"""Export stateful (S=1 por hop) do pDFNet3 Pro v4-presence: o v3 completo + cabeça `p_target` no grafo `enc`.

Contrato: docs/export-contract-pro-v4.md. Igual ao v3 (`pdfnet3_pro_v3.py`) em DSP, condicionamento, decoders e
roteamento; o `enc` ganha a saída `p_target [1,1,1]` (probabilidade de o falante-alvo estar ativo no quadro, sigmoide)
e o par de estado `p_h [1,1,32]` (GRU da cabeça). A cabeça lê a saída da cross-attention (`emb`, já condicionada) e o
`speaker_embedding`, então fica no mesmo grafo que a cross-attention: não há grafo novo.

Gating (manifest `gating`): com `speaker_present`=1, ZERO quando a máquina de estados de presença (histerese τ_on/τ_off
+ hangover) estiver desligada, senão ERB+DF; sem perfil, o gating por lsnr de sempre.
"""

from __future__ import annotations

import copy
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from cctrain.export import pdfnet3_pro_v3 as v3
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmGenerator, load_film_config
from cctrain.models.pdfnet3_pro import PDFNet3Pro

MODEL_ID = "pdfnet3-pro-v4-presence"
OPSET = v3.OPSET
EMB_DIM = v3.EMB_DIM
P_HIDDEN = 32

GRAPHS: dict[str, dict] = copy.deepcopy(v3.GRAPHS)
GRAPHS["enc"]["outputs"]["p_target"] = [1, 1, 1]
GRAPHS["enc"]["states"]["p_h"] = [1, 1, P_HIDDEN]


def input_names(graph: str) -> list[str]:
    g = GRAPHS[graph]
    return list(g["inputs"]) + [f"{s}_in" for s in g["states"]]


def output_names(graph: str) -> list[str]:
    g = GRAPHS[graph]
    return list(g["outputs"]) + [f"{s}_out" for s in g["states"]]


def load_pro_v4(ckpt_path: Path, device: str = "cpu") -> PDFNet3Pro:
    base, _ = load_upstream_dfnet3(device=device)
    gen = FilmGenerator(load_film_config())
    model = PDFNet3Pro(base, gen, cross_attention_config=v3.ATTN_CFG, enable_cross_attention=True,
                       presence_head=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    if model.presence.hidden != P_HIDDEN:
        raise ValueError(f"cabeça com hidden {model.presence.hidden} != {P_HIDDEN}")
    return model.to(device).eval()


class EncStepV4(v3.EncStep):
    def __init__(self, pro: PDFNet3Pro):
        super().__init__(pro)
        self.head = pro.presence

    def forward(self, feat_erb, feat_spec, speaker_embedding, speaker_present, enc_h_in, feat_erb_buf_in,
                feat_spec_buf_in, p_h_in):
        e0, e1, e2, e3, emb, c0, lsnr, gain, h, erb_buf, spec_buf = super().forward(
            feat_erb, feat_spec, speaker_embedding, speaker_present, enc_h_in, feat_erb_buf_in, feat_spec_buf_in)
        hd = self.head
        z = torch.relu(hd.inp(emb))  # [1,1,64]
        s = hd.spk(speaker_embedding)  # [1,64]: broadcast sem Unsqueeze (evita AddAxis antes do GRU no tract)
        y, ph = hd.gru(torch.cat([z, z * s], dim=-1), p_h_in)
        p_target = torch.sigmoid(hd.out(y))  # [1,1,1]
        return e0, e1, e2, e3, emb, c0, lsnr, gain, p_target, h, erb_buf, spec_buf, ph


STEPS = {"enc": EncStepV4, "erb_dec": v3.ErbDecStep, "df_dec": v3.DfDecStep}


def zero_inputs(graph: str) -> dict[str, np.ndarray]:
    g = GRAPHS[graph]
    shapes = {**g["inputs"], **{f"{s}_in": sh for s, sh in g["states"].items()}}
    return {k: np.zeros(v, np.float32) for k, v in shapes.items()}


def _fix_io_shapes(proto: onnx.ModelProto, graph: str) -> None:
    g = GRAPHS[graph]
    shapes = {**g["inputs"], **{f"{s}_in": sh for s, sh in g["states"].items()},
              **g["outputs"], **{f"{s}_out": sh for s, sh in g["states"].items()}}
    for vi in list(proto.graph.input) + list(proto.graph.output):
        dims = vi.type.tensor_type.shape.dim
        for d in list(dims):
            dims.remove(d)
        for w in shapes[vi.name]:
            dims.add().dim_value = int(w)


def export_graph(pro: PDFNet3Pro, graph: str, path: Path) -> Path:
    step = STEPS[graph](pro).eval()
    feeds = zero_inputs(graph)
    if "speaker_embedding" in feeds:
        e = torch.randn(1, EMB_DIM, generator=torch.Generator().manual_seed(0))
        feeds["speaker_embedding"] = (e / e.norm()).numpy()
        feeds["speaker_present"] = np.ones(1, np.float32)
    args = tuple(torch.from_numpy(feeds[n]) for n in input_names(graph))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(step, args, buf, input_names=input_names(graph), output_names=output_names(graph),
                          opset_version=OPSET, do_constant_folding=True, dynamo=False)
    proto = onnx.load_from_string(buf.getvalue())
    proto.producer_name = "cctrain.export.pdfnet3_pro_v4"
    proto.doc_string = f"{MODEL_ID} {graph} (S=1, estados <nome>_in/<nome>_out)"
    _fix_io_shapes(proto, graph)
    onnx.checker.check_model(proto, full_check=True)
    path.write_bytes(proto.SerializeToString())
    return path


def build_manifest(files: dict[str, Path], ckpt_path: Path, ckpt_sha: str, gate: dict) -> dict:
    man = v3.build_manifest(files, ckpt_path, ckpt_sha)
    graphs = {}
    for name, spec in GRAPHS.items():
        graphs[name] = {
            "file": f"{name}.onnx",
            "inputs": {k: {"shape": v, "dtype": "float32"} for k, v in spec["inputs"].items()},
            "outputs": {k: {"shape": v, "dtype": "float32"} for k, v in spec["outputs"].items()},
            "state_pairs": [{"in": f"{s}_in", "out": f"{s}_out", "shape": sh, "dtype": "float32", "init": "zeros"}
                            for s, sh in spec["states"].items()],
        }
    man["model"] = MODEL_ID
    man["graphs"] = graphs
    man["gating"] = {
        "mode": "presence",
        "output": "p_target",
        "graph": "enc",
        "tau_on": float(gate["tau_on"]),
        "tau_off": float(gate["tau_off"]),
        "hangover_frames": int(gate["hangover_frames"]),
        "applies_when": "speaker_present == 1",
        "decision": "ativo = p_target >= tau_off se ativo no quadro anterior, senão p_target >= tau_on; hang = "
                    "hangover_frames se ativo, senão max(hang-1, 0); quadro passa (ERB+DF) se ativo ou hang > 0, "
                    "senão ZERO. Estado inicial: inativo, hang 0 (zerar ao trocar de perfil/reset). Sem passthrough "
                    "nem ERB-only com perfil.",
        "no_profile": {"mode": "lsnr", "min_db": -10.0, "max_db_erb": 30.0, "max_db_df": 20.0},
        "execution": "os três grafos rodam em todo quadro ativo (estados sempre avançam); o gating só seleciona a "
                     "saída. Atalho de silêncio digital (média quadrática < 1e-7) como hoje.",
    }
    man["presence"] = {
        "output": "p_target [1,1,1]: probabilidade (sigmoide) de o falante-alvo estar ativo no quadro",
        "state": "p_h [1,1,32] (GRU da cabeça), zeros no início",
        "label": "VAD relativo à trilha do alvo limpo (quadro a até 30 dB do máximo da trilha, piso -75 dBFS)",
        "meaningful_when": "speaker_present == 1 (sem perfil a saída existe mas não é usada)",
    }
    return man


@dataclass
class ProV4Reference:
    m: np.ndarray
    coefs: np.ndarray
    lsnr: np.ndarray
    p_target: np.ndarray


def pro_reference(pro: PDFNet3Pro, feat_erb: np.ndarray, feat_spec: np.ndarray, embedding: np.ndarray,
                  present: float) -> ProV4Reference:
    """`PDFNet3Pro.forward(..., return_presence=True)` sobre a sequência inteira, sem `pad_feat` (como no v3)."""
    t = feat_erb.shape[0]
    ferb = torch.from_numpy(np.ascontiguousarray(feat_erb, np.float32))[None, None]
    fspec = torch.from_numpy(np.stack([feat_spec.real, feat_spec.imag], -1).astype(np.float32))[None, None]
    spec = torch.zeros(1, 1, t, 481, 2)
    emb = torch.from_numpy(np.asarray(embedding, np.float32).reshape(1, EMB_DIM))
    mask = torch.full((1, 1), float(present))
    saved = pro.base.pad_feat
    pro.base.pad_feat = nn.Identity()
    try:
        with torch.no_grad():
            _, m, lsnr, df, p_logit = pro(spec, ferb, fspec, emb, mask, return_presence=True)
    finally:
        pro.base.pad_feat = saved
    coefs = df.permute(0, 2, 3, 1, 4).reshape(1, t, v3.NB_DF, 2 * v3.DF_ORDER)
    return ProV4Reference(m[0, 0].numpy(), coefs[0].numpy(), lsnr.reshape(-1).numpy(),
                          torch.sigmoid(p_logit).reshape(-1).numpy())


def presence_state_machine(p: np.ndarray, tau_on: float, tau_off: float, hangover: int) -> np.ndarray:
    """Réplica de referência da decisão do runtime (1-D): True = passa (ERB+DF), False = ZERO."""
    active, hang = False, 0
    out = np.zeros(len(p), bool)
    for i, x in enumerate(np.asarray(p, np.float64)):
        active = x >= tau_off if active else x >= tau_on
        hang = hangover if active else max(hang - 1, 0)
        out[i] = active or hang > 0
    return out


pack = v3.pack
sha256_file = v3.sha256_file
