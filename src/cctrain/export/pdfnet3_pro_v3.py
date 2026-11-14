"""Export stateful (S=1 por hop) do pDFNet3 Pro v3 COMPLETO: base + FiLM + FrequencyDWConvMixer + cross-attention.

Contrato: docs/export-contract-pro-v3.md. Três grafos (enc, erb_dec, df_dec) para preservar o gating por lsnr do
runtime (cada decoder só avança nos quadros em que roda). Condicionamento pela entrada `speaker_embedding` [1,192]
(ECAPA L2-normalizado, o mesmo vetor do treino) + `speaker_present` [1]; FiLM, tokens = fallback_proj(e).view(16,128)
e a cross-attention são calculados dentro dos grafos, exatamente como em `PDFNet3Pro.forward`. Sem perfil
(`speaker_present`=0) o grafo é o DFNet3 base do checkpoint (FiLM identidade, delta da atenção e do mixer zerados).

Todo estado temporal é par `<nome>_in`/`<nome>_out` (float32, zeros no início). A cross-attention é por quadro (as
consultas de cada quadro atendem aos 16 tokens estáticos), então não tem cache temporal.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cctrain.models.cross_attention import SubbandCrossAttentionConfig
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmGenerator, load_film_config
from cctrain.models.pdfnet3_pro import PDFNet3Pro

MODEL_ID = "pdfnet3-pro-v3-full"
HOP = 480
SAMPLE_RATE = 48_000
OPSET = 12  # mesmo opset dos grafos aprovados do DFNet3 (tract 0.19.16)
NB_ERB, NB_DF, DF_ORDER = 32, 96, 5
EMB_DIM = 192
ATTN_CFG = SubbandCrossAttentionConfig(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4,
                                       num_tokens=16, token_dim=128)

COND_INPUTS = {"speaker_embedding": [1, EMB_DIM], "speaker_present": [1]}

GRAPHS: dict[str, dict] = {
    "enc": {
        "inputs": {"feat_erb": [1, 1, 1, NB_ERB], "feat_spec": [1, 2, 1, NB_DF], **COND_INPUTS},
        "outputs": {"e0": [1, 64, 1, 32], "e1": [1, 64, 1, 16], "e2": [1, 64, 1, 8], "e3": [1, 64, 1, 8],
                    "emb": [1, 1, 512], "c0": [1, 64, 1, NB_DF], "lsnr": [1, 1, 1], "erb_gain": [1, 1, 1, NB_ERB]},
        "states": {"enc_h": [1, 1, 256], "feat_erb_buf": [1, 1, 2, NB_ERB], "feat_spec_buf": [1, 2, 2, NB_DF]},
    },
    "erb_dec": {
        "inputs": {"emb": [1, 1, 512], "e3": [1, 64, 1, 8], "e2": [1, 64, 1, 8], "e1": [1, 64, 1, 16],
                   "e0": [1, 64, 1, 32], "erb_gain": [1, 1, 1, NB_ERB]},
        "outputs": {"m": [1, 1, 1, NB_ERB]},
        "states": {"erb_h": [2, 1, 256]},
    },
    "df_dec": {
        "inputs": {"emb": [1, 1, 512], "c0": [1, 64, 1, NB_DF], **COND_INPUTS},
        "outputs": {"coefs": [1, 1, NB_DF, 2 * DF_ORDER]},
        "states": {"df_h": [2, 1, 256], "c0_buf": [1, 64, 4, NB_DF]},
    },
}


def input_names(graph: str) -> list[str]:
    g = GRAPHS[graph]
    return list(g["inputs"]) + [f"{s}_in" for s in g["states"]]


def output_names(graph: str) -> list[str]:
    g = GRAPHS[graph]
    return list(g["outputs"]) + [f"{s}_out" for s in g["states"]]


# ------------------------------------------------------------------------------------------------------------------
# Modelo


def load_pro_v3(ckpt_path: Path, device: str = "cpu") -> PDFNet3Pro:
    """PDFNet3Pro exatamente como em scripts/train_pro.py, com o state_dict do checkpoint (strict)."""
    base, _ = load_upstream_dfnet3(device=device)
    gen = FilmGenerator(load_film_config())  # Γ = ln 10, B = 4 (iguais ao StageFHParams do treino)
    model = PDFNet3Pro(base, gen, cross_attention_config=ATTN_CFG, enable_cross_attention=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval()


def _strip_time_pad(seq: nn.Sequential, expected_top: int) -> tuple[nn.Sequential, tuple[int, int]]:
    """Remove o ConstantPad2d causal (topo = atraso temporal) e devolve (resto, pad de frequência)."""
    pad = seq[0]
    if not isinstance(pad, nn.ConstantPad2d):
        raise ValueError(f"esperado ConstantPad2d no início de {seq}")
    left, right, top, bottom = pad.padding
    if top != expected_top or bottom != 0 or pad.value != 0.0:
        raise ValueError(f"pad causal inesperado: {pad.padding}")
    return nn.Sequential(*list(seq)[1:]), (left, right)


def _assert_no_time_context(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.kernel_size[0] != 1:
            raise ValueError(f"conv com contexto temporal não tratado: {m}")
        if isinstance(m, nn.ConstantPad2d) and (m.padding[2] or m.padding[3]):
            raise ValueError(f"pad temporal não tratado: {m}")


def _gru_film(sq: nn.Module, x: Tensor, h: Tensor, film: tuple[Tensor, Tensor] | None) -> tuple[Tensor, Tensor]:
    """SqueezedGRU_S.forward com estado explícito (idêntico ao CCTRAIN-PATCH(film))."""
    y = sq.linear_in(x)
    if film is not None:
        y = y * film[0] + film[1]
    y, h = sq.gru(y, h)
    y = sq.linear_out(y)
    if sq.gru_skip is not None:
        y = y + sq.gru_skip(x)
    return y, h


class EncStep(nn.Module):
    def __init__(self, pro: PDFNet3Pro):
        super().__init__()
        enc = pro.base.enc
        self.enc, self.film, self.mixer, self.ca = enc, pro.film, pro.freq_mixer, pro.cross_attn
        self.lsnr_corr, self.lsnr_corr_scale = getattr(pro, "lsnr_corr", None), getattr(pro, "lsnr_corr_scale", 10.0)
        self.erb_conv0, self.erb_fpad = _strip_time_pad(enc.erb_conv0, 2)
        self.df_conv0, self.df_fpad = _strip_time_pad(enc.df_conv0, 2)
        for m in (enc.erb_conv1, enc.erb_conv2, enc.erb_conv3, enc.df_conv1, self.mixer):
            _assert_no_time_context(m)

    def forward(self, feat_erb, feat_spec, speaker_embedding, speaker_present, enc_h_in, feat_erb_buf_in,
                feat_spec_buf_in):
        enc = self.enc
        mask = speaker_present.reshape(1, 1)
        v = self.film(speaker_embedding, speaker_present)  # ver DfDecStep (tract ChangeAxes)
        # FrequencyDWConvMixer gated pela máscara (PDFNet3Pro.forward), por quadro
        x = feat_erb + (self.mixer(feat_erb) - feat_erb) * mask.reshape(1, 1, 1, 1)
        erb_win = torch.cat([feat_erb_buf_in, x], dim=2)  # [1,1,3,32]
        spec_win = torch.cat([feat_spec_buf_in, feat_spec], dim=2)  # [1,2,3,96]
        e0 = self.erb_conv0(F.pad(erb_win, self.erb_fpad) if any(self.erb_fpad) else erb_win)
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)
        c0 = self.df_conv0(F.pad(spec_win, self.df_fpad) if any(self.df_fpad) else spec_win)
        c1 = enc.df_conv1(c0)
        cemb = enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        emb = enc.combine(e3.permute(0, 2, 3, 1).flatten(2), cemb)
        emb, h = _gru_film(enc.emb_gru, emb, enc_h_in, (v.gamma_enc, v.beta_enc))
        lsnr = enc.lsnr_fc(emb) * enc.lsnr_scale + enc.lsnr_offset
        att = self.ca(emb, speaker_embedding, mask=speaker_present.reshape(1, 1))
        if self.lsnr_corr is not None:  # pro-v5-keep: mesma correção do PDFNet3Pro.forward
            lsnr = lsnr + self.lsnr_corr(att.output) * (self.lsnr_corr_scale * speaker_present.reshape(1, 1, 1))
        return (e0, e1, e2, e3, att.output, c0, lsnr, att.subband_gains, h,
                erb_win[:, :, 1:], spec_win[:, :, 1:])


class ErbDecStep(nn.Module):
    def __init__(self, pro: PDFNet3Pro):
        super().__init__()
        self.dec = pro.base.erb_dec
        _assert_no_time_context(self.dec)

    def forward(self, emb, e3, e2, e1, e0, erb_gain, erb_h_in):
        d = self.dec
        b, _, t, f8 = e3.shape
        y, h = _gru_film(d.emb_gru, emb, erb_h_in, None)
        y = y.view(b, t, f8, -1).permute(0, 3, 1, 2)
        e3 = d.convt3(d.conv3p(e3) + y)
        e2 = d.convt2(d.conv2p(e2) + e3)
        e1 = d.convt1(d.conv1p(e1) + e2)
        m = d.conv0_out(d.conv0p(e0) + e1)
        return m * erb_gain, h


class DfDecStep(nn.Module):
    def __init__(self, pro: PDFNet3Pro):
        super().__init__()
        self.dec, self.film = pro.base.df_dec, pro.film
        self.convp, fpad = _strip_time_pad(self.dec.df_convp, 4)
        if any(fpad):
            raise ValueError("df_convp com pad de frequência inesperado")

    def forward(self, emb, c0, speaker_embedding, speaker_present, df_h_in, c0_buf_in):
        d = self.dec
        # máscara [1] por broadcast, sem Reshape: um AddAxis vindo de Reshape([1]->[1,1]) que chega ao GRU dispara
        # um erro no ChangeAxes do tract 0.19.16 ("Removing non-trivial 2 axis of dim 1,1,256" no Scan do GRU)
        v = self.film(speaker_embedding, speaker_present)
        b, t, _ = emb.shape
        c, h = _gru_film(d.df_gru, emb, df_h_in, (v.gamma_df, v.beta_df))
        if d.df_skip is not None:
            c = c + d.df_skip(emb)
        win = torch.cat([c0_buf_in, c0], dim=2)  # [1,64,5,96]
        cp = self.convp(win).permute(0, 2, 3, 1)
        c = d.df_out(c).view(b, t, d.df_bins, d.df_out_ch) + cp
        return c, h, win[:, :, 1:]


STEPS = {"enc": EncStep, "erb_dec": ErbDecStep, "df_dec": DfDecStep}


def zero_inputs(graph: str) -> dict[str, np.ndarray]:
    g = GRAPHS[graph]
    shapes = {**g["inputs"], **{f"{s}_in": sh for s, sh in g["states"].items()}}
    return {k: np.zeros(v, np.float32) for k, v in shapes.items()}


# ------------------------------------------------------------------------------------------------------------------
# Export


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
    proto.producer_name = "cctrain.export.pdfnet3_pro_v3"
    proto.doc_string = f"{MODEL_ID} {graph} (S=1, estados <nome>_in/<nome>_out)"
    _fix_io_shapes(proto, graph)
    onnx.checker.check_model(proto, full_check=True)
    path.write_bytes(proto.SerializeToString())
    return path


def _fix_io_shapes(proto: onnx.ModelProto, graph: str) -> None:
    g = GRAPHS[graph]
    shapes = {**g["inputs"], **{f"{s}_in": sh for s, sh in g["states"].items()},
              **g["outputs"], **{f"{s}_out": sh for s, sh in g["states"].items()}}
    for vi in list(proto.graph.input) + list(proto.graph.output):
        dims = vi.type.tensor_type.shape.dim
        want = shapes[vi.name]
        for d in list(dims):
            dims.remove(d)
        for w in want:
            dims.add().dim_value = int(w)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_manifest(files: dict[str, Path], ckpt_path: Path, ckpt_sha: str) -> dict:
    graphs = {}
    for name, spec in GRAPHS.items():
        graphs[name] = {
            "file": f"{name}.onnx",
            "inputs": {k: {"shape": v, "dtype": "float32"} for k, v in spec["inputs"].items()},
            "outputs": {k: {"shape": v, "dtype": "float32"} for k, v in spec["outputs"].items()},
            "state_pairs": [{"in": f"{s}_in", "out": f"{s}_out", "shape": sh, "dtype": "float32", "init": "zeros"}
                            for s, sh in spec["states"].items()],
        }
    return {
        "model": MODEL_ID,
        "format_version": 1,
        "hop": HOP,
        "sample_rate": SAMPLE_RATE,
        "fft_size": 960,
        "nb_erb": NB_ERB,
        "nb_df": NB_DF,
        "df_order": DF_ORDER,
        "conv_lookahead": 2,
        "df_lookahead": 2,
        "opset": OPSET,
        "conditioning": {
            "speaker_embedding": "ECAPA-TDNN 192-d L2-normalizado (saída 'embedding' do enrollment); zeros sem perfil",
            "speaker_present": "1.0 com perfil, 0.0 sem perfil (FiLM identidade, atenção e mixer desligados)",
        },
        "routing": {
            "erb_dec": {"emb": "enc.emb", "e3": "enc.e3", "e2": "enc.e2", "e1": "enc.e1", "e0": "enc.e0",
                        "erb_gain": "enc.erb_gain"},
            "df_dec": {"emb": "enc.emb", "c0": "enc.c0", "speaker_embedding": "runtime", "speaker_present": "runtime"},
        },
        "gating": "igual ao DFNet3 stateful atual: enc roda em todo quadro ativo; erb_dec/df_dec só nos quadros em "
                  "que o estágio do lsnr os exige, e só esses avançam os próprios estados",
        "coefs_layout": "[1,1,96,10] reinterpretado como [96, df_order=5, re/im] (igual ao stateful atual)",
        "graphs": graphs,
        "files": {n: {"sha256": sha256_file(p), "size": Path(p).stat().st_size} for n, p in files.items()},
        "source_checkpoint": {"path": str(ckpt_path), "sha256": ckpt_sha},
    }


def pack(out_tar: Path, files: dict[str, Path], manifest: dict) -> str:
    """tar.gz determinístico (mtime 0, uid/gid 0, ordem fixa)."""
    import gzip

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        entries = [(n, Path(p).read_bytes()) for n, p in sorted(files.items())]
        entries.append(("manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()))
        for name, data in entries:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            tar.addfile(ti, io.BytesIO(data))
    out_tar = Path(out_tar)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tar, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw.getvalue())
    return sha256_file(out_tar)


# ------------------------------------------------------------------------------------------------------------------
# Referência PyTorch (o forward de treino)


@dataclass
class ProReference:
    m: np.ndarray  # [T, 32]   máscara ERB final (m_sub * subband_gains)
    coefs: np.ndarray  # [T, 96, 10]
    lsnr: np.ndarray  # [T]


def pro_reference(pro: PDFNet3Pro, feat_erb: np.ndarray, feat_spec: np.ndarray, embedding: np.ndarray,
                  present: float) -> ProReference:
    """`PDFNet3Pro.forward` (o mesmo código do treino) sobre a sequência inteira, sem `pad_feat` (a defasagem de
    lookahead é feita pelos anéis de espectro do runtime) — mesma convenção do `TorchFrameModel`.

    feat_erb [T,32] float32, feat_spec [T,96] complex64, embedding [192].
    """
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
            _, m, lsnr, df = pro(spec, ferb, fspec, emb, mask)
    finally:
        pro.base.pad_feat = saved
    coefs = df.permute(0, 2, 3, 1, 4).reshape(1, t, NB_DF, 2 * DF_ORDER)
    return ProReference(m[0, 0].numpy(), coefs[0].numpy(), lsnr.reshape(-1).numpy())
