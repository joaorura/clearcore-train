"""enrollment.onnx: áudio 16 kHz [1, N] -> gamma_enc, beta_enc, gamma_df, beta_df [256] e embedding [192].

Opset 13, ir_version 8, eixo dinâmico N (spec §2.2, §6.2). No M0 os pesos são aleatórios com semente
(protótipo de operadores); o encoder treinado é do M1.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from torch import Tensor, nn

from cctrain.models.ecapa import ECAPATDNN
from cctrain.models.fbank import TractSafeFbank
from cctrain.models.film import FilmConfig, FilmGenerator

OUTPUT_NAMES = ("gamma_enc", "beta_enc", "gamma_df", "beta_df", "embedding")
OUTPUT_DIMS = {"gamma_enc": 256, "beta_enc": 256, "gamma_df": 256, "beta_df": 256, "embedding": 192}
PAD_ZERO = "cctrain_pad_constant_zero"


class EnrollmentModel(nn.Module):
    def __init__(self, fbank: nn.Module, encoder: ECAPATDNN, generator: FilmGenerator):
        super().__init__()
        self.fbank = fbank
        self.encoder = encoder
        self.generator = generator
        self.register_buffer("mask", torch.ones(1, 1))

    def forward(self, audio: Tensor):
        emb = self.encoder(self.fbank(audio))
        e = emb / torch.sqrt((emb * emb).sum(dim=-1, keepdim=True) + 1e-12)
        v = self.generator(e, self.mask)
        return (v.gamma_enc.reshape(-1), v.beta_enc.reshape(-1), v.gamma_df.reshape(-1), v.beta_df.reshape(-1),
                e.reshape(-1))


def random_generator(seed: int, beta_bound: float = 4.0, out_std: float = 0.5) -> FilmGenerator:
    """Gerador FiLM com a camada de saída aleatória com semente (não identidade), para testes de operadores."""
    torch.manual_seed(seed)
    gen = FilmGenerator(FilmConfig(math.log(10.0), beta_bound))
    with torch.no_grad():
        gen.out.weight.normal_(std=out_std)
        gen.out.bias.normal_(std=out_std)
    return gen


def build_prototype(seed: int, context_mode: str, beta_bound: float = 4.0, out_std: float = 0.5) -> EnrollmentModel:
    torch.manual_seed(seed)
    gen = random_generator(seed, beta_bound, out_std)
    return EnrollmentModel(TractSafeFbank(), ECAPATDNN(context_mode=context_mode), gen).eval()


def randomize_bn_stats(model: nn.Module, seed: int) -> None:
    """Dá a todo BatchNorm estatísticas e parâmetros afins não triviais, com semente.

    A inicialização padrão (média 0, variância 1, γ=1, β=0) faz a BN de inferência valer a identidade e esconde erros
    de dobra da BN no export. Aqui: média ~ N(0, 0.5²), variância ~ U(0.5, 2), γ ~ U(0.5, 1.5), β ~ N(0, 0.2²).
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, nn.modules.batchnorm._BatchNorm):
                continue
            shape = m.running_mean.shape
            m.running_mean.copy_(torch.randn(shape, generator=gen) * 0.5)
            m.running_var.copy_(0.5 + 1.5 * torch.rand(shape, generator=gen))
            if m.affine:
                m.weight.copy_(0.5 + torch.rand(shape, generator=gen))
                m.bias.copy_(torch.randn(shape, generator=gen) * 0.2)


def export_enrollment_onnx(model: EnrollmentModel, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.from_numpy(scale_to_rms_dbfs(speechlike_16k(96_000, 0), -20.0))[None]
    torch.onnx.export(
        model.eval(), (example,), str(path), input_names=["audio"], output_names=list(OUTPUT_NAMES),
        dynamic_axes={"audio": {1: "N"}}, opset_version=13, do_constant_folding=True, dynamo=False,
    )
    proto = onnx.load(str(path))
    proto.ir_version = 8
    tract_fixups(proto)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))
    return path


def tract_fixups(proto: onnx.ModelProto) -> None:
    """Ajustes do grafo exportado que o tract 0.19 exige (spec §6.2).

    1. Todo `Pad` com `constant_value` explícito: o exportador omite o terceiro input (ou o deixa vazio) e o tract
       recusa o nó ("Rules expect 3, node has 2"). O valor 0.0 é inócuo nos Pad de modo reflect.
    2. Saídas com forma concreta: o `Reshape(-1)` final sai com `dim_param` simbólico quando a inferência de forma do
       exportador não atravessa o grafo, e o tract recusa saída simbólica; as formas são as do contrato.
    """
    graph = proto.graph
    if any(n.op_type == "Pad" and (len(n.input) < 3 or not n.input[2]) for n in graph.node):
        graph.initializer.append(helper.make_tensor(PAD_ZERO, TensorProto.FLOAT, [], [0.0]))
        for n in graph.node:
            if n.op_type == "Pad" and (len(n.input) < 3 or not n.input[2]):
                inputs = list(n.input)[:2]
                del n.input[:]
                n.input.extend([*inputs, PAD_ZERO])
    for o in graph.output:
        if o.name in OUTPUT_DIMS:
            dims = o.type.tensor_type.shape.dim
            for d in list(dims):
                dims.remove(d)
            dims.add().dim_value = OUTPUT_DIMS[o.name]


def speechlike_16k(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16_000.0
    f0 = 110.0 + 40.0 * rng.random()
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    voiced = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 20) if f0 * h < 7_500)
    return (envelope * voiced + 0.05 * rng.standard_normal(n)).astype(np.float32)


def scale_to_rms_dbfs(x: np.ndarray, dbfs: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
    return (np.asarray(x, np.float64) * (10 ** (dbfs / 20.0) / rms)).astype(np.float32)


def scale_to_peak(x: np.ndarray, peak: float) -> np.ndarray:
    return (np.asarray(x, np.float64) * (peak / float(np.max(np.abs(x))))).astype(np.float32)


def build_stage_f_enrollment_model(sb_ckpt_dir: Path, ckpt_path: Path) -> EnrollmentModel:
    """Carrega o modelo completo de enrollment com ECAPA SpeechBrain e gerador FiLM do Estágio F."""
    from cctrain.models.film import FilmGenerator, load_film_config
    from cctrain.models.sb_port import build_enrollment, load_speechbrain_state, port_ecapa

    encoder = port_ecapa(load_speechbrain_state(sb_ckpt_dir))
    cfg = load_film_config()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    film_state = {k[len("film."):]: v for k, v in ckpt["model"].items() if k.startswith("film.")}
    gen = FilmGenerator(cfg)
    gen.load_state_dict(film_state)
    return build_enrollment(encoder, gen).eval()


def build_stage_f_enrollment_asset(
    sb_ckpt_dir: Path,
    ckpt_path: Path,
    out_tar_gz: Path,
    out_onnx_path: Path | None = None,
) -> object:
    """Exporta o enrollment.onnx do Estágio F e o empacota em voice-enrollment-asset-v1.tar.gz."""
    from cctrain.export.pack import pack_asset

    out_tar_gz = Path(out_tar_gz)
    onnx_path = out_onnx_path or out_tar_gz.parent / "enrollment.onnx"
    model = build_stage_f_enrollment_model(sb_ckpt_dir, ckpt_path)
    export_enrollment_onnx(model, onnx_path)
    return pack_asset(out_tar_gz, [("enrollment.onnx", onnx_path)])

