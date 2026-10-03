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
from torch import Tensor, nn

from cctrain.models.ecapa import ECAPATDNN
from cctrain.models.fbank import TractSafeFbank
from cctrain.models.film import FilmConfig, FilmGenerator

OUTPUT_NAMES = ("gamma_enc", "beta_enc", "gamma_df", "beta_df", "embedding")


class EnrollmentModel(nn.Module):
    def __init__(self, fbank: TractSafeFbank, encoder: ECAPATDNN, generator: FilmGenerator):
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


def build_prototype(seed: int, context_mode: str, beta_bound: float = 4.0, out_std: float = 0.5) -> EnrollmentModel:
    torch.manual_seed(seed)
    gen = FilmGenerator(FilmConfig(math.log(10.0), beta_bound))
    with torch.no_grad():
        gen.out.weight.normal_(std=out_std)
        gen.out.bias.normal_(std=out_std)
    return EnrollmentModel(TractSafeFbank(), ECAPATDNN(context_mode=context_mode), gen).eval()


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
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))
    return path


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
