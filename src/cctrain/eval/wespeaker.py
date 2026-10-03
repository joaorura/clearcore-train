"""Avaliador independente WeSpeaker ResNet34 (ONNX oficial, onnxruntime) para filtro do CV, trials e G5.

Entrada do ONNX: `feats` [B, T, 80], fbank Kaldi de 80 mels (25/10 ms, dither 0, amostras em escala int16) com CMN por
enunciado, como no `wespeaker/bin/infer_onnx.py`. Saída: `embs` [B, 256]. Roda na CPU: o `onnxruntime` do ambiente não
traz provider CUDA, e o plano M1 proíbe dependência nova pesada. O checkpoint é conferido por SHA-256 contra
`manifests/models/wespeaker-voxceleb-resnet34.yaml` antes de abrir a sessão (falha fechado).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from torchaudio.compliance import kaldi

from cctrain.hashing import sha256_file
from cctrain.paths import Paths, repo_root

MODEL_ID = "wespeaker-voxceleb-resnet34"
MODEL_MANIFEST = repo_root() / "manifests" / "models" / f"{MODEL_ID}.yaml"
ONNX_NAME = "voxceleb_resnet34.onnx"
SR = 16_000
N_MELS = 80
MIN_SAMPLES = 8_000  # 0,5 s: abaixo disso o embedding não é confiável (TSTP sobre poucos quadros)
EMB_DIM = 256


class CheckpointMismatch(RuntimeError):
    """O arquivo do checkpoint não é o pinado no manifest."""


def load_model_manifest(path: Path = MODEL_MANIFEST) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def default_checkpoint(paths: Paths | None = None) -> Path:
    return (paths or Paths.from_env()).data / "models" / MODEL_ID / ONNX_NAME


def kaldi_fbank(x16k: np.ndarray) -> np.ndarray:
    """[T, 80] float32, com CMN por enunciado. `x16k` é mono, float em [-1, 1], 16 kHz."""
    x = np.asarray(x16k, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"esperado áudio mono 1-D, veio {x.shape}")
    if x.size < MIN_SAMPLES:
        raise ValueError(f"áudio curto demais: {x.size} amostras (< {MIN_SAMPLES})")
    if not np.isfinite(x).all():
        raise ValueError("áudio com valores não finitos")
    feats = kaldi.fbank(torch.from_numpy(x * 32768.0)[None], num_mel_bins=N_MELS, frame_length=25.0,
                        frame_shift=10.0, dither=0.0, sample_frequency=float(SR))
    feats = feats - feats.mean(dim=0, keepdim=True)
    return feats.numpy().astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class WeSpeakerEvaluator:
    def __init__(self, onnx_path: Path | None = None, *, threads: int = 1, manifest: Path = MODEL_MANIFEST) -> None:
        onnx_path = Path(onnx_path) if onnx_path is not None else default_checkpoint()
        pin = load_model_manifest(manifest)["files"][ONNX_NAME]
        actual = sha256_file(onnx_path)
        if actual != pin["sha256"]:
            raise CheckpointMismatch(f"{onnx_path}: sha256 {actual} != pinado {pin['sha256']}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self._in = self._sess.get_inputs()[0].name
        self.sha256 = actual

    def embed(self, x16k: np.ndarray) -> np.ndarray:
        feats = kaldi_fbank(x16k)[None]
        out = self._sess.run(None, {self._in: feats})[0]
        return np.asarray(out[0], dtype=np.float32)

    def embed_many(self, xs: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self.embed(x) for x in xs]) if len(xs) else np.zeros((0, EMB_DIM), np.float32)
