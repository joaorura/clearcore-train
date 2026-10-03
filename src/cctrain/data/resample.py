"""resampler-48k-16k-v1 (spec §2.2).

FIR de fase linear, 241 taps, janela Kaiser beta=8,6, corte 7.600 Hz, decimação por 3:
y[n] = sum_k h[k] * x[3n + 120 - k], com zeros fora de [0, L). Decisões de planejamento (item 7):
h projetado em float64 com ganho DC 1 e publicado em f32; saída com ceil(L/3) amostras; acumulação em
float64 sobre os coeficientes f32; saída arredondada para f32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cctrain.hashing import sha256_bytes
from cctrain.paths import repo_root

NAME = "resampler-48k-16k-v1"
TAPS = 241
CENTER = 120
DECIMATION = 3
FS_IN = 48_000
FS_OUT = 16_000
CUTOFF_HZ = 7_600.0
KAISER_BETA = 8.6
V1_DIR = repo_root() / "resampler" / "v1"


def design_coefficients() -> np.ndarray:
    k = np.arange(TAPS, dtype=np.float64) - CENTER
    fc = CUTOFF_HZ / FS_IN
    h = 2.0 * fc * np.sinc(2.0 * fc * k) * np.kaiser(TAPS, KAISER_BETA)
    h = h / h.sum()
    return h.astype(np.float32)


def load_coefficients(directory: Path = V1_DIR) -> np.ndarray:
    raw = (Path(directory) / "coeffs.f32").read_bytes()
    expected = (Path(directory) / "coeffs.sha256").read_text().split()[0]
    if sha256_bytes(raw) != expected:
        raise ValueError(f"{directory}/coeffs.f32: SHA-256 não confere com coeffs.sha256")
    h = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if h.shape != (TAPS,):
        raise ValueError(f"esperados {TAPS} coeficientes, lidos {h.shape}")
    return h


def output_length(n: int) -> int:
    return (n + DECIMATION - 1) // DECIMATION


def resample_48k_to_16k(x: np.ndarray, coeffs: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("entrada mono 1-D")
    h = (load_coefficients() if coeffs is None else np.asarray(coeffs, dtype=np.float32)).astype(np.float64)
    n_out = output_length(len(x))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)
    padded = np.zeros(len(x) + 2 * CENTER + DECIMATION, dtype=np.float64)
    padded[CENTER : CENTER + len(x)] = x
    full = np.convolve(padded, h)  # full[m] = sum_k h[k] * padded[m - k]
    idx = DECIMATION * np.arange(n_out) + 2 * CENTER  # x[3n+120-k] = padded[3n+240-k]
    return full[idx].astype(np.float32)


def golden_inputs() -> dict[str, np.ndarray]:
    t = np.arange(48_000, dtype=np.float64) / 48_000.0
    impulses = np.zeros(3000, dtype=np.float32)
    impulses[[0, 1000, 2001]] = [1.0, 0.5, -0.25]
    chirp_phase = 2 * np.pi * (50.0 * t + (23_950.0 - 50.0) * t**2 / 2.0)
    out = {
        "impulses_3000": impulses,
        "dc_4800": np.full(4800, 0.5, dtype=np.float32),
        "sine_1k_1s": (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32),
        "chirp_50_23950_1s": (0.5 * np.sin(chirp_phase)).astype(np.float32),
        "square_440_099_1s": (0.99 * np.sign(np.sin(2 * np.pi * 440 * t))).astype(np.float32),
        "noise_1s": (0.1 * np.random.default_rng(20261003).standard_normal(48_000)).astype(np.float32),
    }
    for n in (1, 2, 3, 4, 241, 1000):
        out[f"noise_len{n}"] = (0.1 * np.random.default_rng(n).standard_normal(n)).astype(np.float32)
    return out
