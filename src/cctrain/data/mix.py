"""Receita das misturas (spec §4.4), versão do piloto: tipos, frações, SNR/SIR, RIR e alvo seco."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.signal import fftconvolve

SNRS_DB = (-5, 0, 5, 10, 20, 40)
SIR_RANGE_DB = (-5.0, 25.0)
INTERFERER_ONLY_RMS_DBFS = -25.0
PEAK = 0.99


class MixKind(IntEnum):
    TARGET_NOISE = 0
    TARGET_INTERFERER = 1
    TARGET_INTERFERER_NOISE = 2
    INTERFERER_ONLY = 3


def mixture_fractions(s: float = 0.05) -> dict[MixKind, float]:
    if not 0.0 <= s < 1.0:
        raise ValueError("fração de silêncio fora de [0, 1)")
    k = (1.0 - s) / 0.9
    return {MixKind.TARGET_NOISE: 0.35 * k, MixKind.TARGET_INTERFERER: 0.20 * k,
            MixKind.TARGET_INTERFERER_NOISE: 0.35 * k, MixKind.INTERFERER_ONLY: s}


def power(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-20


def scale_for_ratio(ref: np.ndarray, other: np.ndarray, ratio_db: float) -> np.ndarray:
    gain = np.sqrt(power(ref) / (power(other) * 10.0 ** (ratio_db / 10.0)))
    return np.asarray(other, dtype=np.float64) * gain  # float64: f32 rounding breaks the 1e-9 dB contract


def apply_rir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return fftconvolve(np.asarray(x, np.float64), np.asarray(h, np.float64))[: len(x)].astype(np.float32)


def crop_or_pad(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) >= n:
        start = int(rng.integers(0, len(x) - n + 1))
        return np.asarray(x[start : start + n], dtype=np.float32)
    return np.pad(np.asarray(x, np.float32), (0, n - len(x)))


def loop_to_length(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    reps = int(np.ceil(n / len(x))) + 1
    return crop_or_pad(np.tile(np.asarray(x, np.float32), reps), n, rng)


@dataclass(frozen=True)
class MixResult:
    noisy: np.ndarray
    clean: np.ndarray
    kind: MixKind
    snr_db: float
    sir_db: float


def make_mixture(target: np.ndarray, interferers: Sequence[np.ndarray], noise: np.ndarray | None, *,
                 kind: MixKind, snr_db: float, sir_db: float, rir_target: np.ndarray | None = None,
                 rir_interferer: np.ndarray | None = None) -> MixResult:
    n = len(target)
    has_target = kind is not MixKind.INTERFERER_ONLY
    clean = np.asarray(target, np.float32) if has_target else np.zeros(n, np.float32)
    wet = apply_rir(clean, rir_target) if (has_target and rir_target is not None) else clean
    speech = wet.astype(np.float64)
    if kind in (MixKind.TARGET_INTERFERER, MixKind.TARGET_INTERFERER_NOISE, MixKind.INTERFERER_ONLY):
        interf = np.zeros(n)
        for x in interferers:
            interf += apply_rir(x, rir_interferer) if rir_interferer is not None else np.asarray(x, np.float64)
        if has_target:
            interf = scale_for_ratio(wet, interf, sir_db)
        else:
            ref = np.full(n, 10 ** (INTERFERER_ONLY_RMS_DBFS / 20.0))
            interf = scale_for_ratio(ref, interf, 0.0)
        speech = speech + interf
    noisy = speech
    if noise is not None and kind in (MixKind.TARGET_NOISE, MixKind.TARGET_INTERFERER_NOISE,
                                      MixKind.INTERFERER_ONLY):
        noisy = speech + scale_for_ratio(speech, noise, snr_db)
    gain = min(1.0, PEAK / max(float(np.max(np.abs(noisy))), 1e-9), PEAK / max(float(np.max(np.abs(clean))), 1e-9))
    return MixResult((noisy * gain).astype(np.float32), (clean * gain).astype(np.float32), kind, snr_db, sir_db)
