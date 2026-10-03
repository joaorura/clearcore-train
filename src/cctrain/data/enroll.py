"""Validação do áudio de enrollment (16 kHz mono), com paridade contra o Rust do Clearcore.

Réplica de `validate_enrollment_audio` (clearcore crates/model/src/enrollment.rs): mesmas constantes, mesma ordem das
checagens (duração, finitude, clipping, nível, fala ativa) e a mesma aritmética: soma sequencial em f64, dB convertido
para f32, limiares e comparações em f32. O golden de tests/golden/enroll_validation_v1.json vem do próprio código Rust
(scripts/gen_enroll_validation_golden.py). O áudio bruto é biométrico: nada aqui o grava nem o loga.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE_HZ = 16_000
MIN_SAMPLES = 96_000  # 6 s
MAX_SAMPLES = 192_000  # 12 s
MAX_PEAK = np.float32(0.99)
MIN_RMS_DBFS = np.float32(-40.0)
MIN_ACTIVE_FRACTION = np.float32(0.60)
VAD_FRAME_SAMPLES = 320  # 20 ms
VAD_ABSOLUTE_FLOOR_DBFS = np.float32(-50.0)
VAD_RELATIVE_RANGE_DB = np.float32(30.0)


class EnrollmentRejected(ValueError):
    """Áudio recusado. `reason` espelha a variante do erro do Rust; `value` é o campo dela: nº de amostras
    (too_short/too_long), índice (non_finite), pico (clipping), dBFS (too_quiet) ou fração ativa."""

    def __init__(self, reason: str, value: float, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.value = value


@dataclass(frozen=True)
class EnrollmentReport:
    duration_secs: np.float32
    peak: np.float32
    rms_dbfs: np.float32
    active_speech_fraction: np.float32


def _mean_square_dbfs(frames: np.ndarray) -> np.ndarray:
    """dBFS por linha de `frames` (n, m): soma sequencial em f64 (como o `.sum()` do Rust), média, 10·log10(·+1e-12)
    em f64 e depois f32."""
    x = frames.astype(np.float64)
    total = np.cumsum(x * x, axis=1)[:, -1]
    mean = total / max(frames.shape[1], 1)
    return (10.0 * np.log10(mean + 1e-12)).astype(np.float32)


def check_enrollment_audio(x16k: np.ndarray) -> EnrollmentReport:
    """Valida e devolve as medidas; levanta `EnrollmentRejected`. `x16k`: mono 1-D a 16 kHz, convertido para f32."""
    x = np.asarray(x16k)
    if x.ndim != 1:
        raise ValueError("áudio de enrollment deve ser 1-D (mono)")
    x = x.astype(np.float32, copy=False)
    n = len(x)
    if n < MIN_SAMPLES:
        raise EnrollmentRejected("too_short", n, f"{n} amostras, mínimo {MIN_SAMPLES} (6 s)")
    if n > MAX_SAMPLES:
        raise EnrollmentRejected("too_long", n, f"{n} amostras, máximo {MAX_SAMPLES} (12 s)")
    bad = np.flatnonzero(~np.isfinite(x))
    if bad.size:
        raise EnrollmentRejected("non_finite", int(bad[0]), f"amostra não finita no índice {int(bad[0])}")
    peak = np.float32(np.max(np.abs(x)))
    if peak > MAX_PEAK:
        raise EnrollmentRejected("clipping", float(peak), f"pico {float(peak):.3f} > {float(MAX_PEAK)}")
    rms_dbfs = _mean_square_dbfs(x[None, :])[0]
    if rms_dbfs <= MIN_RMS_DBFS:
        raise EnrollmentRejected("too_quiet", float(rms_dbfs), f"{float(rms_dbfs):.1f} dBFS <= {float(MIN_RMS_DBFS)}")
    n_frames = n // VAD_FRAME_SAMPLES
    levels = _mean_square_dbfs(x[: n_frames * VAD_FRAME_SAMPLES].reshape(n_frames, VAD_FRAME_SAMPLES))
    threshold = np.maximum(VAD_ABSOLUTE_FLOOR_DBFS, np.float32(levels.max() - VAD_RELATIVE_RANGE_DB))
    active = int(np.count_nonzero(levels >= threshold))
    fraction = np.float32(active) / np.float32(n_frames)
    if fraction < MIN_ACTIVE_FRACTION:
        raise EnrollmentRejected("insufficient_speech", float(fraction),
                                 f"{float(fraction) * 100:.0f}% de fala ativa, mínimo 60%")
    return EnrollmentReport(np.float32(n) / np.float32(SAMPLE_RATE_HZ), peak, rms_dbfs, fraction)


def validate_enrollment_audio(x16k: np.ndarray) -> None:
    """Levanta `EnrollmentRejected` se o áudio de 16 kHz não cumpre o contrato do enrollment do Clearcore."""
    check_enrollment_audio(x16k)
