"""Validação do áudio de enrollment (16 kHz mono), com paridade contra o Rust do Clearcore.

Réplica de `validate_enrollment_audio` (clearcore crates/model/src/enrollment.rs): mesmas constantes, mesma ordem das
checagens (duração, finitude, clipping, nível, fala ativa) e a mesma aritmética: soma sequencial em f64, dB convertido
para f32, limiares e comparações em f32. O golden de tests/golden/enroll_validation_v1.json vem do próprio código Rust
(scripts/gen_enroll_validation_golden.py). O áudio bruto é biométrico: nada aqui o grava nem o loga.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from cctrain.data.audio import load_audio, resample
from cctrain.data.manifest import FileRecord
from cctrain.data.mix import apply_rir, loop_to_length, scale_for_ratio

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


@dataclass(frozen=True)
class EnrollmentAugmentConfig:
    """Configuração de aumentação ruidosa e reverberante para áudio de enrollment."""

    p_rir: float = 0.5
    p_noise: float = 0.5
    snr_range_db: tuple[float, float] = (5.0, 30.0)
    target_peak: float = 0.95


def augment_enrollment_audio(
    x16k: np.ndarray,
    *,
    rirs: Sequence[np.ndarray] | None = None,
    noises: Sequence[np.ndarray] | None = None,
    p_rir: float = 0.5,
    p_noise: float = 0.5,
    snr_range_db: tuple[float, float] = (5.0, 30.0),
    rng: np.random.Generator | None = None,
    target_peak: float = 0.95,
) -> tuple[np.ndarray, dict]:
    """Aplica aumentação reverberante (RIR) e ruído local de microfone a áudio de enrollment (16 kHz).

    - RIR: aplicada via convolução FFT com probabilidade `p_rir` (padrão 0.5).
    - Ruído local de microfone: adicionado com probabilidade `p_noise` (padrão 0.5) com SNR sorteado em `snr_range_db`.
    - Normalização de pico: escala o sinal para no máximo `target_peak` (padrão 0.95), evitando clipping (> 0.99).

    Devolve `(áudio_aumentado_16k, metadados_de_aumentação)`.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x16k, dtype=np.float32).copy()
    meta: dict[str, object] = {
        "applied_rir": False,
        "applied_noise": False,
        "snr_db": None,
    }

    # 1. Reverberação (RIR)
    if rirs and bool(rng.random() < p_rir):
        rir_idx = int(rng.integers(len(rirs)))
        h = np.asarray(rirs[rir_idx], dtype=np.float32)
        x = apply_rir(x, h)
        meta["applied_rir"] = True
        meta["rir_index"] = rir_idx

    # 2. Ruído local de microfone
    if noises and bool(rng.random() < p_noise):
        noise_idx = int(rng.integers(len(noises)))
        nz = np.asarray(noises[noise_idx], dtype=np.float32)
        nz_looped = loop_to_length(nz, len(x), rng)
        snr = float(rng.uniform(*snr_range_db))
        scaled_noise = scale_for_ratio(x, nz_looped, snr)
        x = (x.astype(np.float64) + scaled_noise).astype(np.float32)
        meta["applied_noise"] = True
        meta["snr_db"] = snr
        meta["noise_index"] = noise_idx

    # 3. Normalização de pico para segurança contra clipping
    peak = float(np.max(np.abs(x))) if len(x) > 0 else 0.0
    if peak > target_peak and peak > 0.0:
        x = (x * (target_peak / peak)).astype(np.float32)

    return x, meta


class EnrollmentAudioLoader:
    """Carregador de áudio de enrollment (16 kHz) com Noisy & Reverberant Augmentation configurável."""

    def __init__(
        self,
        utterances: Sequence[Path | str | FileRecord | np.ndarray],
        *,
        data_root: Path | str = "",
        rirs: Sequence[Path | str | FileRecord | np.ndarray] | None = None,
        noises: Sequence[Path | str | FileRecord | np.ndarray] | None = None,
        p_rir: float = 0.5,
        p_noise: float = 0.5,
        snr_range_db: tuple[float, float] = (5.0, 30.0),
        seed: int = 20261003,
        validate: bool = False,
        target_peak: float = 0.95,
    ):
        self.utterances = list(utterances)
        self.data_root = Path(data_root)
        self.rirs = list(rirs) if rirs is not None else []
        self.noises = list(noises) if noises is not None else []
        self.cfg = EnrollmentAugmentConfig(
            p_rir=p_rir,
            p_noise=p_noise,
            snr_range_db=snr_range_db,
            target_peak=target_peak,
        )
        self.seed = seed
        self.validate = validate
        self._rng = np.random.default_rng(seed)

    def _load_audio_16k(self, item: Path | str | FileRecord | np.ndarray) -> np.ndarray:
        if isinstance(item, np.ndarray):
            return item.astype(np.float32, copy=False)
        rel_path = item.path if isinstance(item, FileRecord) else item
        full_path = self.data_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        x, sr = load_audio(full_path)
        if sr != SAMPLE_RATE_HZ:
            x = resample(x, sr, SAMPLE_RATE_HZ)
        return x.astype(np.float32, copy=False)

    def get_rirs_16k(self) -> list[np.ndarray]:
        return [self._load_audio_16k(r) for r in self.rirs]

    def get_noises_16k(self) -> list[np.ndarray]:
        return [self._load_audio_16k(n) for n in self.noises]

    def load_item(self, idx: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
        """Carrega e aumenta o item do índice `idx`."""
        if rng is None:
            rng = self._rng
        raw = self._load_audio_16k(self.utterances[idx])
        rirs_16k = self.get_rirs_16k()
        noises_16k = self.get_noises_16k()
        audio, meta = augment_enrollment_audio(
            raw,
            rirs=rirs_16k,
            noises=noises_16k,
            p_rir=self.cfg.p_rir,
            p_noise=self.cfg.p_noise,
            snr_range_db=self.cfg.snr_range_db,
            rng=rng,
            target_peak=self.cfg.target_peak,
        )
        if self.validate:
            validate_enrollment_audio(audio)
        return audio, meta

    def sample(self, rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
        """Sorteia aleatoriamente uma fala do conjunto e aplica aumentação."""
        if not self.utterances:
            raise ValueError("Sem falas de enrollment disponíveis para sorteio.")
        if rng is None:
            rng = self._rng
        idx = int(rng.integers(len(self.utterances)))
        return self.load_item(idx, rng=rng)

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict]:
        return self.load_item(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self.load_item(i)
