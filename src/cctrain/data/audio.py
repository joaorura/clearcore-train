"""Leitura de áudio (FLAC/WAV/MP3 via libsndfile) e reamostragem soxr HQ para o loader (spec §4.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    return soxr.resample(np.asarray(x, dtype=np.float32), sr_in, sr_out, quality="HQ").astype(np.float32)
