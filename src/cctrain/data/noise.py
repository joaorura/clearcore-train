"""Ruído colorido sintético (branco, rosa, marrom), 48 kHz, determinístico por seed.

Rótulo: pilot-only, cost measurement. Existe só para que a medição do loader do M0 tenha ruído de split `noise`
treinável (os 3 ruídos CC0 do piloto são split `test` e o loader os recusa); não é a mistura de ruído do treino.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from cctrain.data.manifest import FileRecord
from cctrain.disk import require_free
from cctrain.hashing import sha256_file

SOURCE_ID = "noise-synthetic-v1"
LICENSE = "LicenseRef-cctrain-generated"
COLORS = ("white", "pink", "brown")
SPECTRAL_EXPONENT = {"white": 0.0, "pink": 1.0, "brown": 2.0}  # potência ~ 1/f^exp
TARGET_RMS = 0.1


def generate_noise(seed: int, fs: int = 48_000, seconds: float = 10.0) -> tuple[np.ndarray, str]:
    color = COLORS[seed % len(COLORS)]
    n = int(seconds * fs)
    rng = np.random.default_rng([seed, 0x6E6F697365])
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    f[0] = f[1]  # evita a divisão por zero no DC
    spec *= f ** (-SPECTRAL_EXPONENT[color] / 2.0)
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    x *= TARGET_RMS / float(np.sqrt(np.mean(x**2)))
    x *= min(1.0, 0.95 / float(np.max(np.abs(x))))
    return x.astype(np.float32), color


def write_noise_set(out_dir: Path, seeds: Iterable[int], data_root: Path) -> list[FileRecord]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        x, color = generate_noise(seed)
        path = out_dir / f"noise_{color}_{seed:05d}.flac"
        require_free(out_dir, x.size * 4)
        sf.write(path, x, 48_000, subtype="PCM_16")
        records.append(FileRecord(
            path=str(path.relative_to(data_root)), sha256=sha256_file(path), source_id=SOURCE_ID, license=LICENSE,
            role="train", speaker=None, split="noise", sr=48_000, num_samples=int(x.size), band_hz=None,
            codec="flac"))
    return records
