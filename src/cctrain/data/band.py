"""Banda efetiva por clipe: frequência em que a potência acumulada atinge 99% (spec §4.3).

A definição de max_freq a partir dela fica para o M2 (decisão de planejamento 9).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def effective_band_hz(x: np.ndarray, sr: int, rolloff: float = 0.99) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(x):
        return 0.0
    nperseg = min(x.size, 2048 if sr >= 32_000 else 1024)
    f, p = welch(x, fs=sr, nperseg=nperseg)
    c = np.cumsum(p)
    idx = int(np.searchsorted(c, rolloff * c[-1]))
    return float(f[min(idx, f.size - 1)])
