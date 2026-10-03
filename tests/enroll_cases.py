"""Casos da validação de enrollment, compartilhados entre os testes e o gerador do golden do Rust.

Sinais de borda são constantes (DC): a conta em dBFS é exata e os limiares caem num valor conhecido.
"""

from __future__ import annotations

import numpy as np

FRAME = 320
N6 = 96_000


def _dc(n: int, c: float) -> np.ndarray:
    return np.full(n, np.float32(c), dtype=np.float32)


def _frames(levels: list[float]) -> np.ndarray:
    """Um quadro de 320 amostras por nível (valor DC)."""
    return np.repeat(np.asarray(levels, np.float32), FRAME)


def _speechlike(n: int, seed: int, dbfs: float) -> np.ndarray:
    from cctrain.export.enrollment import scale_to_rms_dbfs, speechlike_16k

    return scale_to_rms_dbfs(speechlike_16k(n, seed), dbfs)


def cases() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    out["ok_dc_6s"] = _dc(N6, 0.1)
    out["ok_dc_12s"] = _dc(192_000, 0.1)
    out["short_95999"] = _dc(N6 - 1, 0.1)
    out["short_empty"] = np.zeros(0, np.float32)
    out["long_192001"] = _dc(192_001, 0.1)
    # a ordem das checagens vale: duração antes de NaN, NaN antes de clipping
    nan_short = _dc(N6 - 1, 0.1)
    nan_short[10] = np.nan
    out["short_with_nan"] = nan_short
    nan = _dc(N6, 0.1)
    nan[1000] = np.nan
    out["nan_at_1000"] = nan
    inf = _dc(N6, 0.1)
    inf[5] = np.inf
    out["inf_at_5"] = inf
    ninf = _dc(N6, 0.1)
    ninf[N6 - 1] = -np.inf
    out["neg_inf_last"] = ninf
    nan_and_clip = _dc(N6, 0.1)
    nan_and_clip[3] = 2.0
    nan_and_clip[4] = np.nan
    out["nan_and_clip"] = nan_and_clip
    out["peak_exactly_099"] = _dc(N6, 0.99)
    out["peak_next_above_099"] = _dc(N6, np.nextafter(np.float32(0.99), np.float32(1.0)))
    out["peak_negative_0995"] = _dc(N6, -0.995)
    one_clip = _dc(N6, 0.1)
    one_clip[50_000] = -1.0
    out["one_clipped_sample"] = one_clip
    out["silence"] = np.zeros(N6, np.float32)
    out["quiet_dc_0.0099"] = _dc(N6, 0.0099)
    out["quiet_dc_0.01_boundary"] = _dc(N6, 0.01)
    out["quiet_dc_0.0101"] = _dc(N6, 0.0101)
    for k in (179, 180, 181):  # 0,6 de 300 quadros: 180 passa, 179 não
        out[f"active_{k}_of_300"] = _frames([0.1] * k + [0.0] * (300 - k))
    # relativo: quadros 30 dB abaixo do mais alto (-20 dBFS), no piso de -50 dBFS
    edge = 0.1 * 10 ** (-30 / 20)
    out["relative_edge_above"] = _frames([0.1] * 150 + [edge * 1.001] * 150)
    out["relative_edge_exact"] = _frames([0.1] * 150 + [edge] * 150)
    out["relative_edge_below"] = _frames([0.1] * 150 + [edge * 0.999] * 150)
    # o limiar relativo (-44 dBFS) fica acima do piso de -50 dBFS e deixa os quadros de -49 dBFS inativos
    out["relative_threshold_above_floor"] = _frames([0.2] * 120 + [0.0035] * 180)
    tail = np.concatenate([_dc(N6, 0.1), _dc(319, 0.5)])
    out["tail_partial_frame_ignored_by_vad"] = tail
    out["tail_partial_frame_clips"] = np.concatenate([_dc(N6, 0.1), _dc(319, 0.995)])
    for sec, seed, db in ((7, 1, -20.0), (9, 2, -30.0), (12, 3, -39.5), (6, 4, -45.0)):
        out[f"speechlike_{sec}s_{abs(db)}dB"] = _speechlike(16_000 * sec, seed, db)
    return out
