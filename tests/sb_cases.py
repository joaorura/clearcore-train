"""Entradas dos vetores de referência do SpeechBrain ECAPA (Tarefa 10), compartilhadas entre o gerador e os testes.

Sintéticos: `speechlike_16k` determinístico em 2/6/8/10/12 s a -20 dBFS RMS, mais níveis de borda (-40 dBFS RMS e pico
0,99). Fala real: os exemplos do próprio repositório HF do checkpoint (não versionados; ficam em data/models).
"""

from __future__ import annotations

import numpy as np

from cctrain.export.enrollment import scale_to_peak, scale_to_rms_dbfs, speechlike_16k

REAL_FILES = ("example1.wav", "example2.flac")
DURATIONS_S = (2, 6, 8, 10, 12)


def build_inputs(real: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for sec in DURATIONS_S:
        out[f"s{sec}_rms-20"] = scale_to_rms_dbfs(speechlike_16k(16_000 * sec, seed=sec), -20.0)
    for sec in (2, 12):
        out[f"s{sec}_rms-40"] = scale_to_rms_dbfs(speechlike_16k(16_000 * sec, seed=sec), -40.0)
        out[f"s{sec}_peak0.99"] = scale_to_peak(speechlike_16k(16_000 * sec, seed=sec), 0.99)
    out["s6_rms-40"] = scale_to_rms_dbfs(speechlike_16k(96_000, seed=6), -40.0)
    out["s6_peak0.99"] = scale_to_peak(speechlike_16k(96_000, seed=6), 0.99)
    if real is not None:
        e1, e2 = (np.asarray(real[n], np.float32) for n in REAL_FILES)
        cat = np.concatenate([e1, e2])
        out["real_ex1"] = e1
        out["real_ex2"] = e2
        out["real_cat"] = cat
        out["real_cat_rms-40"] = scale_to_rms_dbfs(cat, -40.0)
        out["real_cat_peak0.99"] = scale_to_peak(cat, 0.99)
        out["real_12s"] = np.tile(cat, 3)[:192_000]
        out["real_8s"] = np.tile(cat, 2)[:128_000]
    return out


CASES = tuple(build_inputs({n: np.full(4, 0.1, np.float32) for n in REAL_FILES}))
