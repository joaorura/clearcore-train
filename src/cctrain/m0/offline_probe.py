"""Desvio entre a emulação de gating do treino (forward PyTorch inteiro + máscara por estágio) e a réplica do runtime."""

from __future__ import annotations

import numpy as np
import torch
from libdf import DF

from cctrain.data.features import extract_features
from cctrain.eval.runtime_path import HOP, LOOKAHEAD, ReplicaResult, Stage, emulate_gating_offline


def offline_vs_runtime(base, audio: np.ndarray, replica: ReplicaResult) -> dict:
    f = extract_features(audio)
    spec, erb, specf = (torch.from_numpy(a)[None] for a in (f.spec, f.feat_erb, f.feat_spec))
    with torch.no_grad():
        spec_e, m, lsnr, _ = base(spec.clone(), erb, specf)
        spec_m = base.mask(spec.clone(), m)
        out, stages = emulate_gating_offline(spec, spec_m, spec_e, lsnr.reshape(1, -1))
    y = np.asarray(DF(48_000, 960, HOP, 32, 2).synthesis(
        np.ascontiguousarray(torch.view_as_complex(out[:, 0].contiguous()).numpy())), np.float32).reshape(-1)
    shift = LOOKAHEAD * HOP
    rt, off = replica.audio[shift:], y[: y.size - shift]
    n = min(rt.size, off.size)
    diff = rt[:n] - off[:n]
    valid = replica.stages[LOOKAHEAD:] != int(Stage.SILENT)
    agree = stages.numpy()[0][: valid.size][valid] == replica.stages[LOOKAHEAD:][valid]
    silent_fraction = float(1.0 - np.mean(valid))
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "error_to_signal_db": float(10 * np.log10(np.mean(diff.astype(np.float64) ** 2) /
                                                   max(np.mean(rt[:n].astype(np.float64) ** 2), 1e-20))),
        "stage_agreement": float(np.mean(agree)),
        "stage_agreement_frames": int(valid.sum()),
        "silent_fraction_excluded": silent_fraction,
        "offline_stage_fractions": {s.name.lower(): float(np.mean(stages.numpy() == int(s))) for s in Stage},
        "note": "emulação offline sem o estado congelado dos decoders; usada no treino (spec §3.5)",
    }
