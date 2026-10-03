import json

import numpy as np
import pytest

from cctrain.eval.runtime_path import (
    Stage,
    StatefulOrtFrameModel,
    parity_mixture,
    run_replica,
    stage_fractions,
    tract_gating,
)
from cctrain.paths import clearcore_root

pytestmark = [pytest.mark.tract, pytest.mark.clearcore, pytest.mark.data]
ABS, REL = 1e-4, 1e-4


def test_replica_matches_tract_check_on_1000_real_frames(paths, m0_runs, tmp_path):
    x = parity_mixture(paths.data)
    assert x.size == 1000 * 480
    root = clearcore_root()
    rt = tract_gating(root / "vendor/approved/df-compatible-release-asset-v1.bin", x, tmp_path / "rt")
    py = run_replica(x, StatefulOrtFrameModel(root / "models/stateful"))
    excess = np.abs(py.audio - rt.audio) - (ABS + REL * np.abs(rt.audio))
    per_frame = excess.reshape(1000, 480).max(axis=1)
    lsnr_excess = np.abs(py.lsnr - rt.lsnr) - (ABS + REL * np.abs(rt.lsnr))
    report = {
        "frames": 1000,
        "max_abs_diff": float(np.max(np.abs(py.audio - rt.audio))),
        "worst_excess_over_tol": float(excess.max()),
        "worst_frame": int(per_frame.argmax()),
        "first_frames_excess": [float(v) for v in per_frame[:10]],
        "frames_over_tol": int((per_frame > 0).sum()),
        "lsnr_worst_excess": float(lsnr_excess.max()),
        "stage_mismatches": int((py.stages != rt.stages).sum()),
        "fractions_tract": stage_fractions(rt.stages),
        "fractions_replica": stage_fractions(py.stages),
        "pass": bool(excess.max() <= 0 and lsnr_excess.max() <= 0 and np.array_equal(py.stages, rt.stages)),
    }
    (m0_runs / "gating_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    assert np.count_nonzero(rt.stages == Stage.SILENT) == 20
    assert len({int(s) for s in rt.stages} - {int(Stage.SILENT)}) >= 2, report["fractions_tract"]
    assert report["pass"], json.dumps(report, indent=2)
