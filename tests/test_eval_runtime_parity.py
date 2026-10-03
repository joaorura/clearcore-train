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
from cctrain.hashing import sha256_file
from cctrain.paths import clearcore_root
from cctrain.tractcheck import APPROVED_ASSET_SHA256

pytestmark = [pytest.mark.tract, pytest.mark.clearcore, pytest.mark.data]
ABS, REL = 1e-4, 1e-4
MIN_FRAMES_PER_STAGE = 20
EXPECTED_SILENT = 40  # 20 iniciais + 20 no meio (quadros 500-520)


def test_replica_matches_tract_check_on_1000_real_frames(paths, m0_runs, tmp_path):
    x = parity_mixture(paths.data)
    assert x.size == 1000 * 480
    root = clearcore_root()
    asset = root / "vendor/approved/df-compatible-release-asset-v1.bin"
    assert sha256_file(asset) == APPROVED_ASSET_SHA256
    rt = tract_gating(asset, x, tmp_path / "rt")
    py = run_replica(x, StatefulOrtFrameModel(root / "models/stateful"))
    excess = np.abs(py.audio - rt.audio) - (ABS + REL * np.abs(rt.audio))
    per_frame = excess.reshape(1000, 480).max(axis=1)
    lsnr_excess = np.abs(py.lsnr - rt.lsnr) - (ABS + REL * np.abs(rt.lsnr))
    per_stage = {
        st.name.lower(): {
            "frames": int((rt.stages == int(st)).sum()),
            "worst_excess_over_tol": float(per_frame[rt.stages == int(st)].max()) if (rt.stages == int(st)).any()
            else None,
        }
        for st in Stage
    }
    report = {
        "frames": 1000,
        "max_abs_diff": float(np.max(np.abs(py.audio - rt.audio))),
        "worst_excess_over_tol": float(excess.max()),
        "worst_frame": int(per_frame.argmax()),
        "first_frames_excess": [float(v) for v in per_frame[:10]],
        "frames_over_tol": int((per_frame > 0).sum()),
        "lsnr_worst_excess": float(lsnr_excess.max()),
        "stage_mismatches": int((py.stages != rt.stages).sum()),
        "per_stage": per_stage,
        "fractions_tract": stage_fractions(rt.stages),
        "fractions_replica": stage_fractions(py.stages),
        "pass": bool(excess.max() <= 0 and lsnr_excess.max() <= 0 and np.array_equal(py.stages, rt.stages)),
    }
    (m0_runs / "gating_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    counts = {st.name.lower(): int(np.count_nonzero(rt.stages == int(st))) for st in Stage}
    assert counts["silent"] == EXPECTED_SILENT, report["fractions_tract"]
    for name, count in counts.items():
        assert count >= MIN_FRAMES_PER_STAGE, f"estágio {name}: {count} quadros; frações {report['fractions_tract']}"
    assert report["pass"], json.dumps(report, indent=2)
