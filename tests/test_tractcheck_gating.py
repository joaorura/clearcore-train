import json

import numpy as np
import pytest

from cctrain import tractcheck
from cctrain.paths import clearcore_root

pytestmark = [pytest.mark.tract, pytest.mark.clearcore]


def _signal(frames: int = 300) -> np.ndarray:
    t = np.arange(frames * 480) / 48_000.0
    rng = np.random.default_rng(0)
    voiced = sum(np.sin(2 * np.pi * 140 * h * t) / h for h in range(1, 12))
    x = 0.08 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)) * voiced + 0.01 * rng.standard_normal(t.size)
    x[: 50 * 480] = 0.0
    return x.astype("<f4")


def _run(tmp_path, name: str):
    x = _signal()
    inp = tmp_path / "in.f32"
    x.tofile(inp)
    asset = clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin"
    prefix = tmp_path / name
    proc = tractcheck.run("gating", asset, inp, prefix)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = np.fromfile(f"{prefix}.out.f32", dtype="<f4")
    lsnr = np.fromfile(f"{prefix}.lsnr.f32", dtype="<f4")
    stages = np.fromfile(f"{prefix}.stage.u8", dtype=np.uint8)
    return x, out, lsnr, stages, json.loads(open(f"{prefix}.json").read())


def test_gating_outputs_are_consistent(tmp_path):
    x, out, lsnr, stages, report = _run(tmp_path, "a")
    assert out.size == x.size and lsnr.size == stages.size == 300
    assert set(np.unique(stages)) <= {0, 1, 2, 3, 4}
    assert np.all(stages[:50] == 4) and np.all(lsnr[:50] == -15.0) and not np.any(out[:50 * 480])
    assert abs(sum(report["fractions"].values()) - 1.0) < 1e-9
    assert np.isfinite(out).all()


def test_gating_is_deterministic(tmp_path):
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b")
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[3], b[3])
