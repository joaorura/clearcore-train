import json

import numpy as np
import onnx
import pytest
import torch

from cctrain import tractcheck
from cctrain.export.enrollment import (
    OUTPUT_NAMES,
    build_prototype,
    export_enrollment_onnx,
    scale_to_peak,
    scale_to_rms_dbfs,
    speechlike_16k,
)

pytestmark = pytest.mark.tract
DURATIONS_S = (2, 6, 8, 10, 12)
LEVELS = {
    "rms_-39.9dBFS": lambda x: scale_to_rms_dbfs(x, -39.9),
    "rms_-20dBFS": lambda x: scale_to_rms_dbfs(x, -20.0),
    "peak_0.99": lambda x: scale_to_peak(x, 0.99),
}


def _within_limits(name: str, v: np.ndarray) -> bool:
    if not np.isfinite(v).all():
        return False
    if name.startswith("gamma"):
        return bool(v.min() >= 0.001 and v.max() <= 100.0)
    if name.startswith("beta"):
        return bool(np.abs(v).max() <= 50.0)
    return True


def test_enrollment_prototype_loads_and_matches_in_tract(tmp_path, m0_runs):
    results = {}
    for mode in ("expand", "broadcast"):
        model = build_prototype(seed=0, context_mode=mode)
        onnx_path = export_enrollment_onnx(model, tmp_path / f"enroll-{mode}.onnx")
        proto = onnx.load(str(onnx_path))
        # Task 14: tract recusa saída com dimensão simbólica; só a entrada pode ter N.
        assert all(not d.dim_param for o in proto.graph.output for d in o.type.tensor_type.shape.dim)
        assert not proto.graph.value_info
        cases = []
        for seconds in DURATIONS_S:
            for level, scale in LEVELS.items():
                x = scale(speechlike_16k(16_000 * seconds, seed=seconds)).astype("<f4")
                inp, out = tmp_path / f"{mode}-{seconds}-{level}.f32", tmp_path / f"{mode}-{seconds}-{level}.json"
                x.tofile(inp)
                proc = tractcheck.run("enroll", onnx_path, inp, out)
                rep = json.loads(out.read_text())
                case = {"seconds": seconds, "level": level, "ok": rep["ok"], "error": rep.get("error"),
                        "returncode": proc.returncode}
                if rep["ok"]:
                    with torch.no_grad():
                        ref = [t.numpy() for t in model(torch.from_numpy(x)[None])]
                    assert len(ref) == len(OUTPUT_NAMES)
                    excess, limits, abs_err, rel_err = [], [], [], []
                    for name, r in zip(OUTPUT_NAMES, ref):
                        got = np.asarray(rep["outputs"][name], np.float32)
                        assert got.shape == r.shape
                        diff = np.abs(got - r)
                        abs_err.append(float(diff.max()))
                        rel_err.append(float((diff / np.maximum(np.abs(r), 1e-12)).max()))
                        excess.append(float(np.max(np.abs(got - r) - (1e-4 + 1e-4 * np.abs(r)))))
                        limits.append(_within_limits(name, got))
                    case.update(worst_excess=max(excess), max_abs_err=max(abs_err), max_rel_err=max(rel_err),
                                within_limits=all(limits),
                                passed=max(excess) <= 0 and all(limits))
                else:
                    case["passed"] = False
                cases.append(case)
        results[mode] = {"outputs_concrete": True, "all_pass": all(c["passed"] for c in cases), "cases": cases}
    (m0_runs / "ecapa_tract.json").write_text(json.dumps(results, indent=2) + "\n")
    assert any(r["all_pass"] for r in results.values()), json.dumps(
        {m: [c for c in r["cases"] if not c["passed"]][:3] for m, r in results.items()}, indent=2)
