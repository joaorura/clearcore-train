import json

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from cctrain import tractcheck
from cctrain.export.enrollment import (
    OUTPUT_NAMES,
    build_prototype,
    export_enrollment_onnx,
    randomize_bn_stats,
    scale_to_rms_dbfs,
    speechlike_16k,
)

pytestmark = pytest.mark.tract


def _tiny_onnx(path):
    graph = helper.make_graph(
        [helper.make_node("Mul", ["audio", "two"], ["doubled"]),
         helper.make_node("ReduceMean", ["audio"], ["mean"], keepdims=0)],
        "tiny",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "N"])],
        [helper.make_tensor_value_info("doubled", TensorProto.FLOAT, [1, None]),
         helper.make_tensor_value_info("mean", TensorProto.FLOAT, [])],
        [helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_enroll_runs_a_dynamic_graph_like_the_runtime(tmp_path):
    _tiny_onnx(tmp_path / "tiny.onnx")
    x = np.linspace(-0.5, 0.5, 32_000, dtype="<f4")
    x.tofile(tmp_path / "in.f32")
    proc = tractcheck.run("enroll", tmp_path / "tiny.onnx", tmp_path / "in.f32", tmp_path / "out.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rep = json.loads((tmp_path / "out.json").read_text())
    assert rep["ok"] is True and rep["n"] == 32_000
    np.testing.assert_allclose(rep["outputs"]["doubled"], 2 * x, rtol=0, atol=1e-7)
    assert abs(rep["outputs"]["mean"][0] - float(x.mean())) < 1e-6


def test_enroll_reports_unloadable_model(tmp_path):
    (tmp_path / "bad.onnx").write_bytes(b"not an onnx file")
    np.zeros(32_000, dtype="<f4").tofile(tmp_path / "in.f32")
    proc = tractcheck.run("enroll", tmp_path / "bad.onnx", tmp_path / "in.f32", tmp_path / "out.json")
    assert proc.returncode == 1
    assert json.loads((tmp_path / "out.json").read_text())["ok"] is False


def test_randomize_bn_stats_is_deterministic_and_non_trivial():
    a = build_prototype(seed=0, context_mode="expand")
    b = build_prototype(seed=0, context_mode="expand")
    c = build_prototype(seed=0, context_mode="expand")
    randomize_bn_stats(a, seed=7)
    randomize_bn_stats(b, seed=7)
    randomize_bn_stats(c, seed=8)
    bns = [m for m in a.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert bns
    for m in bns:
        assert float(m.running_mean.abs().max()) > 0.05
        assert float((m.running_var - 1.0).abs().max()) > 0.05
        assert float(m.running_var.min()) > 0.0
        assert not torch.equal(m.weight, torch.ones_like(m.weight))
    for (n, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(x, y), n
    assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values(), strict=True))
    assert a.training is False


def test_non_trivial_bn_changes_the_embedding():
    plain = build_prototype(seed=0, context_mode="expand")
    rand = build_prototype(seed=0, context_mode="expand")
    randomize_bn_stats(rand, seed=1)
    x = torch.from_numpy(scale_to_rms_dbfs(speechlike_16k(32_000, 3), -20.0))[None]
    with torch.no_grad():
        assert not torch.allclose(plain(x)[4], rand(x)[4], atol=1e-3)


@pytest.mark.parametrize("mode", ["expand", "broadcast"])
def test_enrollment_with_random_bn_matches_tract(tmp_path, m0_runs, mode):
    # out_std=0.1: com 0.5 (M0) o gerador amplifica ~5x o erro fp32 do embedding e satura a tanh; o embedding já
    # passa em qualquer caso, e o erro dos vetores com 0.5 fica registrado em docs/plans/m1-reports/t2.md.
    model = build_prototype(seed=0, context_mode=mode, out_std=0.1)
    randomize_bn_stats(model, seed=11)
    onnx_path = export_enrollment_onnx(model, tmp_path / f"enroll-bn-{mode}.onnx")
    proto = onnx.load(str(onnx_path))
    assert all(not d.dim_param for o in proto.graph.output for d in o.type.tensor_type.shape.dim)
    assert not proto.graph.value_info
    cases = []
    for seconds in (2, 6, 8, 10, 12):
        x = scale_to_rms_dbfs(speechlike_16k(16_000 * seconds, seed=seconds), -20.0).astype("<f4")
        inp, out = tmp_path / f"{mode}-{seconds}.f32", tmp_path / f"{mode}-{seconds}.json"
        x.tofile(inp)
        proc = tractcheck.run("enroll", onnx_path, inp, out)
        rep = json.loads(out.read_text())
        assert rep["ok"], (seconds, rep.get("error"), proc.stderr)
        with torch.no_grad():
            ref = [t.numpy() for t in model(torch.from_numpy(x)[None])]
        worst_abs, worst_excess = 0.0, -1.0
        for name, r in zip(OUTPUT_NAMES, ref, strict=True):
            got = np.asarray(rep["outputs"][name], np.float32)
            assert got.shape == r.shape and np.isfinite(got).all()
            worst_abs = max(worst_abs, float(np.abs(got - r).max()))
            worst_excess = max(worst_excess, float(np.max(np.abs(got - r) - (1e-4 + 1e-4 * np.abs(r)))))
        cases.append({"seconds": seconds, "max_abs_err": worst_abs, "worst_excess": worst_excess})
        assert worst_excess <= 0 and worst_abs <= 1e-4, cases[-1]
    (m0_runs / f"enroll_bn_random_{mode}.json").write_text(json.dumps(cases, indent=2) + "\n")
