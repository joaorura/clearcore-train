import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from cctrain import tractcheck

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
