import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from cctrain.export.enrollment import (
    OUTPUT_NAMES,
    build_prototype,
    export_enrollment_onnx,
    scale_to_peak,
    scale_to_rms_dbfs,
    speechlike_16k,
)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("mode", ["expand", "broadcast"])
def test_export_contract_and_p1_parity(tmp_path, mode):
    model = build_prototype(seed=0, context_mode=mode)
    path = export_enrollment_onnx(model, tmp_path / f"enroll-{mode}.onnx")
    proto = onnx.load(str(path))
    assert proto.ir_version == 8
    assert proto.opset_import[0].version == 13
    assert [i.name for i in proto.graph.input] == ["audio"]
    assert [o.name for o in proto.graph.output] == list(OUTPUT_NAMES)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for n in (32_000, 96_000, 192_000):
        x = scale_to_rms_dbfs(speechlike_16k(n, seed=n), -20.0)[None]
        with torch.no_grad():
            ref = [t.numpy() for t in model(torch.from_numpy(x))]
        got = sess.run(None, {"audio": x})
        for name, r, g in zip(OUTPUT_NAMES, ref, got):
            assert g.shape == ((192,) if name == "embedding" else (256,))
            np.testing.assert_allclose(g, r, atol=1e-5, rtol=1e-6, err_msg=f"{mode} {name} N={n}")


def test_outputs_within_runtime_limits(tmp_path):
    model = build_prototype(seed=1, context_mode="expand", out_std=5.0)
    for x in (scale_to_rms_dbfs(speechlike_16k(32_000, 1), -39.9), scale_to_peak(speechlike_16k(192_000, 2), 0.99)):
        with torch.no_grad():
            g_enc, b_enc, g_df, b_df, emb = model(torch.from_numpy(x)[None])
        for g in (g_enc, g_df):
            assert torch.isfinite(g).all() and g.min() >= 0.001 and g.max() <= 100.0
        for b in (b_enc, b_df):
            assert torch.isfinite(b).all() and b.abs().max() <= 50.0
        assert abs(float(emb.norm()) - 1.0) < 1e-4


def test_level_helpers():
    x = speechlike_16k(16_000, 0)
    y = scale_to_rms_dbfs(x, -40.0)
    assert abs(10 * np.log10(np.mean(y.astype(np.float64) ** 2)) + 40.0) < 1e-3
    assert abs(np.max(np.abs(scale_to_peak(x, 0.99))) - 0.99) < 1e-6
