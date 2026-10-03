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
    scale_to_peak,
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


# --- Tarefa 10: enrollment.onnx com o encoder SpeechBrain portado (pesos reais) ---

from sb_cases import build_inputs  # noqa: E402

TRACT_ABS_TOL = 1e-4  # nível m1.tract_parity.abs_tol de gates-v1.yaml


def _tract_vs_torch(model, onnx_path, x: np.ndarray, tmp_path, tag: str) -> dict:
    inp, out = tmp_path / f"{tag}.f32", tmp_path / f"{tag}.json"
    x.astype("<f4").tofile(inp)
    proc = tractcheck.run("enroll", onnx_path, inp, out)
    rep = json.loads(out.read_text())
    assert rep["ok"], (tag, rep.get("error"), proc.stderr)
    with torch.no_grad():
        ref = [t.numpy() for t in model(torch.from_numpy(x.astype(np.float32))[None])]
    worst = 0.0
    for name, r in zip(OUTPUT_NAMES, ref, strict=True):
        got = np.asarray(rep["outputs"][name], np.float32)
        assert got.shape == r.shape and np.isfinite(got).all(), (tag, name)
        worst = max(worst, float(np.abs(got - r).max()))
    return {"case": tag, "max_abs_err": worst}


def _record_tract(runs, rows: list[dict]) -> None:
    path = runs / "t10_sb_tract.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update({r["case"]: r["max_abs_err"] for r in rows})
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def test_sb_enrollment_graph_contract(sb_enrollment_onnx):
    proto = onnx.load(str(sb_enrollment_onnx))
    assert [i.name for i in proto.graph.input] == ["audio"]
    assert [o.name for o in proto.graph.output] == list(OUTPUT_NAMES)
    # saídas concretas: só a entrada pode ter o N simbólico (tract recusa saída com dim_param)
    for o in proto.graph.output:
        dims = o.type.tensor_type.shape.dim
        assert dims and all(not d.dim_param and d.dim_value > 0 for d in dims), o.name
    assert not proto.graph.value_info
    ops = {n.op_type for n in proto.graph.node}
    # BN fica como BatchNormalization com as estatísticas reais (vem depois da ReLU, não funde em Conv); o tract a
    # converte em escala/deslocamento e a paridade tract × módulo abaixo prova a conversão
    assert "BatchNormalization" in ops
    assert "STFT" not in ops and "DFT" not in ops
    # o tract 0.19.16 ignora `mode=reflect` do Pad (preenche com zeros): a reflexão do SpeechBrain vai por Gather
    pads = [n for n in proto.graph.node if n.op_type == "Pad"]
    assert all(not any(a.name == "mode" and a.s != b"constant" for a in n.attribute) for n in pads)
    assert all(len(n.input) == 3 and n.input[2] for n in pads)  # constant_value explícito (tract)
    assert "Gather" in ops


@pytest.mark.parametrize(
    "seconds,level",
    [(2, "rms-20"), (6, "rms-20"), (8, "rms-20"), (10, "rms-20"), (12, "rms-20"),
     (2, "rms-40"), (12, "rms-40"), (2, "peak0.99"), (12, "peak0.99")],
)
def test_sb_enrollment_matches_tract(seconds, level, sb_enrollment_model, sb_enrollment_onnx, tmp_path, m1_runs):
    x = build_inputs()[f"s{seconds}_{level}"]
    row = _tract_vs_torch(sb_enrollment_model, sb_enrollment_onnx, x, tmp_path, f"{seconds}s_{level}")
    _record_tract(m1_runs, [row])
    assert row["max_abs_err"] <= TRACT_ABS_TOL, row


def _edge_signals() -> dict[str, np.ndarray]:
    n = 96_000
    rng = np.random.default_rng(0)
    impulse = np.zeros(n, np.float32)
    impulse[n // 2] = 0.99
    return {
        "silence": np.zeros(n, np.float32),
        "dc_0.1": np.full(n, 0.1, np.float32),
        "full_scale_square": np.where((np.arange(n) // 50) % 2 == 0, 1.0, -1.0).astype(np.float32),
        "white_noise_peak0.99": scale_to_peak(rng.standard_normal(n).astype(np.float32), 0.99),
        "noise_-90dBFS": scale_to_rms_dbfs(rng.standard_normal(n).astype(np.float32), -90.0),
        "single_impulse": impulse,
    }


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "name", ["silence", "dc_0.1", "full_scale_square", "white_noise_peak0.99", "noise_-90dBFS", "single_impulse"]
)
def test_sb_enrollment_edge_cases_are_finite(name, sb_enrollment_model, sb_enrollment_onnx, tmp_path, m1_runs):
    """G11: sinais de borda não geram NaN/Inf nem no módulo, nem no ORT, nem no tract; tract × módulo <= 1e-4."""
    import onnxruntime as ort

    x = _edge_signals()[name]
    sess = ort.InferenceSession(str(sb_enrollment_onnx), providers=["CPUExecutionProvider"])
    assert all(np.isfinite(o).all() for o in sess.run(None, {"audio": x[None]})), name
    row = _tract_vs_torch(sb_enrollment_model, sb_enrollment_onnx, x, tmp_path, f"edge_{name}")
    _record_tract(m1_runs, [row])
    assert row["max_abs_err"] <= TRACT_ABS_TOL, row
