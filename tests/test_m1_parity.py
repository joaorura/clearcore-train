"""Paridade tract do enrollment.onnx final (M1 Tarefa 12): tract × módulo PyTorch e ORT × módulo."""

from __future__ import annotations

import numpy as np
import pytest

from cctrain.export.enrollment import build_prototype, export_enrollment_onnx, scale_to_rms_dbfs, speechlike_16k
from cctrain.m1.parity import enrollment_parity, summarize_parity

pytestmark = pytest.mark.tract


@pytest.fixture(scope="module")
def proto(tmp_path_factory):
    model = build_prototype(seed=0, context_mode="expand", out_std=0.1)
    return model, export_enrollment_onnx(model, tmp_path_factory.mktemp("par") / "enrollment.onnx")


def test_parity_report_per_case_and_summary(proto, tmp_path):
    model, onnx_path = proto
    cases = {f"{s}s_rms-20": scale_to_rms_dbfs(speechlike_16k(16_000 * s, s), -20.0) for s in (2, 6)}
    cases["silence_6s"] = np.zeros(96_000, np.float32)
    rows = enrollment_parity(model, onnx_path, cases, tmp_path)
    assert {r["case"] for r in rows} == set(cases)
    for r in rows:  # o protótipo do M0 (aleatório) amplifica o erro no silêncio; aqui só a estrutura e a finitude
        assert r["finite"] and (r["case"] == "silence_6s" or (r["tract_max_abs_diff"] <= 1e-4 and r["ort_ok"])), r
    s = summarize_parity(rows, durations_s=(2, 6), required=("2s_rms-20", "6s_rms-20"))
    assert s["durations_s"] == [2, 6] and s["max_abs_diff"] <= 1e-4 and s["finite"] is True
    assert s["max_abs_diff"] == max(r["tract_max_abs_diff"] for r in rows if r["case"] != "silence_6s")


def test_summary_requires_every_listed_case_and_flags_non_finite():
    rows = [{"case": "2s", "tract_max_abs_diff": 1e-6, "finite": True, "ort_ok": True, "seconds": 2}]
    s = summarize_parity(rows, durations_s=(2, 6), required=("2s", "6s"))
    assert s["durations_s"] == [2] and s["missing"] == ["6s"]
    rows.append({"case": "6s", "tract_max_abs_diff": 1e-6, "finite": False, "ort_ok": True, "seconds": 6})
    assert summarize_parity(rows, durations_s=(2, 6), required=("2s", "6s"))["finite"] is False
