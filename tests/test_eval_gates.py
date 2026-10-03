import shutil
from pathlib import Path

import pytest

from cctrain.eval.gates import BUDGET_V1, GATES_V1, REQUIRED_GATES, GatesError, load_budget, load_gates


def test_gates_v1_is_frozen_and_complete():
    gs = load_gates()
    assert gs.version == 1
    assert set(REQUIRED_GATES) <= set(gs.gates)
    assert {g.cls for g in gs.gates.values()} <= {"B", "M"}
    assert gs.gates["G0"].criteria["config_ini_sha256"].startswith("415eb925")


def test_spec_levels_are_present():
    gs = load_gates()
    assert gs.gates["G1"].criteria["stage_f"]["delta_stoi_min"] == -0.003
    assert gs.gates["G2"].criteria["delta_si_sdr_db_min"] == 3.0
    assert gs.gates["G4"].criteria["trr_min_different_sex"] == 0.90
    assert gs.gates["G4"].criteria["trr_min_same_sex"] == 0.70
    assert gs.gates["G6"].criteria["tar_min"] == 0.97
    assert gs.m1["eer_max"] == {"librispeech_test_clean": 0.03, "vctk_heldout": 0.06, "cv_pt_heldout": 0.08}
    assert gs.probe_rule["go_ci_lower_min"] == 1.0
    assert gs.runtime_path["thresholds_db"] == {"min_db_thresh": -10.0, "max_db_erb_thresh": 30.0,
                                                "max_db_df_thresh": 20.0}
    assert "vad_threshold" in gs.calibrated_in_dev


def _copy(tmp_path: Path, src: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    shutil.copy(src.with_suffix(".sha256"), dst.with_suffix(".sha256"))
    return dst


def test_any_edit_breaks_the_freeze(tmp_path: Path):
    p = _copy(tmp_path, GATES_V1)
    p.write_text(p.read_text().replace("delta_si_sdr_db_min: 3.0", "delta_si_sdr_db_min: 2.0"))
    with pytest.raises(GatesError, match="congelado"):
        load_gates(p)


def test_missing_gate_is_rejected(tmp_path: Path):
    p = _copy(tmp_path, GATES_V1)
    text = p.read_text()
    start = text.index("  G7:")
    end = text.index("  G8:")
    p.write_text(text[:start] + text[end:])
    with pytest.raises(GatesError, match="G7"):
        load_gates(p, verify=False)


def test_bad_class_is_rejected(tmp_path: Path):
    p = _copy(tmp_path, GATES_V1)
    p.write_text(p.read_text().replace("  G7:\n    class: M", "  G7:\n    class: X", 1))
    with pytest.raises(GatesError, match="classe"):
        load_gates(p, verify=False)


def test_budget_is_frozen():
    b = load_budget()
    assert b.reserve_gb == 10.0
    assert b.vram_b16_reserved_gb_max == 7.0
    assert b.loader_rss_gb_max == 6.0
    assert b.t3["go_hours_max"] == 35.0
    p = BUDGET_V1
    assert p.with_suffix(".sha256").is_file()
