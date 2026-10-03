"""Veredito go/no-go do M1 estritamente pelos níveis congelados (M1 Tarefa 12)."""

from __future__ import annotations

import math

import pytest

from cctrain.m1.report import M1Inputs, m1_verdict, render_checks_table

GOOD = M1Inputs(
    eer={"librispeech_test_clean": 0.014, "vctk_heldout": 0.018, "cv_pt_heldout": 0.07},
    tract={"durations_s": [2, 6, 8, 10, 12], "max_abs_diff": 6e-6, "finite": True},
    functional_c3={"ci_low": 0.4, "delta_si_sdr_db": 1.0, "ci_high": 1.7, "n_speakers": 48, "same_sex": True,
                   "sir_db": 0.0, "stage": "C"},
    base_sha256=("a" * 64, "a" * 64),
    probe={"pss_gb_max": 3.2, "vram_peak_reserved_gb": 5.0},
    disk_free_gb=40.0,
    deviation={"error_to_signal_db": -17.0, "stage_agreement": 0.95, "path": "offline"},
    ab1=None,
)


def _checks(v):
    return {c.name: c for c in v.checks}


def test_all_levels_met_is_go_with_levels_from_the_frozen_files():
    v = m1_verdict(GOOD)
    assert v.go and all(c.passed for c in v.checks if c.blocking)
    c = _checks(v)
    assert c["eer_cv_pt_heldout"].level == "<= 0.08" and c["eer_vctk_heldout"].level == "<= 0.06"
    assert c["eer_librispeech_test_clean"].level == "<= 0.03"
    assert c["tract_parity"].level == "<= 0.0001"
    assert c["functional_c3_ci_lower"].level == "> 0.0"
    assert c["budget_ram_pss_gb"].level == "<= 6.0" and c["budget_vram_reserved_gb"].level == "<= 7.0"
    assert v.gates_sha256 and v.budget_sha256 and v.deviation_sha256


@pytest.mark.parametrize("patch,failing", [
    ({"eer": {**GOOD.eer, "cv_pt_heldout": 0.0801}}, "eer_cv_pt_heldout"),
    ({"eer": {**GOOD.eer, "librispeech_test_clean": 0.0301}}, "eer_librispeech_test_clean"),
    ({"tract": {**GOOD.tract, "max_abs_diff": 1.1e-4}}, "tract_parity"),
    ({"tract": {**GOOD.tract, "durations_s": [2, 6, 8, 10]}}, "tract_parity"),
    ({"tract": {**GOOD.tract, "finite": False}}, "tract_parity"),
    ({"functional_c3": {**GOOD.functional_c3, "ci_low": 0.0}}, "functional_c3_ci_lower"),
    ({"functional_c3": {**GOOD.functional_c3, "ci_low": -0.1}}, "functional_c3_ci_lower"),
    ({"base_sha256": ("a" * 64, "b" * 64)}, "base_frozen"),
    ({"probe": {**GOOD.probe, "pss_gb_max": 6.01}}, "budget_ram_pss_gb"),
    ({"probe": {**GOOD.probe, "vram_peak_reserved_gb": 7.01}}, "budget_vram_reserved_gb"),
    ({"disk_free_gb": 9.9}, "budget_disk_free_gb"),
])
def test_each_blocking_level_fails_closed_on_a_miss(patch, failing):
    from dataclasses import replace

    v = m1_verdict(replace(GOOD, **patch))
    assert not v.go and not _checks(v)[failing].passed


def test_boundary_values_pass_where_the_level_is_inclusive():
    from dataclasses import replace

    v = m1_verdict(replace(GOOD, eer={"librispeech_test_clean": 0.03, "vctk_heldout": 0.06, "cv_pt_heldout": 0.08},
                           tract={**GOOD.tract, "max_abs_diff": 1e-4}, probe={"pss_gb_max": 6.0,
                                                                              "vram_peak_reserved_gb": 7.0}))
    assert v.go


def test_missing_or_nan_measurements_are_a_failure_not_a_pass():
    from dataclasses import replace

    for patch in ({"eer": {**GOOD.eer, "vctk_heldout": math.nan}},
                  {"probe": {"pss_gb_max": 3.0, "vram_peak_reserved_gb": None}},
                  {"functional_c3": {**GOOD.functional_c3, "ci_low": math.nan}},
                  {"eer": {"vctk_heldout": 0.01}}):
        assert not m1_verdict(replace(GOOD, **patch)).go


def test_functional_scenario_must_be_the_frozen_one():
    from dataclasses import replace

    for bad in ({"same_sex": False}, {"sir_db": 5.0}, {"stage": "F"}):
        assert not m1_verdict(replace(GOOD, functional_c3={**GOOD.functional_c3, **bad})).go


def test_deviation_gate_and_ab1_never_block_and_the_table_renders_every_check():
    from dataclasses import replace

    v = m1_verdict(replace(GOOD, deviation={"error_to_signal_db": -10.0, "stage_agreement": 0.5, "path": "replica"}))
    assert v.go
    dev = _checks(v)["deviation_gate"]
    assert not dev.blocking and dev.passed is False and "replica" in dev.note
    md = render_checks_table(v)
    assert md.count("\n") >= len(v.checks) + 1 and "eer_vctk_heldout" in md and "não bloqueia" in md
