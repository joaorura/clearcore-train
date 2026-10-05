"""Veredito go/no-go do M2 estritamente pelos níveis congelados (M2 Tarefa 8)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from cctrain.m2.report import Check, M2Inputs, Verdict, m2_verdict, render_checks_table

GOOD = M2Inputs(
    runtime_deviation={"error_to_signal_db": -16.5, "stage_agreement": 0.995, "path": "replica"},
    g1={
        "delta_stoi": {"ci_low": 0.01, "point": 0.05},
        "delta_si_sdr_db": {"ci_low": 5.0, "point": 6.6},
        "delta_dnsmos_ovrl": {"ci_low": 0.10, "point": 0.15},
        "delta_dnsmos_sig": {"ci_low": 0.10, "point": 0.14},
        "delta_dnsmos_bak": {"ci_low": 0.12, "point": 0.18},
        "delta_lsd_above_8k_db": {"ci_high": 0.20, "point": 0.15},
    },
    g2={
        "delta_si_sdr_db": {"point": 3.8, "ci_low": 1.2},
        "delta_dnsmos_p_ovrl": 0.25,
        "wer_ratio_to_dfnet3": 0.80,
        "wer_not_above_noisy": True,
    },
    g3={
        "delta_tsos_over_dfnet3_pp": 0.15,
        "tsos_c0_pct": 0.4,
    },
    g4={
        "trr_diff_sex": 0.95,
        "trr_same_sex": 0.78,
        "mode_sep_db": 18.2,
    },
    g5={"spk_sim_delta_ci_lower": 0.02},
    g8={"dnsmos_p_ovrl": 3.65},
    c8={"tanh_saturation_pct": 0.01, "gating_fractions": {"g1": 0.9, "g2": 0.8}},
)


def _checks(v: Verdict) -> dict[str, Check]:
    return {c.name: c for c in v.checks}


def test_m2_all_levels_met_is_go():
    v = m2_verdict(GOOD)
    assert v.go
    assert len(v.blocking_failures) == 0
    c = _checks(v)
    assert c["G1_delta_stoi_ci_low"].passed
    assert c["G1_delta_si_sdr_ci_low"].passed
    assert c["G1_delta_lsd_above_8k_ci_high"].passed
    assert c["G2_delta_si_sdr_point"].passed
    assert c["G2_delta_si_sdr_ci_lower_gt_0"].passed
    assert c["G3_tsos_max_over_dfnet3_pp"].passed
    assert c["G4_trr_different_sex"].passed
    assert c["G4_mode_separation_db"].passed
    assert v.gates_sha256 and v.selection_sha256 and v.deviation_sha256


@pytest.mark.parametrize("patch,failing", [
    # G1: violações pelo limite do IC
    ({"g1": {**GOOD.g1, "delta_stoi": {"ci_low": -0.004}}}, "G1_delta_stoi_ci_low"),
    ({"g1": {**GOOD.g1, "delta_si_sdr_db": {"ci_low": -0.51}}}, "G1_delta_si_sdr_ci_low"),
    ({"g1": {**GOOD.g1, "delta_dnsmos_ovrl": {"ci_low": -0.06}}}, "G1_delta_dnsmos_ovrl_ci_low"),
    ({"g1": {**GOOD.g1, "delta_lsd_above_8k_db": {"ci_high": 0.55}}}, "G1_delta_lsd_above_8k_ci_high"),
    # G2: violações de separação e WER
    ({"g2": {**GOOD.g2, "delta_si_sdr_db": {"point": 2.9, "ci_low": 0.5}}}, "G2_delta_si_sdr_point"),
    ({"g2": {**GOOD.g2, "delta_si_sdr_db": {"point": 3.5, "ci_low": -0.1}}}, "G2_delta_si_sdr_ci_lower_gt_0"),
    ({"g2": {**GOOD.g2, "delta_dnsmos_p_ovrl": 0.10}}, "G2_delta_dnsmos_p_ovrl"),
    ({"g2": {**GOOD.g2, "wer_ratio_to_dfnet3": 0.86}}, "G2_wer_ratio_to_dfnet3"),
    ({"g2": {**GOOD.g2, "wer_not_above_noisy": False}}, "G2_wer_not_above_noisy"),
    # G3: TSOS acima de +0.5 p.p. vs DFNet3
    ({"g3": {**GOOD.g3, "delta_tsos_over_dfnet3_pp": 0.55}}, "G3_tsos_max_over_dfnet3_pp"),
    # G4: TRR
    ({"g4": {**GOOD.g4, "trr_diff_sex": 0.89}}, "G4_trr_different_sex"),
    ({"g4": {**GOOD.g4, "trr_same_sex": 0.69}}, "G4_trr_same_sex"),
    ({"g4": {**GOOD.g4, "mode_sep_db": 14.5}}, "G4_mode_separation_db"),
])
def test_m2_each_blocking_level_fails_closed_on_a_miss(patch, failing):
    v = m2_verdict(replace(GOOD, **patch))
    assert not v.go
    assert failing in v.blocking_failures
    assert not _checks(v)[failing].passed


def test_m2_missing_or_nan_measurements_fail():
    for patch in [
        {"g1": {**GOOD.g1, "delta_stoi": {"ci_low": math.nan}}},
        {"g2": {**GOOD.g2, "delta_dnsmos_p_ovrl": None}},
        {"g3": {**GOOD.g3, "delta_tsos_over_dfnet3_pp": math.nan}},
        {"g4": {**GOOD.g4, "mode_sep_db": None}},
    ]:
        v = m2_verdict(replace(GOOD, **patch))
        assert not v.go


def test_m2_render_table():
    v = m2_verdict(GOOD)
    tbl = render_checks_table(v)
    assert "| Gate / Métrica |" in tbl
    assert "G1_delta_stoi_ci_low" in tbl
    assert "G2_delta_si_sdr_point" in tbl
    assert "G3_tsos_max_over_dfnet3_pp" in tbl
    assert "G4_trr_different_sex" in tbl
    assert "**PASS**" in tbl
