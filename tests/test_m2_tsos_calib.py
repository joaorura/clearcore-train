import numpy as np
import pytest

from cctrain.eval.gates import GatesError
from cctrain.m2 import tsos_calib as tc

SR = 48_000


def _pairs(n=6):
    rng = np.random.default_rng(0)
    out, spk = [], []
    for k in range(n):
        ref = (0.05 * rng.standard_normal(SR)).astype(np.float32)
        est = ref * np.float32(0.4) + (0.01 * rng.standard_normal(SR)).astype(np.float32)  # supressão parcial
        out.append((est, ref))
        spk.append(f"s{k % 3}")
    return out, spk


def test_convention_is_fixed_float_scale_one_and_papers_parameters():
    c = tc.CONVENTION
    assert (c.amplitude_scale, c.p, c.gamma, c.win_ms, c.hop_ms) == (1.0, 0.3, 0.1, 20.0, 10.0)
    assert not hasattr(tc, "calibrate_amplitude_scale") and not hasattr(tc, "scale_grid")  # sem grade nem ajuste de escala


def test_mean_per_speaker_weights_speakers_equally():
    assert tc.mean_per_speaker([0.0, 0.0, 0.0, 10.0], ["a", "a", "a", "b"]) == pytest.approx(5.0)


def test_measure_uses_the_fixed_convention_and_does_not_choose_a_scale():
    pairs, spk = _pairs()
    v = tc.measure_tsos(pairs, spk, SR)
    assert v == tc.measure_tsos(pairs, spk, SR, tc.CONVENTION)
    assert 0.0 <= v <= 100.0


def test_paper_bands_are_ts3_for_clean_target_not_ts1():
    assert tc.PAPER_BASELINE_PCT["TS3"] == (1.47, 2.42)  # DCCRN sem personalização, R06 l.212 (ICR, VCTK)
    assert tc.paper_band("C0c") == (1.47, 2.42)
    assert tc.paper_band("C0") == (5.33, 7.32) and tc.paper_band("C4") == (5.13, 9.01)
    assert tc.paper_band("C3") is None


@pytest.mark.parametrize(("c0c", "status", "freeze"), [(1.9, "in_band", True), (1.2, "near_band", True),
                                                     (4.5, "near_band", True), (0.2, "far_outside", False),
                                                     (7.07, "far_outside", False), (0.0, "far_outside", False)])
def test_plausibility_gates_freezing_on_ts3_only(c0c, status, freeze):
    r = tc.plausibility({"C0c": c0c, "C0": 1000.0, "C4": 0.001, "C3": 3.0})
    assert r["conditions"]["C0c"]["status"] == status and r["freeze"] is freeze
    assert r["conditions"]["C0"]["status"] == "far_outside" and r["conditions"]["C3"]["status"] == "no_paper_counterpart"
    assert r["decision"] == ("convenção congelada" if freeze else tc.OWNER_DECISION)
    assert "gates-v3" in tc.OWNER_DECISION


def test_plausibility_needs_the_gating_condition():
    with pytest.raises(ValueError, match="C0c"):
        tc.plausibility({"C0": 5.0})


def test_load_tsos_config_needs_freeze(tmp_path):
    p = tmp_path / "tsos-config-v1.yaml"
    p.write_text("version: 1\nconfig: {p: 0.3, gamma: 0.1, win_ms: 20.0, hop_ms: 10.0, amplitude_scale: 0.5}\n")
    with pytest.raises(GatesError, match="não está congelado"):
        tc.load_tsos_config(p)
    from cctrain.hashing import sha256_file

    p.with_suffix(".sha256").write_text(f"{sha256_file(p)}  {p.name}\n")
    cfg = tc.load_tsos_config(p)
    assert cfg.amplitude_scale == 0.5 and cfg.p == 0.3 and cfg.gamma == 0.1
    p.write_text(p.read_text().replace("0.5", "0.7"))
    with pytest.raises(GatesError, match="alterado"):
        tc.load_tsos_config(p)
