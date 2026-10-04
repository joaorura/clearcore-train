import math

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


def test_grid_contains_float_and_int16_scales():
    g = tc.scale_grid()
    assert 1.0 in g and 32768.0 in g and g == sorted(g)


def test_mean_per_speaker_weights_speakers_equally():
    assert tc.mean_per_speaker([0.0, 0.0, 0.0, 10.0], ["a", "a", "a", "b"]) == pytest.approx(5.0)


def test_tsos_grows_with_scale_and_choice_is_nearest_in_log():
    pairs, spk = _pairs()
    res = tc.calibrate_amplitude_scale(pairs, spk, SR, anchor_pct=30.0)
    vals = [r["tsos_pct"] for r in res["table"]]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))  # eq. 3 é monotônica na escala (LHS ~ a^2p, RHS ~ a^p)
    assert vals[0] < 30.0 < vals[-1]
    best = min((r for r in res["table"] if r["tsos_pct"] > 0), key=lambda r: abs(math.log(r["tsos_pct"] / 30.0)))
    assert res["chosen_amplitude_scale"] == best["amplitude_scale"]
    assert res["anchor_pct"] == 30.0


def test_anchor_is_the_papers_non_personalized_ts3_mean():
    assert tc.ANCHOR_PCT == pytest.approx((9.01 + 5.13) / 2)


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
