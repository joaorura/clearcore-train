"""G4 (TRR a 15 dB, separação de modos) e C8 (enrollment inválido) — M2 Tarefa 3, sinais sintéticos de resposta conhecida."""

from __future__ import annotations

import numpy as np
import pytest

from cctrain.eval import c8, trr

SR = 48_000


def burst(seconds=3.0, seed=0, sr=SR):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2  # sempre ativo
    return (0.1 * rng.standard_normal(t.size) * env).astype(np.float32)


def db(x):
    return 10 ** (x / 20.0)


def test_trr_identity_zero_and_silence_one():
    x = burst()
    assert trr.trr(x, x, SR)[0] == 0.0  # nada atenuado
    assert trr.trr(np.zeros_like(x), x, SR)[0] == 1.0  # saída muda: todos os quadros rejeitados


@pytest.mark.parametrize("atten,thr,expected", [(20.0, 15.0, 1.0), (12.0, 15.0, 0.0), (12.0, 10.0, 1.0)])
def test_trr_threshold(atten, thr, expected):
    x = burst()
    assert trr.trr(x * db(-atten), x, SR, atten_db=thr)[0] == expected


def test_trr_counts_only_active_input_frames():
    x = np.concatenate([burst(1.0), np.zeros(SR)])  # 1 s ativo, 1 s de silêncio digital
    cfg = trr.TrrConfig()
    act = trr.active_frames(x, SR, cfg)
    assert 0.4 < act.mean() < 0.55
    assert trr.trr(x, x, SR)[0] == 0.0  # os quadros mudos da entrada não contam como "rejeitados"
    assert np.isnan(trr.trr(np.zeros(SR, np.float32), np.zeros(SR, np.float32), SR)[0])  # sem quadros ativos


def test_median_attenuation_and_curve():
    x = burst()
    assert abs(trr.median_attenuation_db(x * db(-20.0), x, SR)[0] - 20.0) < 0.05
    curve = trr.trr_curve(x * db(-12.0), x, SR, thresholds_db=(6, 10, 15, 20))
    assert [curve[k][0] for k in (6, 10, 15, 20)] == [1.0, 1.0, 0.0, 0.0]


def test_mode_separation_is_profile_minus_no_profile_attenuation():
    x = burst()
    sep = trr.mode_separation_db(x * db(-25.0), x * db(-5.0), x, SR)
    assert abs(sep[0] - 20.0) < 0.05
    assert trr.mode_separation_db(x, x, x, SR)[0] == 0.0


def test_per_utterance_arrays():
    xs = [burst(2.0, 0), burst(2.5, 1)]
    out = trr.trr([x * db(-20) for x in xs], xs, SR)
    assert out.shape == (2,) and (out == 1.0).all()


# ---- C8 ----
@pytest.mark.parametrize("kind", c8.INVALID_KINDS)
def test_invalid_enrollments_are_16k_finite_and_deterministic(kind):
    other = burst(8.0, seed=9, sr=16_000)
    a = c8.make_invalid_enrollment(kind, np.random.default_rng(0), other_speech=other)
    b = c8.make_invalid_enrollment(kind, np.random.default_rng(0), other_speech=other)
    assert a.dtype == np.float32 and np.isfinite(a).all() and (a == b).all()
    assert 0.4 * 16_000 <= a.size <= 12.5 * 16_000
    if kind == "silence":
        assert not a.any()
    if kind == "short":
        assert a.size < 1.0 * 16_000


def test_other_speaker_requires_speech():
    with pytest.raises(ValueError, match="other_speech"):
        c8.make_invalid_enrollment("other_speaker", np.random.default_rng(0))
    with pytest.raises(ValueError, match="tipo"):
        c8.make_invalid_enrollment("nope", np.random.default_rng(0))


def test_c8_verdict_margin_is_one_pp_over_no_profile():
    spk = {"s1": 1.0, "s2": 2.0, "s3": 3.0}
    ok = {k: v + 0.5 for k, v in spk.items()}  # +0,5 p.p.
    bad = {k: v + 3.0 for k, v in spk.items()}  # +3 p.p.
    v = c8.c8_verdict({"silence": ok, "white_noise": bad}, spk, resamples=300, seed=0)
    assert v["margin_pp"] == 1.0
    assert v["kinds"]["silence"]["pass"] is True and v["kinds"]["white_noise"]["pass"] is False
    assert abs(v["kinds"]["silence"]["delta_pp"]["mean"] - 0.5) < 1e-9
    assert v["pass"] is False  # reprova se qualquer tipo reprova
    ci = c8.c8_verdict({"silence": ok}, spk, reading="ci_upper", resamples=300, seed=0)
    assert ci["kinds"]["silence"]["pass"] is True and ci["reading"] == "ci_upper"
