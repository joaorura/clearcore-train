"""Propriedades do gerador v4 que não dependem dos dados: VAD relativo, envelopes, held-out e faixas de sorteio."""

import numpy as np

from cctrain.data.mix_v4 import (
    HOP,
    _level,
    _noise_mode,
    _sir,
    active_rms_db,
    is_heldout_noise,
    is_heldout_speech,
    rel_activity,
    temporal_masks,
)


def test_heldout_patterns():
    assert is_heldout_speech("raw/speech-joao-messias/joao_messias_take4_0003.wav")
    assert is_heldout_speech("raw/speech-joao-messias/joao_messias_take3_0003.wav")
    assert not is_heldout_speech("raw/speech-joao-messias/joao_messias_take2_0003.wav")
    assert not is_heldout_noise("raw/noise-local-mics/wavs/mic_obs_3mic_combo_20261008_143002_seg0001_raw.wav")
    assert not is_heldout_noise("raw/noise-local-mics/wavs/mic_builtin_20261006_105855_seg0001_raw.wav")


def test_relative_vad_works_at_low_level():
    rng = np.random.default_rng(0)
    x = np.zeros(400 * HOP)
    x[100 * HOP : 200 * HOP] = rng.standard_normal(100 * HOP) * 10 ** (-50 / 20)
    act = rel_activity(x, 400)
    assert act[100:200].all() and not act[:100].any() and not act[200:].any()
    assert abs(active_rms_db(x, 400) - (-50.0)) < 0.2


def test_sir_and_level_ranges():
    rng = np.random.default_rng(1)
    s = np.array([_sir(rng, 2) for _ in range(5000)])
    assert s.min() >= -15 and s.max() <= 30
    assert 0.17 < np.mean(s > 15) < 0.23 and 0.12 < np.mean(s < -10) < 0.18
    lv = np.array([_level(rng)[0] for _ in range(5000)])
    assert lv.min() >= -50 and lv.max() <= -12 and 0.08 < np.mean(lv < -40) < 0.16
    modes = [_noise_mode(rng, 2)[0] for _ in range(5000)]
    assert 0.17 < np.mean([m == "silent" for m in modes]) < 0.23


def test_temporal_masks_absent_target():
    rng = np.random.default_rng(2)
    for _ in range(50):
        pat, on = temporal_masks(rng, 3, 400, target_present=False)
        assert not on[0].any() and on[1:].any()
