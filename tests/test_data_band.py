import numpy as np
from scipy.signal import firwin, lfilter

from cctrain.data.band import effective_band_hz


def test_band_of_lowpassed_noise():
    x = np.random.default_rng(0).standard_normal(48_000 * 3)
    y = lfilter(firwin(511, 4000, fs=48_000), [1.0], x)
    assert 3500 <= effective_band_hz(y, 48_000) <= 4300


def test_band_of_white_noise_reaches_nyquist_region():
    x = np.random.default_rng(1).standard_normal(16_000 * 3)
    assert effective_band_hz(x, 16_000) >= 7500


def test_silence_has_zero_band():
    assert effective_band_hz(np.zeros(16_000), 16_000) == 0.0
