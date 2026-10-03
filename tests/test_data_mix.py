import numpy as np
import pytest

from cctrain.data.mix import MixKind, make_mixture, mixture_fractions, power, scale_for_ratio


@pytest.mark.parametrize("s", [0.0, 0.05, 0.10])
def test_fractions_follow_spec(s):
    f = mixture_fractions(s)
    assert abs(sum(f.values()) - 1.0) < 1e-12
    assert f[MixKind.INTERFERER_ONLY] == s


def test_fractions_with_ten_percent_silence():
    f = mixture_fractions(0.10)
    assert [round(f[k], 10) for k in MixKind] == [0.35, 0.20, 0.35, 0.10]


def test_scale_for_ratio_hits_target():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(48_000), 3 * rng.standard_normal(48_000)
    b2 = scale_for_ratio(a, b, 7.5)
    assert abs(10 * np.log10(power(a) / power(b2)) - 7.5) < 1e-9


def test_mixture_kinds_and_targets():
    rng = np.random.default_rng(1)
    t, i, n = (0.1 * rng.standard_normal(48_000) for _ in range(3))
    io = make_mixture(t, [i], n, kind=MixKind.INTERFERER_ONLY, snr_db=10.0, sir_db=float("nan"))
    assert not np.any(io.clean)
    tn = make_mixture(t, [], n, kind=MixKind.TARGET_NOISE, snr_db=5.0, sir_db=float("nan"))
    measured = 10 * np.log10(power(tn.clean) / power(tn.noisy - tn.clean))
    assert abs(measured - 5.0) < 1e-4
    loud = make_mixture(10 * t, [10 * i], 10 * n, kind=MixKind.TARGET_INTERFERER_NOISE, snr_db=0.0, sir_db=0.0)
    assert np.max(np.abs(loud.noisy)) <= 0.99 + 1e-6 and np.max(np.abs(loud.clean)) <= 0.99 + 1e-6
