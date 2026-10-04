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


# --- Tarefa 11: conferência contra spec §4.4 ---


def _rng_signals(seed=2, n=48_000):
    rng = np.random.default_rng(seed)
    return tuple(0.1 * rng.standard_normal(n) for _ in range(3))


def test_noise_snr_is_measured_against_the_target_not_the_interferer():
    t, i, n = _rng_signals()
    for snr in (-5.0, 5.0, 20.0):
        m = make_mixture(t, [i], n, kind=MixKind.TARGET_INTERFERER_NOISE, snr_db=snr, sir_db=0.0)
        no_noise = make_mixture(t, [i], None, kind=MixKind.TARGET_INTERFERER, snr_db=snr, sir_db=0.0)
        target_part = m.clean.astype(np.float64)
        noise_part = m.noisy.astype(np.float64) - no_noise.noisy.astype(np.float64) * _gain(m, no_noise)
        assert abs(10 * np.log10(power(target_part) / power(noise_part)) - snr) < 0.05


def _gain(m, ref):
    # os dois lados são normalizados por pico; recupera a razão entre as escalas pelo alvo seco
    return float(np.dot(m.clean, ref.clean) / np.dot(ref.clean, ref.clean))


def test_dry_target_is_aligned_to_the_direct_path_delay_of_the_rir():
    t, i, n = _rng_signals()
    h = np.zeros(4800)
    h[137], h[900] = 0.9, 0.3  # caminho direto em 137 amostras, eco depois
    m = make_mixture(t, [], None, kind=MixKind.TARGET_NOISE, snr_db=5.0, sir_db=float("nan"), rir_target=h)
    assert not np.any(m.clean[:137])
    shifted = np.concatenate([np.zeros(137), t])[: len(t)]
    k = float(np.dot(m.clean, shifted) / np.dot(shifted, shifted))
    assert np.allclose(m.clean, k * shifted, atol=1e-6)
    # sem RIR não há deslocamento
    m0 = make_mixture(t, [], None, kind=MixKind.TARGET_NOISE, snr_db=5.0, sir_db=float("nan"))
    assert np.allclose(m0.clean, float(np.dot(m0.clean, t) / np.dot(t, t)) * t, atol=1e-6)


def test_interferer_only_has_no_sir_and_a_fixed_level():
    t, i, n = _rng_signals()
    a = make_mixture(t, [i], None, kind=MixKind.INTERFERER_ONLY, snr_db=5.0, sir_db=-5.0)
    b = make_mixture(t, [i], None, kind=MixKind.INTERFERER_ONLY, snr_db=5.0, sir_db=25.0)
    assert np.array_equal(a.noisy, b.noisy) and not np.any(a.clean)
    assert np.isnan(a.sir_db) and np.isnan(b.sir_db)
    rms_dbfs = 10 * np.log10(power(a.noisy))
    assert abs(rms_dbfs - (-25.0)) < 0.05


def test_interferer_only_noise_is_measured_against_the_interferers():
    t, i, n = _rng_signals()
    quiet = make_mixture(t, [i], None, kind=MixKind.INTERFERER_ONLY, snr_db=0.0, sir_db=float("nan"))
    noisy = make_mixture(t, [i], n, kind=MixKind.INTERFERER_ONLY, snr_db=0.0, sir_db=float("nan"))
    g = float(np.dot(noisy.noisy, quiet.noisy) / np.dot(quiet.noisy, quiet.noisy))
    noise_part = noisy.noisy - g * quiet.noisy
    assert abs(10 * np.log10(power(g * quiet.noisy) / power(noise_part))) < 0.05


def test_mix_rng_varies_with_epoch_and_index_but_is_reproducible():
    from cctrain.data.mix import mix_rng

    a = mix_rng(7, 0, 3).random(4)
    assert np.array_equal(a, mix_rng(7, 0, 3).random(4))
    assert not np.array_equal(a, mix_rng(7, 1, 3).random(4))
    assert not np.array_equal(a, mix_rng(7, 0, 4).random(4))
    assert not np.array_equal(a, mix_rng(8, 0, 3).random(4))


def test_clean_all_is_the_sum_of_the_dry_speakers_at_the_mixture_level():
    rng = np.random.default_rng(3)
    t = rng.standard_normal(48_000).astype(np.float32) * 0.1
    i1 = rng.standard_normal(48_000).astype(np.float32) * 0.2
    i2 = rng.standard_normal(48_000).astype(np.float32) * 0.05
    plain = make_mixture(t, [], None, kind=MixKind.TARGET_NOISE, snr_db=0, sir_db=float("nan"))
    assert np.array_equal(plain.clean_all, plain.clean)
    m = make_mixture(t, [i1, i2], None, kind=MixKind.TARGET_INTERFERER, snr_db=0, sir_db=5.0)
    rest = m.clean_all.astype(np.float64) - m.clean.astype(np.float64)
    assert 10 * np.log10(np.mean(m.clean.astype(np.float64) ** 2) / np.mean(rest**2)) == pytest.approx(5.0, abs=1e-3)
    assert np.allclose(m.clean_all, m.noisy, atol=1e-6)  # sem ruído e sem RIR a mistura é a soma das falas secas
    only = make_mixture(t, [i1], None, kind=MixKind.INTERFERER_ONLY, snr_db=0, sir_db=float("nan"))
    assert not only.clean.any() and np.allclose(only.clean_all, only.noisy, atol=1e-6)


def test_clean_all_follows_the_direct_path_delay_of_each_rir():
    rng = np.random.default_rng(4)
    t = rng.standard_normal(4_800).astype(np.float32)
    i1 = rng.standard_normal(4_800).astype(np.float32)
    h = np.zeros(300)
    h[50] = 1.0
    m = make_mixture(t, [i1], None, kind=MixKind.TARGET_INTERFERER, snr_db=0, sir_db=0.0, rir_target=h,
                     rir_interferer=h)
    assert np.allclose(m.clean_all, m.noisy, atol=1e-5)  # RIR de impulso atrasado: alvo seco alinhado = mistura
