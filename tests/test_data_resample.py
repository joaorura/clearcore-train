import json

import numpy as np
import pytest

from cctrain.data import resample as rs
from cctrain.hashing import sha256_bytes

# SHA-256 dos coeficientes f32 calculado no planejamento (numpy do sistema). Se divergir, pare e reporte.
PLANNING_COEFFS_SHA256 = "d230da4dc7dd9ebbdee979dc33a11d3ee61cf2fe29477c768745442865fd7043"


def test_design_matches_published_file_bit_exact():
    assert np.array_equal(rs.design_coefficients(), rs.load_coefficients())


def test_coefficients_hash_is_recorded_and_matches_planning():
    raw = rs.design_coefficients().astype("<f4").tobytes()
    assert sha256_bytes(raw) == (rs.V1_DIR / "coeffs.sha256").read_text().split()[0]
    assert sha256_bytes(raw) == PLANNING_COEFFS_SHA256


def test_linear_phase_and_unit_dc_gain():
    h = rs.load_coefficients().astype(np.float64)
    assert h.shape == (241,)
    assert np.max(np.abs(h - h[::-1])) <= 1e-9
    assert abs(h.sum() - 1.0) < 1e-6


def test_matches_scipy_firwin():
    from scipy.signal import firwin

    ref = firwin(241, 7600.0, window=("kaiser", 8.6), fs=48000.0)
    assert np.max(np.abs(ref.astype(np.float32) - rs.design_coefficients())) <= 1e-8


def test_frequency_response():
    h = rs.load_coefficients().astype(np.float64)
    n = 1 << 18
    mag = 20 * np.log10(np.abs(np.fft.rfft(h, n)) + 1e-300)
    f = np.fft.rfftfreq(n, 1 / 48000)
    assert np.max(np.abs(mag[f <= 7000])) <= 1e-3
    assert np.max(mag[f >= 8200]) <= -85.0
    assert abs(mag[np.argmin(np.abs(f - 7600))] + 6.02) <= 0.1


@pytest.mark.parametrize("n", range(0, 11))
def test_output_length_is_ceil_third(n):
    assert len(rs.resample_48k_to_16k(np.zeros(n, np.float32))) == (n + 2) // 3


def test_impulse_response_is_delay_compensated():
    h64 = rs.load_coefficients().astype(np.float64)
    x = np.zeros(3000, np.float32)
    x[1500] = 1.0
    y = rs.resample_48k_to_16k(x)
    for n in range(len(y)):
        k = 3 * n + 120 - 1500
        want = np.float32(h64[k]) if 0 <= k < 241 else np.float32(0.0)
        assert y[n] == want, n


def test_sine_passes_with_zero_delay():
    t = np.arange(48000) / 48000.0
    y = rs.resample_48k_to_16k((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
    n = np.arange(len(y))
    ref = 0.5 * np.sin(2 * np.pi * 1000 * n / 16000.0)
    assert np.max(np.abs(y[100:-100] - ref[100:-100])) <= 1e-3


def test_goldens_reproduce_bit_exact():
    manifest = json.loads((rs.V1_DIR / "manifest.json").read_text())
    inputs = rs.golden_inputs()
    assert set(manifest["golden"]) == set(inputs)
    for name, x in inputs.items():
        entry = manifest["golden"][name]
        x_file = np.fromfile(rs.V1_DIR / "golden" / f"{name}.in.f32", dtype="<f4")
        y_file = np.fromfile(rs.V1_DIR / "golden" / f"{name}.out.f32", dtype="<f4")
        assert np.array_equal(x_file, x), name
        assert np.array_equal(y_file, rs.resample_48k_to_16k(x)), name
        assert sha256_bytes(x_file.tobytes()) == entry["in_sha256"]
        assert sha256_bytes(y_file.tobytes()) == entry["out_sha256"]
        assert entry["n_out"] == rs.output_length(entry["n_in"])


SRS = (16_000, 24_000, 32_000, 44_100, 48_000)


def _tone(freq: float, sr: int, seconds: float = 2.0, amp: float = 0.5) -> np.ndarray:
    n = np.arange(int(sr * seconds))
    return (amp * np.sin(2 * np.pi * freq * n / sr)).astype(np.float32)


def _amplitude(y: np.ndarray, freq: float, sr: int, trim: int = 400) -> float:
    """Amplitude do componente em `freq` (ajuste de mínimos quadrados sobre o trecho estacionário)."""
    y = y[trim:-trim].astype(np.float64)
    n = np.arange(len(y)) + trim
    basis = np.stack([np.sin(2 * np.pi * freq * n / sr), np.cos(2 * np.pi * freq * n / sr)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return float(np.hypot(*coef))


def _path_length(n: int, sr: int) -> int:
    from cctrain.data.audio import resample

    if sr == 16_000:
        return n
    if sr == 48_000:
        return rs.output_length(n)
    return rs.output_length(len(resample(np.zeros(n, np.float32), sr, 48_000)))


@pytest.mark.parametrize("sr", SRS)
@pytest.mark.parametrize("seconds", [0.5, 1.0, 1.0001, 2.0])
def test_to_16k_length_is_the_output_length_of_the_path(sr, seconds):
    x = _tone(1000.0, sr, seconds)
    y = rs.to_16k(x, sr)
    assert y.dtype == np.float32 and y.ndim == 1
    assert len(y) == _path_length(len(x), sr)
    assert abs(len(y) - len(x) * 16_000 / sr) <= 1.0


@pytest.mark.parametrize("sr", SRS)
def test_to_16k_one_khz_tone_amplitude_within_0_1_db(sr):
    y = rs.to_16k(_tone(1000.0, sr), sr)
    err_db = 20 * np.log10(_amplitude(y, 1000.0, 16_000) / 0.5)
    assert abs(err_db) <= 0.1, err_db


@pytest.mark.parametrize("sr", SRS)
def test_to_16k_rejects_alias_above_8_khz_by_60_db(sr):
    for f in (8_500.0, 10_000.0, 12_000.0, 20_000.0):
        if f >= sr / 2 - 300:  # fora da banda do SR de entrada: o tom não existe nele
            continue
        y = rs.to_16k(_tone(f, sr), sr)
        alias = abs(f - 16_000 * round(f / 16_000))  # onde a decimação por 3 jogaria o tom sem o filtro
        rejection_db = 20 * np.log10(0.5 / max(_amplitude(y, alias, 16_000), 1e-12))
        assert rejection_db >= 60.0, (sr, f, rejection_db)


def test_to_16k_native_16k_is_identity_copy():
    x = _tone(440.0, 16_000, 0.3)
    y = rs.to_16k(x, 16_000)
    assert np.array_equal(x, y) and y is not x


def test_to_16k_48k_is_the_v1_resampler_bit_exact():
    x = _tone(1234.0, 48_000, 0.5)
    assert np.array_equal(rs.to_16k(x, 48_000), rs.resample_48k_to_16k(x))


@pytest.mark.parametrize("sr", [24_000, 32_000, 44_100])
def test_to_16k_other_rates_go_soxr_hq_then_the_v1_filter(sr):
    from cctrain.data.audio import resample

    x = _tone(700.0, sr, 0.5) + _tone(5000.0, sr, 0.5, 0.1)
    assert np.array_equal(rs.to_16k(x, sr), rs.resample_48k_to_16k(resample(x, sr, 48_000)))


def test_to_16k_rejects_bad_input():
    with pytest.raises(ValueError, match="1-D"):
        rs.to_16k(np.zeros((2, 100), np.float32), 48_000)
    for bad in (0, -1, 44_100.5):
        with pytest.raises(ValueError, match="sr"):
            rs.to_16k(np.zeros(100, np.float32), bad)


def test_to_16k_empty_input():
    for sr in SRS:
        assert rs.to_16k(np.zeros(0, np.float32), sr).shape == (0,)

