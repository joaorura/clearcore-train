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
