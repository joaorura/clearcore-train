import numpy as np
import pytest

from cctrain.data.manifest import FILES_DIR, load_sources, read_file_manifest, training_records
from cctrain.data.noise import TARGET_RMS, generate_noise, write_noise_set
from cctrain.hashing import sha256_file


def _slope(x: np.ndarray, fs: int = 48_000) -> float:
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / fs)
    sel = (f > 100) & (f < 20_000)
    return float(np.polyfit(np.log10(f[sel]), np.log10(p[sel]), 1)[0])


def test_noise_is_deterministic_and_normalised():
    a, ca = generate_noise(4)
    b, cb = generate_noise(4)
    assert np.array_equal(a, b) and ca == cb == "pink"
    assert a.dtype == np.float32 and a.size == 480_000
    assert abs(float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) - TARGET_RMS) < 5e-3
    assert float(np.max(np.abs(a))) <= 0.95 + 1e-6
    assert not np.array_equal(generate_noise(5)[0][:100], a[:100])


def test_colors_have_the_expected_spectral_slopes():
    slopes = {c: _slope(generate_noise(i, seconds=4.0)[0]) for i, c in enumerate(("white", "pink", "brown"))}
    assert abs(slopes["white"]) < 0.2
    assert abs(slopes["pink"] + 1.0) < 0.2
    assert abs(slopes["brown"] + 2.0) < 0.2


def test_written_set_has_valid_training_records(tmp_path):
    records = write_noise_set(tmp_path / "n", seeds=range(3), data_root=tmp_path)
    assert len(records) == 3
    assert all(r.split == "noise" and r.sr == 48_000 and r.codec == "flac" for r in records)
    assert training_records(records, load_sources()) == records


@pytest.mark.data
def test_pilot_noise_manifest_matches_files_and_is_trainable(paths):
    records = read_file_manifest(FILES_DIR / "noise-synthetic-v1-pilot.jsonl")
    assert len(records) == 16 and {r.split for r in records} == {"noise"}
    assert training_records(records, load_sources()) == records
    for r in records:
        assert sha256_file(paths.data / r.path) == r.sha256, r.path
