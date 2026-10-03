import numpy as np

from cctrain.data.manifest import load_sources, training_records
from cctrain.data.rir import generate_rir, write_rir_set


def test_rir_is_deterministic_and_bounded():
    h1, p1 = generate_rir(3)
    h2, p2 = generate_rir(3)
    assert np.array_equal(h1, h2)
    assert (p1.seed, p1.room_dim, p1.source, p1.mic, p1.max_order) == (p2.seed, p2.room_dim, p2.source, p2.mic,
                                                                        p2.max_order)
    assert h1.dtype == np.float32
    assert len(h1) <= int(1.2 * 48_000)
    assert abs(float(np.max(np.abs(h1))) - 0.99) < 1e-6
    assert int(np.argmax(np.abs(h1))) <= 48
    assert 0.05 <= p1.rt60_target <= 1.0
    assert not np.array_equal(generate_rir(4)[0][:100], h1[:100])


def test_written_set_has_valid_training_records(tmp_path):
    records, params = write_rir_set(tmp_path / "rir", seeds=range(3), data_root=tmp_path)
    assert len(records) == len(params) == 3
    assert all(r.split == "rir" and r.sr == 48_000 and r.codec == "flac" for r in records)
    assert training_records(records, load_sources()) == records


# --- synthetic-v2: image source de ordem baixa + cauda estocástica ---------------------------------------------

import hashlib  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402

from cctrain.data.manifest import ManifestError  # noqa: E402
from cctrain.data.rir import (  # noqa: E402
    generate_rir_v2,
    measure_rt60_schroeder,
    params_to_jsonl,
    write_rir_v2_set,
)


def test_v2_is_deterministic_and_normalised():
    h1, p1 = generate_rir_v2(5)
    h2, p2 = generate_rir_v2(5)
    assert np.array_equal(h1, h2) and p1 == p2
    assert h1.dtype == np.float32 and np.isfinite(h1).all()
    assert abs(float(np.max(np.abs(h1))) - 0.99) < 1e-6
    assert int(np.argmax(np.abs(h1))) <= 48
    assert p1.max_order <= 3
    assert not np.array_equal(generate_rir_v2(6)[0][:100], h1[:100])


def test_v2_has_no_order_20_ceiling_and_long_tail():
    h, p = generate_rir_v2(1, rt60_range=(0.95, 1.0))
    assert p.rt60_target >= 0.95 and len(h) >= int(1.2 * 48_000)  # v1 truncava em 1,2 s


def test_schroeder_recovers_a_synthetic_exponential_decay():
    fs, rt60 = 48_000, 0.4
    t = np.arange(int(1.0 * fs)) / fs
    x = np.random.default_rng(0).standard_normal(t.size) * 10 ** (-3 * t / rt60)
    assert measure_rt60_schroeder(x, fs) == pytest.approx(rt60, rel=0.1)
    assert np.isnan(measure_rt60_schroeder(np.zeros(100), fs))


def test_v2_rt60_distribution_covers_range_and_tracks_target():
    measured, target = [], []
    for seed in range(256):
        h, p = generate_rir_v2(seed)
        assert np.isfinite(p.rt60_measured)
        assert p.rt60_measured == pytest.approx(measure_rt60_schroeder(h, 48_000), rel=1e-6)
        measured.append(p.rt60_measured)
        target.append(p.rt60_target)
    measured, target = np.array(measured), np.array(target)
    assert np.percentile(measured, 5) <= 0.15
    assert np.percentile(measured, 95) >= 0.85
    assert float(np.median(np.abs(measured - target) / target)) <= 0.15


def test_v2_set_is_deterministic_by_flac_sha256_and_has_valid_v2_records(tmp_path):
    from cctrain.data.manifest import write_file_manifest

    r1, p1 = write_rir_v2_set(tmp_path / "a" / "rir", range(4), tmp_path / "a")
    r2, _ = write_rir_v2_set(tmp_path / "b" / "rir", range(4), tmp_path / "b")
    assert [r.sha256 for r in r1] == [r.sha256 for r in r2]
    for r in r1:
        assert hashlib.sha256((tmp_path / "a" / r.path).read_bytes()).hexdigest() == r.sha256
    assert all(r.schema == 2 and r.source_id == "rir-synthetic-v2" and r.split == "rir" and r.attribution
               for r in r1)
    assert training_records(r1, load_sources()) == r1
    write_file_manifest(r1, tmp_path / "m.jsonl")
    for line in params_to_jsonl(p1).splitlines():
        json.loads(line, parse_constant=lambda c: pytest.fail(f"constante não-JSON: {c}"))


def test_non_finite_rt60_measured_is_refused(tmp_path, monkeypatch):
    import cctrain.data.rir as rir

    monkeypatch.setattr(rir, "measure_rt60_schroeder", lambda h, fs: float("nan"))
    with pytest.raises(ManifestError, match="rt60_measured"):
        write_rir_v2_set(tmp_path / "rir", range(1), tmp_path)
    assert not list((tmp_path / "rir").glob("*.flac"))
    bad = generate_rir(0)[1]
    with pytest.raises(ValueError):
        params_to_jsonl([type(bad)(**{**bad.__dict__, "rt60_measured": float("inf")})])
