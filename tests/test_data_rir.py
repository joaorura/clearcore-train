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
