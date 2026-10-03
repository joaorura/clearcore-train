"""Pools de dados da rodada real do M1 (Tarefa 12): ruído deduplicado e com reserva de dev disjunta."""

from __future__ import annotations

import pytest

from cctrain.data.manifest import FileRecord
from cctrain.m1.pools import split_noise


def _n(i: int, sha: int | None = None) -> FileRecord:
    return FileRecord(path=f"raw/n/{i:04d}.flac", sha256=f"{(i if sha is None else sha):064x}", source_id="dns",
                      license="CC0-1.0", role="train", speaker=None, split="noise", sr=48000, num_samples=48000,
                      band_hz=None, codec="flac", attribution="x")


def test_split_noise_dedupes_by_content_then_holds_out_a_disjoint_dev_fraction():
    recs = [_n(i) for i in range(400)] + [_n(1000 + i, sha=i) for i in range(30)]  # 30 cópias de conteúdo repetido
    train, dev = split_noise(recs, dev_fraction=0.1, seed=7)
    assert len(train) + len(dev) == 400  # as cópias saíram
    assert not {r.sha256 for r in train} & {r.sha256 for r in dev}
    assert len({r.sha256 for r in train + dev}) == 400
    assert 25 <= len(dev) <= 55


def test_split_noise_is_deterministic_and_independent_of_input_order():
    recs = [_n(i) for i in range(200)]
    a = split_noise(recs, dev_fraction=0.1, seed=3)
    b = split_noise(list(reversed(recs)), dev_fraction=0.1, seed=3)
    assert [r.path for r in a[1]] == [r.path for r in b[1]]
    assert [r.path for r in a[1]] != [r.path for r in split_noise(recs, dev_fraction=0.1, seed=4)[1]]


def test_split_noise_rejects_a_bad_fraction():
    with pytest.raises(ValueError):
        split_noise([_n(0)], dev_fraction=1.0, seed=0)
