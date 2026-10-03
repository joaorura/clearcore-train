"""Listas de trials congeladas por SHA-256 e EER (M1 Tarefa 9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cctrain.data.manifest import FileRecord
from cctrain.eval.trials import (
    Trial,
    TrialsError,
    eer_threshold,
    make_trials,
    read_trials,
    verify_trials_hashes,
    write_trials,
)


def _rec(spk: str, i: int, source: str = "src") -> FileRecord:
    return FileRecord(path=f"raw/{source}/{spk}/{i}.flac", sha256=f"{abs(hash((spk, i))) % 10**8:064x}",
                      source_id=source, license="CC-BY-4.0", role="train", speaker=spk, split="test", sr=16000,
                      num_samples=16000, band_hz=None, codec="flac", attribution="x")


def _records(n_spk: int = 6, n_clips: int = 12) -> list[FileRecord]:
    return [_rec(f"s{s}", i) for s in range(n_spk) for i in range(n_clips)]


def test_make_trials_counts_labels_and_distinctness():
    ts = make_trials(_records(), n_target=100, n_nontarget=120, seed=7)
    assert sum(t.target for t in ts) == 100 and sum(not t.target for t in ts) == 120
    for t in ts:
        assert t.a != t.b
        assert (t.spk_a == t.spk_b) == t.target
    assert len({(min(t.a, t.b), max(t.a, t.b)) for t in ts}) == len(ts)  # sem par repetido


def test_make_trials_balanced_over_speakers():
    ts = make_trials(_records(n_spk=5), n_target=100, n_nontarget=10, seed=1)
    counts = {}
    for t in ts:
        if t.target:
            counts[t.spk_a] = counts.get(t.spk_a, 0) + 1
    assert set(counts.values()) == {20}


def test_make_trials_deterministic_and_seed_dependent():
    a = make_trials(_records(), 50, 50, seed=3)
    b = make_trials(list(reversed(_records())), 50, 50, seed=3)
    assert a == b  # independe da ordem dos registros
    assert a != make_trials(_records(), 50, 50, seed=4)


def test_make_trials_uses_only_given_records_and_fails_closed():
    recs = _records(n_spk=1)
    with pytest.raises(TrialsError, match="falantes"):
        make_trials(recs, 10, 10, seed=0)  # um falante só: sem par não-alvo
    with pytest.raises(TrialsError, match="alvo"):
        make_trials(_records(n_spk=3, n_clips=2), n_target=1000, n_nontarget=1, seed=0)  # pares alvo insuficientes


def test_write_read_roundtrip_hash_is_stable(tmp_path: Path):
    ts = make_trials(_records(), 30, 30, seed=2)
    h1 = write_trials(ts, tmp_path / "a.jsonl")
    h2 = write_trials(ts, tmp_path / "b.jsonl")
    assert h1 == h2 and len(h1) == 64
    assert read_trials(tmp_path / "a.jsonl") == ts
    assert isinstance(ts[0], Trial)


def test_verify_trials_hashes(tmp_path: Path):
    ts = make_trials(_records(), 30, 30, seed=2)
    p = tmp_path / "x.jsonl"
    h = write_trials(ts, p)
    lock = tmp_path / "trials.sha256"
    lock.write_text(f"{h}  x.jsonl\n")
    assert verify_trials_hashes(lock, {"x.jsonl": p}) == {"x.jsonl": h}
    p.write_text(p.read_text() + "\n")
    with pytest.raises(TrialsError, match="alterad"):
        verify_trials_hashes(lock, {"x.jsonl": p})
    with pytest.raises(TrialsError, match="ausente"):
        verify_trials_hashes(lock, {"y.jsonl": p})


def test_eer_threshold_separable_and_overlapping():
    scores = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])
    labels = np.array([1, 1, 1, 0, 0, 0])
    eer, thr = eer_threshold(scores, labels)
    assert eer == pytest.approx(0.0) and 0.2 < thr <= 0.8
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 1.0, 20000)
    neg = rng.normal(-1.0, 1.0, 20000)
    eer, thr = eer_threshold(np.r_[pos, neg], np.r_[np.ones(20000), np.zeros(20000)])
    assert eer == pytest.approx(0.1587, abs=0.01)  # Phi(-1)
    assert thr == pytest.approx(0.0, abs=0.05)


def test_eer_requires_both_classes():
    with pytest.raises(TrialsError):
        eer_threshold(np.array([0.1, 0.2]), np.array([1, 1]))
