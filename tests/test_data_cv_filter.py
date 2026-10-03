"""Filtro de qualidade do `client_id` do Common Voice com avaliador independente (spec §4.6; M1 Tarefa 9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cctrain.data.cv_filter import (
    FilterStats,
    calibrate_threshold,
    cv_candidates,
    cv_filter,
    embed_records,
    filter_clips,
    loo_centroid_scores,
)
from cctrain.data.manifest import FileRecord

SR = 16_000
FREQS = {"A": 300.0, "B": 900.0, "C": 1700.0}


class FakeEvaluator:
    """Embedding = one-hot da frequência dominante (determinístico), com 1 pitada de ruído por clipe."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, x16k: np.ndarray) -> np.ndarray:
        self.calls += 1
        spec = np.abs(np.fft.rfft(x16k))
        f = np.argmax(spec) * SR / len(x16k)
        v = np.zeros(len(FREQS) + 1, np.float32)
        v[int(np.argmin([abs(f - q) for q in FREQS.values()]))] = 1.0
        v[-1] = 0.05
        return v


def _write(root: Path, name: str, freq: float) -> str:
    rel = f"raw/cv/{name}.wav"
    (root / "raw" / "cv").mkdir(parents=True, exist_ok=True)
    t = np.arange(SR) / SR
    sf.write(str(root / rel), (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)
    return rel


def _rec(rel: str, client: str) -> FileRecord:
    return FileRecord(path=rel, sha256=f"{abs(hash(rel)) % 10**12:064x}", source_id="cv27-pt", license="CC0-1.0",
                      role="train", speaker=client, split="train", sr=SR, num_samples=SR, band_hz=None, codec="wav",
                      attribution="CV")


def _dataset(root: Path):
    recs = []
    for i in range(24):  # cliente limpo
        recs.append(_rec(_write(root, f"ok_{i}", FREQS["A"]), "ok"))
    for i in range(20):  # cliente misto: 16 de B e 4 de C (as de C são de outra pessoa)
        recs.append(_rec(_write(root, f"mix_{i}", FREQS["B"] if i < 16 else FREQS["C"]), "mix"))
    for i in range(12):  # poucos clipes
        recs.append(_rec(_write(root, f"few_{i}", FREQS["A"]), "few"))
    return recs


def test_candidates_need_min_clips(tmp_path):
    recs = _dataset(tmp_path)
    c = cv_candidates(recs, min_clips=20)
    assert set(c) == {"cv27-pt:ok", "cv27-pt:mix"}


def test_loo_centroid_scores_flag_the_outlier():
    e = {f"p{i}": np.array([1.0, 0.0]) for i in range(9)}
    e["out"] = np.array([0.0, 1.0])
    s = loo_centroid_scores(e)
    assert s["p0"] == pytest.approx(8 / 65**0.5)  # centróide dos outros 9: 8 de [1,0] e 1 de [0,1]
    assert s["out"] == pytest.approx(0.0, abs=1e-6)


def test_filter_drops_foreign_clips(tmp_path):
    recs = _dataset(tmp_path)
    stats = FilterStats()
    kept = cv_filter(recs, FakeEvaluator(), data_dir=tmp_path, threshold=0.5, min_clips=10, min_candidate_clips=20,
                     max_clips_per_client=100, seed=1, cache=tmp_path / "emb.npz", stats=stats, workers=2)
    by: dict[str, list[str]] = {}
    for r in kept:
        by.setdefault(r.speaker, []).append(Path(r.path).stem)
    assert len(by["ok"]) == 24
    assert sorted(by["mix"]) == sorted(f"mix_{i}" for i in range(16))  # os 4 clipes de outra voz saem
    assert "few" not in by  # < 20 clipes brutos: nunca é candidato
    assert (stats.clients_candidates, stats.clients_retained, stats.clips_dropped_low_cosine) == (2, 2, 4)


def test_client_below_min_after_filter_is_dropped(tmp_path):
    recs = _dataset(tmp_path)
    kept = cv_filter(recs, FakeEvaluator(), data_dir=tmp_path, threshold=0.5, min_clips=20, min_candidate_clips=20,
                     max_clips_per_client=100, seed=1, cache=tmp_path / "emb.npz", workers=2)
    assert {r.speaker for r in kept} == {"ok"}


def test_filter_clips_pure_threshold_and_fraction():
    recs = [_rec(f"p{i}", "c") for i in range(30)]
    emb = {r.path: np.array([1.0, 0.0]) for r in recs[:25]} | {r.path: np.array([0.0, 1.0]) for r in recs[25:]}
    kept = filter_clips(recs, emb, threshold=0.5, min_clips=20, min_retained_fraction=0.5)
    assert len(kept) == 25
    kept2 = filter_clips(recs, emb, threshold=0.5, min_clips=26, min_retained_fraction=0.5)
    assert kept2 == []
    kept3 = filter_clips(recs, emb, threshold=0.5, min_clips=20, min_retained_fraction=0.9)
    assert kept3 == []  # 25/30 < 0,9: conta misturada


def test_embed_cache_makes_second_run_free(tmp_path):
    recs = _dataset(tmp_path)[:30]
    ev = FakeEvaluator()
    cache = tmp_path / "emb.npz"
    a = embed_records(recs, ev, tmp_path, cache, workers=2)
    n = ev.calls
    assert n == 30 and len(a) == 30
    b = embed_records(recs, ev, tmp_path, cache, workers=2)
    assert ev.calls == n
    assert all(np.array_equal(a[p], b[p]) for p in a)


def test_short_clips_are_skipped_not_fatal(tmp_path):
    rel = "raw/cv/short.wav"
    (tmp_path / "raw" / "cv").mkdir(parents=True)
    sf.write(str(tmp_path / rel), np.zeros(1000, np.float32), SR)
    out = embed_records([_rec(rel, "x")], FakeEvaluator(), tmp_path, tmp_path / "e.npz", workers=1)
    assert out == {}


def test_calibrate_threshold_returns_eer_between_classes():
    rng = np.random.default_rng(0)
    emb, recs = {}, []
    for c in range(8):
        base = rng.standard_normal(16)
        for i in range(25):
            r = _rec(f"c{c}_{i}", f"c{c}")
            recs.append(r)
            emb[r.path] = (base + 0.2 * rng.standard_normal(16)).astype(np.float32)
    eer, thr = calibrate_threshold(recs, emb, n_pairs=2000, seed=0)
    assert eer < 0.05 and 0.0 < thr < 1.0
