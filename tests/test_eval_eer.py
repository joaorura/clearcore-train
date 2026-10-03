"""EER sobre as trials congeladas, com hash conferido antes de embutir (M1 Tarefa 12)."""

from __future__ import annotations

import numpy as np
import pytest

from cctrain.eval.eer import cosine_scores, eer, eer_bootstrap_ci, evaluate_trials_file
from cctrain.eval.trials import Trial, TrialsError, write_trials
from cctrain.hashing import sha256_file


def test_eer_perfect_and_inverted_and_chance():
    y = np.array([1] * 50 + [0] * 50)
    assert eer(np.r_[np.linspace(0.6, 1, 50), np.linspace(-1, 0.4, 50)], y) == 0.0
    assert eer(np.r_[np.linspace(-1, 0.4, 50), np.linspace(0.6, 1, 50)], y) == 1.0
    rng = np.random.default_rng(0)
    assert abs(eer(rng.standard_normal(20_000), rng.integers(0, 2, 20_000)) - 0.5) < 0.02


def test_eer_matches_the_closed_form_for_two_gaussians():
    from math import erf, sqrt

    rng = np.random.default_rng(1)
    pos, neg = rng.normal(1.0, 1.0, 40_000), rng.normal(-1.0, 1.0, 40_000)
    expected = 0.5 * (1 - erf(1.0 / sqrt(2)))  # EER = Q(d'/2), d' = 2
    got = eer(np.r_[pos, neg], np.r_[np.ones(pos.size), np.zeros(neg.size)])
    assert abs(got - expected) < 0.005


def test_eer_needs_both_classes():
    with pytest.raises(TrialsError):
        eer(np.array([0.1, 0.2]), np.array([1, 1]))


def test_cosine_scores_uses_normalised_dot_and_fails_on_missing_clip():
    emb = {"a": np.array([1.0, 0.0]), "b": np.array([2.0, 0.0]), "c": np.array([0.0, 3.0])}
    trials = [Trial("a", "b", True, "s", "s"), Trial("a", "c", False, "s", "t")]
    s, y = cosine_scores(trials, emb)
    np.testing.assert_allclose(s, [1.0, 0.0], atol=1e-12)
    assert y.tolist() == [True, False]
    with pytest.raises(KeyError):
        cosine_scores([Trial("a", "zz", True, "s", "s")], emb)


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(2)
    s = np.r_[rng.normal(1, 1, 2000), rng.normal(-1, 1, 2000)]
    y = np.r_[np.ones(2000), np.zeros(2000)]
    point = eer(s, y)
    lo, hi = eer_bootstrap_ci(s, y, resamples=200, seed=3)
    assert lo < point < hi and hi - lo < 0.05
    assert (lo, hi) == eer_bootstrap_ci(s, y, resamples=200, seed=3)


def _fake_trials(tmp_path):
    rng = np.random.default_rng(0)
    centers = {f"s{i}": rng.standard_normal(8) * 3 for i in range(4)}
    emb, trials = {}, []
    for spk, c in centers.items():
        for k in range(6):
            emb[f"{spk}/{k}.wav"] = c + rng.standard_normal(8) * 0.3
    keys = sorted(emb)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            sa, sb = a.split("/")[0], b.split("/")[0]
            trials.append(Trial(a, b, sa == sb, sa, sb))
    path = tmp_path / "list.jsonl"
    digest = write_trials(trials, path)
    lock = tmp_path / "lock.sha256"
    lock.write_text(f"{digest}  list.jsonl\n")
    return path, lock, emb


def test_evaluate_trials_file_verifies_hash_embeds_each_clip_once(tmp_path):
    path, lock, emb = _fake_trials(tmp_path)
    calls: list[str] = []

    def embed(p: str) -> np.ndarray:
        calls.append(p)
        return emb[p]

    out = evaluate_trials_file(path, lock, embed, ci_resamples=50)
    assert out["eer"] < 0.05 and out["sha256"] == sha256_file(path)
    assert out["n_clips"] == len(emb) == len(calls) and len(set(calls)) == len(calls)
    assert out["n_target"] + out["n_nontarget"] == out["n_trials"]
    assert out["ci95"][0] <= out["eer"] <= out["ci95"][1]


def test_evaluate_trials_file_refuses_a_list_changed_after_freezing(tmp_path):
    path, lock, emb = _fake_trials(tmp_path)
    with open(path, "a") as f:
        f.write('{"a":"s0/0.wav","b":"s1/0.wav","spk_a":"s0","spk_b":"s1","target":false}\n')
    with pytest.raises(TrialsError, match="alterada"):
        evaluate_trials_file(path, lock, lambda p: emb[p])
