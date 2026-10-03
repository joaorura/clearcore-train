"""EER de verificação de locutor sobre as trials congeladas (M1 Tarefa 12, gates `m1.eer_max`).

O hash da lista é conferido contra `trials-v1.sha256` antes de qualquer embedding: lista alterada depois de congelada
falha fechado. Cada clipe é embutido uma vez; o score é o cosseno e o EER vem de `trials.eer_threshold`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from cctrain.eval.trials import Trial, eer_threshold, read_trials, verify_trials_hashes


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER (fração em [0, 1]): ponto onde a taxa de alvos rejeitados iguala a de não-alvos aceitos."""
    return eer_threshold(scores, labels)[0]


def cosine_scores(trials: Sequence[Trial], emb: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Cosseno de cada par e rótulo alvo. `KeyError` se um clipe da lista não tem embedding."""
    unit = {}
    for p in {p for t in trials for p in (t.a, t.b)}:
        v = np.asarray(emb[p], np.float64).reshape(-1)
        unit[p] = v / max(float(np.linalg.norm(v)), 1e-12)
    s = np.array([float(unit[t.a] @ unit[t.b]) for t in trials])
    return s, np.array([t.target for t in trials], dtype=bool)


def eer_bootstrap_ci(scores: np.ndarray, labels: np.ndarray, *, resamples: int = 1000, ci: float = 0.95,
                     seed: int = 0) -> tuple[float, float]:
    """IC percentil do EER reamostrando os trials (alvos e não-alvos separadamente, com reposição). Informativo."""
    s, y = np.asarray(scores, np.float64), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(resamples):
        sp = pos[rng.integers(0, pos.size, pos.size)]
        sn = neg[rng.integers(0, neg.size, neg.size)]
        vals.append(eer(np.r_[sp, sn], np.r_[np.ones(sp.size, bool), np.zeros(sn.size, bool)]))
    lo, hi = np.percentile(vals, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(lo), float(hi)


def evaluate_trials_file(path: Path, lock: Path, embed: Callable[[str], np.ndarray], *,
                         embed_many: Callable[[Sequence[str]], Mapping[str, np.ndarray]] | None = None,
                         ci_resamples: int = 1000, seed: int = 0) -> dict:
    """Confere o SHA-256 de `path` em `lock`, embute cada clipe uma vez (`embed_many` se dado, senão `embed`) e
    devolve EER, IC 95% e as contagens. `embed` recebe o caminho do clipe como está na lista."""
    path = Path(path)
    digest = verify_trials_hashes(lock, {path.name: path})[path.name]
    trials = read_trials(path)
    clips = sorted({p for t in trials for p in (t.a, t.b)})
    emb = dict(embed_many(clips)) if embed_many is not None else {p: embed(p) for p in clips}
    scores, labels = cosine_scores(trials, emb)
    lo, hi = eer_bootstrap_ci(scores, labels, resamples=ci_resamples, seed=seed) if ci_resamples else (np.nan,) * 2
    value, threshold = eer_threshold(scores, labels)
    return {"list": path.stem, "sha256": digest, "eer": value, "cosine_threshold": threshold,
            "ci95": [lo, hi], "n_trials": len(trials), "n_target": int(labels.sum()),
            "n_nontarget": int((~labels).sum()), "n_clips": len(clips), "n_speakers": len(
                {s for t in trials for s in (t.spk_a, t.spk_b)})}
