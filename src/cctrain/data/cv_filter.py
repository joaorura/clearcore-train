"""Filtro de qualidade do `client_id` do Common Voice por avaliador independente (spec §4.6; M1 Tarefa 9).

Regras: (1) `client_id` com menos de `min_candidate_clips` clipes não é candidato; (2) cada clipe é comparado ao
centróide do próprio `client_id` calculado sem ele (leave-one-out, para que o clipe estranho não puxe o centróide) e
sai se o cosseno ficar abaixo do limiar de EER do avaliador (calibrado no dev do CV); (3) o cliente que, depois disso,
fica com menos de `min_clips` clipes ou com menos de `min_retained_fraction` dos clipes avaliados é descartado inteiro
(conta misturada). O avaliador é o WeSpeaker ResNet34: o encoder embarcado não pode se avaliar.

O `client_id` só agrupa clipes do mesmo gravador; nada aqui o cruza com fonte externa, e embeddings e manifests
derivados ficam em `data/` (fora do git).
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from cctrain.data.audio import load_audio
from cctrain.data.manifest import FileRecord
from cctrain.data.resample import to_16k
from cctrain.eval.trials import eer_threshold, make_trials

MIN_EMBED_SAMPLES = 8_000


class Evaluator(Protocol):
    def embed(self, x16k: np.ndarray) -> np.ndarray: ...


@dataclass
class FilterStats:
    clients_total: int = 0
    clients_candidates: int = 0
    clients_retained: int = 0
    clips_candidates: int = 0
    clips_scored: int = 0
    clips_unreadable: int = 0
    clips_dropped_low_cosine: int = 0
    clips_retained: int = 0


def _group(records: Iterable[FileRecord]) -> dict[str, list[FileRecord]]:
    out: dict[str, list[FileRecord]] = defaultdict(list)
    for r in records:
        if r.speaker_key is None:
            raise ValueError(f"{r.path}: o filtro de CV exige `client_id` em `speaker`")
        out[r.speaker_key].append(r)
    return out


def cv_candidates(records: Iterable[FileRecord], min_clips: int = 20) -> dict[str, list[FileRecord]]:
    return {k: v for k, v in _group(records).items() if len(v) >= min_clips}


def cap_per_client(recs: list[FileRecord], max_clips: int, seed: int) -> list[FileRecord]:
    if len(recs) <= max_clips:
        return sorted(recs, key=lambda r: r.path)
    key = lambda r: hashlib.sha256(f"{seed}:{r.path}".encode()).hexdigest()  # noqa: E731
    return sorted(sorted(recs, key=key)[:max_clips], key=lambda r: r.path)


def _load_cache(cache: Path) -> dict[str, np.ndarray]:
    if not Path(cache).is_file():
        return {}
    z = np.load(cache, allow_pickle=False)
    return {str(p): e for p, e in zip(z["paths"], z["embs"], strict=True)}


def _save_cache(cache: Path, embs: Mapping[str, np.ndarray]) -> None:
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(embs)
    tmp = cache.with_name(cache.name + ".tmp.npz")
    np.savez(tmp, paths=np.array(paths), embs=np.stack([embs[p] for p in paths]) if paths else np.zeros((0, 1)))
    os.replace(tmp, cache)


def embed_records(records: Iterable[FileRecord], evaluator: Evaluator, data_dir: Path, cache: Path, *,
                  workers: int = 8, save_every: int = 2000, log: Callable[[str], None] | None = None,
                  ) -> dict[str, np.ndarray]:
    """Embedding por clipe, retomável: o que já está em `cache` não é recalculado. Clipe ilegível ou curto demais
    (< 0,5 s a 16 kHz) fica de fora do resultado (e é contado por quem chama)."""
    embs = _load_cache(cache)
    todo = [r for r in records if r.path not in embs]
    # clipes que falharam não vão para o cache: o par (ilegível/curto) é refeito em cada rodada, é barato.
    if not todo:
        return {r.path: embs[r.path] for r in records if r.path in embs}

    def work(r: FileRecord):
        try:
            x, sr = load_audio(Path(data_dir) / r.path)
            x16 = to_16k(x, sr)
            if x16.size < MIN_EMBED_SAMPLES:
                return r.path, None
            return r.path, np.asarray(evaluator.embed(x16), dtype=np.float32)
        except Exception:  # noqa: BLE001 - MP3 corrompido não derruba a rodada; vira clipe descartado
            return r.path, None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path, e in ex.map(work, todo):
            done += 1
            if e is not None:
                embs[path] = e
            if done % save_every == 0:
                _save_cache(cache, embs)
                if log:
                    log(f"  embeddings: {done}/{len(todo)}")
    _save_cache(cache, embs)
    return {r.path: embs[r.path] for r in records if r.path in embs}


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def loo_centroid_scores(embs: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Cosseno de cada clipe ao centróide dos demais clipes do mesmo cliente."""
    if len(embs) < 2:
        return {p: 0.0 for p in embs}
    paths = sorted(embs)
    u = np.stack([_unit(embs[p]) for p in paths])
    total = u.sum(axis=0)
    out = {}
    for p, v in zip(paths, u, strict=True):
        c = (total - v) / (len(paths) - 1)
        n = np.linalg.norm(c)
        out[p] = float(v @ c / n) if n > 0 else 0.0
    return out


def filter_clips(records: Iterable[FileRecord], embeddings: Mapping[str, np.ndarray], *, threshold: float,
                 min_clips: int = 20, min_retained_fraction: float = 0.5,
                 stats: FilterStats | None = None) -> list[FileRecord]:
    stats = stats if stats is not None else FilterStats()
    kept_all: list[FileRecord] = []
    for _, recs in sorted(_group(records).items()):
        scored = [r for r in recs if r.path in embeddings]
        scores = loo_centroid_scores({r.path: embeddings[r.path] for r in scored})
        kept = [r for r in scored if scores[r.path] >= threshold]
        stats.clips_scored += len(scored)
        stats.clips_dropped_low_cosine += len(scored) - len(kept)
        if len(kept) < min_clips or not scored or len(kept) / len(scored) < min_retained_fraction:
            continue
        stats.clients_retained += 1
        stats.clips_retained += len(kept)
        kept_all += kept
    return sorted(kept_all, key=lambda r: r.path)


def calibrate_threshold(records: Iterable[FileRecord], embeddings: Mapping[str, np.ndarray], *, n_pairs: int = 5000,
                        seed: int = 0) -> tuple[float, float]:
    """(EER, limiar de cosseno no EER) do avaliador sobre pares do mesmo `client_id` × clientes diferentes."""
    recs = [r for r in records if r.path in embeddings]
    trials = make_trials(recs, n_target=n_pairs, n_nontarget=n_pairs, seed=seed)
    scores = np.array([float(_unit(embeddings[t.a]) @ _unit(embeddings[t.b])) for t in trials])
    labels = np.array([t.target for t in trials])
    return eer_threshold(scores, labels)


def cv_filter(records: Iterable[FileRecord], evaluator: Evaluator, *, data_dir: Path, threshold: float,
              cache: Path, min_clips: int = 20, min_candidate_clips: int = 20, max_clips_per_client: int = 100,
              min_retained_fraction: float = 0.5, seed: int = 0, workers: int = 8,
              stats: FilterStats | None = None, log: Callable[[str], None] | None = None) -> list[FileRecord]:
    """Clipes retidos (de clientes retidos). Candidato: >= `min_candidate_clips` clipes brutos; no máximo
    `max_clips_per_client` clipes por cliente são avaliados (amostra determinística por hash do caminho)."""
    stats = stats if stats is not None else FilterStats()
    groups = _group(records)
    stats.clients_total = len(groups)
    cand = {k: v for k, v in groups.items() if len(v) >= min_candidate_clips}
    stats.clients_candidates = len(cand)
    pool = [r for k in sorted(cand) for r in cap_per_client(cand[k], max_clips_per_client, seed)]
    stats.clips_candidates = len(pool)
    embs = embed_records(pool, evaluator, data_dir, cache, workers=workers, log=log)
    stats.clips_unreadable = len(pool) - len(embs)
    return filter_clips(pool, embs, threshold=threshold, min_clips=min_clips,
                        min_retained_fraction=min_retained_fraction, stats=stats)
