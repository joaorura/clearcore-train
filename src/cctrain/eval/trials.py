"""Listas de trials de verificação de locutor (M1 Tarefa 9), congeladas por SHA-256 antes de qualquer EER.

Um trial é um par de arquivos com rótulo alvo (mesmo falante) ou não-alvo. A geração é determinística: depende só dos
registros (ordenados), das contagens e da seed, nunca da ordem de entrada. O alvo é balanceado entre falantes e o
não-alvo entre pares de falantes, para que um falante com muitos clipes não domine o EER.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cctrain.data.manifest import FileRecord
from cctrain.hashing import sha256_file


class TrialsError(ValueError):
    """Lista de trials impossível de gerar, ou diferente da congelada."""


@dataclass(frozen=True)
class Trial:
    a: str
    b: str
    target: bool
    spk_a: str
    spk_b: str


def make_trials(records: Iterable[FileRecord], n_target: int, n_nontarget: int, seed: int) -> list[Trial]:
    by_spk: dict[str, list[str]] = {}
    for r in records:
        if r.speaker_key is None:
            raise TrialsError(f"{r.path}: registro sem falante não entra em trials")
        by_spk.setdefault(r.speaker_key, []).append(r.path)
    spks = sorted(by_spk)
    for s in spks:
        by_spk[s] = sorted(set(by_spk[s]))
    if len(spks) < 2:
        raise TrialsError(f"são necessários ao menos 2 falantes para trials não-alvo (há {len(spks)})")
    rng = random.Random(seed)

    quota = {s: n_target // len(spks) for s in spks}
    for s in rng.sample(spks, n_target % len(spks)):
        quota[s] += 1
    out: list[Trial] = []
    for s in spks:
        clips = by_spk[s]
        n = len(clips)
        if quota[s] > n * (n - 1) // 2:
            raise TrialsError(f"{s}: só {n * (n - 1) // 2} pares alvo possíveis, pedidos {quota[s]}")
        seen: set[tuple[int, int]] = set()
        while len(seen) < quota[s]:
            i, j = sorted(rng.sample(range(n), 2))
            if (i, j) not in seen:
                seen.add((i, j))
                out.append(Trial(clips[i], clips[j], True, s, s))

    pairs = list(itertools.combinations(spks, 2))
    rng.shuffle(pairs)
    seen_nt: set[tuple[str, str]] = set()
    max_pairs = sum(len(by_spk[a]) * len(by_spk[b]) for a, b in pairs)
    if n_nontarget > max_pairs:
        raise TrialsError(f"só {max_pairs} pares não-alvo possíveis, pedidos {n_nontarget}")
    k = 0
    while len(seen_nt) < n_nontarget:
        sa, sb = pairs[k % len(pairs)]
        k += 1
        pa, pb = rng.choice(by_spk[sa]), rng.choice(by_spk[sb])
        if (pa, pb) not in seen_nt:
            seen_nt.add((pa, pb))
            out.append(Trial(pa, pb, False, sa, sb))
    return out


def _line(t: Trial) -> str:
    return json.dumps(asdict(t), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_trials(trials: Iterable[Trial], path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(_line(t) + "\n" for t in trials), encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def read_trials(path: Path) -> list[Trial]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            out.append(Trial(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise TrialsError(f"{path}:{n}: trial inválido: {exc}") from exc
    return out


def read_hash_lock(lock: Path) -> dict[str, str]:
    out = {}
    for line in Path(lock).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def verify_trials_hashes(lock: Path, files: Mapping[str, Path]) -> dict[str, str]:
    """Confere cada arquivo contra o `<sha256>  <nome>` do lock; falha fechado se faltar ou mudou."""
    pinned = read_hash_lock(lock)
    out = {}
    for name, path in files.items():
        if name not in pinned:
            raise TrialsError(f"{name}: ausente em {lock}")
        actual = sha256_file(path)
        if actual != pinned[name]:
            raise TrialsError(f"{name}: lista alterada depois de congelada ({actual} != {pinned[name]})")
        out[name] = actual
    return out


def eer_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(EER, limiar): aceita se score >= limiar. O limiar é o primeiro onde FAR e FRR mais se aproximam."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if y.all() or not y.any():
        raise TrialsError("EER exige trials alvo e não-alvo")
    pos, neg = np.sort(s[y]), np.sort(s[~y])
    cand = np.unique(s)
    frr = np.searchsorted(pos, cand, side="left") / len(pos)  # alvos rejeitados (score < t)
    far = 1.0 - np.searchsorted(neg, cand, side="left") / len(neg)  # não-alvos aceitos (score >= t)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2.0), float(cand[i])
