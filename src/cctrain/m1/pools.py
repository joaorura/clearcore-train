"""Pools de dados da rodada real do M1 (Tarefa 12): fala por split, ruído deduplicado com reserva de dev, RIR.

O ruído do M1 não tem split próprio nos manifests (todo `noise`). Para a medida adicional C4 (falante concorrente +
ruído) o dev precisa de ruído que o treino nunca viu: separa-se uma fração por hash do SHA-256 do conteúdo, depois de
deduplicar por conteúdo (o DNS traz 24 cópias de 3 hashes e o FSD50K 1; sem isso a mesma gravação cairia nos dois lados).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from cctrain.data.manifest import FILES_DIR, FileRecord, read_file_manifest
from cctrain.data.splits import apply_splits, assign_splits, dedupe_by_sha256, load_split_config
from cctrain.paths import Paths

NOISE_MANIFESTS = ("dns-noise-fullband-cc0", "fsd50k-eval-cc0", "fsd50k-eval-ccby", "noise-local-mics")
RIR_MANIFEST = "rir-synthetic-v2"
NOISE_DEV_FRACTION = 0.05
NOISE_SEED = 20261003


def split_noise(records: Iterable[FileRecord], *, dev_fraction: float, seed: int) -> tuple[list[FileRecord], list[FileRecord]]:
    """(treino, dev): deduplica por SHA-256 e reserva ~`dev_fraction` dos conteúdos para o dev (só C4)."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction fora de (0, 1)")
    train, dev = [], []
    for r in dedupe_by_sha256(records):
        h = int(hashlib.sha256(f"{seed}:{r.sha256}".encode()).hexdigest()[:12], 16) / 16**12
        if h < dev_fraction:
            dev.append(r)  # o `split` continua `noise` (split `dev` exige falante); a reserva é a lista
        else:
            train.append(r)
    return train, dev


def noise_pools(files_dir: Path = FILES_DIR) -> tuple[list[FileRecord], list[FileRecord]]:
    recs: list[FileRecord] = []
    for name in NOISE_MANIFESTS:
        recs += read_file_manifest(Path(files_dir) / f"{name}.jsonl")
    return split_noise(recs, dev_fraction=NOISE_DEV_FRACTION, seed=NOISE_SEED)


def rir_records(files_dir: Path = FILES_DIR) -> list[FileRecord]:
    return read_file_manifest(Path(files_dir) / f"{RIR_MANIFEST}.jsonl")


def speech_records(paths: Paths | None = None, files_dir: Path = FILES_DIR) -> list[FileRecord]:
    """Toda a fala com split de `splits-v1` aplicado (dev/teste viram `test_only`). O CV vem do manifest filtrado em
    `data/` (já com o split), quando existe."""
    paths = paths or Paths.from_env()
    recs: list[FileRecord] = []
    for f in sorted(Path(files_dir).glob("*.jsonl")):
        if f.name.endswith(".params.jsonl"):
            continue
        recs += [r for r in read_file_manifest(f) if r.speaker is not None and r.split != "pilot"]
    recs = dedupe_by_sha256(recs)
    out = apply_splits(recs, assign_splits(recs, load_split_config()))
    cv = paths.data / "manifests" / "cv27-pt-filtered.jsonl"
    if cv.is_file():
        out += read_file_manifest(cv)
    return out


def by_split(records: Sequence[FileRecord], split: str) -> list[FileRecord]:
    return [r for r in records if r.split == split]
