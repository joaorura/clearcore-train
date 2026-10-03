"""Checkpoints atômicos e retomada (spec §5.6).

Gravação: temporário no mesmo diretório + `fsync` + `os.replace`. Mantém os `keep` últimos e o melhor pela métrica
(menor é melhor). Cada checkpoint guarda modelo, otimizador, scheduler, passo, estados de RNG (Python, NumPy,
torch, CUDA), posição do sampler, hash da config e commit. `load_latest` ignora temporários e arquivos ilegíveis e
devolve o último checkpoint válido.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_NAME = re.compile(r"^ckpt-(\d{8})\.pt$")
_BEST = "best.json"
FORMAT = 1


class CheckpointError(RuntimeError):
    """Checkpoint incompatível com a config atual."""


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    step: int
    model: dict
    optimizer: dict
    scheduler: dict | None
    sampler_state: dict
    rng: dict
    cfg_hash: str
    commit: str | None
    metric: float | None
    extra: dict


def config_hash(cfg: Mapping) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def git_commit(repo: Path | None = None) -> str | None:
    from cctrain.paths import repo_root

    try:
        out = subprocess.run(["git", "-C", str(repo or repo_root()), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def capture_rng() -> dict:
    py = random.getstate()
    np_state = np.random.get_state()
    state = {
        "python": {"version": py[0], "state": [int(x) for x in py[1]], "gauss": py[2]},
        "numpy": {"name": np_state[0], "keys": torch.from_numpy(np_state[1].astype(np.int64)),
                  "pos": int(np_state[2]), "has_gauss": int(np_state[3]), "cached_gaussian": float(np_state[4])},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng(ck: Checkpoint) -> None:
    r = ck.rng
    p = r["python"]
    random.setstate((p["version"], tuple(p["state"]), p["gauss"]))
    n = r["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"],
                         n["cached_gaussian"]))
    torch.set_rng_state(r["torch"])
    if r["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(r["cuda"])


def _path(directory: Path, step: int) -> Path:
    return Path(directory) / f"ckpt-{step:08d}.pt"


def load_checkpoint(path: Path) -> Checkpoint:
    ck = _read(Path(path))
    if ck is None:
        raise CheckpointError(f"{path}: checkpoint ilegível ou de outro formato")
    return ck


def list_checkpoints(directory: Path) -> list[Checkpoint]:
    """Metadados dos checkpoints legíveis, em ordem de passo. Arquivos truncados ou temporários ficam de fora."""
    out = []
    for p in sorted(Path(directory).glob("ckpt-*.pt")):
        if not _NAME.match(p.name):
            continue
        ck = _read(p)
        if ck is not None:
            out.append(ck)
    return out


def _read(path: Path) -> Checkpoint | None:
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        if d["format"] != FORMAT:
            return None
        return Checkpoint(path=path, step=int(d["step"]), model=d["model"], optimizer=d["optimizer"],
                          scheduler=d["scheduler"], sampler_state=d["sampler_state"], rng=d["rng"],
                          cfg_hash=d["cfg_hash"], commit=d["commit"], metric=d["metric"], extra=d["extra"])
    except Exception:  # truncado, corrompido ou de outro formato: não é um checkpoint válido
        return None


def save_checkpoint(directory: Path, *, step: int, model: Mapping, optimizer: torch.optim.Optimizer,
                    scheduler=None, sampler_state: Mapping, cfg_hash: str, commit: str | None,
                    metric: float | None = None, keep: int = 3, extra: Mapping | None = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT, "step": int(step), "model": {k: v.detach().cpu() for k, v in model.items()},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
        "sampler_state": dict(sampler_state), "rng": capture_rng(), "cfg_hash": cfg_hash, "commit": commit,
        "metric": None if metric is None else float(metric), "extra": dict(extra or {}),
    }
    final = _path(directory, step)
    tmp = final.with_name(final.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    _fsync_dir(directory)
    _update_best(directory, step, metric)
    _prune(directory, keep)
    return final


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _update_best(directory: Path, step: int, metric: float | None) -> None:
    if metric is None:
        return
    best_path = directory / _BEST
    best = json.loads(best_path.read_text()) if best_path.is_file() else None
    if best is None or float(metric) < float(best["metric"]):
        tmp = best_path.with_name(_BEST + ".tmp")
        tmp.write_text(json.dumps({"step": int(step), "metric": float(metric)}) + "\n")
        os.replace(tmp, best_path)


def best_step(directory: Path) -> int | None:
    p = Path(directory) / _BEST
    return int(json.loads(p.read_text())["step"]) if p.is_file() else None


def _prune(directory: Path, keep: int) -> None:
    files = sorted(p for p in directory.glob("ckpt-*.pt") if _NAME.match(p.name))
    steps = [int(_NAME.match(p.name).group(1)) for p in files]
    protect = set(steps[-keep:])
    best = best_step(directory)
    if best is not None:
        protect.add(best)
    for p, s in zip(files, steps):
        if s not in protect:
            p.unlink()


def load_latest(directory: Path, *, expected_cfg_hash: str | None = None) -> Checkpoint | None:
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for p in sorted((p for p in directory.glob("ckpt-*.pt") if _NAME.match(p.name)), reverse=True):
        ck = _read(p)
        if ck is None:
            continue
        if expected_cfg_hash is not None and ck.cfg_hash != expected_cfg_hash:
            raise CheckpointError(f"{p}: hash da config difere ({ck.cfg_hash[:12]} != {expected_cfg_hash[:12]}); "
                                  "retomar com outra config reproduz outra sequência")
        return ck
    return None
