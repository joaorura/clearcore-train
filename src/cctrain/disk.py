"""Guarda de disco: só disco persistente e falha fechada quando falta espaço.

Liberar espaço é decisão do dono; este módulo nunca apaga nada.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

ENV_MIN_FREE_GB = "CCTRAIN_MIN_FREE_GB"
DEFAULT_MIN_FREE_GB = 10.0
VOLATILE_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs"})
GB = 10**9  # decimal, a mesma unidade de `disk.reserve_gb` do budget-v1 e do relatório M0
MOUNTS = Path("/proc/self/mounts")


class DiskError(RuntimeError):
    """Destino volátil ou espaço insuficiente."""


def _existing_ancestor(path: Path) -> Path:
    p = Path(path).expanduser().resolve()
    while not p.exists():
        p = p.parent
    return p


def _unescape(field: str) -> str:
    return field.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def filesystem_type(path: Path, mounts_file: Path = MOUNTS) -> str:
    target = Path(path).expanduser().resolve()
    best: Path | None = None
    best_type: str | None = None
    for line in Path(mounts_file).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point, fs_type = Path(_unescape(parts[1])), parts[2]
        if target == mount_point or mount_point in target.parents:
            if best is None or len(mount_point.parts) >= len(best.parts):
                best, best_type = mount_point, fs_type
    if best_type is None:
        raise DiskError(f"sem ponto de montagem para {target}")
    return best_type


def assert_persistent(path: Path, mounts_file: Path = MOUNTS) -> None:
    fs = filesystem_type(path, mounts_file)
    if fs in VOLATILE_FILESYSTEMS:
        raise DiskError(
            f"{path} está em {fs} (volátil); dados, checkpoints e builds vão para disco persistente"
        )


def reserve_bytes(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    raw = env.get(ENV_MIN_FREE_GB)
    value = DEFAULT_MIN_FREE_GB if raw in (None, "") else float(raw)
    if value < 0:
        raise DiskError(f"{ENV_MIN_FREE_GB} negativo: {value}")
    return int(value * GB)


def require_free(
    path: Path,
    need_bytes: int,
    *,
    reserve: int | None = None,
    mounts_file: Path = MOUNTS,
    usage: Callable = shutil.disk_usage,
) -> int:
    assert_persistent(path, mounts_file)
    reserve = reserve_bytes() if reserve is None else reserve
    free = usage(_existing_ancestor(Path(path))).free
    if free - need_bytes < reserve:
        raise DiskError(
            f"espaço insuficiente em {path}: livre {free / GB:.2f} GB, necessário {need_bytes / GB:.2f} GB"
            f" + reserva {reserve / GB:.2f} GB. Liberar disco é decisão do dono; nada é apagado aqui."
        )
    return free
