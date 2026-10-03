"""Lock de GPU entre processos: uma tarefa de GPU por vez (flock em runs/.gpu.lock)."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from cctrain.paths import Paths

LOCK_NAME = ".gpu.lock"
_POLL_S = 0.05


@contextmanager
def gpu_lock(timeout_s: float | None = None) -> Iterator[None]:
    """Segura o lock exclusivo de GPU. ``None`` bloqueia sem limite; senão levanta ``TimeoutError``.

    O flock é do descritor aberto: dois ``gpu_lock`` no mesmo processo também se excluem (não reentrante), e o
    kernel solta o lock se o processo morrer.
    """
    runs = Paths.from_env().runs
    runs.mkdir(parents=True, exist_ok=True)
    fd = os.open(runs / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if timeout_s is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"gpu_lock: não obtido em {timeout_s} s ({runs / LOCK_NAME})") from None
                    time.sleep(min(_POLL_S, remaining))
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
