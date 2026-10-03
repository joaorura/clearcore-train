"""SHA-256 de bytes e arquivos (hex minúsculo)."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path: str | Path) -> int:
    return Path(path).stat().st_size
