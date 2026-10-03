"""Empacotamento reprodutível dos assets (spec §2.3) e leitura validada (regras de model_registry.rs)."""

from __future__ import annotations

import gzip
import io
import os
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cctrain.disk import require_free
from cctrain.hashing import sha256_file

DFNET_MEMBERS = ("enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini")
_ALLOWED_PREFIX = ("tmp", "export")


@dataclass(frozen=True)
class PackResult:
    path: Path
    sha256: str
    size: int


def pack_asset(out_path: Path, members: Sequence[tuple[str, Path]]) -> PackResult:
    out_path = Path(out_path)
    names = [n for n, _ in members]
    if len(set(names)) != len(names):
        raise ValueError(f"membros duplicados: {names}")
    for n in names:
        if "/" in n or n in ("", ".", ".."):
            raise ValueError(f"membro {n!r}: o pacote tem membros planos")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, src in members:
            data = Path(src).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.type = tarfile.REGTYPE
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    require_free(out_path.parent, 2 * len(raw))
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
        gz.write(raw)
    os.replace(tmp, out_path)
    return PackResult(out_path, sha256_file(out_path), out_path.stat().st_size)


def read_members(archive: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with tarfile.open(archive, mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = PurePosixPath(m.name).parts
            if not m.isreg():
                raise ValueError(f"{m.name}: só arquivos regulares")
            if m.mode & 0o111:
                raise ValueError(f"{m.name}: bit executável")
            if len(parts) not in (1, 3) or (len(parts) == 3 and parts[:2] != _ALLOWED_PREFIX):
                raise ValueError(f"{m.name}: caminho fora do formato (plano ou tmp/export/)")
            base = parts[-1]
            if base in out:
                raise ValueError(f"{base}: duplicado")
            f = tar.extractfile(m)
            out[base] = f.read() if f is not None else b""
    return out
