"""Download em stream, com hash e guarda de disco; brutos transitórios (spec §4.2).

O bruto inteiro nunca é gravado quando o formato permite extração em stream (tar). Arquivos parciais
criados por esta função são removidos em caso de falha; nada mais é apagado.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

import requests

from cctrain.disk import require_free

USER_AGENT = "clearcore-train/0.1 (research data fetch)"
CHUNK = 1 << 20
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 30.0


class IntegrityError(RuntimeError):
    """Hash, tamanho ou caminho fora do esperado. Falha fechado."""


@dataclass(frozen=True)
class StreamDigest:
    sha256: str
    md5: str
    size: int


def _is_url(source: str | Path) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _retry_delay(r: requests.Response, attempt: int) -> float:
    """Backoff exponencial; honra Retry-After (em segundos) quando presente, com teto."""
    delay = BACKOFF_BASE_S * 2 ** (attempt - 1)
    retry_after = r.headers.get("Retry-After", "")
    if retry_after.isdigit():
        delay = max(delay, float(retry_after))
    return min(delay, BACKOFF_MAX_S)


def _chunks(source: str | Path, *, headers: Mapping[str, str] | None = None,
            session: requests.Session | None = None, timeout: float = 60.0) -> Iterator[bytes]:
    if not _is_url(source):
        with open(source, "rb") as f:
            yield from iter(lambda: f.read(CHUNK), b"")
        return
    s = session or requests.Session()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = s.get(str(source), stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT, **(headers or {})})
        if r.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
            r.close()
            time.sleep(_retry_delay(r, attempt))
            continue
        break
    with r:
        r.raise_for_status()
        for chunk in r.iter_content(CHUNK):
            if chunk:
                yield chunk


class _HashingReader(io.RawIOBase):
    def __init__(self, chunks: Iterator[bytes]):
        self._it = iter(chunks)
        self._buf = bytearray()
        self._sha = hashlib.sha256()
        self._md5 = hashlib.md5()
        self.size = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while len(self._buf) < len(b):
            try:
                self._buf += next(self._it)
            except StopIteration:
                break
        n = min(len(b), len(self._buf))
        data = bytes(self._buf[:n])
        del self._buf[:n]
        b[:n] = data
        self._sha.update(data)
        self._md5.update(data)
        self.size += n
        return n

    def drain(self) -> None:
        buf = bytearray(CHUNK)
        while self.readinto(buf):
            pass

    def digest(self) -> StreamDigest:
        return StreamDigest(self._sha.hexdigest(), self._md5.hexdigest(), self.size)


def _redact(source: str | Path) -> str:
    """Remove query string e fragmento de URLs (podem conter tokens pré-assinados)."""
    text = str(source)
    if _is_url(source):
        parts = urlsplit(text)
        host = parts.netloc.rpartition("@")[2]
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    return text


def _verify(d: StreamDigest, what: str, sha256: str | None, md5: str | None, size: int | None) -> None:
    if size is not None and d.size != size:
        raise IntegrityError(f"{what}: tamanho {d.size} != {size}")
    if sha256 is not None and d.sha256 != sha256.lower():
        raise IntegrityError(f"{what}: sha256 {d.sha256} != {sha256}")
    if md5 is not None and d.md5 != md5.lower():
        raise IntegrityError(f"{what}: md5 {d.md5} != {md5}")


def stream_download(source: str | Path, dest: Path, *, expected_sha256: str | None = None,
                    expected_md5: str | None = None, expected_size: int | None = None,
                    need_bytes: int | None = None, headers: Mapping[str, str] | None = None,
                    session: requests.Session | None = None) -> StreamDigest:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    require_free(dest.parent, need_bytes if need_bytes is not None else (expected_size or 0))
    part = dest.with_name(dest.name + ".part")
    reader = _HashingReader(_chunks(source, headers=headers, session=session))
    try:
        buf = bytearray(CHUNK)
        with open(part, "wb") as f:
            while n := reader.readinto(buf):
                f.write(memoryview(buf)[:n])
        d = reader.digest()
        _verify(d, _redact(source), expected_sha256, expected_md5, expected_size)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return d


def _safe_rel(rel: str) -> PurePosixPath:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise IntegrityError(f"caminho de saída inseguro: {rel!r}")
    return p


def stream_extract_tar(source: str | Path, out_dir: Path, select: Callable[[str], str | None], *,
                       expected_sha256: str | None = None, expected_md5: str | None = None,
                       expected_size: int | None = None, need_bytes: int | None = None,
                       session: requests.Session | None = None) -> tuple[StreamDigest, list[str]]:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise IntegrityError(f"{out_dir} já existe; inspecione antes de reextrair")
    staging = out_dir.with_name(out_dir.name + ".staging")
    if staging.exists():
        raise IntegrityError(f"{staging} existe de uma extração interrompida; inspecione")
    require_free(out_dir.parent, need_bytes or 0)
    staging.mkdir(parents=True)
    reader = _HashingReader(_chunks(source, session=session))
    extracted: list[str] = []
    seen: set[PurePosixPath] = set()
    try:
        with tarfile.open(fileobj=io.BufferedReader(reader, CHUNK), mode="r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                rel = select(member.name)
                if rel is None:
                    continue
                safe = _safe_rel(rel)
                if safe in seen:
                    raise IntegrityError(f"caminho de saída duplicado: {rel!r}")
                seen.add(safe)
                dest = staging / safe
                dest.parent.mkdir(parents=True, exist_ok=True)
                require_free(staging, member.size)
                src = tar.extractfile(member)
                with open(dest, "wb") as f:
                    shutil.copyfileobj(src, f, CHUNK)
                extracted.append(str(safe))
        reader.drain()
        d = reader.digest()
        _verify(d, _redact(source), expected_sha256, expected_md5, expected_size)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    os.replace(staging, out_dir)
    return d, sorted(extracted)


def list_zip_members(url: str) -> dict[str, int]:
    from remotezip import RemoteZip

    with RemoteZip(url, headers={"User-Agent": USER_AGENT}) as z:
        return {i.filename: i.file_size for i in z.infolist()}


def fetch_zip_members(url: str, members: Mapping[str, Path]) -> dict[str, StreamDigest]:
    from remotezip import RemoteZip

    out: dict[str, StreamDigest] = {}
    with RemoteZip(url, headers={"User-Agent": USER_AGENT}) as z:
        infos = {i.filename: i for i in z.infolist()}
        missing = [m for m in members if m not in infos]
        if missing:
            raise IntegrityError(f"membros ausentes no zip: {missing}")
        for member, dest in members.items():
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            require_free(dest.parent, infos[member].file_size)
            part = dest.with_name(dest.name + ".part")
            try:
                with z.open(member) as src:
                    reader = _HashingReader(iter(lambda s=src: s.read(CHUNK), b""))
                    buf = bytearray(CHUNK)
                    with open(part, "wb") as f:
                        while n := reader.readinto(buf):
                            f.write(memoryview(buf)[:n])
                d = reader.digest()
                _verify(d, member, None, None, infos[member].file_size)
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            os.replace(part, dest)
            out[member] = d
    return out


def cut_clip(raw: Path, dest: Path, start_s: float, duration_s: float = 10.0) -> float:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    require_free(dest.parent, int(duration_s * 48_000 * 2) + (1 << 20))
    part = dest.with_name(dest.name + ".part")
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start_s), "-t", str(duration_s),
                        "-i", str(raw), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", "-f", "wav",
                        str(part)], check=True)
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(part)], check=True, capture_output=True, text=True)
        duration = float(out.stdout.strip())
        if abs(duration - duration_s) > 0.01:
            raise IntegrityError(f"{dest}: duração {duration} s, esperado {duration_s} s")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return duration
