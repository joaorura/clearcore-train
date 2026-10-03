"""Download em stream, com hash e guarda de disco; brutos transitórios (spec §4.2).

O bruto inteiro nunca é gravado quando o formato permite extração em stream (tar). Arquivos parciais
criados por esta função são removidos em caso de falha; nada mais é apagado.

Rede: `_chunks` repete em ConnectionError/timeout/corte no meio do corpo e retoma por `Range` (com `If-Range`)
sem o consumidor perceber: hash e tar em stream continuam do offset. Sem `Accept-Ranges`/validador, ou se o recurso
mudou, a retomada falha fechada (`ResumeUnsupportedError`/`SourceChangedError`); as funções de alto nível recomeçam
do zero quando o recurso mudou. Nenhuma mensagem de erro de rede carrega query string ou fragmento da URL
(podem ser pré-assinados): erros viram `RedactedHTTPError`.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
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
NET_CHUNK = 1 << 16  # leitura de rede: um corte descarta no máximo um bloco ainda não entregue (menos reenvio)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3  # tentativas consecutivas sem progresso
MAX_RECONNECTS = 200  # teto de reconexões numa transferência
MAX_RESTARTS = 2  # recomeços do zero quando o recurso muda durante a transferência
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 30.0


class IntegrityError(RuntimeError):
    """Hash, tamanho ou caminho fora do esperado. Falha fechado."""


class ResumeUnsupportedError(IntegrityError):
    """A transferência foi interrompida e não há como retomá-la com segurança (sem Range/validador, ou desligado)."""


class SourceChangedError(IntegrityError):
    """O recurso mudou entre a tentativa e a retomada (If-Range não casou)."""


class RedactedHTTPError(requests.HTTPError):
    """Falha de rede/HTTP sem query string nem fragmento da URL na mensagem; sem causa encadeada."""


@dataclass(frozen=True)
class StreamDigest:
    sha256: str
    md5: str
    size: int


def _is_url(source: str | Path) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _retry_delay(r: requests.Response | None, failures: int) -> float:
    """Backoff exponencial; honra Retry-After (em segundos) quando presente, com teto."""
    delay = BACKOFF_BASE_S * 2 ** max(failures - 1, 0)
    retry_after = r.headers.get("Retry-After", "") if r is not None else ""
    if retry_after.isdigit():
        delay = max(delay, float(retry_after))
    return min(delay, BACKOFF_MAX_S)


def _redact(source: str | Path) -> str:
    """Remove query string e fragmento de URLs (podem conter tokens pré-assinados)."""
    text = str(source)
    if _is_url(source):
        parts = urlsplit(text)
        host = parts.netloc.rpartition("@")[2]
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    return text


def _scrub(message: str, source: str | Path) -> str:
    """Tira de `message` a URL completa, a query string e o fragmento de `source` (mensagens de requests/urllib3/
    remotezip citam a URL ou o caminho com query)."""
    text = str(source)
    if not _is_url(source):
        return message
    parts = urlsplit(text)
    message = message.replace(text, _redact(source))
    for secret in (parts.query, parts.fragment):
        if secret:
            message = message.replace("?" + secret, "").replace("#" + secret, "").replace(secret, "<redigido>")
    return re.sub(r"https?://[^\s'\"]*\?[^\s'\"]*", lambda m: m.group(0).split("?", 1)[0], message)


def _redacted(exc: BaseException, source: str | Path, response: requests.Response | None = None) -> RedactedHTTPError:
    return RedactedHTTPError(f"{type(exc).__name__}: {_scrub(str(exc), source)}", response=response)


def _raise_for_status(r: requests.Response, source: str | Path) -> None:
    if r.status_code >= 400:
        kind = "Client" if r.status_code < 500 else "Server"
        raise RedactedHTTPError(f"{r.status_code} {kind} Error: {r.reason} for url: {_redact(source)}",
                                response=r) from None


_RESUMABLE_ERRORS = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)


class _Truncated(Exception):
    """O corpo terminou antes do tamanho anunciado."""


def _parse_user_range(headers: Mapping[str, str]) -> tuple[int, int | None] | None:
    """`Range: bytes=a-b` / `bytes=a-` do chamador; None se não há ou se não dá para retomar por offset."""
    value = next((v for k, v in headers.items() if k.lower() == "range"), None)
    if value is None:
        return (0, None)
    m = re.fullmatch(r"bytes=(\d+)-(\d*)", value.strip())
    if not m:
        return None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def _after_failure(e: BaseException, source: str | Path, offset: int, window_len: int | None, progressed: bool,
                   failures: int, can_resume: bool) -> tuple[int, bool]:
    """Decide após um corte: (novo contador de falhas, True se a janela já foi toda entregue). Levanta se desiste."""
    if window_len is not None and offset >= window_len:
        return failures, True
    failures = 0 if progressed else failures + 1
    if offset > 0 and not can_resume:
        raise ResumeUnsupportedError(
            f"{_redact(source)}: transferência interrompida em {offset} bytes e sem retomada segura "
            "(sem Accept-Ranges/validador, ou resume=False)") from None
    if failures >= MAX_ATTEMPTS:
        raise _redacted(e, source) from None
    time.sleep(_retry_delay(None, max(failures, 1)))
    return failures, False


def _chunks(source: str | Path, *, headers: Mapping[str, str] | None = None,
            session: requests.Session | None = None, timeout: float = 60.0, resume: bool = True) -> Iterator[bytes]:
    if not _is_url(source):
        with open(source, "rb") as f:
            yield from iter(lambda: f.read(CHUNK), b"")
        return
    url = str(source)
    s = session or requests.Session()
    base = {"User-Agent": USER_AGENT, **(headers or {})}
    window = _parse_user_range(base)  # (início absoluto, fim inclusivo | None); None = Range não retomável
    offset = 0  # bytes já entregues ao consumidor, relativos ao início da janela
    validator: str | None = None
    can_resume = False
    window_len: int | None = None
    failures = reconnects = 0
    while True:
        reconnects += 1
        if reconnects > MAX_RECONNECTS:
            raise RedactedHTTPError(f"transferência desistiu após {MAX_RECONNECTS} reconexões: {_redact(source)}")
        req = dict(base)
        if offset > 0:
            assert window is not None and validator is not None
            req["Range"] = f"bytes={window[0] + offset}-{'' if window[1] is None else window[1]}"
            req["If-Range"] = validator
        progressed = False
        r = None
        try:
            r = s.get(url, stream=True, timeout=timeout, headers=req)
            if r.status_code in RETRY_STATUSES:
                failures += 1
                if failures >= MAX_ATTEMPTS:
                    _raise_for_status(r, source)
                delay = _retry_delay(r, failures)
                r.close()
                time.sleep(delay)
                continue
            if offset > 0:
                if r.status_code == 200:
                    raise SourceChangedError(f"{_redact(source)}: o recurso mudou durante a transferência")
                if r.status_code != 206:
                    _raise_for_status(r, source)
                    raise ResumeUnsupportedError(f"{_redact(source)}: resposta {r.status_code} à retomada")
                m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", r.headers.get("Content-Range", ""))
                if not m or int(m.group(1)) != window[0] + offset:
                    raise ResumeUnsupportedError(f"{_redact(source)}: Content-Range da retomada não confere")
            else:
                _raise_for_status(r, source)
                etag = r.headers.get("ETag", "")
                validator = etag if etag and not etag.startswith("W/") else (r.headers.get("Last-Modified") or None)
                can_resume = (resume and window is not None and validator is not None
                              and "bytes" in r.headers.get("Accept-Ranges", "").lower()
                              and r.headers.get("Content-Encoding", "").lower().strip() in ("", "identity")
                              and r.status_code in (200, 206))
                length = r.headers.get("Content-Length", "")
                if length.isdigit() and r.headers.get("Content-Encoding", "").lower().strip() in ("", "identity"):
                    window_len = int(length)
            for chunk in r.iter_content(NET_CHUNK):
                if chunk:
                    progressed = True
                    offset += len(chunk)
                    yield chunk
            if window_len is not None and offset < window_len:
                raise _Truncated
            return
        except RedactedHTTPError:
            raise
        except requests.RequestException as e:
            if not isinstance(e, _RESUMABLE_ERRORS):
                raise _redacted(e, source, getattr(e, "response", None)) from None
            failures, can_stop = _after_failure(e, source, offset, window_len, progressed, failures, can_resume)
            if can_stop:
                return
        except _Truncated as e:
            failures, can_stop = _after_failure(e, source, offset, window_len, progressed, failures, can_resume)
            if can_stop:
                return
        finally:
            if r is not None:
                r.close()


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
    try:
        for restart in range(MAX_RESTARTS + 1):
            reader = _HashingReader(_chunks(source, headers=headers, session=session))
            try:
                buf = bytearray(CHUNK)
                with open(part, "wb") as f:
                    while n := reader.readinto(buf):
                        f.write(memoryview(buf)[:n])
                break
            except SourceChangedError:
                if restart == MAX_RESTARTS:
                    raise
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


def _extract_once(source: str | Path, staging: Path, select: Callable[[str], str | None],
                  done: Callable[[str], bool] | None, session: requests.Session | None,
                  ) -> tuple[_HashingReader, list[str]]:
    reader = _HashingReader(_chunks(source, session=session))
    extracted: list[str] = []
    seen: set[PurePosixPath] = set()
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
            extracted.append(str(safe))
            dest = staging / safe
            if done is not None and done(str(safe)) and dest.is_file():
                continue  # já extraído e verificado por quem chamou: o tar em stream pula o corpo
            dest.parent.mkdir(parents=True, exist_ok=True)
            require_free(staging, member.size)
            part = dest.with_name(dest.name + ".part")
            try:
                with open(part, "wb") as f:
                    shutil.copyfileobj(tar.extractfile(member), f, CHUNK)
                os.replace(part, dest)  # o membro só existe inteiro: `done` pode confiar na existência
            except BaseException:
                part.unlink(missing_ok=True)
                raise
    reader.drain()
    return reader, extracted


def stream_extract_tar(source: str | Path, out_dir: Path, select: Callable[[str], str | None], *,
                       expected_sha256: str | None = None, expected_md5: str | None = None,
                       expected_size: int | None = None, need_bytes: int | None = None,
                       session: requests.Session | None = None,
                       done: Callable[[str], bool] | None = None) -> tuple[StreamDigest, list[str]]:
    """Extrai em stream os membros escolhidos por `select` para `out_dir.staging` e renomeia no fim.

    `done(rel)` liga a retomada entre processos: com ele, o staging de uma execução interrompida (rede, disco,
    Ctrl-C) é mantido e reaproveitado, e os membros para os quais devolve True são pulados (o corpo é lido do stream
    mas não regravado). Quem passa `done` responde por verificar o membro (ex.: tamanho/hash do manifesto); cada
    membro só aparece em `staging` inteiro (grava em `.part` e renomeia). Hash final divergente apaga o staging.
    Se o recurso mudar durante a transferência, recomeça do zero com o staging limpo.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise IntegrityError(f"{out_dir} já existe; inspecione antes de reextrair")
    staging = out_dir.with_name(out_dir.name + ".staging")
    if staging.exists() and done is None:
        raise IntegrityError(f"{staging} existe de uma extração interrompida; inspecione")
    require_free(out_dir.parent, need_bytes or 0)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for restart in range(MAX_RESTARTS + 1):
            try:
                reader, extracted = _extract_once(source, staging, select, done, session)
                break
            except SourceChangedError:
                if restart == MAX_RESTARTS:
                    raise
                shutil.rmtree(staging)  # o que havia no staging é da versão anterior do recurso
                staging.mkdir()
        d = reader.digest()
        _verify(d, _redact(source), expected_sha256, expected_md5, expected_size)
    except BaseException as e:
        keep = done is not None and (not isinstance(e, IntegrityError) or isinstance(e, ResumeUnsupportedError))
        if not keep:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    os.replace(staging, out_dir)
    return d, sorted(extracted)


@contextlib.contextmanager
def _redact_remote_errors(url: str) -> Iterator[None]:
    """Reemite como `RedactedHTTPError` as falhas do remotezip/requests, cujas mensagens citam a URL com query."""
    from remotezip import RemoteZipError

    try:
        yield
    except (requests.RequestException, RemoteZipError) as e:
        raise _redacted(e, url, getattr(e, "response", None)) from None


def list_zip_members(url: str) -> dict[str, int]:
    from remotezip import RemoteZip

    with _redact_remote_errors(url), RemoteZip(url, headers={"User-Agent": USER_AGENT}) as z:
        return {i.filename: i.file_size for i in z.infolist()}


def fetch_zip_members(url: str, members: Mapping[str, Path]) -> dict[str, StreamDigest]:
    from remotezip import RemoteZip

    out: dict[str, StreamDigest] = {}
    with _redact_remote_errors(url), RemoteZip(url, headers={"User-Agent": USER_AGENT}) as z:
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
