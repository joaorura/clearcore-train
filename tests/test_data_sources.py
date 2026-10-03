import functools
import gzip
import hashlib
import http.server
import io
import socket
import subprocess
import tarfile
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cctrain.data.sources import (
    NET_CHUNK,
    IntegrityError,
    RedactedHTTPError,
    ResumeUnsupportedError,
    SourceChangedError,
    cut_clip,
    fetch_zip_members,
    list_zip_members,
    stream_download,
    stream_extract_tar,
)
from cctrain.disk import DiskError


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # silencioso nos testes
        pass

    def do_GET(self):  # noqa: N802
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        rng = self.headers.get("Range")
        if rng:
            first, last = rng.split("=", 1)[1].split("-")
            if first == "":
                start, end = max(0, len(data) - int(last)), len(data) - 1
            else:
                start, end = int(first), min(int(last) if last else len(data) - 1, len(data) - 1)
            body = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def www(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             functools.partial(_RangeHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join()


def _targz(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue())


def test_stream_download_verifies_and_is_atomic(www, tmp_path):
    root, base = www
    (root / "a.bin").write_bytes(b"x" * 5000)
    sha = hashlib.sha256(b"x" * 5000).hexdigest()
    d = stream_download(f"{base}/a.bin", tmp_path / "out" / "a.bin", expected_sha256=sha, expected_size=5000)
    assert d.sha256 == sha and d.size == 5000
    assert d.md5 == hashlib.md5(b"x" * 5000).hexdigest()
    with pytest.raises(IntegrityError):
        stream_download(f"{base}/a.bin", tmp_path / "out" / "b.bin", expected_sha256="0" * 64)
    assert not (tmp_path / "out" / "b.bin").exists()
    assert not (tmp_path / "out" / "b.bin.part").exists()


def test_stream_download_fails_closed_on_disk(www, tmp_path):
    root, base = www
    (root / "a.bin").write_bytes(b"x")
    with pytest.raises(DiskError):
        stream_download(f"{base}/a.bin", tmp_path / "a.bin", need_bytes=10**16)


def test_range_header_is_honored(www, tmp_path):
    root, base = www
    (root / "big.bin").write_bytes(bytes(range(256)) * 10)
    d = stream_download(f"{base}/big.bin", tmp_path / "head.bin", headers={"Range": "bytes=0-99"})
    assert d.size == 100 and (tmp_path / "head.bin").read_bytes() == bytes(range(100))


def test_stream_extract_tar_selects_members_and_checks_md5(www, tmp_path):
    root, base = www
    raw = _targz({"L/dev/1/a.flac": b"A", "L/dev/1/a.txt": b"T", "L/dev/2/b.flac": b"BB"})
    (root / "x.tar.gz").write_bytes(raw)
    select = lambda name: ("lib/" + name.split("/", 1)[1]) if name.endswith(".flac") else None  # noqa: E731
    d, files = stream_extract_tar(f"{base}/x.tar.gz", tmp_path / "lib-out", select,
                                  expected_md5=hashlib.md5(raw).hexdigest())
    assert files == ["lib/dev/1/a.flac", "lib/dev/2/b.flac"]
    assert (tmp_path / "lib-out" / "lib/dev/2/b.flac").read_bytes() == b"BB"
    assert d.size == len(raw)
    with pytest.raises(IntegrityError):
        stream_extract_tar(f"{base}/x.tar.gz", tmp_path / "bad-out", select, expected_md5="0" * 32)
    assert not (tmp_path / "bad-out").exists()
    assert not (tmp_path / "bad-out.staging").exists()


def test_stream_extract_tar_from_local_file_and_traversal(tmp_path):
    archive = tmp_path / "local.tar.gz"
    archive.write_bytes(_targz({"a.mp3": b"M"}))
    _, files = stream_extract_tar(archive, tmp_path / "cv", lambda n: "clips/" + n)
    assert files == ["clips/a.mp3"]
    with pytest.raises(IntegrityError):
        stream_extract_tar(archive, tmp_path / "evil", lambda n: "../" + n)


def test_fetch_zip_members_by_range(www, tmp_path):
    root, base = www
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("wav48/p1/p1_001_mic1.flac", b"F" * 3000)
        z.writestr("wav48/p1/p1_001_mic2.flac", b"G" * 3000)
        z.writestr("license_text", b"Creative Commons Attribution 4.0")
    (root / "c.zip").write_bytes(buf.getvalue())
    names = list_zip_members(f"{base}/c.zip")
    assert names["license_text"] == len(b"Creative Commons Attribution 4.0")
    got = fetch_zip_members(f"{base}/c.zip", {"wav48/p1/p1_001_mic1.flac": tmp_path / "v" / "a.flac",
                                              "license_text": tmp_path / "v" / "license_text"})
    assert (tmp_path / "v" / "a.flac").read_bytes() == b"F" * 3000
    assert got["license_text"].sha256 == hashlib.sha256(b"Creative Commons Attribution 4.0").hexdigest()
    with pytest.raises(IntegrityError):
        fetch_zip_members(f"{base}/c.zip", {"nope": tmp_path / "v" / "nope"})


def test_cut_clip_produces_10s_mono_48k(tmp_path):
    raw = tmp_path / "raw.wav"
    sf.write(raw, (0.1 * np.random.default_rng(0).standard_normal((44_100 * 20, 2))).astype("f4"), 44_100)
    dur = cut_clip(raw, tmp_path / "clip.wav", start_s=5.0)
    info = sf.info(tmp_path / "clip.wav")
    assert abs(dur - 10.0) <= 0.01
    assert (info.samplerate, info.channels, info.subtype) == (48_000, 1, "PCM_16")


def test_stream_extract_tar_disk_error_leaves_no_staging(tmp_path):
    archive = tmp_path / "local.tar.gz"
    archive.write_bytes(_targz({"a.mp3": b"M"}))
    with pytest.raises(DiskError):
        stream_extract_tar(archive, tmp_path / "cv", lambda n: n, need_bytes=10**16)
    assert not (tmp_path / "cv.staging").exists() and not (tmp_path / "cv").exists()
    _, files = stream_extract_tar(archive, tmp_path / "cv", lambda n: n)  # retry is not refused
    assert files == ["a.mp3"]


def test_stream_extract_tar_rejects_duplicate_destination(tmp_path):
    archive = tmp_path / "dup.tar.gz"
    archive.write_bytes(_targz({"a/x.flac": b"1", "b/x.flac": b"2"}))
    with pytest.raises(IntegrityError, match="duplicado"):
        stream_extract_tar(archive, tmp_path / "out", lambda n: "same/" + n.rsplit("/", 1)[1])
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.staging").exists()


def test_cut_clip_leaves_no_file_on_failure(tmp_path):
    raw = tmp_path / "short.wav"
    sf.write(raw, np.zeros(48_000, dtype="f4"), 48_000)  # 1 s: cannot yield 10 s
    dest = tmp_path / "clip.wav"
    with pytest.raises(IntegrityError):
        cut_clip(raw, dest, start_s=0.0)
    assert not dest.exists() and not (tmp_path / "clip.wav.part").exists()
    with pytest.raises(subprocess.CalledProcessError):
        cut_clip(tmp_path / "missing.wav", dest, start_s=0.0)
    assert not dest.exists() and not (tmp_path / "clip.wav.part").exists()


def test_empty_or_uppercase_hash_does_not_disable_or_break_verification(www, tmp_path):
    root, base = www
    (root / "a.bin").write_bytes(b"abc")
    with pytest.raises(IntegrityError):
        stream_download(f"{base}/a.bin", tmp_path / "e.bin", expected_sha256="")
    assert not (tmp_path / "e.bin").exists()
    sha = hashlib.sha256(b"abc").hexdigest().upper()
    assert stream_download(f"{base}/a.bin", tmp_path / "u.bin", expected_sha256=sha).size == 3


def test_integrity_error_masks_url_query(www, tmp_path):
    root, base = www
    (root / "a.bin").write_bytes(b"abc")
    with pytest.raises(IntegrityError) as exc:
        stream_download(f"{base}/a.bin?X-Amz-Signature=SECRET#frag", tmp_path / "a.bin", expected_size=99)
    assert "SECRET" not in str(exc.value) and "/a.bin" in str(exc.value)


class _FlakyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        self.server.hits += 1
        status = self.server.script[min(self.server.hits, len(self.server.script)) - 1]
        body = b"payload" * 100 if status == 200 else b"err"
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def flaky(monkeypatch):
    monkeypatch.setattr("cctrain.data.sources.BACKOFF_BASE_S", 0.01)
    servers = []

    def make(script):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FlakyHandler)
        server.script, server.hits = script, 0
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, f"http://127.0.0.1:{server.server_address[1]}/f"

    yield make
    for server in servers:
        server.shutdown()
        server.server_close()


def test_stream_download_retries_429_and_5xx_then_succeeds(flaky, tmp_path):
    server, url = flaky([429, 503, 200])
    d = stream_download(url, tmp_path / "ok.bin")
    assert d.size == 700 and server.hits == 3
    assert (tmp_path / "ok.bin").read_bytes() == b"payload" * 100


def test_stream_download_gives_up_after_three_attempts(flaky, tmp_path):
    import requests

    server, url = flaky([429])
    with pytest.raises(requests.HTTPError):
        stream_download(url, tmp_path / "no.bin")
    assert server.hits == 3
    assert not (tmp_path / "no.bin").exists() and not (tmp_path / "no.bin.part").exists()


def test_stream_download_does_not_retry_404(flaky, tmp_path):
    import requests

    server, url = flaky([404])
    with pytest.raises(requests.HTTPError):
        stream_download(url, tmp_path / "no.bin")
    assert server.hits == 1


# ---- retry, retomada por Range e redação (M1 Tarefa 5) ----

PAYLOAD = np.random.default_rng(7).integers(0, 256, 3 * (1 << 20) + 12_345, dtype=np.uint8).tobytes()
PAYLOAD2 = np.random.default_rng(8).integers(0, 256, 2 * (1 << 20) + 777, dtype=np.uint8).tobytes()


class _ResumableHandler(http.server.BaseHTTPRequestHandler):
    """Servidor de teste: honra Range/If-Range, anuncia ETag e corta a conexão no meio do corpo conforme o roteiro."""

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        srv = self.server
        srv.requests.append({"path": self.path, "range": self.headers.get("Range"),
                             "if_range": self.headers.get("If-Range")})
        if srv.fail_status:
            self.send_error(srv.fail_status)
            return
        content, etag = srv.resource()
        start, status = 0, 200
        rng = self.headers.get("Range")
        if rng and srv.ranges and self.headers.get("If-Range") in (None, etag):
            first, _, last = rng.split("=", 1)[1].partition("-")
            start, status = int(first), 206
            end = int(last) if last else len(content) - 1
            body = content[start : end + 1]
        else:
            body = content
        self.send_response(status)
        if srv.ranges:
            self.send_header("Accept-Ranges", "bytes")
        if srv.etag:
            self.send_header("ETag", etag)
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(content)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        cut = srv.cuts.pop(0) if srv.cuts else None
        self.wfile.write(body if cut is None else body[:cut])
        if cut is not None:
            self.wfile.flush()
            self.request.shutdown(socket.SHUT_RDWR)


@pytest.fixture()
def resumable(monkeypatch):
    monkeypatch.setattr("cctrain.data.sources.BACKOFF_BASE_S", 0.01)
    servers = []

    def make(content=PAYLOAD, *, cuts=(), ranges=True, etag=True, fail_status=0):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ResumableHandler)
        server.requests, server.cuts, server.ranges, server.etag = [], list(cuts), ranges, etag
        server.fail_status = fail_status
        server.state = {"content": content, "etag": '"v1"'}
        server.resource = lambda: (server.state["content"], server.state["etag"])
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, f"http://127.0.0.1:{server.server_address[1]}/blob.bin"

    yield make
    for server in servers:
        server.shutdown()
        server.server_close()


def test_download_survives_three_mid_stream_cuts_without_duplicating_bytes(resumable, tmp_path):
    server, url = resumable(cuts=[1_500_007, 100_000, 800_001])
    d = stream_download(url, tmp_path / "blob.bin", expected_sha256=hashlib.sha256(PAYLOAD).hexdigest())
    assert d.size == len(PAYLOAD) and d.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert (tmp_path / "blob.bin").read_bytes() == PAYLOAD
    assert not (tmp_path / "blob.bin.part").exists()
    reqs = server.requests
    assert len(reqs) == 4 and reqs[0]["range"] is None
    starts = [int(r["range"].split("=")[1].split("-")[0]) for r in reqs[1:]]
    assert all(x > 0 for x in starts) and starts == sorted(starts)
    assert 1_500_007 - NET_CHUNK < starts[0] <= 1_500_007  # retoma do último byte entregue, em blocos de leitura
    assert all(r["if_range"] == '"v1"' for r in reqs[1:])


def test_extract_tar_resumes_mid_stream_transparently(resumable, tmp_path):
    members = {f"m/{i}.bin": np.random.default_rng(i).integers(0, 256, 700_000, dtype=np.uint8).tobytes()
               for i in range(4)}
    raw = _targz(members)
    server, url = resumable(raw, cuts=[len(raw) // 3, len(raw) // 3, 5_000])
    d, files = stream_extract_tar(url, tmp_path / "out", lambda n: n, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert d.size == len(raw) and files == sorted(members)
    for name, data in members.items():
        assert (tmp_path / "out" / name).read_bytes() == data
    assert len(server.requests) == 4


def test_gives_up_after_three_consecutive_failures_without_progress(resumable, tmp_path):
    server, url = resumable(cuts=[200_000, 0, 0, 0])
    with pytest.raises(RedactedHTTPError):
        stream_download(url, tmp_path / "blob.bin")
    assert len(server.requests) == 4  # corte com progresso zera a conta; depois 3 sem progresso
    assert not (tmp_path / "blob.bin").exists() and not (tmp_path / "blob.bin.part").exists()


def test_server_without_accept_ranges_fails_closed_on_cut(resumable, tmp_path):
    server, url = resumable(cuts=[1_000_000], ranges=False)
    with pytest.raises(ResumeUnsupportedError):
        stream_download(url, tmp_path / "blob.bin")
    assert len(server.requests) == 1
    assert not (tmp_path / "blob.bin").exists() and not (tmp_path / "blob.bin.part").exists()


def test_server_without_accept_ranges_still_downloads_when_uninterrupted(resumable, tmp_path):
    _, url = resumable(ranges=False)
    assert stream_download(url, tmp_path / "blob.bin").size == len(PAYLOAD)


def test_server_without_validator_fails_closed_on_cut(resumable, tmp_path):
    _, url = resumable(cuts=[1_000_000], etag=False)
    with pytest.raises(ResumeUnsupportedError):
        stream_download(url, tmp_path / "blob.bin")


def test_resume_disabled_fails_on_mid_stream_cut(resumable):
    from cctrain.data.sources import _chunks

    server, url = resumable(cuts=[1_000_000])
    with pytest.raises(ResumeUnsupportedError):
        list(_chunks(url, resume=False))
    assert len(server.requests) == 1


def test_changed_etag_restarts_from_zero(resumable, tmp_path):
    server, url = resumable(cuts=[1_200_000])
    original = server.resource

    def swap_after_first_request():
        if len(server.requests) >= 2:  # a retomada (2ª requisição) já vê o conteúdo novo
            server.state.update(content=PAYLOAD2, etag='"v2"')
        return original()

    server.resource = swap_after_first_request
    d = stream_download(url, tmp_path / "blob.bin", expected_sha256=hashlib.sha256(PAYLOAD2).hexdigest())
    assert (tmp_path / "blob.bin").read_bytes() == PAYLOAD2 and d.size == len(PAYLOAD2)
    last = server.requests[-1]
    assert last["range"] is None and last["if_range"] is None  # recomeço do zero, sem Range
    assert server.requests[1]["if_range"] == '"v1"'


def test_changed_source_every_time_gives_up(resumable, tmp_path):
    server, url = resumable(cuts=[200_000] * 8)
    counter = {"n": 0}

    def always_new():
        counter["n"] += 1
        return PAYLOAD + bytes([counter["n"] % 256]), f'"v{counter["n"]}"'

    server.resource = always_new
    with pytest.raises(SourceChangedError):
        stream_download(url, tmp_path / "blob.bin")
    assert not (tmp_path / "blob.bin").exists() and not (tmp_path / "blob.bin.part").exists()


def test_user_range_header_is_resumed_inside_the_window(resumable, tmp_path):
    server, url = resumable(cuts=[300_000])
    d = stream_download(url, tmp_path / "head.bin", headers={"Range": "bytes=1000-999999"})
    assert (tmp_path / "head.bin").read_bytes() == PAYLOAD[1000:1_000_000] and d.size == 999_000
    first = int(server.requests[1]["range"].split("=")[1].split("-")[0])
    assert 301_000 - NET_CHUNK < first <= 301_000 and server.requests[1]["range"].endswith("-999999")


def test_http_error_message_has_no_query_or_signature(resumable, tmp_path):
    import requests

    server, url = resumable(fail_status=404)
    with pytest.raises(RedactedHTTPError) as exc:
        stream_download(f"{url}?X-Amz-Signature=SECRET&X-Amz-Credential=KEY#frag", tmp_path / "x.bin")
    assert isinstance(exc.value, requests.HTTPError)  # quem já tratava HTTPError continua tratando
    assert "SECRET" not in str(exc.value) and "KEY" not in str(exc.value) and "?" not in str(exc.value)
    assert "404" in str(exc.value) and exc.value.response.status_code == 404
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_retry_exhaustion_message_has_no_query(resumable, tmp_path):
    server, url = resumable(fail_status=503)
    with pytest.raises(RedactedHTTPError) as exc:
        stream_download(f"{url}?sig=SECRET", tmp_path / "x.bin")
    assert len(server.requests) == 3 and "SECRET" not in str(exc.value)


def test_connection_error_message_has_no_query(tmp_path, monkeypatch):
    monkeypatch.setattr("cctrain.data.sources.BACKOFF_BASE_S", 0.01)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # fechado ao sair do with: ninguém escuta
    with pytest.raises(RedactedHTTPError) as exc:
        stream_download(f"http://127.0.0.1:{port}/a.bin?X-Amz-Signature=SECRET", tmp_path / "x.bin")
    assert "SECRET" not in str(exc.value) and "?" not in str(exc.value)
    assert exc.value.__cause__ is None


def test_remotezip_errors_do_not_leak_the_query(resumable):
    server, url = resumable(fail_status=404)
    with pytest.raises(Exception) as exc:  # noqa: PT011
        list_zip_members(f"{url}?X-Amz-Signature=SECRET")
    assert "SECRET" not in str(exc.value) and "SECRET" not in repr(exc.value)
    with pytest.raises(Exception) as exc:  # noqa: PT011
        fetch_zip_members(f"{url}?X-Amz-Signature=SECRET", {"a": Path("/nonexistent/a")})
    assert "SECRET" not in str(exc.value) and "SECRET" not in repr(exc.value)


def _members(n=4, size=500_000):
    return {f"m/{i}.bin": np.random.default_rng(100 + i).integers(0, 256, size, dtype=np.uint8).tobytes()
            for i in range(n)}


def test_extract_tar_resumes_across_processes_skipping_done_members(resumable, tmp_path):
    members = _members()
    raw = _targz(members)
    out = tmp_path / "out"
    staging = tmp_path / "out.staging"
    _, url1 = resumable(raw, cuts=[int(len(raw) * 0.55)], ranges=False)  # morre no meio e sem retomada
    with pytest.raises(ResumeUnsupportedError):
        stream_extract_tar(url1, out, lambda n: n, done=lambda rel: (staging / rel).is_file())
    assert not out.exists() and staging.is_dir()
    kept = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    assert 0 < len(kept) < len(members) and not any(k.endswith(".part") for k in kept)
    first = staging / kept[0]
    first.write_bytes(first.read_bytes())  # marca a identidade: se for reescrito, o mtime_ns muda
    before = first.stat().st_mtime_ns
    asked = []

    def done(rel):
        asked.append(rel)
        return (staging / rel).is_file()

    _, url2 = resumable(raw)
    d, files = stream_extract_tar(url2, out, lambda n: n, done=done, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert files == sorted(members) and d.size == len(raw)
    assert set(asked) == set(members)
    for name, data in members.items():
        assert (out / name).read_bytes() == data
    assert (out / kept[0]).stat().st_mtime_ns == before  # não reescreveu o membro já pronto
    assert not staging.exists()


def test_extract_tar_hash_mismatch_discards_staging_even_with_done(resumable, tmp_path):
    raw = _targz(_members(2))
    _, url = resumable(raw)
    with pytest.raises(IntegrityError):
        stream_extract_tar(url, tmp_path / "out", lambda n: n, expected_sha256="0" * 64, done=lambda rel: False)
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.staging").exists()


def test_extract_tar_stale_staging_without_done_still_refuses(tmp_path):
    (tmp_path / "out.staging").mkdir()
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(_targz({"a": b"x"}))
    with pytest.raises(IntegrityError):
        stream_extract_tar(archive, tmp_path / "out", lambda n: n)


def test_extract_tar_member_never_left_half_written(tmp_path):
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(_targz({"big": b"x" * 100_000}))
    from cctrain.data import sources

    real = sources.shutil.copyfileobj

    def boom(src, dst, *a, **k):
        real(src, dst, 1024)
        raise OSError("disk exploded")

    sources.shutil.copyfileobj = boom
    try:
        with pytest.raises(OSError, match="exploded"):
            stream_extract_tar(archive, tmp_path / "out", lambda n: n, done=lambda rel: False)
    finally:
        sources.shutil.copyfileobj = real
    assert not (tmp_path / "out.staging" / "big").exists()  # só existiria completo e conferido
