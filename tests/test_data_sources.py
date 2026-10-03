import functools
import gzip
import hashlib
import http.server
import io
import subprocess
import tarfile
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cctrain.data.sources import (
    IntegrityError,
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
