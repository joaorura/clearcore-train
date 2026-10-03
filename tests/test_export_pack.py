import gzip
import io
import tarfile
from pathlib import Path

import pytest

from cctrain.export.pack import pack_asset, read_members


def _files(tmp_path: Path) -> list[tuple[str, Path]]:
    out = []
    for name, data in (("enc.onnx", b"E" * 10), ("erb_dec.onnx", b"R"), ("df_dec.onnx", b"D" * 3),
                       ("config.ini", b"[df]\n")):
        p = tmp_path / ("src-" + name)
        p.write_bytes(data)
        out.append((name, p))
    return out


def test_pack_is_reproducible_and_path_independent(tmp_path: Path):
    files = _files(tmp_path)
    a = pack_asset(tmp_path / "a.tar.gz", files)
    (tmp_path / "sub").mkdir()
    b = pack_asset(tmp_path / "sub" / "other-name.tar.gz", files)
    assert a.sha256 == b.sha256
    assert a.size == b.size


def test_pack_layout_follows_spec(tmp_path: Path):
    res = pack_asset(tmp_path / "x.tar.gz", _files(tmp_path))
    raw = res.path.read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert raw[3] & 0x08 == 0, "gzip sem FNAME"
    assert raw[4:8] == b"\x00\x00\x00\x00", "gzip mtime=0"
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == ["enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"]
    for m in members:
        assert m.isreg() and m.mode == 0o644 and m.mtime == 0
        assert (m.uid, m.gid, m.uname, m.gname) == (0, 0, "", "")


def test_pack_refuses_nested_or_duplicate_names(tmp_path: Path):
    files = _files(tmp_path)
    with pytest.raises(ValueError):
        pack_asset(tmp_path / "x.tar.gz", [("tmp/export/enc.onnx", files[0][1])])
    with pytest.raises(ValueError):
        pack_asset(tmp_path / "x.tar.gz", [files[0], files[0]])


def test_read_members_roundtrip_and_prefix(tmp_path: Path):
    res = pack_asset(tmp_path / "x.tar.gz", _files(tmp_path))
    assert read_members(res.path)["enc.onnx"] == b"E" * 10
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("tmp/export/config.ini")
        info.size, info.mode = 3, 0o755
        tar.addfile(info, io.BytesIO(b"abc"))
    bad = tmp_path / "exec.tar.gz"
    bad.write_bytes(gzip.compress(buf.getvalue()))
    with pytest.raises(ValueError, match="executável"):
        read_members(bad)
