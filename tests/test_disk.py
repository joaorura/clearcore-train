from collections import namedtuple
from pathlib import Path

import pytest

from cctrain.disk import GIB, DiskError, assert_persistent, filesystem_type, require_free

Usage = namedtuple("Usage", "total used free")


def _mounts(tmp_path: Path, extra: str = "") -> Path:
    f = tmp_path / "mounts"
    f.write_text(
        "/dev/root / btrfs rw 0 0\n"
        "tmpfs /tmp tmpfs rw 0 0\n"
        "/dev/sdb /media/with\\040space ext4 rw 0 0\n" + extra
    )
    return f


def test_longest_mount_prefix_wins(tmp_path: Path):
    m = _mounts(tmp_path)
    assert filesystem_type(Path("/tmp/x/y"), m) == "tmpfs"
    assert filesystem_type(Path("/home/user"), m) == "btrfs"
    assert filesystem_type(Path("/media/with space/a"), m) == "ext4"


def test_tmpfs_is_refused(tmp_path: Path):
    with pytest.raises(DiskError, match="volátil"):
        assert_persistent(Path("/tmp/data"), _mounts(tmp_path))


def test_require_free_fails_closed(tmp_path: Path):
    m = _mounts(tmp_path)
    usage = lambda _p: Usage(100 * GIB, 95 * GIB, 5 * GIB)  # noqa: E731
    with pytest.raises(DiskError, match="espaço insuficiente"):
        require_free(Path("/home/user/data"), 1 * GIB, reserve=10 * GIB, mounts_file=m, usage=usage)


def test_require_free_passes_with_room(tmp_path: Path):
    m = _mounts(tmp_path)
    usage = lambda _p: Usage(100 * GIB, 50 * GIB, 50 * GIB)  # noqa: E731
    assert require_free(Path("/home/user/data"), 1 * GIB, reserve=10 * GIB, mounts_file=m, usage=usage) == 50 * GIB


def test_real_repo_is_persistent(paths):
    assert_persistent(paths.repo)
