from collections import namedtuple
from pathlib import Path

import pytest

from cctrain.disk import GB, DiskError, assert_persistent, filesystem_type, require_free, reserve_bytes

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
    usage = lambda _p: Usage(100 * GB, 95 * GB, 5 * GB)  # noqa: E731
    with pytest.raises(DiskError, match="espaço insuficiente"):
        require_free(Path("/home/user/data"), 1 * GB, reserve=10 * GB, mounts_file=m, usage=usage)


def test_require_free_passes_with_room(tmp_path: Path):
    m = _mounts(tmp_path)
    usage = lambda _p: Usage(100 * GB, 50 * GB, 50 * GB)  # noqa: E731
    assert require_free(Path("/home/user/data"), 1 * GB, reserve=10 * GB, mounts_file=m, usage=usage) == 50 * GB


def test_real_repo_is_persistent(paths):
    assert_persistent(paths.repo)


def test_reserve_is_decimal_gb_like_the_budget():
    # budget-v1: disk.reserve_gb = 10.0 (GB decimais, como o relatório M0); 1 GB = 1e9 B, não 2**30
    from cctrain.eval.gates import load_budget

    assert GB == 10**9
    assert reserve_bytes({}) == 10_000_000_000 == int(load_budget().reserve_gb * 1e9)
    assert reserve_bytes({"CCTRAIN_MIN_FREE_GB": "2.5"}) == 2_500_000_000


def test_require_free_boundary_uses_decimal_gb(tmp_path: Path):
    m = _mounts(tmp_path)
    usage = lambda _p: Usage(100 * GB, 11 * GB, 11 * GB)  # noqa: E731
    path = Path("/home/user/data")
    # livre 11 GB, reserva padrão 10 GB: pedir exatamente 1 GB passa, 1 GB + 1 B falha
    assert require_free(path, 1 * GB, mounts_file=m, usage=usage) == 11 * GB
    with pytest.raises(DiskError, match="11.00 GB"):
        require_free(path, 1 * GB + 1, mounts_file=m, usage=usage)
