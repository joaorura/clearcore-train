import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cctrain.gpu import gpu_lock


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("CCTRAIN_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def test_lock_file_is_created_under_runs(runs_dir: Path):
    with gpu_lock():
        assert (runs_dir / ".gpu.lock").is_file()


def test_lock_is_released_on_exit_and_on_error(runs_dir: Path):
    with gpu_lock(timeout_s=0.5):
        pass
    with pytest.raises(RuntimeError, match="boom"), gpu_lock(timeout_s=0.5):
        raise RuntimeError("boom")
    with gpu_lock(timeout_s=0.5):
        pass


def test_second_holder_times_out(runs_dir: Path):
    with gpu_lock():
        t0 = time.monotonic()
        with pytest.raises(TimeoutError), gpu_lock(timeout_s=0.3):
            pass
        assert 0.25 <= time.monotonic() - t0 < 5


def test_zero_timeout_is_a_try_lock(runs_dir: Path):
    with gpu_lock(), pytest.raises(TimeoutError), gpu_lock(timeout_s=0):
        pass


_WORKER = textwrap.dedent(
    """
    import sys, time
    from cctrain.gpu import gpu_lock
    log, tag = sys.argv[1], sys.argv[2]
    with gpu_lock():
        with open(log, "a") as f:
            f.write(f"start {tag}\\n"); f.flush()
        time.sleep(0.6)
        with open(log, "a") as f:
            f.write(f"end {tag}\\n"); f.flush()
    """
)


def test_two_processes_serialize(runs_dir: Path, tmp_path: Path):
    log = tmp_path / "log.txt"
    env = {**os.environ, "CCTRAIN_RUNS_DIR": str(runs_dir)}
    procs = [subprocess.Popen([sys.executable, "-c", _WORKER, str(log), t], env=env) for t in "AB"]
    for p in procs:
        assert p.wait(timeout=60) == 0
    lines = log.read_text().split("\n")[:-1]
    assert len(lines) == 4
    # sem intercalação: cada start é seguido do end do mesmo processo
    assert lines[0].split()[0] == "start" and lines[1] == f"end {lines[0].split()[1]}"
    assert lines[2].split()[0] == "start" and lines[3] == f"end {lines[2].split()[1]}"
    assert {lines[0].split()[1], lines[2].split()[1]} == {"A", "B"}
