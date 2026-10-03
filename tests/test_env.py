import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from cctrain.paths import repo_root


def test_numpy_below_2():
    assert int(np.__version__.split(".")[0]) < 2


def test_deepfilterlib_version():
    from importlib.metadata import version

    assert version("deepfilterlib") == "0.5.6"


@pytest.mark.gpu
def test_cuda_blackwell():
    assert torch.cuda.is_available(), "CUDA indisponível"
    assert tuple(int(x) for x in torch.version.cuda.split(".")[:2]) >= (12, 8)
    assert torch.cuda.get_device_capability() == (12, 0)


def _fresh_worktree(tmp_path: Path) -> Path:
    wt = tmp_path / "wt"
    (wt / "scripts").mkdir(parents=True)
    shutil.copy(repo_root() / "scripts" / "env.sh", wt / "scripts" / "env.sh")
    return wt


def test_env_sh_creates_build_and_appends_pythonpath(tmp_path: Path):
    wt = _fresh_worktree(tmp_path)
    assert not (wt / "build").exists()
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": "/pre/existing",
           "CCTRAIN_CLEARCORE_ROOT": "/x"}
    out = subprocess.run(
        ["bash", "-c", f'source "{wt}/scripts/env.sh" && echo "$PYTHONPATH" && echo "$CCTRAIN_WT"'],
        env=env, capture_output=True, text=True, check=True, cwd=tmp_path,
    ).stdout.split("\n")
    assert (wt / "build").is_dir()
    assert out[0] == f"{wt}/src:/pre/existing"
    assert out[1] == str(wt)


def test_env_sh_respects_exported_worktree(tmp_path: Path):
    wt = _fresh_worktree(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "CCTRAIN_WT": str(other), "CCTRAIN_CLEARCORE_ROOT": "/x"}
    subprocess.run(["bash", "-c", f'source "{wt}/scripts/env.sh"'], env=env, check=True, cwd=tmp_path)
    assert (other / "build").is_dir()
    assert not (wt / "build").exists()


def test_tract_check_sh_exports_toolchain_regardless_of_cwd(tmp_path: Path):
    cargo_home = tmp_path / "cargo"
    (cargo_home / "bin").mkdir(parents=True)
    stub = cargo_home / "bin" / "cargo"
    stub.write_text('#!/bin/sh\necho "TOOLCHAIN=$RUSTUP_TOOLCHAIN"\n')
    stub.chmod(0o755)
    root = tmp_path / "root"
    root.mkdir()
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "CARGO_HOME": str(cargo_home), "CCTRAIN_ROOT": str(root),
           "RUSTUP_TOOLCHAIN": "stable"}
    out = subprocess.run(
        ["bash", str(repo_root() / "scripts" / "tract-check.sh"), "build"],
        env=env, capture_output=True, text=True, cwd=cwd, check=True,
    ).stdout
    assert "TOOLCHAIN=1.90.0" in out


def test_gitattributes_marks_resampler_binary():
    out = subprocess.run(
        ["git", "check-attr", "binary", "--", "resampler/v1/coeffs.f32", "resampler/v1/golden/noise_1s.in.f32"],
        cwd=repo_root(), capture_output=True, text=True, check=True,
    ).stdout
    assert out.count("binary: set") == 2
