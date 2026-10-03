"""Invoca o binário tools/tract-check (Rust, tract 0.19.16)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from cctrain.paths import ConfigError, repo_root

ENV_BIN = "CCTRAIN_TRACT_CHECK_BIN"
UPSTREAM_GOLDEN_SHA256 = "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa"
APPROVED_ASSET_SHA256 = "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616"
CLEARCORE_FILM_IDENTITY_SHA256 = "8783baff55cad837953d5420c0f96edfbe40c01e5e2ac715cb4f3fef2b6cb45d"


def binary(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    raw = env.get(ENV_BIN)
    path = Path(raw) if raw else repo_root() / "tools/tract-check/target/release/tract-check"
    if not path.is_file():
        raise ConfigError(f"tract-check não compilado em {path}: rode scripts/tract-check.sh build")
    return path


def run(*args: object, timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary()), *map(str, args)], capture_output=True, text=True, timeout=timeout, check=False
    )
