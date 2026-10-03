"""Caminhos do repositório e configuração externa. Nenhum caminho pessoal no código."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_DATA_DIR = "CCTRAIN_DATA_DIR"
ENV_RUNS_DIR = "CCTRAIN_RUNS_DIR"
ENV_CLEARCORE_ROOT = "CCTRAIN_CLEARCORE_ROOT"
APPROVED_ASSET_RELPATH = "vendor/approved/df-compatible-release-asset-v1.bin"


class ConfigError(RuntimeError):
    """Configuração ausente ou inválida. Sempre falha fechado."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    repo: Path
    data: Path
    runs: Path
    delivery: Path
    build: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Paths:
        env = os.environ if env is None else env
        repo = repo_root()
        data = Path(env.get(ENV_DATA_DIR) or repo / "data").expanduser().resolve()
        runs = Path(env.get(ENV_RUNS_DIR) or repo / "runs").expanduser().resolve()
        return cls(repo=repo, data=data, runs=runs, delivery=repo / "delivery", build=repo / "build")


def clearcore_root(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    raw = env.get(ENV_CLEARCORE_ROOT)
    if not raw:
        raise ConfigError(
            f"{ENV_CLEARCORE_ROOT} não definido: aponte para um checkout do Clearcore com o commit fixado"
        )
    root = Path(raw).expanduser().resolve()
    if not (root / APPROVED_ASSET_RELPATH).is_file():
        raise ConfigError(f"{root} não contém {APPROVED_ASSET_RELPATH}")
    return root
