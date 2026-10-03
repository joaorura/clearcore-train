"""Gerador FiLM (spec §3.2): identidade exata por construção, faixa limitada e configurável."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from cctrain.paths import repo_root

FILM_CONFIG_V1 = repo_root() / "configs" / "pdfnet3" / "film-v1.yaml"
_GAMMA_BOUND_MAX = math.log(100.0)  # [e^-Γ, e^Γ] ⊂ [0,001; 100] exige Γ ≤ ln 100
_BETA_MIN, _BETA_MAX = 4.0, 50.0


class FilmConfigError(ValueError):
    """Configuração do FiLM fora dos limites do runtime."""


@dataclass(frozen=True)
class FilmConfig:
    log_gamma_bound: float
    beta_bound: float | None
    hidden_dim: int = 256
    embedding_dim: int = 192
    mid_dim: int = 256

    def validate(self) -> FilmConfig:
        if not 0.0 < self.log_gamma_bound <= _GAMMA_BOUND_MAX:
            raise FilmConfigError(f"Γ={self.log_gamma_bound} fora de (0, ln 100]")
        if self.beta_bound is None:
            raise FilmConfigError("B ainda não fixado (regra do M0, Task 17)")
        if not _BETA_MIN <= self.beta_bound <= _BETA_MAX:
            raise FilmConfigError(f"B={self.beta_bound} fora de [{_BETA_MIN}, {_BETA_MAX}]")
        if self.hidden_dim != 256:
            raise FilmConfigError("film_hidden do runtime é 256")
        return self


def beta_bound_rule(p99: float) -> float:
    return min(_BETA_MAX, max(_BETA_MIN, 2.0 * float(p99)))


def load_film_config(path: Path = FILM_CONFIG_V1) -> FilmConfig:
    d = yaml.safe_load(Path(path).read_text())
    cfg = FilmConfig(
        log_gamma_bound=float(d["log_gamma_bound"]),
        beta_bound=None if d["beta_bound"] is None else float(d["beta_bound"]),
        hidden_dim=int(d["hidden_dim"]), embedding_dim=int(d["embedding_dim"]), mid_dim=int(d["mid_dim"]),
    )
    return cfg.validate()


@dataclass(frozen=True)
class FilmVectors:
    gamma_enc: Tensor
    beta_enc: Tensor
    gamma_df: Tensor
    beta_df: Tensor


class FilmGenerator(nn.Module):
    def __init__(self, cfg: FilmConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.hidden = nn.Linear(cfg.embedding_dim, cfg.mid_dim)
        self.out = nn.Linear(cfg.mid_dim, 4 * cfg.hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, embedding: Tensor, mask: Tensor) -> FilmVectors:
        d = self.out(torch.relu(self.hidden(embedding))) * mask
        dge, dbe, dgd, dbd = d.chunk(4, dim=-1)
        g_bound, b_bound = self.cfg.log_gamma_bound, self.cfg.beta_bound

        def gamma(x: Tensor) -> Tensor:
            return torch.exp(g_bound * torch.tanh(x)).unsqueeze(1)

        def beta(x: Tensor) -> Tensor:
            return (b_bound * torch.tanh(x)).unsqueeze(1)

        return FilmVectors(gamma(dge), beta(dbe), gamma(dgd), beta(dbd))
