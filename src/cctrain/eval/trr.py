"""G4: rejeição de locutor trocado (TRR a 15 dB) e separação de modos (M2 Tarefa 3).

Cenário C7: só o falante B fala (mistura de entrada `mixture`), o perfil é o do falante A, `atten_lim_db` desligado.
TRR(limiar) = fração dos quadros ATIVOS da entrada cuja saída fica ao menos `limiar` dB abaixo (energia por quadro).
Definição proposta em `docs/superpowers/research/.../06-avaliacao.md` §1.4 (o spec §7 não define TRR); varrer
6/10/15/20 dB no relatório (`trr_curve`). Quadros ativos: energia de entrada até `vad_db_below_peak` dB abaixo do quadro
mais forte e acima do piso absoluto `abs_floor_dbfs` (parâmetro de medida calibrado no dev: `vad_threshold` do gates-v2).
Separação de modos = atenuação mediana de B com o perfil de A menos a atenuação mediana sem perfil (nível: >= 15 dB).
`est` já vem alinhado à entrada (o atraso de 1.440 amostras é compensado por quem chama).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cctrain.eval.metrics import _frames, _pairs, _win_hop

_EPS = 1e-12


@dataclass(frozen=True)
class TrrConfig:
    win_ms: float = 20.0
    hop_ms: float = 10.0
    vad_db_below_peak: float = 30.0
    abs_floor_dbfs: float = -70.0


def frame_energy_db(x: np.ndarray, sr: int, cfg: TrrConfig) -> np.ndarray:
    n, hop = _win_hop(sr, cfg.win_ms, cfg.hop_ms)
    return 10.0 * np.log10((_frames(np.asarray(x, np.float64), n, hop) ** 2).mean(axis=1) + _EPS)


def active_frames(mixture: np.ndarray, sr: int, cfg: TrrConfig | None = None) -> np.ndarray:
    cfg = cfg or TrrConfig()
    e = frame_energy_db(mixture, sr, cfg)
    if e.size == 0:
        return np.zeros(0, bool)
    return (e >= e.max() - cfg.vad_db_below_peak) & (e > cfg.abs_floor_dbfs)


def _attenuation_active(est: np.ndarray, mixture: np.ndarray, sr: int, cfg: TrrConfig) -> np.ndarray:
    """Atenuação por quadro ativo, em dB (positivo = a saída ficou abaixo da entrada)."""
    ein, eout = frame_energy_db(mixture, sr, cfg), frame_energy_db(est, sr, cfg)
    act = active_frames(mixture, sr, cfg)
    return (ein - eout)[act]


def trr(est, mixture, sr: int, *, atten_db: float = 15.0, cfg: TrrConfig | None = None) -> np.ndarray:
    """TRR por enunciado (fração em [0, 1]; NaN se a entrada não tem quadro ativo)."""
    cfg = cfg or TrrConfig()
    out = []
    for a, b in _pairs(est, mixture):
        att = _attenuation_active(a, b, sr, cfg)
        out.append(float((att >= atten_db).mean()) if att.size else np.nan)
    return np.array(out, np.float64)


def trr_curve(est, mixture, sr: int, *, thresholds_db: Sequence[float] = (6, 10, 15, 20),
              cfg: TrrConfig | None = None) -> dict[float, np.ndarray]:
    return {t: trr(est, mixture, sr, atten_db=float(t), cfg=cfg) for t in thresholds_db}


def median_attenuation_db(est, mixture, sr: int, *, cfg: TrrConfig | None = None) -> np.ndarray:
    cfg = cfg or TrrConfig()
    out = []
    for a, b in _pairs(est, mixture):
        att = _attenuation_active(a, b, sr, cfg)
        out.append(float(np.median(att)) if att.size else np.nan)
    return np.array(out, np.float64)


def mode_separation_db(est_profile, est_noprofile, mixture, sr: int, *, cfg: TrrConfig | None = None) -> np.ndarray:
    """Atenuação mediana de B com o perfil de A menos a atenuação mediana no modo sem perfil, por enunciado."""
    return (median_attenuation_db(est_profile, mixture, sr, cfg=cfg)
            - median_attenuation_db(est_noprofile, mixture, sr, cfg=cfg))
