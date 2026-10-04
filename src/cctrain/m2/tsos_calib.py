"""Convenção FIXA do TSOS e relatório de plausibilidade no dev (`calibrated_in_dev: tsos_config` do gates-v2).

Fato (M2 T3): a eq. 3 do TSOS (Eskimez et al. 2021, arXiv 2110.09625) não é invariante à escala, e o paper não fixa a escala
nem a STFT. Convenção (fix round 1, decisão da revisão): áudio float em [-1, 1], `amplitude_scale` = 1, p = 0,3, γ = 0,1 do
paper, STFT de 20/10 ms (a do DeepFilterNet), registrada no `tsos-config-v1.yaml`. NÃO há grade nem ajuste de escala: ajustar a
escala para o DFNet3 bater um número do paper seria circular (o TSOS cresce com a escala) e o "alvo" não existe, porque os
baselines do paper são de outro modelo (DCCRN) em outra escala de áudio desconhecida.

O `calibrate` só MEDE o DFNet3 upstream por condição no `dev-select` e compara com a faixa do baseline NÃO personalizado do
paper (DCCRN sem personalização, tabela 1, como citado em R06) NA CONDIÇÃO CERTA: o alvo limpo sozinho é TS3 (1,47% ICR a
2,42% VCTK), não TS1 (a média 7,07% que a versão anterior usava é de TS1). Se o DFNet3 em TS3 cai muito fora da faixa
(fator `PLAUSIBILITY_FACTOR`), a convenção NÃO é congelada: o resultado fica no relatório e a decisão é do dono (gates-v3: só
critério relativo, ou escala externa por dBFS). O critério relativo do G3 (≤ DFNet3 + 0,5 p.p.) não depende da escala.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

from cctrain.eval.gates import verify_frozen
from cctrain.eval.metrics import TsosConfig, tsos
from cctrain.paths import repo_root

TSOS_V1 = repo_root() / "configs" / "eval" / "tsos-config-v1.yaml"
CONVENTION = TsosConfig(p=0.3, gamma=0.1, win_ms=20.0, hop_ms=10.0, amplitude_scale=1.0)

# DCCRN SEM personalização, tabela 1 de Eskimez et al. 2021 (R06 l.212): (ICR, VCTK) em %, por cenário.
# TS1 = alvo + interferente + ruído, TS2 = alvo + ruído, TS3 = alvo limpo sozinho.
PAPER_BASELINE_PCT = {"TS1": (9.01, 5.13), "TS2": (7.32, 5.33), "TS3": (1.47, 2.42)}
# condição do dev -> cenário do paper (C3 não tem contraparte no paper: só é medida)
PAPER_SCENARIO = {"C0c": "TS3", "C0": "TS2", "C4": "TS1"}
PLAUSIBILITY_FACTOR = 2.0  # "muito fora" = abaixo de lo/2 ou acima de 2*hi da faixa do paper
GATING_CONDITION = "C0c"  # só a condição TS3 (alvo limpo) decide o congelamento; as demais são informativas
OWNER_DECISION = "requer decisão do dono (gates-v3: só critério relativo ou escala externa por dBFS)"


def paper_band(condition: str) -> tuple[float, float] | None:
    sc = PAPER_SCENARIO.get(condition)
    if sc is None:
        return None
    lo, hi = sorted(PAPER_BASELINE_PCT[sc])
    return lo, hi


def mean_per_speaker(values: Sequence[float], speakers: Sequence[str]) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for v, s in zip(values, speakers, strict=True):
        if np.isfinite(v):
            groups[s].append(float(v))
    return float(np.mean([np.mean(g) for g in groups.values()])) if groups else float("nan")


def measure_tsos(pairs: Sequence[tuple[np.ndarray, np.ndarray]], speakers: Sequence[str], sr: int,
                 cfg: TsosConfig = CONVENTION) -> float:
    """TSOS (%, média por falante) de `pairs` = (saída alinhada, alvo limpo) na convenção `cfg`."""
    return mean_per_speaker([float(tsos(e, r, sr, cfg)[0]) for e, r in pairs], speakers)


def plausibility(measured: Mapping[str, float], factor: float = PLAUSIBILITY_FACTOR) -> dict:
    """Compara o TSOS medido do DFNet3, por condição, com a faixa do baseline do paper na condição certa.

    Por condição com contraparte no paper: faixa [lo, hi], `status` = `in_band` | `near_band` (dentro do fator) | `far_outside`.
    `freeze` só depende da condição `GATING_CONDITION`: congelar a convenção se ela não é `far_outside`."""
    rows: dict[str, dict] = {}
    for cond, v in measured.items():
        band = paper_band(cond)
        row: dict = {"tsos_pct": float(v), "paper_scenario": PAPER_SCENARIO.get(cond), "band_pct": band}
        if band is None:
            row["status"] = "no_paper_counterpart"
        elif not math.isfinite(v) or v <= 0:
            row["status"] = "far_outside"
        elif band[0] <= v <= band[1]:
            row["status"] = "in_band"
        elif band[0] / factor <= v <= band[1] * factor:
            row["status"] = "near_band"
        else:
            row["status"] = "far_outside"
        rows[cond] = row
    gate = rows.get(GATING_CONDITION)
    if gate is None:
        raise ValueError(f"falta a condição {GATING_CONDITION} para decidir o congelamento")
    freeze = gate["status"] in ("in_band", "near_band")
    return {"factor": factor, "gating_condition": GATING_CONDITION, "conditions": rows, "freeze": freeze,
            "decision": "convenção congelada" if freeze else OWNER_DECISION}


def load_tsos_config(path: Path = TSOS_V1) -> TsosConfig:
    verify_frozen(Path(path))
    d = yaml.safe_load(Path(path).read_text())
    if d.get("version") != 1:
        raise ValueError("tsos-config: version deve ser 1")
    c = d["config"]
    return TsosConfig(p=float(c["p"]), gamma=float(c["gamma"]), win_ms=float(c["win_ms"]), hop_ms=float(c["hop_ms"]),
                      amplitude_scale=float(c["amplitude_scale"]))
