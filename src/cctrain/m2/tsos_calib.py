"""Calibração do `tsos_config` no dev (`calibrated_in_dev` do gates-v2; os níveis do G3 não mudam).

Fato (M2 T3): a eq. 3 do TSOS (Eskimez et al. 2021, arXiv 2110.09625) não é invariante à escala, e o paper não fixa a
escala nem a STFT. Só `amplitude_scale` fica livre aqui (p = 0,3, γ = 0,1, STFT 20/10 ms são os do paper/DeepFilterNet).

Regra (escolha do implementador, a confirmar pelo orquestrador): ancorar a escala no modelo NÃO personalizado do próprio
paper. Os níveis 1/2/3% do G3 vieram de comparar modelos personalizados com o baseline sem personalização do paper (DCCRN
sem personalização, TS3 = alvo limpo sozinho: 9,01% no ICR e 5,13% no VCTK, tabela 1). Escolhe-se, numa grade logarítmica
2^(k/2), a escala em que o DFNet3 upstream (também um modelo não personalizado), em C0c do `dev-select`, mais se aproxima
(em escala log) da média desses dois valores. Só entram o `dev-select` e o DFNet3 upstream: nem o candidato F nem o
`dev-gate` participam. O critério relativo do G3 (≤ DFNet3 + 0,5 p.p.) não depende desta âncora.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from cctrain.eval.gates import verify_frozen
from cctrain.eval.metrics import TsosConfig, tsos
from cctrain.paths import repo_root

TSOS_V1 = repo_root() / "configs" / "eval" / "tsos-config-v1.yaml"
PAPER_BASELINE_TS3_PCT = {"ICR": 9.01, "VCTK": 5.13}  # DCCRN sem personalização, TS3, tabela 1 do paper
ANCHOR_PCT = float(np.mean(list(PAPER_BASELINE_TS3_PCT.values())))
GRID_K = range(-24, 33)  # escalas 2^(k/2): de 2^-12 a 2^16 (32768 = int16 está na grade como k = 30)


def scale_grid() -> list[float]:
    return [float(2.0 ** (k / 2.0)) for k in GRID_K]


def mean_per_speaker(values: Sequence[float], speakers: Sequence[str]) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for v, s in zip(values, speakers, strict=True):
        if np.isfinite(v):
            groups[s].append(float(v))
    return float(np.mean([np.mean(g) for g in groups.values()])) if groups else float("nan")


def calibrate_amplitude_scale(pairs: Sequence[tuple[np.ndarray, np.ndarray]], speakers: Sequence[str], sr: int, *,
                              anchor_pct: float = ANCHOR_PCT, grid: Sequence[float] | None = None,
                              base: TsosConfig | None = None) -> dict:
    """`pairs` = (saída do DFNet3 alinhada, alvo limpo). Devolve a tabela escala -> TSOS (%, média por falante) e a escolha."""
    base = base or TsosConfig()
    table = []
    for a in grid or scale_grid():
        cfg = dataclasses.replace(base, amplitude_scale=a)
        per_clip = [float(tsos(e, r, sr, cfg)[0]) for e, r in pairs]
        table.append({"amplitude_scale": a, "tsos_pct": mean_per_speaker(per_clip, speakers)})
    usable = [r for r in table if r["tsos_pct"] > 0 and math.isfinite(r["tsos_pct"])]
    if not usable:
        raise ValueError("nenhuma escala da grade dá TSOS > 0 no DFNet3")
    best = min(usable, key=lambda r: abs(math.log(r["tsos_pct"] / anchor_pct)))
    return {"anchor_pct": anchor_pct, "chosen_amplitude_scale": best["amplitude_scale"],
            "chosen_tsos_pct": best["tsos_pct"], "table": table}


def load_tsos_config(path: Path = TSOS_V1) -> TsosConfig:
    verify_frozen(Path(path))
    d = yaml.safe_load(Path(path).read_text())
    if d.get("version") != 1:
        raise ValueError("tsos-config: version deve ser 1")
    c = d["config"]
    return TsosConfig(p=float(c["p"]), gamma=float(c["gamma"]), win_ms=float(c["win_ms"]), hop_ms=float(c["hop_ms"]),
                      amplitude_scale=float(c["amplitude_scale"]))
