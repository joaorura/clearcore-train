"""Re-projeção de disco do M1 e fases de download (plano M1, Tarefa 7).

Lê `configs/data/m1-phases-v1.yaml`. Cada fase acumula o que fica no disco (`gb_final`) e tem um pico transitório
(o maior bruto que precisa coexistir com o extraído; tar/zip lidos em stream têm pico ~ 0). A margem acumulada de uma
fase é `livre - reserva - baseline - Σ gb_final (fases até ela)`; a margem de pico troca o final da própria fase pelo
pico. Fase com margem acumulada ou de pico < 0 fica `blocked_disk`; como o acumulado inclui as fases anteriores
(bloqueadas ou não), uma fase bloqueada bloqueia as seguintes. Nada é baixado nem apagado aqui.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from cctrain.disk import GB
from cctrain.paths import Paths, repo_root

PHASES_V1 = repo_root() / "configs" / "data" / "m1-phases-v1.yaml"
_SOURCE_KEYS = {"id", "hours", "sr", "factor_gb_per_hour", "gb_final", "transient_gb", "basis"}
_TOP_KEYS = {"version", "spec", "baseline", "phases", "excluded_by_owner", "out_of_scope"}


class PlanError(ValueError):
    """Configuração das fases ou entrada inválida."""


@dataclass(frozen=True)
class Phase:
    id: str
    sources: tuple[str, ...]
    gb_final: float
    gb_peak: float
    cumulative_margin_gb: float
    peak_margin_gb: float = math.nan
    status: str = "ok"  # "ok" | "blocked_disk"


def _source_gb(src: Mapping) -> float:
    unknown = set(src) - _SOURCE_KEYS
    if unknown:
        raise PlanError(f"fonte {src.get('id')!r}: chaves desconhecidas {sorted(unknown)}")
    if "gb_final" in src:
        if {"hours", "factor_gb_per_hour"} & set(src):
            raise PlanError(f"fonte {src['id']!r}: gb_final e hours/factor_gb_per_hour são exclusivos")
        gb = float(src["gb_final"])
    else:
        missing = {"hours", "sr", "factor_gb_per_hour"} - set(src)
        if missing:
            raise PlanError(f"fonte {src.get('id')!r}: faltam {sorted(missing)} (ou gb_final)")
        gb = float(src["hours"]) * float(src["factor_gb_per_hour"])
    transient = float(src.get("transient_gb", 0.0))
    if not (math.isfinite(gb) and gb >= 0 and math.isfinite(transient) and transient >= 0):
        raise PlanError(f"fonte {src.get('id')!r}: tamanho inválido")
    return gb


def load_phases_config(path: Path = PHASES_V1) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or cfg.get("version") != 1:
        raise PlanError("m1-phases: version deve ser 1")
    unknown = set(cfg) - _TOP_KEYS
    if unknown:
        raise PlanError(f"m1-phases: chaves desconhecidas {sorted(unknown)}")
    ids = [p["id"] for p in cfg.get("phases") or []]
    if not ids or len(set(ids)) != len(ids):
        raise PlanError("m1-phases: fases ausentes ou com id repetido")
    for p in cfg["phases"]:
        if not p.get("sources"):
            raise PlanError(f"fase {p['id']}: sem fontes")
        for s in p["sources"]:
            _source_gb(s)  # valida
    return cfg


def _baseline_gb(cfg: Mapping) -> float:
    return sum(float(b["gb"]) for b in cfg.get("baseline") or [])


def _check_inputs(free_bytes: float, reserve_gb: float) -> None:
    for name, value in (("free_bytes", free_bytes), ("reserve_gb", reserve_gb)):
        if not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise PlanError(f"{name} deve ser um número finito >= 0, recebido {value!r}")


def plan_phases(free_bytes: int, reserve_gb: float, *, config: Path = PHASES_V1) -> list[Phase]:
    _check_inputs(free_bytes, reserve_gb)
    cfg = load_phases_config(config)
    available = free_bytes / GB - reserve_gb - _baseline_gb(cfg)
    cumulative = 0.0
    phases: list[Phase] = []
    for p in cfg["phases"]:
        finals = [_source_gb(s) for s in p["sources"]]
        gb_final = sum(finals)
        transient = max(float(s.get("transient_gb", 0.0)) for s in p["sources"])
        gb_peak = gb_final + transient
        peak_margin = available - cumulative - gb_peak
        cumulative += gb_final
        margin = available - cumulative
        status = "ok" if margin >= 0 and peak_margin >= 0 else "blocked_disk"
        phases.append(Phase(p["id"], tuple(s["id"] for s in p["sources"]), gb_final, gb_peak, margin, peak_margin,
                            status))
    return phases


def disk_plan(free_bytes: int, reserve_gb: float, *, config: Path = PHASES_V1) -> dict:
    cfg = load_phases_config(config)
    phases = plan_phases(free_bytes, reserve_gb, config=config)
    baseline = _baseline_gb(cfg)
    final_margin = phases[-1].cumulative_margin_gb
    excluded = [{"id": e["id"], "gb_final": float(e["gb_final"]), "basis": e.get("basis", "")}
                for e in cfg.get("excluded_by_owner") or []]
    return {
        "version": 1, "config": str(Path(config).name),
        "free_gb": free_bytes / GB, "reserve_gb": float(reserve_gb),
        "baseline": [{"id": b["id"], "gb": float(b["gb"]), "basis": b.get("basis", "")} for b in cfg["baseline"]],
        "baseline_gb": baseline,
        "phases": [asdict(p) for p in phases],
        "total_gb": baseline + sum(p.gb_final for p in phases),
        "final_margin_gb": final_margin,
        "min_peak_margin_gb": min(p.peak_margin_gb for p in phases),
        "blocked": [p.id for p in phases if p.status == "blocked_disk"],
        "excluded_by_owner": excluded,
        "what_if_with_excluded_margin_gb": final_margin - sum(e["gb_final"] for e in excluded),
        "out_of_scope": cfg.get("out_of_scope") or [],
    }


def write_disk_plan(free_bytes: int, reserve_gb: float, *, config: Path = PHASES_V1,
                    runs_dir: Path | None = None) -> Path:
    plan = disk_plan(free_bytes, reserve_gb, config=config)
    out = Path(runs_dir or Paths.from_env().runs) / "m1" / "disk-plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Projeta o disco do M1 e grava runs/m1/disk-plan.json")
    ap.add_argument("--free-bytes", type=int, help="padrão: espaço livre atual do diretório de dados")
    ap.add_argument("--reserve-gb", type=float, help="padrão: disk.reserve_gb do budget-v1")
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    free = args.free_bytes if args.free_bytes is not None else shutil.disk_usage(paths.data).free
    if args.reserve_gb is not None:
        reserve = args.reserve_gb
    else:
        from cctrain.eval.gates import load_budget

        reserve = load_budget().reserve_gb
    out = write_disk_plan(free, reserve)
    plan = json.loads(out.read_text())
    for p in plan["phases"]:
        print(f"{p['id']:34s} final {p['gb_final']:7.2f}  pico {p['gb_peak']:7.2f}  "
              f"margem {p['cumulative_margin_gb']:7.2f}  {p['status']}")
    print(f"livre {plan['free_gb']:.2f} GB, reserva {plan['reserve_gb']:.1f}, baseline {plan['baseline_gb']:.1f}; "
          f"margem final {plan['final_margin_gb']:.2f} GB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
