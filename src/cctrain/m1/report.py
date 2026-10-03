"""Veredito go/no-go do M1 (Tarefa 12), estritamente pelos níveis congelados.

Níveis: bloco `m1` de `gates-v2.yaml` (EER por lista, paridade tract, gate funcional), `budget-v1.yaml` (RAM do loader,
VRAM do batch 16, reserva de disco) e `runtime-deviation-v1.yaml` (não bloqueante: só escolhe o caminho de medição).
Nada aqui define nível: tudo é lido dos arquivos conferidos por SHA-256. Medida ausente, NaN ou fora do cenário
congelado reprova (falha fechado).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from cctrain.eval.functional import load_deviation_levels
from cctrain.eval.gates import GATES_V2, load_budget, load_gates


@dataclass(frozen=True)
class M1Inputs:
    eer: Mapping[str, float]
    tract: Mapping  # {durations_s, max_abs_diff, finite}
    functional_c3: Mapping  # {ci_low, delta_si_sdr_db, ci_high, n_speakers, same_sex, sir_db, stage}
    base_sha256: tuple[str, str]  # (antes, depois) do estágio C
    probe: Mapping  # {pss_gb_max, vram_peak_reserved_gb}
    disk_free_gb: float
    deviation: Mapping  # {error_to_signal_db, stage_agreement, path}
    ab1: Mapping | None = None


@dataclass(frozen=True)
class Check:
    name: str
    value: float | str | None
    level: str
    passed: bool
    blocking: bool = True
    note: str = ""


@dataclass(frozen=True)
class Verdict:
    go: bool
    checks: tuple[Check, ...]
    gates_sha256: str
    budget_sha256: str
    deviation_sha256: str
    blocking_failures: tuple[str, ...] = field(default=())


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _le(name: str, value, limit: float, note: str = "") -> Check:
    v = _num(value)
    return Check(name, v, f"<= {limit}", v is not None and v <= limit, note=note if v is not None else "ausente")


def m1_verdict(inp: M1Inputs) -> Verdict:
    gates = load_gates(GATES_V2)
    m1 = gates.m1
    budget = load_budget()
    dev_levels = load_deviation_levels()
    checks: list[Check] = []
    for name, limit in m1["eer_max"].items():
        checks.append(_le(f"eer_{name}", inp.eer.get(name), float(limit)))
    tp = m1["tract_parity"]
    want = [int(d) for d in tp["durations_s"]]
    got = sorted(int(d) for d in inp.tract.get("durations_s", []))
    diff = _num(inp.tract.get("max_abs_diff"))
    ok = diff is not None and diff <= float(tp["abs_tol"]) and got == sorted(want) and bool(inp.tract.get("finite"))
    checks.append(Check("tract_parity", diff, f"<= {float(tp['abs_tol'])}", ok,
                        note=f"durações {got} (exigidas {sorted(want)}); finito={inp.tract.get('finite')}"))
    fn = m1["functional"]
    lo = _num(inp.functional_c3.get("ci_low"))
    scenario = (inp.functional_c3.get("stage") == fn["stage"] and bool(inp.functional_c3.get("same_sex")) is bool(
        fn["same_sex"]) and _num(inp.functional_c3.get("sir_db")) == float(fn["sir_db"]))
    level = float(fn["delta_si_sdr_ci_lower_gt"])
    checks.append(Check("functional_c3_ci_lower", lo, f"> {level}", lo is not None and lo > level and scenario,
                        note="" if scenario else "cenário diferente do congelado (estágio/mesmo sexo/SIR)"))
    before, after = inp.base_sha256
    checks.append(Check("base_frozen", after[:12], "SHA-256 da base igual antes e depois",
                        bool(before) and before == after, note="spec §5.3"))
    checks.append(_le("budget_ram_pss_gb", inp.probe.get("pss_gb_max"), float(budget.loader_rss_gb_max)))
    checks.append(_le("budget_vram_reserved_gb", inp.probe.get("vram_peak_reserved_gb"),
                      float(budget.vram_b16_reserved_gb_max)))
    free = _num(inp.disk_free_gb)
    checks.append(Check("budget_disk_free_gb", free, f">= {float(budget.reserve_gb)}",
                        free is not None and free >= float(budget.reserve_gb)))
    e2s, agree = _num(inp.deviation.get("error_to_signal_db")), _num(inp.deviation.get("stage_agreement"))
    dev_ok = (e2s is not None and agree is not None and e2s <= dev_levels.error_to_signal_db_max
              and agree >= dev_levels.stage_agreement_min)
    checks.append(Check("deviation_gate", e2s, f"erro/sinal <= {dev_levels.error_to_signal_db_max} dB e concordância "
                        f">= {dev_levels.stage_agreement_min}", dev_ok, blocking=False,
                        note=f"caminho do dev: {inp.deviation.get('path')} (não bloqueia, só escolhe o caminho)"))
    failures = tuple(c.name for c in checks if c.blocking and not c.passed)
    return Verdict(not failures, tuple(checks), gates.sha256, budget.sha256, dev_levels.sha256, failures)


def render_checks_table(v: Verdict) -> str:
    rows = ["| Verificação | Medido | Nível congelado | Resultado | Bloqueia | Nota |", "|---|---:|---|---|---|---|"]
    for c in v.checks:
        val = f"{c.value:.6g}" if isinstance(c.value, float) else str(c.value)
        res = "passa" if c.passed else "falha"
        rows.append(f"| `{c.name}` | {val} | {c.level} | {res} | {'sim' if c.blocking else 'não bloqueia'} | {c.note} |")
    return "\n".join(rows)
