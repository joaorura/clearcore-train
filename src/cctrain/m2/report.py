"""Veredito go/no-go do M2 (Tarefa 8), estritamente pelos níveis e regras congeladas.

Níveis: bloco `gates` (G1, G2, G3, G4) de `configs/eval/gates-v2.yaml`, decisões do dono pré-registradas em
`configs/eval/m2-selection-v1.yaml` (leitura do G1 pelo limite do IC 95% do lado da reprovação, regra relativa
do G3 com TSOS(pDFNet3) <= TSOS(DFNet3) + 0,5 p.p. segundo Opção A), e `runtime-deviation-v1.yaml` (não-bloqueante:
só escolhe o caminho de medição 'runtime' vs 'replica').

Falha fechado: medida ausente, NaN ou fora do cenário congelado reprova.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cctrain.eval.functional import load_deviation_levels
from cctrain.eval.gates import GATES_V2, load_gates
from cctrain.m2.config import SELECTION_V1, load_selection


@dataclass(frozen=True)
class Check:
    name: str
    value: float | str | None
    level: str
    passed: bool
    blocking: bool = True
    note: str = ""


@dataclass(frozen=True)
class M2Inputs:
    runtime_deviation: Mapping  # {error_to_signal_db, stage_agreement, path}
    g1: Mapping  # {delta_stoi: {ci_low, point}, delta_si_sdr_db: {ci_low, point}, ...}
    g2: Mapping  # {delta_si_sdr_db: {point, ci_low}, delta_dnsmos_p_ovrl: float, wer_ratio: float, wer_ok: bool}
    g3: Mapping  # {delta_tsos_over_dfnet3_pp: float, tsos_c0_pct: float}
    g4: Mapping  # {trr_diff_sex: float, trr_same_sex: float, mode_sep_db: float}
    g5: Mapping | None = None  # {spk_sim_delta_ci_lower: float} (informativo)
    g8: Mapping | None = None  # {dnsmos_p_ovrl: float} (informativo)
    c8: Mapping | None = None  # {tanh_saturation_pct: float, gating_fractions: Mapping} (informativo)


@dataclass(frozen=True)
class Verdict:
    go: bool
    checks: tuple[Check, ...]
    gates_sha256: str
    selection_sha256: str
    deviation_sha256: str
    blocking_failures: tuple[str, ...] = field(default=())


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _ge(name: str, value, limit: float, *, blocking: bool = True, note: str = "") -> Check:
    v = _num(value)
    return Check(name, v, f">= {limit}", v is not None and v >= limit, blocking=blocking,
                 note=note if v is not None else "ausente/não-finito")


def _le(name: str, value, limit: float, *, blocking: bool = True, note: str = "") -> Check:
    v = _num(value)
    return Check(name, v, f"<= {limit}", v is not None and v <= limit, blocking=blocking,
                 note=note if v is not None else "ausente/não-finito")


def m2_verdict(inp: M2Inputs, *, gates_path: Path = GATES_V2, selection_path: Path = SELECTION_V1) -> Verdict:
    gates = load_gates(gates_path)
    selection = load_selection(selection_path)
    dev_levels = load_deviation_levels()

    checks: list[Check] = []

    # 1. Gate de desvio de runtime (informativo/caminho)
    e2s = _num(inp.runtime_deviation.get("error_to_signal_db"))
    agree = _num(inp.runtime_deviation.get("stage_agreement"))
    path_chosen = inp.runtime_deviation.get("path")
    dev_ok = (e2s is not None and agree is not None and e2s <= dev_levels.error_to_signal_db_max
              and agree >= dev_levels.stage_agreement_min)
    checks.append(Check("runtime_deviation", e2s, f"<= {dev_levels.error_to_signal_db_max} dB", dev_ok,
                        blocking=False, note=f"concordância={agree}, caminho escolhido={path_chosen}"))

    # 2. Gate G1: Modo sem perfil (C0), estágio F
    # Decisão do dono: margens de mínimo lidas pelo LIMITE INFERIOR do IC 95%; LSD > 8 kHz pelo LIMITE SUPERIOR.
    g1_crit = gates.gates["G1"].criteria["stage_f"]

    # delta_stoi (min)
    v_stoi = inp.g1.get("delta_stoi", {})
    stoi_val = v_stoi.get("ci_low") if isinstance(v_stoi, Mapping) else v_stoi
    checks.append(_ge("G1_delta_stoi_ci_low", stoi_val, float(g1_crit["delta_stoi_min"]),
                      note="IC 95% inferior (decisão do dono)"))

    # delta_si_sdr_db (min)
    v_sisdr = inp.g1.get("delta_si_sdr_db", {})
    sisdr_val = v_sisdr.get("ci_low") if isinstance(v_sisdr, Mapping) else v_sisdr
    checks.append(_ge("G1_delta_si_sdr_ci_low", sisdr_val, float(g1_crit["delta_si_sdr_db_min"]),
                      note="IC 95% inferior (decisão do dono)"))

    # delta_dnsmos_ovrl (min)
    v_dovrl = inp.g1.get("delta_dnsmos_ovrl", {})
    dovrl_val = v_dovrl.get("ci_low") if isinstance(v_dovrl, Mapping) else v_dovrl
    checks.append(_ge("G1_delta_dnsmos_ovrl_ci_low", dovrl_val, float(g1_crit["delta_dnsmos_ovrl_min"]),
                      note="IC 95% inferior (decisão do dono)"))

    # delta_dnsmos_sig (min)
    v_dsig = inp.g1.get("delta_dnsmos_sig", {})
    dsig_val = v_dsig.get("ci_low") if isinstance(v_dsig, Mapping) else v_dsig
    checks.append(_ge("G1_delta_dnsmos_sig_ci_low", dsig_val, float(g1_crit["delta_dnsmos_sig_min"]),
                      note="IC 95% inferior (decisão do dono)"))

    # delta_dnsmos_bak (min)
    v_dbak = inp.g1.get("delta_dnsmos_bak", {})
    dbak_val = v_dbak.get("ci_low") if isinstance(v_dbak, Mapping) else v_dbak
    checks.append(_ge("G1_delta_dnsmos_bak_ci_low", dbak_val, float(g1_crit["delta_dnsmos_bak_min"]),
                      note="IC 95% inferior (decisão do dono)"))

    # delta_lsd_above_8k_db (max, lido pelo limite superior)
    v_lsd = inp.g1.get("delta_lsd_above_8k_db", {})
    lsd_val = v_lsd.get("ci_high") if isinstance(v_lsd, Mapping) else v_lsd
    checks.append(_le("G1_delta_lsd_above_8k_ci_high", lsd_val, float(g1_crit["delta_lsd_above_8k_db_max"]),
                      note="IC 95% superior (decisão do dono)"))

    # 3. Gate G2: Falante concorrente (C3/C4, SIR <= 5 dB)
    g2_crit = gates.gates["G2"].criteria
    g2_sisdr = inp.g2.get("delta_si_sdr_db", {})
    g2_pt = g2_sisdr.get("point") if isinstance(g2_sisdr, Mapping) else g2_sisdr
    g2_ci_low = g2_sisdr.get("ci_low") if isinstance(g2_sisdr, Mapping) else None

    checks.append(_ge("G2_delta_si_sdr_point", g2_pt, float(g2_crit["delta_si_sdr_db_min"])))
    checks.append(Check("G2_delta_si_sdr_ci_lower_gt_0", _num(g2_ci_low), "> 0.0",
                        _num(g2_ci_low) is not None and _num(g2_ci_low) > float(g2_crit["delta_si_sdr_ci_lower_gt"])))

    checks.append(_ge("G2_delta_dnsmos_p_ovrl", inp.g2.get("delta_dnsmos_p_ovrl"),
                      float(g2_crit["delta_dnsmos_p_ovrl_min"])))
    checks.append(_le("G2_wer_ratio_to_dfnet3", inp.g2.get("wer_ratio_to_dfnet3"),
                      float(g2_crit["wer_ratio_to_dfnet3_max"])))
    wer_not_above = bool(inp.g2.get("wer_not_above_noisy"))
    checks.append(Check("G2_wer_not_above_noisy", wer_not_above, "True", wer_not_above))

    # 4. Gate G3: TSOS (Target Speech Over-Suppression, Opção A aprovada: critério relativo <= +0.5 p.p.)
    g3_crit = gates.gates["G3"].criteria
    delta_tsos = _num(inp.g3.get("delta_tsos_over_dfnet3_pp"))
    max_over = float(g3_crit["tsos_max_over_dfnet3_pp"])
    checks.append(_le("G3_tsos_max_over_dfnet3_pp", delta_tsos, max_over,
                      note="Convenção fixa Opção A (critério estritamente relativo)"))

    tsos_c0 = _num(inp.g3.get("tsos_c0_pct"))
    if tsos_c0 is not None:
        checks.append(_le("G3_tsos_c0_pct", tsos_c0, float(g3_crit["tsos_max_pct"]["C0"])))

    # 5. Gate G4: Locutor trocado (TRR e separação de modos)
    g4_crit = gates.gates["G4"].criteria
    checks.append(_ge("G4_trr_different_sex", inp.g4.get("trr_diff_sex"), float(g4_crit["trr_min_different_sex"])))
    checks.append(_ge("G4_trr_same_sex", inp.g4.get("trr_same_sex"), float(g4_crit["trr_min_same_sex"])))
    checks.append(_ge("G4_mode_separation_db", inp.g4.get("mode_sep_db"), float(g4_crit["mode_separation_db_min"])))

    # Informativos opcionais (G5, G8, C8)
    if inp.g5 is not None:
        checks.append(_ge("G5_spk_sim_delta_ci_lower", inp.g5.get("spk_sim_delta_ci_lower"), -0.02, blocking=False,
                          note="informativo M2"))
    if inp.g8 is not None:
        checks.append(Check("G8_dnsmos_p", _num(inp.g8.get("dnsmos_p_ovrl")), "informativo", True, blocking=False))
    if inp.c8 is not None:
        sat = _num(inp.c8.get("tanh_saturation_pct"))
        checks.append(Check("C8_tanh_saturation", sat, "<= 0.05", sat is not None and sat <= 0.05, blocking=False,
                            note="informativo C8"))

    blocking_fails = tuple(c.name for c in checks if c.blocking and not c.passed)
    go = len(blocking_fails) == 0

    return Verdict(
        go=go,
        checks=tuple(checks),
        gates_sha256=gates.sha256,
        selection_sha256=selection.sha256,
        deviation_sha256=dev_levels.sha256,
        blocking_failures=blocking_fails,
    )


def render_checks_table(verdict: Verdict) -> str:
    lines = [
        "| Gate / Métrica | Valor Medido | Nível Exigido | Status | Bloqueante? | Notas |",
        "| :--- | :--- | :--- | :---: | :---: | :--- |",
    ]
    for c in verdict.checks:
        status = "**PASS**" if c.passed else "**FAIL**"
        blk = "Sim" if c.blocking else "Não"
        val_str = f"{c.value:.4f}" if isinstance(c.value, float) else str(c.value)
        lines.append(f"| `{c.name}` | {val_str} | {c.level} | {status} | {blk} | {c.note} |")
    return "\n".join(lines)
