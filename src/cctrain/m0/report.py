"""Relatório M0: projeções com fatores medidos e go/no-go pelos níveis pré-registrados (budget-v1)."""

from __future__ import annotations

from collections.abc import Mapping

from cctrain.eval.gates import Budget

GB = 1e9
SPEECH_SOURCES = ("vctk-0.92", "librispeech-dev-clean")


def _speech_by_sr(disk: Mapping) -> dict[int, float]:
    """Fator FLAC de fala por SR vindo das fontes de fala medidas; sem recorte por fonte, cai no agregado por SR."""
    by_source = disk.get("by_source") or {}
    out = {int(sr): v["flac_gb_per_hour"] for src in SPEECH_SOURCES for sr, v in by_source.get(src, {}).items()}
    return out or {int(k): v["flac_gb_per_hour"] for k, v in disk["by_sr"].items()}


def _factor(disk: Mapping, sr: int, by_sr: Mapping[int, float] | None = None) -> tuple[float, str]:
    if by_sr is None:
        by_sr = {int(k): v["flac_gb_per_hour"] for k, v in disk["by_sr"].items()}
    if sr in by_sr:
        return by_sr[sr], "medido"
    lower = max((s for s in by_sr if s < sr), default=min(by_sr))
    upper = min((s for s in by_sr if s > sr), default=max(by_sr))
    if lower == upper:
        return by_sr[lower] * sr / lower, "interpolado"
    w = (sr - lower) / (upper - lower)
    return by_sr[lower] + w * (by_sr[upper] - by_sr[lower]), "interpolado"


def project_disk(minimal: Mapping, disk: Mapping) -> dict:
    rows = []
    speech_by_sr = _speech_by_sr(disk)
    for group in ("speech", "noise"):
        for item in minimal[group]:
            if "fixed_gb" in item:
                rows.append({"id": item["id"], "gb": float(item["fixed_gb"]), "factor_source": "fixo (pesquisa)"})
                continue
            factor, source = _factor(disk, int(item["sr"]), speech_by_sr if group == "speech" else None)
            rows.append({"id": item["id"], "gb": item["hours"] * factor, "factor_source": source})
    rir = disk["rir"]
    per_rir = rir["stored_bytes"] / max(rir["count"], 1)
    rows.append({"id": "rir", "gb": minimal["rir"]["count"] * per_rir / GB, "factor_source": "medido"})
    dirs = disk["dirs_bytes"]
    measured = {k: (dirs.get(k) or 0) / GB for k in ("venv", "tract_check_target", "rustup")}
    estimated = dict(minimal["estimated_fixed_gb"])
    total = sum(r["gb"] for r in rows) + sum(measured.values()) + sum(estimated.values())
    return {"rows": rows, "measured_gb": measured, "estimated_gb": estimated, "total_gb": total}


def t3_hours(t3: Mapping, s_per_step: float) -> float:
    steps_per_epoch = t3["hours_audio_per_epoch"] * 3600.0 / t3["segment_s"] / t3["batch"]
    return t3["epochs"] * steps_per_epoch * s_per_step / 3600.0


def go_no_go(budget: Budget, disk_projection_gb: float, free_gb: float, step16: Mapping,
             loader: Mapping) -> list[dict]:
    hours = t3_hours(budget.t3, float(step16["s_per_step_median"])) if "s_per_step_median" in step16 else float("inf")
    time_verdict = ("go" if hours <= budget.t3["go_hours_max"]
                    else "go com ressalva" if hours <= budget.t3["caveat_hours_max"] else "no-go")
    vram = float(step16.get("peak_reserved_gb", float("inf")))
    return [
        {"item": "disco", "value": disk_projection_gb, "limit": free_gb - budget.reserve_gb,
         "verdict": "go" if disk_projection_gb <= free_gb - budget.reserve_gb else "no-go"},
        {"item": "vram_b16", "value": vram, "limit": budget.vram_b16_reserved_gb_max,
         "verdict": "go" if vram <= budget.vram_b16_reserved_gb_max else "no-go"},
        {"item": "ram_loader", "value": float(loader["rss_max_gb"]), "limit": budget.loader_rss_gb_max,
         "verdict": "go" if float(loader["rss_max_gb"]) <= budget.loader_rss_gb_max else "no-go"},
        {"item": "tempo_t3", "value": hours, "limit": budget.t3["go_hours_max"], "verdict": time_verdict},
    ]


def render_markdown(ctx: Mapping) -> str:
    lines = [f"# {ctx['title']}", ""]
    for heading, body in ctx["sections"]:
        lines += [f"## {heading}", "", body.rstrip(), ""]
    lines += ["## Go/no-go do orçamento (níveis de configs/m0/budget-v1.yaml)", "",
              "| item | valor | limite | veredito |", "|---|---:|---:|---|"]
    for r in ctx["go_no_go"]:
        lines.append(f"| {r['item']} | {r['value']:.3f} | {r['limit']:.3f} | {r['verdict']} |")
    return "\n".join(lines) + "\n"
