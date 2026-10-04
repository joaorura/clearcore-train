"""M2 Tarefa 5. Uso: m2.py {smoke,t3,branch,evaluate,report}. Grava em $CCTRAIN_RUNS_DIR/m2/.

smoke     ~20 min de estágio F com dados reais sob o gpu_lock: s/passo, VRAM, PSS, vazão do loader, magnitude de cada
          termo da perda e a estimativa de horas do T3/T7 refeita com o s/passo medido (`t5-smoke.json`).
t3        rodada principal do estágio F (Tarefa 6), idempotente: rode de novo o mesmo comando para retomar.
branch    ramo da AB2 a partir de um checkpoint de fim de época (Tarefa 7).
evaluate  métricas da regra de seleção de um checkpoint no `dev-select` (precisa da Tarefa 4).
report    resumo de um run: avaliações por época, melhor pela regra, parada.

O banco é o v2 (`bank-m2-sbecapa-v2`) quando existe e o formato é lido pelo código desta árvore; senão o v1 do M1, e o
`t5-smoke.json` registra qual foi usado. O `dev-select` vem de `cctrain.m2.devsets.load_select_items()` (Tarefa 4).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from cctrain.paths import Paths

PATHS = Paths.from_env()
OUT = PATHS.runs / "m2"
EMB = PATHS.data / "embeddings"
BANK_V2, BANK_V1 = EMB / "bank-m2-sbecapa-v2.npz", EMB / "bank-m1-sbecapa-v1.npz"
EVAL_HOURS_T3 = 1.7  # avaliações do T3 (plano, Tarefa 6)
EVAL_HOURS_BRANCH = 0.5  # avaliações de um ramo da AB2 (plano, Tarefa 7)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(path: Path, obj: dict) -> None:
    from cctrain.train.stage_c import _atomic_json

    _atomic_json(path, obj)
    log(f"gravado {path}")


def load_bank():
    from cctrain.train.embed_bank import ArrayEmbeddingBank

    if BANK_V2.is_file():
        bank = ArrayEmbeddingBank.load(BANK_V2)
        if getattr(bank, "has_provenance", False):
            return bank, "v2"
        log("banco v2 existe mas esta árvore o lê sem proveniência: usando o v1")
    return ArrayEmbeddingBank.load(BANK_V1), "v1"


def load_inputs():
    from cctrain.m1 import pools
    from cctrain.train.stage_c import StageCData

    speech = pools.by_split(pools.speech_records(PATHS), "train")
    noise_train, _ = pools.noise_pools()
    return StageCData(speech, noise_train, pools.rir_records(), None)


def evaluator_and_rule(enabled: bool):
    if not enabled:
        return None, None
    from cctrain.m2.devsets import load_select_items  # Tarefa 4

    from cctrain.m2.config import load_selection
    from cctrain.m2.select import make_evaluator
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3

    teacher, _ = load_upstream_dfnet3(device="cpu")
    return make_evaluator(load_select_items(), teacher), load_selection()


def estimate(s_per_step: float, steps_per_epoch: int = 7500, epochs: int = 20) -> dict:
    epoch_h = steps_per_epoch * s_per_step / 3600.0
    t3 = epochs * epoch_h + EVAL_HOURS_T3
    ab2 = 4 * (6 * epoch_h + EVAL_HOURS_BRANCH)
    return {"s_per_step": s_per_step, "epoch_hours": epoch_h, "t3_train_hours": epochs * epoch_h,
            "t3_hours_with_eval": t3, "ab2_hours_4_branches": ab2, "ab1_p0_hours": t3,
            "t7_hours_total": ab2 + t3, "plan": {"t3_hours": 12.5, "t7_hours": 28.0, "epoch_hours": 0.54,
                                                  "s_per_step": 0.26}}


def cmd_smoke(a) -> None:
    from cctrain.m2.config import load_stage_f_config
    from cctrain.train.stage_f import StageFHParams, measure_loader_throughput, run_stage_f

    cfg = load_stage_f_config()
    bank, bank_kind = load_bank()
    data = load_inputs()
    run_id = a.run_id
    hp = StageFHParams.from_config(cfg, run_id=run_id, num_workers=a.workers, prefetch_factor=2, probe_every=10,
                                   checkpoint_every_min=a.checkpoint_every_min)
    log(f"banco {bank_kind}: {len(bank.speakers())} falantes; config {cfg.sha256[:12]}; {a.minutes} min")
    loader = measure_loader_throughput(hp, bank, data, n_batches=a.loader_batches)
    log(f"loader sozinho: {loader['batches_per_s']:.2f} batches/s, PSS {loader['pss_gb_max']:.2f} GB")
    ckpt = run_stage_f(hp, bank, data, max_hours=a.minutes / 60.0)
    summary = json.loads((OUT / run_id / "summary.json").read_text())
    rows = [json.loads(x) for x in (OUT / run_id / "loss-log.jsonl").read_text().splitlines()]
    tail = rows[len(rows) // 5:]  # sem o início (aquecimento do LR e do loader)
    terms = {k: sum(r[k] for r in tail) / len(tail) for k in ("mr", "os", "gate", "lsnr", "t", "total")}
    ratios = {k: terms[k] / terms["mr"] for k in ("os", "gate", "lsnr", "t")}
    in_band = {k: 0.01 <= ratios[k] <= 10.0 for k in ("os", "gate")}
    s_step = summary["s_per_step_median"]
    out = {
        "run_id": run_id, "bank": bank_kind, "bank_speakers": len(bank.speakers()), "config_sha256": cfg.sha256,
        "commit": summary["commit"], "steps": summary["steps"], "minutes": a.minutes, "num_workers": a.workers,
        "s_per_step_median": s_step, "s_per_step_mean": summary["s_per_step_mean"],
        "trainer_steps_per_s": 1.0 / s_step, "loader": loader,
        "loader_over_trainer": loader["batches_per_s"] * s_step, "loader_ok_ge_1_2": loader["batches_per_s"] * s_step >= 1.2,
        "loader_wait_s_mean_in_run": summary["loader_wait_s_mean"],
        "vram_peak_allocated_gb": summary["vram_peak_allocated_gb"],
        "vram_peak_reserved_gb": summary["vram_peak_reserved_gb"], "vram_limit_gb": 7.0,
        "pss_gb_max_trainer_plus_workers": summary["pss_gb_max"], "pss_limit_gb": summary["pss_limit_gb"],
        "loss_terms_mean_last_80pct": terms, "ratio_to_base_mr": ratios,
        "os_gate_within_1pct_10x": in_band, "needs_orchestrator_before_t3": not all(in_band.values()),
        "loss_first_last": [rows[0]["total"], rows[-1]["total"]],
        "mask0_fraction": sum(r["n0"] for r in rows) / (len(rows) * hp.batch_size),
        "bn_unchanged": summary["bn_sha256_before"] == summary["bn_sha256_after"],
        "estimate": estimate(s_step), "last_checkpoint": ckpt.name,
        "note": "smoke: sem avaliação no dev-select (Tarefa 4 pendente); estimativa do T3 = 20 épocas x 7.500 passos x "
                f"s/passo mediano + {EVAL_HOURS_T3} h de avaliações (plano).",
    }
    save(OUT / "t5-smoke.json", out)
    log(json.dumps({k: out[k] for k in ("s_per_step_median", "vram_peak_reserved_gb",
                                          "pss_gb_max_trainer_plus_workers", "ratio_to_base_mr", "estimate")},
                   indent=1, default=str))


def cmd_t3(a) -> None:
    from cctrain.m2.config import load_stage_f_config
    from cctrain.train.stage_f import StageFHParams, run_stage_f

    cfg = load_stage_f_config()
    bank, bank_kind = load_bank()
    if bank_kind != "v2":
        raise SystemExit("o T3 exige o banco v2 (proveniência por variante)")
    evaluator, rule = evaluator_and_rule(not a.no_eval)
    hp = StageFHParams.from_config(cfg, run_id=a.run_id, num_workers=a.workers)
    ckpt = run_stage_f(hp, bank, load_inputs(), max_hours=a.max_hours, resume=True, evaluator=evaluator, rule=rule)
    log(f"checkpoint final: {ckpt}")


def cmd_branch(a) -> None:
    from cctrain.train.stage_f import branch

    bank, bank_kind = load_bank()
    if bank_kind != "v2":
        raise SystemExit("os ramos exigem o banco v2")
    over = {k: v for k, v in (("p_drop", a.p_drop), ("s", a.s), ("p_enroll_denoise", a.p_enroll_denoise),
                              ("seed", a.seed)) if v is not None}
    evaluator, rule = evaluator_and_rule(not a.no_eval)
    out = branch(Path(a.from_ckpt), over, a.epochs, bank=bank, data=load_inputs(), run_id=a.run_id,
                 max_hours=a.max_hours, evaluator=evaluator, rule=rule, num_workers=a.workers)
    log(f"checkpoint final do ramo: {out}")


def cmd_evaluate(a) -> None:
    from cctrain.m2.devsets import load_select_items

    from cctrain.m2.config import load_selection
    from cctrain.m2.select import evaluate_select, make_epoch_eval, write_eval
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.train.checkpoint import load_checkpoint
    from cctrain.train.stage_f import StageFHParams, build_models

    ck = load_checkpoint(Path(a.ckpt))
    hp = StageFHParams(device="cpu", **{k: v for k, v in ck.extra["hp"].items() if k in
                                        ("log_gamma_bound", "beta_bound")})
    model, _, _ = build_models(hp, "cpu")
    model.load_state_dict(ck.model)
    teacher, _ = load_upstream_dfnet3(device="cpu")
    metrics = evaluate_select(model, load_select_items(), teacher=teacher)
    ev = make_epoch_eval(a.epoch, ck.step, metrics, load_selection())
    run_dir = OUT / a.run_id
    log(f"gravado {write_eval(run_dir, ev)}: {json.dumps(ev.metrics)} elegível={ev.eligible}")


def cmd_report(a) -> None:
    from cctrain.m2.config import load_selection
    from cctrain.m2.select import best_epoch, read_history, should_stop

    run_dir, rule = OUT / a.run_id, load_selection()
    hist = read_history(run_dir)
    best = best_epoch(hist, rule)
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").is_file() else {}
    out = {"run_id": a.run_id, "epochs_evaluated": [h.epoch for h in hist],
           "eligible": {h.epoch: h.eligible for h in hist}, "scores": {h.epoch: h.score for h in hist},
           "best_epoch": best.epoch if best else None, "best_step": best.step if best else None,
           "early_stopping_would_stop": should_stop(hist, rule) if hist else False, "summary": summary}
    save(run_dir / "report.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--minutes", type=float, default=20.0)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--loader-batches", type=int, default=60)
    s.add_argument("--checkpoint-every-min", type=float, default=30.0)
    s.add_argument("--run-id", default="t5-smoke")
    t = sub.add_parser("t3")
    t.add_argument("--resume", action="store_true", help="aceito por compatibilidade: sempre retoma")
    t.add_argument("--run-id", default="t3")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--max-hours", type=float, default=35.0)
    t.add_argument("--no-eval", action="store_true")
    b = sub.add_parser("branch")
    b.add_argument("--from-ckpt", required=True)
    b.add_argument("--run-id", required=True)
    b.add_argument("--epochs", type=int, default=6)
    b.add_argument("--p-drop", type=float)
    b.add_argument("--s", type=float)
    b.add_argument("--p-enroll-denoise", type=float)
    b.add_argument("--seed", type=int)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--max-hours", type=float, default=8.0)
    b.add_argument("--no-eval", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--run-id", required=True)
    e.add_argument("--epoch", type=int, required=True)
    r = sub.add_parser("report")
    r.add_argument("--run-id", default="t3")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    {"smoke": cmd_smoke, "t3": cmd_t3, "branch": cmd_branch, "evaluate": cmd_evaluate, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
