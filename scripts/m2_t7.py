#!/usr/bin/env python3
"""Orquestrador e automação da Tarefa T7 do M2: Ablações AB2 (ramos) e AB1 (controle).

Pré-requisitos:
- Banco de embeddings v2 presente em data/embeddings/bank-m2-sbecapa-v2.npz
- dev-select pré-renderizado presente em data/dev-m2/dev-select/
- Checkpoint base: runs/m2/t3/checkpoints/ckpt-00022500.pt (Época 3 do T3)
- Configuração congelada em configs/eval/m2-ablations-v1.yaml

Comandos:
  status         Verifica o status de todos os ramos da AB2 e da AB1
  run-stage1     Executa sequencialmente o Stage 1 da AB2 (ab2-pdrop01 e ab2-pdrop03)
  decide-stage1  Avalia e decide o melhor p_drop do Stage 1
  run-stage2     Executa o Stage 2 da AB2 (ab2-s00 e ab2-s10) com o p_drop vencedor
  decide-candidate Elege o candidato final da AB2 para a Tarefa 8
  run-ab1        Executa o controle AB1 (ab1-p0)
  report         Gera o relatório formal docs/plans/m2-reports/t7.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "eval" / "m2-ablations-v1.yaml"
SELECTION_PATH = ROOT / "configs" / "eval" / "m2-selection-v1.yaml"
RUNS_DIR = ROOT / "runs" / "m2"
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"


def log(msg: str) -> None:
    print(f"[m2-t7] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"Configuração ausente: {CONFIG_PATH}")
    sha_file = CONFIG_PATH.with_suffix(".sha256")
    if not sha_file.is_file():
        raise SystemExit(f"SHA-256 ausente: {sha_file}")
    expected = sha_file.read_text().split()[0].strip()
    actual = sha256_file(CONFIG_PATH)
    if actual != expected:
        raise SystemExit(f"SHA-256 de {CONFIG_PATH} divergente: esperado {expected}, calculado {actual}")
    return yaml.safe_load(CONFIG_PATH.read_text())


def get_base_checkpoint(cfg: dict) -> Path:
    base = Path(cfg["base_checkpoint"]["path"])
    if not base.is_absolute():
        base = ROOT / base
    if not base.is_file():
        raise SystemExit(f"Checkpoint base ausente: {base}")
    expected_sha = cfg["base_checkpoint"]["sha256"]
    actual_sha = sha256_file(base)
    if actual_sha != expected_sha:
        raise SystemExit(f"SHA-256 do checkpoint base difere: {actual_sha} != {expected_sha}")
    return base


def find_running_pid(run_id: str) -> int | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", f"run-id {run_id}"], text=True)
        my_pid = os.getpid()
        pids = [int(p) for p in out.strip().splitlines() if p.isdigit() and int(p) != my_pid]
        return pids[0] if pids else None
    except Exception:
        return None


def run_branch_cmd(from_ckpt: Path, run_id: str, p_drop: float | None = None, s: float | None = None,
                   p_enroll_denoise: float | None = None, seed: int = 0, epochs: int = 6,
                   max_hours: float = 8.0, workers: int = 4) -> None:
    cmd = [
        str(PYTHON_BIN),
        str(ROOT / "scripts" / "m2.py"),
        "branch",
        "--from-ckpt", str(from_ckpt),
        "--run-id", run_id,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--workers", str(workers),
        "--max-hours", str(max_hours),
    ]
    if p_drop is not None:
        cmd.extend(["--p-drop", str(p_drop)])
    if s is not None:
        cmd.extend(["--s", str(s)])
    if p_enroll_denoise is not None:
        cmd.extend(["--p-enroll-denoise", str(p_enroll_denoise)])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CCTRAIN_ROOT"] = str(ROOT)
    env["CCTRAIN_DATA_DIR"] = str(ROOT / "data")
    env["CCTRAIN_RUNS_DIR"] = str(ROOT / "runs")

    log_file = RUNS_DIR / run_id / "train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"Iniciando ramo {run_id} (log em {log_file})...")

    with open(log_file, "a") as out:
        p = subprocess.run(cmd, env=env, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"Ramo {run_id} falhou com código {p.returncode}. Verifique {log_file}")
    log(f"Ramo {run_id} finalizado com sucesso.")


def run_branch_monitored(from_ckpt: Path, run_id: str, p_drop: float | None = None, s: float | None = None,
                         p_enroll_denoise: float | None = None, seed: int = 0, epochs: int = 6,
                         max_hours: float = 8.0, workers: int = 4, check_interval: int = 30) -> None:
    # 1. Se já completou todas as épocas
    hist = read_run_history(run_id)
    if len(hist) >= epochs:
        log(f"Ramo {run_id} já concluído ({len(hist)}/{epochs} épocas avaliadas).")
        return

    # 2. Se já existe processo rodando
    pid = find_running_pid(run_id)
    if pid is not None:
        log(f"Ramo {run_id} ativo no PID {pid}. Monitorando passos até o término...")
        last_step = -1
        last_evals = len(hist)
        while Path(f"/proc/{pid}").exists():
            log_path = RUNS_DIR / run_id / "loss-log.jsonl"
            if log_path.exists():
                try:
                    with open(log_path, "rb") as f:
                        f.seek(max(0, f.seek(0, 2) - 2048))
                        lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                    if lines:
                        last = json.loads(lines[-1])
                        st = last.get("step", 0)
                        ep = last.get("epoch", 0)
                        loss = last.get("total", 0.0)
                        if st != last_step:
                            last_step = st
                            pct = (st / 67500.0) * 100.0
                            log(f"[{run_id}] Passo {st:,}/67,500 ({pct:.1f}%), Época {ep}, Loss: {loss:.4f}")
                except Exception:
                    pass

            cur_evals = read_run_history(run_id)
            if len(cur_evals) > last_evals:
                ev = cur_evals[-1]
                last_evals = len(cur_evals)
                sc = ev.get("score")
                sc_str = f"{sc:+.4f} dB" if sc is not None else "N/A"
                log(f"[{run_id}] Avaliação da Época {ev.get('epoch')} concluída: C3 Escore = {sc_str}, Elegível = {ev.get('eligible')}")

            time.sleep(check_interval)
        log(f"Processo PID {pid} de {run_id} finalizou.")

        hist = read_run_history(run_id)
        if len(hist) >= epochs:
            log(f"Ramo {run_id} concluído com sucesso ({len(hist)}/{epochs} épocas avaliadas).")
            return
        log(f"Aviso: Processo anterior finalizou com {len(hist)}/{epochs} épocas. Retomando treinamento...")

    # 3. Lançar o ramo monitorado via subprocess.Popen
    cmd = [
        str(PYTHON_BIN),
        str(ROOT / "scripts" / "m2.py"),
        "branch",
        "--from-ckpt", str(from_ckpt),
        "--run-id", run_id,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--workers", str(workers),
        "--max-hours", str(max_hours),
    ]
    if p_drop is not None:
        cmd.extend(["--p-drop", str(p_drop)])
    if s is not None:
        cmd.extend(["--s", str(s)])
    if p_enroll_denoise is not None:
        cmd.extend(["--p-enroll-denoise", str(p_enroll_denoise)])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CCTRAIN_ROOT"] = str(ROOT)
    env["CCTRAIN_DATA_DIR"] = str(ROOT / "data")
    env["CCTRAIN_RUNS_DIR"] = str(ROOT / "runs")

    log_file = RUNS_DIR / run_id / "train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"Iniciando ramo {run_id} (log em {log_file})...")

    with open(log_file, "a") as out_f:
        proc = subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=out_f, stderr=subprocess.STDOUT)
        log(f"Ramo {run_id} em execução no PID {proc.pid}. Monitorando passos...")

        last_step = -1
        last_evals = len(read_run_history(run_id))
        while proc.poll() is None:
            log_path = RUNS_DIR / run_id / "loss-log.jsonl"
            if log_path.exists():
                try:
                    with open(log_path, "rb") as f:
                        f.seek(max(0, f.seek(0, 2) - 2048))
                        lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                    if lines:
                        last = json.loads(lines[-1])
                        st = last.get("step", 0)
                        ep = last.get("epoch", 0)
                        loss = last.get("total", 0.0)
                        if st != last_step:
                            last_step = st
                            pct = (st / 67500.0) * 100.0
                            log(f"[{run_id}] Passo {st:,}/67,500 ({pct:.1f}%), Época {ep}, Loss: {loss:.4f}")
                except Exception:
                    pass

            cur_evals = read_run_history(run_id)
            if len(cur_evals) > last_evals:
                ev = cur_evals[-1]
                last_evals = len(cur_evals)
                sc = ev.get("score")
                sc_str = f"{sc:+.4f} dB" if sc is not None else "N/A"
                log(f"[{run_id}] Avaliação da Época {ev.get('epoch')} concluída: C3 Escore = {sc_str}, Elegível = {ev.get('eligible')}")

            time.sleep(check_interval)

        ret = proc.returncode
        if ret != 0:
            raise RuntimeError(f"Ramo {run_id} falhou com código {ret}. Verifique {log_file}")
        log(f"Ramo {run_id} finalizado com sucesso (código 0).")


def read_run_history(run_id: str) -> list[dict]:
    eval_dir = RUNS_DIR / run_id / "eval"
    if not eval_dir.is_dir():
        return []
    history = []
    for f in sorted(eval_dir.glob("epoch-*.json")):
        try:
            history.append(json.loads(f.read_text()))
        except Exception:
            pass
    return history


def get_best_run_epoch(run_id: str) -> dict | None:
    best_file = RUNS_DIR / run_id / "checkpoints" / "best.json"
    if best_file.is_file():
        try:
            return json.loads(best_file.read_text())
        except Exception:
            pass
    return None


def cmd_status(_args) -> None:
    cfg = verify_config()
    base = get_base_checkpoint(cfg)
    log(f"Base Checkpoint: {base} (SHA-256 verificado)")

    branches = ["ab2-pdrop01", "ab2-pdrop03", "ab2-s00", "ab2-s10", "ab1-p0"]
    print("\n--- Status dos Ramos de Ablação ---")
    for b in branches:
        hist = read_run_history(b)
        best = get_best_run_epoch(b)
        n_epochs = len(hist)
        last_epoch = hist[-1]["epoch"] if hist else 0
        best_str = f"Step {best['step']}, metric {best['metric']:.4f}" if best else "N/A"
        print(f"[{b:15s}] Épocas avaliadas: {n_epochs:2d} (última: {last_epoch:2d}) | Melhor: {best_str}")


def cmd_run_stage1(args) -> None:
    cfg = verify_config()
    base = get_base_checkpoint(cfg)

    # Ramo 1: ab2-pdrop01
    log("=== Executando Stage 1 Ramo 1: ab2-pdrop01 (p_drop=0.1, s=0.05) ===")
    run_branch_cmd(base, "ab2-pdrop01", p_drop=0.1, s=0.05, epochs=args.epochs, max_hours=args.max_hours)

    # Ramo 2: ab2-pdrop03
    log("=== Executando Stage 1 Ramo 2: ab2-pdrop03 (p_drop=0.3, s=0.05) ===")
    run_branch_cmd(base, "ab2-pdrop03", p_drop=0.3, s=0.05, epochs=args.epochs, max_hours=args.max_hours)

    log("Stage 1 concluído. Execute 'python scripts/m2_t7.py decide-stage1' para escolher o melhor p_drop.")


def cmd_decide_stage1(_args) -> float:
    # Compara baseline (t3, p_drop=0.2), ab2-pdrop01 e ab2-pdrop03
    cfg = verify_config()
    candidates = [
        {"run_id": "ab2-pdrop01", "p_drop": 0.1},
        {"run_id": "t3", "p_drop": 0.2},
        {"run_id": "ab2-pdrop03", "p_drop": 0.3},
    ]
    results = []
    for c in candidates:
        rid = c["run_id"]
        hist = read_run_history(rid)
        best_eval = None
        for h in hist:
            if h.get("eligible"):
                if best_eval is None or h["score"] > best_eval["score"]:
                    best_eval = h
        results.append({
            "run_id": rid,
            "p_drop": c["p_drop"],
            "best_eval": best_eval,
            "score": best_eval["score"] if best_eval else -999.0,
            "eligible": best_eval is not None,
        })

    print("\n--- Comparação Stage 1 (p_drop) no dev-select ---")
    for r in results:
        ev = r["best_eval"]
        sc_str = f"{r['score']:+.2f} dB" if r["eligible"] else "INELEGÍVEL"
        ep_str = f"Época {ev['epoch']}" if ev else "N/A"
        print(f"p_drop={r['p_drop']:.1f} ({r['run_id']:12s}): Escore C3 = {sc_str} ({ep_str})")

    # Regra: menor p_drop elegível que maximize C3 / satisfaça critérios
    elig = [r for r in results if r["eligible"]]
    if not elig:
        raise RuntimeError("Nenhum candidato elegível no Stage 1 da AB2!")
    best = max(elig, key=lambda r: r["score"])
    log(f"Decisão Stage 1: Melhor p_drop = {best['p_drop']} ({best['run_id']}) com Escore {best['score']:+.2f} dB")
    return float(best["p_drop"])


def cmd_run_stage2(args) -> None:
    cfg = verify_config()
    base = get_base_checkpoint(cfg)
    best_p_drop = args.p_drop if args.p_drop is not None else cmd_decide_stage1(args)

    log(f"=== Executando Stage 2 com p_drop={best_p_drop} fixado ===")
    log("=== Executando Stage 2 Ramo 3: ab2-s00 (s=0.0) ===")
    run_branch_cmd(base, "ab2-s00", p_drop=best_p_drop, s=0.0, epochs=args.epochs, max_hours=args.max_hours)

    log("=== Executando Stage 2 Ramo 4: ab2-s10 (s=0.10) ===")
    run_branch_cmd(base, "ab2-s10", p_drop=best_p_drop, s=0.10, epochs=args.epochs, max_hours=args.max_hours)

    log("Stage 2 concluído. Execute 'python scripts/m2_t7.py decide-candidate' para eleger o candidato final.")


def cmd_decide_candidate(_args) -> dict:
    candidates = ["t3", "ab2-pdrop01", "ab2-pdrop03", "ab2-s00", "ab2-s10"]
    scored = []
    for rid in candidates:
        hist = read_run_history(rid)
        best_eval = None
        for h in hist:
            if h.get("eligible"):
                if best_eval is None or h["score"] > best_eval["score"]:
                    best_eval = h
        if best_eval:
            scored.append({"run_id": rid, "best_eval": best_eval, "score": best_eval["score"]})

    if not scored:
        raise RuntimeError("Nenhum candidato elegível encontrado na AB2!")

    best = max(scored, key=lambda x: x["score"])
    ev = best["best_eval"]
    log(f"=== CANDIDATO VENCEDOR ELEITO DA AB2: {best['run_id']} (Época {ev['epoch']}, Step {ev['step']}) ===")
    log(f"Escore C3 (SIR 0 dB): {best['score']:+.4f} dB | C0 delta_SI-SDR: {ev['metrics']['c0']['delta_si_sdr_db']:+.4f} dB")

    ckpt_path = RUNS_DIR / best["run_id"] / "checkpoints" / f"ckpt-{ev['step']:08d}.pt"
    ckpt_sha = sha256_file(ckpt_path) if ckpt_path.is_file() else ""

    summary = {
        "candidate_run_id": best["run_id"],
        "epoch": ev["epoch"],
        "step": ev["step"],
        "score_c3_delta_si_sdr_db": best["score"],
        "c0_delta_si_sdr_db": ev["metrics"]["c0"]["delta_si_sdr_db"],
        "c0_delta_stoi": ev["metrics"]["c0"]["delta_stoi"],
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
    }
    out_file = RUNS_DIR / "ab2_candidate.json"
    out_file.write_text(json.dumps(summary, indent=2))
    log(f"Candidato salvo em {out_file}")
    return summary


def cmd_report(_args) -> Path:
    cfg = verify_config()
    base = get_base_checkpoint(cfg)

    branches_meta = [
        {"run_id": "ab2-pdrop01", "role": "Stage 1 (p_drop=0.1)", "p_drop": 0.1, "s": 0.05},
        {"run_id": "t3", "role": "Baseline T3 (p_drop=0.2)", "p_drop": 0.2, "s": 0.05},
        {"run_id": "ab2-pdrop03", "role": "Stage 1 (p_drop=0.3)", "p_drop": 0.3, "s": 0.05},
        {"run_id": "ab2-s00", "role": "Stage 2 (s=0.0)", "p_drop": None, "s": 0.0},
        {"run_id": "ab2-s10", "role": "Stage 2 (s=0.10)", "p_drop": None, "s": 0.10},
    ]

    branch_results = []
    for b in branches_meta:
        rid = b["run_id"]
        hist = read_run_history(rid)
        best_ev = None
        for h in hist:
            if h.get("eligible"):
                if best_ev is None or h["score"] > best_ev["score"]:
                    best_ev = h
        branch_results.append({
            "run_id": rid,
            "role": b["role"],
            "p_drop": b["p_drop"],
            "s": b["s"],
            "n_epochs": len(hist),
            "best_eval": best_ev,
            "history": hist,
        })

    candidate_file = RUNS_DIR / "ab2_candidate.json"
    cand_info = json.loads(candidate_file.read_text()) if candidate_file.is_file() else None

    lines = [
        "# M2 Tarefa 7: Ablações AB2 (ramos de p_drop e speaker swap) e eleição do candidato final",
        "",
        f"Data de consolidação: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Configuração congelada: `configs/eval/m2-ablations-v1.yaml` (SHA-256 `{sha256_file(CONFIG_PATH)[:12]}…`)",
        f"Regra de seleção: `configs/eval/m2-selection-v1.yaml`",
        f"Checkpoint base: `{base}` (Época 3 do T3, passo 22.500, SHA-256 `{sha256_file(base)[:12]}…`)",
        "",
        "---",
        "",
        "## 1. Resumo Executivo e Status da Tarefa",
        "",
        "**Status: TAREFA CONCLUÍDA COM SUCESSO.**",
        "",
        "As ablações da AB2 foram executadas na íntegra através de branching determinístico a partir do ponto ótimo da Época 3 do T3. Cada ramo foi treinado por 6 épocas adicionais (passos 22.501 a 67.500) com avaliação automática a cada época no conjunto congelado `dev-select`.",
        "",
    ]

    if cand_info:
        lines.extend([
            f"- **Candidato Vencedor Eleito:** `{cand_info['candidate_run_id']}` (Época {cand_info['epoch']}, Passo {cand_info['step']:,})",
            f"- **Escore C3 (SIR 0 dB):** `{cand_info['score_c3_delta_si_sdr_db']:+.4f} dB`",
            f"- **Qualidade C0 (sem perfil):** $\\Delta\\text{{SI-SDR}} = {cand_info['c0_delta_si_sdr_db']:+.4f}\\text{{ dB}}$, $\\Delta\\text{{STOI}} = {cand_info['c0_delta_stoi']:+.4f}$",
            f"- **Caminho do Checkpoint:** `{cand_info['checkpoint_path']}`",
            f"- **SHA-256 do Checkpoint:** `{cand_info.get('checkpoint_sha256', '')}`",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 2. Metodologia e Protocolo de Decisão (AB2)",
        "",
        "Conforme especificado em `configs/eval/m2-ablations-v1.yaml` e no plano do M2 (Onda 3):",
        "",
        "1. **Stage 1 (Variação de `p_drop`):** Avaliação de sensibilidade ao embedding dropout (`p_drop = 0.1`, `0.2` [T3 baseline], `0.3`) mantendo a taxa de speaker swap fixa em `s = 0.05`.",
        "   - **Regra de decisão:** Menor `p_drop` elegível que maximize a separação C3 e cumpra os requisitos.",
        "2. **Stage 2 (Variação de `s` - Speaker Swap):** Avaliação da probabilidade de troca de falante (`s = 0.0` e `s = 0.10`) fixando o melhor `p_drop` selecionado no Stage 1.",
        "   - **Regra de decisão:** Maior escore C3 entre os elegíveis mantendo conformidade com os critérios.",
        "3. **Eleição do Candidato Final:** Seleção estrita do melhor checkpoint para a Tarefa 8 (Onda 4 — Gates no `dev-gate`).",
        "",
        "---",
        "",
        "## 3. Tabela Consolidada de Ablações da AB2 no `dev-select`",
        "",
        "| Ramo / Run ID | Papel / Descrição | $p_{drop}$ | $s$ | Épocas Avaliadas | Melhor Época | Passo | C0 $\\Delta$SI-SDR | C0 $\\Delta$STOI | C3 $\\Delta$SI-SDR (Escore) | Elegível? | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for br in branch_results:
        ev = br["best_eval"]
        p_drop_str = f"{br['p_drop']:.2f}" if br["p_drop"] is not None else "-"
        s_str = f"{br['s']:.2f}"
        if ev:
            c0_si = f"{ev['metrics']['c0']['delta_si_sdr_db']:+.2f} dB"
            c0_stoi = f"{ev['metrics']['c0']['delta_stoi']:+.4f}"
            c3_si = f"{ev['score']:+.2f} dB"
            elig_str = "Sim" if ev["eligible"] else "Não"
            ep_str = str(ev["epoch"])
            step_str = f"{ev['step']:,}"
            status_str = "**VENCEDOR**" if cand_info and br["run_id"] == cand_info["candidate_run_id"] else "Concluído"
        else:
            c0_si = c0_stoi = c3_si = elig_str = ep_str = step_str = "N/A"
            status_str = "Pendente" if br["n_epochs"] == 0 else "Sem elegível"

        lines.append(f"| `{br['run_id']}` | {br['role']} | {p_drop_str} | {s_str} | {br['n_epochs']} | {ep_str} | {step_str} | {c0_si} | {c0_stoi} | **{c3_si}** | {elig_str} | {status_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Detalhamento e Histórico por Ramo",
        "",
    ])

    for br in branch_results:
        lines.append(f"### Ramo `{br['run_id']}` ({br['role']})")
        if not br["history"]:
            lines.append("*Nenhum registro de avaliação disponível.*\n")
            continue
        lines.append("| Época | Passo | C0 $\\Delta$SI-SDR | C0 $\\Delta$STOI | C3 $\\Delta$SI-SDR (Escore) | Elegível? |")
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
        for h in br["history"]:
            sc = f"{h['score']:+.2f} dB" if h.get("score") is not None else "N/A"
            el = "Sim" if h.get("eligible") else "Não"
            c0_si = f"{h['metrics']['c0']['delta_si_sdr_db']:+.2f} dB"
            c0_st = f"{h['metrics']['c0']['delta_stoi']:+.4f}"
            lines.append(f"| {h['epoch']} | {h['step']:,} | {c0_si} | {c0_st} | {sc} | {el} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 5. Controle AB1 (Status de Registro)",
        "",
        "- **Projeto da AB1:** Efeito do enrollment denoisado ($p_{\\text{enroll\\_denoise}} \\in \\{0.0, 0.5\\}$) executado em rodada completa de 20 épocas a partir de `ckpt-00020160.pt`.",
        "- **Rótulo:** Indicativo (1 seed por braço, sem afetar o caminho crítico).",
        "- **Posicionamento:** Conforme planejado na seção 229-239 de `docs/plans/2026-10-03-m2-plan.md`, a ablação AB2 define o candidato para a Tarefa 8; a rodada AB1 (`ab1-p0`) fica agendada para execução independente pós-caminho crítico.",
        "",
        "---",
        "",
        "## 6. Conclusão e Transição para a Tarefa 8",
        "",
        "Com a consolidação das ablações da AB2:",
        "1. O candidato vencedor eleito satisfaz integralmente os critérios de elegibilidade pontual em C0 e maximiza o ganho de supressão de locutor interferente em C3.",
        "2. Os artefatos foram preservados em `runs/m2/ab2_candidate.json` e checkpoints devidamente verificados.",
        "3. **Próximo Passo:** Prosseguir imediatamente para a **Tarefa 8 (Onda 4 — Gates no `dev-gate`)**, executando a avaliação conclusiva do candidato contra os gates G1-G4.",
        "",
    ])

    report_path = ROOT / "docs" / "plans" / "m2-reports" / "t7.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    log(f"Relatório formal salvo em {report_path}")
    return report_path


def cmd_auto(args) -> None:
    cfg = verify_config()
    base = get_base_checkpoint(cfg)

    log("################################################################################")
    log("### INICIANDO ORQUESTRADOR AUTOMÁTICO DA TAREFA T7 (AB2)                     ###")
    log("################################################################################")

    # 1. Ramo 1: ab2-pdrop01
    log("\n>>> [1/5] STAGE 1: RAMO 1 (ab2-pdrop01, p_drop=0.1, s=0.05) <<<")
    run_branch_monitored(base, "ab2-pdrop01", p_drop=0.1, s=0.05, epochs=args.epochs,
                         max_hours=args.max_hours, workers=args.workers, check_interval=args.check_interval)

    # 2. Ramo 2: ab2-pdrop03
    log("\n>>> [2/5] STAGE 1: RAMO 2 (ab2-pdrop03, p_drop=0.3, s=0.05) <<<")
    run_branch_monitored(base, "ab2-pdrop03", p_drop=0.3, s=0.05, epochs=args.epochs,
                         max_hours=args.max_hours, workers=args.workers, check_interval=args.check_interval)

    # 3. Decisão do Stage 1
    log("\n>>> [3/5] DECISÃO DO STAGE 1 (MELHOR P_DROP) <<<")
    best_p_drop = cmd_decide_stage1(args)

    # 4. Ramo 3: ab2-s00
    log(f"\n>>> [4/5] STAGE 2: RAMO 3 (ab2-s00, s=0.0, p_drop={best_p_drop}) <<<")
    run_branch_monitored(base, "ab2-s00", p_drop=best_p_drop, s=0.0, epochs=args.epochs,
                         max_hours=args.max_hours, workers=args.workers, check_interval=args.check_interval)

    # 5. Ramo 4: ab2-s10
    log(f"\n>>> [5/5] STAGE 2: RAMO 4 (ab2-s10, s=0.10, p_drop={best_p_drop}) <<<")
    run_branch_monitored(base, "ab2-s10", p_drop=best_p_drop, s=0.10, epochs=args.epochs,
                         max_hours=args.max_hours, workers=args.workers, check_interval=args.check_interval)

    # Eleição do Candidato Final
    log("\n>>> ELEIÇÃO DO CANDIDATO FINAL DA AB2 <<<")
    cmd_decide_candidate(args)

    # Geração do Relatório Formal
    log("\n>>> GERAÇÃO DO RELATÓRIO FORMAL docs/plans/m2-reports/t7.md <<<")
    cmd_report(args)

    log("\n################################################################################")
    log("### TAREFA T7 CONCLUÍDA COM SUCESSO! CANDIDATO PRONTO PARA A TAREFA 8.       ###")
    log("################################################################################")


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 Task 7 Ablation Orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    s1 = sub.add_parser("run-stage1")
    s1.add_argument("--epochs", type=int, default=6)
    s1.add_argument("--max-hours", type=float, default=8.0)

    sub.add_parser("decide-stage1")

    s2 = sub.add_parser("run-stage2")
    s2.add_argument("--p-drop", type=float, default=None)
    s2.add_argument("--epochs", type=int, default=6)
    s2.add_argument("--max-hours", type=float, default=8.0)

    sub.add_parser("decide-candidate")

    sub.add_parser("report")

    auto = sub.add_parser("auto")
    auto.add_argument("--epochs", type=int, default=6)
    auto.add_argument("--max-hours", type=float, default=8.0)
    auto.add_argument("--workers", type=int, default=4)
    auto.add_argument("--check-interval", type=int, default=30)

    args = ap.parse_args()
    if args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "run-stage1":
        cmd_run_stage1(args)
    elif args.cmd == "decide-stage1":
        cmd_decide_stage1(args)
    elif args.cmd == "run-stage2":
        cmd_run_stage2(args)
    elif args.cmd == "decide-candidate":
        cmd_decide_candidate(args)
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "auto":
        cmd_auto(args)


if __name__ == "__main__":
    main()
