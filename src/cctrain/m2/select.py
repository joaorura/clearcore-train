"""Regra de seleção de checkpoint do M2 (`configs/eval/m2-selection-v1.yaml`) e a avaliação por época no `dev-select`.

Só lógica pura (elegibilidade, escore, empate, early stopping) e a medida offline com gating emulado; nenhuma leitura
de `dev-gate` ou de teste: o chamador passa os itens do `dev-select` (Tarefa 4). Convenções:

- épocas contadas de 1 (a avaliação da época `e` roda ao fim da `e`-ésima época);
- elegibilidade (C0, sem perfil, contra o DFNet3 upstream, estimativa pontual): ΔSI-SDR ≥ −0,5 dB e ΔSTOI ≥ −0,003;
- escore: ΔSI-SDR médio por falante em C3 (SIR 0 dB); entre elegíveis vence o maior; escores a menos de 0,05 dB do
  máximo empatam e vence a época mais antiga;
- early stopping (semântica de `min_delta` do Keras): o melhor escore elegível de referência só sobe quando uma
  avaliação o supera em pelo menos `min_improvement_db`; a partir da época `from_epoch`, cada avaliação sem essa melhora
  conta como "sem melhora", e `patience_evaluations` seguidas param o treino (antes de `from_epoch` só se atualiza o
  melhor, sem contar).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cctrain.m2.config import SelectionRule


@dataclass(frozen=True)
class SelectItem:
    """Um enunciado do `dev-select`. `condition`: "C0" (só ruído, sem perfil) ou "C3" (concorrente sem ruído, com
    perfil, SIR 0 dB). `noisy`/`clean` a 48 kHz; o atraso da síntese (1 hop) é compensado aqui."""

    condition: str
    noisy: np.ndarray
    clean: np.ndarray
    speaker: str
    embedding: np.ndarray | None = None


@dataclass(frozen=True)
class EpochEval:
    epoch: int
    step: int
    eligible: bool
    score: float | None
    metrics: Mapping


# ------------------------------------------------------------------------------------------------ regra
def is_eligible(metrics: Mapping, rule: SelectionRule) -> bool:
    c0, e = metrics["c0"], rule.eligibility
    return bool(c0["delta_si_sdr_db"] >= e.delta_si_sdr_db_min and c0["delta_stoi"] >= e.delta_stoi_min)


def score_of(metrics: Mapping) -> float:
    return float(metrics["c3"]["delta_si_sdr_db"])


def make_epoch_eval(epoch: int, step: int, metrics: Mapping, rule: SelectionRule) -> EpochEval:
    ok = is_eligible(metrics, rule)
    return EpochEval(epoch, step, ok, score_of(metrics) if ok else None, metrics)


def best_epoch(history: Sequence[EpochEval], rule: SelectionRule) -> EpochEval | None:
    """Elegível de maior escore; empate (< tolerância do máximo) vai para o mais antigo. None: nenhum elegível."""
    elig = [h for h in history if h.eligible]
    if not elig:
        return None
    top = max(h.score for h in elig)
    return min((h for h in elig if top - h.score < rule.tie.tolerance_db), key=lambda h: h.epoch)


def should_stop(history: Sequence[EpochEval], rule: SelectionRule) -> bool:
    es = rule.early_stopping
    best: float | None = None
    since = 0
    for h in sorted(history, key=lambda h: h.epoch):
        if h.eligible and (best is None or h.score >= best + es.min_improvement_db):
            best, since = h.score, 0
        elif h.epoch >= es.from_epoch:
            since += 1
            if since >= es.patience_evaluations:
                return True
    return False


# ------------------------------------------------------------------------------------------------ arquivos
def eval_path(run_dir: Path, epoch: int) -> Path:
    return Path(run_dir) / "eval" / f"epoch-{epoch:02d}.json"


def write_eval(run_dir: Path, ev: EpochEval) -> Path:
    path = eval_path(run_dir, ev.epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"epoch": ev.epoch, "step": ev.step, "eligible": ev.eligible, "score": ev.score,
                               "metrics": ev.metrics}, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def read_history(run_dir: Path) -> list[EpochEval]:
    out = []
    for p in sorted((Path(run_dir) / "eval").glob("epoch-*.json")):
        d = json.loads(p.read_text())
        out.append(EpochEval(int(d["epoch"]), int(d["step"]), bool(d["eligible"]), d["score"], d["metrics"]))
    return out


# ------------------------------------------------------------------------------------------------ medida
def _mean_by_speaker(rows: Sequence[tuple[str, float]]) -> tuple[float, int]:
    by: dict[str, list[float]] = defaultdict(list)
    for spk, v in rows:
        by[spk].append(v)
    return float(np.mean([np.mean(v) for v in by.values()])), len(by)


def _si_stoi(est: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    from cctrain.eval import metrics as M

    return float(M.si_sdr(est, ref, 48_000)[0]), float(M.stoi(est, ref, 48_000)[0])


def teacher_reference(items: Sequence[SelectItem], teacher: torch.nn.Module) -> list[tuple[float, float]]:
    """(SI-SDR, STOI) do DFNet3 upstream por item. Não muda entre épocas: calcule uma vez e reuse."""
    from cctrain.eval.functional import OFFLINE_DELAY, _aligned, enhance_offline

    out = []
    for it in items:
        y, _ = enhance_offline(teacher, it.noisy)
        out.append(_si_stoi(_aligned(y, OFFLINE_DELAY, len(it.clean)), it.clean))
    return out


def evaluate_select(model, items: Sequence[SelectItem], *, teacher: torch.nn.Module,
                    reference: Sequence[tuple[float, float]] | None = None) -> dict:
    """Métricas da regra de seleção: ΔSI-SDR e ΔSTOI do aluno (PDFNet3) contra o DFNet3 upstream.

    C0: aluno sem perfil (a base com FiLM identidade), ΔSI-SDR e ΔSTOI médios por falante. C3: aluno com perfil (mask=1,
    `embedding` do item), ΔSI-SDR médio por falante. Caminho offline com gating emulado. Deixa o modelo em `eval()`."""
    from cctrain.eval.functional import OFFLINE_DELAY, _aligned, enhance_offline

    ref = list(reference) if reference is not None else teacher_reference(items, teacher)
    model.eval()
    d_si: dict[str, list[tuple[str, float]]] = {"C0": [], "C3": []}
    d_stoi: list[tuple[str, float]] = []
    for it, (t_si, t_stoi) in zip(items, ref, strict=True):
        if it.condition == "C0":
            y, _ = enhance_offline(model.base, it.noisy)
        elif it.condition == "C3":
            if it.embedding is None:
                raise ValueError("item C3 sem embedding")
            y, _ = enhance_offline(model.base, it.noisy, film_model=model, embedding=it.embedding)
        else:
            raise ValueError(f"condição {it.condition!r}: esperado C0 ou C3")
        si, stoi = _si_stoi(_aligned(y, OFFLINE_DELAY, len(it.clean)), it.clean)
        d_si[it.condition].append((it.speaker, si - t_si))
        if it.condition == "C0":
            d_stoi.append((it.speaker, stoi - t_stoi))
    out: dict = {}
    for name, cond in (("c0", "C0"), ("c3", "C3")):
        if not d_si[cond]:
            raise ValueError(f"dev-select sem itens {cond}")
        mean, n_spk = _mean_by_speaker(d_si[cond])
        out[name] = {"delta_si_sdr_db": mean, "n_speakers": n_spk, "n_items": len(d_si[cond])}
    out["c0"]["delta_stoi"] = _mean_by_speaker(d_stoi)[0]
    return out


def make_evaluator(items: Sequence[SelectItem], teacher: torch.nn.Module):
    """`evaluator(model, epoch) -> métricas` para o trainer; a referência do DFNet3 upstream é calculada uma vez."""
    reference: list[tuple[float, float]] | None = None

    def evaluate(model, epoch: int) -> dict:
        nonlocal reference
        del epoch
        if reference is None:
            reference = teacher_reference(items, teacher)
        with torch.no_grad():
            return evaluate_select(model, items, teacher=teacher, reference=reference)

    return evaluate

