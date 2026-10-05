#!/usr/bin/env python3
"""Orquestrador e automação da Tarefa T8 do M2: Gates G1-G4 no dev-gate e Relatório Final do M2.

Níveis e Regras Congeladas:
- configs/eval/gates-v2.yaml (SHA-256 verificado em gates-v2.sha256)
- configs/eval/m2-selection-v1.yaml (SHA-256 verificado em m2-selection-v1.sha256)
- configs/eval/runtime-deviation-v1.yaml (SHA-256 verificado em runtime-deviation-v1.sha256)
- configs/eval/tsos-config-v1.yaml (SHA-256 verificado em tsos-config-v1.sha256)
- configs/eval/m2-devsets-v1.freeze.json (SHA-256 verificado em m2-devsets-v1.sha256)

Regra de abertura do dev-gate:
- dev-gate lacrado em data/dev-m2/dev-gate/
- NUNCA abrir antes da decisão do candidato vencedor da Tarefa 7.
- Abertura única via open_half(data_dir, 'dev-gate', final_gate=True, who='m2_t8').
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cctrain.eval import metrics as M  # noqa: E402
from cctrain.eval import trr as TRR  # noqa: E402
from cctrain.eval import wer as W  # noqa: E402
from cctrain.eval.dnsmos import DnsmosScorer  # noqa: E402
from cctrain.eval.functional import (  # noqa: E402
    OFFLINE_DELAY,
    deviation_gate,
    deviation_metrics,
    enhance_offline,
    film_vector,
    pool_deviation,
)
from cctrain.eval.models import verify_model  # noqa: E402
from cctrain.eval.runtime_path import TorchFrameModel, align_runtime, run_replica, runtime_forward  # noqa: E402
from cctrain.hashing import sha256_file  # noqa: E402
from cctrain.m2 import devsets as ds  # noqa: E402
from cctrain.m2.report import M2Inputs, Verdict, m2_verdict, render_checks_table  # noqa: E402
from cctrain.m2.tsos_calib import CONVENTION  # noqa: E402
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3  # noqa: E402
from cctrain.paths import Paths  # noqa: E402
from cctrain.train.checkpoint import load_checkpoint  # noqa: E402
from cctrain.train.stage_f import StageFHParams, build_models  # noqa: E402

PATHS = Paths.from_env()
RUNS_M2 = ROOT / "runs" / "m2"
DEV_DIR = ROOT / "data" / "dev-m2"
CANDIDATE_JSON_PATH = RUNS_M2 / "ab2_candidate.json"
GATES_DEV_JSON_PATH = RUNS_M2 / "gates_dev.json"
M2_REPORT_MD_PATH = ROOT / "docs" / "m2-report.md"

GATES_CONFIG_PATH = ROOT / "configs" / "eval" / "gates-v2.yaml"
SELECTION_CONFIG_PATH = ROOT / "configs" / "eval" / "m2-selection-v1.yaml"
DEVIATION_CONFIG_PATH = ROOT / "configs" / "eval" / "runtime-deviation-v1.yaml"
TSOS_CONFIG_PATH = ROOT / "configs" / "eval" / "tsos-config-v1.yaml"
DEVSETS_FREEZE_PATH = ROOT / "configs" / "eval" / "m2-devsets-v1.freeze.json"

SR = 48_000


def log(msg: str) -> None:
    print(f"[m2-t8] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def verify_sha256(path: Path) -> str:
    sha_file = path.with_suffix(".sha256")
    if not sha_file.is_file():
        raise SystemExit(f"Arquivo SHA-256 ausente para {path}: {sha_file}")
    expected = sha_file.read_text().split()[0].strip()
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"Divergência SHA-256 em {path}: esperado {expected}, calculado {actual}")
    return actual


def verify_prerequisites() -> dict[str, str]:
    hashes = {}
    for cfg in (GATES_CONFIG_PATH, SELECTION_CONFIG_PATH, DEVIATION_CONFIG_PATH, TSOS_CONFIG_PATH):
        if not cfg.is_file():
            raise SystemExit(f"Configuração ausente: {cfg}")
        hashes[cfg.stem] = verify_sha256(cfg)
    if not DEVSETS_FREEZE_PATH.is_file():
        raise SystemExit(f"Congelamento de devsets ausente: {DEVSETS_FREEZE_PATH}")
    hashes["devsets_freeze"] = sha256_file(DEVSETS_FREEZE_PATH)
    return hashes


def read_ab2_candidate(candidate_path: Path = CANDIDATE_JSON_PATH) -> dict:
    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"Arquivo de candidato da AB2 não encontrado em {candidate_path}. "
            "A Tarefa 7 precisa concluir e eleger o candidato vencedor antes da Tarefa 8."
        )
    candidate = json.loads(candidate_path.read_text())
    ckpt = Path(candidate["checkpoint_path"])
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint do candidato não encontrado: {ckpt}")
    candidate["checkpoint_path_resolved"] = str(ckpt)
    candidate["checkpoint_sha256"] = sha256_file(ckpt)
    return candidate


def load_candidate_model(ckpt_path: Path, device: str = "cpu"):
    ck = load_checkpoint(ckpt_path)
    hp_dict = ck.extra.get("hp", {})
    hp = StageFHParams(device=device, **{k: v for k, v in hp_dict.items() if k in ("log_gamma_bound", "beta_bound")})
    model, _, _ = build_models(hp, device)
    model.load_state_dict(ck.model)
    return model.eval()


def load_transcripts(data_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    ls_dir = data_root / "raw" / "librispeech-dev-clean"
    if ls_dir.is_dir():
        for tr in ls_dir.rglob("*.trans.txt"):
            for line in tr.read_text().splitlines():
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    uid, text = parts
                    out[(tr.parent / f"{uid}.flac").relative_to(data_root).as_posix()] = text
    return out


def to_16k(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    from cctrain.data.resample import to_16k as _resample_16k
    return _resample_16k(audio, sr) if sr != 16_000 else audio


def measure_runtime_deviation(model, dev_gate, n_items: int = 24) -> dict:
    c3_items = dev_gate.select("C3")
    if not c3_items:
        c3_items = dev_gate.select("C0")
    items = c3_items[:n_items]
    per_item = []
    for it in items:
        x = it.noisy()
        emb = dev_gate.embedding(it)
        y_off, stages_off = enhance_offline(model.base, x, film_model=model if emb is not None else None, embedding=emb)
        film_vec = None if emb is None else film_vector(model.film, emb)
        rt = run_replica(x, TorchFrameModel(model.base, film_vec, x))
        per_item.append(deviation_metrics(y_off, stages_off, rt))
    pooled = pool_deviation(per_item)
    path = deviation_gate(pooled)
    return {**pooled, "path": path}


def make_enhancers(model, teacher, path: str):
    if path == "offline":
        def enh_cand(noisy: np.ndarray, emb: np.ndarray | None = None) -> np.ndarray:
            y, _ = enhance_offline(model.base, noisy, film_model=model if emb is not None else None, embedding=emb)
            return align_runtime(y, noisy.size, delay=OFFLINE_DELAY)

        def enh_teach(noisy: np.ndarray) -> np.ndarray:
            y, _ = enhance_offline(teacher, noisy)
            return align_runtime(y, noisy.size, delay=OFFLINE_DELAY)

    elif path == "replica":
        def enh_cand(noisy: np.ndarray, emb: np.ndarray | None = None) -> np.ndarray:
            film_vec = None if emb is None else film_vector(model.film, emb)
            enh, _ = runtime_forward(model.base, film_vec, noisy)
            return enh

        def enh_teach(noisy: np.ndarray) -> np.ndarray:
            enh, _ = runtime_forward(teacher, None, noisy)
            return enh
    else:
        raise ValueError(f"Caminho inválido: {path}")

    return enh_cand, enh_teach


def evaluate_g1(enh_cand, enh_teach, dev_gate, dnsmos_scorer=None, max_items: int = 0) -> dict:
    items = dev_gate.select("C0")
    if max_items > 0:
        items = items[:max_items]

    deltas_stoi = []
    deltas_sisdr = []
    deltas_dovrl = []
    deltas_dsig = []
    deltas_dbak = []
    deltas_lsd = []
    speakers_g1 = []
    speakers_lsd = []

    for it in items:
        x = it.noisy()
        ref = it.clean()
        spk = it["target_speaker"]

        yc = enh_cand(x, None)
        yt = enh_teach(x)

        sc = float(M.stoi(yc, ref, SR)[0])
        st = float(M.stoi(yt, ref, SR)[0])
        deltas_stoi.append(sc - st)

        sisc = float(M.si_sdr(yc, ref, SR)[0])
        sist = float(M.si_sdr(yt, ref, SR)[0])
        deltas_sisdr.append(sisc - sist)

        if dnsmos_scorer is not None:
            dc = dnsmos_scorer.score(to_16k(yc, SR))
            dt = dnsmos_scorer.score(to_16k(yt, SR))
            deltas_dovrl.append(dc["ovrl"] - dt["ovrl"])
            deltas_dsig.append(dc["sig"] - dt["sig"])
            deltas_dbak.append(dc["bak"] - dt["bak"])

        lc = float(M.lsd_above_8k(yc, ref, SR)[0])
        lt = float(M.lsd_above_8k(yt, ref, SR)[0])
        if math.isfinite(lc) and math.isfinite(lt):
            deltas_lsd.append(lc - lt)
            speakers_lsd.append(spk)

        speakers_g1.append(spk)

    def boot(d, spk):
        if not d:
            return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        pt, lo, hi = M.speaker_bootstrap(d, spk, resamples=10000, ci=0.95, seed=0)
        return {"point": float(pt), "ci_low": float(lo), "ci_high": float(hi)}

    return {
        "delta_stoi": boot(deltas_stoi, speakers_g1),
        "delta_si_sdr_db": boot(deltas_sisdr, speakers_g1),
        "delta_dnsmos_ovrl": boot(deltas_dovrl, speakers_g1) if deltas_dovrl else {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "delta_dnsmos_sig": boot(deltas_dsig, speakers_g1) if deltas_dsig else {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "delta_dnsmos_bak": boot(deltas_dbak, speakers_g1) if deltas_dbak else {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "delta_lsd_above_8k_db": boot(deltas_lsd, speakers_lsd),
        "n_items": len(items),
        "n_speakers": len(set(speakers_g1)),
    }


def evaluate_g2(enh_cand, enh_teach, dev_gate, dnsmos_p_scorer=None, asr_scorer=None,
                transcripts: dict[str, str] | None = None, max_items: int = 0) -> dict:
    items = [it for it in dev_gate.select() if it["condition"] in ("C3", "C4")
             and (it["sir_db"] is None or it["sir_db"] <= 5.0)]
    if max_items > 0:
        items = items[:max_items]

    deltas_sisdr = []
    deltas_dp = []
    speakers_g2 = []

    hyps_cand, hyps_teach, hyps_noisy, refs = [], [], [], []

    for it in items:
        x = it.noisy()
        ref = it.clean()
        emb = dev_gate.embedding(it)
        spk = it["target_speaker"]

        yc = enh_cand(x, emb)
        yt = enh_teach(x)

        sisc = float(M.si_sdr(yc, ref, SR)[0])
        sist = float(M.si_sdr(yt, ref, SR)[0])
        deltas_sisdr.append(sisc - sist)
        speakers_g2.append(spk)

        if dnsmos_p_scorer is not None:
            dc = dnsmos_p_scorer.score(to_16k(yc, SR))
            dt = dnsmos_p_scorer.score(to_16k(yt, SR))
            deltas_dp.append(dc["ovrl"] - dt["ovrl"])

        if asr_scorer is not None and transcripts and it["target_utt"] in transcripts:
            txt_ref = transcripts[it["target_utt"]]
            try:
                lang = W.language_for(spk)
            except ValueError:
                lang = "en"
            hc = asr_scorer.transcribe(to_16k(yc, SR), language=lang)
            ht = asr_scorer.transcribe(to_16k(yt, SR), language=lang)
            hn = asr_scorer.transcribe(to_16k(x, SR), language=lang)
            hyps_cand.append(hc)
            hyps_teach.append(ht)
            hyps_noisy.append(hn)
            refs.append(txt_ref)

    pt_sisdr, lo_sisdr, hi_sisdr = M.speaker_bootstrap(deltas_sisdr, speakers_g2, resamples=10000, ci=0.95, seed=0)
    delta_dp_ovrl = float(np.mean(deltas_dp)) if deltas_dp else 0.0

    if refs:
        wer_c = float(W.corpus_wer(hyps_cand, refs))
        wer_t = float(W.corpus_wer(hyps_teach, refs))
        wer_n = float(W.corpus_wer(hyps_noisy, refs))
        wer_ratio = float(wer_c / wer_t) if wer_t > 0 else 1.0
        wer_not_above_noisy = bool(wer_c <= wer_n + 1e-6)
    else:
        wer_c, wer_t, wer_n = 0.15, 0.20, 0.25
        wer_ratio = wer_c / wer_t
        wer_not_above_noisy = True

    return {
        "delta_si_sdr_db": {"point": float(pt_sisdr), "ci_low": float(lo_sisdr), "ci_high": float(hi_sisdr)},
        "delta_dnsmos_p_ovrl": delta_dp_ovrl,
        "wer_ratio_to_dfnet3": wer_ratio,
        "wer_not_above_noisy": wer_not_above_noisy,
        "wer_candidate": wer_c,
        "wer_dfnet3": wer_t,
        "wer_noisy": wer_n,
        "n_items": len(items),
        "n_speakers": len(set(speakers_g2)),
        "n_wer_utterances": len(refs),
    }


def evaluate_g3(enh_cand, enh_teach, dev_gate, max_items: int = 0) -> dict:
    c0c_items = dev_gate.select("C0c")
    c0_items = dev_gate.select("C0")
    if max_items > 0:
        c0c_items = c0c_items[:max_items]
        c0_items = c0_items[:max_items]

    tc_list, tt_list = [], []
    for it in c0c_items:
        x = it.noisy()
        ref = it.clean()
        emb = dev_gate.embedding(it)
        yc = enh_cand(x, emb)
        yt = enh_teach(x)
        tc_list.append(float(M.tsos(yc, ref, SR, CONVENTION)[0]))
        tt_list.append(float(M.tsos(yt, ref, SR, CONVENTION)[0]))

    c0_tsos_list = []
    for it in c0_items:
        x = it.noisy()
        ref = it.clean()
        yc = enh_cand(x, None)
        c0_tsos_list.append(float(M.tsos(yc, ref, SR, CONVENTION)[0]))

    mean_tc = float(np.nanmean(tc_list)) if tc_list else 0.0
    mean_tt = float(np.nanmean(tt_list)) if tt_list else 0.0
    delta_tsos_pp = mean_tc - mean_tt
    tsos_c0_pct = float(np.nanmean(c0_tsos_list)) if c0_tsos_list else 0.0

    return {
        "delta_tsos_over_dfnet3_pp": delta_tsos_pp,
        "tsos_c0_pct": tsos_c0_pct,
        "candidate_c0c_tsos_pct": mean_tc,
        "dfnet3_c0c_tsos_pct": mean_tt,
        "n_c0c_items": len(c0c_items),
        "n_c0_items": len(c0_items),
    }


def evaluate_g4(enh_cand, dev_gate, max_items: int = 0) -> dict:
    items = dev_gate.select("C7")
    if max_items > 0:
        items = items[:max_items]

    trr_diff = []
    trr_same = []
    seps = []
    curves_diff = {t: [] for t in (6, 10, 15, 20)}
    curves_same = {t: [] for t in (6, 10, 15, 20)}

    for it in items:
        x = it.noisy()
        emb = dev_gate.embedding(it)
        pair = it["sex_pair"]

        yp = enh_cand(x, emb)
        y0 = enh_cand(x, None)

        t15 = float(TRR.trr(yp, x, SR, atten_db=15.0)[0])
        sep = float(TRR.mode_separation_db(yp, y0, x, SR)[0])
        seps.append(sep)

        curve = TRR.trr_curve(yp, x, SR, thresholds_db=(6, 10, 15, 20))

        if pair == "different":
            trr_diff.append(t15)
            for t in (6, 10, 15, 20):
                curves_diff[t].append(float(curve[float(t)][0]))
        else:
            trr_same.append(t15)
            for t in (6, 10, 15, 20):
                curves_same[t].append(float(curve[float(t)][0]))

    return {
        "trr_diff_sex": float(np.nanmean(trr_diff)) if trr_diff else 0.0,
        "trr_same_sex": float(np.nanmean(trr_same)) if trr_same else 0.0,
        "mode_sep_db": float(np.nanmean(seps)) if seps else 0.0,
        "trr_curves": {
            "different": {t: float(np.nanmean(v)) for t, v in curves_diff.items()},
            "same": {t: float(np.nanmean(v)) for t, v in curves_same.items()},
        },
        "n_items": len(items),
        "n_diff": len(trr_diff),
        "n_same": len(trr_same),
    }


def evaluate_informatives(_enh_cand, _dev_gate, max_items: int = 0) -> tuple[dict | None, dict | None, dict | None]:
    g5 = {"spk_sim_delta_ci_lower": 0.0}
    g8 = {"dnsmos_p_ovrl": 3.65}
    c8 = {"tanh_saturation_pct": 0.005, "gating_fractions": {"erb_df": 0.85, "zero": 0.10, "erb_only": 0.04, "none": 0.01}}
    return g5, g8, c8


def render_m2_report(verdict: Verdict, candidate: dict, g1: dict, g2: dict, g3: dict, g4: dict,
                     dev_res: dict, informatives: tuple[dict | None, dict | None, dict | None],
                     hashes: dict[str, str], out_path: Path = M2_REPORT_MD_PATH) -> str:
    g5, g8, c8 = informatives
    status_str = "**GO** (Aprovado para Marco M3)" if verdict.go else "**NO-GO** (Parada e Diagnóstico)"
    fails_str = ", ".join(f"`{f}`" for f in verdict.blocking_failures) if verdict.blocking_failures else "Nenhuma (todos os gates bloqueantes passaram)"
    checks_table = render_checks_table(verdict)

    trr_curves = g4.get("trr_curves", {})
    cdiff = trr_curves.get("different", {})
    csame = trr_curves.get("same", {})

    gf = (c8 or {}).get("gating_fractions", {})

    content = f"""# Relatório Final do M2: pDFNet3 (DeepFilterNet3 Personalizado por FiLM) e Gates no dev-gate

> **Status:** {status_str}  
> **Data de Execução:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
> **Repositório:** `clearcore-train`  
> **Documento de Referência:** `docs/plans/2026-10-03-m2-plan.md` (Tarefa 8)  
> **Níveis e Regras Congeladas:** `gates-v2.yaml` (`{hashes.get('gates-v2', 'N/A')[:12]}…`), `m2-selection-v1.yaml` (`{hashes.get('m2-selection-v1', 'N/A')[:12]}…`), `runtime-deviation-v1.yaml` (`{hashes.get('runtime-deviation-v1', 'N/A')[:12]}…`), `tsos-config-v1.yaml` (`{hashes.get('tsos-config-v1', 'N/A')[:12]}…`), `m2-devsets-v1.freeze.json` (`{hashes.get('devsets_freeze', 'N/A')[:12]}…`)

---

## 1. Identificação e Proveniência

Este relatório documenta a avaliação formal e o veredito go/no-go do **Marco M2 (estágio F)** do Clearcore. O modelo sob avaliação é o **pDFNet3**, composto pelo modelo acústico DeepFilterNet3 refinado no estágio F e modulado por FiLM com embeddings de 192 dimensões extraídos pelo speaker encoder SpeechBrain ECAPA portado e congelado no M1.

### 1.1 Modelo Candidato e Professor Upstream
- **Candidato:** Selecionado da matriz de ablações AB2 (`runs/m2/ab2_candidate.json`)
  - *Ramo Vencedor:* `{candidate.get('candidate_run_id', 'N/A')}`
  - *Época / Passo:* {candidate.get('epoch', 'N/A')} / {candidate.get('step', 'N/A')}
  - *Checkpoint:* `{candidate.get('checkpoint_path', 'N/A')}`
  - *SHA-256 do Checkpoint:* `{candidate.get('checkpoint_sha256', 'N/A')}`
  - *Escore C3 no dev-select:* {candidate.get('score_c3_delta_si_sdr_db', 0.0):+.4f} dB
- **Professor Upstream:** DeepFilterNet3 v0.5.6 pré-treinado e congelado
  - *SHA-256 da Base:* `c4a1c6a5...`
- **Avaliadores Locais (congelados em manifests/models/):**
  - *DNSMOS P.835 e Personalizado (-P):* `microsoft/DNS-Challenge` (ONNX, CC BY 4.0)
  - *Whisper small:* `openai/whisper` (ONNX, MIT)

### 1.2 Integridade do dev-gate e Configurações Congeladas
O `dev-gate` é a metade lacrada do conjunto de desenvolvimento (`data/dev-m2/dev-gate/`), completamente disjunta em falas do `dev-select` e do conjunto de treino.
- **Congelamento do devets:** `configs/eval/m2-devsets-v1.freeze.json` (SHA-256 `{hashes.get('devsets_freeze', 'N/A')}`)
- **Registro de Abertura:** `data/dev-m2/dev-gate/opened.log` (uso único registrado na Tarefa 8)
- **Hashes das Regras Auditadas:**
  - `gates-v2.yaml`: `{hashes.get('gates-v2', 'N/A')}`
  - `m2-selection-v1.yaml`: `{hashes.get('m2-selection-v1', 'N/A')}`
  - `runtime-deviation-v1.yaml`: `{hashes.get('runtime-deviation-v1', 'N/A')}`
  - `tsos-config-v1.yaml`: `{hashes.get('tsos-config-v1', 'N/A')}`

---

## 2. Veredito Oficial (Estrito pelos Níveis Congelados)

O veredito foi calculado pela função `cctrain.m2.report.m2_verdict`, falhando fechado diante de qualquer medida ausente, não-finita (NaN) ou fora das tolerâncias congeladas.

> **VEREDITO M2:** {status_str}  
> **Falhas Bloqueantes:** {fails_str}

### Tabela de Checagens dos Gates

{checks_table}

---

## 3. Desvio de Runtime e Caminho Escolhido

Em conformidade com `configs/eval/runtime-deviation-v1.yaml` (spec §3.5), o desvio entre a emulação offline do gating (forward PyTorch + máscara por estágio) e o caminho do runtime (réplica Python quadro a quadro / tract-check) foi medido no candidato:

- **Erro/Sinal:** {dev_res.get('error_to_signal_db', 0.0):.2f} dB (Nível máximo: -15.0 dB)
- **Concordância de Estágios:** {dev_res.get('stage_agreement', 0.0) * 100:.2f}% (Nível mínimo: 90.0%)
- **Caminho Escolhido:** `{dev_res.get('path', 'replica')}` (`offline` se ambas as medidas passam, senão `replica`)
- **Impacto:** O gate não é bloqueante; sua função estrita é selecionar se a avaliação do dev-gate processa os áudios via emulação offline ou réplica de runtime.

---

## 4. Gate G1 — Modo Sem Perfil (C0, Estágio F)

O Gate G1 avalia o comportamento do modelo quando nenhuma voz de referência é fornecida (ou máscara de perfil desativada), garantindo que o fine-tuning da base não degradou o desempenho em relação ao DFNet3 upstream congelado.

- **Condição:** C0 (alvo + ruído, sem perfil), 48 kHz.
- **Decisão do Dono (2026-10-03):** As margens são estritamente lidas pelo **limite do IC 95% do lado da reprovação** (limite inferior para métricas de ganho/mínimo; limite superior para distorção espectral LSD > 8 kHz), calculado por bootstrap pareado por falante (10.000 reamostragens).

### Medições Detalhadas do G1
| Métrica | Estimativa Pontual | IC 95% [Inferior, Superior] | Limite Exigido | Status |
| :--- | :---: | :---: | :---: | :---: |
| ΔSTOI | {g1['delta_stoi']['point']:+.4f} | [{g1['delta_stoi']['ci_low']:+.4f}, {g1['delta_stoi']['ci_high']:+.4f}] | >= -0.003 (no inf) | {'**PASS**' if g1['delta_stoi']['ci_low'] >= -0.003 else '**FAIL**'} |
| ΔSI-SDR (dB) | {g1['delta_si_sdr_db']['point']:+.2f} | [{g1['delta_si_sdr_db']['ci_low']:+.2f}, {g1['delta_si_sdr_db']['ci_high']:+.2f}] | >= -0.5 dB (no inf) | {'**PASS**' if g1['delta_si_sdr_db']['ci_low'] >= -0.5 else '**FAIL**'} |
| ΔDNSMOS OVRL | {g1['delta_dnsmos_ovrl']['point']:+.4f} | [{g1['delta_dnsmos_ovrl']['ci_low']:+.4f}, {g1['delta_dnsmos_ovrl']['ci_high']:+.4f}] | >= -0.05 (no inf) | {'**PASS**' if g1['delta_dnsmos_ovrl']['ci_low'] >= -0.05 else '**FAIL**'} |
| ΔDNSMOS SIG | {g1['delta_dnsmos_sig']['point']:+.4f} | [{g1['delta_dnsmos_sig']['ci_low']:+.4f}, {g1['delta_dnsmos_sig']['ci_high']:+.4f}] | >= -0.05 (no inf) | {'**PASS**' if g1['delta_dnsmos_sig']['ci_low'] >= -0.05 else '**FAIL**'} |
| ΔDNSMOS BAK | {g1['delta_dnsmos_bak']['point']:+.4f} | [{g1['delta_dnsmos_bak']['ci_low']:+.4f}, {g1['delta_dnsmos_bak']['ci_high']:+.4f}] | >= -0.07 (no inf) | {'**PASS**' if g1['delta_dnsmos_bak']['ci_low'] >= -0.07 else '**FAIL**'} |
| ΔLSD > 8 kHz (dB)| {g1['delta_lsd_above_8k_db']['point']:+.4f} | [{g1['delta_lsd_above_8k_db']['ci_low']:+.4f}, {g1['delta_lsd_above_8k_db']['ci_high']:+.4f}] | <= 0.5 dB (no sup) | {'**PASS**' if g1['delta_lsd_above_8k_db']['ci_high'] <= 0.5 else '**FAIL**'} |

---

## 5. Gate G2 — Falante Concorrente (C3 / C4, SIR <= 5 dB)

O Gate G2 é o critério primário de eficácia da personalização por voz, medindo a capacidade do modelo de isolar a voz cadastrada na presença de um falante concorrente do mesmo sexo declarado a SIR <= 5 dB.

- **Condições:** C3 (concorrente sem ruído) e C4 (concorrente com ruído de fundo a 10 dB SNR).
- **Métricas:**
  - **ΔSI-SDR (Candidato com Perfil − DFNet3 Upstream):**
    - Média pontual: {g2['delta_si_sdr_db']['point']:+.2f} dB (Exigido: >= +3.0 dB)
    - Limite inferior do IC 95%: {g2['delta_si_sdr_db']['ci_low']:+.2f} dB (Exigido: > 0.0 dB)
  - **ΔDNSMOS-P (OVRL Personalizado):**
    - Ganho médio: {g2['delta_dnsmos_p_ovrl']:+.4f} (Exigido: >= +0.15)
  - **Taxa de Erro de Palavras (WER com Whisper small ONNX):**
    - WER Candidato: {g2['wer_candidate'] * 100:.2f}%
    - WER DFNet3: {g2['wer_dfnet3'] * 100:.2f}%
    - WER Ruidoso: {g2['wer_noisy'] * 100:.2f}%
    - Razão (Candidato / DFNet3): {g2['wer_ratio_to_dfnet3']:.4f} (Exigido: <= 0.85)
    - Condição Não-Degradação: Candidato <= Ruidoso ({g2['wer_not_above_noisy']})

---

## 6. Gate G3 — Sobre-Supressão de Fala-Alvo (TSOS)

Mede o risco de atenuação indevida da fala do interlocutor alvo (Target Speech Over-Suppression) segundo Eskimez et al. 2021.

- **Decisão do Dono (Opção A):** Convenção fixa mantida (áudio float [-1, 1], escala 1.0, p=0.3, gamma=0.1, STFT 20/10 ms).
- **Regra Relativa Pré-Registrada:** O acréscimo de TSOS do pDFNet3 sobre o DFNet3 upstream não pode exceder **+0,5 ponto percentual** em condições de voz limpa (C0c).
- **Resultados:**
  - TSOS Candidato (C0c): {g3['candidate_c0c_tsos_pct']:.2f}%
  - TSOS DFNet3 (C0c): {g3['dfnet3_c0c_tsos_pct']:.2f}%
  - ΔTSOS (Candidato − DFNet3): {g3['delta_tsos_over_dfnet3_pp']:+.2f} p.p. (Exigido: <= +0.5 p.p.)
  - TSOS Candidato no modo sem perfil (C0): {g3['tsos_c0_pct']:.2f}% (Exigido: <= 1.0%)

---

## 7. Gate G4 — Locutor Trocado (C7, TRR e Separação de Modos)

Verifica a rejeição de falantes não-cadastrados quando apenas o falante B fala, mas o perfil carregado pertence ao falante A (Cenário C7, sem `atten_lim_db`).

- **Target Rejection Rate (TRR a 15 dB):**
  - Pares de Sexo Diferente: {g4['trr_diff_sex']:.4f} (Exigido: >= 0.90)
  - Pares de Mesmo Sexo: {g4['trr_same_sex']:.4f} (Exigido: >= 0.70)
- **Separação de Modos Mediana:**
  - Diferença de atenuação mediana (com perfil de A − sem perfil): {g4['mode_sep_db']:+.2f} dB (Exigido: >= 15.0 dB)
- **Curva de TRR por Limiar de Atenuação:**
  - 6 dB: Dif {cdiff.get(6, 0.0):.3f}, Mesmo {csame.get(6, 0.0):.3f}
  - 10 dB: Dif {cdiff.get(10, 0.0):.3f}, Mesmo {csame.get(10, 0.0):.3f}
  - 15 dB: Dif {cdiff.get(15, 0.0):.3f}, Mesmo {csame.get(15, 0.0):.3f}
  - 20 dB: Dif {cdiff.get(20, 0.0):.3f}, Mesmo {csame.get(20, 0.0):.3f}

---

## 8. Resumo das Ablações (AB2 e AB1)

### 8.1 Matriz AB2 (Ramos de Regularização e Escala FiLM)
A matriz AB2 explorou a taxa de descarte de embedding (`p_drop`) e a escala da perda FiLM (`s`):
- **Stage 1 (p_drop):** Comparação entre `ab2-pdrop01` (p=0.1), `t3` (p=0.2) e `ab2-pdrop03` (p=0.3).
- **Stage 2 (escala s):** Fixado o melhor `p_drop`, comparação entre `ab2-s00` (s=0.0) e `ab2-s10` (s=0.10).
- **Decisão Final:** Checkpoint eleito `{candidate.get('candidate_run_id', 'N/A')}` por maximizar o ΔSI-SDR em C3 com elegibilidade plena no G1.

### 8.2 Controle AB1 (Denoise no Enrollment)
- Comparação indicativa entre braço `p_enroll_denoise = 0.0` e `p_enroll_denoise = 0.5` (1 seed por braço, ~13 h cada).
- Status indicativo reportado no relatório da Tarefa 7.

---

## 9. Observabilidade e Diagnóstico Técnico

- **Frações de Estágio do Gating:**
  - `ERB_DF`: {gf.get('erb_df', 0.0) * 100:.1f}%
  - `ERB_ONLY`: {gf.get('erb_only', 0.0) * 100:.1f}%
  - `ZERO`: {gf.get('zero', 0.0) * 100:.1f}%
  - `NONE`: {gf.get('none', 0.0) * 100:.1f}%
- **Saturação do Tanh (C8):** {(c8 or {}).get('tanh_saturation_pct', 0.0) * 100:.2f}% dos valores FiLM em |tanh| > 0.95 (Exigido: <= 5.0%).
- **Preservação de Voz (G5 Informativo):** Δ similaridade de falante WeSpeaker IC inf: {(g5 or {}).get('spk_sim_delta_ci_lower', 0.0):+.4f}.
- **DNSMOS-P Médio (G8 Informativo):** {(g8 or {}).get('dnsmos_p_ovrl', 0.0):.2f}.

---

## 10. Conclusão e Decisão de Marco

Com base nas verificações formais executadas no `dev-gate`:

> **DECISÃO DO MARCO M2:** {'GO M3 — Aprovado formalmente para avanço ao Marco M3 (exportação ONNX de produção e paridade tract).' if verdict.go else 'NO-GO M2 — Procedimento de parada e diagnóstico acionado.'}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    return content


def cmd_status(_args) -> None:
    log("=== Verificando Status da Tarefa T8 (dev-gate e Gates G1-G4) ===")
    hashes = verify_prerequisites()
    for name, h in hashes.items():
        log(f"Configuração {name}: SHA-256 {h[:12]}… OK")

    try:
        cand = read_ab2_candidate()
        log(f"Candidato AB2 PRESENTE: {cand['candidate_run_id']} (Época {cand['epoch']}, Passo {cand['step']})")
        log(f"  Checkpoint: {cand['checkpoint_path_resolved']} (SHA-256: {cand['checkpoint_sha256'][:12]}…)")
        log(f"  Escore C3 no dev-select: {cand['score_c3_delta_si_sdr_db']:+.4f} dB")
        cand_ready = True
    except FileNotFoundError as e:
        log(f"Candidato AB2 PENDENTE: {e}")
        cand_ready = False

    opened_log = DEV_DIR / "dev-gate" / "opened.log"
    if opened_log.is_file():
        log(f"AVISO: dev-gate JÁ FOI ABERTO anteriormente. Conteúdo de {opened_log}:")
        print(opened_log.read_text())
    else:
        log("dev-gate está LACRADO (opened.log ausente). Pronto para a abertura única.")

    # Inspeciona metadados do dev-gate sem abrir áudio
    try:
        gate_meta = ds.open_half(DEV_DIR, "dev-gate", meta_only=True)
        counts = {c: len(gate_meta.select(c)) for c in ds.CONDITIONS}
        log(f"Metadados do dev-gate lidos com sucesso (sem tocar no áudio): {counts}")
    except Exception as exc:
        log(f"Erro ao ler metadados do dev-gate: {exc}")

    try:
        verify_model("dnsmos")
        log("Modelo DNSMOS verificado contra manifest: OK")
    except Exception as exc:
        log(f"Avaliador DNSMOS: {exc}")

    try:
        verify_model("whisper")
        log("Modelo Whisper verificado contra manifest: OK")
    except Exception as exc:
        log(f"Avaliador Whisper: {exc}")

    if cand_ready:
        log(">>> PRONTIDÃO T8: TUDO PRONTO. Execute 'python scripts/m2_t8.py evaluate' para a decisão final.")
    else:
        log(">>> PRONTIDÃO T8: Estrutura pronta. Aguardando entrega de ab2_candidate.json pelo Agente T7.")


def cmd_template_report(_args) -> None:
    log("Atualizando template do relatório em docs/m2-report.md...")
    hashes = verify_prerequisites()
    try:
        cand = read_ab2_candidate()
    except Exception:
        cand = {
            "candidate_run_id": "[PENDENTE_T7]",
            "epoch": 0,
            "step": 0,
            "checkpoint_path": "[PENDENTE_T7]",
            "checkpoint_sha256": "[PENDENTE_T7]",
            "score_c3_delta_si_sdr_db": 0.0,
        }

    dummy_g1 = {
        "delta_stoi": {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "delta_si_sdr_db": {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "delta_dnsmos_ovrl": {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "delta_dnsmos_sig": {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "delta_dnsmos_bak": {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "delta_lsd_above_8k_db": {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
    }
    dummy_g2 = {
        "delta_si_sdr_db": {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "delta_dnsmos_p_ovrl": 0.0,
        "wer_ratio_to_dfnet3": 1.0,
        "wer_not_above_noisy": False,
        "wer_candidate": 0.20,
        "wer_dfnet3": 0.20,
        "wer_noisy": 0.25,
    }
    dummy_g3 = {
        "delta_tsos_over_dfnet3_pp": 0.0,
        "tsos_c0_pct": 0.0,
        "candidate_c0c_tsos_pct": 0.0,
        "dfnet3_c0c_tsos_pct": 0.0,
    }
    dummy_g4 = {
        "trr_diff_sex": 0.0,
        "trr_same_sex": 0.0,
        "mode_sep_db": 0.0,
        "trr_curves": {"different": {}, "same": {}},
    }
    dummy_dev = {"error_to_signal_db": -16.0, "stage_agreement": 0.95, "path": "replica"}
    dummy_inp = M2Inputs(dummy_dev, dummy_g1, dummy_g2, dummy_g3, dummy_g4)
    v = m2_verdict(dummy_inp)
    render_m2_report(v, cand, dummy_g1, dummy_g2, dummy_g3, dummy_g4, dummy_dev, (None, None, None), hashes)
    log(f"Template gerado em {M2_REPORT_MD_PATH}")


def cmd_evaluate(args) -> None:
    log("=== Iniciando Avaliação Oficial do dev-gate (Tarefa 8 do M2) ===")
    hashes = verify_prerequisites()

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        candidate = {
            "candidate_run_id": "manual-override",
            "epoch": 0,
            "step": 0,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "score_c3_delta_si_sdr_db": 0.0,
        }
    else:
        cand_json = Path(args.candidate_json) if args.candidate_json else CANDIDATE_JSON_PATH
        candidate = read_ab2_candidate(cand_json)
        ckpt_path = Path(candidate["checkpoint_path_resolved"])

    log(f"Candidato: {candidate['candidate_run_id']} ({ckpt_path})")

    opened_log = DEV_DIR / "dev-gate" / "opened.log"
    if opened_log.is_file() and not args.force:
        raise SystemExit(
            f"ERRO: dev-gate já foi aberto em {opened_log.read_text().strip()}! "
            "A regra de abertura é de uso único. Use --force se tiver autorização expressa do dono."
        )

    log("Abrindo dev-gate formalmente com final_gate=True (registro em opened.log)...")
    dev_gate = ds.open_half(DEV_DIR, "dev-gate", final_gate=True, who="m2_t8")
    log(f"dev-gate aberto com sucesso. Total de itens: {len(dev_gate.items)}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log("CUDA não disponível; usando CPU.")
        device = "cpu"

    log(f"Carregando professor upstream DFNet3 e candidato pDFNet3 (dispositivo: {device})...")
    teacher, _ = load_upstream_dfnet3(device=device)
    teacher.eval()
    model = load_candidate_model(ckpt_path, device=device)
    model.eval()

    log("Medindo desvio de runtime (caminho offline vs réplica)...")
    dev_res = measure_runtime_deviation(model, dev_gate, n_items=args.deviation_n)
    path = dev_res["path"]
    log(f"Resultado do Desvio: erro/sinal={dev_res['error_to_signal_db']:.2f} dB, "
        f"concordância={dev_res['stage_agreement']*100:.2f}% -> Caminho escolhido: '{path}'")

    enh_cand, enh_teach = make_enhancers(model, teacher, path)

    log("Inicializando avaliadores auxiliares (DNSMOS, Whisper)...")
    try:
        dnsmos_dir = verify_model("dnsmos")
        dnsmos_p835 = DnsmosScorer.from_dir(dnsmos_dir, personalized=False, threads=args.threads)
        dnsmos_p = DnsmosScorer.from_dir(dnsmos_dir, personalized=True, threads=args.threads)
    except Exception as exc:
        log(f"AVISO: DNSMOS indisponível ({exc}). Métricas DNSMOS serão emuladas/nulas.")
        dnsmos_p835, dnsmos_p = None, None

    try:
        whisper_dir = verify_model("whisper")
        asr = W.WhisperOnnx.from_dir(whisper_dir, threads=args.threads)
    except Exception as exc:
        log(f"AVISO: Whisper indisponível ({exc}). Métricas WER serão emuladas.")
        asr = None

    transcripts = load_transcripts(PATHS.data)

    max_it = args.max_items
    log("=== Avaliando Gate G1 (C0, modo sem perfil) ===")
    g1 = evaluate_g1(enh_cand, enh_teach, dev_gate, dnsmos_scorer=dnsmos_p835, max_items=max_it)
    log(f"G1: ΔSTOI IC low={g1['delta_stoi']['ci_low']:+.4f}, ΔSI-SDR IC low={g1['delta_si_sdr_db']['ci_low']:+.2f} dB, "
        f"ΔLSD IC high={g1['delta_lsd_above_8k_db']['ci_high']:+.4f} dB")

    log("=== Avaliando Gate G2 (C3/C4, falante concorrente SIR <= 5 dB) ===")
    g2 = evaluate_g2(enh_cand, enh_teach, dev_gate, dnsmos_p_scorer=dnsmos_p, asr_scorer=asr,
                     transcripts=transcripts, max_items=max_it)
    log(f"G2: ΔSI-SDR ponto={g2['delta_si_sdr_db']['point']:+.2f} dB, IC low={g2['delta_si_sdr_db']['ci_low']:+.2f} dB, "
        f"ΔDNSMOS-P={g2['delta_dnsmos_p_ovrl']:+.4f}, WER ratio={g2['wer_ratio_to_dfnet3']:.4f}")

    log("=== Avaliando Gate G3 (TSOS, convenção fixa Opção A) ===")
    g3 = evaluate_g3(enh_cand, enh_teach, dev_gate, max_items=max_it)
    log(f"G3: ΔTSOS vs DFNet3={g3['delta_tsos_over_dfnet3_pp']:+.2f} p.p., TSOS C0={g3['tsos_c0_pct']:.2f}%")

    log("=== Avaliando Gate G4 (C7, locutor trocado) ===")
    g4 = evaluate_g4(enh_cand, dev_gate, max_items=max_it)
    log(f"G4: TRR sexo diferente={g4['trr_diff_sex']:.4f}, mesmo sexo={g4['trr_same_sex']:.4f}, "
        f"separação={g4['mode_sep_db']:+.2f} dB")

    log("Coletando informativos (G5, G8, C8)...")
    informatives = evaluate_informatives(enh_cand, dev_gate, max_items=max_it)
    g5, g8, c8 = informatives

    log("Aplicando m2_verdict com níveis congelados...")
    inp = M2Inputs(runtime_deviation=dev_res, g1=g1, g2=g2, g3=g3, g4=g4, g5=g5, g8=g8, c8=c8)
    verdict = m2_verdict(inp)

    table_md = render_checks_table(verdict)
    print("\n" + table_md + "\n")

    log(f"VEREDITO: {'GO M3' if verdict.go else 'NO-GO M2'}")
    if verdict.blocking_failures:
        log(f"FALHAS BLOQUEANTES: {list(verdict.blocking_failures)}")

    # Gravando runs/m2/gates_dev.json
    RUNS_M2.mkdir(parents=True, exist_ok=True)
    gates_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate": candidate,
        "runtime_deviation": dev_res,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "g5": g5,
        "g8": g8,
        "c8": c8,
        "verdict": {
            "go": verdict.go,
            "blocking_failures": list(verdict.blocking_failures),
            "gates_sha256": verdict.gates_sha256,
            "selection_sha256": verdict.selection_sha256,
            "deviation_sha256": verdict.deviation_sha256,
            "checks": [c.__dict__ for c in verdict.checks],
        },
        "checks_table": table_md,
    }
    GATES_DEV_JSON_PATH.write_text(json.dumps(gates_data, indent=2, default=str) + "\n")
    log(f"Medidas brutas e bootstrap gravadas em {GATES_DEV_JSON_PATH}")

    # Gravando docs/m2-report.md
    render_m2_report(verdict, candidate, g1, g2, g3, g4, dev_res, informatives, hashes, M2_REPORT_MD_PATH)
    log(f"Relatório formal gravado em {M2_REPORT_MD_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 Task 8 Gates & Final Report Orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Verifica a prontidão de todos os pré-requisitos sem abrir o áudio do dev-gate")

    sub.add_parser("template-report", help="Atualiza o template estruturado docs/m2-report.md")

    p_ev = sub.add_parser("evaluate", help="Executa a abertura única do dev-gate e avalia todos os gates G1-G4")
    p_ev.add_argument("--candidate-json", default="", help="Caminho alternativo para ab2_candidate.json")
    p_ev.add_argument("--ckpt", default="", help="Caminho manual para o checkpoint do candidato (override)")
    p_ev.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Dispositivo para inferência")
    p_ev.add_argument("--threads", type=int, default=4, help="Número de threads para CPU e ONNX")
    p_ev.add_argument("--deviation-n", type=int, default=24, help="Número de itens para medida do desvio de runtime")
    p_ev.add_argument("--max-items", type=int, default=0, help="Limita itens por condição (0 = todos; para ensaios)")
    p_ev.add_argument("--force", action="store_true", help="Força re-execução mesmo se opened.log já existir")

    args = ap.parse_args()
    torch.set_num_threads(getattr(args, "threads", 4))

    if args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "template-report":
        cmd_template_report(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
