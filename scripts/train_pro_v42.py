#!/usr/bin/env python3
"""Treino do pDFNet3 Pro v4.2: o alvo sozinho não pode cair depois que o interferente para (C6), sem dano em N=0.

Diagnóstico (`runs/m4/quality-metrics/diag_c6.json`, `runs/m4/pro-v42/diag_c6_vad.json`): no v3.1 o ganho do alvo nos
quadros alvo-sozinho do usuário N=1 depende do tempo desde o último quadro com interferente (antes do 1º interferente
−3,6 dB; 0–0,5 s −15,9; 0,5–1 s −11,3; 1–2 s −6,9) e vem da máscara ERB/DF, não do gating. Com o VAD do interferente
em pico−45 dB e 100 ms após o fim excluídos, a queda diminui (0–0,5 s −9,6 dB) mas não some.

Partida: v3.1 (mesma arquitetura, gating novo `gate_v4`). Perdas (todas sobre a saída com o gating do runtime):
  L_mr + L_os + λ_leak·L_leak + λ_ovl·L_ovl + L_lsnr + L_t + L_dfreg            (como no v4-keep: ruído/reconstrução)
  + λ_turn·L_turn   ganho do alvo-SOMBRA (alvo limpo pela mesma máscara ERB, coeficientes DF e decisão ZERO tomados
                    sobre a mistura — a mesma vista do eval) por quadro, nos quadros alvo-sozinho (alvo ativo, sem
                    interferente, sem silêncio digital) das amostras com perfil e N≥1: hinge relu(h − g)/10 com h=−1,5 dB,
                    g limitado a −40 dB; peso por quadro = min(√(E_alvo/E_média ativa do item), 3) ×
                    (1 + (k−1)·max(0, 1 − t/T)), t = tempo desde o último quadro com interferente, k=4, T=1,5 s.
  + λ_n0·L_n0       amostras N=0 com perfil: erro espectral comprimido (|·|^0,3, magnitude + complexo) entre a saída com
                    perfil e a saída do MESMO modelo sem perfil (sem gradiente, mesmo gating ZERO/ERB+DF), nos quadros
                    fora do silêncio digital: o perfil não pode mudar o que o denoiser faz quando só o alvo fala.
  + λ_g·L_guard     o guard do v4-keep (SI-SDR N=0 com perfil vs v3.1 congelado sem perfil, margem 0,5 dB), com λ_g
                    ADAPTATIVO: λ_g = clip(f·EMA(L_mr)/EMA(L_guard bruto), 1, 40), f = 0,12 (o keep deixou o guard cair a
                    0,6% de L_mr e derivou).
  + λ_sp·L2SP       Σ‖θ − θ_v3.1‖² nos parâmetros treináveis (ancoragem L2-SP); weight decay 0 (o decay puxava para 0,
                    metade da deriva do keep).
L_pres/L_wrec do keep ficam disponíveis (λ=0 por padrão).

Dados: `mix_v4` com o padrão "turnos" (p_turns das amostras N≥1: revezamento alvo/interferente sem sobreposição, pausas
0–500 ms, RIR com p=0,8) e itens de 6 s (cobre recuperação > 2 s).

Avaliação periódica: a cada `--eval-every` passos um snapshot vai para `<run>/snapshots/`; o `eval_quick.py watch`
(processo separado, CPU) avalia, grava `quick-eval.jsonl` e mantém `best/best.pt` (mais PASS; desempate C8, C6, C1).
Retomada, SIGTERM (salva e sai) e `--stop-at` como no keep; ao terminar grava `<run>/TRAIN_DONE`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from train_pro_v4 import _pow, gate_v4, leak_loss, log, lr_factor, ovl_loss, si_sdr, stop_deadline  # noqa: E402
from train_pro_v4_keep import (  # noqa: E402
    PRE_FRAMES,
    _floats,
    grad_probe,
    onset_masks,
    preservation_and_wrec,
    strong_frames,
)
from train_pro_v4_presence import build_model, frozen_reference, resolve  # noqa: E402
from cctrain.dfvendor import ensure_df_importable  # noqa: E402
from cctrain.hashing import sha256_file  # noqa: E402
from cctrain.losses.dfnet import (  # noqa: E402
    asymmetric_formant_loss,
    df_coef_temporal_regularization_loss,
    dfnet_loss,
    local_snr_loss,
)
from cctrain.losses.distill import distill_loss  # noqa: E402
from cctrain.m1 import pools  # noqa: E402
from cctrain.m2.config import load_stage_f_config  # noqa: E402
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3  # noqa: E402
from cctrain.paths import Paths  # noqa: E402
from cctrain.train.checkpoint import (  # noqa: E402
    config_hash,
    git_commit,
    load_checkpoint,
    load_latest,
    restore_rng,
    save_checkpoint,
)
from cctrain.train.embed_bank import ArrayEmbeddingBank  # noqa: E402
from cctrain.train.stage_c import _lsnr_target, _to_device, _truncate_log, total_pss  # noqa: E402
from cctrain.train.stage_f import HOP, StageFHParams, _gated, bn_state_sha256, prepare_student  # noqa: E402
from cctrain.data.mix_v4 import N_PROBS, PATTERN_PROBS, StageV4Dataset, V4Params, build_pools  # noqa: E402

ensure_df_importable()
from df.loss import Istft  # noqa: E402
from df.model import ModelParams  # noqa: E402

GIB = 1024**3
TERMS = ("mr", "os", "leak", "leak_abs", "tail", "ovl", "guard", "lsnr", "turn", "n0g", "n0", "l2sp", "pres", "wrec", "t", "df_reg")
FRAME_S = HOP / 48_000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Treino pDFNet3 Pro v4.2 (alvo sozinho após o interferente, dano N=0)")
    p.add_argument("--run-id", default="pro-v42")
    p.add_argument("--init-ckpt", default=None, help="padrão: v3.1 (runs/m4/pro-v4/checkpoints/ckpt-00079998.pt)")
    p.add_argument("--guard-ckpt", default=None, help="referência congelada do L_guard (padrão: v3.1)")
    p.add_argument("--total-steps", type=int, default=25_000, help="tamanho do cosseno (e da sequência de dados)")
    p.add_argument("--max-steps", type=int, default=None, help="para neste passo (smoke)")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--segment-s", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261114)
    p.add_argument("--n-probs", default=",".join(str(x) for x in N_PROBS))
    p.add_argument("--pattern-probs", default="0.25,0.40,0.20,0.15", help="contínuo, conversa, blocos, esparso")
    p.add_argument("--p-interf-first", type=float, default=0.7)
    p.add_argument("--p-turns", type=float, default=0.4)
    p.add_argument("--turn-gap-max-s", type=float, default=0.5)
    p.add_argument("--p-rir-turns", type=float, default=0.8)
    p.add_argument("--formant-weight", type=float, default=2.5)
    p.add_argument("--lambda-leak", type=float, default=3.0)
    p.add_argument("--lambda-ovl", type=float, default=10.0)
    p.add_argument("--lsnr-factor", type=float, default=2e-3)
    p.add_argument("--lambda-turn", type=float, default=0.7)
    p.add_argument("--turn-hinge-db", type=float, default=-1.5)
    p.add_argument("--turn-boost", type=float, default=4.0, help="peso logo após o interferente parar")
    p.add_argument("--turn-decay-s", type=float, default=1.5)
    p.add_argument("--lambda-n0", type=float, default=0.0, help="L_n0 antigo (casar com a saída sem perfil)")
    p.add_argument("--lambda-n0g", type=float, default=1.0, help="hinge no ganho do alvo-sombra em N=0 com perfil")
    p.add_argument("--n0g-hinge-db", type=float, default=-1.0)
    p.add_argument("--lambda-tail", type=float, default=7.5, help="L_leak extra na cauda (0–tail_s após o interferente)")
    p.add_argument("--tail-s", type=float, default=1.5)
    p.add_argument("--p-target-absent", type=float, default=0.20)
    p.add_argument("--lambda-leak-absent", type=float, default=3.0,
                   help="L_leak extra nas amostras com alvo AUSENTE (C8: vista só-interferente)")
    p.add_argument("--guard-frac", type=float, default=0.12, help="fração-alvo de L_mr para o guard (λ adaptativo)")
    p.add_argument("--guard-lambda-min", type=float, default=1.0)
    p.add_argument("--guard-lambda-max", type=float, default=10.0)
    p.add_argument("--guard-margin", type=float, default=0.5, help="dB")
    p.add_argument("--lambda-sp", type=float, default=0.01, help="L2-SP: Σ‖θ−θ_v3.1‖²")
    p.add_argument("--lambda-pres", type=float, default=0.0)
    p.add_argument("--lambda-wrec", type=float, default=0.0)
    p.add_argument("--onset-weight", type=float, default=4.0)
    p.add_argument("--onset-frames", type=int, default=15)
    p.add_argument("--pres-hinge-db", type=float, default=-3.0)
    p.add_argument("--stop-at", default=None, help="HH:MM local: salva e sai limpo (próxima ocorrência)")
    p.add_argument("--ckpt-every-min", type=float, default=15.0)
    p.add_argument("--keep", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=2000, help="snapshot para o eval_quick (0 = desliga)")
    p.add_argument("--no-update", action="store_true", help="calibração: sem passo do otimizador")
    p.add_argument("--probe-steps", default="", help="passos com medição de gradiente por termo")
    p.add_argument("--runs-subdir", default="m4")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


# --- perdas novas ---------------------------------------------------------------------------------------------------


def target_shadow(model, spec_clean: torch.Tensor, m: torch.Tensor, df_coefs: torch.Tensor,
                  zero: torch.Tensor) -> torch.Tensor:
    """O alvo limpo pelas MESMAS decisões tomadas sobre a mistura (máscara ERB, coeficientes DF, ZERO) [B,1,T,F,2]."""
    base = model.base
    sm = base.mask(spec_clean.clone(), m)
    se = base.df_op(spec_clean.clone(), df_coefs)
    se = torch.cat([se[..., : base.nb_df, :], sm[..., base.nb_df:, :]], dim=-2)
    return torch.where(zero[:, None, :, None, None], torch.zeros_like(se), se)


def since_interferer_s(ia: torch.Tensor) -> torch.Tensor:
    """[B,T] bool -> segundos desde o último quadro com interferente (inf antes do primeiro)."""
    T = ia.shape[1]
    idx = torch.arange(T, device=ia.device)[None].expand_as(ia)
    last = torch.where(ia, idx, torch.full_like(idx, -1)).cummax(1).values
    s = (idx - last).float() * FRAME_S
    return torch.where(last >= 0, s, torch.full_like(s, float("inf")))


def turn_loss(shadow, spec_clean, valid, ta, ia, *, hinge_db: float, boost: float, decay_s: float):
    """L_turn (docstring do módulo). Devolve (perda, stats)."""
    e_sh = _pow(shadow.float())
    e_s = _pow(spec_clean.float())
    E_sh, E_s = e_sh.sum(-1), e_s.sum(-1)  # [B,T]
    g = 10 * torch.log10((E_sh + 1e-12) / (E_s + 1e-12))
    g = torch.clamp(g, min=-40.0)
    l = F.relu(hinge_db - g) / 10.0
    taf = ta.float()
    mean_act = (E_s * taf).sum(1, keepdim=True) / taf.sum(1, keepdim=True).clamp(min=1)
    w_e = torch.sqrt(E_s / mean_act.clamp(min=1e-12)).clamp(max=3.0)
    since = since_interferer_s(ia)
    w_t = 1.0 + (boost - 1.0) * torch.clamp(1.0 - since / decay_s, min=0.0, max=1.0)
    w_t = torch.where(torch.isfinite(since), w_t, torch.ones_like(w_t))
    w = w_e * w_t * valid.float()
    ws = w.sum()
    loss = (l * w).sum() / ws if float(ws) > 0 else shadow.sum() * 0.0
    with torch.no_grad():
        v = valid
        early = v & (since <= 0.5)
        stats = {"g_turn_db": g[v].median() if v.any() else torch.tensor(float("nan")),
                 "g_turn_early_db": g[early].median() if early.any() else torch.tensor(float("nan")),
                 "n_turn": v.sum().float()}
    return loss, stats


def compressed_err(y: torch.Tensor, s: torch.Tensor, frames: torch.Tensor, gamma: float) -> torch.Tensor:
    """Erro espectral comprimido (|·|^γ, magnitude + complexo) médio nos quadros `frames` [B,T]."""
    y, s = y[:, 0].float(), s[:, 0].float()
    my = (y[..., 0] ** 2 + y[..., 1] ** 2 + 1e-10).sqrt()
    ms = (s[..., 0] ** 2 + s[..., 1] ** 2 + 1e-10).sqrt()
    cy, cs = my ** gamma, ms ** gamma
    e = ((cy - cs) ** 2).mean(-1) + (((y * (cy / my)[..., None]) - (s * (cs / ms)[..., None])) ** 2).sum(-1).mean(-1)
    w = frames.float()
    return (e * w).sum() / w.sum().clamp(min=1)


# --- passo ----------------------------------------------------------------------------------------------------------


def step_v42(model, ref, teacher, istft, df_params, hp: StageFHParams, b: dict, a, anchor, lam_guard: float):
    spec, spec_clean, mask = b["spec"], b["spec_clean"], b["mask"]
    nb, T = spec.shape[0], spec.shape[2]
    profile = mask.reshape(nb) > 0
    spec_e, m, lsnr, df_coefs = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], mask)
    spec_m = model.base.mask(spec.clone(), m)
    out, zero = gate_v4(spec, spec_m, spec_e, lsnr, profile)
    est, tgt = istft(out), istft(spec_clean)
    kw = dict(n_ffts=hp.fft_sizes, gamma=hp.loss_gamma)

    ta = b["target_active"][:, :T]
    ia = b["interf_active"][:, :T]
    ds = b["digital_silence"][:, :T]
    prof = profile[:, None]
    f_leak = (prof & ~ta & ia & ~ds).float()
    f_ovl = (prof & ta & ia).float()
    f_keep = prof & ta & ~ds
    has_int = (b["n_interf"] >= 1)[:, None]

    mr = dfnet_loss(est, tgt, b["max_freq"], factor=hp.loss_factor, factor_complex=hp.loss_factor_complex, **kw)
    os_ = asymmetric_formant_loss(est, tgt, b["max_freq"], factor=hp.lambda_os, formant_weight=a.formant_weight,
                                  extra_formant_penalty=1.0, **kw)
    idx_n1 = (b["n_interf"] >= 1).nonzero(as_tuple=True)[0]
    if idx_n1.numel():
        os_ = os_ + (idx_n1.numel() / nb) * asymmetric_formant_loss(
            est[idx_n1], tgt[idx_n1], b["max_freq"][idx_n1], factor=hp.lambda_os, formant_weight=a.formant_weight,
            extra_formant_penalty=1.0, **kw)
    leak = leak_loss(out, spec, f_leak)
    ovl = ovl_loss(out, spec_clean, b["spec_interf"], f_ovl)
    lsnr_t = _lsnr_target(df_params, spec_clean, spec)
    ls = local_snr_loss(lsnr, lsnr_t, ~ds, factor=a.lsnr_factor)

    # L_turn: ganho do alvo-sombra nos quadros alvo-sozinho (perfil, N≥1)
    shadow = target_shadow(model, spec_clean, m, df_coefs, zero)
    valid = prof & has_int & b["has_target"][:, None] & ta & ~ia & ~ds
    turn, tst = turn_loss(shadow, spec_clean, valid, ta, ia, hinge_db=a.turn_hinge_db, boost=a.turn_boost,
                          decay_s=a.turn_decay_s)

    # L_n0g: hinge no ganho do alvo-sombra em N=0 com perfil (mesma forma do L_turn, sem reforço temporal)
    v0 = prof & (b["n_interf"] == 0)[:, None] & b["has_target"][:, None] & ta & ~ds
    n0g, n0st = turn_loss(shadow, spec_clean, v0, ta, torch.zeros_like(ia), hinge_db=a.n0g_hinge_db, boost=1.0,
                          decay_s=1.0)
    # L_tail: vazamento na cauda do interferente (0–tail_s depois do fim, VAD pico−45 na trilha com reverberação)
    iw = b["interf_active_wet45"][:, :T]
    since_i = since_interferer_s(ia)
    f_tail = (prof & ~ta & ~ia & iw & ~ds & (since_i <= a.tail_s)).float()
    tail = leak_loss(out, spec, f_tail)
    leak_abs = leak_loss(out, spec, f_leak * (~b["has_target"]).float()[:, None])

    # L_n0 e L_guard: N=0 com perfil
    idx_g = (profile & (b["n_interf"] == 0) & b["has_target"]).nonzero(as_tuple=True)[0]
    if idx_g.numel():
        with torch.no_grad():
            z = torch.zeros_like(mask[idx_g])
            sg = spec[idx_g]
            on = torch.ones(idx_g.numel(), device=spec.device, dtype=torch.bool)
            e1, m1, l1, _ = model(sg.clone(), b["feat_erb"][idx_g], b["feat_spec"][idx_g], b["embedding"][idx_g], z)
            o1, _ = gate_v4(sg, model.base.mask(sg.clone(), m1), e1, l1, on)  # mesmo modelo, sem perfil
            e0, m0, l0, _ = ref(sg.clone(), b["feat_erb"][idx_g], b["feat_spec"][idx_g], b["embedding"][idx_g], z)
            o0, _ = gate_v4(sg, ref.base.mask(sg.clone(), m0), e0, l0, ~on)  # v3.1 sem perfil (guard do keep)
            sd0 = si_sdr(istft(o0), tgt[idx_g])
        n0 = compressed_err(out[idx_g], o1, ~ds[idx_g], hp.loss_gamma)
        sdp = si_sdr(est[idx_g], tgt[idx_g])
        guard = F.relu(sd0 - a.guard_margin - sdp).mean() / 10.0
        guard_gap = (sdp - sd0).mean().detach()
    else:
        n0 = est.sum() * 0.0
        guard = est.sum() * 0.0
        guard_gap = torch.tensor(float("nan"), device=spec.device)

    # L2-SP
    if a.lambda_sp > 0:
        l2sp = sum(((p - p0) ** 2).sum() for p, p0 in anchor)
    else:
        l2sp = est.sum() * 0.0

    if a.lambda_pres > 0 or a.lambda_wrec > 0:
        strong = strong_frames(spec_clean, ta) & prof & has_int & ~ds
        onset, onset_win = onset_masks(ta & prof, PRE_FRAMES, a.onset_frames)
        pres, wrec, _ = preservation_and_wrec(out, spec_clean, ta & prof, strong, onset, onset_win,
                                              onset_len=a.onset_frames, onset_weight=a.onset_weight,
                                              hinge_db=a.pres_hinge_db, gamma=hp.loss_gamma)
    else:
        pres = wrec = est.sum() * 0.0

    if df_coefs is not None and df_coefs.numel() > 0:
        l_df = df_coef_temporal_regularization_loss(df_coefs, time_dim=1) / (df_coefs.numel() / df_coefs.shape[1]) * 0.05
    else:
        l_df = est.sum() * 0.0

    idx0 = (~profile).nonzero(as_tuple=True)[0]
    if idx0.numel():
        with torch.no_grad():
            s0 = spec[idx0]
            t_e, t_m, t_lsnr, _ = teacher(s0.clone(), b["feat_erb"][idx0], b["feat_spec"][idx0])
            t_out, _ = _gated(teacher, s0, t_e, t_m, t_lsnr)
            t_est = istft(t_out)
    else:
        t_est = est[:0]
    t = distill_loss(est, t_est, mask, b["max_freq"], lambda_t=hp.lambda_t, n_ffts=hp.fft_sizes, gamma=hp.loss_gamma)

    parts = {"mr": mr, "os": os_, "leak": a.lambda_leak * leak, "ovl": a.lambda_ovl * ovl,
             "guard": lam_guard * guard, "lsnr": ls, "turn": a.lambda_turn * turn, "n0": a.lambda_n0 * n0,
             "n0g": a.lambda_n0g * n0g, "tail": a.lambda_tail * tail, "leak_abs": a.lambda_leak_absent * leak_abs,
             "l2sp": a.lambda_sp * l2sp, "pres": a.lambda_pres * pres, "wrec": a.lambda_wrec * wrec, "t": t,
             "df_reg": l_df}
    total = sum(parts.values())
    with torch.no_grad():
        oi = f_leak.bool()
        stats = {
            "raw_leak": leak.detach(), "raw_ovl": ovl.detach(), "raw_guard": guard.detach(), "raw_turn": turn.detach(),
            "raw_n0": n0.detach(), "raw_n0g": n0g.detach(), "raw_tail": tail.detach(),
            "g_n0_db": n0st["g_turn_db"].to(spec.device), "n_tail": f_tail.sum(), "raw_l2sp": l2sp.detach() if torch.is_tensor(l2sp) else torch.tensor(0.0),
            "lam_guard": torch.tensor(lam_guard), "guard_gap_db": guard_gap,
            "g_turn_db": tst["g_turn_db"].to(spec.device), "g_turn_early_db": tst["g_turn_early_db"].to(spec.device),
            "n_turn": tst["n_turn"],
            "zero_frac_only_interf": (zero & oi).sum() / oi.sum().clamp(min=1),
            "zero_frac_target": (zero & f_keep).sum() / f_keep.sum().clamp(min=1),
            "n_guard": torch.tensor(float(idx_g.numel())), "n_noprof": torch.tensor(float(idx0.numel())),
        }
    return {"total": total, **parts, **stats}


# --- dados ----------------------------------------------------------------------------------------------------------


def v42_params(a) -> V4Params:
    npr = _floats(a.n_probs, 5, "n-probs")
    pp = _floats(a.pattern_probs, 4, "pattern-probs")
    return V4Params(seed=a.seed, segment_s=a.segment_s, batch_size=a.batch_size, total_steps=a.total_steps,
                    phase1_frac=0.0, n_probs=npr if npr != N_PROBS else N_PROBS,
                    pattern_probs=pp if pp != PATTERN_PROBS else PATTERN_PROBS, p_interf_first=a.p_interf_first,
                    p_turns=a.p_turns, turn_gap_max_s=a.turn_gap_max_s, p_rir_turns=a.p_rir_turns,
                    p_target_absent=a.p_target_absent)


def make_dataset(paths: Paths, vp: V4Params):
    bank = ArrayEmbeddingBank.load(paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz")
    speech = pools.by_split(pools.speech_records(paths), "train")
    noise_train, _ = pools.noise_pools()
    pl = build_pools(speech, noise_train, pools.rir_records(), bank, paths.data, seed=vp.seed)
    return StageV4Dataset(pl, bank, paths.data, vp), pl


# --- main -----------------------------------------------------------------------------------------------------------


def main():
    a = parse_args()
    paths = Paths.from_env()
    run_dir = paths.runs / a.runs_subdir / a.run_id
    ckpt_dir = run_dir / "checkpoints"
    snap_dir = run_dir / "snapshots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "TRAIN_DONE").unlink(missing_ok=True)
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    deadline = stop_deadline(a.stop_at)
    log(f"run {a.run_id} -> {run_dir} | device {device} | stop-at {a.stop_at} "
        f"({dt.datetime.fromtimestamp(deadline) if deadline else '-'}) | pid {os.getpid()}")
    stop = {"flag": False, "why": ""}

    def on_signal(signum, _frame):
        stop["flag"], stop["why"] = True, signal.Signals(signum).name
        log(f"sinal {stop['why']} recebido: salvando checkpoint e saindo")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    teacher, df_state = load_upstream_dfnet3(device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    cfg = load_stage_f_config()
    hp = StageFHParams.from_config(cfg, run_id=a.run_id, num_workers=a.workers, batch_size=a.batch_size)
    model = build_model(device, hp, presence=False)
    prepare_student(model)
    init = resolve(paths, a.init_ckpt)
    guard_ck = resolve(paths, a.guard_ckpt)
    ref = frozen_reference(guard_ck, device, hp)
    istft = Istft(960, HOP, torch.as_tensor(df_state.fft_window().copy())).to(device)
    params = ModelParams()

    vp = v42_params(a)
    hashed = {"v4": vp.hashable(), "lr": a.lr, "wd": a.weight_decay, "warmup": a.warmup, "total_steps": a.total_steps,
              "batch": a.batch_size,
              "lambdas": [a.lambda_leak, a.lambda_ovl, a.lambda_turn, a.lambda_n0, a.lambda_sp, a.lambda_pres,
                          a.lambda_wrec, a.lsnr_factor, a.lambda_n0g, a.n0g_hinge_db, a.lambda_tail, a.tail_s,
                          a.lambda_leak_absent],
              "turn": [a.turn_hinge_db, a.turn_boost, a.turn_decay_s],
              "guard": [a.guard_frac, a.guard_lambda_min, a.guard_lambda_max, a.guard_margin],
              "fw": a.formant_weight, "guard_ckpt": str(guard_ck), "init": str(init), "v42": 1}
    chash = config_hash(hashed)

    # âncora L2-SP: pesos do init (v3.1), mesmo na retomada
    init_ck = load_checkpoint(init)
    model.load_state_dict(init_ck.model, strict=True)
    prepare_student(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    anchor = [(p, p.detach().clone()) for p in trainable]
    del init_ck
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, a.warmup, a.total_steps))

    log_path = run_dir / "loss-log.jsonl"
    probe_path = run_dir / "probe.jsonl"
    probe_steps = {int(x) for x in a.probe_steps.split(",") if x.strip()}
    existing = load_latest(ckpt_dir, expected_cfg_hash=chash)
    step, spent_before = 0, 0.0
    ema = {"mr": None, "guard": None}
    init_sha = sha256_file(init)
    if existing is not None:
        model.load_state_dict(existing.model)
        opt.load_state_dict(existing.optimizer)
        sched.load_state_dict(existing.scheduler)
        restore_rng(existing)
        step = existing.step
        spent_before = float(existing.extra.get("elapsed_s", 0.0))
        ema = dict(existing.extra.get("ema", ema))
        _truncate_log(log_path, step)
        log(f"RETOMADA de {existing.path.name} (passo {step})")
    else:
        log(f"warm start {init} sha {init_sha[:12]} (strict); guard congelado: {guard_ck}; âncora L2-SP = init "
            f"({sum(p.numel() for p in trainable)} parâmetros treináveis)")
        for pth in (log_path, probe_path):
            if pth.exists():
                pth.unlink()
    prepare_student(model)
    bn_before = bn_state_sha256(model)
    commit = git_commit()

    ds, pl = make_dataset(paths, vp)
    log(f"pools: {json.dumps(pl.stats)} | v4 {json.dumps(vp.hashable())}")
    sampler = list(range(step * a.batch_size, len(ds)))
    dl = DataLoader(ds, batch_size=a.batch_size, sampler=sampler, num_workers=a.workers,
                    prefetch_factor=a.prefetch if a.workers else None, persistent_workers=False)

    t_start = time.monotonic()
    last_ckpt = t_start
    step_times: deque = deque(maxlen=200)
    waits: deque = deque(maxlen=200)
    rel_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    pss_max = 0

    def elapsed() -> float:
        return spent_before + (time.monotonic() - t_start)

    def save(reason: str, directory: Path = ckpt_dir, keep: int = a.keep) -> Path:
        pth = save_checkpoint(directory, step=step, model=model.state_dict(), optimizer=opt, scheduler=sched,
                              keep=keep, sampler_state={"step": step}, cfg_hash=chash, commit=commit,
                              extra={"elapsed_s": elapsed(), "init_sha256": init_sha, "bn_sha256": bn_before,
                                     "reason": reason, "args": vars(a), "v4": vp.hashable(), "presence_head": False,
                                     "ema": ema, "gating": {"mode": "lsnr-profile", "zero_below_db": -10.0},
                                     "wall": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"checkpoint {pth} ({reason})")
        return pth

    def lam_guard() -> float:
        if ema["mr"] is None or ema["guard"] is None:
            return a.guard_lambda_max
        lam = a.guard_frac * ema["mr"] / max(ema["guard"], 1e-6)
        return float(min(max(lam, a.guard_lambda_min), a.guard_lambda_max))

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    end_step = min(a.total_steps, a.max_steps) if a.max_steps else a.total_steps
    stop_reason = "total_steps"
    with open(log_path, "a") as lf:
        it = iter(dl)
        while step < end_step:
            tw = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                stop_reason = "data_end"
                break
            t0 = time.perf_counter()
            waits.append(t0 - tw)
            b = _to_device(batch, device)
            prepare_student(model)
            opt.zero_grad(set_to_none=True)
            lg = lam_guard()
            parts = step_v42(model, ref, teacher, istft, params, hp, b, a, anchor, lg)
            if step in probe_steps:
                pr = _probe(parts, trainable)
                with open(probe_path, "a") as pf:
                    pf.write(json.dumps({"step": step, **pr}) + "\n")
                log(f"sonda passo {step}: normas " + " ".join(f"{k}={v:.2e}" for k, v in pr["norm"].items())
                    + " | cos com total " + " ".join(f"{k}={v:+.2f}" for k, v in pr["cos"].items()
                                                     if k.endswith("|total")))
            total = parts["total"]
            if not torch.isfinite(total):
                log(f"perda não finita no passo {step}: " + json.dumps({k: float(v) for k, v in parts.items()}))
                opt.zero_grad(set_to_none=True)
                step += 1
                sched.step()
                continue
            total.backward()
            gn = nn.utils.clip_grad_norm_(trainable, hp.grad_clip)
            if not a.no_update:
                opt.step()
            sched.step()
            step += 1
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            row = {k: float(v.detach()) for k, v in parts.items()}
            for k, v in (("mr", row["mr"]), ("guard", row["raw_guard"])):
                if row["n_guard"] > 0 or k == "mr":
                    ema[k] = v if ema[k] is None else 0.98 * ema[k] + 0.02 * v
            mr = max(row["mr"], 1e-9)
            for k in TERMS[1:]:
                rel_hist[k].append(row[k] / mr)
            if step % 20 == 0 or step <= 3:
                ta, ia = batch["target_active"], batch["interf_active"]
                cls = {"f_only_t": float((ta & ~ia).float().mean()), "f_only_i": float((~ta & ia).float().mean()),
                       "f_both": float((ta & ia).float().mean())}
                rel = {f"rel_{k}": float(np.median(v)) for k, v in rel_hist.items()}
                entry = {"step": step, "lr": float(sched.get_last_lr()[0]), **row, "grad_norm": float(gn),
                         "dt": step_times[-1], "wait": waits[-1], **cls, **rel, "wall": time.strftime("%H:%M:%S")}
                lf.write(json.dumps(entry) + "\n")
                lf.flush()
            if step % 50 == 0 or (a.no_update and step % 5 == 0):
                pss_max = max(pss_max, total_pss())
                sps = statistics.median(step_times)
                rel = " ".join(f"{k}={100 * np.median(v):.1f}%" for k, v in rel_hist.items())
                eta = f" | até stop-at: {(deadline - time.time()) / 3600:.2f} h" if deadline else ""
                log(f"passo {step}/{end_step} total={row['total']:.3f} mr={row['mr']:.3f} "
                    f"s/passo={sps:.3f} espera={statistics.median(waits):.3f} lr={sched.get_last_lr()[0]:.2e} "
                    f"gTurn={row['g_turn_db']:+.1f} gTurn0.5={row['g_turn_early_db']:+.1f} gN0={row['g_n0_db']:+.1f} "
                    f"zeroOI={row['zero_frac_only_interf']:.2f} zeroT={row['zero_frac_target']:.3f} "
                    f"gap={row['guard_gap_db']:+.2f} λg={row['lam_guard']:.1f} l2sp={row['raw_l2sp']:.2f} "
                    f"| rel: {rel} | pss={pss_max / GIB:.1f} GiB{eta}")
            if stop["flag"]:
                stop_reason = f"signal:{stop['why']}"
                break
            if deadline and time.time() >= deadline:
                stop_reason = "stop_at"
                break
            if a.eval_every > 0 and step % a.eval_every == 0 and not a.no_update:
                save("eval-snapshot", snap_dir, keep=10_000)
            if time.monotonic() - last_ckpt >= a.ckpt_every_min * 60.0:
                save("periodic")
                last_ckpt = time.monotonic()
    if not a.no_update:
        p = save(f"final:{stop_reason}")
        if a.eval_every > 0 and step % a.eval_every != 0:
            save("eval-snapshot-final", snap_dir, keep=10_000)
        log(f"último checkpoint: {p}")
    summary = {
        "run_id": a.run_id, "step": step, "stop_reason": stop_reason, "elapsed_s": elapsed(),
        "s_per_step_median": statistics.median(step_times) if step_times else None,
        "loader_wait_median": statistics.median(waits) if waits else None,
        "rel_median": {k: float(np.median(v)) for k, v in rel_hist.items()},
        "pss_gb_max": pss_max / GIB, "bn_unchanged": bn_state_sha256(model) == bn_before,
        "vram_peak_gb": torch.cuda.max_memory_allocated(device) / GIB if device.startswith("cuda") else None,
        "ema": ema, "args": vars(a),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "TRAIN_DONE").write_text(f"{stop_reason} passo {step} {time.strftime('%F %T')}\n")
    log(f"fim: {stop_reason} no passo {step}; resumo em {run_dir / 'summary.json'}")


def _probe(parts: dict, params: list) -> dict:
    sub = {k: parts[k] for k in TERMS}
    import train_pro_v4_keep as tk

    old = tk.TERMS
    tk.TERMS = TERMS
    try:
        return grad_probe(sub, params)
    finally:
        tk.TERMS = old


if __name__ == "__main__":
    main()
