#!/usr/bin/env python3
"""Treino do pDFNet3 Pro v4-keep: PRESERVAR a voz do alvo (contrato v3, gating novo por lsnr com perfil).

Motivação (exp/soft-gate, `runs/m4/soft-gate-eval/report.md`): em todas as versões (v2, v3, v3.1, v4-presence) o modelo
atenua o alvo em mistura em 10–20 dB e corta os primeiros 100 ms de cada trecho do alvo em ~−24 dB, com ou sem gating,
enquanto o só-interferente já cai 38–48 dB. Este treino ataca o alvo, mantendo o que já funciona:

- Partida: v3.1 (`runs/m4/pro-v4/checkpoints/ckpt-00079998.pt`), mesma arquitetura (sem cabeça de presença).
- Gating emulado = o "novo" do v3.1 (`train_pro_v4.gate_v4`): com perfil, ZERO se lsnr < −10, senão ERB+DF; sem perfil,
  (−10, 30, 20). Sem L_zero/L_keep (brigavam no lsnr, cosseno −0,99).
- Perdas: L_mr + L_os + λ_leak·L_leak + λ_ovl·L_ovl + λ_guard·L_guard (referência CONGELADA: v3.1 sem perfil)
  + L_lsnr (alvo do v3.1: alvo contra todo o resto, todos os quadros menos silêncio digital) + L_t + L_dfreg
  + λ_pres·L_pres + λ_wrec·L_wrec (novas):
  * L_pres (preservação): ganho do alvo na saída pela projeção complexa no alvo limpo, G = Re⟨Y,S⟩/‖S‖², em
    unidades curtas: (a) segmentos alinhados de 100 ms, só nos quadros de alvo FORTE (≥ −20 dB do máximo da trilha)
    das amostras com perfil e N≥1; (b) janelas de início de trecho (os `onset_frames` quadros a partir de um quadro de
    alvo ativo sem alvo ativo nos 100 ms anteriores, em qualquer amostra com perfil), com peso `onset_weight`.
    Perda por unidade = relu(−3 − 10·log10(relu(G)² + 1e-4)) / 10 (hinge em −3 dB, piso em −40 dB).
  * L_wrec (reconstrução ponderada): erro espectral comprimido (|·|^0,3, magnitude + complexo, como o DFNet) por quadro,
    ponderado: 1 nos quadros fortes das misturas com interferente, + `onset_weight` nas janelas de início.
- Dados: `mix_v4` com mais conversa/blocos e o interferente abrindo a conversa / o só-interferente antes do alvo
  (`--pattern-probs`, `--p-interf-first`).

Retomada como no `train_pro_v4.py` (checkpoint periódico, SIGTERM, `--stop-at`). `--probe-steps` mede, nesses passos,
a norma do gradiente de cada termo (já ponderado) no tronco e os cossenos entre pares (`probe.jsonl`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
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
TERMS = ("mr", "os", "leak", "ovl", "guard", "lsnr", "pres", "wrec", "t", "df_reg")
SEG_FRAMES = 10  # 100 ms
PRE_FRAMES = 10  # início = alvo ativo sem alvo ativo nos 100 ms anteriores (o mesmo critério da avaliação)
STRONG_DB = 20.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Treino pDFNet3 Pro v4-keep (preservação do alvo)")
    p.add_argument("--run-id", default="pro-v4-keep")
    p.add_argument("--init-ckpt", default=None, help="padrão: v3.1 (runs/m4/pro-v4/checkpoints/ckpt-00079998.pt)")
    p.add_argument("--guard-ckpt", default=None, help="referência congelada do L_guard (padrão: v3.1)")
    p.add_argument("--total-steps", type=int, default=140_000, help="tamanho do cosseno (e da sequência de dados)")
    p.add_argument("--max-steps", type=int, default=None, help="para neste passo (smoke)")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-5, help="pico (parte do v3.1 já ajustado)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--n-probs", default=",".join(str(x) for x in N_PROBS))
    p.add_argument("--pattern-probs", default="0.25,0.40,0.20,0.15", help="contínuo, conversa, blocos, esparso")
    p.add_argument("--p-interf-first", type=float, default=0.7)
    p.add_argument("--phase1-frac", type=float, default=0.0, help="o v3.1 já passou pelo currículo")
    p.add_argument("--formant-weight", type=float, default=2.5)
    p.add_argument("--lambda-leak", type=float, default=3.0)
    p.add_argument("--lambda-ovl", type=float, default=10.0)
    p.add_argument("--lambda-guard", type=float, default=4.0)
    p.add_argument("--lambda-pres", type=float, default=0.25)
    p.add_argument("--lambda-wrec", type=float, default=300.0)
    p.add_argument("--lsnr-factor", type=float, default=2e-3)
    p.add_argument("--onset-weight", type=float, default=4.0)
    p.add_argument("--onset-frames", type=int, default=15, help="janela de início (quadros de 10 ms)")
    p.add_argument("--pres-hinge-db", type=float, default=-3.0)
    p.add_argument("--guard-margin", type=float, default=0.5, help="dB")
    p.add_argument("--stop-at", default=None, help="HH:MM local: salva e sai limpo (próxima ocorrência)")
    p.add_argument("--ckpt-every-min", type=float, default=15.0)
    p.add_argument("--keep", type=int, default=6)
    p.add_argument("--pin-every", type=int, default=20_000)
    p.add_argument("--no-update", action="store_true", help="calibração: sem passo do otimizador")
    p.add_argument("--probe-steps", default="", help="passos com medição de gradiente por termo (ex.: 0,1,2,298,299)")
    p.add_argument("--runs-subdir", default="m4")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


# --- máscaras de quadro ---------------------------------------------------------------------------------------------


def _win_sum(x: torch.Tensor, start: torch.Tensor, length: int) -> torch.Tensor:
    """Soma de x[b, start:start+length] (cortada em T) para todo start; x [B,T] -> [B,T]."""
    T = x.shape[1]
    cs = F.pad(x.cumsum(1), (1, 0))
    hi = (start + length).clamp(max=T)
    return cs.gather(1, hi.expand_as(x)) - cs.gather(1, start.expand_as(x))


def onset_masks(ta: torch.Tensor, pre: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(início [B,T], janela [B,T]): início = alvo ativo sem alvo ativo nos `pre` quadros anteriores (o início do item
    não conta); janela = quadros de alvo ativo nos `length` quadros a partir de um início."""
    T = ta.shape[1]
    idx = torch.arange(T, device=ta.device)[None]
    taf = ta.float()
    cs = F.pad(taf.cumsum(1), (1, 0))
    prev = cs.gather(1, idx.expand_as(taf)) - cs.gather(1, (idx - pre).clamp(min=0).expand_as(taf))
    onset = ta & (prev == 0) & (idx >= pre)
    co = F.pad(onset.float().cumsum(1), (1, 0))
    win = (co.gather(1, (idx + 1).expand_as(taf)) - co.gather(1, (idx + 1 - length).clamp(min=0).expand_as(taf))) > 0
    return onset, win & ta


def strong_frames(spec_clean: torch.Tensor, ta: torch.Tensor, rel_db: float = STRONG_DB) -> torch.Tensor:
    """Quadros de alvo forte: energia ≥ máximo da trilha − `rel_db` (e alvo ativo)."""
    lv = 10 * torch.log10(_pow(spec_clean).sum(-1) + 1e-12)
    return ta & (lv >= lv.max(1, keepdim=True).values - rel_db)


# --- perdas novas ---------------------------------------------------------------------------------------------------


def _proj(out: torch.Tensor, spec_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Por quadro: Re⟨Y,S⟩ e ‖S‖² (somados nos bins). [B,1,T,F,2] -> [B,T]."""
    y, s = out[:, 0].float(), spec_clean[:, 0].float()
    num = (y[..., 0] * s[..., 0] + y[..., 1] * s[..., 1]).sum(-1)
    den = (s[..., 0] ** 2 + s[..., 1] ** 2).sum(-1)
    return num, den


def _gain_db(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    g = F.relu(num) / den.clamp(min=1e-12)
    return 10 * torch.log10(g * g + 1e-4)


def preservation_and_wrec(out, spec_clean, ta, seg_mask, onset, onset_win, *, onset_len: int, onset_weight: float,
                          hinge_db: float, gamma: float):
    """L_pres e L_wrec (ver docstring do módulo). Devolve (pres, wrec, stats)."""
    num, den = _proj(out, spec_clean)
    B, T = num.shape
    # (a) segmentos de 100 ms nos quadros fortes (perfil, N≥1)
    ns = T // SEG_FRAMES
    L = ns * SEG_FRAMES
    m = seg_mask[:, :L].float()
    sn = (num[:, :L] * m).reshape(B, ns, SEG_FRAMES).sum(-1)
    sd = (den[:, :L] * m).reshape(B, ns, SEG_FRAMES).sum(-1)
    sv = (m.reshape(B, ns, SEG_FRAMES).sum(-1) >= 3).float()
    g_seg = _gain_db(sn, sd)
    l_seg = F.relu(hinge_db - g_seg) / 10.0
    # (b) janelas de início: soma nos quadros de alvo ativo dos `onset_len` quadros a partir de cada início
    idx = torch.arange(T, device=num.device)[None]
    taf = ta.float()
    on_n = _win_sum(num * taf, idx, onset_len)
    on_d = _win_sum(den * taf, idx, onset_len)
    ov = onset.float() * (on_d > 0).float()
    g_on = _gain_db(on_n, on_d)
    l_on = F.relu(hinge_db - g_on) / 10.0
    n_seg, n_on = sv.sum(), ov.sum()
    wsum = n_seg + onset_weight * n_on
    if float(wsum) > 0:
        pres = ((l_seg * sv).sum() + onset_weight * (l_on * ov).sum()) / wsum
    else:
        pres = out.sum() * 0.0
    # L_wrec: erro espectral comprimido por quadro
    y, s = out[:, 0].float(), spec_clean[:, 0].float()
    my = (y[..., 0] ** 2 + y[..., 1] ** 2 + 1e-10).sqrt()
    ms = (s[..., 0] ** 2 + s[..., 1] ** 2 + 1e-10).sqrt()
    cy, cs_ = my ** gamma, ms ** gamma
    e_mag = ((cy - cs_) ** 2).mean(-1)
    e_cplx = (((y * (cy / my)[..., None]) - (s * (cs_ / ms)[..., None])) ** 2).sum(-1).mean(-1)
    w = seg_mask.float() + onset_weight * onset_win.float()
    nw = w.sum()
    wrec = ((e_mag + e_cplx) * w).sum() / nw if float(nw) > 0 else out.sum() * 0.0
    with torch.no_grad():
        stats = {
            "g_seg_db": (g_seg * sv).sum() / n_seg.clamp(min=1) if float(n_seg) else torch.tensor(float("nan")),
            "g_on_db": (g_on * ov).sum() / n_on.clamp(min=1) if float(n_on) else torch.tensor(float("nan")),
            "n_seg": n_seg, "n_on": n_on,
        }
    return pres, wrec, stats


# --- passo ----------------------------------------------------------------------------------------------------------


def step_keep(model, ref, teacher, istft, df_params, hp: StageFHParams, b: dict, a) -> dict[str, torch.Tensor]:
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

    # L_lsnr como no v3.1: alvo contra todo o resto, todos os quadros menos silêncio digital
    lsnr_t = _lsnr_target(df_params, spec_clean, spec)
    ls = local_snr_loss(lsnr, lsnr_t, ~ds, factor=a.lsnr_factor)

    # preservação do alvo e reconstrução ponderada
    strong = strong_frames(spec_clean, ta) & prof & has_int & ~ds
    onset, onset_win = onset_masks(ta & prof, PRE_FRAMES, a.onset_frames)
    pres, wrec, pst = preservation_and_wrec(out, spec_clean, ta & prof, strong, onset, onset_win,
                                            onset_len=a.onset_frames, onset_weight=a.onset_weight,
                                            hinge_db=a.pres_hinge_db, gamma=hp.loss_gamma)

    # L_guard: N=0 com perfil contra a referência congelada (v3.1 sem perfil, gating sem perfil)
    idx_g = (profile & (b["n_interf"] == 0) & b["has_target"]).nonzero(as_tuple=True)[0]
    if idx_g.numel():
        with torch.no_grad():
            z = torch.zeros_like(mask[idx_g])
            e0, m0, l0, _ = ref(spec[idx_g].clone(), b["feat_erb"][idx_g], b["feat_spec"][idx_g],
                                b["embedding"][idx_g], z)
            sm0 = ref.base.mask(spec[idx_g].clone(), m0)
            o0, _ = gate_v4(spec[idx_g], sm0, e0, l0, torch.zeros(idx_g.numel(), device=spec.device, dtype=torch.bool))
            sd0 = si_sdr(istft(o0), tgt[idx_g])
        sdp = si_sdr(est[idx_g], tgt[idx_g])
        guard = F.relu(sd0 - a.guard_margin - sdp).mean() / 10.0
        guard_gap = (sdp - sd0).mean().detach()
    else:
        guard = est.sum() * 0.0
        guard_gap = torch.tensor(float("nan"), device=spec.device)

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
             "guard": a.lambda_guard * guard, "lsnr": ls, "pres": a.lambda_pres * pres,
             "wrec": a.lambda_wrec * wrec, "t": t, "df_reg": l_df}
    total = sum(parts.values())
    with torch.no_grad():
        oi = f_leak.bool()
        stats = {
            "raw_leak": leak.detach(), "raw_ovl": ovl.detach(), "raw_guard": guard.detach(),
            "raw_pres": pres.detach(), "raw_wrec": wrec.detach(), "guard_gap_db": guard_gap,
            "g_seg_db": pst["g_seg_db"].to(spec.device), "g_on_db": pst["g_on_db"].to(spec.device),
            "n_seg": pst["n_seg"], "n_on": pst["n_on"],
            "zero_frac_only_interf": (zero & oi).sum() / oi.sum().clamp(min=1),
            "zero_frac_target": (zero & f_keep).sum() / f_keep.sum().clamp(min=1),
            "zero_frac_onset": (zero & onset_win).sum() / onset_win.sum().clamp(min=1),
            "n_guard": torch.tensor(float(idx_g.numel())), "n0": torch.tensor(float(idx0.numel())),
        }
    return {"total": total, **parts, **stats}


# --- dados ----------------------------------------------------------------------------------------------------------


def _floats(s: str, n: int, name: str) -> tuple[float, ...]:
    v = tuple(float(x) for x in s.split(","))
    if len(v) != n or abs(sum(v) - 1.0) > 1e-6:
        raise SystemExit(f"--{name} inválido: {v}")
    return v


def v4_params(a) -> V4Params:
    npr = _floats(a.n_probs, 5, "n-probs")
    pp = _floats(a.pattern_probs, 4, "pattern-probs")
    return V4Params(seed=a.seed, batch_size=a.batch_size, total_steps=a.total_steps, phase1_frac=a.phase1_frac,
                    n_probs=npr if npr != N_PROBS else N_PROBS, pattern_probs=pp if pp != PATTERN_PROBS else PATTERN_PROBS,
                    p_interf_first=a.p_interf_first)


def make_dataset(paths: Paths, vp: V4Params):
    bank = ArrayEmbeddingBank.load(paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz")
    speech = pools.by_split(pools.speech_records(paths), "train")
    noise_train, _ = pools.noise_pools()
    pl = build_pools(speech, noise_train, pools.rir_records(), bank, paths.data, seed=vp.seed)
    return StageV4Dataset(pl, bank, paths.data, vp), pl


# --- sonda de gradiente ---------------------------------------------------------------------------------------------


def grad_probe(parts: dict, params: list[nn.Parameter]) -> dict:
    vecs, norms = {}, {}
    for k in TERMS:
        v = parts[k]
        if not v.requires_grad:
            norms[k] = 0.0
            continue
        gs = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        vec = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(gs, params)])
        norms[k] = float(vec.norm())
        if norms[k] > 0:
            vecs[k] = vec
    cos = {}
    for x, y in itertools.combinations(list(vecs), 2):
        cos[f"{x}|{y}"] = float(F.cosine_similarity(vecs[x][None], vecs[y][None]).item())
    tot = sum(vecs.values())
    for x in vecs:
        cos[f"{x}|total"] = float(F.cosine_similarity(vecs[x][None], tot[None]).item())
    return {"norm": norms, "cos": cos}


# --- main -----------------------------------------------------------------------------------------------------------


def main():
    a = parse_args()
    paths = Paths.from_env()
    run_dir = paths.runs / a.runs_subdir / a.run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
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

    vp = v4_params(a)
    hashed = {"v4": vp.hashable(), "lr": a.lr, "warmup": a.warmup, "total_steps": a.total_steps, "batch": a.batch_size,
              "lambdas": [a.lambda_leak, a.lambda_ovl, a.lambda_guard, a.lambda_pres, a.lambda_wrec, a.lsnr_factor],
              "onset": [a.onset_weight, a.onset_frames, a.pres_hinge_db], "guard_margin": a.guard_margin,
              "fw": a.formant_weight, "guard_ckpt": str(guard_ck), "keep": 1}
    chash = config_hash(hashed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, a.warmup, a.total_steps))

    log_path = run_dir / "loss-log.jsonl"
    probe_path = run_dir / "probe.jsonl"
    probe_steps = {int(x) for x in a.probe_steps.split(",") if x.strip()}
    existing = load_latest(ckpt_dir, expected_cfg_hash=chash)
    step, init_sha, spent_before = 0, "", 0.0
    if existing is not None:
        model.load_state_dict(existing.model)
        opt.load_state_dict(existing.optimizer)
        sched.load_state_dict(existing.scheduler)
        restore_rng(existing)
        step = existing.step
        init_sha = existing.extra.get("init_sha256", "")
        spent_before = float(existing.extra.get("elapsed_s", 0.0))
        _truncate_log(log_path, step)
        log(f"RETOMADA de {existing.path.name} (passo {step})")
    else:
        init_sha = sha256_file(init)
        ck = load_checkpoint(init)
        model.load_state_dict(ck.model, strict=True)
        log(f"warm start {init} sha {init_sha[:12]}: {len(ck.model)}/{len(model.state_dict())} tensores (strict); "
            f"guard congelado: {guard_ck}")
        for pth in (log_path, probe_path):
            if pth.exists():
                pth.unlink()
    prepare_student(model)
    bn_before = bn_state_sha256(model)
    commit = git_commit()

    ds, pl = make_dataset(paths, vp)
    log(f"pools: {json.dumps(pl.stats)} | v4 {json.dumps(vp.hashable())}")
    (run_dir / "pools-stats.json").write_text(json.dumps({**pl.stats, "heldout_enroll": list(pl.heldout_enroll)},
                                                         indent=1))
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

    def save(reason: str) -> Path:
        pin = a.pin_every > 0 and step % a.pin_every == 0
        pth = save_checkpoint(ckpt_dir, step=step, model=model.state_dict(), optimizer=opt, scheduler=sched,
                              keep=a.keep, sampler_state={"step": step}, cfg_hash=chash, commit=commit, pin=pin,
                              extra={"elapsed_s": elapsed(), "init_sha256": init_sha, "bn_sha256": bn_before,
                                     "reason": reason, "args": vars(a), "v4": vp.hashable(), "presence_head": False,
                                     "gating": {"mode": "lsnr-profile", "zero_below_db": -10.0},
                                     "wall": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"checkpoint {pth.name} ({reason})")
        return pth

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
            parts = step_keep(model, ref, teacher, istft, params, hp, b, a)
            if step in probe_steps:
                pr = grad_probe(parts, trainable)
                with open(probe_path, "a") as pf:
                    pf.write(json.dumps({"step": step, **pr}) + "\n")
                worst = sorted((v, k) for k, v in pr["cos"].items() if not k.endswith("|total"))[:3]
                log(f"sonda passo {step}: normas " + " ".join(f"{k}={v:.2e}" for k, v in pr["norm"].items())
                    + " | piores cossenos " + " ".join(f"{k}={v:+.2f}" for v, k in worst))
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
            mr = max(row["mr"], 1e-9)
            for k in TERMS[1:]:
                rel_hist[k].append(row[k] / mr)
            if step % 20 == 0 or step <= 3:
                ta, ia = batch["target_active"], batch["interf_active"]
                cls = {"f_only_t": float((ta & ~ia).float().mean()), "f_only_i": float((~ta & ia).float().mean()),
                       "f_both": float((ta & ia).float().mean()), "f_none": float((~ta & ~ia).float().mean())}
                rel = {f"rel_{k}": float(np.median(v)) for k, v in rel_hist.items()}
                entry = {"step": step, "lr": float(sched.get_last_lr()[0]), **row, "grad_norm": float(gn),
                         "dt": step_times[-1], "wait": waits[-1], **cls, **rel, "wall": time.strftime("%H:%M:%S")}
                lf.write(json.dumps(entry) + "\n")
                lf.flush()
            if step % 50 == 0:
                pss_max = max(pss_max, total_pss())
                sps = statistics.median(step_times)
                rel = " ".join(f"{k}={100 * np.median(v):.1f}%" for k, v in rel_hist.items())
                eta = f" | até stop-at: {(deadline - time.time()) / 3600:.2f} h" if deadline else ""
                log(f"passo {step}/{end_step} total={row['total']:.3f} mr={row['mr']:.3f} "
                    f"s/passo={sps:.3f} espera={statistics.median(waits):.3f} lr={sched.get_last_lr()[0]:.2e} "
                    f"gSeg={row['g_seg_db']:+.1f} gOn={row['g_on_db']:+.1f} zeroOI={row['zero_frac_only_interf']:.2f} "
                    f"zeroT={row['zero_frac_target']:.3f} zeroOn={row['zero_frac_onset']:.3f} "
                    f"gap={row['guard_gap_db']:+.2f} | rel: {rel} | pss={pss_max / GIB:.1f} GiB{eta}")
                for k, v in rel_hist.items():
                    if len(v) == v.maxlen and np.median(v) < 0.01 and k not in ("df_reg", "lsnr", "t"):
                        log(f"ALARME: termo {k} abaixo de 1% de L_mr (mediana {100 * np.median(v):.2f}%)")
            if stop["flag"]:
                stop_reason = f"signal:{stop['why']}"
                break
            if deadline and time.time() >= deadline:
                stop_reason = "stop_at"
                break
            if time.monotonic() - last_ckpt >= a.ckpt_every_min * 60.0:
                save("periodic")
                last_ckpt = time.monotonic()
    if not a.no_update:
        save(f"final:{stop_reason}")
    summary = {
        "run_id": a.run_id, "step": step, "stop_reason": stop_reason, "elapsed_s": elapsed(),
        "s_per_step_median": statistics.median(step_times) if step_times else None,
        "loader_wait_median": statistics.median(waits) if waits else None,
        "rel_median": {k: float(np.median(v)) for k, v in rel_hist.items()},
        "pss_gb_max": pss_max / GIB, "bn_unchanged": bn_state_sha256(model) == bn_before,
        "vram_peak_gb": torch.cuda.max_memory_allocated(device) / GIB if device.startswith("cuda") else None,
        "args": vars(a),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"fim: {stop_reason} no passo {step}; resumo em {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
