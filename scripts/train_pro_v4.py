#!/usr/bin/env python3
"""Treino do pDFNet3 Pro v4 (`docs/plan-pro-v4.md`): N interferentes, gating novo emulado, perdas corrigidas, retomada.

Perda (§4.2):
    L = L_mr + L_os(×2 com N≥1) + λ_leak·L_leak + λ_ovl·L_ovl + λ_guard·L_guard + L_lsnr + λ_z·L_zero + λ_k·L_keep
        + L_t + L_dfreg
- ArcFace removido (gradiente identicamente zero, §4.1); `target_absence_loss` substituída por `L_leak`.
- Gating emulado por amostra: com perfil (mask=1) só ZERO (lsnr < −10) ou ERB+DF; sem perfil, o gating atual
  (−10, 30, 20).

Retomada: checkpoint (modelo + otimizador + scheduler + RNG + passo) a cada `--ckpt-every-min`, no SIGTERM/SIGINT e
na parada por relógio (`--stop-at HH:MM`, horário local). Ao relançar com o mesmo `--run-id`, retoma do último
checkpoint válido (mesmo hash de config). A sequência de dados é função pura de (seed, passo): a retomada é exata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cctrain.dfvendor import ensure_df_importable
from cctrain.hashing import sha256_file
from cctrain.losses.dfnet import (
    asymmetric_formant_loss,
    df_coef_temporal_regularization_loss,
    dfnet_loss,
    local_snr_loss,
)
from cctrain.losses.distill import distill_loss
from cctrain.m1 import pools
from cctrain.m2.config import load_stage_f_config
from cctrain.models.cross_attention import SubbandCrossAttentionConfig
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmConfig, FilmGenerator
from cctrain.models.pdfnet3_pro import PDFNet3Pro
from cctrain.paths import Paths
from cctrain.train.checkpoint import (
    config_hash,
    git_commit,
    load_checkpoint,
    load_latest,
    restore_rng,
    save_checkpoint,
)
from cctrain.train.embed_bank import ArrayEmbeddingBank
from cctrain.train.stage_c import _lsnr_target, _to_device, _truncate_log, total_pss
from cctrain.train.stage_f import HOP, StageFHParams, _gated, bn_state_sha256, prepare_student
from cctrain.data.mix_v4 import StageV4Dataset, V4Params, build_pools

ensure_df_importable()
from df.loss import Istft  # noqa: E402
from df.model import ModelParams  # noqa: E402

GIB = 1024**3
DEFAULT_INIT = "runs/m2/pro-v2-mhca-k16-tse/checkpoints/ckpt-00002000.pt"
TERMS = ("mr", "os", "leak", "ovl", "guard", "lsnr", "zero", "keep", "t", "df_reg")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Treino pDFNet3 Pro v4")
    p.add_argument("--run-id", default="pro-v4")
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--total-steps", type=int, default=80_000, help="tamanho do cosseno (e da sequência de dados)")
    p.add_argument("--max-steps", type=int, default=None, help="para neste passo (smoke)")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--formant-weight", type=float, default=2.5)
    p.add_argument("--lambda-leak", type=float, default=1.0)
    p.add_argument("--lambda-ovl", type=float, default=1.0)
    p.add_argument("--lambda-guard", type=float, default=1.0)
    p.add_argument("--lambda-zero", type=float, default=1e-3)
    p.add_argument("--lambda-keep", type=float, default=1e-3)
    p.add_argument("--lsnr-factor", type=float, default=1e-3)
    p.add_argument("--stop-at", default=None, help="HH:MM local: salva e sai limpo (próxima ocorrência)")
    p.add_argument("--ckpt-every-min", type=float, default=15.0)
    p.add_argument("--keep", type=int, default=6)
    p.add_argument("--pin-every", type=int, default=10_000)
    p.add_argument("--no-update", action="store_true", help="calibração: forward/backward sem passo do otimizador")
    p.add_argument("--grad-probe", type=int, default=0, help="nos N primeiros passos mede a norma do gradiente por termo")
    p.add_argument("--runs-subdir", default="m4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--soft-gate-tau", type=float, default=0.0, help="diagnóstico: ZERO contínuo (dB) com perfil")
    p.add_argument("--data-offset-step", type=int, default=0, help="diagnóstico: começa a sequência de dados neste passo")
    p.add_argument("--const-lr", type=float, default=0.0, help="diagnóstico: LR constante (sem cosseno/aquecimento)")
    return p.parse_args()


def stop_deadline(s: str | None) -> float | None:
    if not s:
        return None
    hh, mm = (int(x) for x in s.split(":"))
    now = dt.datetime.now()
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    return t.timestamp()


def lr_factor(step: int, warmup: int, total: int) -> float:
    warm = min(1.0, (step + 1) / max(warmup, 1))
    return warm * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(total, 1))))


# --- gating e perdas ------------------------------------------------------------------------------------------------


def gate_v4(spec, spec_m, spec_e, lsnr, profile: torch.Tensor, soft_tau: float = 0.0):
    """Gating por amostra (§3.3): perfil -> ZERO se lsnr < −10, senão ERB+DF; sem perfil -> (−10, 30, 20).

    `soft_tau` > 0 (diagnóstico): nas amostras com perfil o ZERO vira o ganho contínuo 1 − σ((−10 − lsnr)/τ), sem
    detach, para que L_mr/L_leak/L_guard cheguem ao lsnr. `zero` (estatística) continua sendo a decisão dura."""
    level = lsnr.detach().reshape(spec.shape[0], -1)
    prof = profile.reshape(-1, 1).bool()
    zero = level < -10.0
    none = (~zero) & (level > 30.0) & (~prof)
    erb_only = (~zero) & (~none) & (level > 20.0) & (~prof)

    def w(m):
        return m[:, None, :, None, None]

    out = torch.where(w(zero), torch.zeros_like(spec_e), spec_e)
    if soft_tau > 0:
        g = 1.0 - torch.sigmoid((-10.0 - lsnr.reshape(spec.shape[0], -1)) / soft_tau)
        out = torch.where(w(prof.expand_as(zero)), spec_e * w(g), out)
    out = torch.where(w(erb_only), spec_m, out)
    out = torch.where(w(none), spec, out)
    return out, zero


def _pow(spec: torch.Tensor) -> torch.Tensor:
    """|X|² por bin: spec [B,1,T,F,2] -> [B,T,F]."""
    return (spec[:, 0, ..., 0] ** 2 + spec[:, 0, ..., 1] ** 2)


def leak_loss(out, spec, fmask):
    """L_leak (§4.2): mean_q relu(10·log10((E_out+ε)/(E_mix+ε)) + 40)/40 nos quadros perfil ∧ ¬alvo ∧ interf."""
    n = fmask.sum()
    if float(n) == 0.0:
        return out.sum() * 0.0
    eo = _pow(out).sum(-1)
    em = _pow(spec).sum(-1)
    eps = 1e-10
    r = F.relu(10 * torch.log10((eo + eps) / (em + eps)) + 40.0) / 40.0
    return (r * fmask).sum() / n


def ovl_loss(out, spec_clean, spec_interf, fmask):
    """L_ovl (§4.2): relu(⟨|Y|,|I|⟩/‖I‖² − ⟨|S|,|I|⟩/‖I‖²) nos quadros perfil ∧ alvo ∧ interf."""
    n = fmask.sum()
    if float(n) == 0.0:
        return out.sum() * 0.0
    y = (_pow(out) + 1e-12).sqrt()
    s = (_pow(spec_clean) + 1e-12).sqrt()
    i = (_pow(spec_interf) + 1e-12).sqrt()
    ii = (i * i).sum(-1) + 1e-8
    v = F.relu(((y * i).sum(-1) - (s * i).sum(-1)) / ii)
    return (v * fmask).sum() / n


def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + 1e-8)
    s = a * ref
    e = est - s
    return 10 * torch.log10((s * s).sum(-1) + 1e-8) - 10 * torch.log10((e * e).sum(-1) + 1e-8)


def step_v4(model, teacher, istft, df_params, hp: StageFHParams, b: dict, a) -> dict[str, torch.Tensor]:
    spec, spec_clean, mask = b["spec"], b["spec_clean"], b["mask"]
    nb = spec.shape[0]
    T = spec.shape[2]
    profile = mask.reshape(nb) > 0
    spec_e, m, lsnr, df_coefs = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], mask)
    spec_m = model.base.mask(spec.clone(), m)
    out, zero = gate_v4(spec, spec_m, spec_e, lsnr, profile, soft_tau=getattr(a, "soft_gate_tau", 0.0))
    est, tgt = istft(out), istft(spec_clean)
    lsnr_t = _lsnr_target(df_params, spec_clean, spec)
    kw = dict(n_ffts=hp.fft_sizes, gamma=hp.loss_gamma)

    ta = b["target_active"][:, :T]
    ia = b["interf_active"][:, :T]
    ds = b["digital_silence"][:, :T]
    prof = profile[:, None]
    f_leak = (prof & ~ta & ia & ~ds).float()
    f_ovl = (prof & ta & ia).float()
    f_keep = (prof & ta & ~ds)

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

    # L_lsnr: todos os quadros menos silêncio digital (inclui só-interferente com perfil)
    ls = local_snr_loss(lsnr, lsnr_t, ~ds, factor=a.lsnr_factor)
    lv = lsnr.reshape(nb, -1)
    nz = f_leak.sum()
    l_zero = (F.relu(lv + 12.0) ** 2 * f_leak).sum() / nz if float(nz) > 0 else lv.sum() * 0.0
    fk = f_keep.float()
    nk = fk.sum()
    l_keep = (F.relu(-8.0 - lv) ** 2 * fk).sum() / nk if float(nk) > 0 else lv.sum() * 0.0

    # L_guard: amostras N=0 com perfil; referência = o próprio aluno sem perfil (no_grad), ambos com gating emulado
    idx_g = (profile & (b["n_interf"] == 0) & b["has_target"]).nonzero(as_tuple=True)[0]
    if idx_g.numel():
        with torch.no_grad():
            z = torch.zeros_like(mask[idx_g])
            e0, m0, l0, _ = model(spec[idx_g].clone(), b["feat_erb"][idx_g], b["feat_spec"][idx_g],
                                  b["embedding"][idx_g], z)
            sm0 = model.base.mask(spec[idx_g].clone(), m0)
            o0, _ = gate_v4(spec[idx_g], sm0, e0, l0, torch.zeros(idx_g.numel(), device=spec.device, dtype=torch.bool))
            sd0 = si_sdr(istft(o0), tgt[idx_g])
        sdp = si_sdr(est[idx_g], tgt[idx_g])
        guard = F.relu(sd0 - sdp).mean() / 10.0
    else:
        guard = est.sum() * 0.0

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
             "guard": a.lambda_guard * guard, "lsnr": ls, "zero": a.lambda_zero * l_zero,
             "keep": a.lambda_keep * l_keep, "t": t, "df_reg": l_df}
    total = sum(parts.values())
    with torch.no_grad():
        oi = f_leak.bool()
        stats = {
            "raw_leak": leak.detach(), "raw_ovl": ovl.detach(), "raw_guard": guard.detach(),
            "raw_zero": l_zero.detach(), "raw_keep": l_keep.detach(),
            "zero_frac_only_interf": (zero & oi).sum() / oi.sum().clamp(min=1),
            "zero_frac_target": (zero & f_keep).sum() / f_keep.sum().clamp(min=1),
            "n_guard": torch.tensor(float(idx_g.numel())), "n0": torch.tensor(float(idx0.numel())),
        }
    return {"total": total, **parts, **stats}


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

    base, df_state = load_upstream_dfnet3(device=device)
    teacher, _ = load_upstream_dfnet3(device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    cfg = load_stage_f_config()
    hp = StageFHParams.from_config(cfg, run_id=a.run_id, num_workers=a.workers, batch_size=a.batch_size)
    gen = FilmGenerator(FilmConfig(log_gamma_bound=hp.log_gamma_bound, beta_bound=hp.beta_bound)).to(device)
    attn_cfg = SubbandCrossAttentionConfig(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4,
                                           num_tokens=16, token_dim=128)
    model = PDFNet3Pro(base, gen, cross_attention_config=attn_cfg, enable_cross_attention=True).to(device)
    prepare_student(model)
    istft = Istft(960, HOP, torch.as_tensor(df_state.fft_window().copy())).to(device)
    params = ModelParams()

    vp = V4Params(batch_size=a.batch_size, total_steps=a.total_steps)
    hashed = {"v4": vp.hashable(), "lr": a.lr, "warmup": a.warmup, "total_steps": a.total_steps,
              "batch": a.batch_size, "lambdas": [a.lambda_leak, a.lambda_ovl, a.lambda_guard, a.lambda_zero,
                                                 a.lambda_keep, a.lsnr_factor], "fw": a.formant_weight}
    if a.soft_gate_tau or a.data_offset_step or a.const_lr:
        hashed["diag"] = [a.soft_gate_tau, a.data_offset_step, a.const_lr]
    chash = config_hash(hashed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=hp.weight_decay)
    if a.const_lr > 0:
        for g in opt.param_groups:
            g["lr"] = a.const_lr
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, a.warmup, a.total_steps))

    log_path = run_dir / "loss-log.jsonl"
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
        init = Path(a.init_ckpt) if a.init_ckpt else paths.repo.parent / "clearcore-train" / DEFAULT_INIT
        if not init.is_file():
            init = Path("/home/joaorura/orca/projects/clearcore-train") / DEFAULT_INIT
        init_sha = sha256_file(init)
        ck = load_checkpoint(init)
        cur = model.state_dict()
        compat = {k: v for k, v in ck.model.items() if k in cur and v.shape == cur[k].shape}
        missing = sorted(set(cur) - set(compat))
        model.load_state_dict(compat, strict=False)
        log(f"warm start {init} sha {init_sha[:12]}: {len(compat)}/{len(cur)} tensores; faltando {missing[:5]}")
        if log_path.exists():
            log_path.unlink()
    prepare_student(model)
    bn_before = bn_state_sha256(model)
    commit = git_commit()

    bank = ArrayEmbeddingBank.load(paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz")
    speech = pools.by_split(pools.speech_records(paths), "train")
    noise_train, _ = pools.noise_pools()
    pl = build_pools(speech, noise_train, pools.rir_records(), bank, paths.data, seed=vp.seed)
    log(f"pools: {json.dumps(pl.stats)}")
    (run_dir / "pools-stats.json").write_text(json.dumps({**pl.stats, "heldout_enroll": list(pl.heldout_enroll)},
                                                         indent=1))
    ds = StageV4Dataset(pl, bank, paths.data, vp)
    sampler = list(range((step + a.data_offset_step) * a.batch_size, len(ds)))
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
                                     "reason": reason, "args": vars(a), "v4": vp.hashable(),
                                     "wall": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"checkpoint {pth.name} ({reason})")
        return pth

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    end_step = min(a.total_steps, a.max_steps) if a.max_steps else a.total_steps
    stop_reason = "total_steps"
    grad_probe_rows = []
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
            parts = step_v4(model, teacher, istft, params, hp, b, a)
            if a.grad_probe and step < a.grad_probe:
                row = {}
                for k in TERMS:
                    v = parts[k]
                    if not v.requires_grad:
                        row[k] = 0.0
                        continue
                    gs = torch.autograd.grad(v, trainable, retain_graph=True, allow_unused=True)
                    row[k] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)
                                              if any(g is not None for g in gs) else torch.tensor(0.0)))
                grad_probe_rows.append(row)
                log(f"grad-norm passo {step}: " + " ".join(f"{k}={v:.3e}" for k, v in row.items()))
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
            step_times.append(time.perf_counter() - t0)
            row = {k: float(v.detach()) for k, v in parts.items()}
            mr = max(row["mr"], 1e-9)
            for k in ("os", "leak", "ovl", "guard", "lsnr", "zero", "keep", "t", "df_reg"):
                rel_hist[k].append(row[k] / mr)
            if step % 20 == 0 or step <= 3:
                ta, ia = batch["target_active"], batch["interf_active"]
                cls = {"f_only_t": float((ta & ~ia).float().mean()), "f_only_i": float((~ta & ia).float().mean()),
                       "f_both": float((ta & ia).float().mean()), "f_none": float((~ta & ~ia).float().mean())}
                rel = {f"rel_{k}": float(np.median(v)) for k, v in rel_hist.items()}
                entry = {"step": step, "lr": float(sched.get_last_lr()[0]), **row, "grad_norm": float(gn),
                         "dt": step_times[-1], "wait": waits[-1], **cls, **rel,
                         "wall": time.strftime("%H:%M:%S")}
                lf.write(json.dumps(entry) + "\n")
                lf.flush()
            if step % 100 == 0:
                pss_max = max(pss_max, total_pss())
                sps = statistics.median(step_times)
                rel = " ".join(f"{k}={100 * np.median(v):.1f}%" for k, v in rel_hist.items())
                eta = ""
                if deadline:
                    eta = f" | até stop-at: {(deadline - time.time()) / 3600:.2f} h"
                log(f"passo {step}/{end_step} total={row['total']:.3f} mr={row['mr']:.3f} "
                    f"s/passo={sps:.3f} espera={statistics.median(waits):.3f} lr={sched.get_last_lr()[0]:.2e} "
                    f"zeroOI={row['zero_frac_only_interf']:.2f} zeroT={row['zero_frac_target']:.3f} | rel: {rel} "
                    f"| pss={pss_max / GIB:.1f} GiB{eta}")
                for k, v in rel_hist.items():
                    if len(v) == v.maxlen and np.median(v) < 0.01 and k not in ("df_reg",):
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
    bn_after = bn_state_sha256(model)
    summary = {
        "run_id": a.run_id, "step": step, "stop_reason": stop_reason, "elapsed_s": elapsed(),
        "s_per_step_median": statistics.median(step_times) if step_times else None,
        "loader_wait_median": statistics.median(waits) if waits else None,
        "rel_median": {k: float(np.median(v)) for k, v in rel_hist.items()},
        "pss_gb_max": pss_max / GIB, "bn_unchanged": bn_after == bn_before,
        "vram_peak_gb": torch.cuda.max_memory_allocated(device) / GIB if device.startswith("cuda") else None,
        "grad_probe": grad_probe_rows, "args": vars(a),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"fim: {stop_reason} no passo {step}; resumo em {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
