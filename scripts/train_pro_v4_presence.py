#!/usr/bin/env python3
"""Treino do pDFNet3 Pro v4-presence: cabeça `p_target` (presença do falante-alvo) e gating por presença com perfil.

Motivação (diagnóstico do v3.1, `scripts/diag_pro_v31.py`): o gating com perfil decidido pelo `lsnr` (saída de
`base.enc`, antes da cross-attention) não separa alvo de interferente; L_zero e L_keep agem só no `lsnr` com gradientes
opostos (cosseno −0,99) e o `lsnr` fica comprimido em −12..−5 em volta do limiar. Mudanças (b)+(c):

1. `PresenceHead` sobre `emb_conditioned` (DEPOIS da cross-attention, dentro do grafo `enc`), sigmoide por quadro.
   Rótulo = `target_active` do `mix_v4` (VAD relativo à trilha do alvo limpo já escalonada: quadro a até 30 dB do
   máximo da própria trilha, piso −75 dBFS; o mesmo critério em todos os níveis). Perda = BCE com pesos por classe de
   quadro (só-alvo, sobreposição, só-interferente, silêncio), só nas amostras com perfil, fora do silêncio digital.
2. Gating com perfil: ZERO se a presença emulada (histerese τ_on/τ_off + hangover de H quadros, a mesma máquina de
   estados do runtime) estiver desligada, senão ERB+DF. Sem perfil: o gating por `lsnr` de sempre (−10, 30, 20).
   Diferenciável por straight-through: forward = decisão dura (o que o runtime faz), backward = derivada de
   σ((logit − logit(τ_on))/T), centrada no limiar. Assim L_mr/L_os/L_leak/L_ovl chegam à decisão ZERO.
3. Sem L_zero/L_keep. `L_lsnr` só no caminho sem perfil, com o alvo original (`spec_clean` = `clean_all` quando
   mask=0: interferente conta como fala, a semântica do DFNet3 que o runtime sem perfil usa).
4. `L_guard` contra uma cópia CONGELADA do v3.1 sem perfil (mask=0, gating de sempre): relu(SI-SDR(ref) − margem −
   SI-SDR(perfil)) nas amostras N=0 com perfil; P(N=0) sobe para 0,35.
5. Partida do v3.1 final, cabeça nova (viés inicial p=0,9: começa como ERB+DF em todo quadro).

Retomada como no `train_pro_v4.py` (checkpoint periódico, SIGTERM, `--stop-at`).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from train_pro_v4 import leak_loss, log, lr_factor, ovl_loss, si_sdr, stop_deadline  # noqa: E402
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
from cctrain.models.cross_attention import SubbandCrossAttentionConfig  # noqa: E402
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3  # noqa: E402
from cctrain.models.film import FilmConfig, FilmGenerator  # noqa: E402
from cctrain.models.pdfnet3_pro import PDFNet3Pro  # noqa: E402
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
from cctrain.data.mix_v4 import N_PROBS, StageV4Dataset, V4Params, build_pools  # noqa: E402

ensure_df_importable()
from df.loss import Istft  # noqa: E402
from df.model import ModelParams  # noqa: E402

GIB = 1024**3
V31_CKPT = "runs/m4/pro-v4/checkpoints/ckpt-00079998.pt"
TERMS = ("mr", "os", "leak", "ovl", "guard", "lsnr", "bce", "t", "df_reg")
# participação de cada classe de quadro na BCE (só-alvo, sobreposição, só-interferente, silêncio/ruído)
CLASS_SHARE = (0.25, 0.25, 0.30, 0.20)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Treino pDFNet3 Pro v4-presence")
    p.add_argument("--run-id", default="pro-v4-presence")
    p.add_argument("--init-ckpt", default=None, help=f"padrão: {V31_CKPT}")
    p.add_argument("--guard-ckpt", default=None, help=f"referência congelada do L_guard (padrão: {V31_CKPT})")
    p.add_argument("--total-steps", type=int, default=85_000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-5, help="pico do tronco (parte do v3.1 já ajustado)")
    p.add_argument("--head-lr", type=float, default=1e-3, help="pico da cabeça p_target (nova)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--n-probs", default="0.35,0.30,0.17,0.11,0.07")
    p.add_argument("--phase1-frac", type=float, default=0.0, help="o v3.1 já passou pelo currículo")
    p.add_argument("--formant-weight", type=float, default=2.5)
    p.add_argument("--lambda-bce", type=float, default=1.0)
    p.add_argument("--lambda-leak", type=float, default=2.0)
    p.add_argument("--lambda-ovl", type=float, default=20.0)
    p.add_argument("--lambda-guard", type=float, default=4.0)
    p.add_argument("--lsnr-factor", type=float, default=1e-3)
    p.add_argument("--bce-n0-mult", type=float, default=1.0, help="peso extra da BCE nos quadros do alvo com N=0")
    p.add_argument("--bce-trunk-scale", type=float, default=20.0,
                   help="escala do gradiente da cabeça para o tronco (BCE ~5%% da norma do gradiente do tronco)")
    p.add_argument("--guard-margin", type=float, default=0.5, help="dB")
    p.add_argument("--tau-on", type=float, default=0.5)
    p.add_argument("--tau-off", type=float, default=0.35)
    p.add_argument("--hangover", type=int, default=5, help="quadros de 10 ms")
    p.add_argument("--st-temp", type=float, default=1.0, help="temperatura da sigmoide substituta (logit)")
    p.add_argument("--st-scale", type=float, default=1.0,
                   help="escala do gradiente straight-through (L_mr/L_os/L_leak/L_ovl -> decisão)")
    p.add_argument("--stop-at", default=None)
    p.add_argument("--ckpt-every-min", type=float, default=15.0)
    p.add_argument("--keep", type=int, default=6)
    p.add_argument("--pin-every", type=int, default=10_000)
    p.add_argument("--no-update", action="store_true", help="calibração: sem passo do otimizador")
    p.add_argument("--runs-subdir", default="m4")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


# --- modelo ----------------------------------------------------------------------------------------------------------


def build_model(device: str, hp: StageFHParams, presence: bool = True) -> PDFNet3Pro:
    base, _ = load_upstream_dfnet3(device=device)
    gen = FilmGenerator(FilmConfig(log_gamma_bound=hp.log_gamma_bound, beta_bound=hp.beta_bound)).to(device)
    attn_cfg = SubbandCrossAttentionConfig(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4,
                                           num_tokens=16, token_dim=128)
    return PDFNet3Pro(base, gen, cross_attention_config=attn_cfg, enable_cross_attention=True,
                      presence_head=presence).to(device)


def frozen_reference(path: Path, device: str, hp: StageFHParams) -> PDFNet3Pro:
    ref = build_model(device, hp, presence=False)
    ck = load_checkpoint(path)
    ref.load_state_dict(ck.model, strict=True)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def head_params(model: PDFNet3Pro) -> list[nn.Parameter]:
    return [p for p in model.presence.parameters() if p.requires_grad]


def trunk_params(model: PDFNet3Pro) -> list[nn.Parameter]:
    hid = {id(p) for p in model.presence.parameters()}
    return [p for p in model.parameters() if p.requires_grad and id(p) not in hid]


# --- gating ----------------------------------------------------------------------------------------------------------


def presence_state_machine(p: np.ndarray, tau_on: float, tau_off: float, hangover: int) -> np.ndarray:
    """Decisão do runtime por quadro, p [B,T] -> passa [B,T] bool. Estado inicial: inativo, hangover 0.

        ativo = p >= τ_off se ativo no quadro anterior, senão p >= τ_on   (histerese)
        hang  = H se ativo; senão max(hang − 1, 0)                        (hangover)
        passa = ativo ou hang > 0   (passa = ERB+DF; não passa = ZERO)
    """
    b, t = p.shape
    active = np.zeros(b, bool)
    hang = np.zeros(b, np.int64)
    out = np.zeros((b, t), bool)
    for i in range(t):
        pi = p[:, i]
        active = np.where(active, pi >= tau_off, pi >= tau_on)
        hang = np.where(active, hangover, np.maximum(hang - 1, 0))
        out[:, i] = active | (hang > 0)
    return out


def gate_presence(spec, spec_m, spec_e, lsnr, p_logit, profile, a):
    """Saída por amostra. Perfil: g·spec_e, g = decisão dura (máquina de estados) com gradiente straight-through da
    sigmoide centrada em τ_on. Sem perfil: seleção por lsnr (ZERO < −10, ruidoso > 30, só ERB em (20, 30])."""
    nb = spec.shape[0]
    level = lsnr.detach().reshape(nb, -1)
    zl = level < -10.0
    none = (~zl) & (level > 30.0)
    erb_only = (~zl) & (~none) & (level > 20.0)

    def w(m):
        return m[:, None, :, None, None]

    out_np = torch.where(w(zl), torch.zeros_like(spec_e), spec_e)
    out_np = torch.where(w(erb_only), spec_m, out_np)
    out_np = torch.where(w(none), spec, out_np)
    if p_logit is None:
        return out_np, zl, None
    prob = torch.sigmoid(p_logit.detach()).float().cpu().numpy()
    hard = torch.from_numpy(presence_state_machine(prob, a.tau_on, a.tau_off, a.hangover)).to(spec.device)
    soft = torch.sigmoid((p_logit - math.log(a.tau_on / (1.0 - a.tau_on))) / a.st_temp)
    g = hard.to(soft.dtype) + getattr(a, "st_scale", 1.0) * (soft - soft.detach())
    out_p = spec_e * g[:, None, :, None, None].to(spec_e.dtype)
    prof = profile.reshape(-1).bool()
    out = torch.where(prof[:, None, None, None, None], out_p, out_np)
    zero = torch.where(prof[:, None], ~hard, zl)
    return out, zero, hard


def bce_presence(p_logit, ta, ia, valid, n0_target, n0_mult: float):
    """BCE com pesos por classe de quadro: cada classe presente recebe a sua fração de CLASS_SHARE (renormalizada)."""
    y = ta.float()
    classes = (ta & ~ia, ta & ia, ~ta & ia, ~ta & ~ia)
    wt = torch.zeros_like(y)
    shares = [s if bool((c & valid).any()) else 0.0 for c, s in zip(classes, CLASS_SHARE)]
    tot = sum(shares)
    if tot == 0:
        return p_logit.sum() * 0.0
    for c, s in zip(classes, shares):
        cv = c & valid
        n = cv.sum()
        if s > 0:
            wt = wt + cv.float() * (s / tot) / n.float()
    if n0_mult != 1.0:
        wt = wt * torch.where(n0_target & valid, torch.full_like(wt, n0_mult), torch.ones_like(wt))
        wt = wt / wt.sum().clamp(min=1e-12)
    bce = F.binary_cross_entropy_with_logits(p_logit.float(), y, reduction="none")
    return (bce * wt).sum()


# --- passo -----------------------------------------------------------------------------------------------------------


def step_presence(model, ref, teacher, istft, df_params, hp: StageFHParams, b: dict, a) -> dict[str, torch.Tensor]:
    spec, spec_clean, mask = b["spec"], b["spec_clean"], b["mask"]
    nb, T = spec.shape[0], spec.shape[2]
    profile = mask.reshape(nb) > 0
    spec_e, m, lsnr, df_coefs, p_logit = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], mask,
                                               return_presence=True)
    spec_m = model.base.mask(spec.clone(), m)
    out, zero, _hard = gate_presence(spec, spec_m, spec_e, lsnr, p_logit, profile, a)
    est, tgt = istft(out), istft(spec_clean)
    kw = dict(n_ffts=hp.fft_sizes, gamma=hp.loss_gamma)

    ta = b["target_active"][:, :T]
    ia = b["interf_active"][:, :T]
    ds = b["digital_silence"][:, :T]
    prof = profile[:, None]
    f_leak = (prof & ~ta & ia & ~ds).float()
    f_ovl = (prof & ta & ia).float()
    f_keep = prof & ta & ~ds

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

    # L_lsnr: só o caminho sem perfil (alvo original; spec_clean = clean_all quando mask=0)
    lsnr_t = _lsnr_target(df_params, spec_clean, spec)
    ls = local_snr_loss(lsnr, lsnr_t, (~prof) & ~ds, factor=a.lsnr_factor)

    # BCE da presença: amostras com perfil, fora do silêncio digital
    valid = prof & ~ds
    n0t = (b["n_interf"] == 0)[:, None] & ta
    bce = bce_presence(p_logit, ta, ia, valid, n0t, a.bce_n0_mult)

    # L_guard: N=0 com perfil contra a referência congelada (v3.1 sem perfil, gating de sempre)
    idx_g = (profile & (b["n_interf"] == 0) & b["has_target"]).nonzero(as_tuple=True)[0]
    if idx_g.numel():
        with torch.no_grad():
            z = torch.zeros_like(mask[idx_g])
            e0, m0, l0, _ = ref(spec[idx_g].clone(), b["feat_erb"][idx_g], b["feat_spec"][idx_g],
                                b["embedding"][idx_g], z)
            sm0 = ref.base.mask(spec[idx_g].clone(), m0)
            o0, _, _ = gate_presence(spec[idx_g], sm0, e0, l0, None, z.reshape(-1) > 0, a)
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
             "guard": a.lambda_guard * guard, "lsnr": ls, "bce": a.lambda_bce * bce, "t": t, "df_reg": l_df}
    total = sum(parts.values())
    with torch.no_grad():
        oi = f_leak.bool()
        pr = torch.sigmoid(p_logit.float())
        pv = valid
        stats = {
            "raw_leak": leak.detach(), "raw_ovl": ovl.detach(), "raw_guard": guard.detach(), "raw_bce": bce.detach(),
            "guard_gap_db": guard_gap,
            "zero_frac_only_interf": (zero & oi).sum() / oi.sum().clamp(min=1),
            "zero_frac_target": (zero & f_keep).sum() / f_keep.sum().clamp(min=1),
            "p_mean_target": (pr * (pv & ta)).sum() / (pv & ta).sum().clamp(min=1),
            "p_mean_only_interf": (pr * oi).sum() / oi.sum().clamp(min=1),
            "n_guard": torch.tensor(float(idx_g.numel())), "n0": torch.tensor(float(idx0.numel())),
        }
    return {"total": total, **parts, **stats, "_p_logit": p_logit}


# --- dados -----------------------------------------------------------------------------------------------------------


def v4_params(a) -> V4Params:
    npr = tuple(float(x) for x in a.n_probs.split(","))
    if len(npr) != 5 or abs(sum(npr) - 1.0) > 1e-6:
        raise SystemExit(f"--n-probs inválido: {npr}")
    return V4Params(seed=a.seed, batch_size=a.batch_size, total_steps=a.total_steps, phase1_frac=a.phase1_frac,
                    n_probs=npr if npr != N_PROBS else N_PROBS)


def make_dataset(paths: Paths, vp: V4Params):
    bank = ArrayEmbeddingBank.load(paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz")
    speech = pools.by_split(pools.speech_records(paths), "train")
    noise_train, _ = pools.noise_pools()
    pl = build_pools(speech, noise_train, pools.rir_records(), bank, paths.data, seed=vp.seed)
    return StageV4Dataset(pl, bank, paths.data, vp), pl


def resolve(paths: Paths, p: str | None) -> Path:
    if p:
        return Path(p)
    for root in (paths.repo, paths.repo.parent / "clearcore-train", Path.home() / "orca/projects/clearcore-train"):
        c = root / V31_CKPT
        if c.is_file():
            return c
    return paths.runs / "m4/pro-v4/checkpoints/ckpt-00079998.pt"


# --- main ------------------------------------------------------------------------------------------------------------


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
    model = build_model(device, hp, presence=True)
    model.presence_trunk_grad_scale = a.bce_trunk_scale
    prepare_student(model)
    init = resolve(paths, a.init_ckpt)
    guard_ck = resolve(paths, a.guard_ckpt)
    ref = frozen_reference(guard_ck, device, hp)
    istft = Istft(960, HOP, torch.as_tensor(df_state.fft_window().copy())).to(device)
    params = ModelParams()

    vp = v4_params(a)
    hashed = {"v4": vp.hashable(), "lr": a.lr, "head_lr": a.head_lr, "warmup": a.warmup,
              "total_steps": a.total_steps, "batch": a.batch_size,
              "lambdas": [a.lambda_bce, a.lambda_leak, a.lambda_ovl, a.lambda_guard, a.lsnr_factor, a.bce_n0_mult,
                          a.bce_trunk_scale],
              "gate": [a.tau_on, a.tau_off, a.hangover, a.st_temp] + ([a.st_scale] if a.st_scale != 1.0 else []), "guard_margin": a.guard_margin,
              "fw": a.formant_weight, "guard_ckpt": str(guard_ck)}
    chash = config_hash(hashed)
    opt = torch.optim.AdamW([{"params": trunk_params(model), "lr": a.lr},
                             {"params": head_params(model), "lr": a.head_lr}], weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, a.warmup, a.total_steps))
    trainable = [p for g in opt.param_groups for p in g["params"]]

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
        init_sha = sha256_file(init)
        ck = load_checkpoint(init)
        cur = model.state_dict()
        compat = {k: v for k, v in ck.model.items() if k in cur and v.shape == cur[k].shape}
        missing = sorted(set(cur) - set(compat))
        model.load_state_dict(compat, strict=False)
        bad = [k for k in missing if not k.startswith("presence.")]
        if bad:
            raise SystemExit(f"warm start incompleto fora da cabeça nova: {bad[:5]}")
        log(f"warm start {init} sha {init_sha[:12]}: {len(compat)}/{len(cur)} tensores; novos: {len(missing)} "
            f"(cabeça p_target); guard congelado: {guard_ck}")
        if log_path.exists():
            log_path.unlink()
    prepare_student(model)
    bn_before = bn_state_sha256(model)
    commit = git_commit()

    ds, pl = make_dataset(paths, vp)
    log(f"pools: {json.dumps(pl.stats)} | n_probs {vp.n_probs}")
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
                                     "reason": reason, "args": vars(a), "v4": vp.hashable(), "presence_head": True,
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
            parts = step_presence(model, ref, teacher, istft, params, hp, b, a)
            parts.pop("_p_logit")
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
                log(f"passo {step}/{end_step} total={row['total']:.3f} mr={row['mr']:.3f} bce={row['raw_bce']:.3f} "
                    f"s/passo={sps:.3f} espera={statistics.median(waits):.3f} lr={sched.get_last_lr()[0]:.2e} "
                    f"zeroOI={row['zero_frac_only_interf']:.2f} zeroT={row['zero_frac_target']:.3f} "
                    f"pT={row['p_mean_target']:.2f} pOI={row['p_mean_only_interf']:.2f} "
                    f"gap={row['guard_gap_db']:+.2f} | rel: {rel} | pss={pss_max / GIB:.1f} GiB{eta}")
                for k, v in rel_hist.items():
                    if len(v) == v.maxlen and np.median(v) < 0.01 and k not in ("df_reg", "lsnr"):
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
