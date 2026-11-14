#!/usr/bin/env python3
"""Treino do pDFNet3 Pro v5: rede base CONGELADA + camadas do Pro (condicionamento) e adaptadores LoRA/FiLM treináveis.

Base (`--base`): `dfn3` = DFNet3 publicado (upstream, o mesmo do eval `dfn3`), padrão; `v31` = v3.1 (controle).
Com `dfn3` tudo o que o Pro acrescenta começa do zero (init padrão, ramos de saída zerados: FiLM.out, cross_attn.out_proj
e subband_gain_head, freq_mixer.pw_conv, LoRA B, FiLM das GRUs) e nada vem do v3.1: no passo 0 o modelo com perfil é
idêntico ao DFNet3, e sem perfil é sempre bit-idêntico ao DFNet3 (adaptadores desligados, condicionamento × 0).

Variantes (`--variant`): P1 = só condicionamento (film, cross_attn, freq_mixer); P2 = P1 + LoRA r8 nas 3 GRUs
(enc.emb_gru, erb_dec.emb_gru, df_dec.df_gru; entrada e recorrente); P3 = P2 + LoRA r16 no encoder (5 convs 1×1 +
df_fc_emb); ctrl = base v3.1 + condicionamento + LoRA r8 nas GRUs (a variante B anterior). `--gru-film` acrescenta FiLM do
embedding na saída das GRUs.

Perdas: as do smoke C do v4.2 (L_mr, L_os, leak 3, ovl 10, lsnr, turn 0,7, n0g 1,0, tail 7,5, leak alvo-ausente 3 com
alvo ausente 20%, df_reg). Saem as que ancoravam no v3.1 ou na base (guard, L2-SP, L_n0, destilação sem perfil: com a
base congelada a saída sem perfil é a própria base e nenhum parâmetro treinável a altera).

Velocidade: bf16 autocast no forward (`--amp bf16`; perdas, STFT/iSTFT em fp32), batch GLOBAL fixo (`--global-batch`,
repartido entre os ranks: retomar com outro nº de GPUs mantém a mesma sequência de dados e o mesmo cosseno), multi-GPU
por `torchrun` com all-reduce manual só dos parâmetros treináveis, DataLoader com workers persistentes, pin_memory.
LR = `--lr` × sqrt(global_batch / 12). Orçamento em amostras (`--total-samples`, padrão 40k passos × 12).

Guarda: só técnica — `--nan-patience` passos seguidos com perda não finita, ou parâmetro não finito, salvam e param.
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
import warnings
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from train_pro_v4 import gate_v4, leak_loss, log, lr_factor, ovl_loss, stop_deadline  # noqa: E402
from train_pro_v42 import make_dataset, since_interferer_s, target_shadow, turn_loss  # noqa: E402
from train_pro_v4_keep import _floats  # noqa: E402
from train_pro_v4_presence import V31_CKPT, build_model  # noqa: E402
from cctrain.dfvendor import ensure_df_importable  # noqa: E402
from cctrain.losses.dfnet import (  # noqa: E402
    asymmetric_formant_loss,
    df_coef_temporal_regularization_loss,
    dfnet_loss,
    local_snr_loss,
)
from cctrain.m2.config import load_stage_f_config  # noqa: E402
from cctrain.models.adapters import AdapterConfig, apply_trainable, attach_adapters, param_groups  # noqa: E402
from cctrain.paths import Paths  # noqa: E402
from cctrain.train.checkpoint import (  # noqa: E402
    config_hash,
    git_commit,
    load_checkpoint,
    load_latest,
    restore_rng,
    save_checkpoint,
)
from cctrain.train.stage_c import _lsnr_target, _to_device  # noqa: E402
from cctrain.train.stage_f import HOP, StageFHParams, prepare_student  # noqa: E402
from cctrain.data.mix_v4 import N_PROBS, PATTERN_PROBS, V4Params  # noqa: E402
from cctrain.data.workers import init_worker  # noqa: E402
from cctrain.data.gpu.mixer import GpuMixer  # noqa: E402
from cctrain.data.gpu.recipe import StageV4RecipeDataset, collate_recipes  # noqa: E402

ensure_df_importable()
from df.loss import Istft  # noqa: E402
from df.model import ModelParams  # noqa: E402

GIB = 1024**3
DATA_TAG = "mix_v4 fix dry-gain 0145934"  # o fix não muda o hash do V4Params: fica explícito no hash do run
GPU_MIX_TAG = "gpu-mix 56bc2f5"  # equivalente ao caminho CPU (fora do hash); registrado no log/extra/summary
TERMS = ("mr", "os", "leak", "leak_abs", "tail", "ovl", "lsnr", "turn", "n0g", "df_reg")
REF_BATCH = 12
VARIANTS = {  # base, treina condicionamento, gru_rank, enc_rank
    "P1": ("dfn3", True, 0, 0),
    "P2": ("dfn3", True, 8, 0),
    "P3": ("dfn3", True, 8, 16),
    "ctrl": ("v31", True, 8, 0),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Treino pDFNet3 Pro v5 (base congelada + adaptadores)")
    p.add_argument("--run-id", default="pro-v5-p3")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="P3")
    p.add_argument("--base", choices=("dfn3", "v31"), default=None, help="sobrepõe a base da variante")
    p.add_argument("--v31-ckpt", default=None, help="checkpoint do v3.1 (só --base v31)")
    p.add_argument("--gru-rank", type=int, default=None)
    p.add_argument("--enc-rank", type=int, default=None)
    p.add_argument("--alpha-over-rank", type=float, default=1.0, help="escala s = alpha/rank do LoRA")
    p.add_argument("--gru-film", action="store_true", help="FiLM do embedding na saída das GRUs (zero-init)")
    p.add_argument("--no-train-cond", action="store_true", help="congela também o condicionamento (só adaptadores)")
    p.add_argument("--global-batch", type=int, default=48, help="batch global (repartido entre os ranks)")
    p.add_argument("--total-samples", type=int, default=40_000 * REF_BATCH, help="orçamento (40k passos × 12)")
    p.add_argument("--total-steps", type=int, default=None, help="sobrepõe total-samples/global-batch")
    p.add_argument("--max-steps", type=int, default=None, help="para neste passo (smoke)")
    p.add_argument("--warmup-samples", type=int, default=12_000)
    p.add_argument("--lr", type=float, default=3e-4, help="LR no batch de referência 12 (escala sqrt)")
    p.add_argument("--lr-scale", choices=("sqrt", "none"), default="sqrt")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=None, help="padrão: o do stage F")
    p.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    p.add_argument("--compile", action="store_true", help="torch.compile nos decoders (experimental)")
    p.add_argument("--segment-s", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=None, help="por rank; padrão CPUs/rank − 1")
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261114)
    p.add_argument("--n-probs", default=",".join(str(x) for x in N_PROBS))
    p.add_argument("--pattern-probs", default="0.25,0.40,0.20,0.15")
    p.add_argument("--p-interf-first", type=float, default=0.7)
    p.add_argument("--p-turns", type=float, default=0.4)
    p.add_argument("--turn-gap-max-s", type=float, default=0.5)
    p.add_argument("--p-rir-turns", type=float, default=0.8)
    p.add_argument("--p-target-absent", type=float, default=0.20)
    p.add_argument("--formant-weight", type=float, default=2.5)
    p.add_argument("--lambda-leak", type=float, default=3.0)
    p.add_argument("--lambda-ovl", type=float, default=10.0)
    p.add_argument("--lsnr-factor", type=float, default=2e-3)
    p.add_argument("--lambda-turn", type=float, default=0.7)
    p.add_argument("--turn-hinge-db", type=float, default=-1.5)
    p.add_argument("--turn-boost", type=float, default=4.0)
    p.add_argument("--turn-decay-s", type=float, default=1.5)
    p.add_argument("--lambda-n0g", type=float, default=1.0)
    p.add_argument("--n0g-hinge-db", type=float, default=-1.0)
    p.add_argument("--lambda-tail", type=float, default=7.5)
    p.add_argument("--tail-s", type=float, default=1.5)
    p.add_argument("--lambda-leak-absent", type=float, default=3.0)
    p.add_argument("--nan-patience", type=int, default=20)
    p.add_argument("--stop-at", default=None, help="HH:MM local: salva e sai")
    p.add_argument("--ckpt-every-min", type=float, default=15.0)
    p.add_argument("--keep", type=int, default=3)
    p.add_argument("--eval-every-samples", type=int, default=1000 * REF_BATCH,
                   help="snapshot para o eval_quick a cada N amostras (padrão 1.000 passos × 12; 0 desliga)")
    p.add_argument("--runs-subdir", default="m4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-mix", action="store_true",
                   help="workers só sorteiam a receita; mistura, rótulos e features na GPU (cctrain.data.gpu, fora do hash)")
    return p.parse_args(argv)


# --- distribuído ----------------------------------------------------------------------------------------------------


def dist_init() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group("nccl")
        else:  # teste do caminho distribuído em CPU
            dist.init_process_group("gloo")
        return rank, local, world
    return 0, 0, 1


def allreduce_grads(params: list[torch.nn.Parameter], world: int) -> None:
    """Média dos gradientes dos treináveis entre ranks (um único buffer; grad ausente = 0)."""
    if world == 1:
        return
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    dist.all_reduce(flat)
    flat /= world
    o = 0
    for p in params:
        n = p.numel()
        g = flat[o:o + n].view_as(p)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        o += n


def broadcast_params(params: list[torch.nn.Parameter], world: int) -> None:
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)


# --- passo ----------------------------------------------------------------------------------------------------------


def step_v5(model, istft, df_params, hp: StageFHParams, b: dict, a, amp: bool) -> dict:
    spec, spec_clean, mask = b["spec"], b["spec_clean"], b["mask"]
    nb, T = spec.shape[0], spec.shape[2]
    profile = mask.reshape(nb) > 0
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    with ctx:
        spec_e, m, lsnr, df_coefs = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], mask)
    spec_e, m, lsnr, df_coefs = spec_e.float(), m.float(), lsnr.float(), df_coefs.float()
    spec_m = model.base.mask(spec.clone(), m)
    if amp:  # a parte espectral (máscara ERB + filtro DF sobre o espectro) refeita em fp32 a partir de m e dos coefs
        base = model.base
        spec_e = base.df_op(spec.clone(), df_coefs)
        spec_e = torch.cat([spec_e[..., : base.nb_df, :], spec_m[..., base.nb_df:, :]], dim=-2)
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

    shadow = target_shadow(model, spec_clean, m, df_coefs, zero)
    valid = prof & has_int & b["has_target"][:, None] & ta & ~ia & ~ds
    turn, tst = turn_loss(shadow, spec_clean, valid, ta, ia, hinge_db=a.turn_hinge_db, boost=a.turn_boost,
                          decay_s=a.turn_decay_s)
    v0 = prof & (b["n_interf"] == 0)[:, None] & b["has_target"][:, None] & ta & ~ds
    n0g, n0st = turn_loss(shadow, spec_clean, v0, ta, torch.zeros_like(ia), hinge_db=a.n0g_hinge_db, boost=1.0,
                          decay_s=1.0)
    iw = b["interf_active_wet45"][:, :T]
    since_i = since_interferer_s(ia)
    f_tail = (prof & ~ta & ~ia & iw & ~ds & (since_i <= a.tail_s)).float()
    tail = leak_loss(out, spec, f_tail)
    leak_abs = leak_loss(out, spec, f_leak * (~b["has_target"]).float()[:, None])
    if df_coefs.numel() > 1:
        l_df = df_coef_temporal_regularization_loss(df_coefs, time_dim=1) / (df_coefs.numel() / df_coefs.shape[1]) * 0.05
    else:
        l_df = est.sum() * 0.0

    parts = {"mr": mr, "os": os_, "leak": a.lambda_leak * leak, "ovl": a.lambda_ovl * ovl, "lsnr": ls,
             "turn": a.lambda_turn * turn, "n0g": a.lambda_n0g * n0g, "tail": a.lambda_tail * tail,
             "leak_abs": a.lambda_leak_absent * leak_abs, "df_reg": l_df}
    total = sum(parts.values())
    with torch.no_grad():
        oi = f_leak.bool()
        stats = {"raw_leak": leak.detach(), "raw_turn": turn.detach(), "raw_n0g": n0g.detach(),
                 "raw_tail": tail.detach(), "g_turn_db": tst["g_turn_db"].to(spec.device),
                 "g_turn_early_db": tst["g_turn_early_db"].to(spec.device), "g_n0_db": n0st["g_turn_db"].to(spec.device),
                 "zero_frac_only_interf": (zero & oi).sum() / oi.sum().clamp(min=1),
                 "zero_frac_target": (zero & f_keep).sum() / f_keep.sum().clamp(min=1)}
    return {"total": total, **parts, **stats}


# --- montagem -------------------------------------------------------------------------------------------------------


def resolve_v31(paths: Paths, p: str | None) -> Path:
    if p:
        return Path(p)
    for root in (paths.runs, paths.repo / "runs", Path.home() / "orca/projects/clearcore-train/runs"):
        c = root / "m4/pro-v4/checkpoints/ckpt-00079998.pt"
        if c.is_file():
            return c
    return paths.repo / V31_CKPT


def build_v5(a, device: str, hp: StageFHParams, paths: Paths):
    """Modelo + configuração dos adaptadores + lista de treináveis (mesma semente em todos os ranks)."""
    base, train_cond, gr, er = VARIANTS[a.variant]
    base = a.base or base
    gr = gr if a.gru_rank is None else a.gru_rank
    er = er if a.enc_rank is None else a.enc_rank
    train_cond = train_cond and not a.no_train_cond
    torch.manual_seed(a.seed)
    model = build_model(device, hp, presence=False)  # base = DFNet3 publicado; camadas do Pro com init padrão
    init_desc = "dfn3-upstream+fresh"
    if base == "v31":
        ck = resolve_v31(paths, a.v31_ckpt)
        model.load_state_dict(load_checkpoint(ck).model, strict=True)
        init_desc = str(ck)
    cfg = AdapterConfig(gru_rank=gr, enc_rank=er, alpha_over_rank=a.alpha_over_rank, gru_film=a.gru_film)
    sites = attach_adapters(model, cfg) if cfg.any else {}
    if not cfg.any:  # sem adaptadores: ainda marca o modelo para o eval reconhecer a config
        model.adapter_cfg = cfg
    model.to(device)
    return model, cfg, sites, base, train_cond, init_desc


def prepare(model, train_cond: bool) -> list:
    prepare_student(model)  # BN em eval/congelada, RNN em train (backward do cuDNN)
    return apply_trainable(model, train_cond=train_cond, train_adapters=True)


def count_params(model) -> dict:
    g = param_groups(model)
    named = dict(model.named_parameters())
    c = {k: sum(named[n].numel() for n in v) for k, v in g.items()}
    c["total"] = sum(c.values())
    return c


# --- main -----------------------------------------------------------------------------------------------------------


def main():
    a = parse_args()
    warnings.filterwarnings("ignore", message=".*RNN module weights are not part of single contiguous.*")
    rank, local, world = dist_init()
    is0 = rank == 0
    paths = Paths.from_env()
    run_dir = paths.runs / a.runs_subdir / a.run_id
    ckpt_dir, snap_dir = run_dir / "checkpoints", run_dir / "snapshots"
    if is0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        snap_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "TRAIN_DONE").unlink(missing_ok=True)
    if a.device == "cuda" and torch.cuda.is_available():
        device = f"cuda:{local}"
    else:
        device = "cpu"
    amp = a.amp == "bf16" and device.startswith("cuda")
    if a.global_batch % world:
        raise SystemExit(f"global-batch {a.global_batch} não divide por {world} ranks")
    bl = a.global_batch // world
    total_steps = a.total_steps or math.ceil(a.total_samples / a.global_batch)
    warmup = max(1, round(a.warmup_samples / a.global_batch))
    eval_every = max(1, round(a.eval_every_samples / a.global_batch)) if a.eval_every_samples > 0 else 0
    lr = a.lr * (math.sqrt(a.global_batch / REF_BATCH) if a.lr_scale == "sqrt" else 1.0)
    ncpu = len(os.sched_getaffinity(0))
    workers = a.workers if a.workers is not None else max(1, ncpu // world - 1)
    deadline = stop_deadline(a.stop_at)
    if is0:
        log(f"run {a.run_id} -> {run_dir} | {device} × {world} ranks | batch global {a.global_batch} ({bl}/rank) | "
            f"passos {total_steps} | warmup {warmup} | eval a cada {eval_every} | lr {lr:.2e} | amp {amp} | "
            f"workers/rank {workers} (cpus {ncpu}) | dados: {DATA_TAG}{' + ' + GPU_MIX_TAG if a.gpu_mix else ''} | "
            f"pid {os.getpid()}")
    stop = {"flag": False, "why": ""}

    def on_signal(signum, _frame):
        stop["flag"], stop["why"] = True, signal.Signals(signum).name
        log(f"[rank {rank}] sinal {stop['why']}: salvando e saindo")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    cfg_f = load_stage_f_config()
    hp = StageFHParams.from_config(cfg_f, run_id=a.run_id, num_workers=workers, batch_size=bl)
    clip = a.grad_clip if a.grad_clip is not None else hp.grad_clip
    model, acfg, sites, base_kind, train_cond, init_desc = build_v5(a, device, hp, paths)
    trainable = prepare(model, train_cond)
    broadcast_params(trainable, world)
    counts = count_params(model)
    n_tr = sum(p.numel() for p in trainable)
    if is0:
        log(f"base {base_kind} ({init_desc}) | variante {a.variant} | adaptadores {acfg.to_dict()} | sítios {sites} | "
            f"parâmetros {json.dumps(counts)} | treináveis {n_tr} ({100 * n_tr / counts['total']:.2f}%)")
    if a.compile:
        model.base.erb_dec = torch.compile(model.base.erb_dec, dynamic=False)
        model.base.df_dec = torch.compile(model.base.df_dec, dynamic=False)
    istft = Istft(960, HOP, torch.as_tensor(_fft_window())).to(device)
    params = ModelParams()

    vp = V4Params(seed=a.seed, segment_s=a.segment_s, batch_size=a.global_batch, total_steps=total_steps,
                  phase1_frac=0.0, n_probs=_np(a.n_probs, 5, N_PROBS), pattern_probs=_np(a.pattern_probs, 4, PATTERN_PROBS),
                  p_interf_first=a.p_interf_first, p_turns=a.p_turns, turn_gap_max_s=a.turn_gap_max_s,
                  p_rir_turns=a.p_rir_turns, p_target_absent=a.p_target_absent)
    hashed = {"v4": vp.hashable(), "lr": a.lr, "lr_scale": a.lr_scale, "wd": a.weight_decay, "warmup": warmup,
              "total_steps": total_steps, "gbatch": a.global_batch, "variant": a.variant, "base": base_kind,
              "adapter": acfg.to_dict(), "train_cond": train_cond,
              "lambdas": [a.lambda_leak, a.lambda_ovl, a.lambda_turn, a.lsnr_factor, a.lambda_n0g, a.n0g_hinge_db,
                          a.lambda_tail, a.tail_s, a.lambda_leak_absent],
              "turn": [a.turn_hinge_db, a.turn_boost, a.turn_decay_s], "fw": a.formant_weight, "v5": 1, "data": DATA_TAG}
    chash = config_hash(hashed)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warmup, total_steps))
    log_path = run_dir / "loss-log.jsonl"
    existing = load_latest(ckpt_dir, expected_cfg_hash=chash)
    step, spent_before = 0, 0.0
    if existing is not None:
        model.load_state_dict(existing.model, strict=True)
        opt.load_state_dict(existing.optimizer)
        sched.load_state_dict(existing.scheduler)
        restore_rng(existing)
        step = existing.step
        spent_before = float(existing.extra.get("elapsed_s", 0.0))
        trainable = prepare(model, train_cond)
        if is0:
            _truncate(log_path, step)
            log(f"RETOMADA de {existing.path.name} (passo {step}; ranks agora {world})")
    elif is0 and log_path.exists():
        log_path.unlink()
    commit = git_commit()

    ds, pl = make_dataset(paths, vp)
    if is0:
        log(f"pools: {json.dumps(pl.stats)} | v4 {json.dumps(vp.hashable())}")
    sampler = [s * a.global_batch + rank * bl + k for s in range(step, total_steps) for k in range(bl)]
    mixer = None
    if a.gpu_mix:
        ds = StageV4RecipeDataset(ds.pools, ds.bank, ds.data_root, vp)
        mixer = GpuMixer.from_dataset(ds, device)
        if is0:
            log(f"--gpu-mix: receitas nos workers, mistura/rótulos/features em {device}")
    dl = DataLoader(ds, batch_size=bl, sampler=sampler, num_workers=workers,
                    prefetch_factor=a.prefetch if workers else None, persistent_workers=workers > 0,
                    pin_memory=device.startswith("cuda"), worker_init_fn=init_worker,
                    collate_fn=collate_recipes if mixer else None)
    n_refill = 0
    data_tag = DATA_TAG + (f" + {GPU_MIX_TAG}" if mixer else "")

    t_start = time.monotonic()
    last_ckpt = t_start
    step_times: deque = deque(maxlen=200)
    waits: deque = deque(maxlen=200)
    rel_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def elapsed() -> float:
        return spent_before + (time.monotonic() - t_start)

    def save(reason: str, directory: Path = ckpt_dir, keep: int = a.keep) -> Path | None:
        if not is0:
            return None
        pth = save_checkpoint(directory, step=step, model=model.state_dict(), optimizer=opt, scheduler=sched,
                              keep=keep, sampler_state={"step": step}, cfg_hash=chash, commit=commit,
                              extra={"elapsed_s": elapsed(), "reason": reason, "args": vars(a), "v4": vp.hashable(),
                                     "presence_head": False, "adapter": acfg.to_dict(), "data": data_tag, "base": base_kind,
                                     "init": init_desc, "train_cond": train_cond, "world": world,
                                     "gating": {"mode": "lsnr-profile", "zero_below_db": -10.0},
                                     "wall": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"checkpoint {pth} ({reason})")
        return pth

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    end_step = min(total_steps, a.max_steps) if a.max_steps else total_steps
    stop_reason = "total_steps" if end_step == total_steps else "max_steps"
    bad_run = 0
    lf = open(log_path, "a") if is0 else None
    it = iter(dl)
    while step < end_step:
        tw = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            stop_reason = "data_end"
            break
        if mixer is not None:  # a mistura na GPU conta como espera por dados
            batch = mixer(batch)
            n_refill += mixer.fill_invalid(batch, ds)
            batch.pop("valid")
            batch.pop("index")
        t0 = time.perf_counter()
        waits.append(t0 - tw)
        b = _to_device_nb(batch, device)
        opt.zero_grad(set_to_none=True)
        parts = step_v5(model, istft, params, hp, b, a, amp)
        total = parts["total"]
        finite = torch.isfinite(total).float()
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() < 1:
            bad_run += 1
            if is0:
                log(f"perda não finita no passo {step} ({bad_run} seguidos)")
            sched.step()
            step += 1
            if bad_run >= a.nan_patience:
                stop_reason = "nan"
                break
            continue
        bad_run = 0
        total.backward()
        allreduce_grads(trainable, world)
        gn = torch.nn.utils.clip_grad_norm_(trainable, clip)
        opt.step()
        sched.step()
        step += 1
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)
        if step % 200 == 0 and not all(bool(torch.isfinite(p).all()) for p in trainable):
            stop_reason = "nan_params"
            log(f"parâmetro não finito no passo {step}")
            break
        if is0:
            row = {k: float(v.detach()) for k, v in parts.items()}
            mr = max(row["mr"], 1e-9)
            for k in TERMS[1:]:
                rel_hist[k].append(row[k] / mr)
            if step % 20 == 0 or step <= 3:
                rel = {f"rel_{k}": float(np.median(v)) for k, v in rel_hist.items()}
                lf.write(json.dumps({"step": step, "lr": float(sched.get_last_lr()[0]), **row, "grad_norm": float(gn),
                                     "dt": step_times[-1], "wait": waits[-1], **rel,
                                     "wall": time.strftime("%H:%M:%S")}) + "\n")
                lf.flush()
            if step % 25 == 0 or step <= 3:
                sps = statistics.median(step_times)
                wt = statistics.median(waits)
                mem = torch.cuda.max_memory_allocated(device) / GIB if device.startswith("cuda") else 0.0
                eta_h = (end_step - step) * (sps + wt) / 3600
                log(f"passo {step}/{end_step} total={row['total']:.3f} mr={row['mr']:.3f} s/passo={sps:.3f} "
                    f"espera={wt:.3f} amostras/s={a.global_batch / max(sps + wt, 1e-9):.1f} lr={sched.get_last_lr()[0]:.2e} "
                    f"gTurn={row['g_turn_db']:+.1f} gTurn0.5={row['g_turn_early_db']:+.1f} gN0={row['g_n0_db']:+.1f} "
                    f"zeroOI={row['zero_frac_only_interf']:.2f} zeroT={row['zero_frac_target']:.3f} gn={float(gn):.2f} "
                    f"vram={mem:.1f}GiB ETA={eta_h:.1f}h")
        flag = torch.tensor([1.0 if (stop["flag"] or (deadline and time.time() >= deadline)) else 0.0],
                            device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            stop_reason = f"signal:{stop['why']}" if stop["flag"] else "stop_at"
            break
        if eval_every and step % eval_every == 0:
            save("eval-snapshot", snap_dir, keep=10_000)
        if is0 and time.monotonic() - last_ckpt >= a.ckpt_every_min * 60.0:
            save("periodic")
            last_ckpt = time.monotonic()
    p = save(f"final:{stop_reason}")
    if eval_every and step % eval_every != 0:
        save("eval-snapshot-final", snap_dir, keep=10_000)
    if is0:
        summary = {"run_id": a.run_id, "step": step, "stop_reason": stop_reason, "elapsed_s": elapsed(),
                   "world": world, "global_batch": a.global_batch, "amp": amp, "data": data_tag, "gpu_mix_cpu_refills": n_refill if mixer else None,
                   "s_per_step_median": statistics.median(step_times) if step_times else None,
                   "loader_wait_median": statistics.median(waits) if waits else None,
                   "samples_per_s": (a.global_batch / (statistics.median(step_times) + statistics.median(waits)))
                   if step_times else None,
                   "vram_peak_gb": torch.cuda.max_memory_allocated(device) / GIB if device.startswith("cuda") else None,
                   "params": counts, "trainable": n_tr, "last_ckpt": str(p), "args": vars(a)}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        (run_dir / "TRAIN_DONE").write_text(f"{stop_reason} passo {step} {time.strftime('%F %T')}\n")
        log(f"fim: {stop_reason} no passo {step}; resumo em {run_dir / 'summary.json'}")
        lf.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _np(s: str, n: int, default):
    v = _floats(s, n, "probs")
    return v if v != default else default


def _fft_window():
    from df.enhance import init_df  # noqa: F401  (o df_state do upstream define a janela)
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3

    _, st = load_upstream_dfnet3(device="cpu")
    return st.fft_window().copy()


def _to_device_nb(batch: dict, device: str) -> dict:
    if not device.startswith("cuda"):
        return _to_device(batch, device)
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _truncate(path: Path, step: int) -> None:
    if not path.exists():
        return
    rows = [x for x in path.read_text().splitlines() if x.strip() and json.loads(x).get("step", 0) <= step]
    path.write_text("".join(r + "\n" for r in rows))


if __name__ == "__main__":
    main()
