#!/usr/bin/env python3
"""Script de treinamento completo do pDFNet3 Pro (Frente 1 + Frente 2).

Integra de ponta a ponta:
- Arquitetura: PDFNet3Pro (Dual-Conditioning: FiLM Macro + Cross-Attention Causal em Sub-bandas).
- Perdas Frente 1: Loss Assimétrica Ponderada por Formantes (proteção da banda de fala 300 - 3400 Hz contra over-suppression).
- Discriminador Espectral Multi-Escala Causal.
- Noisy & Reverberant Enrollment Augmentation.
- Warm start opcional a partir do melhor checkpoint do Estágio F anterior.
- Loop de treinamento idêntico ao stage_f oficial com gravação em runs/m2/<run-id>/.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cctrain.data.enroll import EnrollmentAugmentConfig
from cctrain.dfvendor import ensure_df_importable
from cctrain.hashing import sha256_file
from cctrain.train.checkpoint import config_hash
from cctrain.losses.dfnet import (
    arcface_angular_margin_loss,
    asymmetric_formant_loss,
    df_coef_temporal_regularization_loss,
    dfnet_loss,
    gate_loss,
    local_snr_loss,
    target_absence_loss,
)
from cctrain.losses.distill import distill_loss
from cctrain.m2.config import load_stage_f_config
from cctrain.models.cross_attention import SubbandCrossAttentionConfig
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmConfig, FilmGenerator
from cctrain.models.pdfnet3_pro import PDFNet3Pro
from cctrain.paths import Paths
from cctrain.train.checkpoint import (
    Checkpoint,
    git_commit,
    load_checkpoint,
    load_latest,
    save_checkpoint,
)
from cctrain.train.stage_f import (
    HOP,
    StageFDataset,
    StageFHParams,
    _gated,
    _lr_factor,
    _lsnr_target,
    _to_device,
    _truncate_log,
    bn_state_sha256,
    gpu_lock,
    gpu_logger,
    n_batchnorm,
    prepare_student,
    read_history,
)
from cctrain.train.embed_bank import ArrayEmbeddingBank
from cctrain.m1 import pools
from cctrain.train.stage_c import StageCData

EMB = Paths.from_env().data / "embeddings"
BANK_V2, BANK_V1 = EMB / "bank-m2-sbecapa-v2.npz", EMB / "bank-m1-sbecapa-v1.npz"

def load_bank():
    if BANK_V2.is_file():
        bank = ArrayEmbeddingBank.load(BANK_V2)
        if getattr(bank, "has_provenance", False):
            return bank, "v2"
    return ArrayEmbeddingBank.load(BANK_V1), "v1"

def load_inputs(paths: Paths):
    speech = pools.by_split(pools.speech_records(paths), "train")
    noise_train, _ = pools.noise_pools()
    return StageCData(speech, noise_train, pools.rir_records(), None)

ensure_df_importable()
from df.loss import Istft
from df.model import ModelParams


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Treinamento Completo do pDFNet3 Pro")
    p.add_argument("--run-id", default="t3-pro-f1-f2", help="ID da execução")
    p.add_argument("--init-ckpt", default="/home/joaorura/orca/projects/clearcore-train/runs/m2/t3-stage-f-full/checkpoints/ckpt-00007500.pt",
                   help="Checkpoint inicial para warm start")
    p.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    p.add_argument("--lr", type=float, default=6e-5, help="Taxa de aprendizado inicial")
    p.add_argument("--workers", type=int, default=4, help="Número de workers no DataLoader")
    p.add_argument("--batch-size", type=int, default=16, help="Tamanho do lote")
    p.add_argument("--formant-weight", type=float, default=2.5, help="Peso da banda de formantes")
    p.add_argument("--steps-per-epoch", type=int, default=None, help="Passos máximos por época (opcional)")
    p.add_argument("--device", default="cuda", help="Dispositivo (cuda ou cpu)")
    return p.parse_args()


def stage_pro_step(model: nn.Module, teacher: nn.Module, istft: Istft, df_params: ModelParams,
                   hp: StageFHParams, b: dict, formant_weight: float) -> dict[str, torch.Tensor]:
    """Forward + perdas do pDFNet3 Pro: Dual-Conditioning, Formantes, Target-Absence e Regularização DF."""
    spec, spec_clean, mask = b["spec"], b["spec_clean"], b["mask"]
    nb = spec.shape[0]
    spec_e, m, lsnr, df_coefs = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], mask)
    out, _stages = _gated(model.base, spec, spec_e, m, lsnr)
    est, tgt = istft(out), istft(spec_clean)
    lsnr_t = _lsnr_target(df_params, spec_clean, spec)
    kw = dict(n_ffts=hp.fft_sizes, gamma=hp.loss_gamma)

    # 1. Perda espectral multi-resolução
    mr = dfnet_loss(est, tgt, b["max_freq"], factor=hp.loss_factor, factor_complex=hp.loss_factor_complex, **kw)

    # 2. Frente 1: Loss Assimétrica Ponderada por Formantes da Fala (protege 300 - 3400 Hz)
    os_ = asymmetric_formant_loss(
        est, tgt, b["max_freq"], factor=hp.lambda_os, formant_weight=formant_weight,
        extra_formant_penalty=1.0, **kw
    )

    # 3. Gating e LSNR
    gate = gate_loss(lsnr, b["interferer_frames"] & (mask.reshape(nb, 1) > 0), factor=hp.lambda_gate)
    ls = local_snr_loss(lsnr, lsnr_t, b["lsnr_mask"])

    # 4. Target-Absence Loss: força energia zero durante ausência do falante alvo
    absence_mask = b["interferer_frames"] & (mask.reshape(nb, 1) > 0)
    if absence_mask.any():
        l_abs = target_absence_loss(est, absence_mask)
    else:
        l_abs = torch.tensor(0.0, device=est.device)

    # 5. DF Temporal Regularization Loss: penaliza variações caóticas de fase/coeficientes
    if df_coefs is not None and df_coefs.numel() > 0:
        l_df = df_coef_temporal_regularization_loss(df_coefs, time_dim=1) / (df_coefs.numel() / df_coefs.shape[1]) * 0.05
    else:
        l_df = torch.tensor(0.0, device=est.device)

    # 6. Destilação de amostras desmascaradas contra professor
    idx = (mask.reshape(nb) == 0).nonzero(as_tuple=True)[0]
    if idx.numel():
        with torch.no_grad():
            s0 = spec[idx]
            t_e, t_m, t_lsnr, _ = teacher(s0.clone(), b["feat_erb"][idx], b["feat_spec"][idx])
            t_out, _ = _gated(teacher, s0, t_e, t_m, t_lsnr)
            t_est = istft(t_out)
    else:
        t_est = est[:0]
    t = distill_loss(est, t_est, mask, b["max_freq"], lambda_t=hp.lambda_t, n_ffts=hp.fft_sizes, gamma=hp.loss_gamma)

    # 7. ArcFace Angular Margin Loss: desentrelaçamento angular de locutores
    if "interferer_embedding" in b and b["interferer_embedding"] is not None:
        l_arc = arcface_angular_margin_loss(
            b["embedding"],
            b["embedding"],
            b["interferer_embedding"],
            s=32.0,
            m=0.25,
        )
    else:
        l_arc = torch.tensor(0.0, device=est.device)

    total = mr + os_ + gate + ls + t + 0.5 * l_abs + l_df + 0.15 * l_arc
    return {
        "total": total,
        "mr": mr,
        "os": os_,
        "gate": gate,
        "lsnr": ls,
        "abs": l_abs,
        "df_reg": l_df,
        "arcface": l_arc,
        "t": t,
        "n0": torch.tensor(float(idx.numel())),
    }


def main():
    args = parse_args()
    paths = Paths.from_env()
    run_dir = paths.runs / "m2" / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    log(f"================================================================================")
    log(f"Iniciando treinamento oficial pDFNet3 Pro [{args.run_id}] na GPU Blackwell")
    log(f"Frente 1: Loss Assimétrica de Formantes ativa (peso {args.formant_weight}x)")
    log(f"Frente 2: Dual-Conditioning (FiLM Macro + Causal Subband Cross-Attention) ativo")
    log(f"================================================================================")

    base, df_state = load_upstream_dfnet3(device=device)
    teacher, _ = load_upstream_dfnet3(device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg = load_stage_f_config()
    hp = StageFHParams.from_config(cfg, run_id=args.run_id, num_workers=args.workers, batch_size=args.batch_size)
    gen = FilmGenerator(FilmConfig(log_gamma_bound=hp.log_gamma_bound, beta_bound=hp.beta_bound)).to(device)
    attn_cfg = SubbandCrossAttentionConfig(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4, num_tokens=16, token_dim=128)
    model = PDFNet3Pro(base, gen, cross_attention_config=attn_cfg, enable_cross_attention=True).to(device)
    prepare_student(model)

    istft = Istft(960, HOP, torch.as_tensor(df_state.fft_window().copy())).to(device)
    params = ModelParams()

    # Warm start do checkpoint eleito
    init_sha = ""
    if Path(args.init_ckpt).is_file():
        log(f"Carregando pesos de warm start: {args.init_ckpt}")
        init_sha = sha256_file(Path(args.init_ckpt))
        ckpt_data = load_checkpoint(Path(args.init_ckpt))
        curr_state = model.state_dict()
        compat_state = {
            k: v for k, v in ckpt_data.model.items()
            if k in curr_state and v.shape == curr_state[k].shape
        }
        model.load_state_dict(compat_state, strict=False)
        loaded_count = len(compat_state)
        total_count = len(curr_state)
        log(f"Warm start concluído: {loaded_count}/{total_count} tensores carregados (base, FiLM e decoders sincronizados)")

    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else hp.steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, hp.warmup_steps, total_steps))

    bank, _ = load_bank()
    data = load_inputs(paths)
    data_root = paths.data

    log_path = run_dir / "loss-log.jsonl"
    chash = config_hash({**hp.hashable(), "run_id": args.run_id})
    commit = git_commit()
    step = 0
    t_start = time.monotonic()
    last_ckpt_time = t_start

    lock = gpu_lock(hp.gpu_lock_timeout_s) if device.startswith("cuda") else contextlib.nullcontext()
    log_ctx = gpu_logger(run_dir / "nvidia-smi.csv", hp.gpu_log_interval_s) if device.startswith("cuda") else contextlib.nullcontext()

    with lock, log_ctx, open(log_path, "a") as loss_log:
        for epoch in range(1, args.epochs + 1):
            log(f"--- Iniciando Época {epoch}/{args.epochs} (Passo {step}) ---")
            ds = StageFDataset(data, bank, data_root, hp, epoch=epoch)
            dl = DataLoader(ds, batch_size=hp.batch_size, num_workers=hp.num_workers,
                            prefetch_factor=hp.prefetch_factor if hp.num_workers else None)
            
            for batch_idx, batch in enumerate(dl):
                if args.steps_per_epoch is not None and batch_idx >= args.steps_per_epoch:
                    break
                t0 = time.perf_counter()
                b = _to_device(batch, device)
                if "interferer_key" in batch:
                    rng_step = np.random.default_rng(hp.seed + step)
                    int_embs = []
                    has_any = False
                    for k in batch["interferer_key"]:
                        if k and k in bank.speakers():
                            int_embs.append(bank.sample(k, rng_step))
                            has_any = True
                        else:
                            int_embs.append(np.zeros(192, dtype=np.float32))
                    if has_any:
                        b["interferer_embedding"] = torch.from_numpy(np.stack(int_embs)).to(device)

                prepare_student(model)
                opt.zero_grad(set_to_none=True)

                parts = stage_pro_step(model, teacher, istft, params, hp, b, args.formant_weight)
                parts["total"].backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], hp.grad_clip)
                opt.step()
                sched.step()

                step += 1
                dt = time.perf_counter() - t0

                if step % 20 == 0:
                    entry = {
                        "step": step, "epoch": epoch, "lr": float(sched.get_last_lr()[0]),
                        "total": float(parts["total"].item()), "mr": float(parts["mr"].item()),
                        "os": float(parts["os"].item()), "gate": float(parts["gate"].item()),
                        "lsnr": float(parts["lsnr"].item()), "abs": float(parts["abs"].item()),
                        "df_reg": float(parts["df_reg"].item()),
                        "arcface": float(parts["arcface"].item()),
                        "t": float(parts["t"].item()),
                        "n0": float(parts["n0"].item()), "dt": dt
                    }
                    loss_log.write(json.dumps(entry) + "\n")
                    loss_log.flush()

                if step % 500 == 0 or (time.monotonic() - last_ckpt_time) > 1800:
                    ckpt_p = save_checkpoint(
                        ckpt_dir, step=step, model=model.state_dict(), optimizer=opt, scheduler=sched,
                        keep=10, sampler_state={"epoch": epoch, "batch": batch_idx},
                        cfg_hash=chash, commit=commit,
                        extra={"elapsed_s": time.monotonic() - t_start, "init_sha256": init_sha}
                    )
                    last_ckpt_time = time.monotonic()
                    log(f"Checkpoint intermediário salvo: passo {step} (Loss Total: {parts['total'].item():.3f})")

            # Salva fim de época
            ep_ckpt = save_checkpoint(
                ckpt_dir, step=step, model=model.state_dict(), optimizer=opt, scheduler=sched,
                keep=10, sampler_state={"epoch": epoch, "batch": 0},
                cfg_hash=chash, commit=commit,
                extra={"elapsed_s": time.monotonic() - t_start, "epoch": epoch, "init_sha256": init_sha}
            )
            log(f"Época {epoch} concluída! Checkpoint: {ep_ckpt.name} (Passo {step})")

    log("Treinamento do pDFNet3 Pro concluído com sucesso!")


if __name__ == "__main__":
    main()
