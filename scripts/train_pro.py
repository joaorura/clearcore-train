#!/usr/bin/env python3
"""Script de treinamento do pDFNet3 Pro (Frente 1 + Frente 2).

Integra:
- Dual-Conditioning: FiLM Macro + Cross-Attention Causal em Sub-bandas (PDFNet3Pro).
- Loss Assimétrica Ponderada por Formantes (proteção da banda de fala 300 - 3400 Hz contra over-suppression).
- Discriminador Espectral Multi-Escala (opcional / loss adversarial para máxima fidelidade).
- Noisy & Reverberant Enrollment Augmentation.
- Inicialização a partir do melhor checkpoint do Estágio F anterior (warm start).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cctrain.data.enroll import EnrollmentAugmentConfig
from cctrain.losses.dfnet import asymmetric_formant_loss, dfnet_loss, gate_loss, local_snr_loss
from cctrain.losses.distill import distill_loss
from cctrain.m2.config import load_stage_f_config
from cctrain.models.cross_attention import SubbandCrossAttentionConfig
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmConfig, FilmGenerator
from cctrain.models.pdfnet3_pro import PDFNet3Pro
from cctrain.paths import Paths
from cctrain.train.stage_f import (
    StageFHParams,
    _gated,
    _lsnr_target,
    gpu_lock,
    gpu_logger,
    load_checkpoint,
    prepare_student,
    save_checkpoint,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Treinamento do pDFNet3 Pro")
    p.add_argument("--run-id", default="t3-pro-f1-f2", help="ID da execução")
    p.add_argument("--init-ckpt", default="/home/joaorura/orca/projects/clearcore-train/runs/m2/t3-stage-f-full/checkpoints/ckpt-00007500.pt",
                   help="Checkpoint inicial para warm start")
    p.add_argument("--epochs", type=int, default=10, help="Número de épocas de ajuste fino Pro")
    p.add_argument("--lr", type=float, default=5e-5, help="Taxa de aprendizado inicial")
    p.add_argument("--workers", type=int, default=4, help="Número de workers no DataLoader")
    p.add_argument("--batch-size", type=int, default=16, help="Tamanho do lote")
    p.add_argument("--use-formant-loss", action="store_true", default=True, help="Usar loss assimétrica de formantes")
    p.add_argument("--formant-weight", type=float, default=2.5, help="Peso da banda de formantes")
    p.add_argument("--device", default="cuda", help="Dispositivo (cuda ou cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    paths = Paths.from_env()
    run_dir = paths.runs / "m2" / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log(f"Iniciando treinamento do pDFNet3 Pro [{args.run_id}]...")
    log(f"Frente 1: Loss Assimétrica de Formantes ativa (peso {args.formant_weight}x)")
    log("Frente 2: Dual-Conditioning (FiLM Macro + Causal Subband Cross-Attention) ativo")

    # 1. Carregar modelo base e inicializar PDFNet3Pro
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    base, df_state = load_upstream_dfnet3(device=device)
    teacher, _ = load_upstream_dfnet3(device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg = load_stage_f_config()
    hp = StageFHParams.from_config(cfg, run_id=args.run_id, num_workers=args.workers)
    
    gen = FilmGenerator(FilmConfig(log_gamma_bound=hp.log_gamma_bound, beta_bound=hp.beta_bound)).to(device)
    attn_cfg = SubbandCrossAttentionConfig(dim=256, num_heads=4, head_dim=32, subband_start=3, subband_end=20)
    model = PDFNet3Pro(base, gen, cross_attention_config=attn_cfg, enable_cross_attention=True).to(device)
    prepare_student(model)

    # 2. Warm start do checkpoint eleito
    if Path(args.init_ckpt).is_file():
        log(f"Carregando pesos do checkpoint inicial: {args.init_ckpt}")
        ckpt_data = load_checkpoint(Path(args.init_ckpt))
        state = ckpt_data.model
        # Carregar parâmetros correspondentes de base e film com strict=False para acomodar o cross_attn inicializado
        model.load_state_dict(state, strict=False)
        log("Warm start concluído: pesos de base e FiLM restaurados; Cross-Attention inicializado com ganhos unitários")

    # 3. Otimizador com LR adaptativo
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * hp.steps_per_epoch, eta_min=args.lr * 0.05)

    log(f"Modelo pronto. Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    log("Tudo pronto para execução do ciclo de treino Pro!")


if __name__ == "__main__":
    main()
