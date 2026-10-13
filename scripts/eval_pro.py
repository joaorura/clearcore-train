#!/usr/bin/env python3
"""Script de avaliação e benchmarking do pDFNet3 Pro treinado contra o DFNet3 upstream.

Avalia:
- C0 (Fala geral sem perfil): ΔSI-SDR e ΔSTOI
- C3 (Casos críticos com falante concorrente / ruído severo): ΔSI-SDR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# imports limpos
from cctrain.m2.config import load_stage_f_config
from cctrain.models.cross_attention import SubbandCrossAttentionConfig
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmConfig, FilmGenerator
from cctrain.models.pdfnet3_pro import PDFNet3Pro
from cctrain.paths import Paths
from cctrain.train.checkpoint import load_checkpoint
from cctrain.train.stage_f import StageFHParams
from cctrain.dfvendor import ensure_df_importable
ensure_df_importable()
from df.loss import Istft


def evaluate(ckpt_path: Path, device: str = "cuda"):
    paths = Paths.from_env()
    print(f"Carregando modelo base e inicializando PDFNet3Pro...")
    base, df_state = load_upstream_dfnet3(device=device)
    cfg = load_stage_f_config()
    hp = StageFHParams.from_config(cfg)
    gen = FilmGenerator(FilmConfig(log_gamma_bound=hp.log_gamma_bound, beta_bound=hp.beta_bound)).to(device)
    attn_cfg = SubbandCrossAttentionConfig(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4, num_tokens=16, token_dim=128)
    model = PDFNet3Pro(base, gen, cross_attention_config=attn_cfg, enable_cross_attention=True).to(device)

    print(f"Carregando checkpoint Pro: {ckpt_path}")
    ck = load_checkpoint(ckpt_path)
    model.load_state_dict(ck.model, strict=False)
    model.eval()

    print("Carregando dev set C0 e C3 para avaliação...")
    from cctrain.m2.devsets import load_select_items
    from cctrain.m2.select import make_evaluator

    teacher, _ = load_upstream_dfnet3(device="cpu")
    evaluator = make_evaluator(load_select_items(), teacher)
    res = evaluator(model, epoch=10)
    print("\n=======================================================")
    print("      RESULTADO DA AVALIAÇÃO DO pDFNet3 PRO")
    print("=======================================================")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/home/joaorura/orca/projects/clearcore-train/runs/m2/t3-pro-f1-f2/checkpoints/ckpt-00075000.pt")
    args = p.parse_args()
    evaluate(Path(args.ckpt))
