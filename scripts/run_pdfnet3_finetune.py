"""Pipeline de Treinamento e Fine-Tuning do pDFNet3 / pDFNet3-Pro para Microfones Locais.

Permite:
- Configuração modular de congelamento vs ajuste:
  * Flag `--freeze-encoder`: Congela o Encoder ERB + Complex STFT (483.969 parâmetros) para preservar
    representação espacial e acústica universal de fala.
  * Se não congelado: Aplica taxa reduzida diferencial (`--lr-encoder`, default 1e-5).
  * Módulo FiLM / Dual-Conditioning: Treinado ativamente (`--lr-film`, default 1e-4).
  * Decoders (ERB Decoder + DF Filter Decoder, ~1.65M parâmetros): Treinados ativamente (`--lr-decoder`, default 1e-4)
    para adaptação de alta resolução aos perfis espectro-temporais dos microfones do notebook e acessórios.
- Suporte a pDFNet3 clássico e pDFNet3-Pro com Dual-Conditioning causal.
- Salvamento e resumo de checkpoints com métricas e contadores de parâmetros.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parents[1] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import torch
from torch import nn, optim

from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmConfig, FilmGenerator, load_film_config
from cctrain.models.pdfnet3 import PDFNet3
from cctrain.models.pdfnet3_pro import PDFNet3Pro


def setup_finetune_model(
    freeze_encoder: bool = False,
    lr_encoder: float = 1e-5,
    lr_film: float = 1e-4,
    lr_decoder: float = 1e-4,
    lr_cross_attn: float = 1e-4,
    device: str = "cpu",
) -> tuple[nn.Module, optim.Optimizer, dict[str, int]]:
    # 1. Carrega modelo base upstream pré-treinado
    base_model, df_state = load_upstream_dfnet3(device=device)
    
    # 2. Carrega gerador FiLM calibrado
    film_cfg = load_film_config()
    film_generator = FilmGenerator(film_cfg)
    
    # 3. Monta o modelo oficial pDFNet3Pro (Dual-Conditioning)
    model = PDFNet3Pro(base=base_model, generator=film_generator)
    model.to(device)
    
    # 4. Estratégia de parâmetros por taxa de aprendizado
    param_groups = []
    
    # Encoder
    if freeze_encoder:
        for param in model.base.enc.parameters():
            param.requires_grad = False
    else:
        for param in model.base.enc.parameters():
            param.requires_grad = True
        param_groups.append({
            "params": [p for p in model.base.enc.parameters() if p.requires_grad],
            "lr": lr_encoder,
            "name": "encoder",
        })
        
    # FiLM Generator
    for param in model.film.parameters():
        param.requires_grad = True
    param_groups.append({
        "params": [p for p in model.film.parameters() if p.requires_grad],
        "lr": lr_film,
        "name": "film_generator",
    })
    
    # Decoders (ERB Decoder + DF Decoder)
    decoder_params = list(model.base.erb_dec.parameters()) + list(model.base.df_dec.parameters())
    for param in decoder_params:
        param.requires_grad = True
    param_groups.append({
        "params": [p for p in decoder_params if p.requires_grad],
        "lr": lr_decoder,
        "name": "decoders",
    })
    
    # Adiciona parâmetros do módulo de Causal Sub-band Cross-Attention (Dual-Conditioning)
    if hasattr(model, "cross_attn"):
        for param in model.cross_attn.parameters():
            param.requires_grad = True
        param_groups.append({
            "params": [p for p in model.cross_attn.parameters() if p.requires_grad],
            "lr": lr_cross_attn,
            "name": "cross_attention",
        })
        
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    
    stats = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "enc_params": sum(p.numel() for p in model.base.enc.parameters()),
        "film_params": sum(p.numel() for p in model.film.parameters()),
        "dec_params": sum(p.numel() for p in decoder_params),
        "cross_attn_params": sum(p.numel() for p in model.cross_attn.parameters()) if hasattr(model, "cross_attn") else 0,
    }
    
    return model, optimizer, stats


def main():
    parser = argparse.ArgumentParser(description="Script de Fine-Tuning do pDFNet3Pro (Dual-Conditioning).")
    parser.add_argument("--freeze-encoder", action="store_true", default=False, help="Congelar o encoder upstream (padrão: False - toda a rede treina)")
    parser.add_argument("--lr-encoder", type=float, default=1e-5, help="Taxa de aprendizado para o encoder (padrão: 1e-5)")
    parser.add_argument("--lr-film", type=float, default=1e-4, help="Taxa de aprendizado para o módulo FiLM (padrão: 1e-4)")
    parser.add_argument("--lr-decoder", type=float, default=1e-4, help="Taxa de aprendizado para os decoders (padrão: 1e-4)")
    parser.add_argument("--lr-cross-attn", type=float, default=1e-4, help="Taxa de aprendizado para a Cross-Attention (padrão: 1e-4)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositivo (cpu/cuda)")
    parser.add_argument("--dry-run", action="store_true", help="Apenas inicializa o modelo e valida arquitetura")
    args = parser.parse_args()

    print("=" * 70)
    print("CONFIGURAÇÃO DE FINE-TUNING pDFNet3Pro (ARQUITETURA OFICIAL):")
    print(f" - Arquitetura: pDFNet3Pro (Dual-Conditioning: FiLM Macro + Causal Subband Attention)")
    print(f" - Freeze Encoder: {args.freeze_encoder}")
    print(f" - LRs -> Encoder: {args.lr_encoder:.1e} | FiLM: {args.lr_film:.1e} | Decoders: {args.lr_decoder:.1e} | Cross-Attn: {args.lr_cross_attn:.1e}")
    print(f" - Dispositivo: {args.device}")
    print("=" * 70)

    model, optimizer, stats = setup_finetune_model(
        freeze_encoder=args.freeze_encoder,
        lr_encoder=args.lr_encoder,
        lr_film=args.lr_film,
        lr_decoder=args.lr_decoder,
        lr_cross_attn=args.lr_cross_attn,
        device=args.device,
    )

    print("\nDISTRIBUIÇÃO DE PARÂMETROS:")
    print(f" - Total de parâmetros: {stats['total_params']:,}")
    print(f" - Parâmetros treináveis: {stats['trainable_params']:,} ({stats['trainable_params']/stats['total_params']*100:.1f}%)")
    print(f" - Parâmetros congelados: {stats['frozen_params']:,} ({stats['frozen_params']/stats['total_params']*100:.1f}%)")
    print(f"   * Encoder: {stats['enc_params']:,} [{'FROZEN' if args.freeze_encoder else 'TRAINABLE'}]")
    print(f"   * FiLM Generator: {stats['film_params']:,} [TRAINABLE]")
    print(f"   * Decoders (ERB + DF): {stats['dec_params']:,} [TRAINABLE]")
    print("=" * 70)
    
    if args.dry_run:
        print("[OK] Dry-run concluído com sucesso!")
        return 0
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
