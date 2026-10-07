#!/usr/bin/env python3
"""Mini Fine-Tuning de Adaptação Rápida para Ruídos Reais e Microfones Locais.

Este script executa um fine-tuning rápido (poucas iterações) adaptando o modelo pDFNet3
para os ruídos acústicos de notebook e vazamentos de voz reais gravados localmente.

Estratégia de Congelamento:
- Encoder (`enc`) congelado: preserva as representações universais de fala intactas e
  evita esquecimento catastrófico.
- BatchNorm congelada em `.eval()`: estatísticas e pesos fixados.
- Decoders (`erb_dec` e `df_dec`) treináveis: adaptam a máscara de ganho ERB e os
  coeficientes de Deep Filtering para anular ressonâncias acústicas do notebook e
  vazamentos de voz dos microfones locais.

Uso:
    python scripts/train_local_adapt.py --steps 1000 --batch-size 8 --device cuda
    python scripts/train_local_adapt.py --steps 50 --device cpu  # smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

# Adiciona o diretório src do repositório ao sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cctrain.data.audio import resample
from cctrain.data.mix import crop_or_pad, loop_to_length, power, scale_for_ratio
from cctrain.dfvendor import ensure_df_importable
from cctrain.eval.functional import si_sdr
from cctrain.eval.runtime_path import HOP, emulate_gating_offline, runtime_forward
from cctrain.losses.dfnet import dfnet_loss, over_suppression_loss
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmGenerator, load_film_config
from cctrain.models.pdfnet3 import PDFNet3
from cctrain.paths import Paths, repo_root
from cctrain.train.checkpoint import load_checkpoint, save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_local_adapt")

SR = 48000
PEAK_TARGET = 0.95


def parse_args() -> argparse.Namespace:
    paths = Paths.from_env()
    default_noise = paths.data / "raw" / "noise-local-mics"
    default_ckpt = paths.runs / "m2" / "t3" / "checkpoints" / "ckpt-00022500.pt"
    default_out = paths.runs / "local_adapt"

    parser = argparse.ArgumentParser(
        description="Mini Fine-Tuning do pDFNet3 para Ruídos Locais"
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        nargs="+",
        default=[default_noise],
        help="Diretório(s) com ruídos reais fatiados ou combinados",
    )
    parser.add_argument(
        "--speech-path",
        type=Path,
        nargs="+",
        default=None,
        help="Caminho(s) do áudio limpo (default: auto-descoberta)",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=default_ckpt,
        help="Checkpoint base de partida (default: runs/m2/t3/checkpoints/ckpt-00022500.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Diretório de saída para checkpoints e logs (default: runs/local_adapt)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Número total de passos de otimização (default: 3000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamanho do mini-lote (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Taxa de aprendizado inicial para os decoders (default: 1e-4)",
    )
    parser.add_argument(
        "--segment-s",
        type=float,
        default=3.0,
        help="Duração de cada segmento de treino em segundos (default: 3.0)",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=200,
        help="Intervalo de passos para avaliação e checkpoint (default: 200)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Paciência para Early Stopping: interrompe se a perda de validação não melhorar por N avaliações consecutivas (default: 5)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.005,
        help="Melhoria mínima na perda de validação para resetar a paciência (default: 0.005)",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=2,
        help="Número máximo de checkpoints intermediários recentes mantidos em disco (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo de execução ('cuda' ou 'cpu')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente pseudo-aleatória (default: 42)",
    )
    return parser.parse_args()


def load_training_data(
    speech_paths: list[Path] | Path | None, noise_dirs: list[Path] | Path, segment_s: float, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Carrega dados e divide em conjuntos de treino e validação (holding-out)."""
    from poc_mix_and_eval import discover_clean_speech, load_clean_speech_segments, load_noise_slices

    if speech_paths is None:
        speech_paths = [discover_clean_speech(None)]
    elif isinstance(speech_paths, Path):
        speech_paths = [speech_paths]

    train_speech = []
    test_speech = []
    for sp in speech_paths:
        if not sp.exists():
            continue
        tr_s, te_s = load_clean_speech_segments(
            sp, segment_s, holding_out_fraction=0.25, seed=seed
        )
        train_speech.extend(tr_s)
        test_speech.extend(te_s)

    if isinstance(noise_dirs, Path):
        noise_dirs = [noise_dirs]

    train_noises = []
    test_noises = []
    for ndir in noise_dirs:
        if not ndir.exists():
            continue
        splits = load_noise_slices(ndir, holding_out_fraction=0.25, seed=seed)
        for cat_name, split in splits.items():
            train_noises.extend(split["train"])
            test_noises.extend(split["test"])

    logger.info(
        f"Dados carregados: {len(train_speech)} trechos de fala treino, {len(test_speech)} teste | "
        f"{len(train_noises)} fatias de ruído treino, {len(test_noises)} teste"
    )
    return train_speech, test_speech, train_noises, test_noises


def prepare_frozen_student(model: PDFNet3) -> list[nn.Parameter]:
    """Congela o encoder e BatchNorm; mantém apenas decoders e gerador treináveis."""
    model.eval()

    # Congela completamente o encoder upstream
    for p in model.base.enc.parameters():
        p.requires_grad_(False)

    # Congela todas as camadas de BatchNorm (estatísticas e pesos)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        elif isinstance(m, nn.RNNBase):
            m.train()

    # Habilita gradientes estritamente nos decoders
    trainable_params = []
    for name, p in model.base.erb_dec.named_parameters():
        p.requires_grad_(True)
        trainable_params.append(p)

    for name, p in model.base.df_dec.named_parameters():
        p.requires_grad_(True)
        trainable_params.append(p)

    logger.info(
        f"Estratégia de congelamento concluída: {len(trainable_params)} tensores de pesos treináveis "
        f"({sum(p.numel() for p in trainable_params):,} parâmetros nos decoders ERB e DF)."
    )
    return trainable_params


def generate_batch(
    speech_pool: list[np.ndarray],
    noise_pool: list[np.ndarray],
    batch_size: int,
    segment_s: float,
    df_state,
    device: str,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """Gera um mini-lote com fala, ruído em SNRs U[-5, 15] dB e features do libdf."""
    from df.enhance import df_features

    n_samples = int(segment_s * SR)
    snr_grid = [-5.0, 0.0, 5.0, 10.0, 15.0]

    noisy_batch = []
    clean_batch = []

    for _ in range(batch_size):
        clean = speech_pool[int(rng.integers(len(speech_pool)))]
        noise = noise_pool[int(rng.integers(len(noise_pool)))]

        clean_seg = crop_or_pad(clean, n_samples, rng)
        noise_seg = loop_to_length(noise, n_samples, rng)

        # 15% de chance de ruído puro (interferente isolado)
        is_noise_only = rng.random() < 0.15
        if is_noise_only:
            target_clean = np.zeros(n_samples, dtype=np.float32)
            noisy_raw = noise_seg
        else:
            snr = float(rng.choice(snr_grid))
            scaled_n = scale_for_ratio(clean_seg, noise_seg, snr)
            noisy_raw = clean_seg + scaled_n
            target_clean = clean_seg

        peak = max(np.max(np.abs(noisy_raw)), np.max(np.abs(target_clean)), 1e-9)
        gain = PEAK_TARGET / peak if peak > PEAK_TARGET else 1.0

        noisy_batch.append((noisy_raw * gain).astype(np.float32))
        clean_batch.append((target_clean * gain).astype(np.float32))

    # Extração de features usando libdf
    audios_noisy_t = [torch.from_numpy(a)[None] for a in noisy_batch]
    audios_clean_t = [torch.from_numpy(a)[None] for a in clean_batch]

    feats = [df_features(a, df_state, 96, device="cpu") for a in audios_noisy_t]
    spec_noisy = torch.cat([f[0] for f in feats], dim=0).to(device)
    feat_erb = torch.cat([f[1] for f in feats], dim=0).to(device)
    feat_spec = torch.cat([f[2] for f in feats], dim=0).to(device)

    # Análise de espectro do sinal limpo alvo
    feats_clean = [df_features(a, df_state, 96, device="cpu") for a in audios_clean_t]
    spec_clean = torch.cat([f[0] for f in feats_clean], dim=0).to(device)

    return {
        "spec": spec_noisy,
        "feat_erb": feat_erb,
        "feat_spec": feat_spec,
        "spec_clean": spec_clean,
        "audio_clean": torch.from_numpy(np.stack(clean_batch)).to(device),
    }


def train_adapt(args: argparse.Namespace) -> None:
    ensure_df_importable()
    from df.loss import Istft
    from df.model import ModelParams

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carrega dados
    train_speech, test_speech, train_noise, test_noise = load_training_data(
        args.speech_path, args.noise_dir, args.segment_s, args.seed
    )

    # 2. Carrega modelo base e inicializa
    logger.info(f"Carregando modelo inicial no dispositivo {args.device}...")
    base, df_state = load_upstream_dfnet3(device=args.device)
    film_cfg = load_film_config()
    generator = FilmGenerator(film_cfg).to(args.device)
    model = PDFNet3(base, generator).to(args.device)

    if args.init_checkpoint.is_file():
        logger.info(f"Iniciando a partir do checkpoint: {args.init_checkpoint}")
        ckpt = load_checkpoint(args.init_checkpoint)
        model.load_state_dict(ckpt.model, strict=False)

    trainable_params = prepare_frozen_student(model)

    istft = Istft(960, HOP, torch.as_tensor(df_state.fft_window().copy())).to(args.device)
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    rng = np.random.default_rng(args.seed)
    max_freq = int(SR // 2)

    logger.info(f"Iniciando mini fine-tuning por {args.steps} passos...")
    t0 = time.perf_counter()

    losses_log = []
    best_val_loss = float("inf")
    patience_counter = 0

    for step in range(1, args.steps + 1):
        batch = generate_batch(
            train_speech,
            train_noise,
            args.batch_size,
            args.segment_s,
            df_state,
            args.device,
            rng,
        )

        spec, feat_erb, feat_spec = batch["spec"], batch["feat_erb"], batch["feat_spec"]
        spec_clean = batch["spec_clean"]

        # Forward pDFNet3 sem perfil de voz (perfil neutro para foco em supressão acústica)
        dummy_emb = torch.zeros(args.batch_size, film_cfg.embedding_dim, device=args.device)
        dummy_mask = torch.zeros(args.batch_size, 1, device=args.device)

        spec_e, m, lsnr, _ = model(spec.clone(), feat_erb, feat_spec, dummy_emb, dummy_mask)

        # Gating emulado
        spec_m = model.base.mask(spec.clone(), m)
        out_spec, _ = emulate_gating_offline(spec, spec_m, spec_e, lsnr.reshape(args.batch_size, -1))

        est = istft(out_spec)
        with torch.no_grad():
            tgt = istft(spec_clean)

        opt.zero_grad(set_to_none=True)

        mr_loss = dfnet_loss(est, tgt, max_freq, factor=500.0, factor_complex=500.0)
        os_loss = over_suppression_loss(est, tgt, max_freq, factor=300.0)
        total_loss = mr_loss + os_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        opt.step()

        if step % 20 == 0 or step == args.steps:
            logger.info(
                f"Passo [{step:4d}/{args.steps:4d}] | Perda Total: {total_loss.item():.4f} "
                f"(MR: {mr_loss.item():.4f}, OS: {os_loss.item():.4f}) | "
                f"Tempo: {time.perf_counter() - t0:.1f}s"
            )
            losses_log.append({"step": step, "loss": float(total_loss.item())})

        if step % args.val_every == 0 or step == args.steps:
            # Avaliação rigorosa no holding-out test split
            model.eval()
            val_losses = []
            val_rng = np.random.default_rng(1337 + step)
            with torch.no_grad():
                for _ in range(8):
                    v_batch = generate_batch(
                        test_speech,
                        test_noise,
                        args.batch_size,
                        args.segment_s,
                        df_state,
                        args.device,
                        val_rng,
                    )
                    v_spec, v_feat_erb, v_feat_spec = v_batch["spec"], v_batch["feat_erb"], v_batch["feat_spec"]
                    v_clean = v_batch["spec_clean"]
                    v_emb = torch.zeros(args.batch_size, film_cfg.embedding_dim, device=args.device)
                    v_mask = torch.zeros(args.batch_size, 1, device=args.device)

                    v_spec_e, v_m, v_lsnr, _ = model(v_spec.clone(), v_feat_erb, v_feat_spec, v_emb, v_mask)
                    v_spec_m = model.base.mask(v_spec.clone(), v_m)
                    v_out_spec, _ = emulate_gating_offline(v_spec, v_spec_m, v_spec_e, v_lsnr.reshape(args.batch_size, -1))

                    v_est = istft(v_out_spec)
                    v_tgt = istft(v_clean)
                    v_loss = dfnet_loss(v_est, v_tgt, max_freq, factor=500.0, factor_complex=500.0) + over_suppression_loss(v_est, v_tgt, max_freq, factor=300.0)
                    val_losses.append(v_loss.item())

            avg_val_loss = float(np.mean(val_losses))
            logger.info(f"==> Validação [Passo {step:4d}] | Perda Val (holding-out): {avg_val_loss:.4f} (Melhor: {best_val_loss:.4f})")

            save_path = ckpt_dir / f"ckpt-adapt-{step:08d}.pt"
            state = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": None,
                "sampler_state": {"step": step},
                "rng": {},
                "cfg_hash": "local-adapt-v1",
                "commit": None,
                "metric": avg_val_loss,
                "extra": {"adapted": True, "init": str(args.init_checkpoint)},
            }
            torch.save(state, save_path)

            # Rotação inteligente para poupar espaço em disco (mantém apenas os N mais recentes)
            all_step_ckpts = sorted(list(ckpt_dir.glob("ckpt-adapt-[0-9]*.pt")))
            if len(all_step_ckpts) > args.max_checkpoints:
                for old_ckpt in all_step_ckpts[:-args.max_checkpoints]:
                    try:
                        old_ckpt.unlink()
                        logger.info(f"Rotação de disco: removido checkpoint antigo {old_ckpt.name}")
                    except Exception:
                        pass

            if avg_val_loss < (best_val_loss - args.min_delta):
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_path = ckpt_dir / "ckpt-adapt-best.pt"
                torch.save(state, best_path)
                logger.info(f"✨ Novo recorde de validação! Checkpoint salvo em: {best_path}")
            else:
                patience_counter += 1
                logger.info(f"Paciência Early Stopping: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    logger.info(
                        f"🛑 Early Stopping disparado no passo {step}! A perda de validação parou de melhorar por {args.patience} épocas. "
                        f"Melhor perda obtida: {best_val_loss:.4f}"
                    )
                    break

            model.train()
            # Mantém BatchNorm e Encoder rigorosamente congelados
            prepare_frozen_student(model)

    logger.info(
        f"Mini fine-tuning concluído com sucesso em {time.perf_counter() - t0:.1f}s!"
    )


def main() -> int:
    args = parse_args()
    train_adapt(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
