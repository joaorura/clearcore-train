#!/usr/bin/env python3
"""PoC de Mistura Sintética e Avaliação Preliminar de Ruídos Reais de Microfone.

Este script executa uma mini-PoC para avaliar a capacidade do modelo pDFNet3 atual
de suprimir ruídos acústicos de notebook e vazamento de voz de fundo gravados localmente,
antes e depois de estratégias de fine-tuning.

Funcionalidades:
1. Carrega a fala limpa (suporta arquivo concatenado ou diretório de enunciados).
2. Carrega ruídos reais fatiados em `data/raw/noise-local-mics/` (acústicos e vozes de fundo).
3. Cria conjuntos segregados (holding-out test split) para validação não enviesada.
4. Gera misturas sintéticas controladas nos SNRs especificados (-5dB, 0dB, 5dB, 10dB)
   e condições de ruído puro / voz de fundo isolada.
5. Executa inferência com o modelo pDFNet3 através da réplica exata do runtime (`runtime_forward`).
6. Computa métricas objetivas (SI-SDR, SNR, STOI, Atenuação de Ruído, Supressão de Voz de Fundo, Gating).
7. Gera relatório estruturado JSON e exporta amostras WAV para inspeção auditiva.

Uso:
    python scripts/poc_mix_and_eval.py
    python scripts/poc_mix_and_eval.py --snrs -5 0 5 10 --save-audio --num-samples 15
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch

# Adiciona o diretório src do repositório ao sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cctrain.data.audio import resample
from cctrain.data.mix import crop_or_pad, loop_to_length, power, scale_for_ratio
from cctrain.eval.functional import si_sdr
from cctrain.eval.metrics import stoi
from cctrain.eval.runtime_path import HOP, Stage, runtime_forward
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmGenerator, load_film_config
from cctrain.models.pdfnet3 import PDFNet3
from cctrain.paths import Paths, repo_root
from cctrain.train.checkpoint import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("poc_eval")

SR = 48000
PEAK_TARGET = 0.95


@dataclass
class MetricSummary:
    count: int
    mean_si_sdr_in: float
    mean_si_sdr_out: float
    mean_delta_si_sdr: float
    mean_snr_in: float
    mean_snr_out: float
    mean_delta_snr: float
    mean_stoi_in: float
    mean_stoi_out: float
    mean_delta_stoi: float
    mean_attenuation_db: float
    stage_percentages: dict[str, float]


def parse_args() -> argparse.Namespace:
    paths = Paths.from_env()
    default_noise = paths.data / "raw" / "noise-local-mics"
    default_ckpt = paths.runs / "m2" / "t3" / "checkpoints" / "ckpt-00022500.pt"
    default_out = paths.runs / "poc_eval"

    parser = argparse.ArgumentParser(
        description="PoC Synthetic Mixing & Evaluation on Local Noises & pDFNet3"
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=default_noise,
        help="Diretório contendo os ruídos fatiados (default: data/raw/noise-local-mics)",
    )
    parser.add_argument(
        "--speech-path",
        type=Path,
        default=None,
        help="Caminho do áudio limpo (arquivo único .wav ou pasta com amostras; default: auto-descoberta)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt,
        help="Checkpoint do pDFNet3 (default: runs/m2/t3/checkpoints/ckpt-00022500.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Diretório de saída para logs e relatórios (default: runs/poc_eval)",
    )
    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=[-5.0, 0.0, 5.0, 10.0],
        help="Lista de SNRs em dB para misturas sintéticas (default: -5 0 5 10)",
    )
    parser.add_argument(
        "--segment-s",
        type=float,
        default=5.0,
        help="Duração de cada segmento de teste em segundos (default: 5.0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Número de pares de teste avaliados por condição (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente pseudo-aleatória (default: 42)",
    )
    parser.add_argument(
        "--holding-out-fraction",
        type=float,
        default=0.3,
        help="Fração de amostras reservada para o teste holding-out (default: 0.3)",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Salva exemplos de áudio WAV (limpo, ruidoso, filtrado) para escuta",
    )
    parser.add_argument(
        "--max-save-audio",
        type=int,
        default=2,
        help="Número máximo de áudios salvos por condição (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dispositivo ('cpu' ou 'cuda'; default: cpu)",
    )
    return parser.parse_args()


def discover_clean_speech(explicit_path: Path | None) -> Path:
    """Localiza fala limpa disponível no ambiente sem caminho pessoal hardcoded."""
    if explicit_path is not None and explicit_path.exists():
        return explicit_path

    # Verifica variável de ambiente opcional
    env_speech = os.environ.get("CLEARCORE_CLEAN_SPEECH")
    if env_speech and Path(env_speech).exists():
        return Path(env_speech)

    paths = Paths.from_env()

    # Opção 0: Gravações reais em speech-joao-messias
    user_speech = paths.data / "raw" / "speech-joao-messias"
    if user_speech.is_dir() and any(user_speech.glob("*.wav")):
        return user_speech

    # Opção A: Arquivo concatenado do subprojeto de áudio de teste se existir
    for candidate_file in [
        repo_root().parents[1] / "test_audio" / "cml_tts_pt_concatenated.wav",
        Path.home() / "test_audio" / "cml_tts_pt_concatenated.wav",
    ]:
        if candidate_file.is_file():
            return candidate_file

    # Opção B: CML-TTS PT dev set do próprio repositório
    cml_dev_audio = paths.data / "raw" / "cml-tts-pt" / "dev" / "audio"
    if cml_dev_audio.is_dir() and any(cml_dev_audio.glob("*/*/*.flac")):
        return cml_dev_audio

    # Opção C: LibriSpeech dev clean
    ls_dev = paths.data / "raw" / "librispeech-dev-clean" / "dev-clean"
    if ls_dev.is_dir() and any(ls_dev.glob("*/*/*.flac")):
        return ls_dev

    raise FileNotFoundError(
        "Nenhuma fonte de fala limpa encontrada. Especifique com --speech-path ou defina CLEARCORE_CLEAN_SPEECH."
    )


def load_clean_speech_segments(
    path: Path, segment_s: float, holding_out_fraction: float, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Carrega segmentos de fala limpa e divide em (train_pool, test_holding_out)."""
    target_len = int(segment_s * SR)
    segments: list[np.ndarray] = []

    if path.is_file():
        logger.info(f"Carregando fala limpa de arquivo único: {path}")
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != SR:
            audio = resample(audio, sr, SR)

        # Fatiamento em janelas sem sobreposição
        total_segs = len(audio) // target_len
        for i in range(total_segs):
            seg = audio[i * target_len : (i + 1) * target_len]
            # Filtra trechos de puro silêncio absoluto (< -60 dBFS)
            p = power(seg)
            if 10 * math.log10(p) > -60.0:
                segments.append(seg)
    else:
        logger.info(f"Carregando enunciados de diretório de fala: {path}")
        files = sorted(list(path.glob("**/*.flac")) + list(path.glob("**/*.wav")))
        rng = np.random.default_rng(seed)
        for f in files:
            try:
                audio, sr = sf.read(f, dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                if sr != SR:
                    audio = resample(audio, sr, SR)
                seg = crop_or_pad(audio, target_len, rng)
                p = power(seg)
                if 10 * math.log10(p) > -50.0:
                    segments.append(seg)
            except Exception:
                continue

    if not segments:
        raise ValueError(f"Nenhum segmento de fala válido extraído de {path}")

    logger.info(f"Total de segmentos de fala limpa extraídos: {len(segments)}")

    # Divisão determinística
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(segments))
    split_idx = int(len(segments) * (1.0 - holding_out_fraction))
    train_pool = [segments[i] for i in perm[:split_idx]]
    test_pool = [segments[i] for i in perm[split_idx:]]

    logger.info(
        f"Divisão de Fala: {len(train_pool)} segmentos de treino/adaptação, "
        f"{len(test_pool)} segmentos de teste holding-out"
    )
    return train_pool, test_pool


def load_noise_slices(
    noise_dir: Path, holding_out_fraction: float, seed: int
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Carrega os ruídos reais fatiados e os organiza por categoria e divisão (train/test)."""
    if not noise_dir.is_dir():
        raise FileNotFoundError(f"Diretório de ruídos não encontrado: {noise_dir}")

    files = sorted(list(noise_dir.glob("**/*.wav")) + list(noise_dir.glob("**/*.flac")))
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo de áudio (.wav/.flac) encontrado em {noise_dir}."
        )

    categories: dict[str, list[np.ndarray]] = defaultdict(list)

    for f in files:
        audio, sr = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != SR:
            import resampy

            audio = resampy.resample(audio, sr, SR)

        parent_name = f.parent.name.lower()
        fname = f.name.lower()

        if "notebook" in parent_name or "acoustic" in parent_name or "notebook" in fname:
            cat = "notebook_acoustic"
        elif "mic1" in parent_name or "mic1" in fname:
            cat = "mic1_background_voice"
        elif "mic2" in parent_name or "mic2" in fname:
            cat = "mic2_background_voice"
        elif "voice" in parent_name or "voice" in fname or "interf" in fname:
            cat = "background_voice"
        else:
            cat = "other_noise"

        categories[cat].append(audio)

    # Divisão por categoria
    rng = np.random.default_rng(seed)
    splits: dict[str, dict[str, list[np.ndarray]]] = {}

    for cat, audios in categories.items():
        perm = rng.permutation(len(audios))
        n_test = max(1, int(len(audios) * holding_out_fraction))
        n_train = max(1, len(audios) - n_test)
        train_list = [audios[i] for i in perm[:n_train]]
        test_list = [audios[i] for i in perm[n_train:]]
        splits[cat] = {"train": train_list, "test": test_list}
        logger.info(
            f"Categoria '{cat}': {len(train_list)} fatias treino/adaptação, "
            f"{len(test_list)} fatias teste holding-out"
        )

    return splits


def load_model(checkpoint_path: Path, device: str = "cpu") -> PDFNet3:
    """Carrega o modelo pDFNet3 com os pesos do checkpoint especificado."""
    logger.info(f"Carregando modelo base DFNet3 upstream no dispositivo '{device}'...")
    base, _ = load_upstream_dfnet3(device=device)
    film_cfg = load_film_config()
    generator = FilmGenerator(film_cfg).to(device)
    model = PDFNet3(base, generator).to(device)

    if checkpoint_path.is_file():
        logger.info(f"Carregando pesos pDFNet3 de checkpoint: {checkpoint_path}")
        try:
            ckpt = load_checkpoint(checkpoint_path)
            state_dict = ckpt.model
        except Exception:
            raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = raw.get("model", raw)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                f"Avisos ao carregar checkpoint: faltantes={len(missing)}, inesperados={len(unexpected)}"
            )
        else:
            logger.info("Pesos do modelo carregados com sucesso.")
    else:
        logger.warning(
            f"Checkpoint {checkpoint_path} não encontrado! Usando modelo inicial DFNet3 upstream."
        )

    model.eval()
    return model


def compute_audio_snr(signal: np.ndarray, noise: np.ndarray, eps: float = 1e-12) -> float:
    """Calcula SNR clássico baseado na potência média."""
    p_sig = power(signal)
    p_noise = power(noise)
    return float(10.0 * np.log10(max(p_sig, eps) / max(p_noise, eps)))


def evaluate_single_sample(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float | None,
    model: PDFNet3,
    rng: np.random.Generator,
    condition_type: str,
) -> dict[str, Any]:
    """Cria a mistura, roda o modelo via runtime_forward e calcula métricas completas."""
    target_len = len(clean)

    if condition_type == "noise_only":
        # Ruído puro (Interferente ou acústico isolado): sem fala primária
        n_seg = loop_to_length(noise, target_len, rng)
        # Normalização de pico para teste
        peak = max(np.max(np.abs(n_seg)), 1e-9)
        noisy = n_seg * (PEAK_TARGET / peak)
        ref_clean = np.zeros(target_len, dtype=np.float32)

        # Inferência
        enh, stages = runtime_forward(model, None, noisy)

        p_in = power(noisy)
        p_out = power(enh)
        atten_db = float(10.0 * np.log10(max(p_in, 1e-12) / max(p_out, 1e-12)))

        return {
            "condition": condition_type,
            "snr_target": None,
            "attenuation_db": atten_db,
            "p_in_dbfs": float(10.0 * np.log10(p_in)),
            "p_out_dbfs": float(10.0 * np.log10(p_out)),
            "stages": stages,
            "audio": {"clean": ref_clean, "noisy": noisy, "enhanced": enh},
        }

    # Mistura fala + ruído no SNR especificado
    assert snr_db is not None
    n_seg = loop_to_length(noise, target_len, rng)
    scaled_n = scale_for_ratio(clean, n_seg, snr_db)
    noisy_raw = clean + scaled_n

    peak = max(np.max(np.abs(noisy_raw)), np.max(np.abs(clean)), 1e-9)
    gain = PEAK_TARGET / peak if peak > PEAK_TARGET else 1.0

    noisy = (noisy_raw * gain).astype(np.float32)
    clean_scaled = (clean * gain).astype(np.float32)
    residual_noise_in = noisy - clean_scaled

    # Inferência via réplica de runtime com delay compensation de 1440 amostras
    enh, stages = runtime_forward(model, None, noisy)
    residual_noise_out = enh - clean_scaled

    # Métricas objetivas
    sdr_in = float(si_sdr(noisy, clean_scaled))
    sdr_out = float(si_sdr(enh, clean_scaled))
    delta_sdr = sdr_out - sdr_in

    snr_in_val = compute_audio_snr(clean_scaled, residual_noise_in)
    snr_out_val = compute_audio_snr(clean_scaled, residual_noise_out)
    delta_snr = snr_out_val - snr_in_val

    stoi_in_val = float(stoi(noisy, clean_scaled, SR)[0])
    stoi_out_val = float(stoi(enh, clean_scaled, SR)[0])
    delta_stoi = stoi_out_val - stoi_in_val

    p_in = power(noisy)
    p_out = power(enh)
    atten_db = float(10.0 * np.log10(max(p_in, 1e-12) / max(p_out, 1e-12)))

    return {
        "condition": condition_type,
        "snr_target": snr_db,
        "si_sdr_in": sdr_in,
        "si_sdr_out": sdr_out,
        "delta_si_sdr": delta_sdr,
        "snr_in": snr_in_val,
        "snr_out": snr_out_val,
        "delta_snr": delta_snr,
        "stoi_in": stoi_in_val,
        "stoi_out": stoi_out_val,
        "delta_stoi": delta_stoi,
        "attenuation_db": atten_db,
        "stages": stages,
        "audio": {"clean": clean_scaled, "noisy": noisy, "enhanced": enh},
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> MetricSummary:
    """Agrega os resultados de uma lista de avaliações."""
    if not results:
        return MetricSummary(
            count=0,
            mean_si_sdr_in=0.0,
            mean_si_sdr_out=0.0,
            mean_delta_si_sdr=0.0,
            mean_snr_in=0.0,
            mean_snr_out=0.0,
            mean_delta_snr=0.0,
            mean_stoi_in=0.0,
            mean_stoi_out=0.0,
            mean_delta_stoi=0.0,
            mean_attenuation_db=0.0,
            stage_percentages={},
        )

    n = len(results)
    attenuations = [r["attenuation_db"] for r in results if "attenuation_db" in r]

    # Para misturas com fala
    speech_results = [r for r in results if "si_sdr_in" in r]
    m_sdr_in = float(np.mean([r["si_sdr_in"] for r in speech_results])) if speech_results else 0.0
    m_sdr_out = float(np.mean([r["si_sdr_out"] for r in speech_results])) if speech_results else 0.0
    m_delta_sdr = float(np.mean([r["delta_si_sdr"] for r in speech_results])) if speech_results else 0.0

    m_snr_in = float(np.mean([r["snr_in"] for r in speech_results])) if speech_results else 0.0
    m_snr_out = float(np.mean([r["snr_out"] for r in speech_results])) if speech_results else 0.0
    m_delta_snr = float(np.mean([r["delta_snr"] for r in speech_results])) if speech_results else 0.0

    m_stoi_in = float(np.mean([r["stoi_in"] for r in speech_results])) if speech_results else 0.0
    m_stoi_out = float(np.mean([r["stoi_out"] for r in speech_results])) if speech_results else 0.0
    m_delta_stoi = float(np.mean([r["delta_stoi"] for r in speech_results])) if speech_results else 0.0

    m_atten = float(np.mean(attenuations)) if attenuations else 0.0

    # Contagem de estágios de gating
    all_stages = np.concatenate([r["stages"] for r in results])
    total_frames = max(len(all_stages), 1)
    stage_counts = {
        "SILENT": float(np.sum(all_stages == int(Stage.SILENT)) / total_frames * 100.0),
        "ZERO": float(np.sum(all_stages == int(Stage.ZERO)) / total_frames * 100.0),
        "ERB_ONLY": float(np.sum(all_stages == int(Stage.ERB_ONLY)) / total_frames * 100.0),
        "ERB_DF": float(np.sum(all_stages == int(Stage.ERB_DF)) / total_frames * 100.0),
        "NONE": float(np.sum(all_stages == int(Stage.NONE)) / total_frames * 100.0),
    }

    return MetricSummary(
        count=n,
        mean_si_sdr_in=m_sdr_in,
        mean_si_sdr_out=m_sdr_out,
        mean_delta_si_sdr=m_delta_sdr,
        mean_snr_in=m_snr_in,
        mean_snr_out=m_snr_out,
        mean_delta_snr=m_delta_snr,
        mean_stoi_in=m_stoi_in,
        mean_stoi_out=m_stoi_out,
        mean_delta_stoi=m_delta_stoi,
        mean_attenuation_db=m_atten,
        stage_percentages=stage_counts,
    )


def run_poc(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = args.out_dir / "audio"
    if args.save_audio:
        audio_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Localização e carga de dados
    clean_speech_path = discover_clean_speech(args.speech_path)
    _, speech_test_pool = load_clean_speech_segments(
        clean_speech_path, args.segment_s, args.holding_out_fraction, args.seed
    )

    noise_splits = load_noise_slices(args.noise_dir, args.holding_out_fraction, args.seed)

    # 2. Carga do Modelo
    model = load_model(args.checkpoint, args.device)

    rng = np.random.default_rng(args.seed)

    all_evaluations: dict[str, dict[str, Any]] = {}
    summary_report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(args.checkpoint),
        "speech_source": str(clean_speech_path),
        "noise_dir": str(args.noise_dir),
        "snrs": args.snrs,
        "segment_s": args.segment_s,
        "num_samples_per_condition": args.num_samples,
        "conditions": {},
    }

    print("\n" + "=" * 90)
    print(f"BENCHMARK PoC: AVALIAÇÃO DE RUÍDOS REAIS & pDFNet3")
    print(f"Modelo: {args.checkpoint.name} | Device: {args.device} | Segmento: {args.segment_s}s")
    print("=" * 90)

    for cat_name, cat_split in noise_splits.items():
        test_noises = cat_split["test"]
        if not test_noises:
            continue

        print(f"\n>>> Categoria de Ruído: [{cat_name}] ({len(test_noises)} fatias de teste)")
        summary_report["conditions"][cat_name] = {}

        # 1. Avalia em cada SNR
        for snr_db in args.snrs:
            cond_key = f"mix_snr_{snr_db:+03.0f}dB"
            results = []

            for i in range(args.num_samples):
                clean_sample = speech_test_pool[int(rng.integers(len(speech_test_pool)))]
                noise_sample = test_noises[int(rng.integers(len(test_noises)))]

                res = evaluate_single_sample(
                    clean_sample, noise_sample, snr_db, model, rng, "speech_and_noise"
                )
                results.append(res)

                # Salva áudio para escuta
                if args.save_audio and i < args.max_save_audio:
                    prefix = f"{cat_name}_snr{snr_db:+02.0f}dB_s{i+1}"
                    sf.write(audio_out_dir / f"{prefix}_clean.wav", res["audio"]["clean"], SR)
                    sf.write(audio_out_dir / f"{prefix}_noisy.wav", res["audio"]["noisy"], SR)
                    sf.write(audio_out_dir / f"{prefix}_enhanced.wav", res["audio"]["enhanced"], SR)

            agg = aggregate_metrics(results)
            summary_report["conditions"][cat_name][cond_key] = asdict(agg)

            print(
                f"  SNR {snr_db:+3.0f} dB: "
                f"ΔSI-SDR: {agg.mean_delta_si_sdr:+5.2f} dB "
                f"(in: {agg.mean_si_sdr_in:5.2f} -> out: {agg.mean_si_sdr_out:5.2f}) | "
                f"ΔSNR: {agg.mean_delta_snr:+5.2f} dB | "
                f"ΔSTOI: {agg.mean_delta_stoi:+5.3f} | "
                f"DF/ERB: {agg.stage_percentages['ERB_DF']:.0f}%"
            )

        # 2. Avalia ruído isolado / vazamento isolado (sem voz primária)
        cond_key = "isolated_noise_only"
        results_iso = []
        for i in range(args.num_samples):
            dummy_clean = np.zeros(int(args.segment_s * SR), dtype=np.float32)
            noise_sample = test_noises[int(rng.integers(len(test_noises)))]
            res = evaluate_single_sample(
                dummy_clean, noise_sample, None, model, rng, "noise_only"
            )
            results_iso.append(res)

            if args.save_audio and i < args.max_save_audio:
                prefix = f"{cat_name}_isolated_s{i+1}"
                sf.write(audio_out_dir / f"{prefix}_in.wav", res["audio"]["noisy"], SR)
                sf.write(audio_out_dir / f"{prefix}_enhanced.wav", res["audio"]["enhanced"], SR)

        agg_iso = aggregate_metrics(results_iso)
        summary_report["conditions"][cat_name][cond_key] = asdict(agg_iso)
        term_label = (
            "Supressão de Vazamento de Voz"
            if "voice" in cat_name
            else "Atenuação de Ruído Acústico"
        )
        print(
            f"  {term_label} Puro: "
            f"{agg_iso.mean_attenuation_db:5.2f} dB de redução "
            f"(Gating ZERO/SILENT: {agg_iso.stage_percentages['ZERO'] + agg_iso.stage_percentages['SILENT']:.0f}%)"
        )

    total_time = time.perf_counter() - t0
    summary_report["total_time_seconds"] = total_time

    # Salva relatório JSON completo
    report_path = args.out_dir / "poc_eval_report.json"
    report_path.write_text(json.dumps(summary_report, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 90)
    print(f"PoC concluída em {total_time:.1f}s. Relatório gravado em: {report_path}")
    if args.save_audio:
        print(f"Amostras WAV exportadas em: {audio_out_dir}")
    print("=" * 90)

    return summary_report


def main() -> int:
    args = parse_args()
    run_poc(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
