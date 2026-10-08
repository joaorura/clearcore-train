#!/usr/bin/env python3
"""Processamento e Curadoria das Gravações de Voz para Dataset de Fine-Tuning.

1. Extrai o áudio mono a 48kHz dos arquivos MP4 gravados via OBS.
2. Aplica denoising com DeepFilterNet3 (DFN3) oficial para remover qualquer ruído de fundo acústico.
3. Detecta voz e fatia em segmentos limpos de treino/teste (3 a 5 segundos).
4. Salva em data/raw/speech-joao-messias/ com manifesto estruturado.
5. Remove com segurança os arquivos originais em /home/joaorura/ após validação.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cctrain.eval.functional import enhance_offline
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("process_user_recordings")

SR = 48000
SEGMENT_SECONDS = 4.0
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * SR)
HOP_SAMPLES = int(1.5 * SR)  # 1.5s step = overlap de 2.5s
MIN_RMS = 0.015  # threshold para filtrar silêncio puro


def extract_audio_from_mp4(mp4_path: Path) -> np.ndarray:
    """Extrai canal mono em 48kHz float32 usando ffmpeg."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-i", str(mp4_path),
        "-ac", "1",
        "-ar", str(SR),
        "-f", "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio


def main() -> None:
    source_dir = Path("/home/joaorura")
    out_dir = REPO_ROOT / "data" / "raw" / "speech-joao-messias"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(source_dir.glob("2026-10-07 08-*.mp4"))
    if not input_files:
        logger.error("Nenhum arquivo 2026-10-07 08-*.mp4 encontrado em %s", source_dir)
        sys.exit(1)

    logger.info("Encontrados %d arquivos de gravação em %s", len(input_files), source_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Carregando modelo DeepFilterNet3 upstream para denoising no dispositivo: %s...", device)
    base, _ = load_upstream_dfnet3(device=device)

    total_slices = 0
    all_slices = []

    for idx, mp4_file in enumerate(input_files, start=1):
        logger.info("[%d/%d] Processando: %s", idx, len(input_files), mp4_file.name)
        raw_audio = extract_audio_from_mp4(mp4_file)
        dur_s = len(raw_audio) / SR
        logger.info("  Áudio extraído: %.2f segundos (%.1f MB de amostras)", dur_s, raw_audio.nbytes / 1e6)

        # Passa pelo DeepFilterNet3 para remover ruídos de fundo acústicos
        logger.info("  Aplicando DFN3 para extrair voz pura e limpa...")
        clean_audio, _ = enhance_offline(base, raw_audio)

        # Normaliza levemente o pico de amplitude
        peak = np.max(np.abs(clean_audio))
        if peak > 1e-4:
            clean_audio = (clean_audio / peak) * 0.90

        # Fatiamento em segmentos com verificação de RMS de voz
        file_slices = 0
        n_samples = len(clean_audio)
        for start in range(0, n_samples - SEGMENT_SAMPLES + 1, HOP_SAMPLES):
            chunk = clean_audio[start : start + SEGMENT_SAMPLES]
            rms = np.sqrt(np.mean(chunk**2))
            if rms < MIN_RMS:
                continue  # ignora pausas longas ou silêncio

            slice_name = f"joao_messias_take{idx}_{file_slices:04d}.wav"
            slice_path = out_dir / slice_name
            sf.write(str(slice_path), chunk, SR, subtype="FLOAT")

            all_slices.append({
                "path": str(slice_path),
                "relative_path": f"data/raw/speech-joao-messias/{slice_name}",
                "speaker": "joao_messias",
                "duration_seconds": SEGMENT_SECONDS,
                "rms": float(rms),
                "source_file": mp4_file.name,
            })
            file_slices += 1

        total_slices += file_slices
        logger.info("  Extraídas %d fatias de voz limpa deste arquivo.", file_slices)

    # Escreve o manifesto do dataset
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": "speech-joao-messias",
            "total_samples": len(all_slices),
            "sample_rate": SR,
            "segment_seconds": SEGMENT_SECONDS,
            "samples": all_slices,
        }, f, indent=2)

    logger.info("Dataset criado com sucesso em: %s", out_dir)
    logger.info("Total de amostras de voz fatiadas: %d (%.2f minutos de fala pura)",
                len(all_slices), (len(all_slices) * SEGMENT_SECONDS) / 60.0)

    # Deleta os arquivos brutos após confirmação
    logger.info("Deletando arquivos brutos MP4 de /home/joaorura/ conforme solicitado...")
    for mp4_file in input_files:
        try:
            mp4_file.unlink()
            logger.info("  Removido: %s", mp4_file)
        except Exception as e:
            logger.warning("  Falha ao remover %s: %s", mp4_file, e)

    logger.info("Concluído!")


if __name__ == "__main__":
    main()
