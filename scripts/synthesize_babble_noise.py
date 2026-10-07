#!/usr/bin/env python3
"""Extrator e Sintetizador de Babble Noise (Vozes Interferentes de Fundo)."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("synthesize_babble")

SR = 48000
SEGMENT_S = 6.0
SAMPLES = int(SR * SEGMENT_S)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

def normalize_rms(x: np.ndarray, target_rms: float = 0.04) -> np.ndarray:
    curr = rms(x)
    if curr > 1e-7:
        x = x * (target_rms / curr)
    peak = np.max(np.abs(x))
    if peak > 0.98:
        x = x * (0.98 / peak)
    return x

def fit_length(x: np.ndarray, target_len: int = SAMPLES) -> np.ndarray:
    if len(x) >= target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - len(x)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp3-dir", type=Path, default=Path("data/raw/cv_clips_tmp"))
    parser.add_argument("--local-noise-dir", type=Path, default=Path("data/raw/noise-local-mics/wavs"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/noise-babble-synthetic/wavs"))
    parser.add_argument("--count", type=int, default=3000, help="Quantidade de ruídos compostos a gerar")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    local_wavs = list(args.local_noise_dir.glob("*.wav"))
    all_mp3s = list(args.mp3_dir.rglob("*.mp3"))

    logger.info(f"Carregando {len(all_mp3s)} arquivos MP3 pré-extraídos...")
    clips = []
    for p in all_mp3s:
        try:
            data, orig_sr = sf.read(p, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=-1)
            if orig_sr != SR:
                data = soxr.resample(data, orig_sr, SR)
            clips.append(data)
        except Exception as e:
            continue

    logger.info(f"Total de {len(clips)} falas em português carregadas em RAM.")

    logger.info(f"Gerando {args.count} misturas ricas de babble noise + ruído acústico de notebook...")
    for i in range(1, args.count + 1):
        num_speakers = random.choice([2, 3, 4, 5])
        chosen_speakers = random.sample(clips, num_speakers)

        babble = np.zeros(SAMPLES, dtype=np.float32)
        for spk in chosen_speakers:
            start_offset = random.randint(0, max(0, len(spk) - SAMPLES)) if len(spk) > SAMPLES else 0
            seg = spk[start_offset:start_offset + SAMPLES]
            seg = fit_length(seg, SAMPLES)
            vol = random.uniform(0.15, 0.6)
            babble += vol * normalize_rms(seg, target_rms=0.03)

        if local_wavs:
            bg_path = random.choice(local_wavs)
            bg, _ = sf.read(bg_path, dtype="float32")
            bg = fit_length(bg, SAMPLES)
            mix = babble + random.uniform(0.5, 1.2) * normalize_rms(bg, target_rms=0.02)
        else:
            mix = babble

        mix = fit_length(mix, SAMPLES)
        mix = normalize_rms(mix, target_rms=random.uniform(0.02, 0.05))
        out_name = f"babble_syn_{i:05d}_spk{num_speakers}.wav"
        sf.write(args.out_dir / out_name, mix.astype(np.float32), SR, subtype="PCM_16")

    logger.info(f"Concluído com sucesso! {args.count} amostras salvas em {args.out_dir}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
