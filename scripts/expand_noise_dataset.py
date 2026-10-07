#!/usr/bin/env python3
"""Expansor e Sintetizador Combinatório de Base de Ruídos (Multi-Noise Combiner)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("expand_noise")

SR = 48000
SEGMENT_S = 6.0
SAMPLES = int(SR * SEGMENT_S)

def make_rir(length_s: float = 0.3, rt60: float = 0.25) -> np.ndarray:
    n = int(SR * length_s)
    decay = np.exp(-3.0 * np.log(10) * np.linspace(0, length_s, n) / max(rt60, 0.05))
    h = np.random.randn(n) * decay
    h[0] = 1.0
    norm = np.sqrt(np.sum(h ** 2))
    return (h / max(norm, 1e-6)).astype(np.float32)

def fit_length(x: np.ndarray, target_len: int = SAMPLES) -> np.ndarray:
    if len(x) >= target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - len(x)))

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

def normalize_rms(x: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    x = fit_length(x, SAMPLES)
    curr = rms(x)
    if curr > 1e-7:
        x = x * (target_rms / curr)
    peak = np.max(np.abs(x))
    if peak > 0.98:
        x = x * (0.98 / peak)
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-dir", type=Path, default=Path("data/raw/noise-local-mics/wavs"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/noise-composite/wavs"))
    parser.add_argument("--count", type=int, default=3500, help="Quantidade de amostras combinadas a gerar")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_wavs = list(args.noise_dir.glob("*.wav"))
    if not all_wavs:
        logger.error(f"Nenhum WAV encontrado em {args.noise_dir}")
        return 1

    fan_wavs = [w for w in all_wavs if "builtin" in w.name or "fan" in w.name]
    voice_wavs = [w for w in all_wavs if "background_voice" in w.name or "babble" in w.name or "142752" in w.name or "110107" in w.name]
    ambient_wavs = [w for w in all_wavs if "external" in w.name or "ambient" in w.name or "111405" in w.name]

    logger.info(f"Arquivos base: {len(all_wavs)} (Fans: {len(fan_wavs)}, Vozes: {len(voice_wavs)}, Ambiente: {len(ambient_wavs)})")

    manifest = []
    logger.info(f"Gerando {args.count} amostras compostas e enriquecidas...")

    for i in range(1, args.count + 1):
        strategy = random.choice([
            "fan_plus_voice",
            "ambient_plus_voice",
            "voice_double_babble",
            "fan_plus_ambient_plus_voice",
            "reverberant_voice_leak",
        ])

        if strategy == "fan_plus_voice" and fan_wavs and voice_wavs:
            f1, _ = sf.read(random.choice(fan_wavs), dtype="float32")
            f2, _ = sf.read(random.choice(voice_wavs), dtype="float32")
            w1 = random.uniform(0.5, 1.2)
            w2 = random.uniform(0.4, 1.5)
            mix = w1 * normalize_rms(f1, 0.03) + w2 * normalize_rms(f2, 0.03)
        elif strategy == "voice_double_babble" and len(voice_wavs) >= 2:
            f1, _ = sf.read(random.choice(voice_wavs), dtype="float32")
            f2, _ = sf.read(random.choice(voice_wavs), dtype="float32")
            mix = normalize_rms(f1, 0.03) + random.uniform(0.6, 1.2) * normalize_rms(f2, 0.03)
        elif strategy == "fan_plus_ambient_plus_voice" and fan_wavs and ambient_wavs and voice_wavs:
            f1, _ = sf.read(random.choice(fan_wavs), dtype="float32")
            f2, _ = sf.read(random.choice(ambient_wavs), dtype="float32")
            f3, _ = sf.read(random.choice(voice_wavs), dtype="float32")
            mix = (
                0.8 * normalize_rms(f1, 0.02)
                + 0.6 * normalize_rms(f2, 0.02)
                + random.uniform(0.7, 1.4) * normalize_rms(f3, 0.03)
            )
        elif strategy == "reverberant_voice_leak" and voice_wavs:
            f1, _ = sf.read(random.choice(voice_wavs), dtype="float32")
            rir = make_rir(length_s=random.uniform(0.2, 0.45), rt60=random.uniform(0.2, 0.4))
            conv = fftconvolve(f1, rir, mode="same")
            mix = normalize_rms(conv, 0.04)
        else:
            w = random.choice(all_wavs)
            f, _ = sf.read(w, dtype="float32")
            mix = normalize_rms(f, random.uniform(0.02, 0.06))

        mix = fit_length(mix, SAMPLES)
        peak = np.max(np.abs(mix))
        if peak > 0.98:
            mix = mix * (0.98 / peak)

        out_name = f"composite_{i:05d}_{strategy}.wav"
        out_path = args.out_dir / out_name
        sf.write(out_path, mix.astype(np.float32), SR, subtype="PCM_16")

        h = hashlib.sha256(mix.tobytes()).hexdigest()
        manifest.append({
            "name": out_name,
            "strategy": strategy,
            "num_samples": SAMPLES,
            "sr": SR,
            "sha256": h,
            "rms_dbfs": float(20 * np.log10(max(rms(mix), 1e-7))),
        })

    meta_file = args.out_dir.parent / "composite_manifest.json"
    with open(meta_file, "w") as fh:
        json.dump({"total": len(manifest), "samples": manifest}, fh, indent=2)

    logger.info(f"Concluído! {len(manifest)} amostras compostas salvas em {args.out_dir}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
