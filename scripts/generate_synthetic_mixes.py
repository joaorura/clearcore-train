"""Gera batches / pares sintéticos de treino combinando fala limpa PT com ruído dos microfones locais.

Fontes de fala limpa:
- CML-TTS PT (data/raw/cml-tts-pt)
- Common Voice PT (data/raw/cv27-pt-full / cv27-pt)
- Fala limpa adicional (ex: speech-joao-messias / tts-portuguese-corpus)

Ruídos:
- Ruídos locais recém-extraídos e variantes (data/raw/noise-local-mics)

Faixa de SNR: -5 dB a +15 dB (configurável).
Taxa de amostragem: 48 kHz mono PCM_16 / Float32.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

_root = Path(__file__).resolve().parents[1]


def load_audio(path: Path, target_sr: int = 48000) -> np.ndarray:
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        import scipy.signal
        gcd = np.gcd(sr, target_sr)
        data = scipy.signal.resample_poly(data, target_sr // gcd, sr // gcd).astype(np.float32)
    return data.astype(np.float32)


def make_mixture(speech: np.ndarray, noise: np.ndarray, snr_db: float, duration_sec: float = 6.0, sr: int = 48000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_samples = int(duration_sec * sr)
    
    # Ajustar fala para o comprimento alvo
    if len(speech) < target_samples:
        reps = int(np.ceil(target_samples / len(speech)))
        speech = np.tile(speech, reps)
    start_s = random.randint(0, max(0, len(speech) - target_samples))
    speech_crop = speech[start_s : start_s + target_samples]
    
    # Ajustar ruído para o comprimento alvo
    if len(noise) < target_samples:
        reps = int(np.ceil(target_samples / len(noise)))
        noise = np.tile(noise, reps)
    start_n = random.randint(0, max(0, len(noise) - target_samples))
    noise_crop = noise[start_n : start_n + target_samples]
    
    # Cálculo de potências
    p_speech = np.mean(speech_crop ** 2) + 1e-12
    p_noise = np.mean(noise_crop ** 2) + 1e-12
    
    # Fator de escala do ruído para atingir o SNR desejado
    # SNR = 10 * log10(p_speech / p_noise_scaled) => p_noise_scaled = p_speech / (10^(SNR/10))
    scale_noise = np.sqrt(p_speech / (p_noise * (10.0 ** (snr_db / 10.0))))
    scaled_noise = noise_crop * scale_noise
    
    mixture = speech_crop + scaled_noise
    
    # Prevenção de clipping na mistura com normalização proporcional
    max_peak = np.max(np.abs(mixture))
    if max_peak > 0.95:
        norm_factor = 0.95 / max_peak
        mixture *= norm_factor
        speech_crop *= norm_factor
        scaled_noise *= norm_factor
        
    return mixture.astype(np.float32), speech_crop.astype(np.float32), scaled_noise.astype(np.float32)


def find_speech_files(limit: int = 500) -> list[Path]:
    candidates = []
    # 1. CML-TTS PT
    cml_dir = _root / "data" / "raw" / "cml-tts-pt"
    if cml_dir.exists():
        candidates.extend(list(cml_dir.rglob("*.wav"))[:limit])
        
    # 2. Speech João Messias (gravações de estúdio limpas locais)
    jm_dir = _root / "data" / "raw" / "speech-joao-messias"
    if jm_dir.exists():
        candidates.extend(list(jm_dir.rglob("*.wav"))[:limit])
        
    # 3. Common Voice PT
    cv_dir = _root / "data" / "raw" / "cv27-pt-full"
    if cv_dir.exists() and len(candidates) < limit:
        candidates.extend(list((cv_dir / "clips").glob("*.mp3"))[: limit - len(candidates)])
        
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Gera pares sintéticos de fala limpa + ruído local.")
    parser.add_argument("--num-samples", type=int, default=20, help="Quantidade de amostras a gerar para validação/treino.")
    parser.add_argument("--snr-min", type=float, default=-5.0, help="SNR mínimo em dB.")
    parser.add_argument("--snr-max", type=float, default=15.0, help="SNR máximo em dB.")
    parser.add_argument("--duration", type=float, default=6.0, help="Duração do chunk em segundos.")
    parser.add_argument("--output-dir", type=str, default=str(_root / "data" / "synthetic_pairs_preview"), help="Diretório de saída.")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    speech_files = find_speech_files()
    noise_dir = _root / "data" / "raw" / "noise-local-mics" / "wavs"
    noise_files = list(noise_dir.glob("*.wav"))
    
    if not speech_files or not noise_files:
        print(f"Erro: Arquivos de fala ({len(speech_files)}) ou ruído ({len(noise_files)}) insuficientes.")
        return 1
        
    print(f"Encontrados {len(speech_files)} arquivos de fala limpa e {len(noise_files)} arquivos de ruído.")
    manifest_items = []
    
    for i in range(args.num_samples):
        sp_path = random.choice(speech_files)
        no_path = random.choice(noise_files)
        snr = round(random.uniform(args.snr_min, args.snr_max), 1)
        
        try:
            sp_audio = load_audio(sp_path)
            no_audio = load_audio(no_path)
            mix, clean, noise_scaled = make_mixture(sp_audio, no_audio, snr, duration_sec=args.duration)
            
            pair_id = f"mix_{i+1:04d}_snr{snr:+05.1f}db"
            mix_path = out_dir / f"{pair_id}_noisy.wav"
            clean_path = out_dir / f"{pair_id}_clean.wav"
            
            sf.write(str(mix_path), mix, 48000, subtype="PCM_16")
            sf.write(str(clean_path), clean, 48000, subtype="PCM_16")
            
            manifest_items.append({
                "id": pair_id,
                "snr_db": snr,
                "duration_sec": args.duration,
                "sample_rate": 48000,
                "noisy_path": str(mix_path.relative_to(_root)),
                "clean_path": str(clean_path.relative_to(_root)),
                "clean_source": sp_path.name,
                "noise_source": no_path.name,
            })
        except Exception as e:
            print(f"Erro ao processar par {i}: {e}")
            
    manifest_out = out_dir / "preview_manifest.json"
    manifest_out.write_text(json.dumps(manifest_items, indent=2), encoding="utf-8")
    print(f"Geradas com sucesso {len(manifest_items)} amostras de treino sintetizadas em: {out_dir}")
    print(f"Manifesto registrado em: {manifest_out}")
    return 0


if __name__ == "__main__":
    main()
