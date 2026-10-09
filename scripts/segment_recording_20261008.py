"""Segmentador e extrator das fontes de áudio da gravação de 2026-10-08 para o dataset noise-local-mics.

Identifica e fatias trechos de:
- fan_steady (ruído contínuo de ventoinha/laptop)
- typing_clicks (ruídos de digitação de teclado e cliques de mouse/trackpad)
- background_babble (conversas ao fundo / fala distante / resíduos vocais)
- primary_speech (trechos com presença de fala direta do usuário)
- room_ambient (ruído composto do ambiente acústico da sala)

Aplica variantes de data augmentation estruturadas (raw, gain_m6db, gain_p3db, lowpass, highpass, speed_0.95x, speed_1.05x).
Atualiza manifest.json, noise_manifest.csv e manifests/files/noise-local-mics.jsonl.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parents[1] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from dataclasses import dataclass
import numpy as np
import scipy.signal
import soundfile as sf

from cctrain.paths import repo_root


@dataclass
class ChunkRecord:
    rel_path: str
    filename: str
    duration_sec: float
    num_samples: int
    sample_rate: int
    channels: int
    codec: str
    rms_dbfs: float
    peak_dbfs: float
    source_file: str
    segment_index: int
    device: str
    augmentation: str
    tags: list[str]
    sha256: str


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_augmented_variants(base_chunk: np.ndarray, sr: int = 48000) -> dict[str, tuple[np.ndarray, list[str]]]:
    variants = {}
    
    # 1. Raw
    variants["raw"] = (base_chunk.copy(), ["variant_raw"])
    
    # 2. Gain -6dB
    gain_m6 = base_chunk * float(10.0 ** (-6.0 / 20.0))
    variants["gain_m6db"] = (gain_m6, ["variant_gain_m6db", "attenuated"])
    
    # 3. Gain +3dB
    gain_p3 = base_chunk * float(10.0 ** (3.0 / 20.0))
    p_max = float(np.max(np.abs(gain_p3)))
    if p_max > 0.98:
        gain_p3 = gain_p3 * (0.98 / p_max)
    variants["gain_p3db"] = (gain_p3, ["variant_gain_p3db", "boosted"])
    
    # 4. Lowpass (4.5 kHz)
    sos_lp = scipy.signal.butter(2, 4500, btype="lowpass", fs=sr, output="sos")
    lp = scipy.signal.sosfilt(sos_lp, base_chunk).astype(np.float32)
    variants["lowpass"] = (lp, ["variant_lowpass", "distant_room_acoustic"])
    
    # 5. Highpass (180 Hz)
    sos_hp = scipy.signal.butter(2, 180, btype="highpass", fs=sr, output="sos")
    hp = scipy.signal.sosfilt(sos_hp, base_chunk).astype(np.float32)
    variants["highpass"] = (hp, ["variant_highpass", "rumble_filtered"])
    
    # 6. Speed 0.95x
    slow = scipy.signal.resample_poly(base_chunk, 20, 19).astype(np.float32)
    variants["speed_0.95x"] = (slow, ["variant_speed_slow", "time_stretched"])
    
    # 7. Speed 1.05x
    fast = scipy.signal.resample_poly(base_chunk, 20, 21).astype(np.float32)
    variants["speed_1.05x"] = (fast, ["variant_speed_fast", "time_compressed"])
    
    return variants


def main():
    root = repo_root()
    video_path = Path("/home/joaorura/2026-10-08 14-30-02.mp4")
    noise_dir = root / "data" / "raw" / "noise-local-mics"
    wav_dir = noise_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    
    # Load 48k audio
    cmd = ["ffmpeg", "-nostdin", "-i", str(video_path), "-vn", "-ar", "48000", "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    raw_audio = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2)
    sr = 48000
    
    # Peak normalize entire recording if needed
    p_max = np.max(np.abs(raw_audio))
    scale = 0.95 / p_max if p_max > 0.95 else 1.0
    audio = raw_audio * scale
    
    ch_l = audio[:, 0]
    ch_r = audio[:, 1]
    ch_mid = (ch_l + ch_r) * 0.5
    
    # Segment into 6-second segments with non-overlapping / lightly stepped chunks
    seg_len_sec = 6.0
    seg_samples = int(seg_len_sec * sr)
    n_total_segs = len(audio) // seg_samples
    
    print(f"Total video length: {len(audio)/sr:.2f}s -> {n_total_segs} segments of {seg_len_sec}s")
    
    new_records: list[ChunkRecord] = []
    source_stem = "20261008_143002"
    device_label = "mic_obs_3mic_combo"
    
    for seg_idx in range(n_total_segs):
        t_start = seg_idx * seg_len_sec
        t_end = (seg_idx + 1) * seg_len_sec
        
        # We process the balanced mono channel (ch_mid) as the canonical acoustic mix of the 3 local mics
        base_chunk = ch_mid[seg_idx * seg_samples : (seg_idx + 1) * seg_samples]
        
        peak = float(np.max(np.abs(base_chunk)))
        rms = float(np.sqrt(np.mean(base_chunk**2))) + 1e-9
        crest = peak / rms
        
        # Energy variation inside the segment (25ms windows)
        flen, fhop = int(0.025 * sr), int(0.010 * sr)
        nfr = (len(base_chunk) - flen) // fhop
        rframes = np.array([np.sqrt(np.mean(base_chunk[j*fhop:j*fhop+flen]**2))+1e-9 for j in range(nfr)])
        db_frames = 20 * np.log10(rframes)
        dyn_range = float(np.percentile(db_frames, 95) - np.percentile(db_frames, 10))
        std_db = float(np.std(db_frames))
        
        # Spectral distribution
        fft = np.abs(np.fft.rfft(base_chunk))
        freqs = np.fft.rfftfreq(len(base_chunk), 1/sr)
        sc = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-9))
        
        tags = [device_label, "local_mics_combo", "obs_recording"]
        
        # Classification of content
        if crest > 6.0 or (dyn_range > 8.0 and sc > 3800):
            tags.extend(["typing_clicks", "transient_clicks", "laptop_keyboard"])
        elif dyn_range > 7.2 and std_db > 2.5:
            # Voice / babble disturbance
            if rms > 0.12:  # louder speech
                tags.extend(["primary_speech", "voice_activity"])
            else:
                tags.extend(["background_babble", "distant_speech"])
        elif std_db < 1.8 and crest < 4.5:
            tags.extend(["fan_steady", "cooling_fan", "continuous_noise"])
        else:
            tags.extend(["room_ambient", "diffuse_noise"])
            
        variants = generate_augmented_variants(base_chunk, sr=sr)
        
        for var_name, (var_audio, var_tags) in variants.items():
            filename = f"{device_label}_{source_stem}_seg{seg_idx+1:04d}_{var_name}.wav"
            out_path = wav_dir / filename
            
            clipped = np.clip(var_audio, -1.0, 1.0)
            sf.write(str(out_path), clipped, sr, subtype="PCM_16")
            
            rms_c = float(np.sqrt(np.mean(clipped**2))) + 1e-9
            peak_c = float(np.max(np.abs(clipped)))
            rms_db = 20 * np.log10(rms_c)
            peak_db = 20 * np.log10(peak_c + 1e-9)
            
            rec = ChunkRecord(
                rel_path=f"wavs/{filename}",
                filename=filename,
                duration_sec=round(len(clipped) / sr, 4),
                num_samples=len(clipped),
                sample_rate=sr,
                channels=1,
                codec="pcm_s16le",
                rms_dbfs=round(rms_db, 2),
                peak_dbfs=round(peak_db, 2),
                source_file=video_path.name,
                segment_index=seg_idx + 1,
                device=device_label,
                augmentation=var_name,
                tags=tags + var_tags,
                sha256=compute_sha256(out_path),
            )
            new_records.append(rec)
            
    print(f"Generated {len(new_records)} new chunk variants ({n_total_segs} base segments * 7 augmentations).")
    
    # 1. Update noise_manifest.csv
    csv_path = noise_dir / "noise_manifest.csv"
    existing_csv_rows = []
    header = None
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            existing_csv_rows = list(reader)
            
    # Filter out any prior entries for this file if rerun
    filtered_csv_rows = [r for r in existing_csv_rows if r[9] != video_path.name]
    
    for r in new_records:
        filtered_csv_rows.append([
            r.rel_path,
            r.filename,
            str(r.duration_sec),
            str(r.num_samples),
            str(r.sample_rate),
            str(r.channels),
            r.codec,
            str(r.rms_dbfs),
            str(r.peak_dbfs),
            r.source_file,
            str(r.segment_index),
            r.device,
            r.augmentation,
            ";".join(r.tags),
            r.sha256,
        ])
        
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header or [
            "path", "filename", "duration_sec", "num_samples", "sample_rate",
            "channels", "codec", "rms_dbfs", "peak_dbfs", "source_file",
            "segment_index", "device", "augmentation", "tags", "sha256"
        ])
        writer.writerows(filtered_csv_rows)
        
    print(f"Updated CSV: {csv_path} (total entries: {len(filtered_csv_rows)})")
    
    # 2. Update manifest.json
    json_path = noise_dir / "manifest.json"
    manifest_data = {}
    if json_path.exists():
        manifest_data = json.loads(json_path.read_text(encoding="utf-8"))
        
    # Recalculate summary from updated CSV rows
    total_samples = len(filtered_csv_rows)
    total_dur_sec = sum(float(r[2]) for r in filtered_csv_rows)
    source_files = set(r[9] for r in filtered_csv_rows)
    
    dev_counts = {}
    aug_counts = {}
    tag_counts = {}
    for r in filtered_csv_rows:
        dev = r[11]
        aug = r[12]
        tags_list = r[13].split(";")
        dev_counts[dev] = dev_counts.get(dev, 0) + 1
        aug_counts[aug] = aug_counts.get(aug, 0) + 1
        for t in tags_list:
            if t:
                tag_counts[t] = tag_counts.get(t, 0) + 1
                
    manifest_data["summary"] = {
        "total_samples": total_samples,
        "total_duration_sec": round(total_dur_sec, 2),
        "total_duration_minutes": round(total_dur_sec / 60.0, 2),
        "source_files_count": len(source_files),
        "device_counts": dev_counts,
        "augmentation_counts": aug_counts,
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
    }
    json_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated JSON manifest: {json_path}")
    
    # 3. Update manifests/files/noise-local-mics.jsonl
    jsonl_path = root / "manifests" / "files" / "noise-local-mics.jsonl"
    existing_jsonl = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                # Filter out previous entries of 20261008 if present
                if "20261008" not in item.get("path", ""):
                    existing_jsonl.append(item)
                    
    for r in new_records:
        rec_jsonl = {
            "attribution": "Clearcore local microphone noise dataset (3-mic OBS session)",
            "band_hz": None,
            "codec": "wav",
            "license": "LicenseRef-cctrain-generated",
            "num_samples": r.num_samples,
            "path": f"raw/noise-local-mics/{r.rel_path}",
            "role": "train",
            "schema": 2,
            "sha256": r.sha256,
            "source_id": "noise-local-mics",
            "speaker": None,
            "split": "noise",
            "sr": r.sample_rate,
        }
        existing_jsonl.append(rec_jsonl)
        
    jsonl_path.write_text("\n".join(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in existing_jsonl) + "\n", encoding="utf-8")
    print(f"Updated JSONL: {jsonl_path} (total lines: {len(existing_jsonl)})")

if __name__ == "__main__":
    main()
