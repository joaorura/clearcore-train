"""Processa e enriquece gravações locais de ruído para o dataset de treino do Clearcore.

Extrai canais mono em WAV 48kHz (pcm_s16le), fatia em segmentos úteis de 5 a 10 segundos,
aplica filtros acústicos, variações de ganho e time-stretch/velocidade, e gera
manifests estruturados (JSON e CSV) sem caminhos absolutos hardcoded.
"""

from __future__ import annotations

import argparse
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

from dataclasses import asdict, dataclass

import numpy as np
import scipy.signal
import soundfile as sf

from cctrain.paths import Paths, repo_root


@dataclass
class NoiseSampleRecord:
    path: str
    rel_path: str
    filename: str
    source_file: str
    segment_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    num_samples: int
    sample_rate: int
    channels: int
    codec: str
    rms_dbfs: float
    peak_dbfs: float
    device: str
    augmentation: str
    tags: list[str]
    sha256: str


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_mono_audio(video_path: Path, sr: int = 48000) -> np.ndarray:
    """Extrai áudio mono 48kHz em ponto flutuante via ffmpeg sem compressão."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "2",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def is_valid_noise_segment(chunk: np.ndarray) -> bool:
    """Verifica se o segmento é válido e livre de saturação/impactos anômalos no microfone."""
    peak = float(np.max(np.abs(chunk)))
    rms = float(np.sqrt(np.mean(chunk**2))) + 1e-9
    crest = peak / rms

    # Descartar silêncio digital ou ruído imperceptível
    if rms < 1e-4:  # < -80 dBFS
        return False

    # Descartar batidas/impactos mecânicos com clipping severo
    if peak >= 0.98 or (peak >= 0.90 and crest > 20.0):
        return False

    return True


def detect_voice_activity(chunk: np.ndarray, sr: int = 48000) -> bool:
    """Detecta atividade de voz/babble secundária via faixa de formantes e modulação silábica."""
    sos = scipy.signal.butter(4, [250, 3500], btype="bandpass", fs=sr, output="sos")
    bp = scipy.signal.sosfilt(sos, chunk)

    frame_len = int(0.03 * sr)  # 30 ms
    hop_len = int(0.015 * sr)  # 15 ms
    n_frames = (len(bp) - frame_len) // hop_len
    if n_frames < 10:
        return False

    frames_e = np.array([np.sum(bp[i * hop_len : i * hop_len + frame_len] ** 2) for i in range(n_frames)])
    frames_db = 10 * np.log10(frames_e + 1e-12)

    dyn_range = float(np.percentile(frames_db, 95) - np.percentile(frames_db, 15))
    std_db = float(np.std(frames_db))

    return (dyn_range > 8.0 and std_db > 2.8) or (dyn_range > 12.0)


def detect_device_category(samples: np.ndarray, sr: int = 48000) -> tuple[str, str]:
    """Classifica o dispositivo com base no acoplamento mecânico de baixa frequência e nível RMS."""
    rms = float(np.sqrt(np.mean(samples**2))) + 1e-9
    rms_db = 20 * np.log10(rms)

    # Microfone embutido de notebook acoplado à ventoinha apresenta alto ganho (RMS ~ -18 dBFS)
    # e forte energia de sub-graves.
    if rms_db > -25.0:
        return "mic_builtin", "fan_noise"
    else:
        return "mic_external", "room_ambient"


def generate_augmented_variants(
    base_chunk: np.ndarray,
    sr: int = 48000,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Gera as 7 variantes acústicas para enriquecimento do dataset."""
    variants: dict[str, tuple[np.ndarray, list[str]]] = {}

    # 1. Base / Original (0 dB)
    variants["raw"] = (base_chunk.copy(), ["variant_raw"])

    # 2. Ganho -6 dB (atenuação leve)
    gain_m6 = base_chunk * float(10.0 ** (-6.0 / 20.0))
    variants["gain_m6db"] = (gain_m6, ["variant_gain_m6db", "attenuated"])

    # 3. Ganho +3 dB (aumento leve com proteção contra clipping)
    gain_p3 = base_chunk * float(10.0 ** (3.0 / 20.0))
    max_peak = float(np.max(np.abs(gain_p3)))
    if max_peak > 0.98:
        gain_p3 = gain_p3 * (0.98 / max_peak)
    variants["gain_p3db"] = (gain_p3, ["variant_gain_p3db", "boosted"])

    # 4. Passa-baixas leve (4500 Hz): atenua agudos, simulando maior distância ou absorção da sala
    sos_lp = scipy.signal.butter(2, 4500, btype="lowpass", fs=sr, output="sos")
    lp_chunk = scipy.signal.sosfilt(sos_lp, base_chunk).astype(np.float32)
    variants["lowpass"] = (lp_chunk, ["variant_lowpass", "distant_room_acoustic"])

    # 5. Passa-altas leve (180 Hz): remove rumble mecânico, simulando microfone sem efeito proximidade
    sos_hp = scipy.signal.butter(2, 180, btype="highpass", fs=sr, output="sos")
    hp_chunk = scipy.signal.sosfilt(sos_hp, base_chunk).astype(np.float32)
    variants["highpass"] = (hp_chunk, ["variant_highpass", "rumble_filtered"])

    # 6. Velocidade 0.95x (time-stretch mais lento, ~+5.2% de duração)
    # Resample racional polyphase de 20 / 19
    slow_chunk = scipy.signal.resample_poly(base_chunk, 20, 19).astype(np.float32)
    variants["speed_0.95x"] = (slow_chunk, ["variant_speed_slow", "time_stretched"])

    # 7. Velocidade 1.05x (time-stretch mais rápido, ~-4.8% de duração)
    # Resample racional polyphase de 20 / 21
    fast_chunk = scipy.signal.resample_poly(base_chunk, 20, 21).astype(np.float32)
    variants["speed_1.05x"] = (fast_chunk, ["variant_speed_fast", "time_compressed"])

    return variants


def process_video_file(
    video_path: Path,
    output_dir: Path,
    segment_sec: float = 6.0,
    sr: int = 48000,
    subtype: str = "PCM_16",
) -> list[NoiseSampleRecord]:
    """Processa um único arquivo de vídeo, extrai, fatia e gera variantes."""
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    samples = extract_mono_audio(video_path, sr=sr)
    device_tag, primary_noise_tag = detect_device_category(samples, sr=sr)

    seg_samples = int(segment_sec * sr)
    n_segs = len(samples) // seg_samples

    base_stem = video_path.stem.replace(" ", "_").replace("-", "")
    records: list[NoiseSampleRecord] = []

    for seg_idx in range(n_segs):
        t_start = seg_idx * segment_sec
        t_end = (seg_idx + 1) * segment_sec
        base_chunk = samples[seg_idx * seg_samples : (seg_idx + 1) * seg_samples]

        if not is_valid_noise_segment(base_chunk):
            continue

        has_voice = detect_voice_activity(base_chunk, sr=sr)
        base_tags = [device_tag, primary_noise_tag]
        if has_voice:
            base_tags.append("background_babble")

        # Gerar as variantes
        variants = generate_augmented_variants(base_chunk, sr=sr)

        for var_name, (var_audio, var_tags) in variants.items():
            filename = f"{device_tag}_{base_stem}_seg{seg_idx + 1:04d}_{var_name}.wav"
            out_path = wav_dir / filename

            # Normalização de segurança para formato PCM
            clipped = np.clip(var_audio, -1.0, 1.0)
            sf.write(str(out_path), clipped, sr, subtype=subtype)

            rms = float(np.sqrt(np.mean(clipped**2))) + 1e-9
            peak = float(np.max(np.abs(clipped)))
            rms_db = 20 * np.log10(rms)
            peak_db = 20 * np.log10(peak + 1e-9)

            rec = NoiseSampleRecord(
                path=str(out_path),
                rel_path=f"wavs/{filename}",
                filename=filename,
                source_file=video_path.name,
                segment_index=seg_idx + 1,
                start_sec=round(t_start, 3),
                end_sec=round(t_end, 3),
                duration_sec=round(len(clipped) / sr, 4),
                num_samples=len(clipped),
                sample_rate=sr,
                channels=1,
                codec="pcm_s16le" if subtype == "PCM_16" else "float32",
                rms_dbfs=round(rms_db, 2),
                peak_dbfs=round(peak_db, 2),
                device=device_tag,
                augmentation=var_name,
                tags=base_tags + var_tags,
                sha256=compute_sha256(out_path),
            )
            records.append(rec)

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Processa gravações locais de microfone em chunks de ruído para IA.")
    parser.add_argument(
        "--input-files",
        nargs="*",
        help="Caminhos para os arquivos de vídeo de entrada. Se omitido, busca no diretório de origem padrão.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Diretório contendo os arquivos .mp4 se nenhum arquivo individual for passado.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Diretório de destino. Padrão: data/raw/noise-local-mics do projeto cctrain.",
    )
    parser.add_argument(
        "--segment-sec",
        type=float,
        default=6.0,
        help="Duração dos segmentos de ruído em segundos (padrão: 6.0s).",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="Taxa de amostragem em Hz (padrão: 48000).",
    )
    parser.add_argument(
        "--format",
        choices=["pcm_s16le", "float32"],
        default="pcm_s16le",
        help="Formato WAV de saída (padrão: pcm_s16le).",
    )
    args = parser.parse_args()

    paths = Paths.from_env()
    output_dir = args.output_dir or (paths.data / "raw" / "noise-local-mics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identificação dos arquivos de entrada
    input_files: list[Path] = []
    if args.input_files:
        for f in args.input_files:
            input_files.append(Path(f).expanduser().resolve())
    elif args.input_dir:
        input_files = sorted(args.input_dir.glob("*.mp4"))
    else:
        # Busca padrão usando variáveis de ambiente ou diretório padrão
        source_dir = os.environ.get("CLEARCORE_NOISE_SOURCE_DIR")
        target_dir = Path(source_dir).expanduser().resolve() if source_dir else Path.home()
        # Arquivos gravados hoje conforme especificado
        candidates = [
            "2026-10-06 10-58-55.mp4",
            "2026-10-06 11-01-07.mp4",
            "2026-10-06 11-14-05.mp4",
            "2026-10-06 11-14-23.mp4",
            "2026-10-06 14-24-29.mp4",
            "2026-10-06 14-27-52.mp4",
            "2026-10-06 14-28-06.mp4",
        ]
        for name in candidates:
            p = target_dir / name
            if p.is_file():
                input_files.append(p)

    if not input_files:
        print("Erro: nenhum arquivo de entrada encontrado.", file=sys.stderr)
        return 1

    print(f"Iniciando processamento de {len(input_files)} arquivos:")
    for f in input_files:
        print(f" - {f.name}")
    print(f"Destino: {output_dir}")
    print(f"Segmento: {args.segment_sec}s | SR: {args.sr}Hz | Formato: {args.format}\n")

    subtype = "PCM_16" if args.format == "pcm_s16le" else "FLOAT"
    all_records: list[NoiseSampleRecord] = []

    for f in input_files:
        print(f"Processando {f.name}...")
        recs = process_video_file(
            f,
            output_dir=output_dir,
            segment_sec=args.segment_sec,
            sr=args.sr,
            subtype=subtype,
        )
        print(f" -> {len(recs)} amostras geradas (incluindo variantes).")
        all_records.extend(recs)

    # Ordenar por caminho relativo para reprodutibilidade estrita
    all_records.sort(key=lambda r: r.filename)

    # Estatísticas globais
    total_samples = len(all_records)
    total_duration_sec = sum(r.duration_sec for r in all_records)
    total_duration_min = total_duration_sec / 60.0

    tag_counts: dict[str, int] = {}
    aug_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    for r in all_records:
        device_counts[r.device] = device_counts.get(r.device, 0) + 1
        aug_counts[r.augmentation] = aug_counts.get(r.augmentation, 0) + 1
        for t in r.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # Gravar manifest.json
    manifest_json_path = output_dir / "manifest.json"
    manifest_data = {
        "dataset": "noise-local-mics",
        "version": 1,
        "sample_rate": args.sr,
        "channels": 1,
        "codec": args.format,
        "summary": {
            "total_samples": total_samples,
            "total_duration_sec": round(total_duration_sec, 2),
            "total_duration_minutes": round(total_duration_min, 2),
            "source_files_count": len(input_files),
            "device_counts": device_counts,
            "augmentation_counts": aug_counts,
            "tag_counts": tag_counts,
        },
        "samples": [asdict(r) for r in all_records],
    }
    manifest_json_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nManifest JSON gravado em: {manifest_json_path}")

    # Gravar noise_manifest.csv
    manifest_csv_path = output_dir / "noise_manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "path",
            "filename",
            "duration_sec",
            "num_samples",
            "sample_rate",
            "channels",
            "codec",
            "rms_dbfs",
            "peak_dbfs",
            "source_file",
            "segment_index",
            "device",
            "augmentation",
            "tags",
            "sha256",
        ])
        for r in all_records:
            writer.writerow([
                r.rel_path,
                r.filename,
                r.duration_sec,
                r.num_samples,
                r.sample_rate,
                r.channels,
                r.codec,
                r.rms_dbfs,
                r.peak_dbfs,
                r.source_file,
                r.segment_index,
                r.device,
                r.augmentation,
                ";".join(r.tags),
                r.sha256,
            ])
    print(f"Manifest CSV gravado em: {manifest_csv_path}")

    # Gravar manifest no padrão cctrain (JSONL em manifests/files/noise-local-mics.jsonl)
    cctrain_jsonl_path = repo_root() / "manifests" / "files" / "noise-local-mics.jsonl"
    jsonl_lines = []
    for r in all_records:
        # Caminho relativo a paths.data
        rel_to_data = f"raw/noise-local-mics/{r.rel_path}"
        jsonl_rec = {
            "attribution": "Clearcore local microphone noise dataset",
            "band_hz": None,
            "codec": "wav",
            "license": "LicenseRef-cctrain-generated",
            "num_samples": r.num_samples,
            "path": rel_to_data,
            "role": "train",
            "schema": 2,
            "sha256": r.sha256,
            "source_id": "noise-local-mics",
            "speaker": None,
            "split": "noise",
            "sr": r.sample_rate,
        }
        jsonl_lines.append(json.dumps(jsonl_rec, sort_keys=True, ensure_ascii=False))

    cctrain_jsonl_path.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    print(f"Manifest cctrain JSONL gravado em: {cctrain_jsonl_path}")

    # Gravar também source YAML em manifests/sources/noise-local-mics.yaml se não existir
    source_yaml_path = repo_root() / "manifests" / "sources" / "noise-local-mics.yaml"
    source_yaml_content = (
        'id: noise-local-mics\n'
        'name: "Gravações locais de microfone próprio com ruído real de sala, ventoinha e vozes de fundo"\n'
        'url: "local://data/raw/noise-local-mics"\n'
        'landing_page: "https://github.com/joaorura/clearcore"\n'
        'license: LicenseRef-cctrain-generated\n'
        'license_evidence: "Gravações autorais registradas para enriquecimento do treinamento acústico do Clearcore"\n'
        'role: train\n'
        'attribution: "Clearcore local microphone noise dataset"\n'
        'native_sr: 48000\n'
        'access: local_directory\n'
        'restrictions: [local_only]\n'
    )
    source_yaml_path.write_text(source_yaml_content, encoding="utf-8")
    print(f"Source YAML gravado em: {source_yaml_path}")

    # Validação e Resumo
    print("\n" + "=" * 60)
    print("RESUMO DO DATASET GERADO:")
    print(f" - Total de amostras geradas: {total_samples}")
    print(f" - Duração total: {total_duration_sec:.1f} s ({total_duration_min:.2f} minutos / {total_duration_min / 60.0:.2f} horas)")
    print(f" - Segmentos base originais: {total_samples // 7}")
    print(" - Distribuição por dispositivo:")
    for dev, cnt in sorted(device_counts.items()):
        print(f"     * {dev}: {cnt} arquivos")
    print(" - Distribuição por aumento de dados:")
    for aug, cnt in sorted(aug_counts.items()):
        print(f"     * {aug}: {cnt} arquivos")
    print(" - Principais tags:")
    for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"     * {tag}: {cnt} ocorrências")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
