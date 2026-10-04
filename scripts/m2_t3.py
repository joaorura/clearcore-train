"""M2 Tarefa 3. Uso: m2_t3.py {fetch,reference}. Grava em $CCTRAIN_RUNS_DIR/m2/.

fetch      baixa (ou confere) os avaliadores locais pinados em manifests/models/{dnsmos,whisper}.yaml e verifica SHA-256.
reference  números de referência no dev do DFNet3 upstream (misturas fala + ruído de dev): STOI/ESTOI, SI-SDR, LSD > 8 kHz,
           TSOS, DNSMOS P.835 e personalizado, e WER (Whisper small) em LibriSpeech dev-clean (en) e Common Voice dev (pt).
           Só dev; nenhuma medida do teste. CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cctrain.hashing import sha256_file  # noqa: E402
from cctrain.paths import Paths, repo_root  # noqa: E402

REPO = repo_root()
PATHS = Paths.from_env()
OUT = PATHS.runs / "m2"
SNRS = (0.0, 5.0, 10.0, 15.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(OUT / name)
    log(f"gravado {name}")


def cmd_fetch(a) -> None:
    import requests

    from cctrain.eval.models import ModelMismatch, load_model_manifest, model_dir, verify_model

    for mid in ("dnsmos", "whisper"):
        man = load_model_manifest(mid)
        base = model_dir(mid, PATHS)
        for rel, meta in man["files"].items():
            f = base / rel
            if f.is_file() and f.stat().st_size == meta["size"] and sha256_file(f) == meta["sha256"]:
                continue
            url = f"{man['base_url']}/{rel}"
            if mid == "dnsmos" and rel in ("LICENSE", "LICENSE-CODE"):
                url = f"{man['base_url'].rsplit('/', 1)[0]}/{rel}"  # na raiz do repositório
            if mid == "dnsmos" and rel == "pDNSMOS/sig_bak_ovr.onnx":
                url = f"{man['base_url']}/pDNSMOS/sig_bak_ovr.onnx"
            if mid == "whisper" and rel == "openai-whisper-LICENSE":
                url = "https://raw.githubusercontent.com/openai/whisper/main/LICENSE"  # não é do commit pinado: só evidência
            log(f"baixando {mid}/{rel}")
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.name + ".part")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(f)
        try:
            verify_model(mid, PATHS)
            log(f"{mid}: arquivos usados conferem com o manifest")
        except ModelMismatch as exc:
            raise SystemExit(str(exc)) from exc


# ------------------------------------------------------------------------------------------------ referência
def _transcripts(cv_validated: Path | None) -> dict[str, str]:
    """{caminho do manifest: transcrição}: LibriSpeech dev-clean (en) e, se informado, Common Voice (pt; coluna `sentence`
    do validated.tsv). As transcrições do CML (dev.csv) NÃO entram: não casam com o áudio segmentado (ex.: referência de 3
    palavras para 10 s de fala), o que daria WER > 100% sem relação com o modelo."""
    out: dict[str, str] = {}
    for tr in (PATHS.data / "raw" / "librispeech-dev-clean").rglob("*.trans.txt"):
        for line in tr.read_text().splitlines():
            uid, text = line.split(" ", 1)
            out[(tr.parent / f"{uid}.flac").relative_to(PATHS.data).as_posix()] = text
    if cv_validated:
        with Path(cv_validated).open(newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                out[f"raw/cv27-pt-full/clips/{row['path']}"] = row["sentence"]
    return out


def _pick(records, per_speaker: int, lo_s: float, hi_s: float):
    by_spk = defaultdict(list)
    for r in records:
        if lo_s <= r.num_samples / r.sr <= hi_s:
            by_spk[r.speaker_key].append(r)
    picked = []
    for spk in sorted(by_spk):
        rs = sorted(by_spk[spk], key=lambda r: hashlib.sha256(r.sha256.encode()).hexdigest())
        picked += rs[:per_speaker]
    return picked


def cmd_reference(a) -> None:
    from cctrain.data.mix import loop_to_length, scale_for_ratio
    from cctrain.data.resample import to_16k
    from cctrain.eval import metrics as M
    from cctrain.eval.dnsmos import DnsmosScorer
    from cctrain.eval.functional import speaker_bootstrap
    from cctrain.eval.models import verify_model
    from cctrain.eval.wer import WhisperOnnx, language_for, wer_per_utterance
    from cctrain.m1 import pools
    from cctrain.train.embed_bank import SR48, _load48, dfnet3_denoiser

    dnsmos_dir, whisper_dir = verify_model("dnsmos", PATHS), verify_model("whisper", PATHS)
    scorers = {"p835": DnsmosScorer.from_dir(dnsmos_dir, threads=a.threads),
               "personalized": DnsmosScorer.from_dir(dnsmos_dir, personalized=True, threads=a.threads)}
    asr = WhisperOnnx.from_dir(whisper_dir, threads=a.threads)
    denoise = dfnet3_denoiser("cpu")
    speech = pools.by_split(pools.speech_records(PATHS), "dev")
    _, noise_dev = pools.noise_pools()
    trans = _transcripts(a.cv_validated)
    recs = _pick(speech, a.per_speaker, 4.0, 12.0)
    if a.max_utts:
        recs = recs[: a.max_utts]
    log(f"{len(recs)} enunciados de dev de {len({r.speaker_key for r in recs})} falantes; ruído de dev: {len(noise_dev)}")
    rows = []
    t0 = time.time()
    for i, r in enumerate(recs):
        rng = np.random.default_rng([20261003, i])
        clean = _load48(PATHS.data, r.path)
        nz = loop_to_length(_load48(PATHS.data, noise_dev[int(rng.integers(len(noise_dev)))].path), clean.size, rng)
        snr = float(SNRS[int(rng.integers(len(SNRS)))])
        noisy = (clean.astype(np.float64) + scale_for_ratio(clean, nz, snr)).astype(np.float32)
        den = denoise(noisy)
        den_c0 = denoise(clean)  # C0: fala-alvo limpa e sozinha (condição do G3 C0)
        row = {"path": r.path, "speaker": r.speaker_key, "corpus": r.source_id, "snr_db": snr, "band_hz": r.band_hz}
        for name, est in (("noisy", noisy), ("dfnet3", den)):
            row[f"{name}.stoi"] = float(M.stoi(est, clean, SR48)[0])
            row[f"{name}.estoi"] = float(M.stoi(est, clean, SR48, extended=True)[0])
            row[f"{name}.si_sdr"] = float(M.si_sdr(est, clean, SR48)[0])
            row[f"{name}.tsos_pct"] = float(M.tsos(est, clean, SR48)[0])
            row[f"{name}.tsos_pct_int16"] = float(M.tsos(est, clean, SR48, M.TsosConfig(amplitude_scale=32768.0))[0])
            if (r.band_hz or 0) > 9000:
                row[f"{name}.lsd_above_8k"] = float(M.lsd_above_8k(est, clean, SR48)[0])
        row["c0.tsos_pct"] = float(M.tsos(den_c0, clean, SR48)[0])
        row["c0.tsos_pct_int16"] = float(M.tsos(den_c0, clean, SR48, M.TsosConfig(amplitude_scale=32768.0))[0])
        row["c0.stoi"] = float(M.stoi(den_c0, clean, SR48)[0])
        row["c0.si_sdr"] = float(M.si_sdr(den_c0, clean, SR48)[0])
        for name, x in (("clean", clean), ("noisy", noisy), ("dfnet3", den)):
            x16 = to_16k(x, SR48)
            for k, sc in scorers.items():
                s = sc.score(x16)
                row.update({f"{name}.dnsmos_{k}_{m}": s[m] for m in ("sig", "bak", "ovrl")})
            if r.path in trans and a.wer:
                row[f"{name}.hyp"] = asr.transcribe(x16, language=language_for(r.speaker_key))
        if r.path in trans and a.wer:
            row["ref_text"] = trans[r.path]
            for name in ("clean", "noisy", "dfnet3"):
                row[f"{name}.wer"] = float(wer_per_utterance([row[f"{name}.hyp"]], [trans[r.path]])[0])
        rows.append(row)
        if (i + 1) % 10 == 0:
            log(f"{i + 1}/{len(recs)} ({(time.time() - t0) / (i + 1):.1f} s/enunciado)")
    # agregados por corpus: média por falante com IC 95% (bootstrap por falante)
    keys = sorted({k for r in rows for k in r if "." in k and not k.endswith(".hyp")})
    agg: dict = {}
    for corpus in sorted({r["corpus"] for r in rows}) + ["ALL"]:
        sub = [r for r in rows if corpus in ("ALL", r["corpus"])]
        agg[corpus] = {"n_utterances": len(sub), "n_speakers": len({r["speaker"] for r in sub})}
        for k in keys:
            vals = [(r[k], r["speaker"]) for r in sub if k in r and r[k] == r[k]]
            if len(vals) >= 2:
                m, lo, hi = speaker_bootstrap([v for v, _ in vals], [s for _, s in vals], resamples=2000, seed=0)
                agg[corpus][k] = {"mean": m, "ci95": [lo, hi], "n": len(vals)}
    save("t3_reference.json", {"note": ("referência do DFNet3 upstream no dev (misturas fala + ruído de dev, SNR em "
                                        f"{list(SNRS)} dB, 48 kHz, CPU); STOI/DNSMOS a 16 kHz pelo caminho fixo; "
                                        "TSOS com p=0,3, gamma=0,1, STFT 20/10 ms, escala float e int16; WER só LibriSpeech "
                                        "dev-clean (en) e CV dev (pt); sem teste, sem calibrar nível"),
                               "per_speaker": a.per_speaker, "aggregate": agg, "rows": rows})
    for corpus, d in agg.items():
        line = [f"{corpus}: n={d['n_utterances']}"]
        for k in ("noisy.stoi", "dfnet3.stoi", "dfnet3.estoi", "dfnet3.si_sdr", "dfnet3.dnsmos_p835_ovrl",
                  "noisy.wer", "dfnet3.wer", "dfnet3.tsos_pct", "c0.tsos_pct", "c0.tsos_pct_int16"):
            if k in d:
                line.append(f"{k}={d[k]['mean']:.3f}")
        log(" ".join(line))


def main() -> int:
    import torch

    torch.set_num_threads(4)  # máquina compartilhada: sem sobreassinatura de threads
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    p = sub.add_parser("reference")
    p.add_argument("--per-speaker", type=int, default=2)
    p.add_argument("--max-utts", type=int, default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cv-validated", default=None, help="validated.tsv do CV 27 (sentence) para o WER em pt")
    p.add_argument("--no-wer", dest="wer", action="store_false")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "reference": cmd_reference}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
