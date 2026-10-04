"""M2 Tarefa 2: banco de embeddings v2 (proveniência por variante). CPU apenas (nenhuma GPU), retomável.

Uso: m2_t2.py bank [--workers N] [--torch-threads N]   (rode de novo o mesmo comando para retomar do cache)
     m2_t2.py stats                                      (estatísticas do banco salvo, só contagens)
Grava `data/embeddings/bank-m2-sbecapa-v2.npz` (+ `.meta.json`) e `$CCTRAIN_RUNS_DIR/m2/bank-v2.stats.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # a GPU é de outra tarefa
os.environ.setdefault("OMP_NUM_THREADS", "1")

from cctrain.m1 import pools  # noqa: E402
from cctrain.m1.encoder import load_encoder  # noqa: E402
from cctrain.paths import Paths  # noqa: E402
from cctrain.train.checkpoint import git_commit  # noqa: E402
from cctrain.train.embed_bank import ArrayEmbeddingBank, build_bank, dfnet3_denoiser  # noqa: E402

PATHS = Paths.from_env()
OUT = PATHS.runs / "m2"
BANK_PATH = PATHS.data / "embeddings" / "bank-m2-sbecapa-v2.npz"
BANK_CACHE = PATHS.data / "embeddings" / "bank-m2-sbecapa-v2.cache"
CKPT_DIR = PATHS.data / "models" / "spkrec-ecapa-voxceleb"
SEED = 20261003
VARIANTS = 20
MAX_TRIES = 16


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_stats(bank: ArrayEmbeddingBank, seconds: float | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = bank.meta
    by_corpus: dict[str, dict] = {}
    for k in bank.speakers():
        c = by_corpus.setdefault(k.split(":")[0], {"speakers": 0, "variants": 0, "lt20": 0, "lt10_raw": 0})
        info = bank.variants(k)
        c["speakers"] += 1
        c["variants"] += len(info)
        c["lt20"] += len(info) < 20
        c["lt10_raw"] += int((~info.denoised).sum()) < 10
    out = {"n_speakers": meta["n_speakers"], "n_embeddings": meta["n_embeddings"], "sha256": meta["sha256"],
           "skipped_speakers": len(meta["skipped_speakers"]), "short_speakers": len(meta["short_speakers"]),
           "stats": bank.stats(), "by_corpus": by_corpus, "generation_seconds": seconds,
           "trim_silence": meta["trim_silence"], "max_tries": meta["max_tries"]}
    (OUT / "bank-v2.stats.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    log(json.dumps(out, sort_keys=True))


def cmd_bank(a) -> None:
    import torch

    torch.set_num_threads(a.torch_threads)
    speech = pools.speech_records(PATHS)
    train = pools.by_split(speech, "train")
    noise_train, _ = pools.noise_pools()
    rirs = pools.rir_records()
    log(f"falantes de treino: {len({r.speaker_key for r in train})}; ruído {len(noise_train)}; RIR {len(rirs)}; "
        f"workers {a.workers}")
    enc = load_encoder(CKPT_DIR, device="cpu")
    t0 = time.time()
    bank = build_bank(enc.embed, train, data_root=PATHS.data, variants=VARIANTS, seed=SEED, noise=noise_train,
                      rirs=rirs, denoise_factory=dfnet3_denoiser, workers=a.workers, cache_dir=BANK_CACHE,
                      encoder_tag=enc.sha256, denoise_tag="dfnet3-upstream-v0.5.6-cpu", max_tries=MAX_TRIES,
                      trim=False, log=log)
    secs = time.time() - t0
    meta = bank.save(BANK_PATH, {"commit": git_commit(), "split": "train", "encoder": "spkrec-ecapa-voxceleb",
                                 "generation_seconds_last_run": round(secs, 1)})
    log(f"banco salvo: {BANK_PATH.name} sha256 {meta['sha256'][:12]}")
    save_stats(bank, secs)


def cmd_stats(_a) -> None:
    save_stats(ArrayEmbeddingBank.load(BANK_PATH), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bank")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=3)
    sub.add_parser("stats")
    a = ap.parse_args()
    {"bank": cmd_bank, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
