"""M1 Tarefa 12. Uso: m1_t12.py {eer,bank,stage-c,evaluate,parity,ab1,report}. Grava em $CCTRAIN_RUNS_DIR/m1/.

eer       EER do encoder portado nas 3 trials congeladas (hash conferido), na GPU sob o gpu_lock; WeSpeaker como referência.
bank      banco de embeddings (20 variantes por falante de treino), CPU em processos + encoder na GPU sob o gpu_lock.
stage-c   rodada real do estágio C (~1 h, retomável: rode de novo o mesmo comando para retomar).
evaluate  gate funcional (C3) + C4 adicional + desvio offline x runtime, no dev, com o último checkpoint.
parity    exporta o enrollment.onnx final (encoder portado + gerador do estágio C) e confere o tract.
ab1       indicador barato da AB1 (cosseno do embedding cru/denoisado com o limpo).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from cctrain.eval.eer import evaluate_trials_file
from cctrain.gpu import gpu_lock
from cctrain.m1 import pools
from cctrain.m1.encoder import clip_embedder, clips_embedder, load_encoder
from cctrain.paths import Paths, repo_root
from cctrain.train.checkpoint import git_commit

REPO = repo_root()
PATHS = Paths.from_env()
OUT = PATHS.runs / "m1"
CKPT_DIR = PATHS.data / "models" / "spkrec-ecapa-voxceleb"
TRIALS = {"librispeech_test_clean": REPO / "configs/eval/trials-v1/librispeech_test_clean.jsonl",
          "vctk_heldout": REPO / "configs/eval/trials-v1/vctk_heldout.jsonl",
          "cv_pt_heldout": PATHS.data / "trials-v1/cv_pt_heldout.jsonl"}
TRIALS_LOCK = REPO / "configs/eval/trials-v1.sha256"
BANK_PATH = PATHS.data / "embeddings" / "bank-m1-sbecapa-v1.npz"
BANK_CACHE = PATHS.data / "embeddings" / "bank-m1-sbecapa-v1.cache"
RUN_ID = "stage-c-m1"
BANK_SEED = 20261003


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(OUT / name)
    log(f"gravado {name}")


def cmd_eer(a) -> None:
    from cctrain.data.audio import load_audio
    from cctrain.data.resample import to_16k
    from cctrain.eval.wespeaker import WeSpeakerEvaluator, kaldi_fbank  # noqa: F401

    result: dict = {"commit": git_commit(), "trials_lock": str(TRIALS_LOCK.relative_to(REPO)),
                    "caveat_cv": "cv_pt_heldout foi filtrado pelo próprio WeSpeaker (spec §4.6): o EER do WeSpeaker "
                                 "nessa lista é otimista; o do encoder portado não tem esse viés"}
    with gpu_lock():
        enc = load_encoder(CKPT_DIR, device="cuda")
        result["encoder_sha256"] = enc.sha256
        many = clips_embedder(enc, PATHS.data)
        for name, path in TRIALS.items():
            t0 = time.time()
            r = evaluate_trials_file(path, TRIALS_LOCK, clip_embedder(enc, PATHS.data), embed_many=many)
            r["seconds"] = time.time() - t0
            result[name] = r
            log(f"{name}: EER {r['eer']:.4f} IC95 {r['ci95']} ({r['n_clips']} clipes, {r['seconds']:.0f} s)")
    if not a.no_wespeaker:
        ws = WeSpeakerEvaluator(threads=2)

        def ws_embed(rel: str) -> np.ndarray:
            x, sr = load_audio(PATHS.data / rel)
            return ws.embed(to_16k(x, sr))

        for name, path in TRIALS.items():
            r = evaluate_trials_file(path, TRIALS_LOCK, ws_embed, ci_resamples=0)
            result[name]["wespeaker_reference_eer"] = r["eer"]
            log(f"{name}: WeSpeaker (referência) EER {r['eer']:.4f}")
    save("t12_eer.json", result)


def cmd_bank(a) -> None:
    from cctrain.train.embed_bank import dfnet3_denoiser

    speech = pools.speech_records(PATHS)
    train = pools.by_split(speech, "train")
    noise_train, _ = pools.noise_pools()
    rirs = pools.rir_records()
    log(f"falantes de treino: {len({r.speaker_key for r in train})}; ruído de treino {len(noise_train)}; RIR {len(rirs)}")
    with gpu_lock():
        enc = load_encoder(CKPT_DIR, device="cuda")
        from cctrain.train.embed_bank import build_bank

        bank = build_bank(enc.embed, train, data_root=PATHS.data, variants=a.variants, seed=BANK_SEED,
                          noise=noise_train, rirs=rirs, denoise_factory=dfnet3_denoiser, workers=a.workers,
                          cache_dir=BANK_CACHE, encoder_tag=enc.sha256, denoise_tag="dfnet3-upstream-v0.5.6-cpu",
                          log=log)
    meta = bank.save(BANK_PATH, {"commit": git_commit(), "split": "train", "encoder": "spkrec-ecapa-voxceleb"})
    log(f"banco salvo: {BANK_PATH.name} sha256 {meta['sha256'][:12]} {meta['n_speakers']} falantes, "
        f"{meta['n_embeddings']} embeddings, pulados {meta['skipped_speakers']}")


def stage_c_inputs():
    from cctrain.data.splits import speaker_sex
    from cctrain.train.embed_bank import ArrayEmbeddingBank
    from cctrain.train.stage_c import StageCData

    speech = pools.by_split(pools.speech_records(PATHS), "train")
    noise_train, _ = pools.noise_pools()
    sex = {k: s for k in {r.speaker_key for r in speech} if (s := speaker_sex(k, PATHS.data))}
    return ArrayEmbeddingBank.load(BANK_PATH), StageCData(speech, noise_train, pools.rir_records(), sex), sex


def cmd_stage_c(a) -> None:
    from cctrain.train.stage_c import StageCConfig, run_stage_c

    bank, data, sex = stage_c_inputs()
    usable = [k for k in bank.speakers() if k in sex]
    log(f"banco: {len(bank.speakers())} falantes; com sexo conhecido (usáveis no mesmo sexo): {len(usable)}")
    cfg = StageCConfig(run_id=RUN_ID, batch_size=16, segment_s=3.0, steps_per_epoch=500, num_workers=a.workers,
                       prefetch_factor=2, checkpoint_every_min=a.checkpoint_every_min, probe_every=20,
                       max_steps=a.max_steps)
    ckpt = run_stage_c(cfg, bank, max_minutes=a.minutes, data=data)
    log(f"checkpoint final: {ckpt}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eer")
    p.add_argument("--no-wespeaker", action="store_true")
    p = sub.add_parser("bank")
    p.add_argument("--variants", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("stage-c")
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint-every-min", type=float, default=10.0)
    for name in ("evaluate", "parity", "ab1", "report"):
        sub.add_parser(name)
    a = ap.parse_args()
    {"eer": cmd_eer, "bank": cmd_bank, "stage-c": cmd_stage_c}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
