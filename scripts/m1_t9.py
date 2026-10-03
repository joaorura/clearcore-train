"""M1 Tarefa 9: filtro do CV com WeSpeaker, splits do CV e listas de trials congeladas.

    python scripts/m1_t9.py cv       # embeddings (CPU, retomável), limiar de EER no dev, filtro, split do CV
    python scripts/m1_t9.py trials   # trials-v1: librispeech_test_clean, vctk_heldout (git), cv_pt_heldout (data/)

Nada com `client_id` vai para o git: o manifest filtrado, o split e a lista de trials do CV ficam em `data/`; só os
SHA-256 deles (e as contagens agregadas) são versionados. Nenhum EER dos conjuntos de teste é calculado aqui.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from cctrain.data.cv_filter import (
    FilterStats,
    calibrate_threshold,
    cap_per_client,
    cv_candidates,
    cv_filter,
    embed_records,
)
from cctrain.data.manifest import read_file_manifest, write_file_manifest
from cctrain.data.splits import apply_splits, assign_splits, load_split_config
from cctrain.eval.trials import make_trials, write_trials
from cctrain.eval.wespeaker import WeSpeakerEvaluator
from cctrain.hashing import sha256_file
from cctrain.paths import Paths, repo_root

REPO = repo_root()
N_TARGET = N_NONTARGET = 5000
TRIALS_SEED = 20261003
MIN_TEST_CLIENTS = 40
MAX_CLIPS_PER_CLIENT = 100
CV_SPLIT_SEED = 20261003


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_cv(paths: Paths, workers: int) -> int:
    torch.set_num_threads(1)
    cfg = load_split_config()
    rule = cfg.sources["cv27-pt"]
    mdir = paths.data / "manifests"
    records = read_file_manifest(mdir / "cv27-pt-full.jsonl")
    _log(f"CV completo: {len(records)} clipes")
    stage1 = assign_splits(records, cfg)  # só dev (a calibração vem antes do filtro)
    dev_keys = {k for k, v in stage1.items() if v == "dev"}
    cand = cv_candidates(records, rule["min_clips"])
    _log(f"candidatos (>= {rule['min_clips']} clipes): {len(cand)} clientes; dev de calibração: {len(dev_keys)}")

    ev = WeSpeakerEvaluator(threads=1)
    cache = paths.data / "embeddings" / "wespeaker-cv27-pt.npz"

    pool = [r for k in sorted(cand) for r in cap_per_client(cand[k], MAX_CLIPS_PER_CLIENT, CV_SPLIT_SEED)]
    _log(f"clipes a embutir (teto {MAX_CLIPS_PER_CLIENT}/cliente): {len(pool)}")
    t0 = time.time()
    embs = embed_records(pool, ev, paths.data, cache, workers=workers, log=_log)
    _log(f"embeddings prontos: {len(embs)}/{len(pool)} em {time.time() - t0:.0f} s")

    dev_pool = [r for r in pool if r.speaker_key in dev_keys]
    eer, thr = calibrate_threshold(dev_pool, embs, n_pairs=5000, seed=CV_SPLIT_SEED)
    _log(f"dev CV: EER {eer:.4f} no limiar de cosseno {thr:.4f}")

    stats = FilterStats()
    kept = cv_filter(records, ev, data_dir=paths.data, threshold=thr, cache=cache, min_clips=rule["min_clips"],
                     min_candidate_clips=rule["min_clips"], max_clips_per_client=MAX_CLIPS_PER_CLIENT,
                     seed=CV_SPLIT_SEED, workers=workers, stats=stats, log=_log)
    retained = {r.speaker_key for r in kept}
    assign = assign_splits(records, cfg, cv_retained=retained)
    final = apply_splits(kept, assign)
    by_split: dict[str, set[str]] = {}
    for r in final:
        by_split.setdefault(r.split, set()).add(r.speaker_key)
    counts = {s: len(v) for s, v in by_split.items()}
    n_test = counts.get("test", 0)
    _log(f"clientes retidos {len(retained)}; por split: {counts}")
    if n_test < MIN_TEST_CLIENTS:
        _log(f"ERRO: só {n_test} clientes de teste retidos (mínimo {MIN_TEST_CLIENTS})")
        return 1

    sha = write_file_manifest(final, mdir / "cv27-pt-filtered.jsonl")
    (mdir / "cv27-pt-splits-v1.json").write_text(
        json.dumps({k: assign[k] for k in sorted(assign)}, indent=0) + "\n", encoding="utf-8")
    sha_split = sha256_file(mdir / "cv27-pt-splits-v1.json")
    (REPO / "manifests" / "files" / "cv27-pt-filtered.sha256").write_text(
        f"{sha}  data/manifests/cv27-pt-filtered.jsonl\n{sha_split}  data/manifests/cv27-pt-splits-v1.json\n")
    summary = {
        "evaluator": "wespeaker-voxceleb-resnet34", "evaluator_sha256": ev.sha256,
        "max_clips_per_client": MAX_CLIPS_PER_CLIENT, "min_clips": rule["min_clips"],
        "dev_calibration": {"clients": len(dev_keys), "eer": eer, "cosine_threshold": thr},
        "stats": vars(stats), "clients_by_split": counts,
        "clips_by_split": {s: sum(1 for r in final if r.split == s) for s in by_split},
        "kept_manifest_sha256": sha, "splits_sha256": sha_split,
    }
    out = paths.runs / "m1" / "t9-cv-filter.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    _log(json.dumps(summary))
    return 0


def cmd_trials(paths: Paths) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from build_splits import non_cv_speech_records

    cfg = load_split_config()
    recs = non_cv_speech_records()
    assign = assign_splits(recs, cfg)
    final = apply_splits(recs, assign)
    out_git = REPO / "configs" / "eval" / "trials-v1"
    out_data = paths.data / "trials-v1"
    lists = {
        "librispeech_test_clean": ([r for r in final if r.source_id == "librispeech-test-clean" and r.split == "test"],
                                   out_git),
        "vctk_heldout": ([r for r in final if r.source_id == "vctk-0.92" and r.split == "test"], out_git),
    }
    cv_kept = paths.data / "manifests" / "cv27-pt-filtered.jsonl"
    if cv_kept.is_file():
        lists["cv_pt_heldout"] = ([r for r in read_file_manifest(cv_kept) if r.split == "test"], out_data)
    hashes, info = {}, {}
    for name, (rs, outdir) in lists.items():
        trials = make_trials(rs, N_TARGET, N_NONTARGET, TRIALS_SEED)
        h = write_trials(trials, outdir / f"{name}.jsonl")
        hashes[f"{name}.jsonl"] = h
        info[name] = {"speakers": len({r.speaker_key for r in rs}), "clips": len(rs), "trials": len(trials),
                      "target": sum(t.target for t in trials), "sha256": h}
    lock = REPO / "configs" / "eval" / "trials-v1.sha256"
    lock.write_text("".join(f"{h}  {n}\n" for n, h in sorted(hashes.items())))
    print(json.dumps(info, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "trials"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    paths = Paths.from_env()
    return cmd_cv(paths, a.workers) if a.cmd == "cv" else cmd_trials(paths)


if __name__ == "__main__":
    sys.exit(main())
