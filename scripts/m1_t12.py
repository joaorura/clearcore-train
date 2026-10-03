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
from pathlib import Path

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


def final_checkpoint() -> Path:
    from cctrain.train.checkpoint import list_checkpoints

    cks = list_checkpoints(OUT / RUN_ID / "checkpoints")
    if not cks:
        raise SystemExit(f"sem checkpoint em {OUT / RUN_ID}")
    return cks[-1].path  # o último: sem seleção de checkpoint pelo dev (nenhum ajuste no dev)


def dev_pool():
    from cctrain.data.splits import speaker_sex

    dev = pools.by_split(pools.speech_records(PATHS), "dev")
    sex = {k: s for k in {r.speaker_key for r in dev} if (s := speaker_sex(k, PATHS.data))}
    return [r for r in dev if r.speaker_key in sex], sex


def cmd_evaluate(a) -> None:
    from cctrain.eval import functional as fn
    from cctrain.export.film import build_film_identity_asset
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.models.film import FilmGenerator, load_film_config
    from cctrain.models.pdfnet3 import PDFNet3
    from cctrain.paths import clearcore_root
    from cctrain.train.checkpoint import load_checkpoint

    ckpt = final_checkpoint()
    dev, sex = dev_pool()
    _, noise_dev = pools.noise_pools()
    enc = load_encoder(CKPT_DIR, device="cpu")
    log(f"checkpoint {ckpt.name}; dev: {len({r.speaker_key for r in dev})} falantes com sexo, ruído de dev {len(noise_dev)}")
    t0 = time.time()
    c3 = fn.build_dev_mixtures(dev, [], PATHS.data, sex, enc.embed, n_mixtures=a.n, condition="C3", seed=0)
    c4 = fn.build_dev_mixtures(dev, noise_dev, PATHS.data, sex, enc.embed, n_mixtures=a.n, condition="C4",
                               snr_db=10.0, seed=0)
    log(f"misturas prontas em {time.time() - t0:.0f} s")
    asset = build_film_identity_asset(clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin",
                                      OUT / "t11-film-asset").path
    runtime = fn.tract_runtime(asset, OUT / "t12-tract")
    base, _ = load_upstream_dfnet3(device="cpu")
    gen = FilmGenerator(load_film_config())
    gen.load_state_dict(load_checkpoint(ckpt).model)
    model = PDFNet3(base, gen.eval()).eval()
    per = []
    for m in c3[: a.deviation_n]:
        y, stages = fn.enhance_offline(base, m.noisy, film_model=model, embedding=m.embedding)
        per.append(fn.deviation_metrics(y, stages, runtime(m.noisy, fn.film_vector(gen, m.embedding))))
    deviation = fn.pool_deviation(per)
    path = fn.deviation_gate(deviation)
    log(f"desvio no checkpoint: {deviation} -> caminho {path}")
    out = {"checkpoint": ckpt.name, "deviation_dev_checkpoint": deviation, "deviation_path": path,
           "deviation_levels_sha256": fn.load_deviation_levels().sha256, "n_mixtures_requested": a.n,
           "dev_speakers": sorted({m.speaker for m in c3}).__len__(), "dev_sources": sorted({r.source_id for r in dev}),
           "dev_noise": "reserva de 5% do ruído deduplicado (nunca visto no treino)", "encoder_sha256": enc.sha256}
    for name, mixes in (("c3_gate", c3), ("c4_additional", c4)):
        t0 = time.time()
        r = fn.functional_gate(ckpt, mixes, path=path, runtime=runtime, base=base, workers=a.workers)
        out[f"functional_{name}"] = {**r.__dict__, "same_sex": True, "sir_db": 0.0, "stage": "C",
                                      "snr_db": None if name == "c3_gate" else 10.0}
        log(f"{name} ({path}): delta {r.delta_si_sdr_db:+.3f} dB IC95 [{r.ci_low:+.3f}, {r.ci_high:+.3f}] "
            f"({time.time() - t0:.0f} s)")
        if path == "replica":  # informativo: o mesmo conjunto pelo forward offline
            ro = fn.functional_gate(ckpt, mixes, path="offline", base=base)
            out[f"functional_{name}_offline_informative"] = ro.__dict__
            log(f"{name} (offline, informativo): delta {ro.delta_si_sdr_db:+.3f} IC95 [{ro.ci_low:+.3f}, {ro.ci_high:+.3f}]")
        save("t12_functional.json", out)


def real_clip(records, rng, n_samples: int) -> np.ndarray:
    from cctrain.data.audio import load_audio
    from cctrain.data.resample import to_16k

    parts, total = [], 0
    for k in rng.permutation(len(records)):
        x, sr = load_audio(PATHS.data / records[int(k)].path)
        parts.append(to_16k(x, sr))
        total += parts[-1].size
        if total >= n_samples:
            break
    return np.concatenate(parts)[:n_samples].astype(np.float32)


def cmd_parity(a) -> None:
    from cctrain.export.enrollment import (
        export_enrollment_onnx,
        scale_to_peak,
        scale_to_rms_dbfs,
        speechlike_16k,
    )
    from cctrain.hashing import sha256_file
    from cctrain.m1.parity import enrollment_parity, summarize_parity
    from cctrain.models.film import FilmGenerator, load_film_config
    from cctrain.models.sb_port import build_enrollment, load_speechbrain_state, port_ecapa
    from cctrain.train.checkpoint import load_checkpoint

    ckpt = final_checkpoint()
    gen = FilmGenerator(load_film_config())
    gen.load_state_dict(load_checkpoint(ckpt).model)
    model = build_enrollment(port_ecapa(load_speechbrain_state(CKPT_DIR)), gen.eval()).eval()
    outdir = OUT / "t12-enrollment-final"
    onnx_path = export_enrollment_onnx(model, outdir / "enrollment.onnx")
    dev, _ = dev_pool()
    pool = [r for r in dev if r.source_id == "librispeech-dev-clean"]
    rng = np.random.default_rng(0)
    cases: dict[str, np.ndarray] = {}
    durations = (2, 6, 8, 10, 12)
    for s in durations:
        real = real_clip(pool, rng, 16_000 * s)
        cases[f"{s}s_real_rms-20"] = scale_to_rms_dbfs(real, -20.0)
        cases[f"{s}s_synthetic_rms-20"] = scale_to_rms_dbfs(speechlike_16k(16_000 * s, s), -20.0)
    for s in (2, 12):
        real = real_clip(pool, rng, 16_000 * s)
        cases[f"{s}s_real_rms-40"] = scale_to_rms_dbfs(real, -40.0)
        cases[f"{s}s_real_peak0.99"] = scale_to_peak(real, 0.99)
    required = tuple(cases)
    n = 96_000
    imp = np.zeros(n, np.float32)
    imp[n // 2] = 0.99
    edges = {"edge_silence": np.zeros(n, np.float32), "edge_dc_0.1": np.full(n, 0.1, np.float32),
             "edge_full_scale_square": np.where((np.arange(n) // 50) % 2 == 0, 1.0, -1.0).astype(np.float32),
             "edge_white_noise_peak0.99": scale_to_peak(np.random.default_rng(0).standard_normal(n).astype("f4"), 0.99),
             "edge_noise_-90dBFS": scale_to_rms_dbfs(np.random.default_rng(1).standard_normal(n).astype("f4"), -90.0),
             "edge_single_impulse": imp}
    rows = enrollment_parity(model, onnx_path, {**cases, **edges}, outdir / "cases")
    summary = summarize_parity(rows, durations_s=durations, required=required)
    summary["edge_cases_max_abs_diff"] = max(r["tract_max_abs_diff"] for r in rows if r["case"].startswith("edge_"))
    save("t12_tract.json", {"checkpoint": ckpt.name, "enrollment_onnx_sha256": sha256_file(onnx_path),
                            "enrollment_onnx_bytes": onnx_path.stat().st_size, "summary": summary, "cases": rows})
    log(json.dumps(summary))


def cmd_ab1(a) -> None:
    from cctrain.m1.ab1 import ab1_indicator
    from cctrain.train.embed_bank import dfnet3_denoiser

    dev, _ = dev_pool()
    utt: dict[str, list[str]] = {}
    for r in dev:
        utt.setdefault(r.speaker_key, []).append(r.path)
    _, noise_dev = pools.noise_pools()
    enc = load_encoder(CKPT_DIR, device="cpu")
    res = ab1_indicator(enc.embed, {k: sorted(v) for k, v in sorted(utt.items())}, PATHS.data,
                        [r.path for r in noise_dev], denoise=dfnet3_denoiser("cpu"), seed=0)
    res["note"] = ("indicador barato (cosseno do embedding cru/denoisado com o limpo), DFNet3 upstream sem perfil, "
                   "ruído de dev; informativo, a AB1 completa é do M4")
    save("t12_ab1.json", res)
    for k, v in res["by_snr_db"].items():
        log(f"SNR {k} dB: cos cru {v['cos_raw_clean']['mean']:.3f} denoisado {v['cos_denoised_clean']['mean']:.3f} "
            f"delta {v['delta_denoised_minus_raw']['mean']:+.3f} IC {v['delta_denoised_minus_raw']['ci95']}")


def cmd_report(a) -> None:
    import shutil

    from cctrain.m1.report import M1Inputs, m1_verdict, render_checks_table

    j = lambda n: json.loads((OUT / n).read_text())  # noqa: E731
    eer, fnr, tr = j("t12_eer.json"), j("t12_functional.json"), j("t12_tract.json")
    probe = j("stage-c-probe.json")
    summ = j(f"{RUN_ID}/summary.json")
    inp = M1Inputs(
        eer={k: eer[k]["eer"] for k in TRIALS}, tract=tr["summary"], functional_c3=fnr["functional_c3_gate"],
        base_sha256=(summ["base_sha256_before"], summ["base_sha256_after"]), probe=probe,
        disk_free_gb=shutil.disk_usage(PATHS.data).free / 1e9, deviation={**fnr["deviation_dev_checkpoint"],
                                                                         "path": fnr["deviation_path"]})
    v = m1_verdict(inp)
    md = render_checks_table(v)
    save("t12_verdict.json", {"go": v.go, "blocking_failures": v.blocking_failures, "gates_sha256": v.gates_sha256,
                              "budget_sha256": v.budget_sha256, "deviation_sha256": v.deviation_sha256,
                              "checks": [c.__dict__ for c in v.checks]})
    (OUT / "t12_verdict.md").write_text(md + "\n")
    print(md)
    log(f"VEREDITO: {'GO' if v.go else 'NO-GO'}; falhas bloqueantes: {list(v.blocking_failures)}")


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
    p = sub.add_parser("evaluate")
    p.add_argument("--n", type=int, default=240)
    p.add_argument("--deviation-n", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    for name in ("parity", "ab1", "report"):
        sub.add_parser(name)
    a = ap.parse_args()
    {"eer": cmd_eer, "bank": cmd_bank, "stage-c": cmd_stage_c, "evaluate": cmd_evaluate, "parity": cmd_parity,
     "ab1": cmd_ab1, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
