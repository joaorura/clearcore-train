"""M2 Tarefa 4. Uso: m2_t4.py {render,freeze,calibrate,smoke}. Grava em $CCTRAIN_RUNS_DIR/m2/ e em $CCTRAIN_DATA_DIR/dev-m2/.

render     pré-renderiza o dev em duas metades disjuntas por fala (dev-select, dev-gate) em `data/dev-m2/` (CPU).
freeze     confere disjunção + SHA-256 de todos os arquivos e grava `configs/eval/m2-devsets-v1.freeze.json` (vai ao git).
calibrate  mede o DFNet3 upstream por condição no dev-select (TSOS na convenção fixa: float, escala 1) e compara com a faixa do
           baseline do paper na condição certa (TS3 = C0c); congela `tsos-config-v1.yaml` só se plausível, senão pede decisão do dono.
smoke      avaliação de fumaça no dev-select (estágio C do M1, caminhos offline e réplica do runtime). Nunca abre o dev-gate.
Só dev; o dev-gate não é lido por nenhum comando daqui (`open_half` o recusa sem `final_gate=True`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cctrain.hashing import sha256_file  # noqa: E402
from cctrain.paths import Paths, repo_root  # noqa: E402

REPO = repo_root()
PATHS = Paths.from_env()
OUT = PATHS.runs / "m2"
DEV_DIR = PATHS.data / "dev-m2"
CKPT_DIR = PATHS.data / "models" / "spkrec-ecapa-voxceleb"
STAGE_C_CKPT = PATHS.runs / "m1" / "stage-c-m1" / "checkpoints" / "ckpt-00020160.pt"
TSOS_YAML = REPO / "configs" / "eval" / "tsos-config-v1.yaml"
SR = 48_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(OUT / name)
    log(f"gravado {name}")


def cmd_render(a) -> None:
    from cctrain.m1 import pools
    from cctrain.m1.encoder import load_encoder
    from cctrain.m2 import devsets as ds
    from cctrain.m2.sex import load_speaker_sex

    dev = pools.by_split(pools.speech_records(PATHS), "dev")
    if a.limit_speakers:
        allk = sorted({r.speaker_key for r in dev})
        keep = set(allk[:: max(1, len(allk) // a.limit_speakers)])  # espaçados: pega todos os corpora
        dev = [r for r in dev if r.speaker_key in keep]
    _, noise_dev = pools.noise_pools()
    sex = load_speaker_sex(REPO / "manifests/speakers/sex-v1.jsonl",
                           PATHS.data / "manifests/speakers/sex-cv27-pt-v1.jsonl")
    enc = load_encoder(CKPT_DIR, device="cpu")
    out = Path(a.out) if a.out else DEV_DIR
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(f"{out} não está vazio (use --force para refazer; nada é apagado por este script)")
    log(f"dev: {len({r.speaker_key for r in dev})} falantes, {len(dev)} falas; ruído de dev {len(noise_dev)}; saída {out}")
    t0 = time.time()
    summary = ds.build_dev(dev, noise_dev, PATHS.data, out, sex, enc.embed, log=log)
    summary["seconds"] = time.time() - t0
    summary["encoder_sha256"] = enc.sha256
    summary["config_sha256"] = ds.load_devsets_config().sha256
    save("t4_render.json" if not a.limit_speakers else "t4_render_dry.json", summary)


def cmd_freeze(a) -> None:
    from cctrain.m2 import devsets as ds

    out = Path(a.dir) if a.dir else DEV_DIR
    sel, gate = ds.open_half(out, "dev-select", meta_only=True), ds.open_half(out, "dev-gate", meta_only=True)
    counts = ds.check_disjoint_and_enrollment_rule(sel, gate)
    files = {h: ds.verify_files(out / h) for h in ds.HALVES}
    freeze = ds.make_freeze(out)
    freeze["disjointness"] = counts
    freeze["files_verified"] = files
    sp = json.loads((out / "speakers.json").read_text())
    freeze["speakers"] = {k: sp[k] for k in ("total", "kept", "excluded", "kept_by_corpus", "kept_by_sex",
                                             "excluded_by_corpus", "excluded_by_sex", "excluded_by_corpus_sex")}
    freeze["speakers_sha256"] = sha256_file(out / "speakers.json")
    freeze["by_half"] = {h: {c: len(half.select(c)) for c in ds.CONDITIONS}
                         for h, half in (("dev-select", sel), ("dev-gate", gate))}
    path = Path(a.freeze) if a.freeze else ds.FREEZE_V1
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    log(f"congelado em {path}: {json.dumps(freeze['halves'])}")


def load_base_and_generator():
    import torch

    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.models.film import FilmGenerator, load_film_config
    from cctrain.train.checkpoint import load_checkpoint

    torch.set_num_threads(4)
    base, _ = load_upstream_dfnet3(device="cpu")
    gen = FilmGenerator(load_film_config())
    gen.load_state_dict(load_checkpoint(STAGE_C_CKPT).model)
    return base, gen.eval()


def dev_dir(a) -> Path:
    return Path(a.dev_dir) if getattr(a, "dev_dir", "") else DEV_DIR


def cmd_calibrate(a) -> None:
    from cctrain.eval.functional import enhance_offline
    from cctrain.eval.runtime_path import align_runtime
    from cctrain.m2 import devsets as ds
    from cctrain.m2.tsos_calib import (
        CONVENTION,
        GATING_CONDITION,
        OWNER_DECISION,
        PAPER_BASELINE_PCT,
        PLAUSIBILITY_FACTOR,
        measure_tsos,
        plausibility,
    )

    half = ds.open_half(dev_dir(a), "dev-select")  # SÓ o dev-select
    base, _ = load_base_and_generator()
    measured, detail = {}, {}
    for cond in ("C0c", "C0", "C3", "C4"):
        items = half.select(cond)
        if a.max_items:
            items = items[:: max(1, len(items) // a.max_items)]
        log(f"medição {cond}: {len(items)} itens do dev-select, DFNet3 upstream (caminho offline), convenção fixa")
        pairs, spk = [], []
        for k, it in enumerate(items):
            x = it.noisy()
            y, _ = enhance_offline(base, x)
            pairs.append((align_runtime(y, x.size, delay=480), it.clean()))
            spk.append(it["target_speaker"])
            if k % 40 == 0:
                log(f"  {cond} {k}/{len(items)}")
        measured[cond] = measure_tsos(pairs, spk, SR, CONVENTION)
        detail[cond] = {"n_items": len(items), "n_speakers": len(set(spk))}
    rep = plausibility(measured)
    for c, d in detail.items():
        rep["conditions"][c].update(d)
    res = {"split": "dev-select", "model": "dfnet3-upstream-v0.5.6 (offline, gating emulado)",
           "convention": {"amplitude_scale": CONVENTION.amplitude_scale, "p": CONVENTION.p, "gamma": CONVENTION.gamma,
                          "win_ms": CONVENTION.win_ms, "hop_ms": CONVENTION.hop_ms, "audio": "float [-1, 1]"},
           "paper_baseline_dccrn_non_personalized_pct_icr_vctk": PAPER_BASELINE_PCT, "plausibility_factor": PLAUSIBILITY_FACTOR,
           "plausibility": rep, "dev_items_sha256": sha256_file(dev_dir(a) / "dev-select" / "items.jsonl")}
    summary = ", ".join(f"{c} {v['tsos_pct']:.2f}% [{v['status']}]" for c, v in rep["conditions"].items())
    if a.dev_dir:  # ensaio em outro diretório: não congela nada
        save("t4_tsos_calibration_dry.json", res)
        log(f"ensaio: {summary}")
        return
    save("t4_tsos_calibration.json", res)
    log(f"TSOS do DFNet3 (convenção fixa): {summary}")
    if not rep["freeze"]:
        log(f"NÃO congelado: DFNet3 em {GATING_CONDITION} muito fora da faixa do paper. {OWNER_DECISION}")
        return
    TSOS_YAML.write_text(
        "# tsos_config v1: convenção FIXA (nada foi calibrado): áudio float [-1, 1], amplitude_scale 1, p e gamma do paper, STFT 20/10 ms\n"
        "# (a do DeepFilterNet). Congelado pelo SHA-256 em tsos-config-v1.sha256 ANTES de qualquer leitura do G3. O DFNet3 upstream foi\n"
        "# MEDIDO no dev-select (plausibilidade contra a faixa do baseline do paper na condição certa): docs/plans/m2-reports/t4.md.\n"
        "# Os níveis do G3 não mudam.\n"
        "version: 1\n"
        f"plausibility: {{split: dev-select, gating_condition: {GATING_CONDITION}, "
        f"dfnet3_tsos_pct: {json.dumps({c: round(v['tsos_pct'], 3) for c, v in rep['conditions'].items()})}, "
        f"dev_select_items_sha256: {res['dev_items_sha256']}}}\n"
        f"config:\n  p: {CONVENTION.p}\n  gamma: {CONVENTION.gamma}\n  win_ms: {CONVENTION.win_ms}\n  hop_ms: {CONVENTION.hop_ms}\n"
        f"  amplitude_scale: {CONVENTION.amplitude_scale}\n")
    (TSOS_YAML.with_suffix(".sha256")).write_text(f"{sha256_file(TSOS_YAML)}  {TSOS_YAML.name}\n")
    log(f"convenção congelada em {TSOS_YAML}")


def cmd_smoke(a) -> None:
    from cctrain.eval import metrics as M
    from cctrain.eval.dnsmos import DnsmosScorer, dnsmos
    from cctrain.eval.functional import enhance_offline, film_vector
    from cctrain.eval.metrics import TsosConfig
    from cctrain.eval.models import verify_model
    from cctrain.eval.runtime_path import align_runtime, runtime_forward
    from cctrain.eval.trr import mode_separation_db, trr
    from cctrain.m2 import devsets as ds
    from cctrain.m2.tsos_calib import load_tsos_config
    from cctrain.models.pdfnet3 import PDFNet3

    half = ds.open_half(dev_dir(a), "dev-select")
    base, gen = load_base_and_generator()
    model = PDFNet3(base, gen).eval()
    from cctrain.m2.tsos_calib import CONVENTION

    tsos_cfg = CONVENTION  # convenção fixa (float, escala 1); o yaml congelado, se existir, tem de coincidir
    if not a.dev_dir and TSOS_YAML.exists():
        assert load_tsos_config() == TsosConfig(**{k: getattr(CONVENTION, k) for k in ("p", "gamma", "win_ms", "hop_ms", "amplitude_scale")})
    n = a.n
    rng = np.random.default_rng(0)

    def pick(cond: str, k: int, **where):
        its = half.select(cond, **where)
        idx = rng.permutation(len(its))[:k]
        return [its[int(i)] for i in sorted(idx)]

    def offline(x, emb=None):
        y, _ = enhance_offline(base, x) if emb is None else enhance_offline(base, x, film_model=model, embedding=emb)
        return align_runtime(y, x.size, delay=480)

    def replica(x, emb=None):
        return runtime_forward(base, None if emb is None else film_vector(gen, emb), x)[0]

    # os mesmos itens nos dois caminhos (sorteio uma vez só)
    chosen = {"C0": pick("C0", n), "C3": pick("C3", n, sir_db=0.0), "C0c": pick("C0c", n),
              "C7same": pick("C7", n, sex_pair="same"), "C7different": pick("C7", n, sex_pair="different")}
    paths = {"offline": offline, "replica": replica}
    out: dict = {"checkpoint": STAGE_C_CKPT.name, "checkpoint_sha256": sha256_file(STAGE_C_CKPT),
                 "tsos_config": vars(tsos_cfg), "split": "dev-select", "n_per_condition": n, "paths": {}}
    timing = {k: [0.0, 0] for k in paths}
    for pname, fn in paths.items():
        res: dict = {}
        # C0 sem perfil: F (mask=0) contra DFNet3: no estágio C a base é congelada, então o delta tem de ser 0
        d_sisdr, d_stoi = [], []
        for it in chosen["C0"]:
            x, ref = it.noisy(), it.clean()
            t0 = time.time()
            yf = fn(x, None)
            timing[pname][0] += time.time() - t0
            timing[pname][1] += 1
            yb = offline(x) if pname == "offline" else replica(x, None)
            d_sisdr.append(float(M.si_sdr(yf, ref, SR)[0] - M.si_sdr(yb, ref, SR)[0]))
            d_stoi.append(float(M.stoi(yf, ref, SR)[0] - M.stoi(yb, ref, SR)[0]))
        res["C0_delta_vs_dfnet3"] = {"si_sdr_db_mean": float(np.mean(d_sisdr)), "stoi_mean": float(np.mean(d_stoi)),
                                     "n": len(d_sisdr)}
        # C3, SIR 0: ΔSI-SDR (estágio C com perfil − DFNet3)
        d = []
        for it in chosen["C3"]:
            x, ref = it.noisy(), it.clean()
            emb = half.embedding(it)
            t0 = time.time()
            yf = fn(x, emb)
            timing[pname][0] += time.time() - t0
            timing[pname][1] += 1
            yb = fn(x, None) if pname == "replica" else offline(x)
            d.append(float(M.si_sdr(yf, ref, SR)[0] - M.si_sdr(yb, ref, SR)[0]))
        res["C3_sir0_delta_si_sdr_db"] = {"mean": float(np.mean(d)), "n": len(d)}
        # C7: TRR(15 dB) e separação de modos, por par de sexo
        for pair in ("same", "different"):
            trrs, seps = [], []
            for it in chosen[f"C7{pair}"]:
                x = it.noisy()
                emb = half.embedding(it)
                yp, y0 = fn(x, emb), fn(x, None)
                trrs.append(float(trr(yp, x, SR)[0]))
                seps.append(float(mode_separation_db(yp, y0, x, SR)[0]))
            res[f"C7_{pair}_sex"] = {"trr15_mean": float(np.nanmean(trrs)), "mode_separation_db_mean": float(np.nanmean(seps)),
                                     "n": len(trrs)}
        # C0c: TSOS na config calibrada (DFNet3 sem perfil e estágio C com o próprio perfil)
        tb, tf = [], []
        for it in chosen["C0c"]:
            x, ref = it.noisy(), it.clean()
            tb.append(float(M.tsos(fn(x, None), ref, SR, tsos_cfg)[0]))
            tf.append(float(M.tsos(fn(x, half.embedding(it)), ref, SR, tsos_cfg)[0]))
        res["C0c_tsos_pct"] = {"dfnet3": float(np.nanmean(tb)), "stage_c_with_profile": float(np.nanmean(tf)), "n": len(tb)}
        out["paths"][pname] = res
        log(f"{pname}: {json.dumps(res)}")
    out["seconds_per_clip"] = {k: v[0] / max(v[1], 1) for k, v in timing.items()}
    try:  # DNSMOS em poucos itens: prova a cadeia (modelos locais com SHA-256 conferido)
        scorer = DnsmosScorer.from_dir(verify_model("dnsmos", PATHS), threads=2)
        items = pick("C3", 4, sir_db=0.0)
        out["dnsmos_p835_ovrl_c3_4items"] = {
            "noisy": float(np.mean([dnsmos(it.noisy(), None, SR, scorer=scorer)["ovrl"][0] for it in items])),
            "stage_c_replica": float(np.mean([dnsmos(replica(it.noisy(), half.embedding(it)), None, SR,
                                                     scorer=scorer)["ovrl"][0] for it in items]))}
    except Exception as e:  # noqa: BLE001
        out["dnsmos_p835_ovrl_c3_4items"] = f"não executado: {e}"
    save("t4_smoke.json" if not a.dev_dir else "t4_smoke_dry.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--limit-speakers", type=int, default=0)
    r.add_argument("--out", default="")
    r.add_argument("--force", action="store_true")
    f = sub.add_parser("freeze")
    f.add_argument("--dir", default="")
    f.add_argument("--freeze", default="")
    c = sub.add_parser("calibrate")
    c.add_argument("--dev-dir", default="", help="ensaio em outro diretório (não congela nada)")
    c.add_argument("--max-items", type=int, default=0, help="limita itens por condição (0 = todos)")
    s = sub.add_parser("smoke")
    s.add_argument("--n", type=int, default=12)
    s.add_argument("--dev-dir", default="")
    a = ap.parse_args()
    {"render": cmd_render, "freeze": cmd_freeze, "calibrate": cmd_calibrate, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
