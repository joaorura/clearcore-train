#!/usr/bin/env python3
"""Avaliação do pDFNet3 Pro v4 (plano `docs/plan-pro-v4.md` §8). Subcomandos:

  prep     monta banco de falantes de dev, trilha do usuário real (take4) e as receitas congeladas (261 + extras + usuário)
  run      roda um checkpoint nas receitas (gating `current` ou `new`), em N processos; retoma de onde parou
  report   agrega por célula, confere os 11 critérios e escreve o relatório markdown
  compare  tabela lado a lado de várias execuções (critérios principais)
  eval     run + report num comando só (é o que se usa no ckpt final do v4)

Exemplo (v4 na GPU, ~1 h): `scripts/eval_pro_v4_final.sh runs/m4/pro-v4/checkpoints/<ckpt-final>.pt`.
Vive no ambiente do `uv` do repositório principal (PYTHONPATH apontando para o `src` deste worktree).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cctrain.eval import tse_matrix as tm  # noqa: E402
from cctrain.paths import Paths, repo_root  # noqa: E402

PATHS = Paths.from_env()
SET_DIR_DEFAULT = PATHS.data / "eval-v4"
CKPT_ENCODER = PATHS.data / "models" / "spkrec-ecapa-voxceleb"
USER_DIR = PATHS.data / "raw" / "speech-joao-messias"
MAIN_ROOT = Path(os.environ.get("CCTRAIN_ROOT", Path.home() / "orca/projects/clearcore-train"))
FILES_DIR = MAIN_ROOT / "manifests" / "files"  # manifests de arquivos não vão ao git: vêm do checkout principal


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- prep ------------------------------------------------------------------------------------------------------------


def _sex_map():
    from cctrain.m2.sex import load_speaker_sex

    a = MAIN_ROOT / "manifests/speakers/sex-v1.jsonl"
    b = PATHS.data / "manifests/speakers/sex-cv27-pt-v1.jsonl"
    return load_speaker_sex(a, b)


def build_bank_dev() -> tm.Bank:
    """Falantes do dev (as 95 do dev-select). Enrollment = o do dev-select (disjunto das falas das misturas); falas de
    mistura = pool de misturas da metade `dev-select` (nunca o `dev-gate`). Ruído = reserva de dev (5% por hash)."""
    from cctrain.m1 import pools
    from cctrain.m2 import devsets as ds

    cfg = ds.load_devsets_config()
    recs = pools.by_split(pools.speech_records(PATHS, FILES_DIR), "dev")
    parts = ds.partition_utterances(recs, cfg)
    half = PATHS.data / "dev-m2" / "dev-select"
    rows = [json.loads(line) for line in (half / "enroll.jsonl").read_text().splitlines() if line.strip()]
    emb_all = np.load(half / "enroll_emb.npy")
    sex = _sex_map()
    keys, utts, corpus, embs, sx = [], {}, {}, [], {}
    for row in rows:
        spk = row["speaker"]
        used = set(row["utterances"])
        pool = [r for r in parts[spk]["dev-select"].pool if r.path not in used
                and cfg.target_min_s <= r.num_samples / r.sr <= cfg.target_max_s]
        if not pool:
            continue
        keys.append(spk)
        utts[spk] = [{"path": r.path, "n": r.num_samples, "sr": r.sr} for r in sorted(pool, key=lambda r: r.path)]
        corpus[spk] = spk.split(":", 1)[0]
        sx[spk] = sex.get(spk)
        embs.append(emb_all[row["embedding_row"]])
    _, noise_dev = pools.noise_pools(FILES_DIR)
    noise = sorted(r.path for r in noise_dev if r.num_samples >= 2 * tm.SR)
    return tm.Bank(keys, sx, corpus, utts, np.stack(embs).astype(np.float32), noise)


def build_user(out_dir: Path) -> dict:
    """Trilha do usuário real: take4 (reservado para teste) costurada a 48 kHz; perfis de takes 1-2 (take3 é o gêmeo
    simultâneo do take4: mesma fala em outro canal, então fica fora do perfil)."""
    import soundfile as sf

    from cctrain.data.enroll import EnrollmentRejected, check_enrollment_audio
    from cctrain.data.resample import to_16k
    from cctrain.m1.encoder import load_encoder
    from cctrain.m2.devsets import trim_edge_silence

    hop = int(1.5 * tm.SR)
    sl = [sf.read(str(USER_DIR / f"joao_messias_take4_{i:04d}.wav"), dtype="float32")[0] for i in range(24)]
    stream = np.concatenate([s[:hop] for s in sl[:-1]] + [sl[-1]]).astype(np.float32)
    path = out_dir / "user_take4_stream_48k.wav"
    sf.write(str(path), stream, tm.SR, subtype="FLOAT")
    win, t = [], 0
    while t + tm.SEG <= len(stream):
        if tm.activity(stream[t : t + tm.SEG]).mean() >= 0.5:
            win.append(t)
        t += int(2.5 * tm.SR)
    enc = load_encoder(CKPT_ENCODER, device="cpu")
    embs, used = [], []
    for e in range(5):
        rng = tm.rng_for("user-enroll", tm.SEED, e)
        for _ in range(40):
            want = float(rng.uniform(6.5, 10.0))
            parts, chosen, total = [], [], 0.0
            cand = [(tk, i) for tk in (1, 2) for i in range(70)]
            for k in rng.permutation(len(cand)):
                tk, i = cand[int(k)]
                if any(tk == c[0] and abs(i - c[1]) < 3 for c in chosen):
                    continue
                f = USER_DIR / f"joao_messias_take{tk}_{i:04d}.wav"
                if not f.is_file():
                    continue
                x, sr = sf.read(str(f), dtype="float32")
                x16 = trim_edge_silence(to_16k(x, sr))
                if parts:
                    parts.append(np.zeros(int(rng.uniform(0.15, 0.40) * 16000), np.float32))
                parts.append(x16)
                chosen.append((tk, i))
                total = sum(len(p) for p in parts) / 16000
                if total >= want:
                    break
            audio = np.concatenate(parts)[: int(want * 16000)].astype(np.float32)
            try:
                check_enrollment_audio(audio)
            except EnrollmentRejected:
                continue
            embs.append(enc.embed(audio))
            used.append([f"take{tk}_{i:04d}" for tk, i in chosen])
            break
    np.save(out_dir / "user_emb.npy", np.stack(embs).astype(np.float32))
    return {"key": "joao_messias", "windows": win, "embs_n": len(embs), "enroll_slices": used,
            "stream_seconds": len(stream) / tm.SR, "stream": str(path)}


def cmd_prep(a) -> None:
    out = Path(a.set)
    out.mkdir(parents=True, exist_ok=True)
    tm.set_roots(PATHS.data, out / "user_take4_stream_48k.wav")
    log("banco de dev...")
    bank = build_bank_dev()
    (out / "bank.json").write_text(json.dumps(bank.to_json()))
    np.save(out / "bank_emb.npy", bank.emb)
    log(f"banco: {len(bank.keys)} falantes, {sum(len(v) for v in bank.utts.values())} falas, {len(bank.noise)} ruídos de dev")
    log("usuário real (take4)...")
    user = build_user(out)
    (out / "user.json").write_text(json.dumps(user, indent=1))
    log(f"usuário: {len(user['windows'])} janelas de 5 s em {user['stream_seconds']:.1f} s; {user['embs_n']} perfis")
    recs, redraws = [], 0
    t0 = time.time()
    for cell in tm.all_cells():
        for idx in range(tm.n_items(cell)):
            for attempt in range(6):
                rec = tm.make_recipe(bank, cell, idx, user=user if cell.axis == "user" else None, attempt=attempt)
                if cell.axis == "temporal":
                    break
                r = tm.render(rec)
                if r.tgt_act.mean() >= 0.30:
                    break
                redraws += 1
            recs.append(rec)
    (out / "recipes.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n")
    cells = tm.all_cells()
    by_axis = {ax: sum(1 for c in cells if c.axis == ax) for ax in ("matrix", "similar", "temporal", "user")}
    man = {"seed": tm.SEED, "recipes": len(recs), "cells": by_axis, "redraws": redraws,
           "recipes_sha256": tm.recipes_digest(out / "recipes.jsonl"), "bank_speakers": len(bank.keys),
           "seconds": time.time() - t0, "user": {k: user[k] for k in ("windows", "enroll_slices", "stream_seconds")},
           "noise_pool": "reserva de dev (5% dos conteúdos por hash, ver m1/pools.noise_pools)"}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"{len(recs)} receitas, {man['cells']} células, {redraws} redraws, sha256 {man['recipes_sha256'][:12]}")


# --- run -------------------------------------------------------------------------------------------------------------

_W: dict = {}


def _init_worker(set_dir: str, ckpt: str, arch: str, gating: str, device: str, threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    from cctrain.eval import tse_eval as te

    sd = Path(set_dir)
    tm.set_roots(PATHS.data, sd / "user_take4_stream_48k.wav")
    bank = tm.Bank.from_json(json.loads((sd / "bank.json").read_text()), np.load(sd / "bank_emb.npy"))
    _W["bank"] = bank
    _W["user_emb"] = np.load(sd / "user_emb.npy")
    _W["runner"] = te.Runner(Path(ckpt), arch, gating, device)
    _W["te"] = te


def _emb_of(rec: dict) -> np.ndarray:
    p = rec["profile"]
    return _W["user_emb"][p["row"]] if p["kind"] == "user" else _W["bank"].emb[_W["bank"].idx(p["spk"])]


def _swap_emb_of(rec: dict):
    return None if rec["swap_spk"] is None else _W["bank"].emb[_W["bank"].idx(rec["swap_spk"])]


def _work(chunk: list[dict]) -> list[dict]:
    rows = []
    for rec in chunk:
        t0 = time.time()
        got = _W["te"].run_item(_W["runner"], rec, _emb_of, _swap_emb_of)
        dt = time.time() - t0
        for r in got:
            r["sec"] = round(dt / len(got), 3)
        rows += got
    return rows


def select_recipes(recs: list[dict], subset: str) -> list[dict]:
    if subset == "full":
        return recs
    keep = []
    for r in recs:
        c = r["cell"]
        if c["axis"] != "matrix":
            keep.append(r)  # extras e usuário sempre entram (são poucos)
            continue
        if subset == "cpu":
            # subconjunto representativo da CPU: nível nominal -30 dBFS, todos os N e ruídos, SIR {-10, 0, +10, +30};
            # nível -45/-18 só em N=0 e N=1 (critérios 6 e 8), SIR 0 e +30
            lv, sir, n = c["level"], c["sir"], c["n"]
            if lv == -30.0 and (n == 0 or sir in (-10.0, 0.0, 10.0, 30.0)):
                keep.append(r)
            elif lv != -30.0 and (n == 0 or (n == 1 and sir in (0.0, 30.0))):
                keep.append(r)
        elif subset == "smoke":
            if c["level"] == -30.0 and c["noise"] == "silent" and (c["n"] in (0, 1) and c["sir"] in (None, 0.0)):
                keep.append(r)
    return keep


def cmd_run(a) -> Path:
    sd = Path(a.set)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = select_recipes(tm.read_recipes(sd / "recipes.jsonl"), a.subset)
    if a.limit:
        recs = recs[: a.limit]
    rows_path = out / "rows.jsonl"
    done: set[tuple[str, str]] = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["id"], d["view"]))

    def pending(rec):
        cell = tm.Cell(**{k: v for k, v in rec["cell"].items() if k != "cid"})
        return any((rec["id"], v) not in done for v in cell.views())

    todo = [r for r in recs if pending(r)]
    log(f"{a.ckpt} [{a.arch}, gating {a.gating}, {a.device}]: {len(recs)} itens, {len(todo)} pendentes, "
        f"{a.workers} processos x {a.threads} threads")
    meta = {"ckpt": str(a.ckpt), "arch": a.arch, "gating": a.gating, "device": a.device, "subset": a.subset,
            "workers": a.workers, "threads": a.threads, "recipes_sha256": tm.recipes_digest(sd / "recipes.jsonl"),
            "label": a.label or f"{a.arch}_{a.gating}", "started": time.strftime("%F %T")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    t0 = time.time()
    chunks = [todo[i : i + a.chunk] for i in range(0, len(todo), a.chunk)]
    if chunks:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(str(sd), str(a.ckpt), a.arch, a.gating, a.device, a.threads)) as ex, \
                open(rows_path, "a") as fh:
            futs = [ex.submit(_work, c) for c in chunks]
            n = 0
            for f in as_completed(futs):
                rows = f.result()
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += len(rows)
                if (n // len(rows)) % 10 == 0:
                    el = time.time() - t0
                    log(f"{n} linhas ({el:.0f} s; {el / max(n, 1):.2f} s/linha de parede)")
    meta["seconds"] = time.time() - t0
    meta["finished"] = time.strftime("%F %T")
    rows_all = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    meta["rows"] = len(rows_all)
    meta["cpu_seconds_per_row"] = float(np.mean([r["sec"] for r in rows_all])) if rows_all else None
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"pronto: {len(rows_all)} linhas em {meta['seconds']:.0f} s")
    return out


# --- report ----------------------------------------------------------------------------------------------------------


def _fmt(x, f="{:+.2f}"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def build_report(run: Path, ref: Path | None, title: str | None = None) -> str:
    from cctrain.eval import tse_eval as te

    meta = json.loads((run / "meta.json").read_text())
    rows = te.load_rows(run / "rows.jsonl")
    t = te.make_table(rows)
    rt = te.make_table(te.load_rows(ref / "rows.jsonl")) if ref else None
    ref_meta = json.loads((ref / "meta.json").read_text()) if ref else None
    crit = te.evaluate_criteria(t, rt)
    L = [f"# Avaliação pDFNet3 Pro v4: {title or meta['label']}", ""]
    L += [f"- checkpoint: `{meta['ckpt']}` (arch {meta['arch']}), gating com perfil: **{meta['gating']}**, device {meta['device']}",
          f"- subconjunto: `{meta['subset']}`; linhas: {len(rows)}; células: {len({r['cid'] for r in rows})}; "
          f"receitas sha256 `{meta['recipes_sha256'][:12]}`",
          f"- tempo: {meta.get('seconds', 0):.0f} s de parede ({meta['workers']} processos x {meta['threads']} threads), "
          f"{(meta.get('cpu_seconds_per_row') or 0):.2f} s de CPU por linha (item x vista)"]
    if ref_meta:
        L.append(f"- referência dos critérios 7 e 9: `{ref_meta['label']}` ({ref_meta['ckpt']})")
    L += ["", "## Critérios de aceite (nominal = nível do alvo -30 dBFS; 'todos' = -45, -30 e -18)", "",
          "| # | Critério | Meta | Nominal | Todos os níveis |", "|---|---|---|---|---|"]
    for c in crit:
        L.append(f"| {c.num} | {c.name} | {c.target} | **{c.status}**: {c.nominal} | {c.all_levels} |")

    # N=0: dano do perfil
    L += ["", "## N=0 (só alvo + ruído): dano do perfil", "",
          "| nível | ruído | dSI-SDR perfil | dSI-SDR sem perfil | dano | dSTOI perfil | dSTOI sem perfil | alvo em ZERO % | alvo em SILENT % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p, a in t.sel("mix", n=0):
        npf = t.cells.get((tm.Cell("matrix", 0, None, p["level"], p["noise"]).cid, "noprof"), {})
        L.append(f"| {p['level']:.0f} | {p['noise']} | {_fmt(a['d_sisdr'])} | {_fmt(npf.get('d_sisdr'))} | "
                 f"{_fmt(a['d_sisdr'] - npf.get('d_sisdr', float('nan')))} | {_fmt(a['d_stoi'], '{:+.3f}')} | "
                 f"{_fmt(npf.get('d_stoi'), '{:+.3f}')} | {_fmt(a['tgt_zero_only_pct'], '{:.1f}')} | "
                 f"{_fmt(a['tgt_zero_pct'] - a['tgt_zero_only_pct'], '{:.1f}')} |")

    # N x SIR no nível nominal
    for nz in tm.NOISES:
        L += ["", f"## Mistura N x SIR (nível -30 dBFS, ruído {nz}): dSI-SDR / dSTOI / supressão / ganho do alvo / alvo ZERO%", "",
              "| N | " + " | ".join(f"SIR {s:+.0f}" for s in tm.SIRS) + " |", "|---|" + "---|" * len(tm.SIRS)]
        for n in tm.NS[1:]:
            cells = []
            for s in tm.SIRS:
                a = t.cells.get((tm.Cell("matrix", n, s, -30.0, nz).cid, "mix"))
                cells.append("-" if not a else f"{a['d_sisdr']:+.1f} / {a['d_stoi']:+.2f} / {a['interf_supp_db']:.0f} / "
                             f"{a['target_gain_db']:+.1f} / {a['tgt_zero_only_pct']:.0f}")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
    for nz in ("silent", "snr5"):
        L += ["", f"## Só-interferente, perfil ligado (nível -30 dBFS, ruído {nz}): %ZERO (ZERO+SILENT) / vazamento mediano dB", "",
              "| N | " + " | ".join(f"SIR {s:+.0f}" for s in tm.SIRS) + " |", "|---|" + "---|" * len(tm.SIRS)]
        for n in tm.NS[1:]:
            cells = []
            for s in tm.SIRS:
                a = t.cells.get((tm.Cell("matrix", n, s, -30.0, nz).cid, "int"))
                cells.append("-" if not a else f"{a['stage_zero_pct']:.0f}% / {a.get('leak_db_median', float('nan')):+.0f}")
            L.append(f"| {n} | " + " | ".join(cells) + " |")

    # robustez de nível
    L += ["", "## Robustez de nível (N=1, SIR 0): dSI-SDR por nível do alvo", "",
          "| ruído | -45 | -30 | -18 |", "|---|---|---|---|"]
    for nz in tm.NOISES:
        vals = [t.cells.get((tm.Cell("matrix", 1, 0.0, lv, nz).cid, "mix")) for lv in tm.LEVELS]
        L.append(f"| {nz} | " + " | ".join("-" if not v else f"{v['d_sisdr']:+.2f}" for v in vals) + " |")

    # extras
    sim = t.cells.get((tm.Cell("similar", 1, 0.0, -30.0, "silent").cid, "mix"))
    if sim:
        base_c = t.cells.get((tm.Cell("matrix", 1, 0.0, -30.0, "silent").cid, "mix"))
        L += ["", "## Eixo 'parecido' (N=1, SIR 0, silencioso, interferente entre os 10 vizinhos, mesmo sexo)", "",
              "| célula | dSI-SDR | dSTOI | supressão dB | ganho do alvo dB |", "|---|---|---|---|---|",
              f"| parecido (40 itens) | {_fmt(sim['d_sisdr'])} | {_fmt(sim['d_stoi'], '{:+.3f}')} | {_fmt(sim['interf_supp_db'])} | {_fmt(sim['target_gain_db'])} |"]
        if base_c:
            L.append(f"| matriz N=1 SIR 0 silencioso (aleatório) | {_fmt(base_c['d_sisdr'])} | {_fmt(base_c['d_stoi'], '{:+.3f}')} | "
                     f"{_fmt(base_c['interf_supp_db'])} | {_fmt(base_c['target_gain_db'])} |")
    L += ["", "## Eixo temporal (conversa e blocos; 5 itens por linha)", "",
          "| N | padrão | ruído | dSI-SDR | alvo ZERO% (só-alvo) | alvo ZERO% (ambos) | só-interf ZERO% | vazamento só-interf dB | silêncio ZERO% |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p, a in t.sel("mix", axis="temporal"):
        L.append(f"| {p['n']} | {p['pattern']} | {p['noise']} | {_fmt(a['d_sisdr'])} | {_fmt(a.get('t_only_zero_pct'), '{:.1f}')} | "
                 f"{_fmt(a.get('both_zero_pct'), '{:.1f}')} | {_fmt(a.get('int_only_zero_pct'), '{:.1f}')} | "
                 f"{_fmt(a.get('leak_intonly_frames_db'), '{:+.1f}')} | {_fmt(a.get('sil_zero_pct'), '{:.1f}')} |")
    L += ["", "## Usuário real (alvo = take4 de joao_messias; perfil de takes 1-2; interferentes reais de dev; nível -30)", "",
          "| ruído | N | SIR | dSI-SDR | dSTOI | supressão dB | ganho alvo dB | alvo ZERO% | só-interf %ZERO | só-interf vazamento dB |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p, a in t.sel("mix", axis="user"):
        i = t.cells.get((tm.Cell("user", p["n"], p["sir"], p["level"], p["noise"]).cid, "int"), {})
        npf = t.cells.get((tm.Cell("user", 0, None, p["level"], p["noise"]).cid, "noprof"), {}) if p["n"] == 0 else {}
        extra = f" (sem perfil {_fmt(npf.get('d_sisdr'))})" if npf else ""
        sir_s = "-" if p["sir"] is None else f"{p['sir']:+.0f}"
        L.append(f"| {p['noise']} | {p['n']} | {sir_s} | {_fmt(a['d_sisdr'])}{extra} | "
                 f"{_fmt(a['d_stoi'], '{:+.3f}')} | {_fmt(a.get('interf_supp_db'))} | {_fmt(a['target_gain_db'])} | "
                 f"{_fmt(a['tgt_zero_only_pct'], '{:.1f}')} | {_fmt(i.get('stage_zero_pct'), '{:.0f}')} | "
                 f"{_fmt(i.get('leak_db_median'), '{:+.0f}')} |")
    L += ["", "Notas: ΔSTOI/ΔESTOI e SI-SDR contra o alvo seco (sem reverberação na matriz); supressão e ganho por projeção de "
          "mínimos quadrados da saída sobre [alvo, interferentes]; ZERO% = só o estágio ZERO decidido pelo modelo; "
          "SILENT = atalho de silêncio do runtime (média quadrática < 1e-7), contado à parte e somado em %ZERO do só-interferente. "
          "DNSMOS não calculado neste harness."]
    return "\n".join(L) + "\n"


def cmd_report(a) -> None:
    md = build_report(Path(a.run), Path(a.ref) if a.ref else None, a.title)
    out = Path(a.md) if a.md else Path(a.run) / "report.md"
    out.write_text(md)
    from cctrain.eval import tse_eval as te

    t = te.make_table(te.load_rows(Path(a.run) / "rows.jsonl"))
    rt = te.make_table(te.load_rows(Path(a.ref) / "rows.jsonl")) if a.ref else None
    crit = te.evaluate_criteria(t, rt)
    (Path(a.run) / "summary.json").write_text(json.dumps(
        {"criteria": [c.__dict__ for c in crit], "cells": {f"{k[0]}||{k[1]}": v for k, v in t.cells.items()}}, indent=1))
    log(f"relatório em {out}")
    for c in crit:
        print(f"  C{c.num:<2} {c.status:5s} {c.nominal}")


def compare_md(runs: list[Path], ref_label_idx: int = 0) -> str:
    from cctrain.eval import tse_eval as te

    tabs, metas = [], []
    for r in runs:
        tabs.append(te.make_table(te.load_rows(r / "rows.jsonl")))
        metas.append(json.loads((r / "meta.json").read_text()))
    labels = [m["label"] for m in metas]
    ref = tabs[ref_label_idx]
    crits = [te.evaluate_criteria(t, ref) for t in tabs]
    L = ["# Comparação de execuções", "", "| execução | checkpoint | gating | linhas | tempo (s) |", "|---|---|---|---|---|"]
    for m, t in zip(metas, tabs, strict=True):
        L.append(f"| {m['label']} | `{Path(m['ckpt']).parent.parent.name}/{Path(m['ckpt']).name}` | {m['gating']} | {len(t.rows)} | {m.get('seconds', 0):.0f} |")
    L += ["", "## Critérios (nominal -30 dBFS)", "", "| # | Critério | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for i in range(len(crits[0])):
        L.append(f"| {crits[0][i].num} | {crits[0][i].name} | " + " | ".join(f"**{c[i].status}**: {c[i].nominal}" for c in crits) + " |")

    def row(name, fn):
        vals = []
        for t in tabs:
            try:
                vals.append(fn(t))
            except Exception:
                vals.append("-")
        L.append(f"| {name} | " + " | ".join(vals) + " |")

    L += ["", "## Números principais (nível -30 dBFS)", "", "| métrica | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    cc = lambda *a, **k: tm.Cell(*a, **k).cid  # noqa: E731
    for nz in tm.NOISES:
        row(f"N=0 {nz}: dSI-SDR perfil", lambda t, nz=nz: f"{t.cells[(cc('matrix', 0, None, -30.0, nz), 'mix')]['d_sisdr']:+.2f}")
        row(f"N=0 {nz}: dSI-SDR sem perfil", lambda t, nz=nz: f"{t.cells[(cc('matrix', 0, None, -30.0, nz), 'noprof')]['d_sisdr']:+.2f}")
        row(f"N=0 {nz}: dano do perfil (dB)", lambda t, nz=nz: f"{t.cells[(cc('matrix', 0, None, -30.0, nz), 'mix')]['d_sisdr'] - t.cells[(cc('matrix', 0, None, -30.0, nz), 'noprof')]['d_sisdr']:+.2f}")
        row(f"N=0 {nz}: dSTOI perfil", lambda t, nz=nz: f"{t.cells[(cc('matrix', 0, None, -30.0, nz), 'mix')]['d_stoi']:+.3f}")
        row(f"N=0 {nz}: alvo zerado (ZERO %)", lambda t, nz=nz: f"{t.cells[(cc('matrix', 0, None, -30.0, nz), 'mix')]['tgt_zero_only_pct']:.2f}")
    for nz in ("silent", "snr5"):
        for n in tm.NS[1:]:
            row(f"N={n} SIR 0 {nz}: dSI-SDR", lambda t, n=n, nz=nz: f"{t.cells[(cc('matrix', n, 0.0, -30.0, nz), 'mix')]['d_sisdr']:+.2f}")
        row(f"N=1 SIR 0 {nz}: supressão / ganho do alvo", lambda t, nz=nz: "{:.1f} / {:+.1f}".format(*(t.cells[(cc('matrix', 1, 0.0, -30.0, nz), 'mix')][k] for k in ("interf_supp_db", "target_gain_db"))))
        row(f"só-interferente {nz} (N=1..3 média, SIR 0): %ZERO / vazamento mediano", lambda t, nz=nz: "{:.0f}% / {:+.0f} dB".format(
            np.mean([t.cells[(cc('matrix', n, 0.0, -30.0, nz), 'int')]['stage_zero_pct'] for n in (1, 2, 3)]),
            np.mean([t.cells[(cc('matrix', n, 0.0, -30.0, nz), 'int')]['leak_db_median'] for n in (1, 2, 3)])))
    for s in (20.0, 30.0):
        row(f"N=1 SIR {s:+.0f} silencioso: supressão (dB)", lambda t, s=s: f"{t.cells[(cc('matrix', 1, s, -30.0, 'silent'), 'mix')]['interf_supp_db']:.1f}")
        row(f"só-interferente N=1 SIR {s:+.0f} silencioso: %ZERO", lambda t, s=s: f"{t.cells[(cc('matrix', 1, s, -30.0, 'silent'), 'int')]['stage_zero_pct']:.0f}")
    row("perfil trocado N=1 SIR 0 silencioso: ganho do alvo (dB)", lambda t: f"{t.cells[(cc('matrix', 1, 0.0, -30.0, 'silent'), 'swap')]['target_gain_db']:+.1f}")
    row("usuário real N=0 snr5: dSI-SDR perfil / sem perfil", lambda t: "{:+.2f} / {:+.2f}".format(
        t.cells[(cc('user', 0, None, -30.0, 'snr5'), 'mix')]['d_sisdr'], t.cells[(cc('user', 0, None, -30.0, 'snr5'), 'noprof')]['d_sisdr']))
    for n in (1, 2, 3, 4):
        row(f"usuário real N={n} SIR 0 silencioso: dSI-SDR / supressão / ganho", lambda t, n=n: "{:+.2f} / {:.1f} / {:+.1f}".format(
            *(t.cells[(cc('user', n, 0.0, -30.0, 'silent'), 'mix')][k] for k in ("d_sisdr", "interf_supp_db", "target_gain_db"))))
    row("usuário real só-interferente N=1 SIR 0 silencioso: %ZERO", lambda t: f"{t.cells[(cc('user', 1, 0.0, -30.0, 'silent'), 'int')]['stage_zero_pct']:.0f}")
    return "\n".join(L) + "\n"


def cmd_compare(a) -> None:
    md = compare_md([Path(r) for r in a.runs], a.ref_index)
    Path(a.md).write_text(md)
    log(f"comparação em {a.md}")


def cmd_eval(a) -> None:
    run = cmd_run(a)
    a.run = str(run)
    a.md = None
    cmd_report(a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--set", default=str(SET_DIR_DEFAULT), help="diretório do conjunto congelado (prep)")

    p = sub.add_parser("prep")
    common(p)
    p.set_defaults(fn=cmd_prep)

    def runargs(p):
        common(p)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--arch", choices=["v2", "v3", "v4"], default="v4")
        p.add_argument("--gating", choices=["current", "new"], default="new")
        p.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else "cpu")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--chunk", type=int, default=4)
        p.add_argument("--subset", choices=["full", "cpu", "smoke"], default="full")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--out", required=True)
        p.add_argument("--label", default="")

    p = sub.add_parser("run")
    runargs(p)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--ref", default="")
    p.add_argument("--md", default="")
    p.add_argument("--title", default="")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("compare")
    p.add_argument("runs", nargs="+")
    p.add_argument("--md", required=True)
    p.add_argument("--ref-index", type=int, default=0)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("eval")
    runargs(p)
    p.add_argument("--ref", default="")
    p.add_argument("--md", default="")
    p.add_argument("--title", default="")
    p.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
