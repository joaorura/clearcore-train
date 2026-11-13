#!/usr/bin/env python3
"""Avaliação rápida, sem treino, do gating SUAVE por presença (exp/soft-gate) contra o ZERO duro.

Variantes (todas com perfil; sem perfil = DFNet3 base de cada modelo, gating atual):
  v2_current      v2 (ckpt do dev v4), estágios por lsnr de hoje (ZERO < -10 < ERB+DF <= 20 < ERB-only <= 30 < passthrough)
  v31_new         v3.1, gating novo por lsnr (ZERO se lsnr < -10, senão ERB+DF)
  C_hard          ckpt C (cabeça p_target), presença dura: τ_on 0,3 / τ_off 0,1 / hangover 12 quadros -> ERB+DF ou ZERO
  C_open          ckpt C, sempre ERB+DF sem ganho (piso 0 dB): o que o gating suave deixa de fazer
  C_soft_f{F}_r{R}  ckpt C, sempre ERB+DF; g = floor + (1-floor)·s, s = p_target suavizado (ataque 15 ms, liberação R ms)

Os decoders rodam uma vez por (item, vista, gating) sobre o MESMO forward do encoder. Métricas por linha (item x vista x
variante) saem de `tse_eval.item_metrics`; o subcomando `report` agrega (somando contagens), pareia o dano do perfil em N=0
com a vista sem perfil do mesmo modelo e confere as três metas.

Uso: python scripts/eval_soft_gate.py run --out <dir> [--workers 3]; python scripts/eval_soft_gate.py report --out <dir>

`--model KEY=CKPT:ARCH:GATING` (repetível, pro-v4-keep): troca o conjunto de modelos; cada um vira uma variante
`KEY_GATING` com o gating por lsnr de `GATINGS` (`current` ou `new`), ex.: `--model v2=...:v2:current
--model v31=...:v4:new --model keep=...:v4:new` -> v2_current, v31_new, keep_new. A lista fica em `meta.json` e o
`report` a relê de lá.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cctrain.eval import tse_matrix as tm  # noqa: E402
from cctrain.paths import Paths  # noqa: E402

PATHS = Paths.from_env()
SET_DIR = PATHS.data / "eval-v4"
RUNS = PATHS.runs
MODELS = {  # chave -> (ckpt, arch)
    "v2": (RUNS / "m2/pro-v2-dualpath-tse-20261008/checkpoints/ckpt-00075000.pt", "v2"),
    "v31": (RUNS / "m4/pro-v4/checkpoints/ckpt-00079998.pt", "v4"),
    "C": (RUNS / "m4/v4-presence-smoke-c/checkpoints/ckpt-00000500.pt", "v4p"),
}
FLOORS = (-10.0, -15.0, -20.0, -30.0)
RELEASES = (150.0, 300.0)
ATTACK_MS = 15.0


LATENCY_MODEL_MS = 30.0  # RUNTIME_DELAY: 1.440 amostras a 48 kHz (2 quadros de lookahead + 1 da STFT)
LOOKAHEADS = (0.0, 20.0, 40.0, 80.0)
WINDOWS = (100.0, 200.0)
_MODE = {"set": "main", "floor": -20.0, "release": 200.0}


def latency_ms(name: str, g) -> float:
    """Latência algorítmica total: a do modelo (30 ms) + o atraso extra da decisão (L do lookahead; W da janela fixa)."""
    return LATENCY_MODEL_MS + (g.lookahead_ms if g.kind == "soft" else g.window_ms if g.kind == "window" else 0.0)


def parse_models(specs: list[str]) -> dict[str, tuple[str, str, str]]:
    """`KEY=CKPT:ARCH:GATING` -> {KEY: (CKPT, ARCH, GATING)}."""
    out = {}
    for sp in specs:
        key, rest = sp.split("=", 1)
        ck, arch, gating = rest.rsplit(":", 2)
        out[key] = (ck, arch, gating)
    return out


def variants() -> dict[str, tuple[str, object]]:
    from cctrain.eval.tse_eval import Gating

    if _MODE.get("models"):
        return {f"{k}_{g}": (k, Gating.of(g)) for k, (_c, _a, g) in _MODE["models"].items()}

    if _MODE["set"] == "la":
        fl, rel = _MODE["floor"], _MODE["release"]
        v = {"v2_current": ("v2", Gating.of("current")), "v31_new": ("v31", Gating.of("new")),
             "C_hard": ("C", Gating("hard", tau_on=0.3, tau_off=0.1, hangover=12)),
             "C_open": ("C", Gating("soft", floor_db=0.0, attack_ms=ATTACK_MS, release_ms=150.0))}
        for la in LOOKAHEADS:
            v[f"C_soft_f{-fl:.0f}_r{rel:.0f}_L{la:.0f}"] = ("C", Gating("soft", floor_db=fl, attack_ms=ATTACK_MS,
                                                                        release_ms=rel, lookahead_ms=la))
        for w in WINDOWS:
            v[f"C_win{w:.0f}_f{-fl:.0f}"] = ("C", Gating("window", floor_db=fl, window_ms=w))
        return v

    v: dict[str, tuple[str, object]] = {"v2_current": ("v2", Gating.of("current")), "v31_new": ("v31", Gating.of("new")),
                                        "C_hard": ("C", Gating("hard", tau_on=0.3, tau_off=0.1, hangover=12)),
                                        "C_open": ("C", Gating("soft", floor_db=0.0, attack_ms=ATTACK_MS, release_ms=150.0))}
    for rel in RELEASES:
        for fl in FLOORS:
            v[f"C_soft_f{-fl:.0f}_r{rel:.0f}"] = ("C", Gating("soft", floor_db=fl, attack_ms=ATTACK_MS, release_ms=rel))
    return v


def select(recs: list[dict]) -> list[dict]:
    """-30 dBFS, ruído silencioso e SNR 5: N=0 e N=1..4 com SIR 0; usuário real idem; eixo temporal inteiro."""
    keep = []
    for r in recs:
        c = r["cell"]
        if c["axis"] == "temporal":
            keep.append(r)
        elif c["axis"] in ("matrix", "user") and c["level"] == -30.0 and c["noise"] in ("silent", "snr5") \
                and (c["n"] == 0 or c["sir"] == 0.0):
            keep.append(r)
    return keep


_W: dict = {}


def _init(threads: int, mode: dict) -> None:
    import torch

    _MODE.update(mode)

    torch.set_num_threads(threads)
    from cctrain.eval import tse_eval as te

    tm.set_roots(PATHS.data, SET_DIR / "user_take4_stream_48k.wav")
    _W["bank"] = tm.Bank.from_json(json.loads((SET_DIR / "bank.json").read_text()), np.load(SET_DIR / "bank_emb.npy"))
    _W["user_emb"] = np.load(SET_DIR / "user_emb.npy")
    models = {k: (Path(c), a) for k, (c, a, _g) in mode["models"].items()} if mode.get("models") else MODELS
    _W["runners"] = {k: te.Runner(ck, arch, "new", "cpu") for k, (ck, arch) in models.items()}
    _W["te"] = te
    _W["var"] = variants()


def _work(chunk: list[dict]) -> list[dict]:
    te, bank, var = _W["te"], _W["bank"], _W["var"]
    rows: list[dict] = []
    for rec in chunk:
        t0 = time.time()
        cell = tm.Cell(**{k: v for k, v in rec["cell"].items() if k != "cid"})
        r = tm.render(rec)
        p = rec["profile"]
        prof = _W["user_emb"][p["row"]] if p["kind"] == "user" else bank.emb[bank.idx(p["spk"])]
        noisy_mix = r.noisy(True)
        base = te.input_baseline(r, noisy_mix)
        got = []
        for view in cell.views():
            if view == "swap":
                continue
            if view == "noprof":
                for mk, rn in _W["runners"].items():
                    enh, st = rn.run(noisy_mix, None)
                    got.append((mk, view, te.item_metrics("noprof", rec, r, noisy_mix, enh, st, base)))
                continue
            noisy = noisy_mix if view == "mix" else r.noisy(False)
            for mk, rn in _W["runners"].items():
                names = {n: g for n, (m, g) in var.items() if m == mk}
                for n, (enh, st) in rn.run_multi(noisy, prof, names).items():
                    got.append((n, view, te.item_metrics(view, rec, r, noisy, enh, st, base)))
        dt = time.time() - t0
        for name, view, m in got:
            rows.append({"id": rec["id"], "cid": rec["cell"]["cid"], "view": view, "variant": name,
                         "target": rec["target"]["spk"], "sec": round(dt / len(got), 3), **m})
    return rows


def cmd_run(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _MODE.update(set=a.set, floor=a.floor, release=a.release, models=parse_models(a.model) if a.model else None)
    recs = select(tm.read_recipes(SET_DIR / "recipes.jsonl"))
    if a.limit:
        recs = recs[: a.limit]
    rows_path = out / ("rows.jsonl" if a.set == "main" else f"rows-{a.set}.jsonl")
    done = set()
    if rows_path.exists():
        done = {json.loads(line)["id"] for line in rows_path.read_text().splitlines() if line.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} itens, {len(todo)} pendentes, {a.workers} processos x {a.threads} thread(s)", flush=True)
    models = {k: [c, ar, g] for k, (c, ar, g) in _MODE["models"].items()} if _MODE["models"] else \
        {k: [str(v[0]), v[1]] for k, v in MODELS.items()}
    meta = {"models": models, "custom_models": bool(_MODE["models"]), "variants": list(variants()),
            "recipes_sha256": tm.recipes_digest(SET_DIR / "recipes.jsonl"), "items": len(recs), "attack_ms": ATTACK_MS,
            "hard": {"tau_on": 0.3, "tau_off": 0.1, "hangover": 12}, "started": time.strftime("%F %T")}
    (out / ("meta.json" if a.set == "main" else f"meta-{a.set}.json")).write_text(json.dumps(meta, indent=1))
    t0 = time.time()
    chunks = [todo[i : i + a.chunk] for i in range(0, len(todo), a.chunk)]
    if chunks:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                 initargs=(a.threads, _MODE)) as ex, open(rows_path, "a") as fh:
            futs = [ex.submit(_work, c) for c in chunks]
            n = 0
            for f in as_completed(futs):
                for row in f.result():
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
                if n % 5 == 0:
                    print(f"{n}/{len(chunks)} blocos ({time.time() - t0:.0f} s)", flush=True)
    meta["seconds"] = time.time() - t0
    (out / ("meta.json" if a.set == "main" else f"meta-{a.set}.json")).write_text(json.dumps(meta, indent=1))
    print(f"pronto em {meta['seconds']:.0f} s", flush=True)


# --- agregação ------------------------------------------------------------------------------------------------------


def cellinfo(cid: str) -> dict:
    axis, n, sir, lv, noise, pat = cid.split("|")
    return {"axis": axis, "n": int(n[1:]), "sir": None if sir[3:] == "na" else float(sir[3:]), "noise": noise,
            "pattern": pat}


def _mean(rs, k):
    v = np.array([r[k] for r in rs if r.get(k) is not None], float)
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"), len(v)) if len(v) \
        else (float("nan"), float("nan"), 0)


def _pct(rs, num, den):
    d = sum(r.get(den, 0) for r in rs)
    return 100.0 * sum(r.get(num, 0) for r in rs) / d if d else float("nan")


def _att_mean(rs, pre):
    d = sum(r.get(f"{pre}_n", 0) for r in rs)
    return sum(r.get(f"{pre}_att_sum", 0.0) for r in rs) / d if d else float("nan")


def agg(rows: list[dict], variant: str, **sel) -> dict:
    """Agrega as linhas de `variant` no recorte `sel` (axis, n, noise, nset). `dif`: supressão + ganho do alvo
    (mínimos quadrados, global); `dif_f`: atenuação média por quadro no só-interferente (vista `int`) + ganho do alvo nos
    quadros fortes (vista `mix`); `dif_f_mix` usa os quadros só-interferente da própria mistura."""

    def pick(view, var):
        out = []
        for r in rows:
            if r["variant"] != var or r["view"] != view:
                continue
            c = cellinfo(r["cid"])
            if all((c[k] in v if isinstance(v, (tuple, set, list)) else c[k] == v) for k, v in sel.items()):
                out.append(r)
        return out

    mix, itv = pick("mix", variant), pick("int", variant)
    res: dict = {"n_items": len(mix)}
    for k in ("d_sisdr", "d_stoi", "d_estoi", "interf_supp_db", "target_gain_db", "target_gain_hi_db", "supp_overlap_db",
              "tgt_gain_overlap_db"):
        res[k] = _mean(mix, k)[0]
    ds = [r["interf_supp_db"] + r["target_gain_db"] for r in mix if "interf_supp_db" in r]
    res["dif"] = float(np.mean(ds)) if ds else float("nan")
    res["dif_se"] = float(np.std(ds, ddof=1) / np.sqrt(len(ds))) if len(ds) > 1 else float("nan")
    res["io_att"] = _att_mean(itv, "io")
    res["io_ge10"], res["io_ge20"] = _pct(itv, "io_ge10", "io_n"), _pct(itv, "io_ge20", "io_n")
    res["oi_att"] = _att_mean(mix, "oi")
    res["oi_ge10"], res["oi_ge20"] = _pct(mix, "oi_ge10", "oi_n"), _pct(mix, "oi_ge20", "oi_n")
    res["dif_f"] = res["io_att"] + res["target_gain_hi_db"]
    res["dif_f_mix"] = res["oi_att"] + res["target_gain_hi_db"]
    res["seg_io"] = _pct(itv, "seg_io_ge15", "seg_io_n")
    res["seg_oi"] = _pct(mix, "seg_oi_ge15", "seg_oi_n")
    res["seg_t"] = _pct(mix, "seg_t_gt3", "seg_t_n")
    res["onset_gain"] = (sum(r.get("onset_gain_sum", 0.0) for r in mix) / max(sum(r.get("onset_n", 0) for r in mix), 1)
                         if any(r.get("onset_n", 0) for r in mix) else float("nan"))
    res["onset_gt3"] = _pct(mix, "onset_gt3", "onset_n")
    res["n_onsets"] = sum(r.get("onset_n", 0) for r in mix)
    res["n_io_frames"], res["n_seg_io"], res["n_seg_oi"] = sum(r.get("io_n", 0) for r in itv), \
        sum(r.get("seg_io_n", 0) for r in itv), sum(r.get("seg_oi_n", 0) for r in mix)
    return res


def n0_damage(rows: list[dict], variant: str, model: str, **sel) -> dict:
    """Dano do perfil em N=0: métrica(variante, vista mix) - métrica(modelo, vista noprof) pareado por item."""
    nop = {r["id"]: r for r in rows if r["view"] == "noprof" and r["variant"] == model}
    out: dict = {}
    for k in ("d_sisdr", "d_sisdr_hi", "target_gain_hi_db", "d_stoi"):
        d = []
        for r in rows:
            if r["variant"] != variant or r["view"] != "mix" or r["id"] not in nop:
                continue
            c = cellinfo(r["cid"])
            if c["n"] != 0 or not all(c[kk] == v for kk, v in sel.items()):
                continue
            if k in r and k in nop[r["id"]]:
                d.append(r[k] - nop[r["id"]][k])
        out[k] = float(np.mean(d)) if d else float("nan")
        out[k + "_se"] = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    seg = [r for r in rows if r["variant"] == variant and r["view"] == "mix" and cellinfo(r["cid"])["n"] == 0
           and all(cellinfo(r["cid"])[kk] == v for kk, v in sel.items())]
    out["seg_t"] = _pct(seg, "seg_t_gt3", "seg_t_n")
    seg0 = [r for r in rows if r["variant"] == model and r["view"] == "noprof" and all(cellinfo(r["cid"])[kk] == v for kk, v in sel.items())]
    out["seg_t_noprof"] = _pct(seg0, "seg_t_gt3", "seg_t_n")
    return out


def f(x, fmt="{:+.1f}"):
    return "-" if x is None or not np.isfinite(x) else fmt.format(x)


def build_report_la(out: Path) -> str:
    """Variantes com lookahead (atraso L na saída) e decisão por janela fixa, contra os baselines, com o corte de início
    de trecho do alvo (primeiros 100 ms após silêncio ou só-interferente) e a latência algorítmica total."""
    rows = [json.loads(line) for line in (out / "rows-la.jsonl").read_text().splitlines() if line.strip()]
    var = variants()
    names = [n for n in var if any(r["variant"] == n for r in rows)]
    model_of = {n: m for n, (m, _) in var.items()}
    lat = {n: latency_ms(n, g) if model_of[n] == "C" and g.kind != "hard" else LATENCY_MODEL_MS for n, (_, g) in var.items()}
    L = ["# Gating suave com lookahead e janela fixa", "",
         f"- linhas: {len(rows)}; itens: {len({r['id'] for r in rows})}; soft base: piso {-_MODE['floor']:.0f} dB, liberação "
         f"{_MODE['release']:.0f} ms, ataque {ATTACK_MS:.0f} ms; latência do modelo {LATENCY_MODEL_MS:.0f} ms; lookahead = máximo de "
         "p_target em [t, t+L] antes da suavização; janela = mediana de p_target na janela fixa aplicada à janela inteira "
         "(latência = 30 ms + W, pior caso do primeiro quadro)", ""]
    hdr = ("| variante | latência ms | ΔSI-SDR | ΔSTOI | ganho alvo (fortes) | **dif** | **dif_f** | só-int att (int) | ≥10 | ≥20 | "
           "seg só-int ≥15 (int) | seg só-int ≥15 (mix) | seg alvo >3 | **início: ganho médio** | **início atenuado >3 (%)** |\n|---|" + "---|" * 14)
    groups = [("Matriz N=1..4, SIR 0, silencioso", dict(axis="matrix", n=(1, 2, 3, 4), noise="silent")),
              ("Matriz N=1..4, SIR 0, SNR 5", dict(axis="matrix", n=(1, 2, 3, 4), noise="snr5")),
              ("Usuário real N=1..4, SIR 0, silencioso", dict(axis="user", n=(1, 2, 3, 4), noise="silent")),
              ("Usuário real N=1..4, SIR 0, SNR 5", dict(axis="user", n=(1, 2, 3, 4), noise="snr5")),
              ("Usuário real N=1, SIR 0, silencioso", dict(axis="user", n=1, noise="silent")),
              ("Usuário real N=1, SIR 0, SNR 5", dict(axis="user", n=1, noise="snr5")),
              ("Temporal (blocos e conversa, N=1 e 3)", dict(axis="temporal"))]
    for title, sel in groups:
        L += [f"## {title}", "", hdr]
        for n in names:
            a = agg(rows, n, **sel)
            L.append(f"| {n} | {lat[n]:.0f} | {f(a['d_sisdr'])} | {f(a['d_stoi'], '{:+.3f}')} | {f(a['target_gain_hi_db'])} | "
                     f"{f(a['dif'])} | {f(a['dif_f'])} | {f(a['io_att'])} | {f(a['io_ge10'], '{:.0f}')} | {f(a['io_ge20'], '{:.0f}')} | "
                     f"{f(a['seg_io'], '{:.0f}')} | {f(a['seg_oi'], '{:.0f}')} | {f(a['seg_t'], '{:.0f}')} | "
                     f"{f(a['onset_gain'])} | {f(a['onset_gt3'], '{:.0f}')} |")
        L.append("")
    L += ["## N=0: dano do perfil (perfil - sem perfil do modelo, pareado)", "",
          "| variante | recorte | Δ ΔSI-SDR | Δ ganho alvo (fortes) | seg alvo >3 (perfil / sem perfil) |", "|---|---|---|---|---|"]
    for n in names:
        for lab, sel in (("matriz silencioso", dict(axis="matrix", noise="silent")), ("matriz SNR 5", dict(axis="matrix", noise="snr5")),
                         ("usuário silencioso", dict(axis="user", noise="silent")), ("usuário SNR 5", dict(axis="user", noise="snr5"))):
            d = n0_damage(rows, n, model_of[n], **sel)
            L.append(f"| {n} | {lab} | {f(d['d_sisdr'], '{:+.2f}')} | {f(d['target_gain_hi_db'], '{:+.2f}')} | "
                     f"{f(d['seg_t'], '{:.0f}')} / {f(d['seg_t_noprof'], '{:.0f}')} |")
    return "\n".join(L) + "\n"


def build_report(out: Path) -> str:
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines() if line.strip()]
    var = variants()
    names = [n for n in var if any(r["variant"] == n for r in rows)]
    model_of = {n: m for n, (m, _) in var.items()}
    if _MODE.get("models"):
        L = ["# Avaliação TSE (gating por lsnr, caminho do runtime)", "",
             f"- linhas: {len(rows)}; itens: {len({r['id'] for r in rows})}"] + \
            [f"- `{k}_{g}`: `{c}` (arch {ar}, gating {g})" for k, (c, ar, g) in _MODE["models"].items()] + [""]
    else:
        L = ["# Gating suave por presença: avaliação sem treino", "",
             f"- linhas: {len(rows)}; itens: {len({r['id'] for r in rows})}; ckpt C = `{MODELS['C'][0]}` (500 passos, smoke c); "
             f"v3.1 = `{MODELS['v31'][0]}`; v2 = `{MODELS['v2'][0]}`",
             f"- gating soft: ataque {ATTACK_MS:.0f} ms, liberação 150/300 ms (polo único), piso em dB no nome (`f10` = -10 dB); "
             "`C_open` = ERB+DF sempre, sem ganho (piso 0 dB); `C_hard` = presença 0,3/0,1/12 (ZERO duro)", ""]
    groups = [("Matriz N=1..4, SIR 0, -30 dBFS, silencioso", dict(axis="matrix", n=(1, 2, 3, 4), noise="silent")),
              ("Matriz N=1..4, SIR 0, -30 dBFS, SNR 5", dict(axis="matrix", n=(1, 2, 3, 4), noise="snr5")),
              ("Matriz N=1, SIR 0, -30 dBFS, silencioso", dict(axis="matrix", n=1, noise="silent")),
              ("Matriz N=1, SIR 0, -30 dBFS, SNR 5", dict(axis="matrix", n=1, noise="snr5")),
              ("Usuário real (take4) N=1..4, SIR 0, silencioso", dict(axis="user", n=(1, 2, 3, 4), noise="silent")),
              ("Usuário real (take4) N=1..4, SIR 0, SNR 5", dict(axis="user", n=(1, 2, 3, 4), noise="snr5")),
              ("Usuário real N=1, SIR 0, silencioso", dict(axis="user", n=1, noise="silent")),
              ("Usuário real N=1, SIR 0, SNR 5", dict(axis="user", n=1, noise="snr5"))]
    hdr = ("| variante | ΔSI-SDR | ΔSTOI | ΔESTOI | ganho alvo | ganho alvo (fortes) | supressão | supr. sobrep. | **dif** | "
           "**dif_f** | só-int att (int) | ≥10 | ≥20 | só-int att (mix) | seg só-int ≥15 (int) | seg só-int ≥15 (mix) | "
           "seg alvo >3 | início: ganho | início >3 (%) |\n|---|" + "---|" * 18)
    for title, sel in groups:
        L += [f"## {title}", "", hdr]
        for n in names:
            a = agg(rows, n, **sel)
            L.append(f"| {n} | {f(a['d_sisdr'])} | {f(a['d_stoi'], '{:+.3f}')} | {f(a['d_estoi'], '{:+.3f}')} | "
                     f"{f(a['target_gain_db'])} | {f(a['target_gain_hi_db'])} | {f(a['interf_supp_db'])} | "
                     f"{f(a['supp_overlap_db'])} | {f(a['dif'])} | {f(a['dif_f'])} | {f(a['io_att'])} | "
                     f"{f(a['io_ge10'], '{:.0f}')} | {f(a['io_ge20'], '{:.0f}')} | {f(a['oi_att'])} | "
                     f"{f(a['seg_io'], '{:.0f}')} | {f(a['seg_oi'], '{:.0f}')} | {f(a['seg_t'], '{:.0f}')} | "
                     f"{f(a['onset_gain'])} | {f(a['onset_gt3'], '{:.0f}')} |")
        L.append("")
    L += ["## Por N (matriz, SIR 0, -30 dBFS) e por ruído: dif / dif_f / seg só-int ≥15 (int) / seg alvo >3", "",
          "| variante | ruído | " + " | ".join(f"N={n}" for n in (1, 2, 3, 4)) + " |", "|---|---|---|---|---|---|"]
    for n in names:
        for nz in ("silent", "snr5"):
            cells = []
            for k in (1, 2, 3, 4):
                a = agg(rows, n, axis="matrix", n=k, noise=nz)
                cells.append(f"{f(a['dif'])} / {f(a['dif_f'])} / {f(a['seg_io'], '{:.0f}')} / {f(a['seg_t'], '{:.0f}')}")
            L.append(f"| {n} | {nz} | " + " | ".join(cells) + " |")
    L += ["", "## N=0: dano do perfil (perfil - sem perfil do mesmo modelo, pareado por item)", "",
          "| variante | recorte | Δ ΔSI-SDR | Δ ΔSI-SDR (fortes) | Δ ganho alvo (fortes) | Δ ΔSTOI | seg alvo >3 (perfil / sem perfil) |",
          "|---|---|---|---|---|---|---|"]
    for n in names:
        for lab, sel in (("matriz silencioso", dict(axis="matrix", noise="silent")), ("matriz SNR 5", dict(axis="matrix", noise="snr5")),
                         ("usuário silencioso", dict(axis="user", noise="silent")), ("usuário SNR 5", dict(axis="user", noise="snr5"))):
            d = n0_damage(rows, n, model_of[n], **sel)
            L.append(f"| {n} | {lab} | {f(d['d_sisdr'], '{:+.2f}')} ± {f(d['d_sisdr_se'], '{:.2f}')} | {f(d['d_sisdr_hi'], '{:+.2f}')} | "
                     f"{f(d['target_gain_hi_db'], '{:+.2f}')} | {f(d['d_stoi'], '{:+.3f}')} | {f(d['seg_t'], '{:.0f}')} / {f(d['seg_t_noprof'], '{:.0f}')} |")
    L += ["", "## Eixo temporal (blocos/conversa; alvo e interferentes com janelas): dif_f_mix (só-interferente na mistura) / seg só-int ≥15 (mix) / seg alvo >3", "",
          "| variante | N=1 blocks | N=1 conv | N=3 blocks | N=3 conv |", "|---|---|---|---|---|"]
    for n in names:
        cells = []
        for k in (1, 3):
            for pat in ("blocks", "conv"):
                a = agg(rows, n, axis="temporal", n=k, pattern=pat)
                cells.append(f"{f(a['dif_f_mix'])} / {f(a['seg_oi'], '{:.0f}')} / {f(a['seg_t'], '{:.0f}')}")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L += ["", "Definições: `dif` = supressão do interferente (mínimos quadrados, global) + ganho do alvo (dB, negativo = atenuação). "
          "`dif_f` = atenuação média por quadro nos quadros só-interferente (vista `int`: mesma mistura sem o alvo) + ganho do alvo nos "
          "quadros com energia >= -20 dB do máximo (mínimos quadrados só nesses quadros). Segmentos de 300 ms, sem sobreposição; "
          "só-interferente = sem alvo ativo e interferente ativo em >= 50% dos quadros; segmento com alvo = alvo ativo em >= 50%, "
          "atenuação do alvo por mínimos quadrados no segmento. Atenuações por quadro limitadas a 60 dB. Com ruído SNR 5 a energia "
          "total inclui o ruído (a atenuação do só-interferente mistura interferente e ruído)."]
    return "\n".join(L) + "\n"


def cmd_report(a) -> None:
    out = Path(a.out)
    _MODE.update(set=a.set, floor=a.floor, release=a.release)
    meta_p = out / ("meta.json" if a.set == "main" else f"meta-{a.set}.json")
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("custom_models"):
            _MODE["models"] = {k: tuple(v) for k, v in meta["models"].items()}
    md = build_report_la(out) if a.set == "la" else build_report(out)
    name = "report.md" if a.set == "main" else f"report-{a.set}.md"
    (out / name).write_text(md)
    print(f"relatório em {out / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--chunk", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--set", choices=["main", "la"], default="main")
    p.add_argument("--floor", type=float, default=-20.0, help="set la: piso (dB) da melhor combinação")
    p.add_argument("--release", type=float, default=200.0, help="set la: liberação (ms) da melhor combinação")
    p.add_argument("--model", action="append", default=[], help="KEY=CKPT:ARCH:GATING (repetível)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--set", choices=["main", "la"], default="main")
    p.add_argument("--floor", type=float, default=-20.0)
    p.add_argument("--release", type=float, default=200.0)
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
