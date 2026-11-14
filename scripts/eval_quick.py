#!/usr/bin/env python3
"""Avaliação rápida (pro-v42) de um checkpoint: subconjunto do `eval_quality` — usuário take4 N=0 e N=1 (SIR 0,
silencioso + SNR 5) e ruído SNR 0 (24 itens) — com os critérios C1–C8 e R0 do relatório completo, contra o v3.1 e o
DFNet3 publicado (linhas de referência calculadas uma vez e guardadas em `<cache>/ref-rows.jsonl`).

Também mede o ganho do alvo nos quadros "alvo sozinho" do usuário N=1 silencioso por tempo desde o último quadro com
interferente (VAD do eval, pico−30 dB, e o VAD conservador pico−45 dB + 100 ms após o fim excluídos).

Uso:
  eval_quick.py one --ckpt CKPT [--step N] --cache DIR [--jsonl OUT]      (uma avaliação, 1 processo, CPU)
  eval_quick.py watch --snap-dir DIR --cache DIR --jsonl OUT --best-dir DIR [--done-file F]
      (laço: avalia cada snapshot novo em DIR, apaga o que não for o melhor; termina quando `done-file` existir e não
       houver pendentes)
Melhor = mais PASS; desempate: C8, depois C6, depois C1 (`--tiebreak` muda a ordem; pro-v5 usa c6,c1).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# CPU sempre: o `init_df` do DeepFilterNet (DFNet3 upstream, carregado pelo Runner) faz `model.to(get_device())`, que é
# CUDA quando há GPU visível, ignorando o device "cpu" pedido. Com o treino ocupando a GPU, cada snapshot do watch
# falhava com CUDA OOM e o jsonl (e a guarda de 2k/4k) ficava vazio até o fim do treino.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import eval_quality as eq  # noqa: E402
from cctrain.eval import tse_matrix as tm  # noqa: E402

V31 = eq.MODELS["v31"]
BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9))
N_NOISE = 24


def subset() -> tuple[list[dict], list[dict]]:
    recs = eq.all_recipes()
    main = [r for r in eq.select_main(recs) if r["cell"]["axis"] == "user" and r["cell"]["n"] in (0, 1)]
    noise = [r for r in eq.noise_recipes(recs) if r["cell"]["noise"] == "snr0"][:N_NOISE]
    return main, noise


def _dilate(m: np.ndarray, k: int) -> np.ndarray:
    out = m.copy()
    for s in range(1, k + 1):
        out[s:] |= m[:-s]
    return out


def since_bins(r, tshadow: np.ndarray) -> dict:
    """Ganho do alvo-sombra nos quadros alvo-sozinho por tempo desde o interferente: {vad: {bin: [ganhos]}}."""
    te = eq._W["te"]
    nf = len(r.target) // tm.HOP
    act = te.target_active_local(r.target, nf)
    g = te.frame_gain_db(r.target, tshadow, nf)
    out = {}
    for name, rel, ext in (("p30", 30.0, 0), ("p45x100", 45.0, 10)):
        ia = np.zeros(nf, bool)
        for x in r.interf:
            ia |= tm.activity(x, rel_db=rel)[:nf]
        tonly = act & ~(_dilate(ia, ext) if ext else ia)
        since = np.full(nf, np.inf)
        last = None
        for f in range(nf):
            if ia[f]:
                last = f
            elif last is not None:
                since[f] = (f - last) * eq.FRAME_S
        d = {"antes": g[tonly & ~np.isfinite(since)].tolist()}
        for lo, hi in BINS:
            m = tonly & np.isfinite(since) & (since > lo if lo > 0 else since >= 0) & (since <= hi)
            d[f"{lo:g}-{hi:g}s" if hi < 1e8 else f">{lo:g}s"] = g[m].tolist()
        out[name] = d
    return out


def run_items(recs: list[dict], runners: dict, with_dfn3: bool, bins_acc: dict | None) -> list[dict]:
    te = eq._W["te"]
    rows = []
    for rec in recs:
        c = rec["cell"]
        r = tm.render(rec)
        noisy = r.noisy(True)
        base = te.quality_baseline(r, noisy, te.input_baseline(r, noisy))
        comps = [r.target, r.noise]
        prof = eq.profile_of(rec)
        got = []
        for mk, rn in runners.items():
            enh, st, sh = rn.run(noisy, prof, shadow=comps)
            got.append((f"{mk}_new", "mix", te.item_metrics("mix", rec, r, noisy, enh, st, base, sh, None)))
            if bins_acc is not None and c["axis"] == "user" and c["n"] == 1 and c["noise"] == "silent":
                for vad, d in since_bins(r, sh[0]).items():
                    for k, v in d.items():
                        bins_acc.setdefault(mk, {}).setdefault(vad, {}).setdefault(k, []).extend(v)
            if c["axis"] == "user" and c["n"] == 0:
                enh, st, sh = rn.run(noisy, None, shadow=comps)
                got.append((f"{mk}_noprof", "noprof", te.item_metrics("noprof", rec, r, noisy, enh, st, base, sh, None)))
            if c["axis"] == "user" and c["n"] > 0:
                ni = r.noisy(False)
                enh, st = rn.run(ni, prof)
                got.append((f"{mk}_new", "int", te.item_metrics("int", rec, r, ni, enh, st, base)))
        if with_dfn3 and c["n"] == 0 and c["axis"] == "noise":
            enh, st, sh = eq._W["dfn3"].run(noisy, None, shadow=comps)
            got.append(("dfn3_noprof", "noprof", te.item_metrics("noprof", rec, r, noisy, enh, st, base, sh, None)))
        for name, view, m in got:
            rows.append({"id": rec["id"], "cid": c["cid"], "view": view, "variant": name,
                         "src_axis": rec.get("src_axis", c["axis"]), **m})
    return rows


def init_env(threads: int) -> None:
    import torch

    from cctrain.eval import tse_eval as te

    torch.set_num_threads(threads)
    tm.set_roots(eq.PATHS.data, eq.SET_DIR / "user_take4_stream_48k.wav")
    eq._W["bank"] = tm.Bank.from_json(json.loads((eq.SET_DIR / "bank.json").read_text()),
                                      np.load(eq.SET_DIR / "bank_emb.npy"))
    eq._W["user_emb"] = np.load(eq.SET_DIR / "user_emb.npy")
    eq._W["te"] = te


def refs(cache: Path) -> tuple[list[dict], dict]:
    """Linhas do v3.1 (perfil, sem perfil, int) e do DFNet3 publicado (ruído) no subconjunto, com cache."""
    cache.mkdir(parents=True, exist_ok=True)
    p, pb = cache / "ref-rows.jsonl", cache / "ref-bins.json"
    if p.exists() and pb.exists():
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()], json.loads(pb.read_text())
    te = eq._W["te"]
    main, noise = subset()
    eq._W["dfn3"] = te.BaseRunner()
    rn = {"v31": te.Runner(V31[0], V31[1], V31[2], "cpu")}
    bins: dict = {}
    rows = run_items(main + noise, rn, True, bins)
    del eq._W["dfn3"]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pb.write_text(json.dumps(bins))
    return rows, bins


def bins_summary(b: dict) -> dict:
    return {vad: {k: {"n": len(v), "g_med": float(np.median(v)) if v else None} for k, v in d.items()}
            for vad, d in b.items()}


def evaluate(ckpt: Path, step: int | None, cache: Path) -> dict:
    t0 = time.time()
    ref_rows, ref_bins = refs(cache)
    te = eq._W["te"]
    main, noise = subset()
    rn = {"cand": te.Runner(ckpt, "v4", "new", "cpu")}
    bins: dict = {}
    rows = run_items(main + noise, rn, False, bins)
    del rn
    allr = ref_rows + rows
    mrows = [r for r in allr if not r["cid"].startswith("noise|")]
    nrows = [r for r in allr if r["cid"].startswith("noise|")]
    eq.PROF = ("v31_new", "cand_new")
    eq.MODEL_OF = {"v31_new": "v31", "cand_new": "cand"}
    eq.NOISE_SNRS = (0.0,)
    _, res = eq.criteria(mrows, nrows)
    crit = {}
    for v in ("v31_new", "cand_new"):
        crit[v] = {k.split(" ")[0]: [bool(p), s] for k, (p, s) in res[v].items()}
    c = crit["cand_new"]
    u1 = eq.qagg(eq.pick(mrows, "cand_new", "mix", axis="user", n=1))
    pd = eq.paired(mrows, "cand_new", "cand_noprof", ["d_sisdr"], axis="user", n=0, noise="silent")
    dmg, dmg_se = pd["d_sisdr"], pd["d_sisdr_se"]
    try:
        c8 = float(c["C8"][1].rstrip("%"))
    except (KeyError, ValueError):
        c8 = float("nan")
    out = {"c8_pct": c8, "step": step, "ckpt": str(ckpt), "pass": sum(p for p, _ in c.values()), "n_crit": len(c),
           "c6_db": u1["tonly_g_med"], "c1_db": dmg, "c1_se": dmg_se, "crit": c, "v31_crit": crit["v31_new"],
           "bins": bins_summary(bins.get("cand", {})), "v31_bins": bins_summary(ref_bins.get("v31", {})),
           "sec": round(time.time() - t0, 1), "wall": time.strftime("%F %T")}
    return out


TIEBREAK_KEYS = {"c8": "c8_pct", "c6": "c6_db", "c1": "c1_db"}


def better(a: dict, b: dict | None, tiebreak: str = "c8,c6,c1") -> bool:
    if b is None:
        return True
    order = [TIEBREAK_KEYS[k.strip()] for k in tiebreak.split(",") if k.strip()]

    def key(r):  # mais PASS; desempate na ordem de `tiebreak` (padrão C8, C6, C1; pro-v5: C6, C1)
        return (r["pass"], *(np.nan_to_num(r.get(k, float("nan")), nan=-99) for k in order))
    ka, kb = key(a), key(b)
    return ka > kb


def pick_c8gate(rows: list[dict], c8_min: float, c1_floor: float, tiebreak: str = "c6,c1") -> dict | None:
    """pro-v5: candidato = C8 ≥ c8_min (isolamento real); entre candidatos, mais PASS, desempate `tiebreak`. Sem
    candidato: o de maior C8 entre os com C1 ≥ c1_floor (desempate pelo mesmo critério)."""
    def f(r, k):
        return float(np.nan_to_num(r.get(k, float("nan")), nan=-99))
    cand = [r for r in rows if f(r, "c8_pct") >= c8_min]
    if cand:
        best = None
        for r in cand:
            if better(r, best, tiebreak):
                best = r
        return best
    ok = [r for r in rows if f(r, "c1_db") >= c1_floor]
    if not ok:
        return None
    order = [TIEBREAK_KEYS[k.strip()] for k in tiebreak.split(",") if k.strip()]
    return max(ok, key=lambda r: (f(r, "c8_pct"), r["pass"], *(f(r, k) for k in order)))


def line(r: dict) -> str:
    c = r["crit"]
    return (f"passo {r['step']}: {r['pass']}/{r['n_crit']} PASS | C6 {r['c6_db']:+.1f} dB | C1 {r['c1_db']:+.2f}±{1.96 * r.get('c1_se', float('nan')):.2f} dB (IC95) | "
            + " ".join(f"{k}={'P' if p else 'F'}({s})" for k, (p, s) in c.items()) + f" | {r['sec']:.0f} s")


def cmd_watch(a) -> None:
    snap, best_dir, jl = Path(a.snap_dir), Path(a.best_dir), Path(a.jsonl)
    best_dir.mkdir(parents=True, exist_ok=True)
    bestp = best_dir / "best.json"
    best = json.loads(bestp.read_text()) if bestp.exists() else None
    done_steps = set()
    if jl.exists():
        done_steps = {json.loads(x)["step"] for x in jl.read_text().splitlines() if x.strip()}
    print(f"watch {snap} (pid {os.getpid()}); já avaliados: {sorted(done_steps)}", flush=True)
    fails: dict[int, int] = {}
    while True:
        pend = sorted(p for p in snap.glob("ckpt-*.pt") if int(p.stem.split("-")[1]) not in done_steps)
        if not pend:
            if a.done_file and Path(a.done_file).exists():
                print("fim: treino terminou e não há snapshots pendentes", flush=True)
                return
            time.sleep(30)
            continue
        p = pend[0]
        time.sleep(2)  # escrita atômica (tmp + rename) no save_checkpoint, mas folga
        step = int(p.stem.split("-")[1])
        try:
            r = evaluate(p, step, Path(a.cache))
        except Exception as e:  # noqa: BLE001
            fails[step] = fails.get(step, 0) + 1
            print(f"passo {step}: ERRO ({fails[step]}/3) {type(e).__name__}: {e}", flush=True)
            if fails[step] >= 3:  # desiste, mas mantém o snapshot para reavaliação manual
                done_steps.add(step)
            else:
                time.sleep(60)
            continue
        with open(jl, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        done_steps.add(step)
        print(line(r), flush=True)
        if a.select == "c8gate":  # recalcula sobre todas as linhas cujo snapshot ainda existe (ou é o best atual)
            rows = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
            avail = [x for x in rows if (snap / f"ckpt-{x['step']:08d}.pt").exists()
                     or (best is not None and x["step"] == best.get("step"))]
            nb = pick_c8gate(avail, a.c8_min, a.c1_floor, a.tiebreak)
            if nb is not None and (best is None or nb["step"] != best.get("step")):
                src = snap / f"ckpt-{nb['step']:08d}.pt"
                if src.exists():
                    shutil.copy2(src, best_dir / "best.pt")
                    best = dict(nb, best_ckpt=str(best_dir / "best.pt"), rule=f"c8>={a.c8_min} | c1>={a.c1_floor}")
                    bestp.write_text(json.dumps(best, indent=1))
                    print(f"  -> novo melhor (passo {nb['step']}, regra c8gate)", flush=True)
        elif better(r, best, a.tiebreak):
            shutil.copy2(p, best_dir / "best.pt")
            r2 = dict(r, best_ckpt=str(best_dir / "best.pt"))
            bestp.write_text(json.dumps(r2, indent=1))
            best = r2
            print(f"  -> novo melhor (passo {step})", flush=True)
        if not a.keep_all:
            p.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("one")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--cache", required=True)
    p.add_argument("--jsonl", default=None)
    p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("watch")
    p.add_argument("--snap-dir", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--best-dir", required=True)
    p.add_argument("--done-file", default=None)
    p.add_argument("--tiebreak", default="c8,c6,c1", help="desempate depois do nº de PASS (pro-v5: c6,c1)")
    p.add_argument("--select", choices=("default", "c8gate"), default="default",
                   help="c8gate (pro-v5): só C8 ≥ --c8-min é candidato; sem candidato, maior C8 com C1 ≥ --c1-floor")
    p.add_argument("--c8-min", type=float, default=85.0)
    p.add_argument("--c1-floor", type=float, default=-3.0)
    p.add_argument("--keep-all", action="store_true", help="não apaga snapshots avaliados")
    p.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    init_env(a.threads)
    if a.cmd == "one":
        r = evaluate(Path(a.ckpt), a.step, Path(a.cache))
        if a.jsonl:
            with open(a.jsonl, "a") as fh:
                fh.write(json.dumps(r) + "\n")
        print(line(r), flush=True)
        print(json.dumps({"bins": r["bins"], "v31_bins": r["v31_bins"], "v31_crit": r["v31_crit"]}, ensure_ascii=False))
    else:
        cmd_watch(a)


if __name__ == "__main__":
    main()
