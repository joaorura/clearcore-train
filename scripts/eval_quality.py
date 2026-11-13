#!/usr/bin/env python3
"""Métricas de qualidade do alvo (picote, instabilidade do ganho, ESTOI ativo) e de remoção de ruído, pelo caminho do
runtime, sobre v2_current / v31_new / keep_new e os modelos sem perfil.

Base: `tse_eval.quality_metrics` sobre a saída-sombra do alvo e do ruído (`runtime_path.run_replica(shadow=...)`).

Conjuntos:
  main   o mesmo recorte de `eval_soft_gate.select` (-30 dBFS, silencioso e SNR 5, N=0 e N=1..4 com SIR 0, matriz e
         usuário real take4, eixo temporal). Vistas: `mix` com perfil (KEY_GATING), `noprof` (KEY_noprof = DFNet3 base do
         próprio checkpoint, gating atual), `int` (só-interferente, com perfil) e `dfn3_noprof` (DFNet3 publicado) em N=0.
  noise  remoção de ruído: os alvos N=0 a -30 dBFS (usuário take4 silencioso, matriz silencioso/SNR 20/SNR 5: 48 alvos)
         re-renderizados com ruído do pool de dev em SNR 0, 5 e 10 dB, ruído por categoria em rodízio:
         office_0810 (mic_obs_3mic_combo_20261008, escritório de 08/10), local_mics (demais gravações locais),
         dns (porta/ventilador/cadeira: não estacionário), fsd50k. Sem interferente humano.

Uso: eval_quality.py run --set main|noise --out DIR [--workers 3]; eval_quality.py report --out DIR;
     eval_quality.py examples --out DIR (WAVs + CSV de dropouts do usuário take4, N=0 e N=1, v31 e keep).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cctrain.eval import tse_matrix as tm  # noqa: E402
from cctrain.paths import Paths  # noqa: E402

PATHS = Paths.from_env()
SET_DIR = PATHS.data / "eval-v4"
RUNS = PATHS.runs
MODELS = {  # chave -> (ckpt, arch, gating com perfil)
    "v2": (RUNS / "m2/pro-v2-dualpath-tse-20261008/checkpoints/ckpt-00075000.pt", "v2", "current"),
    "v31": (RUNS / "m4/pro-v4/checkpoints/ckpt-00079998.pt", "v4", "new"),
    "keep": (RUNS / "m4/pro-v4-keep/checkpoints/ckpt-00153093.pt", "v4", "new"),
}
NOISE_SNRS = (0.0, 5.0, 10.0)
NOISE_CATS = ("office_0810", "local_mics", "dns", "fsd50k")
FRAME_S = tm.HOP / tm.SR


def noise_cat(path: str) -> str:
    if "mic_obs_3mic_combo_20261008" in path:
        return "office_0810"
    if "noise-local-mics" in path:
        return "local_mics"
    if "dns-noise" in path:
        return "dns"
    return "fsd50k"


def select_main(recs: list[dict]) -> list[dict]:
    from eval_soft_gate import select

    return select(recs)


def noise_recipes(recs: list[dict]) -> list[dict]:
    """48 alvos N=0 a -30 dBFS x SNR 0/5/10, ruído do pool de dev (os arquivos de ruído das receitas) por categoria."""
    pool: dict[str, list[str]] = {c: [] for c in NOISE_CATS}
    for r in recs:
        if r["noise"]["kind"] == "file":
            p = r["noise"]["path"]
            c = noise_cat(p)
            if p not in pool[c]:
                pool[c].append(p)
    for c in pool:
        pool[c].sort()
    tg = [r for r in recs if r["cell"]["n"] == 0 and r["cell"]["level"] == -30.0
          and (r["cell"]["axis"] == "matrix" or (r["cell"]["axis"] == "user" and r["cell"]["noise"] == "silent"))]
    out = []
    for i, r in enumerate(tg):
        cat = NOISE_CATS[i % len(NOISE_CATS)]
        h = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16)
        path = pool[cat][h % len(pool[cat])]
        start = (h >> 64) % (2 * tm.SR)
        for snr in NOISE_SNRS:
            q = json.loads(json.dumps(r))
            src = r["cell"]["axis"]
            q["noise"] = {"kind": "file", "path": path, "snr_db": snr, "start": int(start)}
            q["cell"] = {"axis": "noise", "n": 0, "sir": None, "level": -30.0, "noise": f"snr{snr:.0f}", "pattern": cat}
            q["cell"]["cid"] = f"noise|n0|sirna|lv-30|snr{snr:.0f}|{cat}"
            q["id"] = f"{q['cell']['cid']}|{src}#{r['idx']:03d}"
            q["src_axis"] = src
            out.append(q)
    return out


_W: dict = {}


def _init(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    from cctrain.eval import tse_eval as te
    from cctrain.eval.dnsmos import DnsmosScorer

    tm.set_roots(PATHS.data, SET_DIR / "user_take4_stream_48k.wav")
    _W["bank"] = tm.Bank.from_json(json.loads((SET_DIR / "bank.json").read_text()), np.load(SET_DIR / "bank_emb.npy"))
    _W["user_emb"] = np.load(SET_DIR / "user_emb.npy")
    _W["runners"] = {k: te.Runner(ck, arch, g, "cpu") for k, (ck, arch, g) in MODELS.items()}
    _W["dfn3"] = te.BaseRunner()
    _W["te"] = te
    _W["scorer"] = DnsmosScorer.from_dir(PATHS.data / "models" / "dnsmos", threads=1)


def profile_of(rec: dict) -> np.ndarray:
    p = rec["profile"]
    return _W["user_emb"][p["row"]] if p["kind"] == "user" else _W["bank"].emb[_W["bank"].idx(p["spk"])]


def _work(chunk: list[dict]) -> list[dict]:
    te, sc = _W["te"], _W["scorer"]
    rows: list[dict] = []
    for rec in chunk:
        t0 = time.time()
        c = rec["cell"]
        r = tm.render(rec)
        noisy = r.noisy(True)
        base = te.quality_baseline(r, noisy, te.input_baseline(r, noisy))
        comps = [r.target, r.noise]
        prof = profile_of(rec)
        got = []
        from cctrain.data.resample import to_16k

        got.append(("input", "input", {f"dnsmos_{k}": v for k, v in sc.score(to_16k(noisy, tm.SR)).items()}
                    | {"estoi_in_act": base.get("estoi_in_act", float("nan")), "sisdr_in": base["sisdr_in"]}))
        for mk, rn in _W["runners"].items():
            enh, st, sh = rn.run(noisy, prof, shadow=comps)
            got.append((f"{mk}_{MODELS[mk][2]}", "mix", te.item_metrics("mix", rec, r, noisy, enh, st, base, sh, sc)))
            enh, st, sh = rn.run(noisy, None, shadow=comps)
            got.append((f"{mk}_noprof", "noprof", te.item_metrics("noprof", rec, r, noisy, enh, st, base, sh, sc)))
            if c["axis"] != "noise" and c["n"] > 0 and c["axis"] != "temporal":
                ni = r.noisy(False)
                enh, st = rn.run(ni, prof)
                got.append((f"{mk}_{MODELS[mk][2]}", "int", te.item_metrics("int", rec, r, ni, enh, st, base)))
        if c["n"] == 0:
            enh, st, sh = _W["dfn3"].run(noisy, None, shadow=comps)
            got.append(("dfn3_noprof", "noprof", te.item_metrics("noprof", rec, r, noisy, enh, st, base, sh, sc)))
        dt = time.time() - t0
        for name, view, m in got:
            rows.append({"id": rec["id"], "cid": c["cid"], "view": view, "variant": name, "target": rec["target"]["spk"],
                         "src_axis": rec.get("src_axis", c["axis"]), "sec": round(dt / len(got), 3), **m})
    return rows


def all_recipes() -> list[dict]:
    return tm.read_recipes(SET_DIR / "recipes.jsonl")


def cmd_run(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = all_recipes()
    recs = select_main(recs) if a.set == "main" else noise_recipes(recs)
    if a.limit:
        recs = recs[: a.limit]
    rows_path = out / f"rows-{a.set}.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(line)["id"] for line in rows_path.read_text().splitlines() if line.strip()}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{a.set}: {len(recs)} itens, {len(todo)} pendentes, {a.workers} processos", flush=True)
    meta = {"set": a.set, "models": {k: [str(c), ar, g] for k, (c, ar, g) in MODELS.items()}, "items": len(recs),
            "recipes_sha256": tm.recipes_digest(SET_DIR / "recipes.jsonl"), "started": time.strftime("%F %T")}
    t0 = time.time()
    chunks = [todo[i : i + a.chunk] for i in range(0, len(todo), a.chunk)]
    if chunks:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                 initargs=(a.threads,)) as ex, open(rows_path, "a") as fh:
            futs = [ex.submit(_work, c) for c in chunks]
            for n, f in enumerate(as_completed(futs), 1):
                for row in f.result():
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                if n % 10 == 0:
                    print(f"{n}/{len(chunks)} blocos ({time.time() - t0:.0f} s)", flush=True)
    meta["seconds"] = time.time() - t0
    (out / f"meta-{a.set}.json").write_text(json.dumps(meta, indent=1))
    print(f"pronto em {meta['seconds']:.0f} s", flush=True)


# --- agregação -----------------------------------------------------------------------------------------------------


def cellinfo(cid: str) -> dict:
    axis, n, sir, lv, noise, pat = cid.split("|")
    return {"axis": axis, "n": int(n[1:]), "sir": None if sir[3:] == "na" else float(sir[3:]), "noise": noise,
            "pattern": pat}


def pick(rows, variant, view, **sel):
    out = []
    for r in rows:
        if r["variant"] != variant or r["view"] != view:
            continue
        c = cellinfo(r["cid"]) | {"src": r.get("src_axis")}
        if all((c[k] in v if isinstance(v, (tuple, set, list)) else c[k] == v) for k, v in sel.items()):
            out.append(r)
    return out


def _m(rs, k):
    v = np.array([r[k] for r in rs if r.get(k) is not None], float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _sum(rs, k):
    return float(sum(r.get(k, 0) or 0 for r in rs))


def qagg(rs: list[dict]) -> dict:
    """Agregado de qualidade: contagens somadas (taxas sobre o tempo total de fala ativa), demais médias por item."""
    act = _sum(rs, "q_act_frames")
    mins = act * FRAME_S / 60.0
    nd = _sum(rs, "q_drop_n")
    a = {"n": len(rs), "speech_min": mins,
         "drop_per_min": nd / mins if mins else float("nan"),
         "drop_pct": 100 * _sum(rs, "q_drop_frames") / act if act else float("nan"),
         "long_pct": 100 * _sum(rs, "q_long_frames") / act if act else float("nan"),
         "drop_dur_ms": 1000 * FRAME_S * _sum(rs, "q_drop_frames") / nd if nd else float("nan")}
    for k in ("q_g_p10", "q_g_med", "q_g_std", "q_g_dstep", "d_estoi_act", "estoi_act", "d_estoi", "d_stoi", "d_sisdr",
              "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "n_att_sil_mean", "n_att_sil_p10", "n_att_speech",
              "snr_out_speech", "snr_in_speech", "target_gain_hi_db", "interf_supp_db"):
        a[k] = _m(rs, k)
    return a


def seg_io(rows, variant, **sel) -> float:
    rs = pick(rows, variant, "int", **sel)
    d = _sum(rs, "seg_io_n")
    return 100 * _sum(rs, "seg_io_ge15") / d if d else float("nan")


def paired(rows, variant, ref_variant, keys, view_a="mix", view_b="noprof", **sel) -> dict:
    """Média pareada por item de métrica(variant, view_a) - métrica(ref, view_b)."""
    b = {r["id"]: r for r in pick(rows, ref_variant, view_b, **sel)}
    out = {}
    for k in keys:
        d = [r[k] - b[r["id"]][k] for r in pick(rows, variant, view_a, **sel)
             if r["id"] in b and r.get(k) is not None and b[r["id"]].get(k) is not None]
        d = np.array(d, float)
        d = d[np.isfinite(d)]
        out[k] = float(d.mean()) if d.size else float("nan")
        out[k + "_se"] = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
    return out


def f(x, fmt="{:+.2f}"):
    return "-" if x is None or not np.isfinite(x) else fmt.format(x)


PROF = ("v2_current", "v31_new", "keep_new")
NOPROF = ("v2_noprof", "v31_noprof", "keep_noprof", "dfn3_noprof")
MODEL_OF = {"v2_current": "v2", "v31_new": "v31", "keep_new": "keep"}


def variant_rows(rows, **sel):
    """[(nome exibido, variante, vista)]: com perfil (mix) e sem perfil (noprof) de cada modelo."""
    out = [(v, v, "mix") for v in PROF]
    out += [(v, v, "noprof") for v in NOPROF if pick(rows, v, "noprof", **sel)]
    return out


QHDR = ("| variante | itens | fala (min) | dropouts/min | % fala em dropout | dur. média (ms) | % supressão longa (>300 ms) | "
        "p10 ganho (dB) | mediana ganho | desvio ganho | Δ quadro a quadro | ESTOI ativo | ΔESTOI ativo | ΔESTOI global | "
        "ΔSI-SDR | DNSMOS OVRL |\n|---|" + "---|" * 15)


def qline(name, a) -> str:
    return (f"| {name} | {a['n']} | {f(a['speech_min'], '{:.1f}')} | {f(a['drop_per_min'], '{:.1f}')} | {f(a['drop_pct'], '{:.1f}')} | "
            f"{f(a['drop_dur_ms'], '{:.0f}')} | {f(a['long_pct'], '{:.1f}')} | {f(a['q_g_p10'], '{:+.1f}')} | "
            f"{f(a['q_g_med'], '{:+.1f}')} | {f(a['q_g_std'], '{:.1f}')} | {f(a['q_g_dstep'], '{:.2f}')} | "
            f"{f(a['estoi_act'], '{:.3f}')} | {f(a['d_estoi_act'], '{:+.3f}')} | {f(a['d_estoi'], '{:+.3f}')} | "
            f"{f(a['d_sisdr'], '{:+.1f}')} | {f(a['dnsmos_ovrl'], '{:.2f}')} |")


MAIN_GROUPS = [
    ("Usuário real (take4) N=0, silencioso", dict(axis="user", n=0, noise="silent")),
    ("Usuário real (take4) N=0, SNR 5", dict(axis="user", n=0, noise="snr5")),
    ("Usuário real (take4) N=1, SIR 0, silencioso", dict(axis="user", n=1, noise="silent")),
    ("Usuário real (take4) N=1, SIR 0, SNR 5", dict(axis="user", n=1, noise="snr5")),
    ("Usuário real (take4) N=1..4, SIR 0, silencioso + SNR 5", dict(axis="user", n=(1, 2, 3, 4))),
    ("Matriz N=0, silencioso", dict(axis="matrix", n=0, noise="silent")),
    ("Matriz N=0, SNR 5", dict(axis="matrix", n=0, noise="snr5")),
    ("Matriz N=1, SIR 0, silencioso + SNR 5", dict(axis="matrix", n=1)),
    ("Matriz N=1..4, SIR 0, silencioso + SNR 5", dict(axis="matrix", n=(1, 2, 3, 4))),
    ("Temporal (blocos/conversa, N=1 e 3)", dict(axis="temporal")),
]


def criteria(rows: list[dict], nrows: list[dict] | None) -> tuple[list[str], dict]:
    """PASS/FAIL por variante com perfil. Limiares em `CRIT` (justificativa no relatório)."""
    res: dict = {}
    u0 = dict(axis="user", n=0)
    u1 = dict(axis="user", n=1)
    v31_u1 = qagg(pick(rows, "v31_new", "mix", **u1))
    for v in PROF:
        mk = MODEL_OF[v]
        c = {}
        dmg = paired(rows, v, f"{mk}_noprof", ["d_sisdr"], axis="user", n=0, noise="silent")["d_sisdr"]
        c["C1 dano N=0 usuário silencioso (ΔSI-SDR perfil−sem) ≥ −1 dB"] = (dmg >= -1.0, f"{dmg:+.2f} dB")
        a0, b0 = qagg(pick(rows, v, "mix", **u0)), qagg(pick(rows, f"{mk}_noprof", "noprof", **u0))
        c["C2 dropouts/min N=0 usuário ≤ sem perfil + 1"] = (a0["drop_per_min"] <= b0["drop_per_min"] + 1.0,
                                                             f"{a0['drop_per_min']:.1f} vs {b0['drop_per_min']:.1f}+1")
        c["C3 p10 ganho N=0 usuário ≥ sem perfil − 3 dB"] = (a0["q_g_p10"] >= b0["q_g_p10"] - 3.0,
                                                           f"{a0['q_g_p10']:+.1f} vs {b0['q_g_p10']:+.1f}−3")
        a1 = qagg(pick(rows, v, "mix", **u1))
        c["C4 dropouts/min N=1 usuário ≤ v31 + 1"] = (a1["drop_per_min"] <= v31_u1["drop_per_min"] + 1.0,
                                                      f"{a1['drop_per_min']:.1f} vs {v31_u1['drop_per_min']:.1f}+1")
        c["C5 % fala em dropout N=1 usuário ≤ 5%"] = (a1["drop_pct"] <= 5.0, f"{a1['drop_pct']:.1f}%")
        c["C6 ΔESTOI ativo N=1 usuário ≥ v31 − 0,01"] = (a1["d_estoi_act"] >= v31_u1["d_estoi_act"] - 0.01,
                                                        f"{a1['d_estoi_act']:+.3f} vs {v31_u1['d_estoi_act']:+.3f}")
        s = seg_io(rows, v, axis="user", n=(1, 2, 3, 4))
        c["C7 seg só-int ≥15 dB (usuário N=1..4, vista int) ≥ 90%"] = (s >= 90.0, f"{s:.0f}%")
        if nrows:
            for snr in NOISE_SNRS:
                sel = dict(axis="noise", noise=f"snr{snr:.0f}")
                a, ref = qagg(pick(nrows, v, "mix", **sel)), qagg(pick(nrows, "v2_current", "mix", **sel))
                c[f"R{snr:.0f} ruído SNR {snr:.0f}: ΔSI-SDR e atenuação (silêncio) ≥ v2_current − 1 dB"] = (
                    a["d_sisdr"] >= ref["d_sisdr"] - 1 and a["n_att_sil_mean"] >= ref["n_att_sil_mean"] - 1,
                    f"{a['d_sisdr']:+.1f}/{a['n_att_sil_mean']:.1f} vs {ref['d_sisdr']:+.1f}/{ref['n_att_sil_mean']:.1f}")
        res[v] = c
    names = list(next(iter(res.values())).keys())
    L = ["| critério | " + " | ".join(PROF) + " |", "|---|" + "---|" * len(PROF)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{'PASS' if res[v][n][0] else 'FAIL'} ({res[v][n][1]})" for v in PROF) + " |")
    L.append("| **total** | " + " | ".join(f"{sum(x[0] for x in res[v].values())}/{len(names)}" for v in PROF) + " |")
    return L, res


def build_report(out: Path) -> str:
    rows = [json.loads(x) for x in (out / "rows-main.jsonl").read_text().splitlines() if x.strip()]
    np_ = out / "rows-noise.jsonl"
    nrows = [json.loads(x) for x in np_.read_text().splitlines() if x.strip()] if np_.exists() else None
    L = ["# Qualidade do alvo (picote, instabilidade, inteligibilidade) e remoção de ruído", "",
         f"- conjunto principal: {len({r['id'] for r in rows})} itens (mesmo recorte do relatório "
         "`runs/m4/pro-v4-keep/eval-ckpt-00153093`); remoção de ruído: "
         f"{len({r['id'] for r in nrows}) if nrows else 0} itens", ""]
    L += [f"- `{k}_{g}`: `{c}` (arch {ar}, gating {g}); `{k}_noprof` = DFNet3 base do mesmo checkpoint, sem perfil, gating atual"
          for k, (c, ar, g) in MODELS.items()]
    L += ["- `dfn3_noprof` = DFNet3 publicado (upstream, `load_upstream_dfnet3`), sem perfil, gating atual (só em N=0 e ruído)", ""]
    L += ["## Definições e limiares", "",
          "- **Ganho do alvo por quadro** (10 ms): 10·log10(E_out/E_in) da **saída-sombra do alvo** — o alvo limpo processado "
          "pelas mesmas decisões (estágio, máscara ERB, coeficientes DF) que o modelo tomou sobre a mistura, na réplica do "
          "runtime. Fixadas as decisões o processamento é linear: alvo-sombra + ruído-sombra + interferente-sombra = saída "
          "(erro medido −136 dB), então o ganho é do alvo, sem viés de projeção. Limitado a [−60, +20] dB.",
          "- **Quadro ativo**: nível do alvo ≥ máximo local (±500 ms) − 30 dB e > −75 dBFS. Local e não pelo pico do item, "
          "para não descartar frases mais baixas.",
          "- **Quadro caído**: ativo e ganho ≤ max(mediana do item − 10 dB, −10 dB). 10 dB = metade da sonoridade percebida, "
          "um corte audível; o piso absoluto −10 dB pega o item que já sai inteiro atenuado.",
          "- **Dropout**: trecho contíguo de quadros caídos (um quadro inativo interrompe) com 20–300 ms (2–30 quadros). "
          "1 quadro (10 ms) é ignorado: abaixo da resolução da STFT de 20 ms do modelo. > 300 ms é **supressão longa** "
          "(apagamento de palavra/frase, não picote), reportada à parte.",
          "- **dropouts/min**: dropouts ÷ minutos de fala ativa (somados no recorte); **% fala em dropout**: quadros em "
          "dropout ÷ quadros ativos; **dur. média**; **p10**: percentil 10 do ganho por quadro nos quadros ativos (média dos "
          "itens); **desvio** e **Δ quadro a quadro** (média de |g[t]−g[t−1]| entre quadros ativos consecutivos): "
          "instabilidade/tremulação do ganho (média dos itens).",
          "- **ESTOI ativo**: ESTOI (pystoi, 16 kHz) sobre a concatenação das amostras dos quadros ativos (saída real vs. alvo "
          "limpo); Δ = saída − mistura de entrada na mesma máscara. Exige ≥ 1 s de fala ativa.",
          "- **DNSMOS P.835 OVRL** (local, `data/models/dnsmos`, sem rede): qualidade perceptual não intrusiva da saída. "
          "**PESQ não medido**: a lib `pesq` não está instalada no venv (não instalada, sem rede). Nenhuma métrica "
          "perceptual de inteligibilidade não intrusiva disponível localmente.",
          "- Ruído: **atenuação no silêncio** = 10·log10(E_ruído_in/E_ruído-sombra_out) por quadro nos quadros sem alvo "
          "(média e p10; limitado a 60 dB); **atenuação durante a fala** = mesma razão somada nos quadros ativos do alvo "
          "(ruído residual pela sombra do ruído); **SNR de saída na fala** = alvo-sombra ÷ ruído-sombra nesses quadros.",
          ""]
    L += ["## Critérios de aceitação", ""]
    cl, _ = criteria(rows, nrows)
    L += cl + [""]
    for title, sel in MAIN_GROUPS:
        L += [f"## {title}", "", QHDR]
        for name, v, view in variant_rows(rows, **sel):
            rs = pick(rows, v, view, **sel)
            if rs:
                L.append(qline(name, qagg(rs)))
        if sel.get("n") != 0 and sel.get("axis") != "temporal":
            L.append("")
            L.append("seg só-int ≥15 dB (vista int): " + ", ".join(f"{v} {f(seg_io(rows, v, **sel), '{:.0f}')}%" for v in PROF))
        L.append("")
    L += ["## N=0: dano do perfil (com perfil − sem perfil do mesmo checkpoint, pareado por item)", "",
          "| variante | recorte | Δ ΔSI-SDR | Δ ΔESTOI ativo | Δ p10 ganho | Δ mediana ganho | dropouts/min perfil / sem | % dropout perfil / sem |",
          "|---|---|---|---|---|---|---|---|"]
    for v in PROF:
        mk = MODEL_OF[v]
        for lab, sel in (("usuário silencioso", dict(axis="user", n=0, noise="silent")),
                         ("usuário SNR 5", dict(axis="user", n=0, noise="snr5")),
                         ("matriz silencioso", dict(axis="matrix", n=0, noise="silent")),
                         ("matriz SNR 5", dict(axis="matrix", n=0, noise="snr5"))):
            d = paired(rows, v, f"{mk}_noprof", ["d_sisdr", "d_estoi_act", "q_g_p10", "q_g_med"], **sel)
            a, b = qagg(pick(rows, v, "mix", **sel)), qagg(pick(rows, f"{mk}_noprof", "noprof", **sel))
            L.append(f"| {v} | {lab} | {f(d['d_sisdr'])} ± {f(d['d_sisdr_se'], '{:.2f}')} | {f(d['d_estoi_act'], '{:+.3f}')} | "
                     f"{f(d['q_g_p10'], '{:+.1f}')} | {f(d['q_g_med'], '{:+.1f}')} | {f(a['drop_per_min'], '{:.1f}')} / "
                     f"{f(b['drop_per_min'], '{:.1f}')} | {f(a['drop_pct'], '{:.1f}')} / {f(b['drop_pct'], '{:.1f}')} |")
    L.append("")
    if nrows:
        L += noise_section(nrows)
    return "\n".join(L) + "\n"


NHDR = ("| variante | itens | ΔSI-SDR | ΔESTOI ativo | atenuação ruído silêncio média (dB) | p10 | atenuação ruído na fala | "
        "SNR saída na fala (entrada) | dropouts/min | % dropout | p10 ganho alvo | DNSMOS SIG/BAK/OVRL |\n|---|" + "---|" * 11)


def nline(name, a, a_in=None) -> str:
    return (f"| {name} | {a['n']} | {f(a['d_sisdr'], '{:+.1f}')} | {f(a['d_estoi_act'], '{:+.3f}')} | "
            f"{f(a['n_att_sil_mean'], '{:.1f}')} | {f(a['n_att_sil_p10'], '{:.1f}')} | {f(a['n_att_speech'], '{:.1f}')} | "
            f"{f(a['snr_out_speech'], '{:+.1f}')} ({f(a['snr_in_speech'], '{:+.1f}')}) | {f(a['drop_per_min'], '{:.1f}')} | "
            f"{f(a['drop_pct'], '{:.1f}')} | {f(a['q_g_p10'], '{:+.1f}')} | "
            f"{f(a['dnsmos_sig'], '{:.2f}')}/{f(a['dnsmos_bak'], '{:.2f}')}/{f(a['dnsmos_ovrl'], '{:.2f}')} |")


def noise_section(nrows) -> list[str]:
    L = ["## Remoção de ruído (alvo + ruído, sem interferente humano)", "",
         "48 alvos N=0 a −30 dBFS (12 do usuário take4, 36 da matriz) × SNR 0/5/10 dB; ruído do pool de dev em rodízio: "
         "`office_0810` = gravação do escritório de 08/10 (`mic_obs_3mic_combo_20261008`), `local_mics` = demais gravações "
         "locais (06–07/10), `dns` = porta/ventilador/cadeira (não estacionário), `fsd50k`. Com perfil = vista `mix`; sem "
         "perfil = DFNet3 base de cada checkpoint; `dfn3_noprof` = DFNet3 publicado.", ""]
    for snr in NOISE_SNRS:
        sel = dict(axis="noise", noise=f"snr{snr:.0f}")
        a_in = qagg(pick(nrows, "input", "input", **sel))
        L += [f"### SNR {snr:.0f} dB (entrada: DNSMOS OVRL {f(a_in['dnsmos_ovrl'], '{:.2f}')})", "", NHDR]
        for name, v, view in variant_rows(nrows, **sel):
            L.append(nline(name, qagg(pick(nrows, v, view, **sel))))
        L.append("")
    L += ["### Por categoria de ruído (SNR 0/5/10 juntos): ΔSI-SDR / atenuação no silêncio (dB) / dropouts/min", "",
          "| variante | " + " | ".join(NOISE_CATS) + " |", "|---|" + "---|" * len(NOISE_CATS)]
    for name, v, view in variant_rows(nrows, axis="noise"):
        cells = []
        for cat in NOISE_CATS:
            a = qagg(pick(nrows, v, view, axis="noise", pattern=cat))
            cells.append(f"{f(a['d_sisdr'], '{:+.1f}')} / {f(a['n_att_sil_mean'], '{:.1f}')} / {f(a['drop_per_min'], '{:.1f}')}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def cmd_report(a) -> None:
    out = Path(a.out)
    md = build_report(out)
    (out / "report.md").write_text(md)
    print(f"relatório em {out / 'report.md'}")


# --- exemplos ------------------------------------------------------------------------------------------------------


def cmd_examples(a) -> None:
    """Usuário take4: N=0 silencioso e N=1 SIR 0 silencioso, o item com mais dropouts no keep (e o mesmo para o v31).
    Salva mistura, alvo limpo, saída de v31/keep e CSV com os dropouts (início, fim, duração, ganho mínimo, limiar)."""
    import soundfile as sf

    from cctrain.eval import tse_eval as te

    out = Path(a.out)
    ex = out / "examples"
    ex.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (out / "rows-main.jsonl").read_text().splitlines() if x.strip()]
    _init(1)
    recs = {r["id"]: r for r in all_recipes()}
    picks = []
    for sel in (dict(axis="user", n=0, noise="silent"), dict(axis="user", n=1, noise="silent")):
        rs = sorted(pick(rows, "keep_new", "mix", **sel), key=lambda r: -(r.get("q_drop_n", 0)))
        picks += [r["id"] for r in rs[:2]]
    lines = []
    for iid in picks:
        rec = recs[iid]
        r = tm.render(rec)
        noisy = r.noisy(True)
        tag = iid.replace("|", "_").replace("#", "_")
        sf.write(ex / f"{tag}__mix.wav", noisy, tm.SR)
        sf.write(ex / f"{tag}__target.wav", r.target, tm.SR)
        prof = profile_of(rec)
        nf = len(r.target) // tm.HOP
        act = te.target_active_local(r.target, nf)
        for mk in ("v31", "keep"):
            enh, _st, sh = _W["runners"][mk].run(noisy, prof, shadow=[r.target, r.noise])
            name = f"{mk}_{MODELS[mk][2]}"
            sf.write(ex / f"{tag}__{name}.wav", enh, tm.SR)
            d = te.dropouts(te.frame_gain_db(r.target, sh[0], nf), act)
            for s, e, gmin in d["events"]:
                lines.append([f"{tag}__{name}.wav", f"{s * FRAME_S:.2f}", f"{e * FRAME_S:.2f}", f"{(e - s) * 10}",
                              f"{gmin:+.1f}", f"{d['thr']:+.1f}", f"{d['med']:+.1f}"])
            print(f"{tag} {name}: {len(d['events'])} dropouts, mediana {d['med']:+.1f} dB, p10 {d['p10']:+.1f}", flush=True)
    with open(ex / "dropouts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arquivo", "inicio_s", "fim_s", "duracao_ms", "ganho_min_db", "limiar_db", "mediana_item_db"])
        w.writerows(lines)
    print(f"exemplos em {ex}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--set", choices=["main", "noise"], default="main")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--chunk", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("examples")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_examples)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
