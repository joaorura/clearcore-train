#!/usr/bin/env python3
"""Áudios de escuta v3.1 x smoke C (pro-v42) no usuário take4 (held-out, local_only: não sai da máquina).

Itens: 2 N=0 silencioso; 3 N=1 em turnos (receitas do usuário N=1 silencioso com janelas alternadas alvo/interferente e
pausas curtas; 2 com maior diferença de ganho alvo-sozinho v3.1→smokeC e 1 típica); 1 N=1 sobreposto; 1 N=0 SNR 5
(ruído de escritório). Para cada item: mixture, v31, smokeC, sem-perfil (v31 sem perfil), alvo limpo — mesmo ganho para o
conjunto do item (limitado a −1 dBFS de pico) — e AB_<item>.wav (mixture, bip, v31, bip duplo, smokeC). Escreve LEIAME.md.
Uso: make_listen_v42.py --smoke CKPT --out DIR"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import eval_quality as eq  # noqa: E402
import eval_quick as qk  # noqa: E402
from cctrain.eval import tse_matrix as tm  # noqa: E402

SR = tm.SR
PEAK = 10 ** (-1 / 20)
TURNS = [  # (janelas do alvo, janelas do interferente) em s, item de 5 s: o interferente abre, pausas de 0,1–0,4 s
    ([(1.3, 2.7), (3.6, 5.0)], [(0.0, 1.1), (2.9, 3.4)]),
    ([(1.0, 2.2), (3.3, 5.0)], [(0.0, 0.8), (2.5, 3.1)]),
    ([(1.6, 3.0), (3.8, 5.0)], [(0.0, 1.4), (3.2, 3.6)]),
]


def turns_recipe(rec: dict, k: int) -> dict:
    q = json.loads(json.dumps(rec))
    tw, iw = TURNS[k % len(TURNS)]
    q["target"]["windows"] = [list(w) for w in tw]
    for it in q["interferers"]:
        it["windows"] = [list(w) for w in iw]
    q["id"] = rec["id"] + f"|turns{k % len(TURNS)}"
    q["cell"] = dict(q["cell"], pattern="turns")
    return q


def run_item(rec: dict, runners: dict) -> dict:
    te = eq._W["te"]
    r = tm.render(rec)
    noisy = r.noisy(True)
    base = te.quality_baseline(r, noisy, te.input_baseline(r, noisy))
    prof = eq.profile_of(rec)
    out = {"rec": rec, "r": r, "noisy": noisy, "wav": {}, "m": {}}
    for name, rn in runners.items():
        enh, st, sh = rn.run(noisy, prof, shadow=[r.target, r.noise])
        out["wav"][name] = enh
        m = te.item_metrics("mix", rec, r, noisy, enh, st, base, sh, None)
        if rec["cell"]["n"] > 0:
            ni = r.noisy(False)
            e2, s2 = rn.run(ni, prof)
            mi = te.item_metrics("int", rec, r, ni, e2, s2, base)
            m["seg_io_pct"] = 100 * mi.get("seg_io_ge15", 0) / mi["seg_io_n"] if mi.get("seg_io_n") else float("nan")
        out["m"][name] = m
    enh, _st = runners["v31"].run(noisy, None)
    out["wav"]["semperfil"] = enh
    return out


def tonly(m: dict) -> float:
    return m.get("q_tonly_g_med", float("nan"))


def dpm(m: dict) -> float:
    a = m.get("q_act_frames", 0)
    return m.get("q_drop_n", 0) / (a * eq.FRAME_S / 60) if a else float("nan")


def events(r) -> list[str]:
    """Momentos 'o interferente para e o alvo começa' (≤ 0,6 s depois) e início do alvo."""
    ta, ia = r.tgt_act, r.int_act
    ev = []
    nf = len(ta)
    for f in range(1, nf):
        if ia[f - 1] and not ia[f]:
            nxt = np.flatnonzero(ta[f: f + 60])
            if nxt.size:
                ev.append(f"aos {f * eq.FRAME_S:.1f} s a outra pessoa para e o usuário fala em {(f + nxt[0]) * eq.FRAME_S:.1f} s")
    return ev[:3]


def beep(n: int) -> np.ndarray:
    t = np.arange(int(0.15 * SR)) / SR
    b = 0.1 * np.sin(2 * np.pi * 1000 * t) * np.hanning(t.size)
    gap = np.zeros(int(0.1 * SR))
    return np.concatenate([b, gap, b]) if n == 2 else b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qk.init_env(4)
    te = eq._W["te"]
    v31 = eq.MODELS["v31"]
    runners = {"v31": te.Runner(v31[0], v31[1], v31[2], "cpu"), "smokeC": te.Runner(Path(a.smoke), "v4", "new", "cpu")}
    recs = eq.all_recipes()
    u = [r for r in recs if r["cell"]["axis"] == "user" and r["cell"]["level"] == -30.0]
    n0s = [r for r in u if r["cell"]["n"] == 0 and r["cell"]["noise"] == "silent"]
    n1s = [r for r in u if r["cell"]["n"] == 1 and r["cell"]["noise"] == "silent" and r["cell"].get("sir") == 0.0]
    office = [r for r in eq.noise_recipes(recs) if r["cell"]["noise"] == "snr5" and r["src_axis"] == "user"
              and r["cell"]["pattern"] == "office_0810"]
    # turnos: varre os 12 primeiros N=1 silenciosos em turnos e escolhe pela diferença de ganho alvo-sozinho
    cand = []
    for k, rec in enumerate(n1s[:12]):
        res = run_item(turns_recipe(rec, k), runners)
        d = tonly(res["m"]["smokeC"]) - tonly(res["m"]["v31"])
        cand.append((d, k, res))
        print(f"turnos {k}: v31 {tonly(res['m']['v31']):+.1f} smokeC {tonly(res['m']['smokeC']):+.1f} Δ {d:+.1f}", flush=True)
    cand = [c for c in cand if np.isfinite(c[0])]
    cand.sort(key=lambda c: -c[0])
    med = float(np.median([c[0] for c in cand]))
    typ = min(cand[2:], key=lambda c: abs(c[0] - med))
    items = [("n0_silencioso_a", "N=0 silencioso", run_item(n0s[0], runners)),
             ("n0_silencioso_b", "N=0 silencioso", run_item(n0s[5], runners)),
             ("n1_turnos_a", "N=1 turnos (maior Δ)", cand[0][2]),
             ("n1_turnos_b", "N=1 turnos (2º maior Δ)", cand[1][2]),
             ("n1_turnos_tipico", "N=1 turnos (Δ típico)", typ[2])]
    # sobreposição: N=1 silencioso padrão com mais quadros dos dois ao mesmo tempo
    best, bov = None, -1.0
    for rec in n1s[:12]:
        r = tm.render(rec)
        ov = float((r.tgt_act & r.int_act).mean())
        if ov > bov:
            best, bov = rec, ov
    items.append(("n1_sobreposto", f"N=1 sobreposto ({100 * bov:.0f}% dos quadros com os dois)", run_item(best, runners)))
    items.append(("n0_ruido_snr5", "N=0 ruído de escritório SNR 5", run_item(office[0], runners)))

    rows = []
    for tag, scen, res in items:
        r = res["r"]
        sigs = {"mixture": res["noisy"], "v31": res["wav"]["v31"], "smokeC": res["wav"]["smokeC"],
                "semperfil": res["wav"]["semperfil"], "alvo": r.target}
        pk = max(float(np.max(np.abs(x))) for x in sigs.values())
        g = PEAK / max(pk, 1e-9)
        for k, x in sigs.items():
            sf.write(out / f"{tag}__{k}.wav", (np.asarray(x, np.float64) * g).astype(np.float32), SR, subtype="FLOAT")
        sil = np.zeros(int(0.5 * SR))
        ab = np.concatenate([sigs["mixture"] * g, sil, beep(1), sil, sigs["v31"] * g, sil, beep(2), sil,
                             sigs["smokeC"] * g])
        sf.write(out / f"AB_{tag}.wav", np.clip(ab, -1, 1).astype(np.float32), SR, subtype="FLOAT")
        mv, ms = res["m"]["v31"], res["m"]["smokeC"]
        rows.append({"tag": tag, "scen": scen, "id": res["rec"]["id"], "dur": len(r.target) / SR, "gain_db": 20 * np.log10(g),
                     "tonly_v31": tonly(mv), "tonly_c": tonly(ms), "gmed_v31": mv.get("q_g_med"),
                     "gmed_c": ms.get("q_g_med"), "dpm_v31": dpm(mv), "dpm_c": dpm(ms),
                     "io_v31": mv.get("seg_io_pct"), "io_c": ms.get("seg_io_pct"), "ev": events(r)})
    (out / "itens.json").write_text(json.dumps(rows, indent=1, default=float))
    print(json.dumps(rows, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
