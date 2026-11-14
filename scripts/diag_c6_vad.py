#!/usr/bin/env python3
"""Ressalva do C6: refaz o ganho do alvo nos quadros "alvo sozinho" por tempo desde o interferente com VADs do
interferente mais permissivos (pico-30 dB, pico-45 dB, pico-45 dB + 100 ms após o fim excluídos). Só v3.1, gating new,
usuário take4 N=1 silencioso (mesma seleção do diag_c6). Uso: diag_c6_vad.py [--model v31] [--ckpt PATH]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_quality as eq  # noqa: E402
from cctrain.eval import tse_eval as te  # noqa: E402
from cctrain.eval import tse_matrix as tm  # noqa: E402

BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9))
VARIANTS = {"pico-30": (30.0, 0), "pico-45": (45.0, 0), "pico-30+100ms": (30.0, 10), "pico-45+100ms": (45.0, 10)}


def dilate_fwd(m: np.ndarray, k: int) -> np.ndarray:
    out = m.copy()
    for s in range(1, k + 1):
        out[s:] |= m[:-s]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v31")
    a = ap.parse_args()
    for k in list(eq.MODELS):
        if k != a.model:
            eq.MODELS.pop(k)
    eq._init(4)
    recs = [r for r in eq.all_recipes() if r["cell"]["axis"] == "user" and r["cell"]["n"] == 1
            and r["cell"]["level"] == -30.0 and r["cell"]["noise"] == "silent" and r["cell"].get("sir") == 0.0]
    acc = {v: {"g": [], "since": []} for v in VARIANTS}
    rn = eq._W["runners"][a.model]
    for rec in recs:
        r = tm.render(rec)
        noisy = r.noisy(True)
        nf = len(r.target) // tm.HOP
        act = te.target_active_local(r.target, nf)
        enh, st, sh = rn.run(noisy, eq.profile_of(rec), shadow=[r.target])
        g = te.frame_gain_db(r.target, sh[0], nf)
        for v, (rel, ext) in VARIANTS.items():
            ia = np.zeros(nf, bool)
            for x in r.interf:
                ia |= tm.activity(x, rel_db=rel)[:nf]
            ia_x = dilate_fwd(ia, ext) if ext else ia
            tonly = act & ~ia_x
            since = np.full(nf, np.inf)
            last = None
            for f in range(nf):
                if ia[f]:
                    last = f
                elif last is not None:
                    since[f] = (f - last) * eq.FRAME_S
            acc[v]["g"].append(g[tonly])
            acc[v]["since"].append(since[tonly])
    out = {}
    for v, d in acc.items():
        g, since = np.concatenate(d["g"]), np.concatenate(d["since"])
        res = {"frames": int(g.size), "g_med_all": float(np.median(g)), "since": {}}
        m = ~np.isfinite(since)
        if m.any():
            res["since"]["antes"] = {"n": int(m.sum()), "g_med": float(np.median(g[m]))}
        for lo, hi in BINS:
            m = (since > lo) & (since <= hi) if lo > 0 else (since >= 0) & (since <= hi)
            if m.any():
                res["since"][f"{lo:g}-{hi:g}s" if hi < 1e8 else f">{lo:g}s"] = {"n": int(m.sum()),
                                                                                  "g_med": float(np.median(g[m]))}
        out[v] = res
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
