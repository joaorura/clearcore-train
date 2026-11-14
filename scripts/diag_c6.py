#!/usr/bin/env python3
"""Diagnóstico do C6: atenuação do alvo nos quadros "alvo sozinho" (ativo, sem interferente) do usuário take4 N=1
silencioso. Decompõe por estágio do gating, por tempo desde o último quadro com interferente e compara com o gating por
lsnr desligado (ERB+DF sempre). Uso: diag_c6.py [--axis user --noise silent --n 1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_quality as eq  # noqa: E402
from cctrain.eval import runtime_path as rp  # noqa: E402
from cctrain.eval import tse_eval as te  # noqa: E402
from cctrain.eval import tse_matrix as tm  # noqa: E402

# a saída alinhada (atraso 1.440 = 3 quadros) do quadro f sai do passo f + 2 da réplica (2 de lookahead + 1 da STFT,
# o passo t sintetiza o quadro t - 2 dos anéis); conferido abaixo: quadros com estágio ZERO têm ganho ~ -60 dB.
STEP_OFFSET = 2
BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9))
STAGES = {int(rp.Stage.ZERO): "ZERO", int(rp.Stage.ERB_ONLY): "ERB-only", int(rp.Stage.ERB_DF): "ERB+DF",
          int(rp.Stage.NONE): "passthrough", int(rp.Stage.SILENT): "silêncio"}
OPEN = te.Gating("lsnr", (np.float32(-np.inf), np.float32(np.inf), np.float32(np.inf)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="user")
    ap.add_argument("--noise", default="silent")
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    eq._init(2)
    recs = [r for r in eq.all_recipes() if r["cell"]["axis"] == a.axis and r["cell"]["n"] == a.n
            and r["cell"]["level"] == -30.0 and r["cell"]["noise"] == a.noise and r["cell"].get("sir") == 0.0]
    acc: dict = {}
    for rec in recs:
        r = tm.render(rec)
        noisy = r.noisy(True)
        prof = eq.profile_of(rec)
        nf = len(r.target) // tm.HOP
        act = te.target_active_local(r.target, nf)
        ia = r.int_act[:nf]
        tonly = act & ~ia
        # tempo desde o último quadro com interferente (inf antes do primeiro)
        since = np.full(nf, np.inf)
        last = None
        for f in range(nf):
            if ia[f]:
                last = f
            elif last is not None:
                since[f] = (f - last) * eq.FRAME_S
        for mk in ("v31", "keep"):
            rn = eq._W["runners"][mk]
            audio, n = rn._padded(noisy)
            pm = te.ProfileFrameModel(rn.model, prof, audio)
            lsnr = pm.lsnr_true
            outs = rn.run_multi(noisy, prof, {"new": te.Gating.of("new"), "open": OPEN}, shadow=[r.target, r.noise])
            for gname, (enh, st, sh) in outs.items():
                g = te.frame_gain_db(r.target, sh[0], nf)
                stf = np.full(nf, int(rp.Stage.SILENT))
                k = min(nf, len(st) - STEP_OFFSET)
                stf[:k] = st[STEP_OFFSET: STEP_OFFSET + k]
                ls = np.full(nf, np.nan)
                k2 = min(nf, len(lsnr) - STEP_OFFSET)
                ls[:k2] = lsnr[STEP_OFFSET: STEP_OFFSET + k2]
                d = acc.setdefault((mk, gname), {"g": [], "st": [], "since": [], "ls": [], "g_n0": []})
                d["g"].append(g[tonly])
                d["st"].append(stf[tonly])
                d["since"].append(since[tonly])
                d["ls"].append(ls[tonly])
                d.setdefault("zero_all", []).append(g[act & (stf == int(rp.Stage.ZERO))])
    out = {}
    for (mk, gname), d in acc.items():
        g, st, since, ls = (np.concatenate(d[k]) for k in ("g", "st", "since", "ls"))
        res = {"frames": int(g.size), "g_med": float(np.median(g)), "g_mean_db": float(g.mean()),
               "lsnr_mean": float(np.nanmean(ls)), "lsnr_med": float(np.nanmedian(ls)),
               "zero_check_g_med_all_active": float(np.median(np.concatenate(d["zero_all"]))) if
               np.concatenate(d["zero_all"]).size else None, "stages": {}, "since": {}}
        for code, name in STAGES.items():
            m = st == code
            if m.any():
                res["stages"][name] = {"pct": float(100 * m.mean()), "g_med": float(np.median(g[m])),
                                       "g_mean": float(g[m].mean()), "lsnr_mean": float(np.nanmean(ls[m]))}
        m = ~np.isfinite(since)
        if m.any():
            res["since"]["antes do 1º interferente"] = {"n": int(m.sum()), "g_med": float(np.median(g[m])),
                                                         "zero_pct": float(100 * (st[m] == 0).mean()),
                                                         "lsnr_mean": float(np.nanmean(ls[m]))}
        for lo, hi in BINS:
            m = (since > lo) & (since <= hi) if lo > 0 else (since >= 0) & (since <= hi)
            if m.any():
                res["since"][f"{lo:g}-{hi:g}s" if hi < 1e8 else f">{lo:g}s"] = {
                    "n": int(m.sum()), "g_med": float(np.median(g[m])), "zero_pct": float(100 * (st[m] == 0).mean()),
                    "lsnr_mean": float(np.nanmean(ls[m]))}
        out[f"{mk}_{gname}"] = res
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
