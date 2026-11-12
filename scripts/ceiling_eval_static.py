#!/usr/bin/env python3
"""Partes 1 e 3 do teto: (1) rótulos estritos para o p_target do ckpt C (e o lsnr do v3.1); (3) cosseno ECAPA em janelas
causais de 0,5/1,0/1,5 s (e cosseno × peso de energia) como score por quadro. Tudo sobre o cache do dev.

Uso: python scripts/ceiling_eval_static.py --cache <npz> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

import ceiling_common as C


def load_cache(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    c = {k: z[k] for k in z.files if k != "emb31"}
    return c


def strict_masks(c: dict) -> dict[str, np.ndarray]:
    """Subconjuntos de positivos (quadros com alvo ativo pelo rótulo original, VAD a 30 dB do máximo)."""
    tdb = c["tgt_db"]
    mx = tdb.max(axis=1, keepdims=True)
    rel20 = tdb >= mx - 20.0
    rel10 = tdb >= mx - 10.0
    # VAD estrito: a 20 dB do máximo, sem bordas (erosão de 3 quadros de cada lado) e sem trechos < 100 ms
    strict = np.zeros_like(rel20)
    for i in range(len(rel20)):
        m = binary_erosion(rel20[i], structure=np.ones(7, bool), border_value=0)
        # remove trechos curtos
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(st, en):
            if e - s < 10:
                m[s:e] = False
        strict[i] = m
    # alvo dominante no quadro: nível do alvo ≥ nível da soma dos interferentes (SIR local ≥ 0 dB)
    dom = tdb >= c["int_db"]
    return {"orig_rel30": np.ones_like(rel20), "rel20": rel20, "rel10": rel10, "vad_strict": strict,
            "rel20_dominant": rel20 & dom, "solo_target": ~c["ia"]}


def summary_row(r: dict) -> dict:
    g = r["gate"]
    return {"auc": round(r["auc"], 4), "ba_at_tau_on": round(r["ba_at_tau_on"], 4), "ba_max": round(r["ba_max"], 4),
            "only_interf_zero_pct": round(g["only_interf_zero_pct"], 2), "tgt_zero_pct": round(g["tgt_zero_pct"], 2),
            "gate": {k: g[k] for k in ("tau_on", "tau_off", "hangover")}, "n_pos": r["n_pos"],
            "groups": r.get("groups")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = load_cache(a.cache)
    rep = {"n_views": int(len(c["view"])), "n_recipes": int(len(set(c["id"].tolist())))}
    gate_c = {"tau_on": 0.3, "tau_off": 0.1, "hangover": 12}  # ponto do eval-dev-500 do ckpt C
    # sanidade: reproduz o eval-dev-500 do ckpt C
    rep["ckptC_repro_fixed_gate"] = summary_row(C.evaluate_score(c, c["pC"], gate=gate_c))
    rep["v31_lsnr_repro"] = summary_row(C.evaluate_score(c, c["lsnr31"], gate={"tau_on": -10.0, "tau_off": -10.0,
                                                                                    "hangover": 0}))
    # 1. rótulos estritos
    sm = strict_masks(c)
    pos_all, _ = C.pos_neg(c)
    rep["label_fraction_kept"] = {k: round(float((pos_all & m).sum() / pos_all.sum()), 4) for k, m in sm.items()}
    rep["labels"] = {}
    for name, m in sm.items():
        rep["labels"][name] = {
            "ckptC_fixed_gate": summary_row(C.evaluate_score(c, c["pC"], pos_mask=m, gate=gate_c, groups=False)),
            "ckptC_resweep": summary_row(C.evaluate_score(c, c["pC"], pos_mask=m)),
            "v31_lsnr_resweep": summary_row(C.evaluate_score(c, c["lsnr31"], pos_mask=m, prob=False, groups=False)),
        }
        print(name, {k: (v["auc"], v["ba_max"], v["only_interf_zero_pct"], v["tgt_zero_pct"])
                     for k, v in rep["labels"][name].items()}, flush=True)
    # quanto do %ZERO-alvo do ckpt C (ponto fixo) cai em quadros de baixa energia
    zc = ~C.presence_state_machine(c["pC"], **gate_c)
    tdb = c["tgt_db"]
    rel = tdb - tdb.max(axis=1, keepdims=True)
    zt = zc & pos_all
    rep["ckptC_target_zero_by_rel_db"] = {
        b: {"frames": int((pos_all & sel).sum()), "zero_pct_in_bin": round(100 * float((zt & sel).sum() / max((pos_all & sel).sum(), 1)), 2),
            "share_of_target_zeros": round(100 * float((zt & sel).sum() / max(zt.sum(), 1)), 2)}
        for b, sel in (("0..-10", rel >= -10), ("-10..-20", (rel < -10) & (rel >= -20)), ("-20..-30", rel < -20))}
    # 3. ECAPA
    ew = C.energy_weight(c["mix_db"])
    rep["ecapa"] = {}
    for w in ("0.5", "1.0", "1.5"):
        cos = c[f"cos{w}"]
        rep["ecapa"][f"cos_{w}s"] = summary_row(C.evaluate_score(c, cos, prob=False))
        rep["ecapa"][f"cos_x_energy_{w}s"] = summary_row(C.evaluate_score(c, cos * ew, prob=False))
        rep["ecapa"][f"cos_{w}s_rel20"] = summary_row(C.evaluate_score(c, cos, pos_mask=sm["rel20"], prob=False,
                                                                       groups=False))
        print("ecapa", w, {k: (v["auc"], v["ba_max"], v["only_interf_zero_pct"], v["tgt_zero_pct"])
                           for k, v in rep["ecapa"].items() if w in k}, flush=True)
    # referência: ckpt C e lsnr com varredura própria (mesma rotina)
    rep["ckptC_resweep"] = summary_row(C.evaluate_score(c, c["pC"]))
    rep["v31_lsnr_resweep"] = summary_row(C.evaluate_score(c, c["lsnr31"], prob=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("ckptC_repro_fixed_gate", "v31_lsnr_repro", "label_fraction_kept",
                                          "ckptC_target_zero_by_rel_db")}, indent=1, default=float))


if __name__ == "__main__":
    main()
