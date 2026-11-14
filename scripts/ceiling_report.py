#!/usr/bin/env python3
"""Tabela final do teto: junta o estático (v3.1 lsnr, ckpt C, rótulos estritos, ECAPA) com as cabeças treinadas
(varredura do gate e % ZERO por N) e o custo do ECAPA.

Uso: python scripts/ceiling_report.py --cache <npz> --static <json> --heads-dir <dir> [--cost <json>] --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ceiling_common as C
from ceiling_eval_static import load_cache, strict_masks, summary_row


def line(name: str, r: dict) -> str:
    g = r.get("groups") or {}
    per_n = " ".join(f"N{k}:{g.get(f'N{k}', {}).get('only_interf_zero_pct', float('nan')):.0f}/"
                     f"{g.get(f'N{k}', {}).get('tgt_zero_pct', float('nan')):.1f}" for k in range(5))
    return (f"{name:<28} AUC {r['auc']:.3f} BA {r['ba_max']:.3f} (@τ {r['ba_at_tau_on']:.3f}) "
            f"ZERO só-interf {r['only_interf_zero_pct']:5.1f}% alvo {r['tgt_zero_pct']:4.2f}% | {per_n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--static", required=True)
    ap.add_argument("--heads-dir", required=True)
    ap.add_argument("--cost", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = load_cache(a.cache)
    st = json.loads(Path(a.static).read_text())
    rows = {"v31_lsnr (gate -10)": st["v31_lsnr_repro"], "v31_lsnr (varrido)": st["v31_lsnr_resweep"],
            "ckptC (gate 0.3/0.1/12)": st["ckptC_repro_fixed_gate"], "ckptC (varrido)": st["ckptC_resweep"]}
    for k in ("rel20", "rel10", "vad_strict", "rel20_dominant", "solo_target"):
        rows[f"ckptC rótulo {k}"] = st["labels"][k]["ckptC_resweep"]
    for k, v in st["ecapa"].items():
        if not k.endswith("rel20"):
            rows[f"ECAPA {k}"] = v
    sm = strict_masks(c)
    rel20 = sm["rel20"]
    heads = {}
    for f in sorted(Path(a.heads_dir).glob("dev-scores-*.npy")):
        name = f.stem.removeprefix("dev-scores-")
        s = np.load(f).astype(np.float32)
        r = summary_row(C.evaluate_score(c, s))
        heads[name] = {"all": r, "rel20": summary_row(C.evaluate_score(c, s, pos_mask=rel20, groups=False)),
                       "solo_target": summary_row(C.evaluate_score(c, s, pos_mask=sm["solo_target"], groups=False))}
        rows[f"congelado {name}"] = r
        rows[f"congelado {name} rótulo rel20"] = heads[name]["rel20"]
        rows[f"congelado {name} só-alvo"] = heads[name]["solo_target"]
    # só-interferente separado: vista sem alvo (alvo ausente o item todo) × trechos sem alvo dentro da mistura
    scores = {"v31_lsnr": (c["lsnr31"], False), "ckptC": (c["pC"], True),
              "ECAPA cos_x_energy_1.0s": (c["cos1.0"] * C.energy_weight(c["mix_db"]), False)}
    for f in sorted(Path(a.heads_dir).glob("dev-scores-*.npy")):
        scores[f"congelado {f.stem.removeprefix('dev-scores-')}"] = (np.load(f).astype(np.float32), True)
    split = {}
    is_int = (c["view"] == "int")[:, None]
    for name, (s, prob) in scores.items():
        r = C.evaluate_score(c, s, prob=prob, groups=False)
        g = r["gate"]
        z = ~C.presence_state_machine(s, g["tau_on"], g["tau_off"], g["hangover"])
        _, neg = C.pos_neg(c)
        nv, nm = neg & is_int, neg & ~is_int
        # AUC alvo × vista só-interf. apenas
        pos, _ = C.pos_neg(c)
        sc = np.concatenate([s[pos], s[nv]])
        y = np.concatenate([np.ones(pos.sum(), bool), np.zeros(nv.sum(), bool)])
        split[name] = {"zero_only_interf_view_pct": round(100 * float((z & nv).sum() / nv.sum()), 2),
                       "zero_oi_in_mix_pct": round(100 * float((z & nm).sum() / nm.sum()), 2),
                       "auc_target_vs_only_interf_view": round(C.auc(sc, y), 4),
                       "frames_view": int(nv.sum()), "frames_in_mix": int(nm.sum())}
        print("split", name, split[name], flush=True)
    curve = json.loads((Path(a.heads_dir) / "curve.json").read_text())
    rep = {"rows": rows, "heads": heads, "curve": curve, "split_only_interf": split,
           "cost": json.loads(Path(a.cost).read_text()) if a.cost else None}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("variante | AUC | BA máx (@τ_on) | %ZERO só-interf | %ZERO alvo | por N: só-interf/alvo")
    for k, r in rows.items():
        print(line(k, r))
    print("\ncurva (AUC por passo):")
    for row in curve["curve"]:
        print(row["step"], {k: v["auc"] for k, v in row.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
