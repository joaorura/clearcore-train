#!/usr/bin/env python3
"""Avaliação curta da presença (pro-v4-presence) contra o lsnr do v3.1, em lotes FIXOS de dev (falantes de dev,
receitas congeladas de `data/eval-v4`, enrollment do dev-select).

Itens: matriz com nível −30 dBFS, ruído silencioso (piso −70 dBFS) e SNR 5, N=0..4 (SIR 0 e +20), N=0/1 também a −45
dBFS, e o eixo `temporal` (conversa/blocos, N=1 e 3). Para cada item com interferente roda também a vista
"só-interferente" (a mesma mistura sem o alvo, com o perfil do alvo).

Métricas por quadro (fora do silêncio digital, média quadrática < 1e-7, que o runtime já curto-circuita):
- AUC e acurácia balanceada da p_target (positivo = alvo ativo) e do −lsnr do v3.1, em dois conjuntos: (i) alvo × só-
  interferente (como o diag do v3.1) e (ii) todos os quadros com perfil;
- % ZERO no só-interferente (vista só-interferente, quadros com interferente ativo) por N e ruído;
- % ZERO no alvo (quadros com alvo ativo na mistura) por N e ruído.
Decisão da presença = máquina de estados do runtime (τ_on/τ_off/hangover); do v3.1 = lsnr < −10 (gating (a)).
`--sweep` varre (τ_on, τ_off, H) e escolhe o de maior % ZERO no só-interferente com % ZERO no alvo ≤ 2% (agregado).

Uso: python scripts/eval_presence.py --ckpt <ckpt presença> [--v31 <ckpt>] --out <json> [--sweep]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import train_pro_v4_presence as P  # noqa: E402
from cctrain.data.features import extract_features  # noqa: E402
from cctrain.eval import tse_matrix as tm  # noqa: E402
from cctrain.m2.config import load_stage_f_config  # noqa: E402
from cctrain.paths import Paths  # noqa: E402
from cctrain.train.checkpoint import load_checkpoint  # noqa: E402
from cctrain.train.stage_f import StageFHParams  # noqa: E402

SILENCE_MS = 1e-7


def select(recs: list[dict]) -> list[dict]:
    keep = []
    for r in recs:
        c = r["cell"]
        if c["axis"] == "temporal":
            keep.append(r)
        elif c["axis"] == "matrix" and c["noise"] in ("silent", "snr5"):
            if c["level"] == -30.0 and (c["n"] == 0 or c["sir"] in (0.0, 20.0)):
                keep.append(r)
            elif c["level"] == -45.0 and (c["n"] == 0 or (c["n"] == 1 and c["sir"] == 0.0)):
                keep.append(r)
    return keep


def auc(s, y):
    s, y = np.asarray(s, np.float64), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    # empates: posto médio
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bal_acc(pred_pos, y):
    y = np.asarray(y, bool)
    tpr = (pred_pos & y).sum() / max(y.sum(), 1)
    tnr = (~pred_pos & ~y).sum() / max((~y).sum(), 1)
    return float((tpr + tnr) / 2)


@torch.no_grad()
def forward(model, x: np.ndarray, emb: np.ndarray, dev: str, presence: bool):
    f = extract_features(x)
    t = lambda a: torch.from_numpy(np.asarray(a))[None].to(dev)  # noqa: E731
    spec, ferb, fspec = t(f.spec), t(f.feat_erb), t(f.feat_spec)
    e = torch.from_numpy(emb.astype(np.float32))[None].to(dev)
    mask = torch.ones(1, 1, device=dev)
    out = model(spec, ferb, fspec, e, mask, return_presence=presence)
    lsnr = out[2].reshape(-1).float().cpu().numpy()
    p = torch.sigmoid(out[4].float()).reshape(-1).cpu().numpy() if presence else None
    return lsnr, p


def frame_ms(x: np.ndarray, frames: int) -> np.ndarray:
    return np.mean(x[: frames * tm.HOP].astype(np.float64).reshape(frames, tm.HOP) ** 2, axis=1)


def run_items(model_p, model_v31, recs, bank, dev):
    """Por item/vista: p, lsnr_p (modelo novo), lsnr_v31, rótulos e máscara de quadros válidos."""
    out = []
    for k, rec in enumerate(recs):
        r = tm.render(rec)
        emb = bank.emb[bank.idx(rec["profile"]["spk"])]
        c = rec["cell"]
        views = [("mix", r.noisy(True))]
        if c["n"] >= 1 and c["axis"] != "temporal":
            views.append(("int", r.noisy(False)))
        for vname, x in views:
            ls_p, p = forward(model_p, x, emb, dev, True)
            ls_3, _ = forward(model_v31, x, emb, dev, False)
            T = min(len(p), len(r.tgt_act))
            ta = r.tgt_act[:T] if vname == "mix" else np.zeros(T, bool)
            ia = r.int_act[:T]
            valid = frame_ms(x, T) >= SILENCE_MS
            out.append({"id": rec["id"], "axis": c["axis"], "n": c["n"], "noise": c["noise"], "level": c["level"],
                        "sir": c["sir"], "pattern": c.get("pattern"), "view": vname, "p": p[:T], "lsnr": ls_p[:T],
                        "lsnr_v31": ls_3[:T], "ta": ta, "ia": ia, "valid": valid})
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(recs)} itens", flush=True)
    return out


def decide(items, tau_on, tau_off, hang):
    """ZERO por quadro (presença: máquina de estados por item, do início do item)."""
    return [~P.presence_state_machine(it["p"][None], tau_on, tau_off, hang)[0] for it in items]


def summarize(items, zero_p, zero_v31):
    groups = defaultdict(lambda: defaultdict(list))
    for it, zp, z3 in zip(items, zero_p, zero_v31):
        v = it["valid"]
        if it["axis"] == "matrix":
            key = f"lv{it['level']:.0f}|{it['noise']}|N{it['n']}" + ("" if it["sir"] is None else f"|sir{it['sir']:+.0f}")
        else:
            key = f"temporal|{it['pattern']}|{it['noise']}|N{it['n']}"
        g = groups[key]
        if it["view"] == "mix":
            m = it["ta"] & v
            g["tgt_frames"].append(m.sum())
            g["tgt_zero_p"].append((zp & m).sum())
            g["tgt_zero_v31"].append((z3 & m).sum())
            oi = ~it["ta"] & it["ia"] & v
            g["oi_mix_frames"].append(oi.sum())
            g["oi_mix_zero_p"].append((zp & oi).sum())
            g["oi_mix_zero_v31"].append((z3 & oi).sum())
        else:
            m = it["ia"] & v
            g["oi_frames"].append(m.sum())
            g["oi_zero_p"].append((zp & m).sum())
            g["oi_zero_v31"].append((z3 & m).sum())
    res = {}
    for key, g in sorted(groups.items()):
        s = {k: int(np.sum(v)) for k, v in g.items()}
        row = {}
        if s.get("tgt_frames"):
            row["tgt_zero_pct_presence"] = round(100 * s["tgt_zero_p"] / s["tgt_frames"], 2)
            row["tgt_zero_pct_v31"] = round(100 * s["tgt_zero_v31"] / s["tgt_frames"], 2)
        if s.get("oi_frames"):
            row["only_interf_zero_pct_presence"] = round(100 * s["oi_zero_p"] / s["oi_frames"], 2)
            row["only_interf_zero_pct_v31"] = round(100 * s["oi_zero_v31"] / s["oi_frames"], 2)
        if s.get("oi_mix_frames"):
            row["oi_in_mix_zero_pct_presence"] = round(100 * s["oi_mix_zero_p"] / s["oi_mix_frames"], 2)
            row["oi_in_mix_zero_pct_v31"] = round(100 * s["oi_mix_zero_v31"] / s["oi_mix_frames"], 2)
        res[key] = row
    return res


def aggregate(items, zero, which="p"):
    tf = tz = of = oz = 0
    for it, z in zip(items, zero):
        v = it["valid"]
        if it["view"] == "mix":
            m = it["ta"] & v
            tf += m.sum()
            tz += (z & m).sum()
            oi = ~it["ta"] & it["ia"] & v
        else:
            oi = it["ia"] & v
        of += oi.sum()
        oz += (z & oi).sum()
    return 100 * tz / max(tf, 1), 100 * oz / max(of, 1)


def discrim(items, tau):
    """AUC / acurácia balanceada (p e −lsnr) em: alvo × só-interferente e todos os quadros com perfil."""
    P_, L_, L3, Y, OI = [], [], [], [], []
    for it in items:
        v = it["valid"]
        P_.append(it["p"][v]); L_.append(it["lsnr"][v]); L3.append(it["lsnr_v31"][v])
        Y.append(it["ta"][v]); OI.append((~it["ta"] & it["ia"])[v])
    p, l3, y, oi = map(np.concatenate, (P_, L3, Y, OI))
    sel = y | oi
    return {
        "target_vs_only_interf": {"n": int(sel.sum()), "frac_target": float(y[sel].mean()),
                                  "auc_p": auc(p[sel], y[sel]), "balacc_p@tau": bal_acc(p[sel] >= tau, y[sel]),
                                  "auc_lsnr_v31": auc(l3[sel], y[sel]),
                                  "balacc_lsnr_v31@-10": bal_acc(l3[sel] >= -10, y[sel])},
        "all_profile_frames": {"n": int(len(y)), "frac_target": float(y.mean()), "auc_p": auc(p, y),
                               "balacc_p@tau": bal_acc(p >= tau, y), "auc_lsnr_v31": auc(l3, y),
                               "balacc_lsnr_v31@-10": bal_acc(l3 >= -10, y)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint com cabeça de presença ('init' = v3.1 + cabeça nova)")
    ap.add_argument("--v31", default=None)
    ap.add_argument("--tau-on", type=float, default=None)
    ap.add_argument("--tau-off", type=float, default=None)
    ap.add_argument("--hangover", type=int, default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    paths = Paths.from_env()
    sd = paths.data / "eval-v4"
    tm.set_roots(paths.data, sd / "user_take4_stream_48k.wav")
    bank = tm.Bank.from_json(json.loads((sd / "bank.json").read_text()), np.load(sd / "bank_emb.npy"))
    recs = select(tm.read_recipes(sd / "recipes.jsonl"))
    if a.limit:
        recs = recs[:: max(1, len(recs) // a.limit)]
    hp = StageFHParams.from_config(load_stage_f_config(), run_id="eval", num_workers=0, batch_size=1)
    v31 = P.resolve(paths, a.v31)
    model_p = P.build_model(dev, hp, presence=True)
    src = v31 if a.ckpt == "init" else Path(a.ckpt)
    ck = load_checkpoint(src)
    cur = model_p.state_dict()
    model_p.load_state_dict({k: v for k, v in ck.model.items() if k in cur and v.shape == cur[k].shape},
                            strict=a.ckpt != "init")
    model_p.eval()
    targs = ck.extra.get("args") or {}
    tau_on = a.tau_on if a.tau_on is not None else targs.get("tau_on", 0.5)
    tau_off = a.tau_off if a.tau_off is not None else targs.get("tau_off", 0.35)
    hang = a.hangover if a.hangover is not None else targs.get("hangover", 5)
    model_3 = P.frozen_reference(v31, dev, hp)
    print(f"{len(recs)} receitas de dev; presença {src}; v3.1 {v31}", flush=True)
    items = run_items(model_p, model_3, recs, bank, dev)
    z3 = [it["lsnr_v31"] < -10.0 for it in items]
    rep = {"ckpt": str(src), "v31": str(v31), "n_recipes": len(recs), "n_views": len(items),
           "recipes_sha256": tm.recipes_digest(sd / "recipes.jsonl")}
    sweep = []
    if a.sweep:
        for t_on in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for d in (0.0, 0.1, 0.2):
                t_off = round(t_on - d, 2)
                if t_off <= 0:
                    continue
                for h in (0, 3, 5, 8, 12):
                    tz, oz = aggregate(items, decide(items, t_on, t_off, h))
                    sweep.append({"tau_on": t_on, "tau_off": t_off, "hangover": h, "tgt_zero_pct": round(tz, 3),
                                  "only_interf_zero_pct": round(oz, 3)})
        ok = [s for s in sweep if s["tgt_zero_pct"] <= 2.0]
        best = max(ok, key=lambda s: (s["only_interf_zero_pct"], -s["tgt_zero_pct"])) if ok else \
            min(sweep, key=lambda s: s["tgt_zero_pct"])
        rep["sweep_best"] = best
        rep["sweep"] = sweep
        tau_on, tau_off, hang = best["tau_on"], best["tau_off"], best["hangover"]
    zp = decide(items, tau_on, tau_off, hang)
    rep["gate"] = {"tau_on": tau_on, "tau_off": tau_off, "hangover_frames": hang}
    rep["discrimination"] = discrim(items, tau_on)
    tz, oz = aggregate(items, zp)
    tz3, oz3 = aggregate(items, z3)
    rep["aggregate"] = {"presence": {"tgt_zero_pct": round(tz, 3), "only_interf_zero_pct": round(oz, 3)},
                        "v31_lsnr": {"tgt_zero_pct": round(tz3, 3), "only_interf_zero_pct": round(oz3, 3)}}
    rep["cells"] = summarize(items, zp, z3)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("gate", "discrimination", "aggregate")}, indent=1, default=float))
    print(json.dumps(rep["cells"], indent=0))


if __name__ == "__main__":
    main()
