#!/usr/bin/env python3
"""Teste de gradiente do pro-v4-presence (como `diag_pro_v31.py`): normas por termo (já ponderado pelo λ do treino) na
cabeça `p_target` e no tronco, e cossenos entre termos (por lote e da soma dos lotes). Só leitura.

Uso: python scripts/diag_pro_v4p.py --ckpt <ckpt ou 'init'> --out <json> [--steps 1000,1001,2000,2001]
'init' = v3.1 + cabeça nova (o ponto de partida do treino).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import train_pro_v4_presence as P  # noqa: E402
from cctrain.m2.config import load_stage_f_config  # noqa: E402
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3  # noqa: E402
from cctrain.paths import Paths  # noqa: E402
from cctrain.train.checkpoint import load_checkpoint  # noqa: E402
from cctrain.train.stage_c import _to_device  # noqa: E402
from cctrain.train.stage_f import HOP, StageFHParams, prepare_student  # noqa: E402
from df.loss import Istft  # noqa: E402
from df.model import ModelParams  # noqa: E402

TERMS = ("mr", "os", "leak", "ovl", "guard", "lsnr", "bce", "t", "total")


def flat(gs, ps):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(gs, ps)])


def cos(a, b):
    return float(F.cosine_similarity(a[None], b[None]).item()) if a.norm() > 0 and b.norm() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-args", default=None, help="args extras do treino (lambdas), ex.: '--lambda-ovl 30'")
    ap.add_argument("--steps", default="1000,1001,5000,5001")
    ap.add_argument("--half", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    paths = Paths.from_env()
    targs = P.parse_args((a.train_args or "").split())
    hp = StageFHParams.from_config(load_stage_f_config(), run_id="diag", num_workers=0, batch_size=a.half)
    model = P.build_model(dev, hp, presence=True)
    init = P.resolve(paths, None)
    src = init if a.ckpt == "init" else Path(a.ckpt)
    ck = load_checkpoint(src)
    cur = model.state_dict()
    compat = {k: v for k, v in ck.model.items() if k in cur and v.shape == cur[k].shape}
    model.load_state_dict(compat, strict=False)
    if a.ckpt != "init":
        for k, v in (ck.extra.get("args") or {}).items():
            if k.startswith(("lambda_", "tau_", "hangover", "st_", "guard_margin", "bce_", "lsnr_factor")):
                setattr(targs, k, v)
    model.presence_trunk_grad_scale = targs.bce_trunk_scale
    prepare_student(model)
    ref = P.frozen_reference(P.resolve(paths, None), dev, hp)
    teacher, df_state = load_upstream_dfnet3(device=dev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    istft = Istft(960, HOP, torch.as_tensor(df_state.fft_window().copy())).to(dev)
    targs.batch_size = 16
    ds, _ = P.make_dataset(paths, P.v4_params(targs))
    idx = [s * 16 + k for s in map(int, a.steps.split(",")) for k in range(16)]
    batches = [_to_device(b, dev) for b in DataLoader(ds, batch_size=a.half, sampler=idx, num_workers=0)]
    print(f"{len(batches)} lotes de {a.half}; loaded {len(compat)}/{len(cur)}", flush=True)

    hid = {id(p) for p in model.presence.parameters()}
    groups = {
        "p_head": [p for p in model.presence.parameters() if p.requires_grad],
        "trunk": [p for p in model.parameters() if p.requires_grad and id(p) not in hid],
        "emb_gru": [p for p in model.base.enc.emb_gru.parameters() if p.requires_grad],
        "cross_attn": [p for p in model.cross_attn.parameters() if p.requires_grad],
        "film": [p for p in model.film.parameters() if p.requires_grad],
    }
    allp = groups["p_head"] + groups["trunk"]
    norms = {t: {g: [] for g in groups} for t in TERMS}
    sums = {t: {g: None for g in ("p_head", "trunk")} for t in TERMS}
    pairs = list(itertools.combinations([t for t in TERMS if t != "total"], 2)) + \
        [(t, "total") for t in TERMS if t != "total"]
    cos_b = {f"{x}|{y}": {"p_head": [], "trunk": []} for x, y in pairs}
    vals = {t: [] for t in TERMS}
    for b in batches:
        prepare_student(model)
        model.zero_grad(set_to_none=True)
        parts = P.step_presence(model, ref, teacher, istft, ModelParams(), hp, b, targs)
        fl = {}
        for t in TERMS:
            v = parts[t]
            vals[t].append(float(v.detach()))
            if not v.requires_grad:
                for g in groups:
                    norms[t][g].append(0.0)
                continue
            gs = torch.autograd.grad(v, allp, retain_graph=True, allow_unused=True)
            gmap = {id(p): gg for p, gg in zip(allp, gs)}
            for g, ps in groups.items():
                vec = flat([gmap.get(id(p)) for p in ps], ps)
                norms[t][g].append(float(vec.norm()))
                if g in ("p_head", "trunk"):
                    fl[(t, g)] = vec
                    sums[t][g] = vec.clone() if sums[t][g] is None else sums[t][g] + vec
        for x, y in pairs:
            for g in ("p_head", "trunk"):
                if (x, g) in fl and (y, g) in fl:
                    cos_b[f"{x}|{y}"][g].append(cos(fl[(x, g)], fl[(y, g)]))
        del fl, parts
        torch.cuda.empty_cache()
    cos_sum = {f"{x}|{y}": {g: cos(sums[x][g], sums[y][g]) for g in ("p_head", "trunk")
                            if sums[x][g] is not None and sums[y][g] is not None} for x, y in pairs}
    cos_mean = {k: {g: (float(np.nanmean(v)) if v else None) for g, v in d.items()} for k, d in cos_b.items()}
    worst = sorted(((v.get("trunk", 1.0), k) for k, v in cos_sum.items() if v.get("trunk") == v.get("trunk")))[:5]
    rep = {"ckpt": str(src), "lambdas": {k: getattr(targs, k) for k in vars(targs) if k.startswith("lambda_")},
           "loss_mean": {t: float(np.mean(v)) for t, v in vals.items()},
           "grad_norm_mean": {t: {g: float(np.mean(v)) for g, v in d.items()} for t, d in norms.items()},
           "cos_summed": cos_sum, "cos_per_batch_mean": cos_mean, "most_negative_trunk_summed": worst}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("loss_mean", "grad_norm_mean", "most_negative_trunk_summed")}, indent=1))


if __name__ == "__main__":
    main()
