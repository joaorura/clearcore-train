#!/usr/bin/env python3
"""Partes 2 e 4 do teto: tronco v3.1 CONGELADO, só cabeças de presença (BCE com os pesos por classe do treino),
várias variantes nos MESMOS lotes (ver `ceiling_common.head_specs`): h32 (a PresenceHead atual), h128x2, h128x2 com
contexto causal ~1,3 s, e as combinadas com o cosseno ECAPA da janela causal de 1 s (+ uma só-ECAPA).
Dev (cache do `ceiling_dev_cache.py`) a cada `--eval-every` passos: AUC / BA; no fim, varredura do gate e % ZERO.

Uso: python scripts/ceiling_train_heads.py --cache <npz> --run-dir <dir> [--steps 3500]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import ceiling_common as C
import train_pro_v4_presence as P
import cctrain.data.mix_v4 as MV
from cctrain.data.resample import to_16k
from cctrain.m1 import pools
from cctrain.m2.config import load_stage_f_config
from cctrain.paths import Paths
from cctrain.train.embed_bank import ArrayEmbeddingBank
from cctrain.train.stage_c import _to_device
from cctrain.train.stage_f import StageFHParams

T = 500
_LAST: dict = {}
_orig_extract = MV.extract_features


def _capture(x):
    _LAST["x"] = x
    return _orig_extract(x)


MV.extract_features = _capture
MV.analysis = lambda x: np.zeros(1, np.float32)  # spec_clean/spec_interf não são usados aqui


class DS(MV.StageV4Dataset):
    def __getitem__(self, i):
        d = super().__getitem__(i)
        x = _LAST["x"]
        n = len(x) // 480
        d["x16"] = to_16k(x, 48_000)[: n * 160].astype(np.float32)
        fr = x[: n * 480].astype(np.float64).reshape(n, 480)
        d["mix_db"] = (10 * np.log10((fr * fr).mean(1) + 1e-12)).astype(np.float32)
        for k in ("spec_clean", "spec_interf", "target_key", "target_path"):
            d.pop(k, None)
        return d


def dev_tensors(cache: str, dev: str):
    z = np.load(cache)
    c = {k: z[k] for k in z.files if k != "emb31"}
    t = {"emb31": torch.from_numpy(z["emb31"]).to(dev), "prof": torch.from_numpy(c["profiles"]).to(dev),
         "extra": C.extra_feats(torch.from_numpy(c["cos1.0"]).float().to(dev),
                                torch.from_numpy(c["mix_db"]).float().to(dev))}
    return c, t


@torch.no_grad()
def dev_scores(head, t, needs_extra: bool):
    out = []
    for i in range(0, t["emb31"].shape[0], 64):
        x = t["emb31"][i: i + 64].float()
        ex = t["extra"][i: i + 64] if needs_extra else None
        out.append(torch.sigmoid(head(x, t["prof"][i: i + 64], ex)).cpu().numpy())
    return np.concatenate(out)


def quick(c, s):
    pos, neg = C.pos_neg(c)
    sc = np.concatenate([s[pos], s[neg]])
    y = np.concatenate([np.ones(pos.sum(), bool), np.zeros(neg.sum(), bool)])
    qs = np.quantile(sc, np.linspace(0.01, 0.99, 99))
    return {"auc": round(C.auc(sc, y), 4), "ba_max": round(max(C.bal_acc(sc >= q, y) for q in qs), 4),
            "ba@0.5": round(C.bal_acc(sc >= 0.5, y), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    paths = Paths.from_env()
    hp = StageFHParams.from_config(load_stage_f_config(), run_id="ceil", num_workers=a.workers,
                                   batch_size=a.batch_size)
    trunk = P.frozen_reference(P.resolve(paths, None), dev, hp)
    fbank, ecapa = C.load_ecapa(paths.data, dev)
    for p in ecapa.parameters():
        p.requires_grad_(False)
    specs = C.head_specs()
    heads = {k: C.Head(**v).to(dev) for k, v in specs.items()}
    needs = {k: v.get("extra", 0) > 0 for k, v in specs.items()}
    nparams = {k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()}
    print("parâmetros:", nparams, flush=True)
    params = [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    vp = MV.V4Params(seed=a.seed, batch_size=a.batch_size, total_steps=a.steps, phase1_frac=0.0,
                     n_probs=(0.35, 0.30, 0.17, 0.11, 0.07))
    bank = ArrayEmbeddingBank.load(paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz")
    speech = pools.by_split(pools.speech_records(paths), "train")
    noise_train, _ = pools.noise_pools()
    pl = MV.build_pools(speech, noise_train, pools.rir_records(), bank, paths.data, seed=vp.seed)
    ds = DS(pl, bank, paths.data, vp)
    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers, prefetch_factor=2, shuffle=False)
    c, dt = dev_tensors(a.cache, dev)

    curve = []
    logf = open(run / "train-log.jsonl", "w")
    t0 = time.time()
    step = 0
    run_loss = {k: 0.0 for k in heads}

    def do_eval(final=False):
        row = {"step": step, "elapsed_s": round(time.time() - t0, 1)}
        for k, h in heads.items():
            h.eval()
            s = dev_scores(h, dt, needs[k])
            row[k] = quick(c, s)
            if final:
                np.save(run / f"dev-scores-{k}.npy", s.astype(np.float16))
            h.train()
        curve.append(row)
        print(json.dumps(row), flush=True)

    for b in dl:
        if step >= a.steps:
            break
        b = _to_device(b, dev)
        with torch.no_grad():
            nb = b["spec"].shape[0]
            (_, _, _, _), att = trunk(b["spec"], b["feat_erb"], b["feat_spec"], b["embedding"],
                                      torch.ones(nb, 1, device=dev), return_attention_details=True)
            tb = b["mix_db"].shape[1]
            x = att.output[:, :tb].float()
            e = b["embedding"].float()
            e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            cos1 = C.cos_per_frame(C.window_embeddings(fbank, ecapa, b["x16"], 1.0), e, tb)
            bad = int((~torch.isfinite(cos1)).sum())
            if bad:
                print(f"AVISO passo {step}: {bad} quadros com cosseno não finito", flush=True)
                cos1 = torch.nan_to_num(cos1, nan=0.0, posinf=0.0, neginf=0.0)
            extra = C.extra_feats(cos1, b["mix_db"])
        ta, ia = b["target_active"][:, :tb], b["interf_active"][:, :tb]
        valid = ~b["digital_silence"][:, :tb]
        n0t = (b["n_interf"] == 0)[:, None] & ta
        total = 0.0
        for k, h in heads.items():
            logit = h(x, e, extra if needs[k] else None)
            loss = P.bce_presence(logit, ta, ia, valid, n0t, 1.0)
            run_loss[k] = 0.98 * run_loss[k] + 0.02 * float(loss.detach()) if step else float(loss.detach())
            total = total + loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        for h in heads.values():
            torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            ent = {"step": step, "s": round(time.time() - t0, 1), **{k: round(v, 4) for k, v in run_loss.items()}}
            logf.write(json.dumps(ent) + "\n")
            logf.flush()
            if step % 250 == 0:
                print(json.dumps(ent), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            do_eval(final=step == a.steps)
    torch.save({k: h.state_dict() for k, h in heads.items()}, run / "heads.pt")
    (run / "curve.json").write_text(json.dumps({"params": nparams, "curve": curve, "args": vars(a)}, indent=1))
    print(f"fim: {step} passos em {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
