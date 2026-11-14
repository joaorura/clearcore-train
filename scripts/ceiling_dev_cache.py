#!/usr/bin/env python3
"""Cache do dev para o experimento de teto (as mesmas 304 receitas / 520 vistas do `eval_presence.py`).

Por vista: rótulos (ta, ia, valid), níveis por quadro (alvo limpo, soma dos interferentes, mistura), p_target e lsnr
do ckpt C, lsnr do v3.1, `emb_conditioned` do v3.1 congelado (fp16) e cosseno ECAPA×perfil em janelas causais de
0,5 / 1,0 / 1,5 s a cada 100 ms.

Uso: python scripts/ceiling_dev_cache.py --ckpt-c <ckpt> --out <npz>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import ceiling_common as C
import eval_presence as E
import train_pro_v4_presence as P
from cctrain.data.features import extract_features
from cctrain.data.resample import to_16k
from cctrain.eval import tse_matrix as tm
from cctrain.m2.config import load_stage_f_config
from cctrain.paths import Paths
from cctrain.train.checkpoint import load_checkpoint
from cctrain.train.stage_f import StageFHParams

T = 500
WINS = (0.5, 1.0, 1.5)


@torch.no_grad()
def v31_forward(model, x, emb, dev):
    f = extract_features(x)
    t = lambda a: torch.from_numpy(np.asarray(a))[None].to(dev)  # noqa: E731
    e = torch.from_numpy(emb.astype(np.float32))[None].to(dev)
    (_, _, lsnr, _), att = model(t(f.spec), t(f.feat_erb), t(f.feat_spec), e, torch.ones(1, 1, device=dev),
                                 return_attention_details=True)
    return lsnr.reshape(-1).float().cpu().numpy(), att.output[0].half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-c", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    paths = Paths.from_env()
    sd = paths.data / "eval-v4"
    tm.set_roots(paths.data, sd / "user_take4_stream_48k.wav")
    bank = tm.Bank.from_json(json.loads((sd / "bank.json").read_text()), np.load(sd / "bank_emb.npy"))
    recs = E.select(tm.read_recipes(sd / "recipes.jsonl"))
    hp = StageFHParams.from_config(load_stage_f_config(), run_id="ceil", num_workers=0, batch_size=1)
    v31 = P.resolve(paths, None)
    model_c = P.build_model(dev, hp, presence=True)
    model_c.load_state_dict(load_checkpoint(Path(a.ckpt_c)).model, strict=True)
    model_c.eval()
    model_3 = P.frozen_reference(v31, dev, hp)
    fbank, ecapa = C.load_ecapa(paths.data, dev)
    t0 = time.time()
    cols = {k: [] for k in ("id", "axis", "n", "noise", "level", "sir", "pattern", "view", "spk_idx")}
    arr = {k: [] for k in ("ta", "ia", "valid", "tgt_db", "int_db", "mix_db", "pC", "lsnrC", "lsnr31", "emb31",
                           "cos0.5", "cos1.0", "cos1.5")}
    audio16 = []
    for k, rec in enumerate(recs):
        r = tm.render(rec)
        si = bank.idx(rec["profile"]["spk"])
        emb = bank.emb[si]
        c = rec["cell"]
        views = [("mix", r.noisy(True))]
        if c["n"] >= 1 and c["axis"] != "temporal":
            views.append(("int", r.noisy(False)))
        tdb = tm.frame_db(r.target)[:T]
        idb = tm.frame_db(r.interf.sum(axis=0))[:T] if len(r.interf) else np.full(T, -120.0)
        for vname, x in views:
            ls_c, p = E.forward(model_c, x, emb, dev, True)
            ls_3, e31 = v31_forward(model_3, x, emb, dev)
            assert len(p) >= T and len(r.tgt_act) >= T, (len(p), len(r.tgt_act))
            for kk, v in (("id", rec["id"]), ("axis", c["axis"]), ("n", c["n"]), ("noise", c["noise"]),
                          ("level", c["level"]), ("sir", np.nan if c["sir"] is None else c["sir"]),
                          ("pattern", c.get("pattern") or ""), ("view", vname), ("spk_idx", si)):
                cols[kk].append(v)
            arr["ta"].append(r.tgt_act[:T] if vname == "mix" else np.zeros(T, bool))
            arr["ia"].append(r.int_act[:T])
            arr["valid"].append(E.frame_ms(x, T) >= E.SILENCE_MS)
            arr["tgt_db"].append(tdb if vname == "mix" else np.full(T, -120.0))
            arr["int_db"].append(idb)
            arr["mix_db"].append(tm.frame_db(x)[:T])
            arr["pC"].append(p[:T])
            arr["lsnrC"].append(ls_c[:T])
            arr["lsnr31"].append(ls_3[:T])
            arr["emb31"].append(e31[:T])
            audio16.append(to_16k(x, tm.SR))
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(recs)} receitas ({time.time() - t0:.0f} s)", flush=True)
    # ECAPA nas vistas (mistura de entrada), em lotes
    prof = torch.from_numpy(np.stack([bank.emb[i] for i in cols["spk_idx"]]).astype(np.float32)).to(dev)
    x16 = torch.from_numpy(np.stack([a_[: 80_000] for a_ in audio16])).to(dev)
    for w in WINS:
        cs = []
        for i in range(0, len(x16), 16):
            ew = C.window_embeddings(fbank, ecapa, x16[i: i + 16], w)
            cs.append(C.cos_per_frame(ew, prof[i: i + 16], T).cpu().numpy())
        arr[f"cos{w:.1f}"] = list(np.concatenate(cs))
        print(f"ECAPA {w} s pronto ({time.time() - t0:.0f} s)", flush=True)
    out = {k: np.asarray(v) for k, v in cols.items()}
    out.update({k: np.stack(v) for k, v in arr.items()})
    out["profiles"] = prof.cpu().numpy()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"{len(cols['id'])} vistas -> {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
