#!/usr/bin/env python3
"""Decompõe o dano do perfil em N=0 (a quantidade do L_guard): SI-SDR sem perfil (gating atual) × com perfil (gating
v4) × com perfil sem as decisões ZERO, por faixa de SNR. Só leitura."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import train_pro_v4 as T  # noqa: E402
import diag_pro_v31 as D  # noqa: E402
from cctrain.m1 import pools  # noqa: E402
from cctrain.paths import Paths  # noqa: E402
from cctrain.train.embed_bank import ArrayEmbeddingBank  # noqa: E402
from cctrain.train.stage_c import _to_device  # noqa: E402
from cctrain.data.mix_v4 import StageV4Dataset, V4Params, build_pools  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--start", type=int, default=80500 * 16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    paths = Paths.from_env()
    model, _, istft, _ = D.build(dev)
    D.load_into(model, a.ckpt)
    model.eval()
    vp = V4Params(batch_size=16, total_steps=85500)
    bank = ArrayEmbeddingBank.load(paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz")
    pl = build_pools(pools.by_split(pools.speech_records(paths), "train"), pools.noise_pools()[0],
                     pools.rir_records(), bank, paths.data, seed=vp.seed)
    ds = StageV4Dataset(pl, bank, paths.data, vp)
    keep, i = [], a.start
    while len(keep) < a.n:
        s = ds[i]
        if s["n_interf"] == 0 and s["mask"][0] > 0 and s["has_target"]:
            keep.append(i)
        i += 1
    rows = []
    for b in DataLoader(ds, batch_size=8, sampler=keep, num_workers=0):
        b = _to_device(b, dev)
        nb, Tn = b["spec"].shape[0], b["spec"].shape[2]
        with torch.no_grad():
            spec = b["spec"]
            tgt = istft(b["spec_clean"])
            e1, m1, l1, _ = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], b["mask"])
            o1, z1 = T.gate_v4(spec, model.base.mask(spec.clone(), m1), e1, l1, torch.ones(nb, dtype=torch.bool, device=dev))
            e0, m0, l0, _ = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], torch.zeros_like(b["mask"]))
            o0, z0 = T.gate_v4(spec, model.base.mask(spec.clone(), m0), e0, l0, torch.zeros(nb, dtype=torch.bool, device=dev))
            sdp = T.si_sdr(istft(o1), tgt)
            sdp_nz = T.si_sdr(istft(e1), tgt)  # perfil sem ZERO (sempre ERB+DF)
            sd0 = T.si_sdr(istft(o0), tgt)
            sdin = T.si_sdr(istft(spec), tgt)
            ta = b["target_active"][:, :Tn]
            for k in range(nb):
                snr = float("inf") if bool(b["noise_silent"][k]) else float(b["noise_val"][k])
                rows.append({"snr": snr, "sd_in": float(sdin[k]), "sd0": float(sd0[k]), "sdp": float(sdp[k]),
                             "sdp_nozero": float(sdp_nz[k]), "zeroT": float(z1[k][ta[k]].float().mean()),
                             "zero_all": float(z1[k].float().mean()),
                             "lsnr_tgt_med_prof": float(l1.reshape(nb, -1)[k][ta[k]].median()),
                             "lsnr_tgt_med_noprof": float(l0.reshape(nb, -1)[k][ta[k]].median())})
    out = {}
    for name, lo, hi in (("snr<5", -99, 5), ("5-20", 5, 20), ("20-40", 20, 40.01), ("silent", 1e9, 1e10)):
        r = [x for x in rows if (lo <= x["snr"] < hi) or (name == "silent" and x["snr"] == float("inf"))]
        if not r:
            continue
        f = lambda k: float(np.mean([x[k] for x in r]))  # noqa: E731
        out[name] = {"n": len(r), "sd_in": f("sd_in"), "sd0": f("sd0"), "sdp": f("sdp"), "sdp_nozero": f("sdp_nozero"),
                     "dano(sd0-sdp)": f("sd0") - f("sdp"), "dano_sem_zero": f("sd0") - f("sdp_nozero"),
                     "guard_hinge_mean/10": float(np.mean([max(x["sd0"] - x["sdp"], 0) for x in r])) / 10,
                     "zeroT": f("zeroT"), "zero_all": f("zero_all"),
                     "lsnr_tgt_med_prof": f("lsnr_tgt_med_prof"), "lsnr_tgt_med_noprof": f("lsnr_tgt_med_noprof")}
    print(json.dumps(out, indent=1))
    Path(a.out).write_text(json.dumps({"bins": out, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
