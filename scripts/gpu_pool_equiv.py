#!/usr/bin/env python
"""Pool residente (`--gpu-pool-gb`): equivalência com o `GpuMixer` atual, erro do codec e lei de amostragem.

  equiv  receitas do pool -> `PoolMixer`, contra as MESMAS receitas materializadas no CPU (áudio lido do disco, como
         o caminho atual) -> `GpuMixer`: max|diff| relativo ao pico, features, % de quadros de rótulo divergentes.
  codec  erro do int16 (escala por clipe) nos clipes que não são sem perda (MP3 / reamostrados), por tipo de fonte.
  stats  lei de amostragem pool vs original (sem áudio): TV de falantes-alvo/interferentes/faixas/nº de
         interferentes, contra a TV entre duas faixas de índices do próprio original (ruído amostral).

Uso: PYTHONPATH=src python scripts/gpu_pool_equiv.py --mode equiv --gb 1 --n 96 --batch 12
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_mix_equiv import AUDIO, FEATS, LABELS, make, mixer_kwargs  # noqa: E402

from cctrain.data.gpu.mixer import PoolMixer  # noqa: E402
from cctrain.data.gpu.pool import GpuClipPool, StageV4PoolRecipeDataset, encode  # noqa: E402
from cctrain.data.gpu.recipe import StageV4RecipeDataset, collate_recipes  # noqa: E402


def build(dev: str, gb: float, dtype: str, batch: int, codec: str = "exact"):
    ds, mixer, _ = make(dev, mixer_kwargs(dtype), batch_size=batch)
    pool = GpuClipPool(ds, dev, gb, codec=codec, seed=7)
    pds = StageV4PoolRecipeDataset(ds, pool.index)
    pm = PoolMixer.from_dataset(pds, dev, pool=pool, **mixer_kwargs(dtype))
    return ds, mixer, pool, pds, pm


def equiv(a) -> dict:
    ds, mixer, pool, pds, pm = build(a.device, a.gb, a.dtype, a.batch, a.codec)
    wa = {k: 0.0 for k in AUDIO}
    wf = {k: 0.0 for k in FEATS}
    lab = {k: 0 for k in LABELS}
    tot, inval, meta_bad, n_tilt, n_pad, refill = 0, 0, 0, 0, 0, 0
    for s in range(a.start, a.start + a.n, a.batch):
        items = [pds[i] for i in range(s, s + a.batch)]
        n_tilt += sum(int(np.sum(it["dry_ref"][:, 4])) for it in items)
        n_pad += sum(int(np.sum(it["dry_ref"][:, 3] < it["n_samples"])) for it in items)
        op = pm(collate_recipes(items), debug=True)
        od = mixer(collate_recipes([pds.materialize(it, pool) for it in items]), debug=True)
        inval += int((~op["valid"]).sum())
        for k in AUDIO:
            x, y = op["debug"][k].double(), od["debug"][k].double()
            pk = y.abs().amax(dim=1).clamp(min=1e-30)
            wa[k] = max(wa[k], float(((x - y).abs().amax(dim=1) / pk).max()))
        for k in FEATS:
            x, y = op[k].double(), od[k].double()
            wf[k] = max(wf[k], float((x - y).abs().max() / y.abs().max().clamp(min=1e-30)))
        for k in LABELS:
            lab[k] += int((op[k] != od[k]).sum())
        tot += op["target_active"].numel()
        for k in ("n_interf", "has_target", "pattern", "reverb", "mask", "noise_silent", "phase", "embedding"):
            meta_bad += int(not torch.equal(op[k], od[k]))
        refill += pm.fill_invalid(op, pds)
        assert bool(op["valid"].all())
    return {"refilled": refill, "n": a.n, "dtype": a.dtype, "codec": a.codec, "audio_rel_peak": wa, "feat_rel_peak": wf,
            "label_frames_diff_pct": {k: 100.0 * v / max(tot, 1) for k, v in lab.items()}, "meta_mismatch": meta_bad,
            "gpu_invalid": inval, "tracks_tilted": n_tilt, "tracks_padded": n_pad, "pool": pool.summary()}


def refresh(a) -> dict:
    """Renovação agressiva (`--refresh` %/min, quarentena curta) com DataLoader de 2 workers: cada lote do pool tem de
    bater com a mesma receita materializada do disco (prova que nenhum slot é reaproveitado com receita viva)."""
    ds, mixer, pool, pds, pm = build(a.device, a.gb, a.dtype, a.batch, a.codec)
    pool.refresh_pct, pool.delay = a.refresh, 2 * 2 * 2 + 2
    dl = torch.utils.data.DataLoader(pds, batch_size=a.batch, sampler=list(range(a.start, a.start + a.n)),
                                     num_workers=2, prefetch_factor=2, collate_fn=lambda x: x, persistent_workers=True)
    it = iter(dl)
    pool.start()
    worst, nb = 0.0, 0
    for items in it:
        op = pm(collate_recipes(items), debug=True)
        od = mixer(collate_recipes([pds.materialize(r, pool) for r in items]), debug=True)
        for k in AUDIO:
            x, y = op["debug"][k].double(), od["debug"][k].double()
            worst = max(worst, float(((x - y).abs().amax(dim=1) / y.abs().amax(dim=1).clamp(min=1e-30)).max()))
        nb += 1
    pool.close()
    return {"batches": nb, "audio_rel_peak_worst": worst, "pool": pool.summary()}


def codec(a) -> dict:
    ds, _, _ = make("cpu", batch_size=a.batch)
    from cctrain.data.gpu.pool import PoolIndex

    ix = PoolIndex(ds.pools)
    rng = np.random.default_rng(3)
    res: dict = collections.defaultdict(lambda: {"n": 0, "lossless": 0, "max_err_rel_peak": 0.0,
                                                 "max_err_rel_rms": 0.0, "max_err_abs": 0.0})
    for gid in rng.choice(len(ix.paths), a.n, replace=False):
        p = ix.paths[gid]
        inf = sf.info(str(ds.data_root / p))
        kind = f"{inf.subtype}@{inf.samplerate}"
        x = ds._load48(p).astype(np.float32)
        is16, sc, q = encode(x, "int16")
        y = q.astype(np.float32) * np.float32(sc)
        r = res[kind]
        r["n"] += 1
        lossless = bool(np.array_equal(y, x))
        r["lossless"] += lossless
        e = float(np.max(np.abs(y.astype(np.float64) - x)))
        pk = float(np.max(np.abs(x))) or 1.0
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) or 1.0
        r["max_err_abs"] = max(r["max_err_abs"], e)
        r["max_err_rel_peak"] = max(r["max_err_rel_peak"], e / pk)
        r["max_err_rel_rms"] = max(r["max_err_rel_rms"], e / rms)
    return dict(res)


def stats(a) -> dict:
    """Original (áudio trocado por uns do comprimento certo: só a lei dos sorteios importa) vs pool (CPU, sem GPU)."""
    ds, _, _ = make("cpu", batch_size=a.batch)
    L = {}

    def fake48(rel):
        if rel not in L:
            inf = sf.info(str(ds.data_root / rel))
            L[rel] = int(round(inf.frames * 48000 / inf.samplerate))
        return np.ones(L[rel], np.float32)

    ds._load48 = fake48
    pool = GpuClipPool(ds, "cpu", a.gb, seed=11, threads=4)
    pds = StageV4PoolRecipeDataset(ds, pool.index)

    def law(d, lo, hi):
        c = {"target": collections.Counter(), "interf": collections.Counter(), "tier": collections.Counter(),
             "n_interf": collections.Counter(), "noise_mode": collections.Counter()}
        pick = type(d)._pick_interferer

        def counted(rng, tkey, taken, p_hard):
            k = pick(d, rng, tkey, taken, p_hard)
            c["interf"][k] += 1
            return k

        d._pick_interferer = counted
        for i in range(lo, hi):
            r = d[i]
            c["target"][r["target_key"]] += 1
            c["tier"][d.slots[i % d.p.batch_size]] += 1
            c["n_interf"][int(r["n_interf"])] += 1
            c["noise_mode"]["silent" if r["noise_silent"] else "snr"] += 1
        del d._pick_interferer
        return c

    def tv(c1, c2):
        n1, n2 = sum(c1.values()), sum(c2.values())
        return 0.5 * sum(abs(c1[k] / n1 - c2[k] / n2) for k in set(c1) | set(c2))

    N = a.n
    o1, o2, pp = law(ds, 0, N), law(ds, N, 2 * N), law(pds, 2 * N, 3 * N)
    return {"n": N, "pool": pool.summary(),
            "tv_orig_vs_orig": {k: tv(o1[k], o2[k]) for k in o1},
            "tv_pool_vs_orig": {k: tv(pp[k], o2[k]) for k in o1},
            "n_interf_pool": dict(pp["n_interf"]), "n_interf_orig": dict(o2["n_interf"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("equiv", "codec", "stats", "refresh"), default="equiv")
    ap.add_argument("--refresh", type=float, default=200.0, help="--mode refresh: %% do pool por minuto")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--gb", type=float, default=1.0)
    ap.add_argument("--dtype", choices=("f64", "f32"), default="f32")
    ap.add_argument("--codec", choices=("exact", "int16"), default="exact")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = {"equiv": equiv, "codec": codec, "stats": stats, "refresh": refresh}[a.mode](a)
    txt = json.dumps(res, indent=1, default=float)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt)


if __name__ == "__main__":
    main()
