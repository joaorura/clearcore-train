#!/usr/bin/env python
"""Amostras/s do gerador v4.2: caminho CPU (StageV4Dataset) vs --gpu-mix (receitas nos workers + GpuMixer).

Uso: PYTHONPATH=src python scripts/gpu_mix_bench.py --workers 1,2,4 --batches 20 [--batch 12 --segment-s 6]
Mede só o gerador (sem modelo): o laço consome lotes o mais rápido possível; no modo GPU inclui a mistura e o
`torch.cuda.synchronize()`. Também mede o tempo da mistura GPU isolada num lote fixo (limite superior do GPU)."""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_mix_equiv import make, mixer_kwargs  # noqa: E402

from cctrain.data.gpu.recipe import collate_recipes  # noqa: E402
from cctrain.data.workers import init_worker  # noqa: E402


def bench(ds, mixer, workers, batch, n_batches, warm, start, device, wire="dense"):
    sampler = list(range(start, start + (n_batches + warm) * batch))
    dl = DataLoader(ds if mixer else ds.cpu_view, batch_size=batch, sampler=sampler, num_workers=workers,
                    prefetch_factor=2 if workers else None, worker_init_fn=init_worker,
                    collate_fn=functools.partial(collate_recipes, sparse=wire == "sparse") if mixer else None, pin_memory=bool(mixer))
    it = iter(dl)
    t0 = None
    gpu_s = 0.0
    for k in range(n_batches + warm):
        if k == warm:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0, gpu_s = time.perf_counter(), 0.0
        b = next(it)
        if mixer:
            tg = time.perf_counter()
            out = mixer(b)
            mixer.fill_invalid(out, ds)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            gpu_s += time.perf_counter() - tg
    dt = time.perf_counter() - t0
    del it, dl
    return n_batches * batch / dt, gpu_s / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="1,2,4")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--segment-s", type=float, default=6.0)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modes", default="cpu,gpu")
    ap.add_argument("--dtype", choices=("f64", "f32"), default="f64")
    ap.add_argument("--lp-dtype", choices=("f64", "f32"), default=None)
    ap.add_argument("--label-dtype", choices=("f64", "f32"), default=None)
    ap.add_argument("--wire", choices=("dense", "sparse"), default="dense")
    ap.add_argument("--iso-only", action="store_true", help="só a mistura isolada (sem DataLoader)")
    a = ap.parse_args()
    ds, mixer, _ = make(a.device, mixer_kwargs(a.dtype, a.lp_dtype, a.label_dtype), batch_size=a.batch,
                        segment_s=a.segment_s)

    class _Cpu(torch.utils.data.Dataset):  # mesma amostra pelo caminho CPU de referência
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return ds.cpu_sample(i)

    ds.cpu_view = _Cpu()
    # mistura GPU isolada (lote fixo, já na memória)
    rb = collate_recipes([ds[i] for i in range(a.batch)], sparse=a.wire == "sparse")
    for _ in range(3):
        mixer(rb)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        mixer(rb)
    torch.cuda.synchronize()
    iso = (time.perf_counter() - t) / 10
    res = {"batch": a.batch, "segment_s": a.segment_s, "dtype": a.dtype, "lp_dtype": a.lp_dtype,
           "label_dtype": a.label_dtype, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
           "gpu_mix_isolated_ms_per_batch": 1000 * iso,
           "gpu_mix_isolated_samples_per_s": a.batch / iso,
           "gpu": torch.cuda.get_device_name(0), "runs": []}
    print(json.dumps(res), flush=True)
    if a.iso_only:
        return
    start = 5000
    for w in [int(x) for x in a.workers.split(",")]:
        for mode in a.modes.split(","):
            sps, gpu_ms = bench(ds, mixer if mode == "gpu" else None, w, a.batch, a.batches, a.warm, start,
                                a.device, a.wire)
            start += (a.batches + a.warm) * a.batch  # receitas novas (sem cache de página quente do mesmo item)
            row = {"mode": mode, "workers": w, "samples_per_s": round(sps, 2),
                   "gpu_mix_ms_per_batch": round(1000 * gpu_ms, 1) if mode == "gpu" else None}
            res["runs"].append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
