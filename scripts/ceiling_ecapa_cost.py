#!/usr/bin/env python3
"""Custo do ECAPA (SpeechBrain spkrec-ecapa-voxceleb portado) no esquema de janelas deslizantes: tempo de CPU por
janela (PyTorch, fp32, 1 e 4 threads), MACs por janela (contagem por hooks em Conv1d/Linear + front-end) e o custo por
segundo de áudio com uma janela a cada 100 ms.

Uso: python scripts/ceiling_ecapa_cost.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import ceiling_common as C
from cctrain.paths import Paths


def count_macs(fbank, ecapa, n: int) -> dict:
    macs = {"conv": 0, "linear": 0}

    def conv_hook(m, inp, out):
        macs["conv"] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin_hook(m, inp, out):
        macs["linear"] += out.numel() * m.in_features

    hs = []
    for mod in list(fbank.modules()) + list(ecapa.modules()):
        if isinstance(mod, nn.Conv1d):
            hs.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hs.append(mod.register_forward_hook(lin_hook))
    x = torch.randn(1, n) * 0.05
    with torch.no_grad():
        f = fbank(x)
        frames = f.shape[-1]
        fb = sum(macs.values())
        ecapa(f)
    for h in hs:
        h.remove()
    # front-end (funcional, fora dos hooks): conv de Fourier 402×400 + MatMul mel 80×201 por quadro
    mel = frames * (402 * 400 + 80 * 201)
    total = sum(macs.values()) + mel
    return {"frames": frames, "fbank_macs": fb + mel, "total_macs": total}


def time_window(fbank, ecapa, n: int, threads: int, reps: int) -> float:
    torch.set_num_threads(threads)
    x = torch.randn(1, n) * 0.05
    with torch.no_grad():
        for _ in range(3):
            ecapa(fbank(x))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ecapa(fbank(x))
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    fbank, ecapa = C.load_ecapa(Paths.from_env().data, "cpu")
    rep = {"torch": torch.__version__, "params": sum(p.numel() for p in ecapa.parameters()), "windows": {}}
    for w in (0.5, 1.0, 1.5):
        n = int(w * C.ECAPA_SR)
        row = count_macs(fbank, ecapa, n)
        for th in (1, 4):
            t = time_window(fbank, ecapa, n, th, a.reps)
            row[f"ms_per_window_{th}t"] = round(1000 * t, 2)
            # uma janela a cada 100 ms -> 10 janelas por segundo de áudio
            row[f"cpu_s_per_audio_s_hop100ms_{th}t"] = round(10 * t, 3)
        row["gmacs_per_audio_s_hop100ms"] = round(10 * row["total_macs"] / 1e9, 2)
        row["gmacs_per_window"] = round(row["total_macs"] / 1e9, 3)
        rep["windows"][f"{w}s"] = row
        print(w, row, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
