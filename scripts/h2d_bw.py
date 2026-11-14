"""Banda H2D (memória pinned -> GPU) por placa, sozinha e com várias placas copiando ao mesmo tempo.

Uma thread por placa, cada uma na sua stream, copiando o mesmo bloco pinned de `--mb` MB em laço por `--secs` s.
Uso: `python scripts/h2d_bw.py --mb 128 --secs 5` (mede cada placa visível sozinha e depois todas juntas).
"""

from __future__ import annotations

import argparse
import threading
import time

import torch


def run(devs: list[int], mb: int, secs: float) -> dict[int, float]:
    n = mb * 2**20
    src = {d: torch.empty(n, dtype=torch.uint8).pin_memory() for d in devs}
    dst = {d: torch.empty(n, dtype=torch.uint8, device=f"cuda:{d}") for d in devs}
    for d in devs:  # aquecimento
        dst[d].copy_(src[d], non_blocking=True)
        torch.cuda.synchronize(d)
    out: dict[int, float] = {}
    barrier = threading.Barrier(len(devs))

    def work(d: int):
        s = torch.cuda.Stream(d)
        with torch.cuda.device(d), torch.cuda.stream(s):
            barrier.wait()
            t0, k = time.perf_counter(), 0
            while time.perf_counter() - t0 < secs:
                dst[d].copy_(src[d], non_blocking=True)
                s.synchronize()
                k += 1
            out[d] = k * n / (time.perf_counter() - t0) / 1e9

    ths = [threading.Thread(target=work, args=(d,)) for d in devs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=128)
    ap.add_argument("--secs", type=float, default=5.0)
    a = ap.parse_args()
    devs = list(range(torch.cuda.device_count()))
    for d in devs:
        print(f"sozinha cuda:{d} ({torch.cuda.get_device_properties(d).name}): {run([d], a.mb, a.secs)[d]:.3f} GB/s",
              flush=True)
    if len(devs) > 1:
        r = run(devs, a.mb, a.secs)
        print("juntas: " + " ".join(f"cuda:{d}={v:.3f}" for d, v in r.items()) + f" | soma {sum(r.values()):.3f} GB/s")


if __name__ == "__main__":
    main()
