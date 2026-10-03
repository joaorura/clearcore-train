"""Vazão do loader (amostras/s, s por hop produzido) e RSS com 4 workers e prefetch 2 (spec §5.5)."""

from __future__ import annotations

import time

import psutil
from torch.utils.data import DataLoader

HOPS_PER_SECOND = 100


def total_rss() -> int:
    p = psutil.Process()
    procs = [p, *p.children(recursive=True)]
    total = 0
    for proc in procs:
        try:
            total += proc.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def probe_loader(dataset, *, batch_size: int = 16, num_workers: int = 4, prefetch_factor: int = 2,
                 warmup_batches: int = 3, batches: int = 30) -> dict:
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor,
                    persistent_workers=True)
    it = iter(dl)
    for _ in range(warmup_batches):
        next(it)
    rss_max = 0
    t0 = time.perf_counter()
    for _ in range(batches):
        batch = next(it)
        rss_max = max(rss_max, total_rss())
    dt = time.perf_counter() - t0
    samples = batches * batch_size
    seconds_audio = float(batch["noisy"].shape[1]) / 48_000.0
    del it, dl
    return {
        "batch_size": batch_size, "num_workers": num_workers, "prefetch_factor": prefetch_factor,
        "batches": batches, "seconds": dt, "samples_per_s": samples / dt,
        "wall_s_per_hop": dt / (samples * seconds_audio * HOPS_PER_SECOND),
        "rss_max_gb": rss_max / 1024**3, "segment_s": seconds_audio,
    }
