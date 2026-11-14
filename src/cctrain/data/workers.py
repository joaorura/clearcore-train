"""Inicialização dos workers do DataLoader: 1 thread de BLAS/OpenMP/FFT por worker.

Cada worker é um processo que roda numpy/scipy/torch em série; sem limite, o OpenBLAS do numpy sobe uma pool com um
thread por núcleo POR WORKER (16 threads × N workers numa máquina de 16 núcleos) e os workers brigam entre si e com o
laço de treino. O PyTorch já fixa seus próprios threads em 1 dentro do worker; o resto (OpenBLAS do numpy, OpenMP de
outras libs, FFT do scipy) fica por conta deste `init_worker`, passado como `worker_init_fn=`.
"""

from __future__ import annotations

import ctypes
import os
import re

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
_OPENBLAS_SETTERS = ("openblas_set_num_threads64_", "openblas_set_num_threads", "scipy_openblas_set_num_threads64_",
                     "scipy_openblas_set_num_threads")
_keep: list = []  # referências que mantêm limites/contextos vivos durante o processo


def _openblas_limit(n: int) -> int:
    """Fixa os threads de cada OpenBLAS já carregado (as variáveis de ambiente só valem antes do import do numpy)."""
    done = 0
    try:
        with open("/proc/self/maps") as f:
            libs = sorted({line.split()[-1] for line in f if re.search(r"openblas", line)})
    except OSError:
        return 0
    for lib in libs:
        try:
            dll = ctypes.CDLL(lib)
        except OSError:
            continue
        for sym in _OPENBLAS_SETTERS:
            fn = getattr(dll, sym, None)
            if fn is not None:
                fn(ctypes.c_int(n))
                done += 1
                break
    return done


def limit_threads(n: int = 1) -> None:
    for k in _THREAD_ENV:  # vale para processos filhos e libs carregadas depois
        os.environ[k] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
    try:  # threadpoolctl cobre OpenBLAS/MKL/OpenMP de uma vez, quando instalado
        from threadpoolctl import threadpool_limits

        _keep.append(threadpool_limits(limits=n))
    except ImportError:
        _openblas_limit(n)
    try:  # scipy.fft (fftconvolve/oaconvolve): 1 worker explícito, em vez de depender do default
        import scipy.fft

        ctx = scipy.fft.set_workers(n)
        ctx.__enter__()
        _keep.append(ctx)
    except ImportError:
        pass


def init_worker(worker_id: int = 0) -> None:
    """`DataLoader(worker_init_fn=init_worker)`."""
    limit_threads(1)
