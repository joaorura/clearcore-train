"""Paridade das primitivas de DSP em torch (cctrain.data.gpu.dsp) com a implementação CPU de mix_v4 / stage_f.

Rode em CPU: CUDA_VISIBLE_DEVICES="" nice -n 10 pytest tests/test_gpu_dsp.py -s
RIRs reais: `$CCTRAIN_DATA_DIR/rir/*/*.flac` (pula se ausente). O teste de GPU só roda com CUDA disponível.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from cctrain.data.audio import load_audio, resample
from cctrain.data.gpu import dsp
from cctrain.data.mix_v4 import HOP, SR, _conv, _ramp_env, early_rir, pink_noise
from cctrain.train.stage_f import lowpass48 as lowpass48_cpu

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
_REPORT: dict[str, float] = {}


def _rel(a: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(erro máximo absoluto, erro máximo relativo ao pico de `ref`)."""
    a, ref = np.asarray(a, np.float64), np.asarray(ref, np.float64)
    err = float(np.max(np.abs(a - ref)))
    return err, err / (float(np.max(np.abs(ref))) + 1e-30)


def _report(name: str, err: tuple[float, float]) -> None:
    _REPORT[name] = max(_REPORT.get(name, 0.0), err[1])
    print(f"\n[{name}] max_abs={err[0]:.3e} max_rel_pico={err[1]:.3e}")


def _real_rirs(k: int = 6) -> list[np.ndarray]:
    root = Path(os.environ.get("CCTRAIN_DATA_DIR", "")) / "rir"
    files = sorted(root.glob("*/*.flac"))
    if not files:
        pytest.skip(f"sem RIRs reais em {root}")
    step = max(1, len(files) // k)
    return [np.ascontiguousarray(resample(*load_audio(f)[:1], load_audio(f)[1], SR), np.float32) for f in files[::step][:k]]


# --- ramp_env ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,ramp", [(HOP * 100, 480), (HOP * 100 + 137, 480), (HOP * 50, 2), (HOP * 50, 479)])
def test_ramp_env(n, ramp):
    rng = np.random.default_rng(0)
    frames = n // HOP
    on = rng.random((5, frames)) < 0.5
    on[0] = True
    on[1] = False
    ref = np.stack([_ramp_env(o, n, ramp) for o in on])
    got = dsp.ramp_env(torch.from_numpy(on), n, ramp).numpy()
    err = _rel(got, ref)
    _report("ramp_env", err)
    assert got.shape == ref.shape and err[0] == 0.0


# --- fft_conv / early_len ------------------------------------------------------------------------------------------


def test_fft_conv_random_varlen():
    rng = np.random.default_rng(1)
    n = 48_000 * 3
    lens = np.array([1, 500, 4800, 48_000, 90_000])
    L = int(lens.max())
    x = rng.standard_normal((len(lens), n)) * 0.1
    h = np.zeros((len(lens), L), np.float32)
    for i, ln in enumerate(lens):
        h[i, :ln] = (rng.standard_normal(ln) * np.exp(-np.arange(ln) / max(ln / 6, 1))).astype(np.float32)
    ref = np.stack([_conv(x[i], h[i, : lens[i]]) for i in range(len(lens))])
    got = dsp.fft_conv(torch.from_numpy(x), torch.from_numpy(h), torch.from_numpy(lens)).numpy()
    err = _rel(got, ref)
    _report("fft_conv(random)", err)
    assert got.dtype == np.float64 and err[1] < 1e-5


def test_fft_conv_early_len_real_rirs():
    rirs = _real_rirs()
    rng = np.random.default_rng(2)
    n = 48_000 * 6
    lens = np.array([len(h) for h in rirs])
    L = int(lens.max())
    h = np.zeros((len(rirs), L), np.float32)
    for i, r in enumerate(rirs):
        h[i, : len(r)] = r
    x = rng.standard_normal((len(rirs), n)) * 0.1
    ht, lt = torch.from_numpy(h), torch.from_numpy(lens)
    ref = np.stack([_conv(x[i], rirs[i]) for i in range(len(rirs))])
    got = dsp.fft_conv(torch.from_numpy(x), ht, lt).numpy()
    _report("fft_conv(RIR real)", _rel(got, ref))
    assert _rel(got, ref)[1] < 1e-5
    el = dsp.early_len(ht, lt).numpy()
    ref_el = np.array([len(early_rir(r)) for r in rirs])
    assert (el == ref_el).all(), (el, ref_el)
    ref_e = np.stack([_conv(x[i], early_rir(rirs[i])) for i in range(len(rirs))])
    got_e = dsp.fft_conv(torch.from_numpy(x), ht, torch.from_numpy(el)).numpy()
    _report("fft_conv(early real)", _rel(got_e, ref_e))
    assert _rel(got_e, ref_e)[1] < 1e-5


def test_early_len_edge_cases():
    h = np.zeros((3, 10_000), np.float32)
    h[0, 10] = 1.0  # pico cedo: 10 + 2401
    h[1, 9000] = -1.0  # pico perto do fim: limitado a h_len
    h[2, 5] = 0.5
    h[2, 5000] = 0.5  # empate: primeiro máximo
    lens = np.array([10_000, 9_500, 10_000])
    got = dsp.early_len(torch.from_numpy(h), torch.from_numpy(lens)).numpy()
    ref = np.array([len(early_rir(h[i, : lens[i]])) for i in range(3)])
    assert (got == ref).all(), (got, ref)


# --- lowpass48 -----------------------------------------------------------------------------------------------------


def test_lowpass48_parity():
    rng = np.random.default_rng(3)
    n = 48_000 * 4
    cut = np.array([800.0, 3000.0, 7900.0, 8000.0, 12_000.0, 16_000.0, 23_900.0, 24_000.0])
    # ruído branco, rosa e com DC (as bordas é que mais divergem se zi/extensão estiverem errados)
    x = np.stack([rng.standard_normal(n) for _ in cut]).astype(np.float32)
    x[1] += 0.7
    x[2] = (x[2] * 0.05 + np.sin(2 * np.pi * 440 * np.arange(n) / SR)).astype(np.float32)
    ref = np.stack([lowpass48_cpu(x[i], float(cut[i])) for i in range(len(cut))])
    got = dsp.lowpass48(torch.from_numpy(x), torch.from_numpy(cut)).numpy()
    err = _rel(got, ref)
    edge = _rel(got[:, :2000], ref[:, :2000]), _rel(got[:, -2000:], ref[:, -2000:])
    _report("lowpass48", err)
    print(f"[lowpass48] bordas (2000 amostras) max_abs/rel: ini={edge[0]}, fim={edge[1]}")
    assert got.dtype == np.float32 and err[1] < 1e-5
    assert (got[7] == x[7]).all()  # corte >= 0.999*24k: x inalterado


def test_lowpass48_float64_low_cutoffs_chunked():
    """Em float64 (sem o arredondamento da saída f32) o erro é ~1e-14, inclusive com cortes baixos (warm-up longo) e
    sinais com DC; n grande o bastante para ativar o processamento em blocos."""
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(7)
    n = 48_000 * 3
    cut = np.array([200.0, 500.0, 1500.0, 20_000.0])
    x = rng.standard_normal((4, n)) + np.array([[0.0], [1.0], [-3.0], [0.2]])
    ref = np.stack([sosfiltfilt(butter(8, c / 24_000, output="sos"), x[i]) for i, c in enumerate(cut)])
    got = dsp.lowpass48(torch.from_numpy(x), torch.from_numpy(cut)).numpy()
    err = _rel(got, ref)
    _report("lowpass48(f64)", err)
    assert err[1] < 1e-11


def test_lowpass48_short_tail_lengths():
    rng = np.random.default_rng(4)
    for n in (28, 100, 4801):
        x = rng.standard_normal((2, n)).astype(np.float32)
        cut = np.array([5000.0, 9000.0])
        ref = np.stack([lowpass48_cpu(x[i], float(cut[i])) for i in range(2)])
        got = dsp.lowpass48(torch.from_numpy(x), torch.from_numpy(cut)).numpy()
        assert _rel(got, ref)[1] < 1e-5


# --- pink_from_white -----------------------------------------------------------------------------------------------


def test_pink_from_white():
    n = 48_000 * 6
    seeds = [10, 11, 12]
    ref = np.stack([pink_noise(n, np.random.default_rng(s)) for s in seeds])
    w = np.stack([np.random.default_rng(s).standard_normal(n) for s in seeds])
    got = dsp.pink_from_white(torch.from_numpy(w)).numpy()
    err = _rel(got, ref)
    _report("pink_from_white", err)
    assert err[1] < 1e-10


# --- timing --------------------------------------------------------------------------------------------------------


def _bench(M: int, n: int, device: str) -> dict[str, float]:
    rng = np.random.default_rng(5)
    x32 = torch.from_numpy(rng.standard_normal((M, n)).astype(np.float32)).to(device)
    x64 = x32.double()
    L = 48_000
    h = torch.from_numpy((rng.standard_normal((M, L)) * np.exp(-np.arange(L) / 8000)).astype(np.float32)).to(device)
    hl = torch.full((M,), L, dtype=torch.long, device=device)
    cut = torch.from_numpy(rng.uniform(4000, 20000, M))
    on = torch.from_numpy(rng.random((M, n // HOP)) < 0.5).to(device)
    fns = {
        "ramp_env": lambda: dsp.ramp_env(on, n),
        "fft_conv": lambda: dsp.fft_conv(x64, h, hl),
        "early_len": lambda: dsp.early_len(h, hl),
        "lowpass48": lambda: dsp.lowpass48(x32, cut),
        "pink_from_white": lambda: dsp.pink_from_white(x64),
    }
    out = {}
    for k, f in fns.items():
        f()  # aquecimento
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        f()
        if device == "cuda":
            torch.cuda.synchronize()
        out[k] = time.perf_counter() - t
    return out


def _cpu_reference_times(M: int, n: int) -> dict[str, float]:
    """Tempo da implementação CPU atual (loop por amostra, 1 thread de numpy/scipy) sobre M trilhas."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((M, n))
    L = 48_000
    h = (rng.standard_normal(L) * np.exp(-np.arange(L) / 8000)).astype(np.float32)
    cut = rng.uniform(4000, 20000, M)
    on = rng.random(n // HOP) < 0.5
    rg = np.random.default_rng(0)
    jobs = {
        "ramp_env": lambda: [_ramp_env(on, n) for _ in range(M)],
        "fft_conv": lambda: [_conv(x[i], h) for i in range(M)],
        "early_len": lambda: [early_rir(h) for _ in range(M)],
        "lowpass48": lambda: [lowpass48_cpu(x[i].astype(np.float32), float(cut[i])) for i in range(M)],
        "pink_from_white": lambda: [pink_noise(n, rg) for _ in range(M)],
    }
    t = {}
    for k, f in jobs.items():
        s = time.perf_counter()
        f()
        t[k] = time.perf_counter() - s
    return t


@pytest.mark.skipif(os.environ.get("CCTRAIN_BENCH_M") is None, reason="defina CCTRAIN_BENCH_M=96 para medir tempos")
def test_timing_batch():
    M, n = int(os.environ["CCTRAIN_BENCH_M"]), 288_000
    cpu_ref = _cpu_reference_times(M, n)
    torch_cpu = _bench(M, n, "cpu")
    gpu = _bench(M, n, "cuda") if torch.cuda.is_available() else {}
    print(f"\nTempos M={M}, n={n} (s)  {'função':<16}{'CPU atual':>11}{'torch CPU':>11}{'torch GPU':>11}")
    for k in cpu_ref:
        print(f"  {k:<16}{cpu_ref[k]:>11.3f}{torch_cpu[k]:>11.3f}{gpu.get(k, float('nan')):>11.3f}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_cuda_matches_cpu():
    rng = np.random.default_rng(6)
    M, n = 4, 48_000 * 2
    x = torch.from_numpy(rng.standard_normal((M, n)))
    h = torch.from_numpy((rng.standard_normal((M, 9000)) * 0.1).astype(np.float32))
    hl = torch.tensor([9000, 5000, 100, 9000])
    cut = torch.tensor([3000.0, 9000.0, 16000.0, 24000.0], dtype=torch.float64)
    cases = {
        "fft_conv": lambda d: dsp.fft_conv(x.to(d), h.to(d), hl.to(d)),
        "lowpass48": lambda d: dsp.lowpass48(x.float().to(d), cut),
        "pink": lambda d: dsp.pink_from_white(x.to(d)),
        "ramp": lambda d: dsp.ramp_env((x[:, : n // HOP] > 0).to(d), n),
    }
    for k, f in cases.items():
        a, b = f("cpu").numpy(), f("cuda").cpu().numpy()
        assert _rel(b, a)[1] < 1e-5, k
