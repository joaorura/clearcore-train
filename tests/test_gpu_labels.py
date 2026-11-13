"""Paridade exata dos rótulos/VAD em lote (torch) com o gerador CPU `mix_v4`. Rode com CUDA_VISIBLE_DEVICES=""."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cctrain.data.gpu import labels as L
from cctrain.data import mix_v4 as C

HOP = C.HOP


def _cpu_frame_levels(tr, F):
    return np.stack([C.frame_levels(t, F) for t in tr])


def _cpu_act(tr, F, **kw):
    return np.stack([C.rel_activity(t, F, **kw) for t in tr])


def _cpu_rms(tr, F):
    return np.array([C.active_rms_db(t, F) for t in tr])


def _check(tr: np.ndarray, F: int | None = None) -> dict:
    """Compara tudo; devolve nº de quadros divergentes e erro máximo de nível."""
    n = tr.shape[1]
    F = n // HOP if F is None else F
    x = torch.from_numpy(tr)
    out = {"lv_err": float(np.abs(L.frame_levels(x, F).numpy() - _cpu_frame_levels(tr, F)).max()), "div": 0}
    for kw in ({}, {"range_db": 45.0}, {"range_db": 10.0, "floor_dbfs": -40.0}, {"floor_dbfs": -200.0}):
        g = L.rel_activity(x, F, **kw).numpy()
        out["div"] += int((g != _cpu_act(tr, F, **kw)).sum())
    out["rms_err"] = float(np.abs(L.active_rms_db(x, F).numpy() - _cpu_rms(tr, F)).max())
    n32 = tr.astype(np.float32)
    cpu_ds = np.stack([np.mean(t[: F * HOP].reshape(F, HOP).astype(np.float64) ** 2, axis=1) < C.SILENCE_MS for t in n32])
    out["div"] += int((L.digital_silence(torch.from_numpy(n32), F).numpy() != cpu_ds).sum())
    return out


def _assert_parity(r: dict) -> None:
    assert r["div"] == 0, r
    assert r["lv_err"] < 1e-9, r
    assert r["rms_err"] < 1e-9, r


def test_random_signals():
    rng = np.random.default_rng(0)
    n = 48_000 * 3
    amp = 10.0 ** rng.uniform(-6, 0, size=(16, 1))
    _assert_parity(_check(rng.standard_normal((16, n)) * amp))


def test_signals_with_silence_and_bursts():
    rng = np.random.default_rng(1)
    n, F = 48_000 * 4, 400
    tr = rng.standard_normal((12, n)) * 0.1
    tr[0] = 0.0  # tudo zero: piso, rms -200
    tr[1, : n // 2] = 0.0  # metade zero
    tr[2] *= 1e-5  # abaixo do piso de -75 dBFS
    tr[3] *= 1e-3
    tr[3, 1000:20_000] *= 1e3  # rajada sobre fundo baixo
    for k in range(4, 12):  # envelope por trechos, com buracos de silêncio digital
        env = np.repeat(rng.choice([0.0, 1e-6, 1e-3, 0.3], size=F), HOP)
        tr[k] *= env
    r = _check(tr)
    _assert_parity(r)
    x = torch.from_numpy(tr)
    assert L.active_rms_db(x, F)[0].item() == -200.0
    assert not L.rel_activity(x, F)[0].any() and not L.rel_activity(x, F)[2].any()


def test_unaligned_length_ignores_tail():
    rng = np.random.default_rng(2)
    tr = rng.standard_normal((4, 48_000 + 123)) * 0.05
    _assert_parity(_check(tr, F=100))


def test_threshold_ties_floor_edge():
    """Quadros constantes no limiar exato: >= no piso e no máximo-range (igual ao CPU)."""
    F = 50
    lv = np.full(F, -80.0)
    lv[0] = -45.0  # máximo; max-30 = -75 = piso -> quadros em -75 contam
    lv[1:10] = -75.0
    lv[10:20] = -75.0 - 1e-9
    tr = np.repeat(np.sqrt(10.0 ** (lv / 10.0)), HOP)[None]
    r = _check(tr)
    assert r["div"] == 0, r


def test_device_independence_dtype():
    rng = np.random.default_rng(3)
    tr = rng.standard_normal((3, 48_000)) * 0.2
    a = L.rel_activity(torch.from_numpy(tr).float(), 100)  # entrada float32 é promovida a float64
    assert a.dtype == torch.bool and L.frame_levels(torch.from_numpy(tr).float(), 100).dtype == torch.float64


def _real_tracks():
    data = os.environ.get("CCTRAIN_DATA_DIR")
    if not data:
        pytest.skip("CCTRAIN_DATA_DIR não definido")
    root = Path(data) / "raw"
    files = sorted(root.glob("librispeech-dev-clean/**/*.flac"))[:40]
    files += sorted(root.glob("dns-noise-fullband-cc0/**/*.wav"))[:6]
    files += sorted(root.glob("fsd50k-eval-cc0/**/*.wav"))[:6]
    if not files:
        pytest.skip("áudio real inacessível")
    from cctrain.data.audio import load_audio, resample

    n = 48_000 * 8
    out = []
    for f in files:
        try:
            x, sr = load_audio(f)
        except Exception:
            continue
        x = resample(x, sr, 48_000).astype(np.float64)
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))  # cauda de zeros: silêncio digital real
        out.append(x[:n])
    if len(out) < 4:
        pytest.skip("poucos arquivos legíveis")
    return np.stack(out)


@pytest.mark.parametrize("gain_db", [0.0, -30.0, -60.0, -90.0])
def test_real_audio(gain_db):
    tr = _real_tracks() * 10.0 ** (gain_db / 20.0)
    r = _check(tr)
    print(f"real audio gain={gain_db} dB tracks={len(tr)} frames={tr.shape[1] // HOP}:", r)
    _assert_parity(r)
