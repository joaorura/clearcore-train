"""Features em torch (cctrain.data.gpu.features) vs libdf (cctrain.data.features): erro máximo abs/rel."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from cctrain.data.features import analysis as cpu_analysis
from cctrain.data.features import extract_features as cpu_extract
from cctrain.data.gpu import features as gf

N = 6 * 48_000
TOL = 1e-4


def _real_signals() -> list[tuple[str, np.ndarray]]:
    root = Path(os.environ.get("CCTRAIN_DATA_DIR", "data")) / "synthetic_pairs_preview"
    out = []
    for pat in ("*_clean.wav", "*_noisy.wav"):
        for f in sorted(root.glob(pat))[:2]:
            x, sr = sf.read(f, dtype="float32")
            if sr != 48_000 or len(x) < N:
                continue
            out.append((f.name, np.ascontiguousarray(x[:N] if x.ndim == 1 else x[:N, 0])))
    return out


def _signals() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(1234)
    sigs = []
    for i in range(8):
        amp = 10 ** rng.uniform(-3, -0.3)
        x = amp * rng.standard_normal(N)
        if i % 2:  # modulação de envelope + trechos quase silenciosos
            x *= np.repeat(rng.random(N // 4800 + 1) ** 4, 4800)[:N]
        sigs.append((f"rand{i}", x.astype(np.float32)))
    sigs += _real_signals()
    sigs.append(("silence", np.zeros(N, np.float32)))
    # fala-like: silêncio digital no meio de ruído (testa estados das EMAs e o log de zero)
    mid = (0.05 * rng.standard_normal(N)).astype(np.float32)
    mid[N // 3 : 2 * N // 3] = 0
    sigs.append(("noise_gap", mid))
    return sigs


def _err(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    scale = np.abs(b).max() + 1e-30
    elem = (d / (np.abs(b) + 1e-3 * scale)).max()
    return float(d.max()), float(d.max() / scale), float(elem)


def _ref(x: np.ndarray):
    f = cpu_extract(x)
    return f.spec, f.feat_erb, f.feat_spec


def test_shapes_and_dtypes():
    x = torch.randn(3, 4 * 48_000 + 100)  # resto de 100 amostras descartado
    spec, erb, fs = gf.extract_features(x)
    F = x.shape[1] // 480
    assert spec.shape == (3, 1, F, 481, 2) and erb.shape == (3, 1, F, 32) and fs.shape == (3, 1, F, 96, 2)
    assert all(t.dtype == torch.float32 for t in (spec, erb, fs))
    assert torch.equal(gf.analysis(x), spec)
    assert cpu_analysis(x[0].numpy()).shape == (1, F, 481, 2)


def test_matches_libdf_batch():
    sigs = _signals()
    x = np.stack([s for _, s in sigs])
    with torch.no_grad():
        spec, erb, fs = (t.numpy() for t in gf.extract_features(torch.from_numpy(x)))
        spec_a = gf.analysis(torch.from_numpy(x)).numpy()
    assert np.array_equal(spec_a, spec)
    worst = {"spec": 0.0, "feat_erb": 0.0, "feat_spec": 0.0}
    lines = []
    for i, (name, s) in enumerate(sigs):
        rs, re_, rf = _ref(s)
        for key, a, b in (("spec", spec[i], rs), ("feat_erb", erb[i], re_), ("feat_spec", fs[i], rf)):
            assert a.shape == b.shape, (name, key, a.shape, b.shape)
            assert np.isfinite(a).all(), (name, key)
            mabs, mrel, melem = _err(a, b)
            worst[key] = max(worst[key], mrel)
            lines.append(f"{name:28s} {key:9s} max_abs={mabs:.3e} rel_to_max={mrel:.3e} rel_elem={melem:.3e}")
            assert mrel < TOL, lines[-1]
    print("\n" + "\n".join(lines))
    print("worst rel_to_max:", worst)


def test_silence_exact_values():
    z = torch.zeros(2, N)
    spec, erb, fs = gf.extract_features(z)
    assert spec.abs().max() == 0
    # erb_db = -100 constante; EMA parte de linspace(-60,-90) -> saída finita e igual ao libdf
    r = cpu_extract(np.zeros(N, np.float32))
    assert np.allclose(erb[0].numpy(), r.feat_erb, rtol=0, atol=1e-5)
    assert np.isfinite(fs.numpy()).all() and float(fs.abs().max()) == 0.0
    assert float(erb[0, 0, 0, 0]) == pytest.approx(float(r.feat_erb[0, 0, 0]), abs=1e-6)


@pytest.mark.parametrize("n", [480, 960, 480 * 7 + 13, 480 * 100])
def test_short_and_odd_lengths(n):
    x = (0.1 * np.random.default_rng(n).standard_normal(n)).astype(np.float32)
    spec, erb, fs = (t.numpy() for t in gf.extract_features(torch.from_numpy(x)[None]))
    rs, re_, rf = _ref(x)
    for a, b in ((spec[0], rs), (erb[0], re_), (fs[0], rf)):
        assert _err(a, b)[1] < TOL


def test_batch_independence():
    rng = np.random.default_rng(7)
    x = torch.from_numpy((0.1 * rng.standard_normal((4, 48_000))).astype(np.float32))
    full = gf.extract_features(x)
    for i in range(4):
        one = gf.extract_features(x[i : i + 1])
        for a, b in zip(full, one):
            assert torch.allclose(a[i : i + 1], b, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_cuda_matches_cpu():
    x = torch.randn(4, 48_000)
    a = gf.extract_features(x)
    b = gf.extract_features(x.cuda())
    for u, v in zip(a, b):
        assert _err(v.cpu().numpy(), u.numpy())[1] < TOL
