"""Pool residente (`--gpu-pool-gb`): mistura do pool = mesma receita materializada do disco no `GpuMixer`, também com
renovação agressiva; codec sem perda nos PCM16 a 48 kHz. Pula sem o banco de embeddings ou sem ~3,5 GB livres."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _need(paths):
    if not (paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz").is_file():
        pytest.skip("banco de embeddings ausente")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4.5e9:
        pytest.skip("sem GPU com ~4,5 GB livres")


def test_encode_lossless_and_scaled():
    from cctrain.data.gpu.pool import encode

    x = (np.arange(-32768, 32768, 7, dtype=np.int64) / 32768.0).astype(np.float32)
    is16, sc, q = encode(x, "exact")
    assert is16 and np.array_equal(q.astype(np.float32) * np.float32(sc), x)
    y = (x * 0.3 + 1e-7).astype(np.float32)
    assert not encode(y, "exact")[0]
    is16, sc, q = encode(y, "int16")
    assert is16 and np.max(np.abs(q * sc - y)) <= 0.5 * sc * 1.001


def test_pool_matches_gpu_mixer(paths):
    _need(paths)
    from gpu_pool_equiv import equiv, refresh

    a = SimpleNamespace(device="cuda", gb=3.4, dtype="f32", batch=6, codec="exact", n=24, start=0, refresh=300.0)
    r = equiv(a)
    assert max(r["audio_rel_peak"].values()) <= 1e-5 and max(r["feat_rel_peak"].values()) <= 1e-5
    assert max(r["label_frames_diff_pct"].values()) == 0.0 and r["meta_mismatch"] == 0
    a.n = 60
    r = refresh(a)
    assert r["pool"]["swaps"] > 0 and r["pool"]["gen_mismatch_rows"] == 0 and r["audio_rel_peak_worst"] <= 1e-5
