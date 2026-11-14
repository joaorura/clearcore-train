"""Equivalência do `mix_v4` otimizado (envelope por soma acumulada, convolução float32, RIRs em RAM) contra a
implementação anterior, mantida aqui só como referência."""

import numpy as np
import pytest
from scipy.signal import fftconvolve

from cctrain.data import mix_v4 as M


def _ramp_env_ref(on_frames, n, ramp=480):
    env = np.repeat(on_frames.astype(np.float64), M.HOP)[:n]
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    if ramp > 1:
        env = np.convolve(env, np.ones(ramp) / ramp, mode="same")
    return np.clip(env, 0.0, 1.0)


def _conv_ref(x, h):
    return fftconvolve(np.asarray(x, np.float64), np.asarray(h, np.float64))[: len(x)]


@pytest.mark.parametrize("ramp", [480, 481, 64, 2])
def test_ramp_env_matches_convolve(ramp):
    rng = np.random.default_rng(0)
    worst = 0.0
    for p in (0.1, 0.5, 0.9):
        on = rng.random(600) < p
        a, b = M._ramp_env(on, 288_000, ramp), _ramp_env_ref(on, 288_000, ramp)
        assert a.shape == b.shape
        worst = max(worst, float(np.max(np.abs(a - b))))
    print(f"ramp_env ramp={ramp}: max|diff| = {worst:.3g}")
    assert worst < 1e-12


def test_ramp_env_edges_and_short_signal():
    on = np.ones(10, bool)
    assert np.allclose(M._ramp_env(on, 10 * M.HOP), _ramp_env_ref(on, 10 * M.HOP), atol=1e-12)
    assert np.allclose(M._ramp_env(on, 4800 + 100), _ramp_env_ref(on, 4800 + 100), atol=1e-12)  # n > frames*HOP
    assert np.allclose(M._ramp_env(on, 300), _ramp_env_ref(on, 300), atol=1e-12)  # n < ramp


def test_conv_float32_close_to_float64():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(288_000) * np.repeat(rng.random(600), M.HOP)
    for n_h in (2449, 36_000):
        h = (rng.standard_normal(n_h) * np.exp(-np.arange(n_h) / (n_h / 6))).astype(np.float32)
        a, b = M._conv(x, h), _conv_ref(x, h)
        assert a.dtype == np.float64 and a.shape == b.shape
        rel = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
        print(f"conv |h|={n_h}: max|diff|/peak = {rel:.3g}")
        assert rel < 1e-5  # ~-100 dB; observado ~2e-7


def test_early_rir_is_a_view_with_same_values():
    h = np.random.default_rng(2).standard_normal(20_000).astype(np.float32)
    h[300] = 50.0
    e = M.early_rir(h)
    assert len(e) == 300 + 2400 + 1 and np.shares_memory(e, h)


def test_limit_threads_sets_one_thread():
    import os

    import torch

    from cctrain.data.workers import limit_threads

    before = torch.get_num_threads()
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    try:
        limit_threads(1)
        assert torch.get_num_threads() == 1
        assert os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == "1"
    finally:
        torch.set_num_threads(before)
        for k, v in env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# --- ponta a ponta com dados reais (pula quando o banco/dados não estão no disco) ----------------------------------

LABELS = ("target_active", "interf_active", "interf_active_wet45", "digital_silence", "n_interf", "has_target",
          "max_freq", "level_db", "sir_min_db", "noise_silent", "noise_val", "reverb", "pattern", "phase", "mask",
          "target_key", "target_path")
FLOATS = ("spec", "feat_erb", "feat_spec", "spec_clean", "spec_interf")


@pytest.fixture(scope="module")
def real_dataset(paths):
    from cctrain.m1 import pools
    from cctrain.train.embed_bank import ArrayEmbeddingBank

    bank_p = paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz"
    if not bank_p.is_file():
        pytest.skip(f"banco de embeddings ausente em {bank_p}")
    bank = ArrayEmbeddingBank.load(bank_p)
    speech = pools.by_split(pools.speech_records(paths), "train")
    noise_train, _ = pools.noise_pools()
    pl = M.build_pools(speech, noise_train, pools.rir_records(), bank, paths.data, seed=20261110)
    vp = M.V4Params(batch_size=16, total_steps=1000, phase1_frac=0.0, p_turns=0.3)
    return M.StageV4Dataset(pl, bank, paths.data, vp)


def test_real_samples_old_vs_new(real_dataset, monkeypatch, capsys):
    ds = real_dataset
    idx = list(range(40))
    new = [ds[i] for i in idx]
    with monkeypatch.context() as mp:
        mp.setattr(M, "_ramp_env", _ramp_env_ref)
        mp.setattr(M, "_conv", _conv_ref)
        old = [ds[i] for i in idx]
    worst = {k: 0.0 for k in FLOATS}
    n_rev = 0
    for a, b in zip(new, old):
        n_rev += bool(a["reverb"])
        for k in LABELS:
            assert a[k] == b[k] if isinstance(a[k], str) else np.array_equal(a[k], b[k], equal_nan=True), k
        for k in FLOATS:
            worst[k] = max(worst[k], float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))))
    with capsys.disabled():
        print(f"\n{len(idx)} amostras ({n_rev} com reverb): max|diff| por saída float = "
              + ", ".join(f"{k}={v:.3g}" for k, v in worst.items()))
    assert n_rev > 0
    assert max(worst.values()) < 1e-3
