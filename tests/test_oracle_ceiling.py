"""exp/oracle-ceiling: oráculo de presença (cctrain.models.oracle).

(a) passo 0 (zero-init): saída com oráculo bit-idêntica à sem oráculo, com e sem perfil;
(b) sem perfil, mesmo com o oráculo treinado: bit-idêntica (× máscara);
(c) o oráculo recebe gradiente; as GRUs liberadas também;
(d) alinhamento do eval: o encoder no índice t vê o rótulo do quadro de saída t − 2; corrupção do O2 (~10%, atraso).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cctrain.dfvendor import ensure_df_importable  # noqa: E402

ensure_df_importable()

from test_adapters_v5 import _freeze, _inputs, _p3, env  # noqa: E402,F401


def _oracle_model(env, cfg=None):
    from cctrain.models.oracle import OracleConfig, attach_oracle
    from cctrain.models.pdfnet3_pro import attach_lsnr_correction

    m, _, _ = _p3(env)
    attach_lsnr_correction(m, 10.0)
    attach_oracle(m, cfg or OracleConfig("clean"))
    return _freeze(m)


def _orc(b, t, seed=3):
    from cctrain.models.oracle import features

    g = torch.Generator().manual_seed(seed)
    return features(torch.rand(b, t, generator=g) > 0.5, torch.rand(b, t, generator=g) > 0.5)


def test_a_step0_identity(env):
    m = _oracle_model(env)
    spec, ferb, fspec, emb = _inputs()
    for mv in (1.0, 0.0):
        mask = torch.full((3, 1), mv)
        with torch.no_grad():
            r0 = m(spec.clone(), ferb, fspec, emb, mask)
            r1 = m(spec.clone(), ferb, fspec, emb, mask, oracle=_orc(3, spec.shape[2]))
        for x, y in zip(r0, r1):
            assert torch.equal(x, y)


def test_b_noprofile_identity_trained(env):
    m = _oracle_model(env)
    with torch.no_grad():
        for p in m.oracle_cond.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    spec, ferb, fspec, emb = _inputs()
    mask = torch.tensor([[0.0], [1.0], [0.0]])
    o = _orc(3, spec.shape[2])
    with torch.no_grad():
        r0 = m(spec.clone(), ferb, fspec, emb, mask)
        r1 = m(spec.clone(), ferb, fspec, emb, mask, oracle=o)
    for x, y in zip(r0, r1):
        assert torch.equal(x[0], y[0]) and torch.equal(x[2], y[2])
    assert not torch.equal(r0[2][1], r1[2][1])  # com perfil o lsnr muda


def test_c_gradients(env):
    import train_pro_v5 as tp

    m = _oracle_model(env)
    trainable = tp.prepare(m, True)
    names = {id(p): n for n, p in m.named_parameters()}
    assert any(names[id(p)].startswith("oracle_cond.") for p in trainable)
    gru = tp.free_gru_params(m)
    assert len(gru) >= 12 and len(gru) % 4 == 0  # 4 tensores por camada
    spec, ferb, fspec, emb = _inputs()
    m.train()
    # um passo para tirar o oráculo do zero (gradiente do peso = dL/dy · o)
    out = m(spec.clone(), ferb, fspec, emb, torch.ones(3, 1), oracle=_orc(3, spec.shape[2]))
    (out[2].sum() + out[1].sum() + out[3].float().pow(2).sum()).backward()
    assert m.oracle_cond.lsnr.weight.grad.abs().sum() > 0
    assert m.oracle_cond.enc_b.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in gru)


def test_d_eval_alignment_and_noise():
    from cctrain.models.oracle import CONV_LOOKAHEAD, OracleConfig, corrupt_np, eval_oracle_frames

    n = 400 * 480
    rng = np.random.default_rng(0)
    tgt = np.zeros(n, np.float32)
    tgt[100 * 480:200 * 480] = rng.standard_normal(100 * 480) * 0.1
    ints = np.zeros((1, n), np.float32)
    ints[0, 250 * 480:300 * 480] = rng.standard_normal(50 * 480) * 0.1
    active = np.arange(400)
    o = eval_oracle_frames(tgt, ints, active, OracleConfig("clean"), 0)
    assert o.shape == (400, 3)
    assert np.flatnonzero(o[:, 0]).tolist() == list(range(100 + CONV_LOOKAHEAD, 200 + CONV_LOOKAHEAD))
    assert np.flatnonzero(o[:, 1]).tolist() == list(range(250 + CONV_LOOKAHEAD, 300 + CONV_LOOKAHEAD))
    cfg = OracleConfig("noisy", delay_frames=15, err_frac=0.1, burst_frames=20)
    lab = np.zeros(200_000, bool)
    x = corrupt_np(lab, cfg, np.random.default_rng(1))
    assert 0.07 < x.mean() < 0.11
    lab = np.zeros(100, bool)
    lab[50] = True
    assert np.flatnonzero(corrupt_np(lab, OracleConfig("noisy", delay_frames=15), np.random.default_rng(0))).tolist() == [65]


def test_e_corrupt_torch_matches_rate():
    from cctrain.models.oracle import OracleConfig, corrupt_torch

    cfg = OracleConfig("noisy", delay_frames=15, err_frac=0.1, burst_frames=20)
    g = torch.Generator().manual_seed(0)
    x = corrupt_torch(torch.zeros(64, 600, dtype=torch.bool), cfg, g)
    assert 0.07 < x.float().mean().item() < 0.11
    lab = torch.zeros(2, 100, dtype=torch.bool)
    lab[:, 50] = True
    y = corrupt_torch(lab, OracleConfig("noisy", delay_frames=15), None)
    assert y[0].nonzero().flatten().tolist() == [65]
