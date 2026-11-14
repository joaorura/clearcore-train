"""pro-v5-keep: correção do lsnr pelo perfil, gate straight-through e as perdas L_zero_tgt / L_p10.

(a) perfil desligado com a correção do lsnr (já treinada, ≠ 0) -> bit-idêntico ao DFNet3 publicado;
(b) passo 0 com perfil (correção zerada) -> bit-idêntico ao DFNet3;
(c) `gate_st` = `gate_v4` no forward; com τ > 0 o gradiente chega ao lsnr só nas amostras com perfil;
(d) L_zero_tgt / L_p10: valores conhecidos, gradiente empurra o lsnr para cima; quantil = numpy "lower";
(e) passo do treino com as perdas novas: a correção do lsnr recebe gradiente; flags desligadas = passo de antes;
(f) checkpoint com a correção: o eval (`tse_eval.build_model`) a reconstrói e reproduz o forward.
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
sys.path.insert(0, str(ROOT / "tests"))

from cctrain.dfvendor import ensure_df_importable  # noqa: E402

ensure_df_importable()

import cctrain.losses.zero_tgt as zt  # noqa: E402
from test_adapters_v5 import _eq, _freeze, _inputs, _p3, _randomize, env  # noqa: E402,F401
from test_v5_fast_step import _batch  # noqa: E402


def _p3c(env, film=True):
    from cctrain.models.pdfnet3_pro import attach_lsnr_correction

    m, cfg, sites = _p3(env, film)
    attach_lsnr_correction(m, 10.0)
    return m, cfg, sites


def test_a_profile_off_bit_identical_with_corr(env):
    _, ref, _ = env
    m, _, _ = _p3c(env)
    _randomize(m)
    assert m.lsnr_corr.weight.abs().sum() > 0
    _freeze(m)
    spec, ferb, fspec, emb = _inputs()
    with torch.no_grad():
        want = ref(spec.clone(), ferb, fspec)
        got = m(spec.clone(), ferb, fspec, emb, torch.zeros(3, 1))
    assert _eq(got, want)


def test_b_step0_profile_on_identical(env):
    _, ref, _ = env
    m, _, _ = _p3c(env)
    _freeze(m)
    spec, ferb, fspec, emb = _inputs()
    with torch.no_grad():
        got = m(spec.clone(), ferb, fspec, emb, torch.ones(3, 1))
        want = ref(spec.clone(), ferb, fspec)
    assert _eq(got, want)


def test_b2_trained_corr_moves_lsnr_only(env):
    m, _, _ = _p3c(env)
    _freeze(m)
    spec, ferb, fspec, emb = _inputs()
    on = torch.ones(3, 1)
    with torch.no_grad():
        a = m(spec.clone(), ferb, fspec, emb, on)
        m.lsnr_corr.bias.fill_(0.5)  # +5 dB
        b = m(spec.clone(), ferb, fspec, emb, on)
    torch.testing.assert_close(b[2], a[2] + 5.0)
    assert _eq([a[0], a[1], a[3]], [b[0], b[1], b[3]])


def _gate_inputs(seed=0, b=3, t=30):
    g = torch.Generator().manual_seed(seed)
    s = lambda: torch.randn(b, 1, t, 481, 2, generator=g)  # noqa: E731
    lsnr = torch.randn(b, t, 1, generator=g) * 20.0
    return s(), s(), s(), lsnr, torch.tensor([True, False, True])


@pytest.mark.parametrize("tau", [0.0, 2.0])
def test_c_gate_st_forward_equals_gate_v4(tau):
    from train_pro_v4 import gate_v4

    spec, spec_m, spec_e, lsnr, prof = _gate_inputs()
    want, zw = gate_v4(spec, spec_m, spec_e, lsnr, prof)
    got, zg, g = zt.gate_st(spec, spec_m, spec_e, lsnr, prof, tau)
    assert torch.equal(zw, zg)
    assert torch.equal(got.abs(), want.abs()) and torch.equal(got == 0, want == 0)
    assert torch.equal(g[1], torch.ones_like(g[1]))


def test_c_gate_st_grad_reaches_lsnr_only_with_profile():
    spec, spec_m, spec_e, lsnr, prof = _gate_inputs()
    lsnr = lsnr.clone().requires_grad_(True)
    out, _, _ = zt.gate_st(spec, spec_m, spec_e, lsnr, prof, 2.0)
    (out.square().sum()).backward()
    gr = lsnr.grad.reshape(3, -1)
    assert gr[0].abs().sum() > 0 and gr[2].abs().sum() > 0
    assert torch.equal(gr[1], torch.zeros_like(gr[1]))
    lsnr.grad = None
    out, _, _ = zt.gate_st(spec, spec_m, spec_e, lsnr, prof, 0.0)
    assert not out.requires_grad  # τ = 0: sem caminho (como o gate_v4, que usa lsnr.detach())


def test_d_zero_target_loss_values_and_grad():
    t = 10
    frames = torch.ones(2, t, dtype=torch.bool)
    e = torch.ones(2, t)
    lsnr = torch.tensor([[40.0] * t, [-40.0] * t]).requires_grad_(True)
    p = zt.pass_prob(lsnr, 2.0)
    g_eff = zt.effective_gain_db(e, e, p)
    loss, st = zt.zero_target_loss(g_eff, p, frames, -1.0)
    # item 0: passa, ganho 0 dB -> 0; item 1: ZERO, P(ZERO)≈1 e g_eff≈−40 -> 1 + 3,9
    assert abs(float(loss) - (0.0 + (1.0 + 3.9)) / 2) < 0.01
    assert abs(float(st["zero_tgt_p"]) - 0.5) < 1e-3
    lsnr2 = torch.full((1, t), -9.0, requires_grad=True)
    p2 = zt.pass_prob(lsnr2, 2.0)
    l2, _ = zt.zero_target_loss(zt.effective_gain_db(e[:1], e[:1], p2), p2, frames[:1], -1.0)
    l2.backward()
    assert (lsnr2.grad < 0).all()  # descer a perda = subir o lsnr (sair do ZERO)
    # quadros fora de `frames` não contam
    l3, _ = zt.zero_target_loss(g_eff, p, torch.tensor([[True] * t, [False] * t]), -1.0)
    assert float(l3) < 1e-6


def test_d_lowq_matches_numpy_lower():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 50, generator=g) * 10
    fr = torch.rand(4, 50, generator=g) > 0.3
    fr[3] = False
    q, n = zt.lowq_per_item(x, fr, 0.1)
    for i in range(3):
        assert float(q[i]) == pytest.approx(float(np.quantile(x[i][fr[i]].numpy(), 0.1, method="lower")))
    assert int(n[3]) == 0 and float(q[3]) == 0.0
    loss, st = zt.p10_loss(x, fr, 0.1, -3.0, 5)
    want = np.mean([max(0.0, -3.0 - float(q[i])) / 10 for i in range(3)])
    assert float(loss) == pytest.approx(want, rel=1e-6)


@pytest.fixture(scope="module")
def tp():
    return pytest.importorskip("train_pro_v5")


def _setup(tp, extra):
    a = tp.parse_args(["--variant", "P3", *extra])
    hp = tp.StageFHParams.from_config(tp.load_stage_f_config(), run_id="t", num_workers=0, batch_size=4)
    model, _, _, _, train_cond, _ = tp.build_v5(a, "cpu", hp, tp.Paths.from_env())
    tr = tp.prepare(model, train_cond)
    from df.loss import Istft

    istft = Istft(960, tp.HOP, torch.hann_window(960).sqrt())
    return a, hp, model, tr, istft, tp.ModelParams()


KEEP = ["--lsnr-corr", "10", "--gate-st-tau", "2", "--lambda-zero-tgt", "2", "--lambda-p10", "1",
        "--p10-min-frames", "5"]


def test_e_step_with_keep_losses_trains_lsnr_corr(tp):
    a, hp, model, tr, istft, params = _setup(tp, KEEP)
    names = {id(p): n for n, p in model.named_parameters()}
    assert any(names[id(p)].startswith("lsnr_corr.") for p in tr)
    b = _batch()
    parts = tp.step_v5(model, istft, params, hp, b, a, False)
    for k in ("zero_tgt", "p10", "zero_tgt_p", "g_p10_db", "raw_zero_tgt"):
        assert k in parts and torch.isfinite(parts[k]), k
    parts["total"].backward()
    assert model.lsnr_corr.weight.grad is not None and model.lsnr_corr.weight.grad.abs().sum() > 0


def test_e_flags_off_same_as_before(tp):
    """Sem as flags novas o passo é o de antes (mesmas chaves, mesma perda que com a correção zerada desligada)."""
    a, hp, model, tr, istft, params = _setup(tp, [])
    assert model.lsnr_corr is None
    b = _batch()
    torch.manual_seed(0)
    p0 = tp.step_v5(model, istft, params, hp, b, a, False)
    assert "zero_tgt" not in p0 and "zero_tgt_p" not in p0
    # mesma semente + correção zerada + flags de perda desligadas: perda idêntica
    a2, _, m2, _, _, _ = _setup(tp, ["--lsnr-corr", "10"])
    p1 = tp.step_v5(m2, istft, params, hp, b, a2, False)
    assert torch.equal(p0["total"], p1["total"])


def test_f_checkpoint_roundtrip_with_corr(env, tmp_path):
    from cctrain.eval.tse_eval import build_model as eval_build
    from cctrain.train.checkpoint import save_checkpoint

    m, cfg, _ = _p3c(env)
    _randomize(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    p = save_checkpoint(tmp_path, step=3, model=m.state_dict(), optimizer=opt, sampler_state={"step": 3},
                        cfg_hash="x", commit=None, extra={"adapter": cfg.to_dict(), "lsnr_corr": 10.0})
    m2 = eval_build(p, "v4", "cpu")
    assert m2.lsnr_corr is not None
    spec, ferb, fspec, emb = _inputs()
    _freeze(m)
    _freeze(m2)
    with torch.no_grad():
        a = m(spec.clone(), ferb, fspec, emb, torch.ones(3, 1))
        b = m2(spec.clone(), ferb, fspec, emb, torch.ones(3, 1))
    assert _eq(a, b)
