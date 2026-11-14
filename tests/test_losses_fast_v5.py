"""Paridade das perdas rápidas do pro-v5 (`cctrain.losses.fast_v5`) com as de referência: valores e gradientes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cctrain.losses.fast_v5 as fv5  # noqa: E402
from cctrain.losses.dfnet import asymmetric_formant_loss, dfnet_loss, local_snr_loss  # noqa: E402

train_pro_v4 = pytest.importorskip("train_pro_v4")
train_pro_v42 = pytest.importorskip("train_pro_v42")

RTOL = 1e-5
N_FFTS = (256, 512, 1024, 2048)


def _close(x: torch.Tensor, y: torch.Tensor, rtol: float = RTOL) -> None:
    x, y = x.detach().double(), y.detach().double()
    assert float((x - y).abs().max()) <= rtol * max(float(y.abs().max()), 1e-12), (x, y)


def _grad(f, x):
    x = x.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(f(x), x)
    return g


@pytest.mark.parametrize("max_freq", ["mixed", "full"])
@pytest.mark.parametrize("n1", ["some", "none", "all"])
def test_multires_matches_reference(max_freq, n1):
    g = torch.Generator().manual_seed(1)
    B, N = 5, 24_000
    tgt = 0.1 * torch.randn(B, N, generator=g)
    est = tgt + 0.05 * torch.randn(B, N, generator=g)
    mf = torch.tensor([8000.0, 24000.0, 16000.0, 24000.0, 4000.0]) if max_freq == "mixed" else torch.full((B,), 24000.0)
    nint = {"some": torch.tensor([0, 2, 1, 0, 3]), "none": torch.zeros(B, dtype=torch.long),
            "all": torch.ones(B, dtype=torch.long)}[n1]
    kw = dict(n_ffts=N_FFTS, gamma=0.3)

    def ref(e):
        mr = dfnet_loss(e, tgt, mf, factor=500.0, factor_complex=500.0, **kw)
        os_ = asymmetric_formant_loss(e, tgt, mf, factor=1000.0, formant_weight=2.5, extra_formant_penalty=1.0, **kw)
        idx = (nint >= 1).nonzero(as_tuple=True)[0]
        if idx.numel():
            os_ = os_ + (idx.numel() / B) * asymmetric_formant_loss(e[idx], tgt[idx], mf[idx], factor=1000.0,
                                                                    formant_weight=2.5, extra_formant_penalty=1.0, **kw)
        return mr, os_

    def fast(e):
        return fv5.multires_mr_os(e, tgt, mf, nint >= 1, n_ffts=N_FFTS, gamma=0.3, factor=500.0, factor_complex=500.0,
                                  os_factor=1000.0, formant_weight=2.5)

    (mr_r, os_r), (mr_f, os_f) = ref(est), fast(est)
    _close(mr_f, mr_r)
    _close(os_f, os_r)
    for i in range(2):
        _close(_grad(lambda e: fast(e)[i], est), _grad(lambda e: ref(e)[i], est))


def _spec(g, B=3, T=40, F=481, scale=1.0):
    return scale * torch.randn(B, 1, T, F, 2, generator=g)


@pytest.mark.parametrize("empty", [False, True])
def test_leak_ovl_lsnr_match_reference(empty):
    g = torch.Generator().manual_seed(2)
    B, T = 3, 40
    spec, clean, interf = _spec(g), _spec(g, scale=0.5), _spec(g, scale=0.5)
    out = 0.3 * _spec(g)
    f = (torch.rand(B, T, generator=g) > 0.5).float() * (0.0 if empty else 1.0)
    leak_r = lambda o: train_pro_v4.leak_loss(o, spec, f)  # noqa: E731
    leak_f = lambda o: fv5.masked_frac(fv5.leak_ratio(fv5.pow_bins(o).sum(-1), fv5.pow_bins(spec).sum(-1)), f)  # noqa: E731
    ovl_r = lambda o: train_pro_v4.ovl_loss(o, clean, interf, f)  # noqa: E731
    ovl_f = lambda o: fv5.ovl_value(fv5.pow_bins(o), fv5.pow_bins(clean), fv5.pow_bins(interf), f)  # noqa: E731
    for r, fa in ((leak_r, leak_f), (ovl_r, ovl_f)):
        _close(fa(out), r(out))
        _close(_grad(fa, out), _grad(r, out))
    lsnr = 10 * torch.randn(B, T, 1, generator=g)
    lt = 10 * torch.randn(B, T, generator=g)
    keep = f.bool()
    ls_r = lambda x: local_snr_loss(x, lt, keep, factor=2e-3)  # noqa: E731
    ls_f = lambda x: fv5.local_snr_value(x, lt, keep, 2e-3)  # noqa: E731
    _close(ls_f(lsnr), ls_r(lsnr))
    _close(_grad(ls_f, lsnr), _grad(ls_r, lsnr))


@pytest.mark.parametrize("empty", [False, True])
def test_turn_pair_matches_two_turn_losses(empty):
    g = torch.Generator().manual_seed(3)
    B, T = 4, 300
    clean = _spec(g, B, T)
    shadow = clean * torch.rand(B, 1, T, 1, 1, generator=g)
    ta = torch.rand(B, T, generator=g) > 0.3
    ia = torch.rand(B, T, generator=g) > 0.8
    v1 = ta & ~ia & (torch.rand(B, T, generator=g) > 0.2) & (not empty)
    v0 = ta & (torch.rand(B, T, generator=g) > 0.5) & (not empty)
    kw = dict(hinge_db=-1.5, boost=4.0, decay_s=1.5)

    def ref(sh):
        t, st = train_pro_v42.turn_loss(sh, clean, v1, ta, ia, **kw)
        n, sn = train_pro_v42.turn_loss(sh, clean, v0, ta, torch.zeros_like(ia), hinge_db=-1.0, boost=1.0, decay_s=1.0)
        return t, st, n, sn

    def fast(sh):
        return fv5.turn_pair(fv5.pow_bins(sh), fv5.pow_bins(clean), ta, train_pro_v42.since_interferer_s(ia), v1, v0,
                             hinge_t=-1.5, hinge_0=-1.0, boost=4.0, decay_s=1.5)

    tr, str_, nr, snr = ref(shadow)
    tf, stf, nf, snf = fast(shadow)
    _close(tf, tr)
    _close(nf, nr)
    for a_, b_ in ((stf["g_turn_db"], str_["g_turn_db"]), (stf["g_turn_early_db"], str_["g_turn_early_db"]),
                   (snf["g_turn_db"], snr["g_turn_db"])):
        if torch.isnan(b_):
            assert torch.isnan(a_)
        else:
            _close(a_, b_)
    for i in (0, 2):
        _close(_grad(lambda s: fast(s)[i], shadow), _grad(lambda s: ref(s)[i], shadow))
