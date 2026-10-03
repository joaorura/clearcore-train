import pytest
import torch

from cctrain.dfvendor import ensure_df_importable
from cctrain.losses.dfnet import (
    dfnet_loss,
    gate_loss,
    local_snr_loss,
    over_suppression_loss,
    stage_c_loss,
)

ensure_df_importable()
from df.loss import LocalSnrLoss, MultiResSpecLoss  # noqa: E402


def _wave(seed, b=2, n=24_000):
    g = torch.Generator().manual_seed(seed)
    return 0.1 * torch.randn(b, n, generator=g)


def _upstream():
    return MultiResSpecLoss([256, 512, 1024, 2048], gamma=0.3, factor=500.0, f_complex=500.0)


def test_dfnet_loss_matches_upstream_multires_loss():
    pred, target = _wave(0), _wave(1)
    ref = _upstream()(pred, target)
    out = dfnet_loss(pred, target, None)
    assert abs(float(out) - float(ref)) <= 1e-6 * max(1.0, abs(float(ref)))
    assert abs(float(dfnet_loss(pred, target, 24_000.0)) - float(ref)) <= 1e-6 * max(1.0, abs(float(ref)))


def test_dfnet_loss_gradient_matches_upstream():
    pred, target = _wave(2), _wave(3)
    a, b = pred.clone().requires_grad_(), pred.clone().requires_grad_()
    _upstream()(a, target).backward()
    dfnet_loss(b, target, None).backward()
    assert float((a.grad - b.grad).abs().max()) <= 1e-6 * float(a.grad.abs().max())  # float32, ordem das somas


def test_dfnet_loss_excludes_bins_above_max_freq():
    target = _wave(4)
    pred_low = target.clone()
    t = torch.arange(target.shape[-1]) / 48_000.0
    hi = 0.05 * torch.sin(2 * torch.pi * 18_000.0 * t)  # energia só acima de 8 kHz
    pred = target + hi
    full = float(dfnet_loss(pred, target, None))
    cut = float(dfnet_loss(pred, target, 8_000.0))
    assert full > 0.0
    assert cut < 0.02 * full  # o tom de 18 kHz some quase por inteiro (vazamento espectral à parte)
    assert float(dfnet_loss(pred_low, target, 8_000.0)) == 0.0


def test_dfnet_loss_accepts_per_sample_max_freq():
    pred, target = _wave(5), _wave(6)
    mf = torch.tensor([8_000.0, 24_000.0])
    per = dfnet_loss(pred, target, mf)
    only_second = dfnet_loss(pred[1:], target[1:], None)
    only_first = dfnet_loss(pred[:1], target[:1], 8_000.0)
    assert torch.isfinite(per)
    assert abs(float(per) - 0.5 * (float(only_first) + float(only_second))) <= 1e-5 * float(per)


def test_over_suppression_is_zero_when_not_attenuating_and_asymmetric():
    target = _wave(7)
    assert float(over_suppression_loss(target.clone(), target)) == 0.0
    over = over_suppression_loss(0.5 * target, target)  # atenuou o alvo: penaliza
    under = over_suppression_loss(1.5 * target, target)  # amplificou: sem penalidade
    assert float(over) > 0.0 and float(under) == 0.0


def test_over_suppression_ignores_bins_above_max_freq():
    target = _wave(8)
    t = torch.arange(target.shape[-1]) / 48_000.0
    pred = target - 0.05 * torch.sin(2 * torch.pi * 18_000.0 * t)
    assert float(over_suppression_loss(pred, target, 8_000.0)) < 0.05 * float(over_suppression_loss(pred, target))


def test_gate_loss_penalizes_lsnr_above_30_only_on_active_frames():
    lsnr = torch.tensor([[[35.0], [40.0], [10.0], [50.0]]])  # [B, T, 1]
    active = torch.tensor([[True, True, True, False]])
    expected = 1e-3 * ((35.0 - 30.0) ** 2 + (40.0 - 30.0) ** 2 + 0.0) / 3
    assert abs(float(gate_loss(lsnr, active)) - expected) < 1e-7
    assert float(gate_loss(lsnr, torch.zeros_like(active))) == 0.0
    lsnr.requires_grad_()
    gate_loss(lsnr, active).backward()
    assert lsnr.grad[0, 3, 0] == 0.0 and lsnr.grad[0, 0, 0] != 0.0


def test_local_snr_loss_matches_upstream_without_mask_and_masks_silence():
    g = torch.Generator().manual_seed(9)
    pred, tgt = torch.randn(2, 10, 1, generator=g), torch.randn(2, 10, generator=g)
    ref = LocalSnrLoss(1e-3)(pred, tgt)
    assert abs(float(local_snr_loss(pred, tgt)) - float(ref)) < 1e-9
    keep = torch.ones(2, 10, dtype=torch.bool)
    keep[1] = False  # a amostra 1 é "sem alvo": mascarada
    only0 = LocalSnrLoss(1e-3)(pred[:1], tgt[:1])
    assert abs(float(local_snr_loss(pred, tgt, keep)) - float(only0)) < 1e-9
    assert float(local_snr_loss(pred, tgt, torch.zeros_like(keep))) == 0.0


def test_stage_c_loss_is_the_sum_of_its_components():
    pred, target = _wave(10), _wave(11)
    lsnr = torch.randn(2, 20, 1)
    lsnr_t = torch.randn(2, 20)
    active = torch.ones(2, 20, dtype=torch.bool)
    parts = stage_c_loss(pred, target, lsnr, lsnr_t, lsnr_mask=torch.ones_like(active), interferer_frames=active,
                         max_freq=None)
    assert set(parts) == {"total", "mr", "os", "gate", "lsnr"}
    total = parts["mr"] + parts["os"] + parts["gate"] + parts["lsnr"]
    assert torch.allclose(parts["total"], total)
    assert all(torch.isfinite(v) for v in parts.values())


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_max_freq_must_be_positive(bad):
    with pytest.raises(ValueError):
        dfnet_loss(_wave(0), _wave(1), bad)
