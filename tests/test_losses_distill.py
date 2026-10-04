import pytest
import torch

from cctrain.losses.dfnet import dfnet_loss
from cctrain.losses.distill import distill_loss


def _w(seed, b, n=12_000):
    return 0.1 * torch.randn(b, n, generator=torch.Generator().manual_seed(seed))


def test_distill_loss_is_zero_when_no_sample_is_without_profile_and_keeps_the_graph():
    s = _w(0, 3).requires_grad_(True)
    loss = distill_loss(s, _w(1, 0), torch.ones(3, 1), 24_000.0)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert s.grad is not None and float(s.grad.abs().sum()) == 0.0


def test_distill_loss_uses_only_the_mask0_rows_weighted_by_their_share_of_the_batch():
    s, t = _w(0, 4), _w(1, 2)
    mask = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mf = torch.tensor([24_000.0, 12_000.0, 24_000.0, 8_000.0])
    out = distill_loss(s, t, mask, mf, lambda_t=500.0)
    ref = dfnet_loss(s[[1, 3]], t, mf[[1, 3]], factor=500.0, factor_complex=500.0) * 0.5
    assert torch.allclose(out, ref)
    # as linhas com perfil não entram: trocá-las não muda a perda
    s2 = s.clone()
    s2[[0, 2]] = _w(5, 2)
    assert torch.allclose(distill_loss(s2, t, mask, mf), out)


def test_distill_loss_gives_gradient_only_to_mask0_rows_and_is_zero_for_identical_signals():
    s = _w(0, 4).requires_grad_(True)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    t = _w(1, 2)
    distill_loss(s, t, mask, None).backward()
    assert float(s.grad[[0, 2]].abs().sum()) == 0.0 and float(s.grad[[1, 3]].abs().sum()) > 0.0
    same = _w(7, 2)
    full = torch.stack([_w(9, 1)[0], same[0], _w(9, 1)[0], same[1]])
    assert float(distill_loss(full, same, mask, None)) == pytest.approx(0.0, abs=1e-9)


def test_distill_loss_rejects_a_teacher_with_the_wrong_row_count():
    with pytest.raises(ValueError, match="mask=0"):
        distill_loss(_w(0, 4), _w(1, 3), torch.tensor([1.0, 0.0, 1.0, 0.0]))
