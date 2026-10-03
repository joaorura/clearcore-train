import torch

from cctrain.models.ecapa import ECAPATDNN


def test_shapes_and_size():
    torch.manual_seed(0)
    m = ECAPATDNN().eval()
    out = m(torch.randn(2, 80, 198))
    assert out.shape == (2, 192)
    assert 5.5e6 <= sum(p.numel() for p in m.parameters()) <= 7.0e6


def test_context_modes_are_numerically_equivalent():
    torch.manual_seed(0)
    a = ECAPATDNN(context_mode="expand").eval()
    b = ECAPATDNN(context_mode="broadcast").eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 80, 300)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6, rtol=0)
