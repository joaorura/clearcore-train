import torch
from torch import nn

from cctrain.models.state_hash import state_dict_sha256


def test_hash_is_stable_and_detects_any_change():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    h1 = state_dict_sha256(m)
    m.eval()
    with torch.no_grad():
        m(torch.randn(5, 4))
    assert state_dict_sha256(m) == h1
    with torch.no_grad():
        m[0].weight[0, 0] += 1e-7
    assert state_dict_sha256(m) != h1
