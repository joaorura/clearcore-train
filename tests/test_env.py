import numpy as np
import pytest
import torch


def test_numpy_below_2():
    assert int(np.__version__.split(".")[0]) < 2


def test_deepfilterlib_version():
    from importlib.metadata import version

    assert version("deepfilterlib") == "0.5.6"


@pytest.mark.gpu
def test_cuda_blackwell():
    assert torch.cuda.is_available(), "CUDA indisponível"
    assert tuple(int(x) for x in torch.version.cuda.split(".")[:2]) >= (12, 8)
    assert torch.cuda.get_device_capability() == (12, 0)
