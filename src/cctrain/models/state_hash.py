"""SHA-256 do state_dict (integridade da base congelada no estágio C, spec §5.3)."""

from __future__ import annotations

import hashlib

from torch import nn


def state_dict_sha256(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()
