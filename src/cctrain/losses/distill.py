"""Destilação do estágio F (spec §5.4): `L_T = λ_T · MR(Ŝ_aluno, Ŝ_DFNet3 congelado)` só nas amostras `mask=0`.

O professor é o DFNet3 upstream congelado; aluno e professor passam pelo mesmo gating emulado (§3.5), então a perda
compara o que o runtime produziria. `MR` é a perda multi-resolução do DFNet3 (`losses.dfnet.dfnet_loss`) com os dois
fatores iguais a `λ_T`, bins acima de `max_freq` fora. A normalização é a da perda base (média sobre o lote inteiro):
cada amostra `mask=0` pesa `1/B`, então um lote com poucas amostras sem perfil recebe pouca destilação, e um lote
sem nenhuma recebe zero (com grafo, para o `backward` não falhar).
"""

from __future__ import annotations

import torch
from torch import Tensor

from cctrain.losses.dfnet import GAMMA, N_FFTS, dfnet_loss

LAMBDA_T = 500.0


def distill_loss(student: Tensor, teacher: Tensor, mask: Tensor, max_freq: float | Tensor | None = None, *,
                 lambda_t: float = LAMBDA_T, n_ffts: tuple[int, ...] = N_FFTS, gamma: float = GAMMA) -> Tensor:
    """`student`: `[B, T]` no tempo; `teacher`: `[n0, T]`, uma linha por amostra com `mask == 0`, na ordem do lote
    (o professor só roda nelas); `mask`: `[B]` ou `[B, 1]` (1 = com perfil)."""
    b = student.shape[0]
    idx = (mask.reshape(b) == 0).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return (student * 0.0).sum()
    mf = None
    if max_freq is not None:
        mf = torch.as_tensor(max_freq, dtype=torch.float32, device=student.device)
        mf = mf.reshape(-1)[idx] if mf.numel() == b else mf
    if teacher.shape[0] != idx.numel():
        raise ValueError(f"professor com {teacher.shape[0]} linhas, mas há {idx.numel()} amostras mask=0")
    mr = dfnet_loss(student[idx], teacher.detach(), mf, n_ffts=n_ffts, gamma=gamma, factor=lambda_t,
                    factor_complex=lambda_t)
    return mr * (idx.numel() / b)


__all__ = ["LAMBDA_T", "distill_loss"]
