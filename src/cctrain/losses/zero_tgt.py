"""pro-v5-keep: perdas que penalizam o ZERO (e a atenuação) do alvo, com caminho diferenciável até o lsnr.

No P3 (DFNet3 congelado + Pro do zero) o estágio ZERO (lsnr < −10 dB com perfil) é decidido sobre `lsnr.detach()`: o
único gradiente que chega ao lsnr é o L_lsnr, cujo alvo é o SNR da MISTURA (interferente conta como ruído), que empurra
o lsnr para baixo justamente nos quadros de alvo em SNR baixo. Já no passo 0 (= DFNet3) ~14% dos quadros com alvo ativo
caem em ZERO, e nada no treino corrige isso. Aqui:

- `gate_st`: o mesmo gating do `gate_v4` (forward idêntico, bit a bit), com estimador straight-through nas amostras com
  perfil: o fator do quadro é `hard + p − p.detach()`, p = σ((lsnr + 10)/τ) (probabilidade de passar), e o gradiente
  das perdas sobre a saída (L_mr, L_leak, L_tail, ...) e sobre a sombra (L_turn, L_n0g) chega ao lsnr.
- `zero_target_loss` (L_zero_tgt): nos quadros com alvo ativo, média de P(ZERO) = 1 − p mais o hinge do ganho efetivo
  do alvo-sombra, g_eff = ganho antes do gate (máscara ERB + DF) + 20·log10(p + 0,01), abaixo de `hinge_db`.
- `p10_loss`: hinge no percentil baixo (p10) de g_eff por item (pior caso), além da média.

Tudo sem sincronizar com o host (CUDA Graph do StepRunner).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

ZERO_DB = -10.0


def pass_prob(lsnr: Tensor, tau: float) -> Tensor:
    """P(passar) suave do gate, [B,T]: σ((lsnr − (−10))/τ)."""
    return torch.sigmoid((lsnr.reshape(lsnr.shape[0], -1).float() - ZERO_DB) / tau)


def gate_st(spec, spec_m, spec_e, lsnr, profile, tau: float):
    """`gate_v4` com straight-through nas amostras com perfil. Devolve (out, zero, fator [B,T] das amostras com perfil;
    1 sem perfil). Forward: idêntico ao `gate_v4` (o fator vale exatamente 0 ou 1)."""
    nb = spec.shape[0]
    level = lsnr.detach().reshape(nb, -1)
    prof = profile.reshape(-1, 1).bool()
    zero = level < ZERO_DB
    none = (~zero) & (level > 30.0) & (~prof)
    erb_only = (~zero) & (~none) & (level > 20.0) & (~prof)

    def w(m):
        return m[:, None, :, None, None]

    hard = (~zero).to(spec_e.dtype)
    if tau > 0:
        p = pass_prob(lsnr, tau).to(spec_e.dtype)
        g = hard + (p - p.detach())
    else:
        g = hard
    g = torch.where(prof.expand_as(g), g, torch.ones_like(g))
    out_p = spec_e * w(g)
    out = torch.where(w(zero & ~prof), torch.zeros_like(spec_e), spec_e)
    out = torch.where(w(prof.expand_as(zero)), out_p, out)
    out = torch.where(w(erb_only), spec_m, out)
    out = torch.where(w(none), spec, out)
    return out, zero, g


def frame_energy(spec: Tensor) -> Tensor:
    """[B,1,T,F,2] -> energia por quadro [B,T]."""
    return (spec[:, 0, ..., 0].float() ** 2 + spec[:, 0, ..., 1].float() ** 2).sum(-1)


def effective_gain_db(e_pre: Tensor, e_clean: Tensor, p_pass: Tensor) -> Tensor:
    """Ganho efetivo do alvo-sombra por quadro (dB): o da máscara/DF antes do gate (limitado a −40) mais o do gate
    suave em amplitude, 20·log10(p + 0,01) (≈ 0 dB passando, ≈ −40 dB em ZERO), sempre diferenciável."""
    g_pre = torch.clamp(10 * torch.log10((e_pre + 1e-12) / (e_clean + 1e-12)), min=-40.0)
    return g_pre + 20 * torch.log10(p_pass + 0.01)


def zero_target_loss(g_eff: Tensor, p_pass: Tensor, frames: Tensor, hinge_db: float) -> tuple[Tensor, dict]:
    """L_zero_tgt = média nos quadros `frames` [B,T] de (1 − p) + relu(hinge − g_eff)/10."""
    f = frames.float()
    n = f.sum().clamp(min=1.0)
    p_zero = 1.0 - p_pass
    hinge = F.relu(hinge_db - g_eff) / 10.0
    loss = ((p_zero + hinge) * f).sum() / n
    with torch.no_grad():
        st = {"zero_tgt_p": (p_zero * f).sum() / n, "g_eff_mean_db": (g_eff * f).sum() / n}
    return loss, st


def lowq_per_item(g: Tensor, frames: Tensor, q: float) -> tuple[Tensor, Tensor]:
    """Quantil baixo `q` de g [B,T] nos quadros `frames` de cada item (o elemento de ordem floor(q·(n−1)) — sem
    interpolação, diferenciável pelo gather) e n por item. Itens sem quadro: valor 0 (ver o n)."""
    big = torch.full_like(g, 1e4)
    gs, _ = torch.sort(torch.where(frames, g, big), dim=1)
    n = frames.sum(1)
    k = torch.clamp(torch.floor(q * (n - 1).clamp(min=0).float()).long(), min=0)
    val = gs.gather(1, k[:, None])[:, 0]
    return torch.where(n > 0, val, torch.zeros_like(val)), n


def p10_loss(g_eff: Tensor, frames: Tensor, q: float, hinge_db: float, min_frames: int) -> tuple[Tensor, dict]:
    """Hinge no percentil baixo do ganho efetivo do alvo por item: média nos itens com ≥ min_frames quadros de
    relu(hinge − g_q)/10."""
    gq, n = lowq_per_item(g_eff, frames, q)
    ok = (n >= min_frames).float()
    m = ok.sum().clamp(min=1.0)
    loss = (F.relu(hinge_db - gq) / 10.0 * ok).sum() / m
    with torch.no_grad():
        st = {"g_p10_db": (gq * ok).sum() / m}
    return loss, st
