"""Perdas do estágio C (spec §5.4): multi-resolução do DFNet3, over-suppression, L_gate e LocalSnr.

`dfnet_loss` reproduz `df.loss.MultiResSpecLoss` do upstream vendorizado (FFT 256/512/1024/2048, gamma 0,3,
fator de módulo e complexo 500, STFT Hann normalizada com hop = n_fft/4). A única diferença é `max_freq`: bins
acima dele ficam fora da média (spec §5.4, "bins acima de max_freq excluídos de todas as perdas"); sem `max_freq`
(ou com `max_freq >= sr/2`) o resultado é o do upstream. Com `max_freq`, a média é sobre os bins mantidos, por
amostra, e depois sobre o lote. Entradas no domínio do tempo `[B, T]` (saída da iSTFT com o gating emulado).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from cctrain.dfvendor import ensure_df_importable

SR = 48_000
N_FFTS = (256, 512, 1024, 2048)
GAMMA = 0.3
FACTOR = 500.0
FACTOR_COMPLEX = 500.0
OS_FACTOR = 500.0
GATE_FACTOR = 1e-3
GATE_LSNR_DB = 30.0
LSNR_FACTOR = 1e-3
_EPS = 1e-12


def _stft(x: Tensor, n_fft: int) -> Tensor:
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return torch.stft(x.reshape(-1, x.shape[-1]), n_fft=n_fft, hop_length=n_fft // 4, window=window,
                      normalized=True, return_complex=True)


def _bin_weights(max_freq: float | Tensor | None, n_fft: int, batch: int, sr: int, device) -> Tensor | None:
    """Máscara de bins mantidos `[B, F, 1]` (None: todos). `max_freq` em Hz, escalar ou `[B]`."""
    if max_freq is None:
        return None
    mf = torch.as_tensor(max_freq, dtype=torch.float32, device=device)
    if bool((mf <= 0).any()):
        raise ValueError("max_freq deve ser positivo (Hz)")
    mf = mf.reshape(-1).expand(batch) if mf.numel() in (1, batch) else mf
    if mf.numel() != batch:
        raise ValueError("max_freq deve ser escalar ou ter um valor por amostra")
    if bool((mf >= sr / 2).all()):
        return None
    freqs = torch.arange(n_fft // 2 + 1, device=device, dtype=torch.float32) * (sr / n_fft)
    return (freqs[None, :] <= mf[:, None]).to(torch.float32).unsqueeze(-1)


def _masked_mean(err: Tensor, w: Tensor | None) -> Tensor:
    """`err`: `[B, F, T]` ou `[B, F, T, 2]`; média sobre os bins mantidos por amostra, depois sobre o lote."""
    if w is None:
        return err.mean()
    if err.ndim == 4:
        w = w.unsqueeze(-1)
    per = (err * w).flatten(1).sum(dim=1) / (w.expand_as(err).flatten(1).sum(dim=1).clamp_min(1.0))
    return per.mean()


def dfnet_loss(pred: Tensor, target: Tensor, max_freq: float | Tensor | None = None, *, sr: int = SR,
               n_ffts: tuple[int, ...] = N_FFTS, gamma: float = GAMMA, factor: float = FACTOR,
               factor_complex: float = FACTOR_COMPLEX) -> Tensor:
    angle = _angle()
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for n_fft in n_ffts:
        y, s = _stft(pred, n_fft), _stft(target, n_fft)
        y_abs, s_abs = y.abs(), s.abs()
        if gamma != 1:
            y_abs, s_abs = y_abs.clamp_min(_EPS).pow(gamma), s_abs.clamp_min(_EPS).pow(gamma)
        w = _bin_weights(max_freq, n_fft, pred.shape[0], sr, pred.device)
        loss = loss + _masked_mean((y_abs - s_abs) ** 2, w) * factor
        if factor_complex:
            if gamma != 1:
                y = y_abs * torch.exp(1j * angle.apply(y))
                s = s_abs * torch.exp(1j * angle.apply(s))
            loss = loss + _masked_mean((torch.view_as_real(y) - torch.view_as_real(s)) ** 2, w) * factor_complex
    return loss


def over_suppression_loss(pred: Tensor, target: Tensor, max_freq: float | Tensor | None = None, *, sr: int = SR,
                          n_ffts: tuple[int, ...] = N_FFTS, gamma: float = GAMMA, factor: float = OS_FACTOR
                          ) -> Tensor:
    """`L_OS = λ_OS · média_res média_TF h(|S|^γ − |Ŝ|^γ)²`, `h(x) = max(x, 0)`: só pune módulo abaixo do alvo."""
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for n_fft in n_ffts:
        y_abs = _stft(pred, n_fft).abs().clamp_min(_EPS).pow(gamma)
        s_abs = _stft(target, n_fft).abs().clamp_min(_EPS).pow(gamma)
        w = _bin_weights(max_freq, n_fft, pred.shape[0], sr, pred.device)
        loss = loss + _masked_mean(F.relu(s_abs - y_abs) ** 2, w)
    return loss / len(n_ffts) * factor


def gate_loss(lsnr: Tensor, interferer_frames: Tensor, *, factor: float = GATE_FACTOR,
              limit_db: float = GATE_LSNR_DB) -> Tensor:
    """`L_gate = λ_g · média(relu(lsnr − 30)²)` nos quadros com interferente ativo e perfil presente (§3.5).

    `lsnr`: `[B, T, 1]` (ou `[B, T]`); `interferer_frames`: `[B, T]` booleano. Sem quadro ativo, zero com grafo."""
    x = lsnr.reshape(lsnr.shape[0], -1)
    m = interferer_frames.to(x.dtype)
    n = m.sum()
    if float(n) == 0.0:
        return (x * 0.0).sum()
    return factor * ((F.relu(x - limit_db) ** 2) * m).sum() / n


def local_snr_loss(lsnr: Tensor, target_lsnr: Tensor, keep: Tensor | None = None, *, factor: float = LSNR_FACTOR
                   ) -> Tensor:
    """`LocalSnrLoss` do upstream (MSE · fator), com máscara opcional de quadros: a classe "sem alvo" (silêncio)
    é mascarada para não ensinar o modelo a forçar -15 dB (§3.5). Sem máscara, é idêntica à do upstream."""
    x = lsnr.reshape(lsnr.shape[0], -1)
    if keep is None:
        return F.mse_loss(x, target_lsnr) * factor
    k = keep.to(x.dtype)
    n = k.sum()
    if float(n) == 0.0:
        return (x * 0.0).sum()
    return factor * (((x - target_lsnr) ** 2) * k).sum() / n


def stage_c_loss(pred: Tensor, target: Tensor, lsnr: Tensor, lsnr_target: Tensor, *, lsnr_mask: Tensor,
                 interferer_frames: Tensor, max_freq: float | Tensor | None = None) -> dict[str, Tensor]:
    """Perda do estágio C sem destilação: MR do DFNet3 + over-suppression + L_gate + LocalSnr (mascarada)."""
    mr = dfnet_loss(pred, target, max_freq)
    os_ = over_suppression_loss(pred, target, max_freq)
    gate = gate_loss(lsnr, interferer_frames)
    ls = local_snr_loss(lsnr, lsnr_target, lsnr_mask)
    return {"total": mr + os_ + gate + ls, "mr": mr, "os": os_, "gate": gate, "lsnr": ls}


def _angle():
    ensure_df_importable()
    from df.utils import angle

    return angle
