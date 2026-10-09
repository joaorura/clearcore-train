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


def formant_frequency_weights(
    n_fft: int,
    sr: int = SR,
    device: torch.device | None = None,
    *,
    formant_band: tuple[float, float] = (300.0, 3400.0),
    formant_weight: float = 2.5,
    outer_low_freq: float = 150.0,
    outer_low_weight: float = 0.2,
    outer_high_freq: float = 5000.0,
    outer_high_weight: float = 0.2,
) -> tuple[Tensor, Tensor]:
    """Máscara de frequências psicoacústicas para proteção de formantes da fala.

    - Banda de formantes (300 Hz a 3400 Hz): peso elevado (ex: 2.5x).
    - Bandas externas (< 150 Hz e > 5000 Hz): peso reduzido (ex: 0.2x), tolerando
      cancelamento agressivo de ruídos de ventoinha e sub-graves.
    - Faixas de transição: interpolação linear suave.

    Retorna `(weights, formant_mask)` com forma `[F, 1]`.
    """
    freqs = torch.arange(n_fft // 2 + 1, device=device, dtype=torch.float32) * (sr / n_fft)
    weights = torch.ones_like(freqs)

    f_low, f_fstart = outer_low_freq, formant_band[0]
    f_fend, f_high = formant_band[1], outer_high_freq

    # Banda externa grave (< f_low): tolerância alta (peso reduzido)
    weights[freqs < f_low] = outer_low_weight

    # Transição f_low -> f_fstart
    mask_trans_low = (freqs >= f_low) & (freqs < f_fstart)
    if bool(mask_trans_low.any()) and (f_fstart > f_low):
        t = (freqs[mask_trans_low] - f_low) / (f_fstart - f_low)
        weights[mask_trans_low] = outer_low_weight + t * (formant_weight - outer_low_weight)

    # Banda de formantes de voz (300 a 3400 Hz): peso elevado
    mask_formant = (freqs >= f_fstart) & (freqs <= f_fend)
    weights[mask_formant] = formant_weight

    # Transição f_fend -> f_high
    mask_trans_high = (freqs > f_fend) & (freqs <= f_high)
    if bool(mask_trans_high.any()) and (f_high > f_fend):
        t = (freqs[mask_trans_high] - f_fend) / (f_high - f_fend)
        weights[mask_trans_high] = formant_weight + t * (outer_high_weight - formant_weight)

    # Banda externa aguda (> f_high): tolerância alta (peso reduzido)
    weights[freqs > f_high] = outer_high_weight

    return weights.unsqueeze(-1), mask_formant.to(torch.float32).unsqueeze(-1)


def asymmetric_formant_loss(
    pred: Tensor,
    target: Tensor,
    max_freq: float | Tensor | None = None,
    *,
    sr: int = SR,
    n_ffts: tuple[int, ...] = N_FFTS,
    gamma: float = GAMMA,
    factor: float = OS_FACTOR,
    formant_band: tuple[float, float] = (300.0, 3400.0),
    formant_weight: float = 2.5,
    outer_low_freq: float = 150.0,
    outer_low_weight: float = 0.2,
    outer_high_freq: float = 5000.0,
    outer_high_weight: float = 0.2,
    extra_formant_penalty: float = 1.0,
) -> Tensor:
    """Loss Assimétrica Ponderada por Formantes da Fala.

    Aplica máscara psicoacústica em múltiplas resoluções STFT:
    - Banda de formantes (300 Hz a 3.400 Hz): peso elevado (2.5x) com penalidade
      quadrática extra (1.0 + extra_formant_penalty * err) para punir rigorosamente
      over-suppression e evitar corte de vogais e consoantes.
    - Bandas externas (< 150 Hz e > 5.000 Hz): peso reduzido (0.2x), com alta tolerância
      para cancelamento agressivo de ruídos de ventoinha e sub-graves.
    - Só penaliza atenuação abaixo do alvo (h(x) = relu(x)), como L_OS.
    """
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for n_fft in n_ffts:
        y_abs = _stft(pred, n_fft).abs().clamp_min(_EPS).pow(gamma)
        s_abs = _stft(target, n_fft).abs().clamp_min(_EPS).pow(gamma)
        w = _bin_weights(max_freq, n_fft, pred.shape[0], sr, pred.device)
        fw, fmask = formant_frequency_weights(
            n_fft,
            sr,
            pred.device,
            formant_band=formant_band,
            formant_weight=formant_weight,
            outer_low_freq=outer_low_freq,
            outer_low_weight=outer_low_weight,
            outer_high_freq=outer_high_freq,
            outer_high_weight=outer_high_weight,
        )
        diff = F.relu(s_abs - y_abs)
        err = diff ** 2
        if extra_formant_penalty > 0.0:
            err = err * (1.0 + fmask * extra_formant_penalty * err)
        err = err * fw
        loss = loss + _masked_mean(err, w)
    return loss / len(n_ffts) * factor


def over_suppression_loss(
    pred: Tensor,
    target: Tensor,
    max_freq: float | Tensor | None = None,
    *,
    sr: int = SR,
    n_ffts: tuple[int, ...] = N_FFTS,
    gamma: float = GAMMA,
    factor: float = OS_FACTOR,
    use_formant_weights: bool = False,
    formant_band: tuple[float, float] = (300.0, 3400.0),
    formant_weight: float = 2.5,
    outer_low_freq: float = 150.0,
    outer_low_weight: float = 0.2,
    outer_high_freq: float = 5000.0,
    outer_high_weight: float = 0.2,
    extra_formant_penalty: float = 1.0,
) -> Tensor:
    """`L_OS = λ_OS · média_res média_TF h(|S|^γ − |Ŝ|^γ)²`, `h(x) = max(x, 0)`: só pune módulo abaixo do alvo.

    Quando `use_formant_weights=True`, aplica a ponderação de formantes psicoacústica
    e penalidade quadrática extra da `asymmetric_formant_loss`.
    """
    if use_formant_weights:
        return asymmetric_formant_loss(
            pred,
            target,
            max_freq,
            sr=sr,
            n_ffts=n_ffts,
            gamma=gamma,
            factor=factor,
            formant_band=formant_band,
            formant_weight=formant_weight,
            outer_low_freq=outer_low_freq,
            outer_low_weight=outer_low_weight,
            outer_high_freq=outer_high_freq,
            outer_high_weight=outer_high_weight,
            extra_formant_penalty=extra_formant_penalty,
        )
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


def target_absence_loss(pred: Tensor, absence_mask: Tensor) -> Tensor:
    """Target-Absence Loss: força energia zero na saída durante a ausência do alvo."""
    b = pred.shape[0]
    p_flat = pred.reshape(b, -1)
    m_flat = absence_mask.to(pred.dtype).reshape(b, -1)
    if m_flat.shape[-1] != p_flat.shape[-1]:
        m_flat = F.interpolate(m_flat.unsqueeze(1), size=p_flat.shape[-1], mode="nearest").squeeze(1)
    denom = m_flat.sum()
    if denom == 0.0:
        return pred.sum() * 0.0
    return ((p_flat ** 2) * m_flat).sum() / denom


def interferer_rejection_loss(pred_emb: Tensor, target_emb: Tensor, interferer_emb: Tensor, temperature: float = 0.1) -> Tensor:
    """Interferer Rejection Loss (InfoNCE contrastiva)."""
    pred_norm = F.normalize(pred_emb, dim=-1)
    pos_norm = F.normalize(target_emb, dim=-1)
    neg_norm = F.normalize(interferer_emb, dim=-1)
    pos_sim = (pred_norm * pos_norm).sum(dim=-1)
    neg_sim = (pred_norm * neg_norm).sum(dim=-1)
    logits = torch.stack([pos_sim, neg_sim], dim=-1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def df_coef_temporal_regularization_loss(df_coefs: Tensor, time_dim: int = -2) -> Tensor:
    """DF Coefficient Temporal Regularization: penaliza variações bruscas."""
    idx1 = [slice(None)] * df_coefs.ndim
    idx1[time_dim] = slice(1, None)
    idx2 = [slice(None)] * df_coefs.ndim
    idx2[time_dim] = slice(None, -1)
    diff = df_coefs[tuple(idx1)] - df_coefs[tuple(idx2)]
    return diff.abs().sum()


def stage_c_loss(pred: Tensor, target: Tensor, lsnr: Tensor, lsnr_target: Tensor, *, lsnr_mask: Tensor,
                 interferer_frames: Tensor, max_freq: float | Tensor | None = None,
                 use_formant_loss: bool = False,
                 absence_mask: Tensor | None = None,
                 pred_emb: Tensor | None = None, target_emb: Tensor | None = None, interferer_emb: Tensor | None = None,
                 df_coefs: Tensor | None = None, df_coefs_time_dim: int = -2,
                 lambda_abs: float = 0.5, lambda_conf: float = 0.1, lambda_df: float = 0.05,
                 **os_kwargs) -> dict[str, Tensor]:
    """Perda do estágio C sem destilação: MR do DFNet3 + over-suppression + L_gate + LocalSnr (mascarada)."""
    mr = dfnet_loss(pred, target, max_freq)
    os_ = over_suppression_loss(pred, target, max_freq, use_formant_weights=use_formant_loss, **os_kwargs)
    gate = gate_loss(lsnr, interferer_frames)
    ls = local_snr_loss(lsnr, lsnr_target, lsnr_mask)
    
    total = mr + os_ + gate + ls
    res = {"total": total, "mr": mr, "os": os_, "gate": gate, "lsnr": ls}
    
    if absence_mask is not None:
        l_abs = target_absence_loss(pred, absence_mask)
        res["abs"] = l_abs
        res["total"] = res["total"] + lambda_abs * l_abs
        
    if pred_emb is not None and target_emb is not None and interferer_emb is not None:
        l_conf = interferer_rejection_loss(pred_emb, target_emb, interferer_emb)
        res["conf"] = l_conf
        res["total"] = res["total"] + lambda_conf * l_conf
        
    if df_coefs is not None:
        l_df = df_coef_temporal_regularization_loss(df_coefs, time_dim=df_coefs_time_dim)
        res["df_reg"] = l_df
        res["total"] = res["total"] + lambda_df * l_df
        
    return res


def _angle():
    ensure_df_importable()
    from df.utils import angle

    return angle
