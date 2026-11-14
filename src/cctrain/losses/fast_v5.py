"""Perdas do passo pro-v5 sem recomputação e sem sincronização host-device (mesma matemática das de referência).

Equivalências (paridade nos testes `tests/test_losses_fast_v5.py`):

- `multires_mr_os` = `dfnet_loss(est, tgt)` + `asymmetric_formant_loss(est, tgt)` + `(k/B)·asymmetric_formant_loss(
  est[n1], tgt[n1])` (k = nº de amostras com interferente). Cada resolução faz UMA STFT da saída (compartilhada pelas
  três perdas) e uma do alvo (sem grad); a média sobre o subconjunto vira soma ponderada por amostra
  ((k/B)·média_n1 = Σ_n1/B), sem `nonzero`. A máscara de bins (`max_freq`) é sempre aplicada: com todos os bins
  mantidos ela é a média simples (o atalho `None` da referência só existia para isso) e sem os `bool(...)` que
  sincronizavam; o denominador é contado analiticamente (Σ bins mantidos × T).
- `masked_frac` = `(x·f).sum()/n` com `n = f.sum()`, 0 quando `n == 0` (a referência fazia `float(n) == 0`).
- `turn_pair` = os dois `turn_loss` (L_turn e L_n0g) com a energia da sombra, o ganho g, os pesos de energia e o tempo
  desde o interferente calculados uma vez; as medianas das estatísticas por `nanmedian` (mesma mediana inferior do
  `torch.median`, NaN se vazio) em vez de indexação booleana.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from cctrain.losses.dfnet import _EPS, SR, _stft, formant_frequency_weights

_FW_CACHE: dict = {}


def formant_weights_cached(n_fft: int, sr: int, device, **kw) -> tuple[Tensor, Tensor]:
    """`formant_frequency_weights` memoizado (constante por n_fft; a função original sincroniza com `bool(...)`)."""
    key = (n_fft, sr, str(device), tuple(sorted(kw.items())))
    v = _FW_CACHE.get(key)
    if v is None:
        v = formant_frequency_weights(n_fft, sr, device, **kw)
        _FW_CACHE[key] = v
    return v


def bin_weights(max_freq: Tensor | float, n_fft: int, batch: int, sr: int, device) -> Tensor:
    """`[B, F]` 1/0 dos bins mantidos (bins ≤ max_freq), sem sincronizar."""
    mf = torch.as_tensor(max_freq, dtype=torch.float32, device=device).reshape(-1).expand(batch)
    freqs = torch.arange(n_fft // 2 + 1, device=device, dtype=torch.float32) * (sr / n_fft)
    return (freqs[None, :] <= mf[:, None]).to(torch.float32)


def _angle_re_im():
    from cctrain.dfvendor import ensure_df_importable

    ensure_df_importable()
    from df.utils import angle_re_im  # o mesmo gradiente robusto do `angle` usado por dfnet_loss, em (re, im)

    return angle_re_im


def _res_terms(yr: Tensor, s_abs: Tensor, s_c: Tensor, w: Tensor, fw: Tensor, fmask: Tensor, gamma: float,
               pen: float) -> tuple[Tensor, Tensor, Tensor]:
    """Termos por amostra `[B]` de uma resolução: (|·|^γ do L_mr, complexo do L_mr, L_os), em aritmética real.

    `yr` = view_as_real(STFT da saída) `[B,F,T,2]`. Equivale a `y.abs().clamp_min(ε).pow(γ)` (= (re²+im²)
    limitado a ε² elevado a γ/2; o gradiente é zero abaixo de ε nos dois) e a `|y|^γ·exp(i·angle(y))` (cos/sin do
    `angle_re_im`, mesmo backward do `angle` do upstream). Uma função só, para o torch.compile fundir tudo."""
    re, im = yr[..., 0], yr[..., 1]
    p = (re * re + im * im).clamp_min(_EPS * _EPS).pow(gamma / 2)
    phi = _angle_re_im().apply(re, im)
    T = yr.shape[-2]
    den = w.sum(-1).clamp_min(1.0 / T) * T  # Σ bins mantidos × T (≥ 1, como o clamp da referência)
    mag = ((p - s_abs) ** 2).sum(-1)
    cpx = ((p * torch.cos(phi) - s_c[..., 0]) ** 2 + (p * torch.sin(phi) - s_c[..., 1]) ** 2).sum(-1)
    err = F.relu(s_abs - p) ** 2
    if pen > 0.0:
        err = err * (1.0 + fmask * pen * err)
    err = (err * fw).sum(-1)
    return (mag * w).sum(-1) / den, (cpx * w).sum(-1) / (2 * den), (err * w).sum(-1) / den


_RES_TERMS = {"fn": _res_terms}


def set_compile(enabled: bool) -> None:
    """Liga o torch.compile nos termos por resolução (forward e backward fundidos)."""
    _RES_TERMS["fn"] = torch.compile(_res_terms, dynamic=None) if enabled else _res_terms


def multires_mr_os(est: Tensor, tgt: Tensor, max_freq, n1: Tensor, *, n_ffts, gamma: float, factor: float,
                   factor_complex: float, os_factor: float, formant_weight: float, extra_formant_penalty: float = 1.0,
                   sr: int = SR) -> tuple[Tensor, Tensor]:
    """(L_mr, L_os total) — ver docstring do módulo. `n1`: `[B]` 1/0 das amostras com interferente."""
    if gamma == 1 or not factor_complex:
        raise ValueError("multires_mr_os supõe gamma != 1 e o termo complexo ligado (o do treino)")
    B = est.shape[0]
    n1 = n1.to(est.dtype)
    mr = torch.zeros((), device=est.device, dtype=est.dtype)
    os_ = torch.zeros((), device=est.device, dtype=est.dtype)
    fn = _RES_TERMS["fn"]
    for n_fft in n_ffts:
        yr = torch.view_as_real(_stft(est, n_fft))
        with torch.no_grad():
            s = _stft(tgt, n_fft)
            s_abs = s.abs().clamp_min(_EPS).pow(gamma)
            s_c = torch.view_as_real(s_abs * torch.exp(1j * torch.atan2(s.imag, s.real)))
        w = bin_weights(max_freq, n_fft, B, sr, est.device)
        fw, fmask = formant_weights_cached(n_fft, sr, est.device, formant_weight=formant_weight)
        mag, cpx, per = fn(yr, s_abs, s_c, w, fw, fmask, gamma, extra_formant_penalty)
        mr = mr + mag.mean() * factor + cpx.mean() * factor_complex
        os_ = os_ + per.mean() + (per * n1).sum() / B
    return mr, os_ / len(n_ffts) * os_factor


def masked_frac(x: Tensor, f: Tensor) -> Tensor:
    """`(x·f).sum() / f.sum()`, 0 se a máscara é vazia (sem sincronizar)."""
    return (x * f).sum() / f.sum().clamp(min=1)


def pow_bins(spec: Tensor) -> Tensor:
    """|X|² por bin: spec [B,1,T,F,2] -> [B,T,F]."""
    return spec[:, 0, ..., 0] ** 2 + spec[:, 0, ..., 1] ** 2


def leak_ratio(eo: Tensor, em: Tensor) -> Tensor:
    """O termo por quadro do L_leak (`leak_loss`), a partir das energias por quadro [B,T]."""
    eps = 1e-10
    return F.relu(10 * torch.log10((eo + eps) / (em + eps)) + 40.0) / 40.0


def ovl_value(p_out: Tensor, p_clean: Tensor, p_interf: Tensor, fmask: Tensor) -> Tensor:
    """`ovl_loss` a partir das potências por bin [B,T,F]."""
    y = (p_out + 1e-12).sqrt()
    with torch.no_grad():
        s = (p_clean + 1e-12).sqrt()
        i = (p_interf + 1e-12).sqrt()
        ii = (i * i).sum(-1) + 1e-8
        si = (s * i).sum(-1)
    v = F.relu(((y * i).sum(-1) - si) / ii)
    return masked_frac(v, fmask)


def local_snr_value(lsnr: Tensor, target: Tensor, keep: Tensor, factor: float) -> Tensor:
    x = lsnr.reshape(lsnr.shape[0], -1)
    return factor * masked_frac((x - target) ** 2, keep.to(x.dtype))


def _guarded_div(num: Tensor, den: Tensor) -> Tensor:
    pos = den > 0
    return torch.where(pos, num / torch.where(pos, den, torch.ones_like(den)), torch.zeros_like(num))


def _nanmedian(g: Tensor, v: Tensor) -> Tensor:
    return torch.where(v, g, torch.full_like(g, float("nan"))).nanmedian()


def turn_pair(e_sh: Tensor, e_s: Tensor, ta: Tensor, since: Tensor, valid_t: Tensor, valid_0: Tensor, *,
              hinge_t: float, hinge_0: float, boost: float, decay_s: float):
    """(L_turn, stats_turn, L_n0g, stats_n0g) — os dois `turn_loss` do passo v5 (o do n0g com ia = 0: w_t = 1).

    `e_sh`, `e_s`: |·|² por bin [B,T,F] da sombra e do alvo; `since`: segundos desde o interferente [B,T]."""
    E_sh, E_s = e_sh.sum(-1), e_s.sum(-1)
    g = torch.clamp(10 * torch.log10((E_sh + 1e-12) / (E_s + 1e-12)), min=-40.0)
    with torch.no_grad():
        taf = ta.float()
        mean_act = (E_s * taf).sum(1, keepdim=True) / taf.sum(1, keepdim=True).clamp(min=1)
        w_e = torch.sqrt(E_s / mean_act.clamp(min=1e-12)).clamp(max=3.0)
        w_t = 1.0 + (boost - 1.0) * torch.clamp(1.0 - since / decay_s, min=0.0, max=1.0)
        w_t = torch.where(torch.isfinite(since), w_t, torch.ones_like(w_t))
        w1 = w_e * w_t * valid_t.float()
        w0 = w_e * valid_0.float()
    l1 = F.relu(hinge_t - g) / 10.0
    l0 = F.relu(hinge_0 - g) / 10.0
    turn = _guarded_div((l1 * w1).sum(), w1.sum())
    n0g = _guarded_div((l0 * w0).sum(), w0.sum())
    with torch.no_grad():
        gd = g.detach()
        st1 = {"g_turn_db": _nanmedian(gd, valid_t), "g_turn_early_db": _nanmedian(gd, valid_t & (since <= 0.5)),
               "n_turn": valid_t.sum().float()}
        st0 = {"g_turn_db": _nanmedian(gd, valid_0), "n_turn": valid_0.sum().float()}
    return turn, st1, n0g, st0
