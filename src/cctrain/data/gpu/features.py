"""Features do DFNet3 em torch, em lote e em qualquer device (réplica de cctrain.data.features / libdf 0.5.6).

Interface:
  analysis(x[B,n])         -> spec[B,1,F,481,2]            (STFT do libdf, real/imag no último eixo)
  extract_features(x[B,n]) -> (spec, feat_erb[B,1,F,32], feat_spec[B,1,F,96,2])

Fórmulas (libDF/src/lib.rs, transforms.rs):
  * STFT: frame t = [x[(t-1)H : tH] | x[tH : (t+1)H]] (zeros antes do início), janela vorbis, rfft, escala 1/fft_size
    (wnorm = 2H/N^2 = 1/960). F = n // H (o resto é descartado, como em chunks_exact).
  * ERB: média de re^2+im^2 por banda (larguras de erb_widths), 10*log10(v + 1e-10).
  * erb_norm: EMA s = x*(1-a) + s*a, saída (x - s)/40, estado inicial linspace(-60, -90, 32).
  * unit_norm (96 bins): s = |x|*(1-a) + s*a, saída x/sqrt(s), estado inicial linspace(0.001, 0.0001, 96).
  * a = 0.99, estados reiniciados a cada amostra (um DF novo por chamada no gerador CPU).

As EMAs são vetorizadas por blocos (matriz de Toeplitz exponencial dentro do bloco + laço curto só sobre os blocos
para propagar o estado), em vez de um laço por quadro.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

SR = 48_000
FFT = 960
HOP = 480
NB_ERB = 32
NB_DF = 96
MIN_NB_ERB_FREQS = 2
NORM_ALPHA = 0.99
MEAN_NORM_INIT = (-60.0, -90.0)
UNIT_NORM_INIT = (0.001, 0.0001)
_EMA_BLOCK = 32


@lru_cache(maxsize=1)
def _libdf_constants() -> tuple[np.ndarray, np.ndarray]:
    from libdf import DF

    st = DF(SR, FFT, HOP, NB_ERB, MIN_NB_ERB_FREQS)
    return np.asarray(st.fft_window(), dtype=np.float32).copy(), np.asarray(st.erb_widths(), dtype=np.int64).copy()


def _linspace_f32(a: float, b: float, n: int) -> np.ndarray:
    # ndarray::linspace em f32 (libdf): passo em f32, a + i*step
    step = np.float32((np.float32(b) - np.float32(a)) / np.float32(n - 1))
    return (np.float32(a) + np.arange(n, dtype=np.float32) * step).astype(np.float32)


class _Consts:
    """Buffers por device."""

    def __init__(self, device: torch.device):
        window, widths = _libdf_constants()
        self.window = torch.from_numpy(window).to(device)
        n_freq = FFT // 2 + 1
        assert int(widths.sum()) == n_freq and len(widths) == NB_ERB
        maxw = int(widths.max())
        idx = np.full((NB_ERB, maxw), n_freq, dtype=np.int64)  # n_freq -> coluna de zeros
        off = 0
        for b, w in enumerate(widths):
            idx[b, :w] = np.arange(off, off + w)
            off += int(w)
        self.erb_idx = torch.from_numpy(idx).to(device).reshape(-1)
        self.erb_k = torch.from_numpy((1.0 / widths.astype(np.float32)).astype(np.float32)).to(device)
        self.erb_shape = (NB_ERB, maxw)
        self.mean_init = torch.from_numpy(_linspace_f32(*MEAN_NORM_INIT, NB_ERB)).to(device)
        self.unit_init = torch.from_numpy(_linspace_f32(*UNIT_NORM_INIT, NB_DF)).to(device)
        self.wnorm = np.float32(1.0 / (FFT**2 / (2 * HOP)))


_CONSTS: dict[str, _Consts] = {}


def _consts(device: torch.device) -> _Consts:
    key = str(device)
    if key not in _CONSTS:
        _CONSTS[key] = _Consts(device)
    return _CONSTS[key]


def _check(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"esperado x[B,n], recebido {tuple(x.shape)}")
    return x.to(torch.float32)


def _stft(x: torch.Tensor, c: _Consts) -> torch.Tensor:
    """x[B,n] -> complex64 [B,F,481]."""
    n_frames = x.shape[-1] // HOP
    x = x[:, : n_frames * HOP]
    xp = torch.nn.functional.pad(x, (FFT - HOP, 0))  # frame 0 vê zeros antes do início
    frames = xp.unfold(-1, FFT, HOP)  # [B,F,960]
    return torch.fft.rfft(frames * c.window) * float(c.wnorm)


def _ema(x: torch.Tensor, init: torch.Tensor, alpha: float, block: int = _EMA_BLOCK) -> torch.Tensor:
    """y_t = alpha*y_{t-1} + (1-alpha)*x_t ao longo do eixo 1; x[B,F,K], init[K]; y_{-1} = init. Em float32."""
    B, F, K = x.shape
    dev, dt = x.device, x.dtype
    L = min(block, max(F, 1))
    nb = -(-F // L)
    pad = nb * L - F
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    xb = x.reshape(B, nb, L, K)
    j = torch.arange(L, device=dev, dtype=torch.float64)
    expo = j[:, None] - j[None, :]
    # M[j,i] = (1-a) a^(j-i) para i<=j (calculado em f64, usado em f32)
    M = torch.where(expo >= 0, (1.0 - alpha) * torch.pow(torch.tensor(alpha, dtype=torch.float64, device=dev), expo.clamp(min=0)), 0.0)
    M = M.to(dt)
    pw = torch.pow(torch.tensor(alpha, dtype=torch.float64, device=dev), j + 1).to(dt)  # a^(j+1)
    aL = float(alpha) ** L
    # parte local (estado inicial zero) dentro de cada bloco: [B,nb,L,K]
    local = torch.einsum("ji,bnik->bnjk", M, xb)
    # laço curto sobre os blocos para o estado no fim de cada bloco
    carry = init.to(dt).expand(B, K)
    carries = []
    last = local[:, :, -1, :]  # [B,nb,K]
    for b in range(nb):
        carries.append(carry)
        carry = last[:, b] + aL * carry
    carries_t = torch.stack(carries, dim=1)  # estado de entrada de cada bloco [B,nb,K]
    y = local + pw[None, None, :, None] * carries_t[:, :, None, :]
    return y.reshape(B, nb * L, K)[:, :F]


def analysis(x: torch.Tensor) -> torch.Tensor:
    """x f32[B,n] -> spec f32[B,1,F,481,2]."""
    x = _check(x)
    return torch.view_as_real(_stft(x, _consts(x.device)))[:, None]


def _features_from_complex(spec: torch.Tensor, c: _Consts) -> tuple[torch.Tensor, torch.Tensor]:
    B, F, _ = spec.shape
    power = spec.real * spec.real + spec.imag * spec.imag
    power = torch.nn.functional.pad(power, (0, 1))  # coluna de zeros para o preenchimento das bandas
    gathered = power[..., c.erb_idx].reshape(B, F, *c.erb_shape)
    erb = gathered.sum(-1) * c.erb_k
    erb_db = (erb + 1e-10).log10() * 10.0
    s = _ema(erb_db, c.mean_init, NORM_ALPHA)
    feat_erb = (erb_db - s) / 40.0

    sdf = spec[..., :NB_DF]
    s = _ema(sdf.abs(), c.unit_init, NORM_ALPHA)
    feat_spec = sdf / s.sqrt()
    return feat_erb, torch.view_as_real(feat_spec)


def extract_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x f32[B,n] -> (spec[B,1,F,481,2], feat_erb[B,1,F,32], feat_spec[B,1,F,96,2])."""
    x = _check(x)
    c = _consts(x.device)
    spec = _stft(x, c)
    feat_erb, feat_spec = _features_from_complex(spec, c)
    return torch.view_as_real(spec)[:, None], feat_erb[:, None], feat_spec[:, None]
