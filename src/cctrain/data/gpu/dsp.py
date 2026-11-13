"""Primitivas de DSP do gerador `mix_v4` em torch, em lote (sem loop por amostra; CPU ou GPU).

Convenções: trilhas empacotadas `[M, n]`, quadros `F = n // HOP` (HOP = 480), float64 salvo indicação. Cada função tem
paridade numérica com a versão CPU de referência:

    ramp_env        <-> mix_v4._ramp_env           (exato: somas de 0/1 em float64)
    fft_conv        <-> mix_v4._conv               (FFT em float32, erro relativo ao pico ~2e-7 como a CPU)
    early_len       <-> mix_v4.early_rir           (pico + 50 ms + 1 amostras)
    lowpass48       <-> stage_f.lowpass48          (Butterworth ordem 8, sosfiltfilt, padlen 27, zi do scipy)
    pink_from_white <-> mix_v4.pink_noise          (mesma formula, ruído branco dado de fora)

Nada aqui mexe em ganhos/SIR/SNR/mistura.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.fft import next_fast_len
from scipy.signal import butter

HOP = 480
SR = 48_000
NYQUIST_HZ = 24_000.0
EARLY_MS = 50.0
_PADLEN = 27  # scipy.sosfiltfilt: 3 * (2 * n_sections + 1) com 4 seções de ordem 2 sem zeros/polos na origem


def ramp_env(on: torch.Tensor, n: int, ramp: int = 480) -> torch.Tensor:
    """Envelope por amostra (0/1 por quadro, `on` bool `[M, F]`) com rampas de `ramp` amostras: média móvel por soma
    acumulada, saída[i] = média de env[i-(ramp-1-(ramp-1)//2) .. i+(ramp-1)//2], fora = 0 (ramp=480: [i-240, i+239]),
    o mesmo alinhamento de `np.convolve(..., 'same')`. Retorna float64 `[M, n]` em [0, 1]."""
    m = on.shape[0]
    env = on.to(torch.float64).repeat_interleave(HOP, dim=1)[:, :n]
    if env.shape[1] < n:
        env = torch.nn.functional.pad(env, (0, n - env.shape[1]))
    if ramp > 1:
        if n < ramp:
            raise ValueError("ramp_env: n < ramp não suportado (o 'same' do numpy troca os papéis nesse caso)")
        right = (ramp - 1) // 2
        left = ramp - 1 - right
        c = torch.zeros(m, n + ramp, dtype=torch.float64, device=env.device)
        c[:, left + 1 : left + 1 + n] = torch.cumsum(env, dim=1)
        c[:, left + 1 + n :] = c[:, left + n : left + n + 1]
        env = (c[:, ramp : ramp + n] - c[:, :n]) / ramp
    return env.clamp_(0.0, 1.0)


def fft_conv(x: torch.Tensor, h: torch.Tensor, h_len: torch.Tensor, *, chunk: int = 32) -> torch.Tensor:
    """Convolução `x * h` truncada a `n` amostras, FFT em float32 (como `mix_v4._conv`). `x` f64 `[M, n]`; `h` f32
    `[M, L]` (RIRs de comprimentos diferentes, zero-pad até L); `h_len` long `[M]`: amostras de `h` realmente usadas
    (o resto é zerado). Retorna float64 `[M, n]`. `chunk` limita as linhas por FFT (memória) e não altera o resultado."""
    m, n = x.shape
    length = min(h.shape[1], n)  # só as primeiras n amostras de h influem nas n primeiras saídas
    h = h[:, :length]
    keep = torch.arange(length, device=h.device)[None, :] < h_len.to(h.device)[:, None]
    h = torch.where(keep, h, torch.zeros((), dtype=h.dtype, device=h.device)).to(torch.float32)
    nfft = next_fast_len(n + length - 1, real=True)
    out = torch.empty(m, n, dtype=torch.float64, device=x.device)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        fx = torch.fft.rfft(x[s:e].to(torch.float32), n=nfft, dim=1)
        fh = torch.fft.rfft(h[s:e], n=nfft, dim=1)
        out[s:e] = torch.fft.irfft(fx * fh, n=nfft, dim=1)[:, :n].to(torch.float64)
    return out


def early_len(h: torch.Tensor, h_len: torch.Tensor) -> torch.Tensor:
    """Comprimento da parte "early" de cada RIR: índice do pico |h| (primeiro máximo) + 50 ms + 1 amostra, limitado a
    `h_len` (igual a `len(early_rir(h))`). Use como `h_len` de `fft_conv` para a convolução early."""
    keep = torch.arange(h.shape[1], device=h.device)[None, :] < h_len.to(h.device)[:, None]
    peak = torch.argmax(torch.where(keep, h.abs(), torch.full((), -1.0, dtype=h.dtype, device=h.device)), dim=1)
    return torch.minimum(peak + (int(EARLY_MS * SR / 1000.0) + 1), h_len.to(h.device)).to(torch.long)


def pink_from_white(w: torch.Tensor) -> torch.Tensor:
    """Ruído rosa (PSD ~ 1/f) a partir de branco `w` f64 `[M, n]`: espectro / sqrt(k) (k=0 -> 1), normalizado a RMS 1
    por trilha."""
    n = w.shape[1]
    spec = torch.fft.rfft(w, dim=1)
    f = torch.arange(spec.shape[1], dtype=torch.float64, device=w.device)
    f[0] = 1.0
    x = torch.fft.irfft(spec / torch.sqrt(f), n=n, dim=1)
    return x / (torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)) + 1e-20)


# --- passa-baixa Butterworth de fase zero ------------------------------------------------------------------------


def _sos_for(cutoffs: np.ndarray) -> np.ndarray:
    """SOS `[k, 4, 6]` (butter(8, fc/Nyquist)) por corte; cada corte distinto é projetado uma vez."""
    cache: dict[float, np.ndarray] = {}
    out = np.empty((len(cutoffs), 4, 6))
    for i, fc in enumerate(cutoffs):
        fc = float(fc)
        if fc not in cache:
            cache[fc] = butter(8, fc / NYQUIST_HZ, output="sos")
        out[i] = cache[fc]
    return out


def _sos_pass(y: torch.Tensor, sos: torch.Tensor, warm: int, chunk: int) -> torch.Tensor:
    """Cascata de biquads (`sos` `[k, 4, 6]`) sobre `y` `[k, T]` com estado inicial de regime para a entrada constante
    `y[:, 0]` (= `sosfilt_zi * y[:, 0]` do scipy). Em cada seção, y = H(y - y0) de estado zero + G*y0 (G = ganho DC da
    seção), que é exatamente a solução com zi de regime por linearidade.

    O IIR é inerentemente sequencial no tempo (o `lfilter` do torchaudio percorre as amostras de cada linha em série),
    então o sinal é cortado em blocos de `chunk` amostras que viram linhas extras do lote, cada um precedido de `warm`
    amostras do bloco anterior (overlap-save para IIR; o prefixo do primeiro bloco repete y[:, 0], que equivale ao zi de
    regime). O transiente do estado inicial errado de cada bloco decai como r^warm (r = maior módulo de polo) e é
    descartado com o prefixo; `warm` é escolhido para r^warm < 1e-16."""
    import torchaudio.functional as taf

    k, t = y.shape
    if chunk >= t:
        warm, chunk = 0, t
    nc = -(-t // chunk)
    pad_r = nc * chunk - t
    yp = torch.cat([y[:, :1].expand(k, warm), y, y[:, -1:].expand(k, pad_r)], dim=1)
    rows = yp.unfold(1, warm + chunk, chunk).reshape(k * nc, warm + chunk)
    sos_r = sos.repeat_interleave(nc, dim=0)
    z = rows
    for s in range(sos.shape[1]):
        b, a = sos_r[:, s, :3], sos_r[:, s, 3:]
        b, a = b / a[:, :1], a / a[:, :1]
        g = b.sum(dim=1, keepdim=True) / a.sum(dim=1, keepdim=True)
        z0 = z[:, :1]
        z = taf.lfilter(z - z0, a, b, clamp=False, batching=True) + g * z0
    return z[:, warm:].reshape(k, nc * chunk)[:, :t]


def _warm_len(sos: np.ndarray) -> int:
    """Amostras para o transiente de qualquer seção decair a < 1e-16 (com folga de 1,5x pela cascata de 4 seções)."""
    r = 0.0
    for sec in sos.reshape(-1, 6):
        r = max(r, float(np.max(np.abs(np.roots(sec[3:])))))
    return int(math.ceil(1.5 * math.log(1e-16) / math.log(r)))


def lowpass48(x: torch.Tensor, cutoff_hz: torch.Tensor) -> torch.Tensor:
    """Passa-baixa Butterworth de ordem 8, fase zero (`sosfiltfilt`: extensão ímpar de 27 amostras, zi de regime,
    filtro direto + reverso) para 48 kHz, em lote. `x` f32 `[M, n]`, `cutoff_hz` f64 `[M]` (corte por linha; linhas com
    corte >= 0,999 * 24 kHz voltam inalteradas, como a CPU). Filtragem em float64, saída float32. Paridade com
    `stage_f.lowpass48`: erro ~1e-14 relativo ao pico em float64 (bordas incluídas: mesma extensão e mesmos zi do
    scipy); na saída float32 sobra no máximo 1 ulp de arredondamento (~1e-8 do pico). Requer n > 27 (o scipy também recusa)."""
    cut = torch.as_tensor(cutoff_hz, dtype=torch.float64).detach().cpu().numpy()
    act = np.nonzero(cut < NYQUIST_HZ * 0.999)[0]
    if len(act) == 0:
        return x
    n = x.shape[1]
    if n <= _PADLEN:
        raise ValueError(f"lowpass48: n={n} <= padlen={_PADLEN}")
    out = x.clone()
    idx = torch.as_tensor(act, device=x.device)
    sos = torch.as_tensor(_sos_for(cut[act]), device=x.device)
    xs = x[idx].to(torch.float64)
    ext = torch.cat(
        [
            2 * xs[:, :1] - xs[:, 1 : _PADLEN + 1].flip(1),
            xs,
            2 * xs[:, -1:] - xs[:, -_PADLEN - 1 : -1].flip(1),
        ],
        dim=1,
    )
    warm = _warm_len(sos.cpu().numpy())
    chunk = max(4096, 4 * warm)
    y = _sos_pass(ext, sos, warm, chunk)
    y = _sos_pass(y.flip(1), sos, warm, chunk).flip(1)
    out[idx] = y[:, _PADLEN : _PADLEN + n].to(x.dtype)
    return out
