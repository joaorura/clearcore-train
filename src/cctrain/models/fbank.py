"""Front-end do encoder dentro do grafo, só com operadores comuns (spec §3.1).

Hamming de 25 ms, hop de 10 ms, DFT como Conv1d com base de Fourier fixa (sem op STFT), potência,
80 mel (MatMul), Log(x + 1e-6) e subtração da média no tempo. Parâmetros (decisão de planejamento 11):
n_fft 512, mel HTK de 20 a 7.600 Hz, sem padding (T = 1 + (N - 400) // 160).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

SR = 16_000
WIN = 400
HOP = 160
N_FFT = 512
N_MELS = 80


def num_frames(n_samples: int) -> int:
    return 1 + (n_samples - WIN) // HOP


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR, f_min: float = 20.0,
                   f_max: float = 7600.0) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    hz = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))
    bins = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, bins.size))
    for i in range(n_mels):
        lo, center, hi = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((bins - lo) / (center - lo), (hi - bins) / (hi - center)))
    return fb


class TractSafeFbank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n_bins = N_FFT // 2 + 1
        window = np.hamming(WIN)
        k = np.arange(n_bins)[:, None]
        n = np.arange(WIN)[None, :]
        angle = 2.0 * math.pi * k * n / N_FFT
        basis = np.concatenate([np.cos(angle) * window, -np.sin(angle) * window], axis=0)
        self.n_bins = n_bins
        self.register_buffer("basis", torch.from_numpy(basis).float().unsqueeze(1))  # [2F, 1, WIN]
        self.register_buffer("mel", torch.from_numpy(mel_filterbank()).float())  # [80, F]

    def forward(self, audio: Tensor) -> Tensor:
        spec = nn.functional.conv1d(audio.unsqueeze(1), self.basis, stride=HOP)  # [B, 2F, T]
        re, im = spec[:, : self.n_bins], spec[:, self.n_bins :]
        mel = torch.matmul(self.mel, re * re + im * im)  # [B, 80, T]
        logmel = torch.log(mel + 1e-6)
        return logmel - logmel.mean(dim=-1, keepdim=True)
