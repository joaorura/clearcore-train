"""Front-ends do encoder dentro do grafo, só com operadores comuns (spec §3.1).

`TractSafeFbank` (protótipo do M0): Hamming de 25 ms, hop de 10 ms, DFT como Conv1d com base de Fourier fixa (sem op
STFT), potência, 80 mel (MatMul), Log(x + 1e-6) e subtração da média no tempo. n_fft 512, mel HTK de 20 a 7.600 Hz,
sem padding (T = 1 + (N - 400) // 160).

`SpeechBrainFbank` (M1 Tarefa 10): reproduz o `Fbank` + `InputNormalization` do SpeechBrain ECAPA (ver docstring).
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


SB_N_FFT = 400
SB_N_STFT = SB_N_FFT // 2 + 1
SB_F_MAX = 8000.0
SB_AMIN = 1e-10
SB_TOP_DB = 80.0


def speechbrain_mel_filterbank(n_mels: int = N_MELS) -> np.ndarray:
    """Matriz [n_mels, 201] do `speechbrain.processing.features.Filterbank` (triangular, f 0-8000 Hz).

    Fiel ao SpeechBrain, inclusive o detalhe de a banda de cada filtro ser a distância ao centro anterior
    (`hz[i+1] - hz[i]`) nos dois lados: o triângulo é simétrico.
    """
    def to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    mel = np.linspace(to_mel(0.0), to_mel(SB_F_MAX), n_mels + 2)
    hz = 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    band = (hz[1:] - hz[:-1])[:-1]
    central = hz[1:-1]
    freqs = np.linspace(0.0, SR // 2, SB_N_STFT)
    slope = (freqs[None, :] - central[:, None]) / band[:, None]
    return np.maximum(0.0, np.minimum(slope + 1.0, -slope + 1.0))


class SpeechBrainFbank(nn.Module):
    """`Fbank(n_mels=80)` + `InputNormalization(sentence, std_norm=False)` do SpeechBrain, sem op STFT.

    `torch.stft(center=True, pad_mode="constant")` vira Pad constante de 200 amostras de cada lado e um Conv1d com a
    base de Fourier fixa (Hamming periódica de 400, n_fft 400, 201 bins reais + 201 imaginários), potência, MatMul
    com o banco mel, `10*log10(max(x, 1e-10))`, piso de `top_db=80` dB abaixo do máximo do enunciado e subtração
    da média no tempo. Entrada [B, N] (16 kHz), saída [B, 80, 1 + N // 160] (canal antes do tempo; o SpeechBrain
    devolve [B, T, 80]). Só Pad, Conv, MatMul, Mul, Add, Log, Clip, ReduceMax, Max e ReduceMean.
    """

    def __init__(self) -> None:
        super().__init__()
        window = np.hamming(WIN + 1)[:-1]  # `torch.hamming_window(400)` é periódica
        k = np.arange(SB_N_STFT)[:, None]
        n = np.arange(WIN)[None, :]
        angle = 2.0 * math.pi * k * n / SB_N_FFT
        basis = np.concatenate([np.cos(angle) * window, -np.sin(angle) * window], axis=0)
        self.register_buffer("basis", torch.from_numpy(basis).float().unsqueeze(1))  # [402, 1, 400]
        self.register_buffer("mel", torch.from_numpy(speechbrain_mel_filterbank()).float())  # [80, 201]
        self._db_scale = 10.0 / math.log(10.0)

    def log_mel_db(self, audio: Tensor) -> Tensor:
        """Saída do `Fbank` do SpeechBrain (dB com piso `top_db`), antes da média por sentença: [B, 80, T]."""
        x = nn.functional.pad(audio, (WIN // 2, WIN // 2)).unsqueeze(1)
        spec = nn.functional.conv1d(x, self.basis, stride=HOP)  # [B, 402, T]
        re, im = spec[:, :SB_N_STFT], spec[:, SB_N_STFT:]
        mel = torch.matmul(self.mel, re * re + im * im)  # [B, 80, T]
        db = self._db_scale * torch.log(torch.clamp(mel, min=SB_AMIN))
        return torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - SB_TOP_DB)

    def forward(self, audio: Tensor) -> Tensor:
        db = self.log_mel_db(audio)
        return db - db.mean(dim=-1, keepdim=True)
