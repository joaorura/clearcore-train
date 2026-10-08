"""Discriminador Espectral Multi-Escala (MS-STFT Discriminator) e perdas adversariais.

Inspirado em arquiteturas de síntese e restauração de alta fidelidade como BigVGAN, Vocos e HiFi-GAN:
analisa o sinal no domínio tempo-frequência (STFT) em múltiplas resoluções temporais e espectrais
concomitantes para avaliar se a fala sintetizada/denoisada possui textura espectral natural ou se
apresenta artefatos como fala robótica, vazamentos de fase, corte de harmônicos ou abafamento.

Contém:
- `STFTSubDiscriminator`: discriminador 2D sobre representação real/imaginária de STFT em uma resolução.
- `MultiScaleSpectralDiscriminator`: conjunto multi-resolução de sub-discriminadores STFT.
- `discriminator_loss`: perda adversarial do discriminador (LSGAN ou Hinge).
- `generator_adversarial_loss`: perda adversarial do gerador para enganar o discriminador.
- `feature_matching_loss`: perda L1 de matching de ativações intermediárias para estabilidade acústica.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_RESOLUTIONS: tuple[tuple[int, int, int], ...] = (
    # (n_fft, hop_length, win_length)
    (512, 128, 512),
    (1024, 256, 1024),
    (2048, 512, 2048),
)


class STFTSubDiscriminator(nn.Module):
    """Sub-discriminador 2D operando sobre o espectrograma STFT (canais real e imaginário)."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int | None = None,
        win_length: int | None = None,
        channels: int = 32,
        max_channels: int = 512,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 4)
        self.win_length = win_length or n_fft

        # Camadas convolucionais 2D operando em (B, 2, F, T)
        c = channels
        self.convs = nn.ModuleList([
            nn.Conv2d(2, c, kernel_size=(3, 9), padding=(1, 4)),
            nn.Conv2d(c, c * 2, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(c * 2, c * 4, kernel_size=(3, 9), stride=(2, 2), padding=(1, 4)),
            nn.Conv2d(c * 4, c * 8, kernel_size=(3, 9), stride=(2, 2), padding=(1, 4)),
            nn.Conv2d(c * 8, min(c * 16, max_channels), kernel_size=(3, 3), padding=(1, 1)),
        ])
        self.post_conv = nn.Conv2d(min(c * 16, max_channels), 1, kernel_size=(3, 3), padding=(1, 1))

    def _stft(self, x: Tensor) -> Tensor:
        """Gera representação [B, 2, F, T] (canais real e imaginário) da STFT."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        batch, time_len = x.shape
        if time_len < self.win_length:
            pad_amount = self.win_length - time_len
            x = F.pad(x, (0, pad_amount))

        window = torch.hann_window(self.win_length, device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            normalized=True,
            return_complex=True,
        )
        # stft: [B, F, T] complex
        real = stft.real
        imag = stft.imag
        return torch.stack([real, imag], dim=1)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Processa a forma de onda de áudio [B, T].

        Retorna (score [B, 1, F', T'], fmaps [lista de tensores intermediários]).
        """
        h = self._stft(x)
        fmaps: list[Tensor] = []
        for conv in self.convs:
            h = conv(h)
            h = F.leaky_relu(h, 0.2)
            fmaps.append(h)
        score = self.post_conv(h)
        fmaps.append(score)
        return score, fmaps


class MultiScaleSpectralDiscriminator(nn.Module):
    """Discriminador Espectral Multi-Escala para avaliação de naturalidade da fala."""

    def __init__(
        self,
        resolutions: Sequence[tuple[int, int, int]] = DEFAULT_RESOLUTIONS,
        channels: int = 32,
        max_channels: int = 512,
    ):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.discriminators = nn.ModuleList([
            STFTSubDiscriminator(
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                channels=channels,
                max_channels=max_channels,
            )
            for n_fft, hop, win in self.resolutions
        ])

    def forward(self, x: Tensor) -> tuple[list[Tensor], list[list[Tensor]]]:
        """Aplica todos os sub-discriminadores em x [B, T].

        Retorna:
        - scores: lista de scores [B, 1, F_i, T_i] por resolução.
        - fmaps: lista de listas de ativações intermediárias por resolução.
        """
        scores: list[Tensor] = []
        fmaps: list[list[Tensor]] = []
        for disc in self.discriminators:
            score, fmap = disc(x)
            scores.append(score)
            fmaps.append(fmap)
        return scores, fmaps


def discriminator_loss(
    real_scores: Sequence[Tensor],
    fake_scores: Sequence[Tensor],
    *,
    loss_type: Literal["lsgan", "hinge"] = "lsgan",
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Calcula a perda do discriminador em múltiplas escalas.

    - "lsgan": 0.5 * (mean((D(x) - 1)^2) + mean(D(x_hat)^2)) por escala.
    - "hinge": mean(relu(1 - D(x))) + mean(relu(1 + D(x_hat))) por escala.

    Retorna (total_loss, real_losses, fake_losses).
    """
    if len(real_scores) != len(fake_scores):
        raise ValueError("real_scores e fake_scores devem ter o mesmo número de escalas")

    total_loss = torch.zeros((), device=real_scores[0].device, dtype=real_scores[0].dtype)
    real_losses: list[Tensor] = []
    fake_losses: list[Tensor] = []

    for r_score, f_score in zip(real_scores, fake_scores, strict=True):
        if loss_type == "lsgan":
            r_loss = torch.mean((r_score - 1.0) ** 2)
            f_loss = torch.mean(f_score ** 2)
            scale_loss = 0.5 * (r_loss + f_loss)
        elif loss_type == "hinge":
            r_loss = torch.mean(F.relu(1.0 - r_score))
            f_loss = torch.mean(F.relu(1.0 + f_score))
            scale_loss = r_loss + f_loss
        else:
            raise ValueError(f"Tipo de perda desconhecido: {loss_type}")

        real_losses.append(r_loss)
        fake_losses.append(f_loss)
        total_loss = total_loss + scale_loss

    total_loss = total_loss / len(real_scores)
    return total_loss, real_losses, fake_losses


def generator_adversarial_loss(
    fake_scores: Sequence[Tensor],
    *,
    loss_type: Literal["lsgan", "hinge"] = "lsgan",
) -> tuple[Tensor, list[Tensor]]:
    """Calcula a perda adversarial do gerador em múltiplas escalas.

    - "lsgan": mean((D(x_hat) - 1)^2) por escala.
    - "hinge": -mean(D(x_hat)) por escala.

    Retorna (total_loss, scale_losses).
    """
    total_loss = torch.zeros((), device=fake_scores[0].device, dtype=fake_scores[0].dtype)
    scale_losses: list[Tensor] = []

    for f_score in fake_scores:
        if loss_type == "lsgan":
            s_loss = torch.mean((f_score - 1.0) ** 2)
        elif loss_type == "hinge":
            s_loss = -torch.mean(f_score)
        else:
            raise ValueError(f"Tipo de perda desconhecido: {loss_type}")

        scale_losses.append(s_loss)
        total_loss = total_loss + s_loss

    total_loss = total_loss / len(fake_scores)
    return total_loss, scale_losses


def feature_matching_loss(
    real_fmaps: Sequence[Sequence[Tensor]],
    fake_fmaps: Sequence[Sequence[Tensor]],
    *,
    factor: float = 1.0,
) -> Tensor:
    """Perda L1 de Feature Matching entre ativações intermediárias do discriminador."""
    if len(real_fmaps) != len(fake_fmaps):
        raise ValueError("real_fmaps e fake_fmaps devem ter o mesmo número de escalas")

    total_loss = torch.zeros((), device=real_fmaps[0][0].device, dtype=real_fmaps[0][0].dtype)
    for r_scale_fmaps, f_scale_fmaps in zip(real_fmaps, fake_fmaps, strict=True):
        scale_loss = torch.zeros((), device=r_scale_fmaps[0].device, dtype=r_scale_fmaps[0].dtype)
        for r_f, f_f in zip(r_scale_fmaps, f_scale_fmaps, strict=True):
            scale_loss = scale_loss + F.l1_loss(f_f, r_f.detach())
        scale_loss = scale_loss / len(r_scale_fmaps)
        total_loss = total_loss + scale_loss

    return (total_loss / len(real_fmaps)) * factor
