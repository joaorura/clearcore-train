import pytest
import torch

from cctrain.losses.discriminator import (
    MultiScaleSpectralDiscriminator,
    STFTSubDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)


def _audio(b: int = 2, t: int = 16_000, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 0.2 * torch.randn(b, t, generator=g)


def test_stft_sub_discriminator_shapes():
    sub = STFTSubDiscriminator(n_fft=1024, hop_length=256, win_length=1024, channels=16)
    x = _audio(b=2, t=16_000, seed=1)
    score, fmaps = sub(x)

    # Score deve ser um mapa 2D (B, 1, F_out, T_out)
    assert score.ndim == 4
    assert score.shape[0] == 2
    assert score.shape[1] == 1
    # Lista de ativações intermediárias não vazia
    assert len(fmaps) == len(sub.convs) + 1  # camadas + post_conv
    assert all(isinstance(f, torch.Tensor) for f in fmaps)


def test_multiscale_spectral_discriminator_forward():
    ms_disc = MultiScaleSpectralDiscriminator(
        resolutions=((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)),
        channels=16,
    )
    x = _audio(b=2, t=24_000, seed=2)
    scores, fmaps = ms_disc(x)

    assert len(scores) == 3
    assert len(fmaps) == 3
    for s, f in zip(scores, fmaps, strict=True):
        assert s.ndim == 4
        assert s.shape[0] == 2
        assert len(f) > 0


def test_discriminator_loss_lsgan_and_hinge():
    ms_disc = MultiScaleSpectralDiscriminator(
        resolutions=((512, 128, 512), (1024, 256, 1024)),
        channels=16,
    )
    real_x = _audio(b=2, t=12_000, seed=3)
    fake_x = _audio(b=2, t=12_000, seed=4)

    real_scores, _ = ms_disc(real_x)
    fake_scores, _ = ms_disc(fake_x)

    # 1. LSGAN
    d_loss_lsgan, r_losses, f_losses = discriminator_loss(real_scores, fake_scores, loss_type="lsgan")
    assert d_loss_lsgan.ndim == 0
    assert torch.isfinite(d_loss_lsgan)
    assert float(d_loss_lsgan.detach()) > 0.0
    assert len(r_losses) == 2
    assert len(f_losses) == 2

    d_loss_lsgan.backward()
    # Verifica que gradientes fluem para os parâmetros do discriminador
    has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in ms_disc.parameters())
    assert has_grad

    # 2. Hinge
    ms_disc.zero_grad()
    d_loss_hinge, _, _ = discriminator_loss(real_scores, fake_scores, loss_type="hinge")
    assert d_loss_hinge.ndim == 0
    assert torch.isfinite(d_loss_hinge)


def test_generator_adversarial_loss_lsgan_and_hinge():
    ms_disc = MultiScaleSpectralDiscriminator(
        resolutions=((512, 128, 512), (1024, 256, 1024)),
        channels=16,
    )
    fake_x = _audio(b=2, t=12_000, seed=5).requires_grad_()
    fake_scores, _ = ms_disc(fake_x)

    # 1. LSGAN
    g_loss_lsgan, scale_losses = generator_adversarial_loss(fake_scores, loss_type="lsgan")
    assert g_loss_lsgan.ndim == 0
    assert torch.isfinite(g_loss_lsgan)
    assert len(scale_losses) == 2

    g_loss_lsgan.backward()
    assert fake_x.grad is not None
    assert torch.norm(fake_x.grad) > 0

    # 2. Hinge
    g_loss_hinge, _ = generator_adversarial_loss(fake_scores, loss_type="hinge")
    assert torch.isfinite(g_loss_hinge)


def test_feature_matching_loss():
    ms_disc = MultiScaleSpectralDiscriminator(
        resolutions=((512, 128, 512), (1024, 256, 1024)),
        channels=16,
    )
    x = _audio(b=2, t=12_000, seed=6)
    y = _audio(b=2, t=12_000, seed=7)

    _, fmaps_x = ms_disc(x)
    _, fmaps_y = ms_disc(y)

    # Ativações idênticas dão perda 0
    loss_self = feature_matching_loss(fmaps_x, fmaps_x)
    assert float(loss_self.detach()) == 0.0

    # Ativações diferentes dão perda > 0
    loss_diff = feature_matching_loss(fmaps_x, fmaps_y)
    assert float(loss_diff.detach()) > 0.0


def test_discriminator_handles_short_audio_via_padding():
    # Áudio menor que a maior janela de STFT (ex: 800 amostras vs win_length=1024)
    sub = STFTSubDiscriminator(n_fft=1024, hop_length=256, win_length=1024, channels=16)
    short_x = torch.randn(1, 800)
    score, fmaps = sub(short_x)
    assert score.shape[0] == 1
    assert len(fmaps) > 0
