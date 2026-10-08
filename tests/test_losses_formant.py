import pytest
import torch

from cctrain.losses.dfnet import (
    asymmetric_formant_loss,
    formant_frequency_weights,
    over_suppression_loss,
    stage_c_loss,
)


def _wave(seed: int, b: int = 2, n: int = 24_000) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 0.1 * torch.randn(b, n, generator=g)


def test_formant_frequency_weights_bands_and_interpolation():
    n_fft = 2048
    sr = 48_000
    weights, mask = formant_frequency_weights(
        n_fft,
        sr=sr,
        formant_band=(300.0, 3400.0),
        formant_weight=2.5,
        outer_low_freq=150.0,
        outer_low_weight=0.2,
        outer_high_freq=5000.0,
        outer_high_weight=0.2,
    )

    freqs = torch.arange(n_fft // 2 + 1) * (sr / n_fft)

    # Sub-graves e ruídos de ventoinha (< 150 Hz): peso reduzido 0.2
    idx_sub = torch.nonzero(freqs < 140.0).flatten()
    assert torch.allclose(weights[idx_sub, 0], torch.tensor(0.2))

    # Formantes da fala (300 a 3400 Hz): peso elevado 2.5 e máscara ativa
    idx_formant = torch.nonzero((freqs >= 350.0) & (freqs <= 3300.0)).flatten()
    assert torch.allclose(weights[idx_formant, 0], torch.tensor(2.5))
    assert torch.allclose(mask[idx_formant, 0], torch.tensor(1.0))

    # Altas frequências (> 5000 Hz): peso reduzido 0.2
    idx_high = torch.nonzero(freqs > 5100.0).flatten()
    assert torch.allclose(weights[idx_high, 0], torch.tensor(0.2))
    assert torch.allclose(mask[idx_high, 0], torch.tensor(0.0))

    # Transição suave entre 150 Hz e 300 Hz
    idx_trans = torch.nonzero((freqs > 150.0) & (freqs < 300.0)).flatten()
    for i in idx_trans:
        assert 0.2 <= float(weights[i, 0]) <= 2.5


def test_asymmetric_formant_loss_is_zero_when_not_attenuating():
    target = _wave(100)
    assert float(asymmetric_formant_loss(target.clone(), target)) == 0.0

    # Amplificação não sofre penalidade (assimétrica: só pune over-suppression)
    amplified = 1.4 * target
    assert float(asymmetric_formant_loss(amplified, target)) == 0.0


def test_asymmetric_formant_loss_heavily_punishes_formant_cut_vs_subbass_and_high_cut():
    target = _wave(101, b=1, n=48_000)
    t = torch.arange(48_000) / 48_000.0

    # Mesmo nível de corte senoidal em 3 regiões diferentes:
    amp = 0.04
    # 1. Região de formantes (1000 Hz - formante de vogal F1/F2)
    pred_formant_cut = target - amp * torch.sin(2 * torch.pi * 1000.0 * t)
    # 2. Região de sub-grave / ventoinha (60 Hz)
    pred_subbass_cut = target - amp * torch.sin(2 * torch.pi * 60.0 * t)
    # 3. Região de agudos / hiss (8000 Hz)
    pred_hiss_cut = target - amp * torch.sin(2 * torch.pi * 8000.0 * t)

    loss_formant = float(asymmetric_formant_loss(pred_formant_cut, target))
    loss_subbass = float(asymmetric_formant_loss(pred_subbass_cut, target))
    loss_hiss = float(asymmetric_formant_loss(pred_hiss_cut, target))

    assert loss_formant > 0.0
    assert loss_subbass > 0.0
    assert loss_hiss > 0.0

    # A perda na banda de formantes deve ser muito mais rigorosa que nos extremos (> 3.5x maior)
    assert loss_formant > 3.5 * loss_subbass
    assert loss_formant > 3.5 * loss_hiss


def test_extra_quadratic_penalty_punishes_severe_cuts_superlinearly():
    target = _wave(102, b=1, n=48_000)
    t = torch.arange(48_000) / 48_000.0

    # Corte moderado vs corte severo em 1500 Hz
    tone = torch.sin(2 * torch.pi * 1500.0 * t)
    pred_moderate = target - 0.02 * tone
    pred_severe = target - 0.06 * tone  # 3x maior corte

    loss_no_extra = float(asymmetric_formant_loss(pred_severe, target, extra_formant_penalty=0.0))
    loss_with_extra = float(asymmetric_formant_loss(pred_severe, target, extra_formant_penalty=2.0))

    # A penalidade quadrática extra amplifica drasticamente cortes severos
    assert loss_with_extra > loss_no_extra


def test_over_suppression_loss_backwards_compatibility_and_formant_flag():
    pred, target = _wave(103), _wave(104)

    # Sem flag (padrão): idêntica à baseline
    base_loss = over_suppression_loss(pred, target, use_formant_weights=False)
    assert base_loss > 0.0

    # Com flag: delega para asymmetric_formant_loss
    formant_os_loss = over_suppression_loss(pred, target, use_formant_weights=True)
    expected = asymmetric_formant_loss(pred, target)
    assert torch.allclose(formant_os_loss, expected)


def test_stage_c_loss_integrates_formant_loss_option():
    pred, target = _wave(105), _wave(106)
    lsnr = torch.randn(2, 20, 1)
    lsnr_t = torch.randn(2, 20)
    active = torch.ones(2, 20, dtype=torch.bool)

    res_default = stage_c_loss(pred, target, lsnr, lsnr_t, lsnr_mask=active, interferer_frames=active)
    res_formant = stage_c_loss(pred, target, lsnr, lsnr_t, lsnr_mask=active, interferer_frames=active,
                               use_formant_loss=True)

    assert set(res_formant) == {"total", "mr", "os", "gate", "lsnr"}
    assert torch.allclose(res_formant["total"],
                          res_formant["mr"] + res_formant["os"] + res_formant["gate"] + res_formant["lsnr"])
    assert res_formant["os"] != res_default["os"]  # over-suppression calculada com ponderação de formantes
