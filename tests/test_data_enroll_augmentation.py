import numpy as np
import pytest

from cctrain.data.enroll import (
    EnrollmentAudioLoader,
    EnrollmentAugmentConfig,
    augment_enrollment_audio,
    check_enrollment_audio,
    validate_enrollment_audio,
)
from cctrain.export.enrollment import scale_to_rms_dbfs, speechlike_16k


def _synthetic_rir_16k() -> np.ndarray:
    t = np.arange(1600) / 16_000.0  # 100 ms
    return (np.exp(-30.0 * t) * np.random.default_rng(123).standard_normal(1600)).astype(np.float32)


def _synthetic_noise_16k(n: int = 16_000) -> np.ndarray:
    return (0.1 * np.random.default_rng(456).standard_normal(n)).astype(np.float32)


def test_augment_enrollment_audio_p_zero_identity():
    x = np.full(16_000, 0.2, dtype=np.float32)
    rir = _synthetic_rir_16k()
    noise = _synthetic_noise_16k()

    out, meta = augment_enrollment_audio(
        x,
        rirs=[rir],
        noises=[noise],
        p_rir=0.0,
        p_noise=0.0,
    )

    assert np.allclose(out, x)
    assert meta["applied_rir"] is False
    assert meta["applied_noise"] is False
    assert meta["snr_db"] is None


def test_augment_enrollment_audio_applies_rir():
    x = np.zeros(16_000, dtype=np.float32)
    x[0] = 1.0  # delta de Dirac
    rir = _synthetic_rir_16k()

    out, meta = augment_enrollment_audio(
        x,
        rirs=[rir],
        p_rir=1.0,
        p_noise=0.0,
        target_peak=0.95,
    )

    assert meta["applied_rir"] is True
    assert meta["applied_noise"] is False
    # A resposta ao impulso da convolução deve reproduzir o RIR
    assert not np.allclose(out, x)


def test_augment_enrollment_audio_applies_noise_within_snr_range():
    x = np.ones(16_000, dtype=np.float32) * 0.1
    noise = _synthetic_noise_16k()

    out, meta = augment_enrollment_audio(
        x,
        noises=[noise],
        p_rir=0.0,
        p_noise=1.0,
        snr_range_db=(10.0, 20.0),
        rng=np.random.default_rng(42),
    )

    assert meta["applied_noise"] is True
    assert meta["snr_db"] is not None
    assert 10.0 <= meta["snr_db"] <= 20.0
    assert not np.allclose(out, x)


def test_augment_enrollment_audio_prevents_clipping():
    # Sinal alto com ruído alto que levaria a clipping (> 1.0)
    x = np.ones(16_000, dtype=np.float32) * 0.9
    loud_noise = np.ones(16_000, dtype=np.float32) * 0.9

    out, _ = augment_enrollment_audio(
        x,
        noises=[loud_noise],
        p_rir=0.0,
        p_noise=1.0,
        snr_range_db=(-5.0, 0.0),  # ruído muito alto
        target_peak=0.95,
    )

    assert np.max(np.abs(out)) <= 0.95 + 1e-5


def test_enrollment_audio_loader_dataset_interface():
    # Cria conjunto de 5 amostras em memória
    samples = [np.full(16_000, 0.05 * (i + 1), dtype=np.float32) for i in range(5)]
    rirs = [_synthetic_rir_16k()]
    noises = [_synthetic_noise_16k()]

    loader = EnrollmentAudioLoader(
        utterances=samples,
        rirs=rirs,
        noises=noises,
        p_rir=0.5,
        p_noise=0.5,
        snr_range_db=(5.0, 25.0),
        seed=123,
    )

    assert len(loader) == 5

    # Indexação
    item, meta = loader[0]
    assert len(item) == 16_000
    assert isinstance(meta, dict)

    # Sorteio
    sample_item, sample_meta = loader.sample()
    assert len(sample_item) == 16_000

    # Iteração
    count = 0
    for it, _ in loader:
        count += 1
        assert len(it) == 16_000
    assert count == 5


def test_enrollment_audio_loader_validates_against_clearcore_contract():
    # Gera fala sintética válida (6 segundos = 96.000 amostras a 16 kHz)
    speech = scale_to_rms_dbfs(speechlike_16k(96_000, seed=777), -22.0)
    # Valida sinal limpo base
    validate_enrollment_audio(speech)

    rir = _synthetic_rir_16k()
    noise = _synthetic_noise_16k(96_000)

    loader = EnrollmentAudioLoader(
        utterances=[speech],
        rirs=[rir],
        noises=[noise],
        p_rir=0.5,
        p_noise=0.5,
        snr_range_db=(15.0, 25.0),
        seed=888,
        validate=True,
    )

    # Com validate=True, cada amostra gerada deve cumprir o contrato de enrollment
    for _ in range(5):
        audio, meta = loader.sample()
        rep = check_enrollment_audio(audio)
        assert rep.duration_secs == 6.0
        assert rep.peak <= 0.99
        assert rep.active_speech_fraction >= 0.60
