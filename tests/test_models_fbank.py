import numpy as np
import torch

from cctrain.models.fbank import TractSafeFbank, mel_filterbank, num_frames


def _reference(x: np.ndarray) -> np.ndarray:
    n = num_frames(len(x))
    frames = np.stack([x[i * 160 : i * 160 + 400] for i in range(n)]) * np.hamming(400)
    power = np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2
    mel = np.log(power @ mel_filterbank().T + 1e-6).T
    return mel - mel.mean(axis=1, keepdims=True)


def test_frame_count():
    assert num_frames(32_000) == 198
    assert num_frames(192_000) == 1198


def test_matches_numpy_reference():
    x = (0.1 * np.random.default_rng(0).standard_normal(16_000)).astype(np.float32)
    got = TractSafeFbank()(torch.from_numpy(x)[None])[0].numpy()
    assert got.shape == (80, num_frames(16_000))
    assert np.max(np.abs(got - _reference(x.astype(np.float64)))) <= 1e-3


def test_mel_filterbank_is_triangular_and_covers_range():
    fb = mel_filterbank()
    assert fb.shape == (80, 257)
    # filtro 3 tem pico 0,4934: resolução de 31,25 Hz com n_fft 512 em 20-7600 Hz (n_fft mantido p/ não reabrir o contrato do fbank)
    assert np.all(fb >= 0) and np.all(fb.max(axis=1) > 0.45)
