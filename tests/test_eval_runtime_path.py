import numpy as np
import torch

from cctrain.eval.runtime_path import (
    HOP,
    EncoderStep,
    Stage,
    apply_band_gains,
    apply_stages,
    deep_filter,
    emulate_gating_offline,
    mean_square_f32,
    run_replica,
    stage_of,
)


def test_apply_stages_boundaries_match_rust():
    assert apply_stages(-10.0001) == (False, True, False)
    assert apply_stages(-10.0) == (True, False, True)
    assert apply_stages(20.0) == (True, False, True)
    assert apply_stages(20.0001) == (True, False, False)
    assert apply_stages(30.0) == (True, False, False)
    assert apply_stages(30.0001) == (False, False, False)
    assert stage_of(apply_stages(0.0)) is Stage.ERB_DF


def test_mean_square_is_sequential_f32():
    x = np.random.default_rng(0).standard_normal(480).astype(np.float32)
    acc = np.float32(0.0)
    for v in x:
        acc = np.float32(acc + v * v)
    assert mean_square_f32(x) == np.float32(acc / np.float32(480))


def test_band_gains_and_deep_filter():
    spec = (np.arange(10) + 1j * np.arange(10)).astype(np.complex64)
    apply_band_gains(spec, np.array([0.5, 2.0], np.float32), np.array([4, 6]))
    assert np.array_equal(spec[:4], (np.arange(4) * 0.5 * (1 + 1j)).astype(np.complex64))
    assert np.array_equal(spec[4:], (np.arange(4, 10) * 2.0 * (1 + 1j)).astype(np.complex64))
    frames = [np.full(481, k + 1, np.complex64) for k in range(5)]
    coefs = np.zeros((96, 5), np.complex64)
    coefs[:, 2] = 1.0
    assert np.array_equal(deep_filter(frames, coefs), np.full(96, 3, np.complex64))


class _FakeModel:
    def __init__(self, lsnrs):
        self.lsnrs = lsnrs
        self.calls = []

    def reset(self):
        self.t = 0

    def encode(self, feat_erb, feat_spec):
        step = EncoderStep(np.float32(self.lsnrs[self.t]), {})
        self.t += 1
        return step

    def erb_decode(self, step):
        self.calls.append(("erb", self.t - 1))
        return np.ones(32, np.float32)

    def df_decode(self, step):
        self.calls.append(("df", self.t - 1))
        c = np.zeros((96, 5), np.complex64)
        c[:, 2] = 1.0
        return c


def test_decoders_only_run_on_their_stages_and_silence_skips_everything():
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(8 * HOP)).astype(np.float32)
    x[:HOP] = 0.0
    lsnrs = [-20.0, 40.0, 25.0, 0.0, 0.0, 40.0, 25.0]
    model = _FakeModel(lsnrs)
    res = run_replica(x, model)
    assert list(res.stages) == [Stage.SILENT, Stage.ZERO, Stage.NONE, Stage.ERB_ONLY, Stage.ERB_DF, Stage.ERB_DF,
                                Stage.NONE, Stage.ERB_ONLY]
    assert model.calls == [("erb", 2), ("erb", 3), ("df", 3), ("erb", 4), ("df", 4), ("erb", 6)]
    assert res.lsnr[0] == np.float32(-15.0) and not np.any(res.audio[:HOP])


def test_no_processing_path_has_contract_latency_of_1440_samples():
    x = (0.1 * np.random.default_rng(1).standard_normal(200 * HOP)).astype(np.float32)
    res = run_replica(x, _FakeModel([40.0] * 200))
    np.testing.assert_allclose(res.audio[1440:], x[:-1440], atol=1e-5)


def test_offline_emulation_selects_by_stage():
    shape = (1, 1, 4, 3, 2)
    noisy, masked, enhanced = torch.full(shape, 1.0), torch.full(shape, 2.0), torch.full(shape, 3.0)
    lsnr = torch.tensor([[-20.0, 40.0, 25.0, 0.0]])
    out, stages = emulate_gating_offline(noisy, masked, enhanced, lsnr)
    assert out[0, 0, :, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert stages.tolist() == [[0, 3, 1, 2]]
