"""Features do DFNet3 pelo libdf (deepfilterlib 0.5.6), idênticas a df.enhance.df_features (spec §5.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from libdf import DF, erb, erb_norm, unit_norm

SR = 48_000
FFT = 960
HOP = 480
NB_ERB = 32
NB_DF = 96
MIN_NB_ERB_FREQS = 2
NORM_ALPHA = 0.99  # calc_norm_alpha(48000, 480, tau=1) arredondado como no libDF


@dataclass(frozen=True)
class Features:
    spec: np.ndarray
    feat_erb: np.ndarray
    feat_spec: np.ndarray


def _as_real(c: np.ndarray) -> np.ndarray:
    return np.stack([c.real, c.imag], axis=-1).astype(np.float32)


def _state() -> DF:
    return DF(SR, FFT, HOP, NB_ERB, MIN_NB_ERB_FREQS)


def analysis(audio_48k: np.ndarray) -> np.ndarray:
    return _as_real(_state().analysis(np.ascontiguousarray(audio_48k, dtype=np.float32)[None]))


def extract_features(audio_48k: np.ndarray) -> Features:
    state = _state()
    spec = state.analysis(np.ascontiguousarray(audio_48k, dtype=np.float32)[None])
    feat_erb = np.asarray(erb_norm(erb(spec, state.erb_widths()), NORM_ALPHA), dtype=np.float32)
    feat_spec = unit_norm(np.ascontiguousarray(spec[..., :NB_DF]), NORM_ALPHA)
    return Features(_as_real(spec), feat_erb, _as_real(np.asarray(feat_spec)))
