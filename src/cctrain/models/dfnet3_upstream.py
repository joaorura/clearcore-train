"""DFNet3 upstream v0.5.6 (pesos do DeepFilterNet3.zip vendorizado) e features do libdf em lote."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from cctrain.dfvendor import ensure_df_importable, ensure_dfnet3_checkpoint

NB_DF = 96


def load_upstream_dfnet3(model_dir: Path | None = None, device: str = "cpu") -> tuple[nn.Module, object]:
    ensure_df_importable()
    from df.enhance import init_df

    model_dir = model_dir or ensure_dfnet3_checkpoint()
    model, df_state, *_ = init_df(
        str(model_dir), post_filter=False, log_level="ERROR", log_file=None,
        config_allow_defaults=True, epoch="best",
    )
    return model.to(device).eval(), df_state


def df_features_batch(audios: Sequence[np.ndarray], df_state) -> tuple[Tensor, Tensor, Tensor]:
    ensure_df_importable()
    from df.enhance import df_features

    feats = [df_features(torch.from_numpy(np.asarray(a, np.float32))[None], df_state, NB_DF, device="cpu")
             for a in audios]
    return tuple(torch.cat([f[k] for f in feats], dim=0) for k in range(3))
