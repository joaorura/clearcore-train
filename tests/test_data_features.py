import numpy as np
import torch

from cctrain.data.features import NORM_ALPHA, extract_features
from cctrain.dfvendor import ensure_df_importable, ensure_dfnet3_checkpoint


def test_features_are_identical_to_upstream_df_features():
    ensure_df_importable()
    from df.enhance import df_features, init_df
    from df.utils import get_norm_alpha

    _, df_state, *_ = init_df(str(ensure_dfnet3_checkpoint()), post_filter=False, log_level="ERROR",
                              log_file=None, config_allow_defaults=True, epoch="best")
    assert get_norm_alpha(False) == NORM_ALPHA
    x = (0.1 * np.random.default_rng(0).standard_normal(48_000 * 2)).astype(np.float32)
    spec, erb, specf = df_features(torch.from_numpy(x)[None], df_state, 96, device="cpu")
    ours = extract_features(x)
    assert np.array_equal(ours.spec, spec[:, 0].numpy())
    assert np.array_equal(ours.feat_erb, erb[:, 0].numpy())
    assert np.array_equal(ours.feat_spec, specf[:, 0].numpy())
    assert ours.spec.shape == (1, 200, 481, 2)
