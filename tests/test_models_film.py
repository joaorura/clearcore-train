import math

import pytest
import torch

from cctrain.models.film import (
    FilmConfig,
    FilmConfigError,
    FilmGenerator,
    beta_bound_rule,
    load_film_config,
)

LN10 = math.log(10.0)


def test_rule_for_beta_bound():
    assert beta_bound_rule(0.5) == 4.0
    assert beta_bound_rule(3.0) == 6.0
    assert beta_bound_rule(30.0) == 50.0


@pytest.mark.parametrize("gamma_bound,beta", [(0.0, 4.0), (math.log(100.0) + 1e-6, 4.0), (LN10, 3.9),
                                              (LN10, 50.1)])
def test_validator_refuses_out_of_range(gamma_bound, beta):
    with pytest.raises(FilmConfigError):
        FilmConfig(log_gamma_bound=gamma_bound, beta_bound=beta).validate()


def test_config_v1_has_gamma_ln10_and_beta_pending_until_m0():
    with pytest.raises(FilmConfigError, match="B"):
        load_film_config()


def test_zero_init_gives_exact_identity_for_any_embedding():
    gen = FilmGenerator(FilmConfig(LN10, 4.0))
    v = gen(torch.randn(3, 192), torch.ones(3, 1))
    for t in (v.gamma_enc, v.gamma_df):
        assert t.shape == (3, 1, 256) and torch.equal(t, torch.ones_like(t))
    for t in (v.beta_enc, v.beta_df):
        assert torch.equal(t, torch.zeros_like(t))


def test_mask_zero_gives_identity_even_with_trained_weights():
    gen = FilmGenerator(FilmConfig(LN10, 4.0))
    with torch.no_grad():
        gen.out.weight.normal_(std=1.0)
        gen.out.bias.normal_(std=1.0)
    x = torch.randn(4, 1, 256).abs()
    v = gen(torch.randn(4, 192), torch.zeros(4, 1))
    assert torch.equal(x * v.gamma_enc + v.beta_enc, x)
    assert torch.equal(x * v.gamma_df + v.beta_df, x)


def test_outputs_respect_runtime_limits():
    gen = FilmGenerator(FilmConfig(LN10, 50.0))
    with torch.no_grad():
        gen.out.weight.normal_(std=100.0)
    v = gen(torch.randn(16, 192), torch.ones(16, 1))
    for g in (v.gamma_enc, v.gamma_df):
        assert g.min() >= 0.1 - 1e-6 and g.max() <= 10.0 + 1e-5
    for b in (v.beta_enc, v.beta_df):
        assert b.abs().max() <= 50.0
    assert sum(p.numel() for p in gen.parameters()) == 192 * 256 + 256 + 256 * 1024 + 1024
