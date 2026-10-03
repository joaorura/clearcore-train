import json
import math

import numpy as np
import pytest
import torch

from cctrain.models.dfnet3_upstream import df_features_batch, load_upstream_dfnet3
from cctrain.models.film import FilmConfig, FilmGenerator
from cctrain.models.pdfnet3 import PDFNet3
from cctrain.models.state_hash import state_dict_sha256


@pytest.fixture(scope="module")
def upstream():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model, df_state = load_upstream_dfnet3(device="cpu")
    yield model, df_state
    torch.use_deterministic_algorithms(False)


def _audio(n: int = 2, seconds: float = 1.0, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(int(48_000 * seconds)) / 48_000.0
    return [(0.1 * np.sin(2 * np.pi * (150 + 50 * i) * t) + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
            for i in range(n)]


def _generator(scale: float) -> FilmGenerator:
    gen = FilmGenerator(FilmConfig(math.log(10.0), 4.0))
    if scale:
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            gen.out.weight.copy_(torch.randn(gen.out.weight.shape, generator=g) * scale)
            gen.out.bias.copy_(torch.randn(gen.out.bias.shape, generator=g) * scale)
    return gen


def test_i1_mask_zero_is_bit_exact_with_upstream(upstream, m0_runs):
    base, df_state = upstream
    spec, erb, specf = df_features_batch(_audio(), df_state)
    emb = torch.randn(2, 192, generator=torch.Generator().manual_seed(2))
    before = state_dict_sha256(base)
    with torch.no_grad():
        ref = base(spec.clone(), erb.clone(), specf.clone())
        out = PDFNet3(base, _generator(0.5)).eval()(spec.clone(), erb.clone(), specf.clone(), emb, torch.zeros(2, 1))
    diffs = [float((r - o).abs().max()) if r.numel() else 0.0 for r, o in zip(ref, out)]
    (m0_runs / "i1.json").write_text(json.dumps(
        {"test": "I1", "device": "cpu", "deterministic": True, "max_abs_diff_per_output": diffs,
         "pass": all(torch.equal(r, o) for r, o in zip(ref, out))}, indent=2) + "\n")
    assert all(torch.equal(r, o) for r, o in zip(ref, out)), diffs
    assert state_dict_sha256(base) == before


def test_zero_initialized_generator_is_identity_with_mask_one(upstream):
    base, df_state = upstream
    spec, erb, specf = df_features_batch(_audio(seed=3), df_state)
    with torch.no_grad():
        ref = base(spec.clone(), erb.clone(), specf.clone())
        out = PDFNet3(base, _generator(0.0)).eval()(spec.clone(), erb.clone(), specf.clone(),
                                                     torch.randn(2, 192), torch.ones(2, 1))
    assert all(torch.equal(r, o) for r, o in zip(ref, out))


def test_non_identity_film_changes_enhancement(upstream):
    base, df_state = upstream
    spec, erb, specf = df_features_batch(_audio(seed=4), df_state)
    with torch.no_grad():
        ref = base(spec.clone(), erb.clone(), specf.clone())
        out = PDFNet3(base, _generator(0.5)).eval()(spec.clone(), erb.clone(), specf.clone(),
                                                     torch.randn(2, 192), torch.ones(2, 1))
    assert not torch.equal(ref[0], out[0])
    assert torch.isfinite(out[0]).all()


def test_film_sites_are_the_relu_outputs(upstream):
    base, _ = upstream
    sites = PDFNet3(base, _generator(0.0)).film_sites()
    assert set(sites) == {"enc", "df_dec"}
    for module in sites.values():
        assert isinstance(list(module.children())[-1], torch.nn.ReLU)
