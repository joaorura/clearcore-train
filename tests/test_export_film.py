import gzip
import io
import json
import tarfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from cctrain.export.film import (
    CONFIG_INI_SHA256,
    HIDDEN,
    SITES,
    build_film_identity_asset,
    build_film_identity_members,
    verify_film_graph,
)
from cctrain.export.pack import read_members
from cctrain.hashing import sha256_bytes
from cctrain.paths import clearcore_root

pytestmark = pytest.mark.clearcore


def _approved():
    return clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin"


def _feeds(name: str, steps: int, seed: int) -> dict:
    r = np.random.RandomState(seed)
    if name == "enc":
        return {"feat_erb": r.randn(1, 1, steps, 32).astype("f4"), "feat_spec": r.randn(1, 2, steps, 96).astype("f4")}
    return {"emb": r.randn(1, steps, 512).astype("f4"), "c0": r.randn(1, 64, steps, 96).astype("f4")}


def _run(model_bytes: bytes, feeds: dict):
    return ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"]).run(None, feeds)


def test_film_graphs_keep_initializers_and_add_only_mul_add(tmp_path):
    out = build_film_identity_members(_approved(), tmp_path)
    original = read_members(_approved())
    for name, site in SITES.items():
        verify_film_graph(original[f"{name}.onnx"], out[f"{name}.onnx"].read_bytes(), site)
        m = onnx.load(str(out[f"{name}.onnx"]))
        assert [i.name for i in m.graph.input][-2:] == ["gamma", "beta"]
        assert len(m.graph.input) == 4
    assert out["erb_dec.onnx"].read_bytes() == original["erb_dec.onnx"]
    assert sha256_bytes(out["config.ini"].read_bytes()) == CONFIG_INI_SHA256


def test_identity_is_bit_exact_and_non_identity_changes_output(tmp_path):
    out = build_film_identity_members(_approved(), tmp_path)
    original = read_members(_approved())
    steps = 8
    for name in SITES:
        orig, film = original[f"{name}.onnx"], out[f"{name}.onnx"].read_bytes()
        for seed in range(5):
            f = _feeds(name, steps, seed)
            base = _run(orig, f)
            ident = _run(film, {**f, "gamma": np.ones((1, steps, HIDDEN), "f4"),
                                "beta": np.zeros((1, steps, HIDDEN), "f4")})
            assert all(np.array_equal(a, b) for a, b in zip(base, ident)), (name, seed)
        f = _feeds(name, steps, 99)
        moved = _run(film, {**f, "gamma": np.full((1, steps, HIDDEN), 1.5, "f4"),
                            "beta": np.full((1, steps, HIDDEN), 0.1, "f4")})
        assert not all(np.array_equal(a, b) for a, b in zip(_run(orig, f), moved)), name


def test_asset_pack_is_reproducible_and_compared_with_clearcore_fixture(tmp_path, m0_runs):
    a = build_film_identity_asset(_approved(), tmp_path / "a")
    b = build_film_identity_asset(_approved(), tmp_path / "b")
    assert a.sha256 == b.sha256
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(a.path.read_bytes()))) as tar:
        assert [m.name for m in tar.getmembers()] == ["enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"]
    ours = read_members(a.path)
    fixture = read_members(clearcore_root() / "fixtures/film/film-identity-asset.tar.gz")
    same = {k: ours[k] == fixture[k] for k in ours}
    (m0_runs / "film_asset.json").write_text(json.dumps(
        {"sha256": a.sha256, "size": a.size, "members_equal_to_clearcore_fixture": same}, indent=2) + "\n")
