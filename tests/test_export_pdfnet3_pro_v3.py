"""Export stateful do pDFNet3 Pro v3: grafos S=1 hop a hop (ORT) == PDFNet3Pro.forward com pesos de condicionamento
não triviais (FiLM, cross-attention e mixer aleatórios), com e sem perfil."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cctrain.export import pdfnet3_pro_v3 as X

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def pro():
    try:
        from cctrain.models.dfnet3_upstream import load_upstream_dfnet3

        base, _ = load_upstream_dfnet3()
    except Exception as e:  # pragma: no cover - depende do checkpoint DFNet3 local
        pytest.skip(f"DFNet3 upstream indisponível: {e}")
    from cctrain.models.film import FilmGenerator, load_film_config
    from cctrain.models.pdfnet3_pro import PDFNet3Pro

    torch.manual_seed(0)
    m = PDFNet3Pro(base, FilmGenerator(load_film_config()), cross_attention_config=X.ATTN_CFG).eval()
    with torch.no_grad():
        for p in (m.film.out.weight, m.film.out.bias, m.cross_attn.out_proj.weight,
                  m.cross_attn.subband_gain_head.weight, m.freq_mixer.pw_conv.weight):
            p.normal_(std=0.1)
    return m


@pytest.mark.parametrize("present", [0.0, 1.0])
def test_stateful_onnx_matches_forward(pro, tmp_path, present):
    from pro_v3_stream_runner import run

    torch.set_num_threads(1)
    for g in X.GRAPHS:
        X.export_graph(pro, g, tmp_path / f"{g}.onnx")
    files = {f"{g}.onnx": tmp_path / f"{g}.onnx" for g in X.GRAPHS}
    (tmp_path / "manifest.json").write_text(json.dumps(X.build_manifest(files, Path("ckpt"), "-")))
    rng = np.random.default_rng(1)
    t = 24
    ferb = rng.standard_normal((t, 32)).astype(np.float32)
    fspec = (0.3 * (rng.standard_normal((t, 96)) + 1j * rng.standard_normal((t, 96)))).astype(np.complex64)
    emb = rng.standard_normal(192).astype(np.float32)
    emb /= np.linalg.norm(emb)
    if present == 0.0:
        emb[:] = 0.0
    ref = X.pro_reference(pro, ferb, fspec, emb, present)
    got = run(tmp_path, "ort", {"feat_erb": ferb, "feat_spec": np.stack([fspec.real, fspec.imag], 1),
                                "speaker_embedding": emb, "speaker_present": np.full(1, present, np.float32)})
    for k in ("m", "coefs", "lsnr"):
        r = getattr(ref, k)
        np.testing.assert_allclose(got[k].reshape(r.shape), r, atol=2e-5, rtol=1e-4)
