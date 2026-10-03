"""Encoder portado como função áudio 16 kHz -> embedding 192-d L2 (M1 Tarefa 12)."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from cctrain.export.enrollment import scale_to_rms_dbfs, speechlike_16k
from cctrain.m1.encoder import clip_embedder, load_encoder


def test_encoder_function_matches_the_enrollment_graph_embedding(sb_ckpt_dir, sb_enrollment_model):
    enc = load_encoder(sb_ckpt_dir, device="cpu")
    x = scale_to_rms_dbfs(speechlike_16k(96_000, 3), -20.0)
    emb = enc.embed(x)
    with torch.no_grad():
        ref = sb_enrollment_model(torch.from_numpy(x)[None])[4].numpy()
    assert emb.shape == (192,) and abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5
    np.testing.assert_allclose(emb, ref, atol=1e-5)
    assert len(enc.sha256) == 64


def test_clip_embedder_reads_resamples_and_embeds(tmp_path, sb_ckpt_dir):
    enc = load_encoder(sb_ckpt_dir, device="cpu")
    x = scale_to_rms_dbfs(speechlike_16k(48_000, 1), -20.0)
    sf.write(tmp_path / "a.wav", x, 16_000, subtype="FLOAT")
    x48 = np.repeat(x, 3)
    sf.write(tmp_path / "b.wav", x48, 48_000, subtype="FLOAT")
    embed = clip_embedder(enc, tmp_path)
    a, b = embed("a.wav"), embed("b.wav")
    assert a.shape == b.shape == (192,) and float(a @ b) > 0.98
