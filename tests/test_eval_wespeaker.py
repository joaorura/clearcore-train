"""Avaliador WeSpeaker ResNet34 (M1 Tarefa 9): fbank Kaldi, checkpoint pinado, embedding de 256 dimensões."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import yaml

from cctrain.eval.wespeaker import (
    MODEL_MANIFEST,
    CheckpointMismatch,
    WeSpeakerEvaluator,
    cosine,
    kaldi_fbank,
    load_model_manifest,
)


def test_manifest_pins_license_revision_and_hash():
    m = load_model_manifest()
    assert m["license"] == "Apache-2.0"
    assert m["role"] == "tool_only"
    assert len(m["revision"]) == 40
    f = m["files"]["voxceleb_resnet34.onnx"]
    assert len(f["sha256"]) == 64 and f["size"] > 1_000_000
    assert "Apache" in yaml.safe_load(MODEL_MANIFEST.read_text())["license"]


def test_fbank_shape_and_cmn():
    x = np.random.default_rng(0).standard_normal(16_000).astype(np.float32) * 0.1
    f = kaldi_fbank(x)
    assert f.shape == (98, 80)  # snip_edges: 1 + (16000 - 400) // 160
    assert np.allclose(f.mean(axis=0), 0.0, atol=1e-4)  # CMN por enunciado
    assert np.isfinite(f).all()


def test_fbank_rejects_too_short_and_non_finite():
    with pytest.raises(ValueError, match="curto"):
        kaldi_fbank(np.zeros(1000, np.float32))
    with pytest.raises(ValueError, match="finit"):
        kaldi_fbank(np.full(16_000, np.nan, np.float32))


def test_cosine():
    a = np.array([1.0, 0.0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, np.array([0.0, 2.0])) == pytest.approx(0.0)
    assert cosine(a, -a) == pytest.approx(-1.0)


def test_checkpoint_hash_mismatch_fails_closed(paths, tmp_path):
    src = paths.data / "models" / "wespeaker-voxceleb-resnet34" / "voxceleb_resnet34.onnx"
    if not src.is_file():
        pytest.skip(f"checkpoint WeSpeaker ausente em {src}")
    bad = tmp_path / "voxceleb_resnet34.onnx"
    shutil.copy(src, bad)
    with bad.open("r+b") as f:
        f.seek(1000)
        f.write(b"\x00\x01\x02\x03")
    with pytest.raises(CheckpointMismatch, match="sha256"):
        WeSpeakerEvaluator(bad)


@pytest.fixture(scope="module")
def evaluator(paths):
    p = paths.data / "models" / "wespeaker-voxceleb-resnet34" / "voxceleb_resnet34.onnx"
    if not p.is_file():
        pytest.skip(f"checkpoint WeSpeaker ausente em {p}")
    return WeSpeakerEvaluator(p, threads=2)


def test_embed_shape_batch_consistency(evaluator):
    rng = np.random.default_rng(1)
    a = (rng.standard_normal(32_000) * 0.05).astype(np.float32)
    b = (rng.standard_normal(40_000) * 0.05).astype(np.float32)
    ea, eb = evaluator.embed(a), evaluator.embed(b)
    assert ea.shape == (256,) and ea.dtype == np.float32 and np.isfinite(ea).all()
    out = evaluator.embed_many([a, b])
    assert out.shape == (2, 256)
    np.testing.assert_allclose(out[0], ea, atol=1e-4)
    np.testing.assert_allclose(out[1], eb, atol=1e-4)
    np.testing.assert_allclose(evaluator.embed(a), ea, atol=0)  # determinístico


@pytest.mark.data
def test_same_speaker_closer_than_different(evaluator, paths):
    """Fala real do LibriSpeech test-clean: mesmo falante tem cosseno maior que falantes diferentes."""
    import soundfile as sf

    root = paths.data / "raw" / "librispeech-test-clean" / "test-clean"
    if not root.is_dir():
        pytest.skip("LibriSpeech test-clean ausente")
    spk = sorted(p.name for p in root.iterdir())[:3]
    embs = {}
    for s in spk:
        files = sorted((root / s).rglob("*.flac"))[:6]
        embs[s] = [evaluator.embed(sf.read(str(f), dtype="float32")[0]) for f in files]
    same = [cosine(e[i], e[j]) for e in embs.values() for i in range(len(e)) for j in range(i + 1, len(e))]
    diff = [cosine(embs[a][i], embs[b][i]) for a in spk for b in spk if a < b for i in range(6)]
    assert np.mean(same) > np.mean(diff) + 0.2
