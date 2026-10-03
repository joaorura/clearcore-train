import numpy as np
import pytest
import soundfile as sf

from cctrain.data.licenses import LicenseRefused
from cctrain.data.loader import PilotLoaderConfig, PilotMixDataset
from cctrain.data.manifest import FileRecord
from cctrain.hashing import sha256_file


def _write(root, rel, x, sr, source, lic, speaker, split, role="train"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, x.astype("f4"), sr, subtype="PCM_16")
    return FileRecord(path=rel, sha256=sha256_file(p), source_id=source, license=lic, role=role, speaker=speaker,
                      split=split, sr=sr, num_samples=x.size, band_hz=None, codec="flac")


@pytest.fixture()
def corpus(tmp_path):
    rng = np.random.default_rng(0)
    speech = [_write(tmp_path, f"s/{spk}_{k}.flac", 0.1 * rng.standard_normal(sr * 2), sr, "vctk-0.92",
                     "CC-BY-4.0", spk, "pilot") for spk, sr in (("a", 48_000), ("b", 16_000), ("c", 48_000))
              for k in range(2)]
    noise = [_write(tmp_path, "n/n.flac", 0.05 * rng.standard_normal(48_000 * 4), 48_000, "noise-wikimedia-cc0",
                    "CC0-1.0", None, "test")]
    h = np.zeros(4800)
    h[0], h[100] = 0.99, 0.3
    rirs = [_write(tmp_path, "r/r.flac", h, 48_000, "rir-synthetic-v1", "LicenseRef-cctrain-generated", None,
                   "rir")]
    return tmp_path, speech, noise, rirs


def test_items_have_dfnet_shapes(corpus):
    root, speech, noise, rirs = corpus
    it = iter(PilotMixDataset(speech, noise, rirs, root, PilotLoaderConfig(segment_s=1.0)))
    item = next(it)
    assert item["noisy"].shape == (48_000,) and item["clean"].shape == (48_000,)
    assert item["spec"].shape == (1, 100, 481, 2)
    assert item["spec_clean"].shape == (1, 100, 481, 2)
    assert item["feat_erb"].shape == (1, 100, 32)
    assert item["feat_spec"].shape == (1, 100, 96, 2)
    assert np.isfinite(item["noisy"]).all()


def test_dataset_is_deterministic(corpus):
    root, speech, noise, rirs = corpus
    cfg = PilotLoaderConfig(segment_s=1.0, seed=7)
    a = [next(iter(PilotMixDataset(speech, noise, rirs, root, cfg)))["noisy"] for _ in range(2)]
    assert np.array_equal(a[0], a[1])


def test_dataset_refuses_test_only_or_single_speaker(corpus):
    root, speech, noise, rirs = corpus
    bad = [FileRecord(**{**speech[0].to_dict(), "role": "test_only"})]
    with pytest.raises(LicenseRefused):
        PilotMixDataset(bad + speech, noise, rirs, root)
    with pytest.raises(ValueError, match="falantes"):
        PilotMixDataset([r for r in speech if r.speaker == "a"], noise, rirs, root)
