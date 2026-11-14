"""Equivalência fim a fim CPU (`StageV4Dataset`) vs `--gpu-mix` (receita + `GpuMixer`) com dados reais; pula sem o
banco de embeddings. Na GPU quando houver, senão em CPU (mesmo código torch)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_gpu_mix_matches_cpu(paths, capsys):
    if not (paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz").is_file():
        pytest.skip("banco de embeddings ausente")
    from gpu_mix_equiv import make, run

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds, mixer, _ = make(dev, batch_size=6)
    r = run(ds, mixer, list(range(24)), 6)
    with capsys.disabled():
        print(f"\n{dev}: {r}")
    assert r["reverb"] > 0 and r["dry_with_interf"] > 0
    assert not r["meta_mismatch"]
    assert max(r["label_frames_diff_pct"].values()) == 0.0
    assert max(r["audio_rel_peak"].values()) < 1e-5
    assert max(r[k] for k in ("level_db_maxdiff", "snr_eff_db_maxdiff")) < 1e-3
    fr = r["feat_rel_peak"]
    assert max(fr["spec"], fr["spec_clean"], fr["spec_interf"], fr["feat_spec"]) < 1e-5 and fr["feat_erb"] < 1e-3
