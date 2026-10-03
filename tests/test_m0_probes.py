import io

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from cctrain.data.manifest import FileRecord
from cctrain.hashing import sha256_file
from cctrain.m0.beta_probe import beta_rule_report, write_beta_bound
from cctrain.m0.disk_probe import flac_ratio_by_sr
from cctrain.m0.gpu_monitor import parse_line
from cctrain.m0.step_probe import MultiResSpecLossProxy, istft_proxy


def test_parse_nvidia_smi_line():
    s = parse_line("2026/10/03 10:00:00.000, 71, 2100, 48.5, 3120, 0x0000000000000004")
    assert (s.temperature_c, s.sm_clock_mhz, s.power_w, s.memory_used_mib) == (71.0, 2100.0, 48.5, 3120.0)
    assert s.reasons == "0x0000000000000004"


def test_flac_ratio_by_sr(tmp_path):
    x = (0.1 * np.random.default_rng(0).standard_normal(16_000 * 2)).astype("f4")
    p = tmp_path / "a.wav"
    sf.write(p, x, 16_000, subtype="PCM_16")
    rec = FileRecord(path="a.wav", sha256=sha256_file(p), source_id="s", license="CC0-1.0", role="train",
                     speaker=None, split="noise", sr=16_000, num_samples=x.size, band_hz=None, codec="wav", attribution="test")
    out = flac_ratio_by_sr([rec], tmp_path)["16000"]
    y, _ = sf.read(p, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, y, 16_000, format="FLAC", subtype="PCM_16")
    assert out["pcm16_bytes"] == x.size * 2
    assert out["flac_bytes"] == len(buf.getvalue())
    assert abs(out["hours"] - 2 / 3600) < 1e-9


def test_beta_rule_report_pools_both_sites():
    acts = {"enc": np.linspace(0, 1, 1001), "df_dec": np.linspace(0, 3, 1001)}
    rep = beta_rule_report(acts)
    pooled = float(np.percentile(np.concatenate(list(acts.values())), 99))
    assert rep["p99_pooled"] == pytest.approx(pooled)
    assert rep["beta_bound"] == min(50.0, max(4.0, 2 * pooled))


def test_write_beta_bound_only_touches_that_line(tmp_path):
    p = tmp_path / "film.yaml"
    p.write_text("version: 1\nlog_gamma_bound: 2.302585092994046\nbeta_bound: null\nmid_dim: 256\n")
    write_beta_bound(5.25, p)
    assert p.read_text().splitlines()[2].startswith("beta_bound: 5.25")
    with pytest.raises(ValueError):
        write_beta_bound(6.0, p)


def test_loss_proxy_is_zero_for_identical_inputs_and_positive_otherwise():
    spec = torch.randn(2, 1, 20, 481, 2)
    loss = MultiResSpecLossProxy()
    y = istft_proxy(spec)
    assert float(loss(y, y)) == 0.0
    assert float(loss(y, istft_proxy(spec * 0.5))) > 0.0
    assert isinstance(loss, nn.Module)


def _wav_record(root, name, sr, source, seed):
    x = (0.1 * np.random.default_rng(seed).standard_normal(sr)).astype("f4")
    sf.write(root / name, x, sr, subtype="PCM_16")
    return FileRecord(path=name, sha256=sha256_file(root / name), source_id=source, license="CC0-1.0",
                      role="train", speaker=None, split="noise", sr=sr, num_samples=x.size, band_hz=None,
                      codec="wav", attribution="test")


def test_disk_report_splits_by_source_and_uses_main_checkout_for_tract_target(tmp_path, monkeypatch):
    from cctrain.m0.disk_probe import disk_report
    from cctrain.paths import Paths

    main = tmp_path / "main"
    (main / "tools/tract-check/target").mkdir(parents=True)
    (main / "tools/tract-check/target/a.bin").write_bytes(b"x" * 1000)
    data = tmp_path / "data"
    data.mkdir()
    recs = [_wav_record(data, "a.wav", 48_000, "src-a", 1), _wav_record(data, "b.wav", 48_000, "src-b", 2),
            _wav_record(data, "c.wav", 16_000, "src-b", 3)]
    monkeypatch.setenv("CCTRAIN_ROOT", str(main))
    worktree = tmp_path / "wt"
    worktree.mkdir()
    paths = Paths(repo=worktree, data=data, runs=tmp_path / "runs", delivery=worktree, build=worktree)
    rep = disk_report(paths, recs)
    assert set(rep["by_source"]) == {"src-a", "src-b"}
    assert set(rep["by_source"]["src-a"]) == {"48000"} and set(rep["by_source"]["src-b"]) == {"48000", "16000"}
    assert rep["by_source"]["src-a"]["48000"]["files"] == 1
    assert rep["dirs_bytes"]["tract_check_target"] == 1000


def test_dir_size_counts_hardlinks_once(tmp_path):
    from cctrain.m0.disk_probe import dir_size

    (tmp_path / "a").write_bytes(b"x" * 500)
    (tmp_path / "b").hardlink_to(tmp_path / "a")
    (tmp_path / "c").symlink_to(tmp_path / "a")
    assert dir_size(tmp_path) == 500


def test_beta_limitation_note_and_current_bound(tmp_path):
    from cctrain.m0.beta_probe import current_beta_bound, limitation_note

    assert "7.4 > B=4" in limitation_note(7.43, 4.0) and "<= B=" in limitation_note(3.0, 4.0)
    p = tmp_path / "film.yaml"
    p.write_text("beta_bound: null\n")
    assert current_beta_bound(p) is None
    p.write_text("beta_bound: 4.0  # fixado\n")
    assert current_beta_bound(p) == 4.0
    enc = np.linspace(0, 1, 1001)
    enc[-1] = 7.4  # cauda acima de B, p99 baixo
    rep = beta_rule_report({"enc": enc, "df_dec": np.linspace(0, 1, 1001)})
    assert "7.4 > B=4" in rep["limitation"]
