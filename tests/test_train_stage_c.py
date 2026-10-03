import json
import math
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
import torch

from cctrain.data.manifest import FileRecord, ManifestError
from cctrain.data.mix import MixKind
from cctrain.hashing import sha256_file
from cctrain.models.film import FilmGenerator, load_film_config
from cctrain.train.checkpoint import load_latest
from cctrain.train.stage_c import (
    ArrayEmbeddingBank,
    BaseDriftError,
    StageCConfig,
    StageCData,
    StageCDataset,
    assert_base_unchanged,
    run_stage_c,
    total_pss,
)


def _write(root, rel, x, sr, source, lic, speaker, split, band=None):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, x.astype("f4"), sr, subtype="PCM_16")
    return FileRecord(path=rel, sha256=sha256_file(p), source_id=source, license=lic, role="train", speaker=speaker,
                      split=split, sr=sr, num_samples=x.size, band_hz=band, codec="flac", attribution="test")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("stagec")
    rng = np.random.default_rng(0)
    t = np.arange(48_000 * 3) / 48_000
    speech = []
    for i, spk in enumerate(("f1", "f2", "m1", "m2")):
        for k in range(3):
            env = 0.5 + 0.5 * np.sin(2 * np.pi * (2 + k) * t) ** 2
            x = 0.1 * env * np.sin(2 * np.pi * (110 + 35 * i) * t) + 0.003 * rng.standard_normal(t.size)
            speech.append(_write(root, f"s/{spk}_{k}.flac", x, 48_000, "vctk-0.92", "CC-BY-4.0", spk, "pilot",
                                 band=7_000.0))
    noise = [_write(root, "n/n.flac", 0.05 * rng.standard_normal(48_000 * 4), 48_000, "noise-wikimedia-cc0",
                    "CC0-1.0", None, "noise")]
    h = np.zeros(4800)
    h[40], h[400] = 0.9, 0.2
    rirs = [_write(root, "r/r.flac", h, 48_000, "rir-synthetic-v1", "LicenseRef-cctrain-generated", None, "rir")]
    sex = {"vctk-0.92:f1": "F", "vctk-0.92:f2": "F", "vctk-0.92:m1": "M", "vctk-0.92:m2": "M"}
    brng = np.random.default_rng(1)
    bank = {}
    for k in sex:
        v = brng.standard_normal((5, 192)).astype(np.float32)
        bank[k] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return root, StageCData(speech, noise, rirs, sex), ArrayEmbeddingBank(bank)


def _cfg(tmp_path, **kw):
    base = StageCConfig(run_id="t", batch_size=2, segment_s=1.0, steps_per_epoch=4, num_workers=0, device="cpu",
                        warmup_steps=3, checkpoint_every_min=1e9, runs_dir=tmp_path)
    return replace(base, **kw)


def test_dataset_item_shapes_and_contract(corpus, tmp_path):
    root, data, bank = corpus
    ds = StageCDataset(data, bank, root, _cfg(tmp_path), epoch=0)
    assert len(ds) == 8
    it = ds[0]
    assert it["spec"].shape == (1, 100, 481, 2) and it["spec_clean"].shape == (1, 100, 481, 2)
    assert it["feat_erb"].shape == (1, 100, 32) and it["feat_spec"].shape == (1, 100, 96, 2)
    assert it["embedding"].shape == (192,) and it["mask"].shape == (1,) and float(it["mask"][0]) == 1.0
    assert it["interferer_frames"].shape == (100,) and it["lsnr_mask"].shape == (100,)
    assert np.isfinite(it["spec"]).all() and 0 < it["max_freq"] <= 24_000


def test_dataset_is_deterministic_per_epoch_and_index_and_varies_across_epochs(corpus, tmp_path):
    root, data, bank = corpus
    cfg = _cfg(tmp_path)
    a, b = StageCDataset(data, bank, root, cfg, epoch=0), StageCDataset(data, bank, root, cfg, epoch=0)
    assert np.array_equal(a[3]["spec"], b[3]["spec"])
    assert not np.array_equal(a[3]["spec"], StageCDataset(data, bank, root, cfg, epoch=1)[3]["spec"])
    assert not np.array_equal(a[3]["spec"], a[4]["spec"])


def test_same_sex_sir_and_silence_class_semantics(corpus, tmp_path):
    root, data, bank = corpus
    cfg = _cfg(tmp_path, silence_fraction=0.5, p_rir=0.0)
    ds = StageCDataset(data, bank, root, cfg, epoch=0)
    seen = set()
    for i in range(60):
        it = ds[i]
        kind = MixKind(int(it["kind"]))
        seen.add(kind)
        if kind in (MixKind.TARGET_INTERFERER, MixKind.TARGET_INTERFERER_NOISE):
            assert it["sir_db"] == 0.0
        if kind is MixKind.INTERFERER_ONLY:
            assert math.isnan(float(it["sir_db"]))
            assert not it["lsnr_mask"].any()  # sem alvo: a LocalSnrLoss é mascarada (§3.5)
            assert it["interferer_frames"].any()  # mas o L_gate se aplica (perfil presente, interferente ativo)
        else:
            assert it["lsnr_mask"].all()
        if kind is MixKind.TARGET_NOISE:
            assert not it["interferer_frames"].any()
        if it["interferer_key"]:
            assert data.sex[it["interferer_key"]] == data.sex[it["target_key"]]
            assert it["interferer_key"] != it["target_key"]
    assert seen == set(MixKind)


def test_same_sex_without_a_sex_map_fails_closed(corpus, tmp_path):
    root, data, bank = corpus
    with pytest.raises(ValueError, match="sex"):
        StageCDataset(replace(data, sex=None), bank, root, _cfg(tmp_path), epoch=0)
    StageCDataset(replace(data, sex=None), bank, root, _cfg(tmp_path, same_sex=False), epoch=0)  # explícito: ok


def test_dev_and_test_records_never_enter_training(corpus, tmp_path):
    root, data, bank = corpus
    bad = [FileRecord(**{**r.to_dict(), "split": "dev"}) for r in data.speech[:4]] + list(data.speech[4:])
    with pytest.raises(ManifestError):
        StageCDataset(replace(data, speech=bad), bank, root, _cfg(tmp_path), epoch=0)


def test_bank_speaker_must_cover_training_targets(corpus, tmp_path):
    root, data, bank = corpus
    with pytest.raises(ValueError, match="banco"):
        StageCDataset(data, ArrayEmbeddingBank({}), root, _cfg(tmp_path), epoch=0)


@pytest.fixture(scope="module")
def trained(corpus, tmp_path_factory):
    root, data, bank = corpus
    out = tmp_path_factory.mktemp("run20")
    cfg = _cfg(out, max_steps=6, data_root=root)
    ckpt = run_stage_c(cfg, bank, max_minutes=5, data=data)
    return cfg, ckpt, out


def test_run_trains_only_the_generator_and_keeps_the_base_bit_identical(trained):
    cfg, ckpt, out = trained
    summary = json.loads((out / "m1" / cfg.run_id / "summary.json").read_text())
    assert summary["steps"] == 6 and summary["base_sha256_before"] == summary["base_sha256_after"]
    assert summary["trainable_params"] == sum(p.numel() for p in FilmGenerator(load_film_config()).parameters())
    ck = load_latest(ckpt.parent)
    assert ck.step == 6 and ckpt.name == "ckpt-00000006.pt"
    assert float(ck.model["out.weight"].abs().max()) > 0.0  # a camada zerada saiu do zero: houve gradiente
    log = [json.loads(line) for line in (out / "m1" / cfg.run_id / "loss-log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3, 4, 5, 6] and all(math.isfinite(r["total"]) for r in log)
    assert {"mr", "os", "gate", "lsnr"} <= set(log[0])


def test_probe_reports_pss_of_trainer_plus_workers(trained):
    cfg, _, out = trained
    probe = json.loads((out / "m1" / "stage-c-probe.json").read_text())
    assert probe["pss_gb_max"] > 0 and probe["limit_gb"] == 6.0 and probe["num_workers"] == 0
    assert probe["pass"] == (probe["pss_gb_max"] <= 6.0)
    assert total_pss() > 0


def test_resume_reproduces_the_loss_for_20_steps_on_cpu(corpus, tmp_path):
    root, data, bank = corpus
    straight = _cfg(tmp_path / "a", run_id="straight", max_steps=20, data_root=root, steps_per_epoch=3)
    run_stage_c(straight, bank, max_minutes=10, data=data)
    first = _cfg(tmp_path / "b", run_id="resumed", max_steps=10, data_root=root, steps_per_epoch=3)
    run_stage_c(first, bank, max_minutes=10, data=data)
    second = replace(first, max_steps=20)
    run_stage_c(second, bank, max_minutes=10, data=data)

    def losses(cfg):
        lines = (cfg.runs_dir / "m1" / cfg.run_id / "loss-log.jsonl").read_text().splitlines()
        return {json.loads(x)["step"]: json.loads(x)["total"] for x in lines}

    a, b = losses(straight), losses(second)
    assert sorted(a) == sorted(b) == list(range(1, 21))
    assert all(a[s] == b[s] for s in range(1, 21)), {s: (a[s], b[s]) for s in range(1, 21) if a[s] != b[s]}


def test_resume_with_a_different_config_is_refused(corpus, tmp_path):
    root, data, bank = corpus
    cfg = _cfg(tmp_path, max_steps=2, data_root=root)
    run_stage_c(cfg, bank, max_minutes=5, data=data)
    from cctrain.train.checkpoint import CheckpointError

    with pytest.raises(CheckpointError):
        run_stage_c(replace(cfg, max_steps=4, lr=2e-3), bank, max_minutes=5, data=data)


def test_time_budget_stops_the_loop_after_at_least_one_step(corpus, tmp_path):
    root, data, bank = corpus
    cfg = _cfg(tmp_path, max_steps=1000, data_root=root)
    ckpt = run_stage_c(cfg, bank, max_minutes=0.0, data=data)
    assert 1 <= load_latest(ckpt.parent).step < 1000


def test_base_drift_is_detected(corpus):
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.models.state_hash import state_dict_sha256

    base, _ = load_upstream_dfnet3(device="cpu")
    before = state_dict_sha256(base)
    assert_base_unchanged(base, before)
    with torch.no_grad():
        next(iter(base.parameters())).add_(1e-3)
    with pytest.raises(BaseDriftError):
        assert_base_unchanged(base, before)
    with torch.no_grad():
        next(iter(base.parameters())).sub_(1e-3)
