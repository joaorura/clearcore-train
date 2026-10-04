"""Dev do M2 (Tarefa 4): partição por fala, disjunção, regra do enrollment, recusa de train/test, congelamento."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from cctrain.data.manifest import FileRecord
from cctrain.m2 import devsets as ds

N_SPK, N_UTT = 10, 16


def _rec(path, speaker, n, sr, source="syn", split="dev", sha=None):
    return FileRecord(path=path, sha256=sha or hashlib.sha256(path.encode()).hexdigest(), source_id=source,
                      license="CC0-1.0", role="test_only", speaker=speaker, split=split, sr=sr, num_samples=n,
                      band_hz=None, codec="wav", attribution="sintético")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("devcorpus")
    rng = np.random.default_rng(0)
    speech, sex = [], {}
    for s in range(N_SPK):
        spk = f"s{s:02d}"
        if s != 9:  # s09 sem sexo declarado
            sex[f"syn:{spk}"] = "F" if s < 5 else "M"
        f0 = 100 + 20 * s
        for u in range(N_UTT):
            n = int(rng.uniform(4.0, 8.0) * 16_000)
            t = np.arange(n) / 16_000
            x = 0.2 * sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
            x = (x + 0.01 * rng.standard_normal(n)).astype(np.float32)
            rel = f"speech/{spk}_{u:02d}.wav"
            (root / "speech").mkdir(exist_ok=True)
            sf.write(root / rel, x, 16_000)
            speech.append(_rec(rel, spk, n, 16_000))
    noise = []
    for k in range(8):
        n = 10 * 48_000
        rel = f"noise/n{k}.wav"
        (root / "noise").mkdir(exist_ok=True)
        sf.write(root / rel, (0.1 * rng.standard_normal(n)).astype(np.float32), 48_000)
        noise.append(_rec(rel, None, n, 48_000, source="synnoise", split="noise"))
    return root, speech, noise, sex


def fake_embed(a):
    h = hashlib.sha256(np.asarray(a, np.float32).tobytes()).digest()
    v = np.random.default_rng(int.from_bytes(h[:8], "big")).standard_normal(192).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def built(corpus, tmp_path_factory):
    root, speech, noise, sex = corpus
    out = tmp_path_factory.mktemp("devout")
    summary = ds.build_dev(speech, noise, root, out, sex, fake_embed)
    return out, summary, sex


def test_config_is_frozen_and_valid():
    cfg = ds.load_devsets_config()
    assert cfg.halves == ("dev-select", "dev-gate") and cfg.source_split == "dev"
    assert cfg.per("C3", "items_per_speaker_per_sir", "vctk-0.92") == 3
    assert cfg.per("C3", "items_per_speaker_per_sir", "librispeech-dev-clean") == 1


def test_config_edit_is_refused(tmp_path):
    cfg_text = ds.DEVSETS_V1.read_text()
    p = tmp_path / "m2-devsets-v1.yaml"
    p.write_text(cfg_text.replace("seed: 20261004", "seed: 1"))
    (tmp_path / "m2-devsets-v1.sha256").write_text(ds.DEVSETS_V1.with_suffix(".sha256").read_text())
    with pytest.raises(Exception, match="alterado depois de congelado"):
        ds.load_devsets_config(p)


@pytest.mark.parametrize("split", ["train", "test"])
def test_build_dev_refuses_train_and_test(corpus, tmp_path, split):
    root, speech, noise, sex = corpus
    with pytest.raises(ds.DevSetsError, match="recusa"):
        ds.build_dev(speech, noise, root, tmp_path, sex, fake_embed, split=split)
    bad = [dataclasses.replace(speech[0], split=split)] + speech[1:]
    with pytest.raises(ds.DevSetsError, match="recusa"):
        ds.build_dev(bad, noise, root, tmp_path, sex, fake_embed)


def test_partition_is_disjoint_across_halves(corpus):
    _, speech, _, _ = corpus
    cfg = ds.load_devsets_config()
    parts = ds.partition_utterances(speech, cfg)
    for spk, halves in parts.items():
        sel, gate = halves["dev-select"], halves["dev-gate"]
        assert not ({r.path for r in sel.pool} & {r.path for r in gate.pool}), spk
        assert abs(len(sel.pool) - len(gate.pool)) <= 1
    assert sum(len(h.pool) for hs in parts.values() for h in hs.values()) == len(speech)


def test_enrollment_uses_only_pool_utterances_and_leaves_the_rest_for_mixtures(corpus):
    root, speech, _, _ = corpus
    cfg = ds.load_devsets_config()
    pool = ds.partition_utterances(speech, cfg)["syn:s00"]["dev-select"]
    audio, used = ds.choose_enrollment(root, pool.pool, ds._rng(1, "t"), cfg)
    assert 6.0 * 16_000 <= len(audio) <= 10.0 * 16_000 and used
    assert {r.path for r in used} <= {r.path for r in pool.pool}
    mix = pool.mix_pool(used, cfg)
    assert not ({r.path for r in used} & {r.path for r in mix}) and mix
    assert all(cfg.target_min_s <= r.num_samples / r.sr <= cfg.target_max_s for r in mix)


def test_partition_is_deterministic(corpus):
    _, speech, _, _ = corpus
    cfg = ds.load_devsets_config()
    a = ds.partition_utterances(speech, cfg)
    b = ds.partition_utterances(list(reversed(speech)), cfg)
    assert a == b


def test_halves_disjoint_by_utterance_and_noise_and_enrollment_rule(built):
    out, summary, _ = built
    sel, gate = ds.open_half(out, "dev-select"), ds.open_half(out, "dev-gate", final_gate=True, who="test")
    counts = ds.check_disjoint_and_enrollment_rule(sel, gate)
    assert counts["utterances_select"] > 0 and counts["utterances_gate"] > 0
    assert set(summary) == {"dev-select", "dev-gate"}


def test_disjointness_check_detects_a_leak(built, tmp_path):
    out, _, _ = built
    sel = ds.open_half(out, "dev-select")
    gate = ds.open_half(out, "dev-gate", final_gate=True, who="test")
    leaked = dataclasses.replace(gate.items[0], meta={**gate.items[0].meta, "target_utt": sel.items[0]["target_utt"]})
    gate2 = ds.DevHalf.__new__(ds.DevHalf)
    gate2.name, gate2.root, gate2.enrollments = gate.name, gate.root, gate.enrollments
    gate2.items = [leaked, *gate.items[1:]]
    with pytest.raises(ds.DevSetsError, match="nas duas metades"):
        ds.check_disjoint_and_enrollment_rule(sel, gate2)


def test_enrollment_rule_detects_mixture_utterance_in_enrollment(built):
    out, _, _ = built
    sel = ds.open_half(out, "dev-select")
    i = sel.select("C3")[0]
    bad = dataclasses.replace(i, meta={**i.meta, "target_utt": sel.enrollments[i["profile_speaker"]].utterances[0]})
    sel2 = ds.DevHalf.__new__(ds.DevHalf)
    sel2.name, sel2.root, sel2.enrollments, sel2.items = sel.name, sel.root, sel.enrollments, [bad]
    gate = ds.open_half(out, "dev-gate", final_gate=True, who="test")
    with pytest.raises(ds.DevSetsError, match="enrollment contém a fala"):
        ds.check_disjoint_and_enrollment_rule(sel2, gate)


def test_conditions_sex_and_sir_rules(built):
    out, _, sex = built
    for name in ds.HALVES:
        half = ds.open_half(out, name, final_gate=True, who="test")
        for cond in ds.CONDITIONS:
            assert half.select(cond), (name, cond)
        for i in half.select("C3") + half.select("C4"):
            assert sex[i["target_speaker"]] == sex[i["interferer_speaker"]]  # só mesmo sexo declarado
            assert i["interferer_speaker"] != i["target_speaker"] and i["profile_speaker"] == i["target_speaker"]
            assert i["sir_db"] in (0.0, 5.0)
            assert (i["noise"] is None) == (i["condition"] == "C3")
        for i in half.select("C7"):
            a, b = i["profile_speaker"], i["target_speaker"]
            assert a != b and a in sex and b in sex
            assert (sex[a] == sex[b]) == (i["sex_pair"] == "same")
            assert i["clean"] is None
        undeclared = "syn:s09"
        for i in half.items:
            if i["condition"] in ("C3", "C4", "C7"):
                assert undeclared not in (i["target_speaker"], i["interferer_speaker"], i["profile_speaker"])
        assert any(i["target_speaker"] == undeclared for i in half.select("C0"))  # fora do mesmo sexo, não do dev
        for i in half.select("C0"):
            assert i["profile_speaker"] is None and half.embedding(i) is None


def test_sir_is_exact_and_audio_roundtrip(built):
    out, _, _ = built
    half = ds.open_half(out, "dev-select")
    i = half.select("C3", sir_db=0.0)[0]
    noisy, clean = i.noisy(), i.clean()
    assert noisy.dtype == np.float32 and noisy.shape == clean.shape and noisy.size == i["samples"]
    interf = noisy - clean
    sir = 10 * np.log10(np.mean(clean.astype(np.float64) ** 2) / np.mean(interf.astype(np.float64) ** 2))
    assert abs(sir - 0.0) < 0.05
    assert np.abs(noisy).max() <= 0.99 + 1e-4
    c0c = half.select("C0c")[0]
    assert np.array_equal(c0c.noisy(), c0c.clean())


def test_noise_halves_are_disjoint_by_file(built):
    out, _, _ = built
    a = {i["noise"] for i in ds.open_half(out, "dev-select").items if i["noise"]}
    b = {i["noise"] for i in ds.open_half(out, "dev-gate", final_gate=True, who="test").items if i["noise"]}
    assert a and b and not (a & b)


def test_dev_gate_is_locked_and_opening_is_logged(built):
    out, _, _ = built
    with pytest.raises(ds.DevGateLocked):
        ds.open_half(out, "dev-gate")
    with pytest.raises(ds.DevGateLocked):
        ds.open_half(out, "dev-gate", final_gate=False)
    ds.open_half(out, "dev-gate", final_gate=True, who="teste")
    assert "teste" in (out / "dev-gate" / "opened.log").read_text()
    with pytest.raises(ds.DevSetsError):
        ds.open_half(out, "test")


def test_build_is_deterministic_and_freeze_verifies(corpus, built, tmp_path):
    root, speech, noise, sex = corpus
    out, _, _ = built
    ds.build_dev(speech, noise, root, tmp_path / "again", sex, fake_embed)
    a, b = ds.make_freeze(out), ds.make_freeze(tmp_path / "again")
    assert a == b  # mesmas listas, mesmos áudios, mesmos embeddings
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(a))
    ds.verify_freeze(out, freeze, deep=True)


def test_verify_freeze_detects_tampering(corpus, tmp_path):
    root, speech, noise, sex = corpus
    ds.build_dev(speech, noise, root, tmp_path / "d", sex, fake_embed)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(ds.make_freeze(tmp_path / "d")))
    victim = sorted((tmp_path / "d" / "dev-select" / "audio").glob("*.noisy.npy"))[0]
    arr = np.load(victim)
    arr[0] += 1
    np.save(victim, arr)
    with pytest.raises(ds.DevSetsError, match="SHA-256 não confere"):
        ds.verify_freeze(tmp_path / "d", freeze, deep=True)


def test_meta_only_open_has_no_audio_and_leaves_no_trace(built):
    out, _, _ = built
    log = out / "dev-gate" / "opened.log"
    before = log.read_text() if log.exists() else ""
    gate = ds.open_half(out, "dev-gate", meta_only=True)
    assert gate.items and gate.enrollments
    with pytest.raises(ds.DevGateLocked):
        gate.items[0].noisy()
    with pytest.raises(ds.DevGateLocked):
        next(iter(gate.enrollments.values())).audio16k()
    assert (log.read_text() if log.exists() else "") == before
