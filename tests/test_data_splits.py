"""Splits disjuntos por falante, comuns ao M1 e ao M2 (spec §4.5; M1 Tarefa 9)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from cctrain.data.licenses import LicenseRefused, check_train_allowed
from cctrain.data.manifest import FileRecord, ManifestError, load_sources, read_file_manifest, training_records
from cctrain.data.splits import (
    SPLITS_V1,
    SplitError,
    apply_splits,
    assign_splits,
    dedupe_by_sha256,
    load_split_config,
    speaker_sex,
)
from cctrain.hashing import sha256_file
from cctrain.paths import repo_root

FILES = repo_root() / "manifests" / "files"
VCTK_FIXED_TEST = {"vctk-0.92:p225", "vctk-0.92:p226", "vctk-0.92:p232", "vctk-0.92:p257"}
GOLDEN_VCTK_DRAWN = [f"vctk-0.92:p{n}" for n in (231, 250, 256, 269, 273, 278, 292, 308, 318, 363)]


def _rec(source: str, spk: str | None, i: int = 0, split: str = "train", role: str = "train", sha: str | None = None):
    return FileRecord(path=f"raw/{source}/{spk}/{i}.flac", sha256=sha or f"{abs(hash((source, spk, i))) % 10**12:064x}",
                      source_id=source, license="CC-BY-4.0", role=role, speaker=spk, split=split, sr=16000,
                      num_samples=16000, band_hz=None, codec="flac", attribution="x")


@pytest.fixture(scope="module")
def cfg():
    return load_split_config(SPLITS_V1)


def _vctk(n=109):
    return [_rec("vctk-0.92", f"p{225 + k}", i) for k in range(n) for i in range(2)]


def test_vctk_fixed_test_plus_ten_random_and_dev(cfg):
    a = assign_splits(_vctk(), cfg)
    test = {k for k, v in a.items() if v == "test"}
    assert VCTK_FIXED_TEST <= test and len(test) == 14
    dev = {k for k, v in a.items() if v == "dev"}
    assert len(dev) == cfg.sources["vctk-0.92"]["dev_random"] and not dev & test
    assert sum(v == "train" for v in a.values()) == 109 - 14 - len(dev)


def test_assignment_is_deterministic_and_order_independent(cfg):
    recs = _vctk()
    assert assign_splits(recs, cfg) == assign_splits(list(reversed(recs)), cfg)


def test_random_pick_depends_on_seed(cfg):
    import dataclasses

    other = dataclasses.replace(cfg, seed=cfg.seed + 1)
    assert assign_splits(_vctk(), cfg) != assign_splits(_vctk(), other)


def test_official_splits_are_kept_and_must_be_consistent(cfg):
    recs = [_rec("cml-tts-pt", "1", split="dev"), _rec("cml-tts-pt", "2", split="test"),
            _rec("cml-tts-pt", "3", split="train")]
    assert assign_splits(recs, cfg) == {"cml-tts-pt:1": "dev", "cml-tts-pt:2": "test", "cml-tts-pt:3": "train"}
    with pytest.raises(SplitError, match="mais de um split"):
        assign_splits([_rec("cml-tts-pt", "1", split="dev"), _rec("cml-tts-pt", "1", 1, split="train")], cfg)


def test_unknown_source_fails_closed(cfg):
    with pytest.raises(SplitError, match="sem regra"):
        assign_splits([_rec("fonte-nova", "a")], cfg)


def test_noise_records_are_ignored(cfg):
    assert assign_splits([_rec("dns-noise-fullband-cc0", None, split="noise")], cfg) == {}


def test_cv_dev_pool_before_filter_and_test_pool_from_retained(cfg):
    big = [_rec("cv27-pt", f"c{c:03d}", i) for c in range(300) for i in range(25)]
    small = [_rec("cv27-pt", f"s{c:03d}", i) for c in range(50) for i in range(3)]
    stage1 = assign_splits(big + small, cfg)
    dev = {k for k, v in stage1.items() if v == "dev"}
    assert len(dev) == cfg.sources["cv27-pt"]["dev_clients"]
    assert all(k.startswith("cv27-pt:c") for k in dev)  # só quem tem >= min_clips
    assert "test" not in stage1.values()  # sem retidos, nenhum teste ainda
    retained = {f"cv27-pt:c{c:03d}" for c in range(0, 300, 2)}
    stage2 = assign_splits(big + small, cfg, cv_retained=retained)
    assert {k for k, v in stage2.items() if v == "dev"} == dev  # o dev não muda com o filtro
    test = {k for k, v in stage2.items() if v == "test"}
    assert len(test) == cfg.sources["cv27-pt"]["test_clients"] and test <= retained and not test & dev
    assert all(v == "train" for k, v in stage2.items() if k not in dev | test)


def test_apply_splits_marks_dev_test_as_non_trainable():
    recs = [_rec("cml-tts-pt", "1", split="test", role="train"), _rec("cml-tts-pt", "2", split="train")]
    out = apply_splits(recs, {"cml-tts-pt:1": "test", "cml-tts-pt:2": "train"})
    assert [r.split for r in out] == ["test", "train"] and out[0].role == "test_only" and out[1].role == "train"
    with pytest.raises(ManifestError, match="dev/teste"):
        training_records([out[0]], load_sources())
    with pytest.raises(LicenseRefused):
        check_train_allowed(out[0].license, out[0].role, what=out[0].path)
    with pytest.raises(SplitError, match="sem split"):
        apply_splits(recs, {"cml-tts-pt:1": "test"})


def test_dedupe_by_sha256_keeps_first_by_path():
    a = _rec("dns", None, 1, split="noise", sha="a" * 64)
    b = _rec("dns", None, 0, split="noise", sha="a" * 64)
    c = _rec("dns", None, 2, split="noise", sha="b" * 64)
    out = dedupe_by_sha256([a, b, c])
    assert [r.path for r in out] == [b.path, c.path]


def test_speaker_sex_tables(tmp_path: Path):
    d = tmp_path
    (d / "raw" / "librispeech-test-clean").mkdir(parents=True)
    (d / "raw" / "librispeech-test-clean" / "SPEAKERS.TXT").write_text(
        "; c\n;ID  |SEX| SUBSET | MINUTES | NAME\n1089 | M | test-clean | 1 | A\n1188 | F | test-clean | 1 | B\n")
    (d / "raw" / "vctk-0.92").mkdir(parents=True)
    (d / "raw" / "vctk-0.92" / "speaker-info.txt").write_text(
        "ID  AGE  GENDER  ACCENTS  REGION COMMENTS \np225  23  F    English    Southern  England\np226  22  M    English    Surrey\n")
    assert speaker_sex("librispeech-test-clean:1089", d) == "M"
    assert speaker_sex("librispeech-test-clean:1188", d) == "F"
    assert speaker_sex("vctk-0.92:p225", d) == "F"
    assert speaker_sex("vctk-0.92:p226", d) == "M"
    assert speaker_sex("cml-tts-pt:1", d) is None
    assert speaker_sex("vctk-0.92:p999", d) is None


# ---- teste de disjunção sobre todos os manifests versionados (e o CV local, se houver) ----


def _all_speech_records(paths):
    recs = []
    for f in sorted(FILES.glob("*.jsonl")):
        if f.name.endswith(".params.jsonl"):
            continue
        for r in read_file_manifest(f):
            if r.speaker is not None and r.split != "pilot":
                recs.append(r)
    cv = paths.data / "manifests" / "cv27-pt-full.jsonl"
    return recs, cv


def test_disjointness_over_all_manifests(cfg, paths):
    recs, cv = _all_speech_records(paths)
    if cv.is_file():
        recs += [r for r in read_file_manifest(cv)]
    assign = assign_splits(recs, cfg)
    final = apply_splits(recs, assign)
    split_of: dict[str, set[str]] = defaultdict(set)
    for r in final:
        split_of[r.speaker_key].add(r.split)
    assert all(len(v) == 1 for v in split_of.values())  # nenhum falante em dois splits
    by_split: dict[str, set[str]] = defaultdict(set)
    for k, v in split_of.items():
        by_split[next(iter(v))].add(k)
    assert not by_split["train"] & (by_split["dev"] | by_split["test"])
    assert not by_split["dev"] & by_split["test"]
    # namespace: nenhuma chave sem `source:`
    assert all(":" in k for k in split_of)
    # o mesmo áudio (sha256) nunca aparece em splits diferentes
    sha_split: dict[str, set[str]] = defaultdict(set)
    for r in final:
        sha_split[r.sha256].add(r.split)
    assert not [h for h, s in sha_split.items() if len(s) > 1]
    # dev/test nunca são treináveis
    for r in final:
        if r.split in ("dev", "test"):
            assert r.role == "test_only"


def test_vctk_fixed_and_drawn_test_speakers_on_real_manifest(cfg, paths):
    recs, _ = _all_speech_records(paths)
    non_cv = [r for r in recs if r.source_id != "cv27-pt"]
    assign = assign_splits(non_cv, cfg)
    test = sorted(k for k, v in assign.items() if v == "test" and k.startswith("vctk-0.92:"))
    assert VCTK_FIXED_TEST <= set(test) and len(test) == 14
    golden = json.loads((repo_root() / "configs" / "data" / "splits-v1.resolved.json").read_text())
    assert golden["assignment"] == assign  # o split versionado é o que a configuração produz
    assert golden["config_sha256"] == sha256_file(SPLITS_V1)
    # as 10 sorteadas ficam fixas (a ordem vem do sha256, não de uma biblioteca)
    assert [k for k in test if k not in VCTK_FIXED_TEST] == GOLDEN_VCTK_DRAWN


def test_splits_config_is_frozen():
    lock = SPLITS_V1.with_suffix(".sha256").read_text().split()[0]
    assert lock == sha256_file(SPLITS_V1)
