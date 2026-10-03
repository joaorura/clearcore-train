import json
from pathlib import Path

import pytest

from cctrain.data.licenses import LicenseRefused
from cctrain.data.manifest import (
    MANIFEST_SCHEMA,
    SPLITS,
    TRAIN_SPLITS,
    FileRecord,
    ManifestError,
    load_sources,
    lock_entry,
    migrate_v1,
    read_file_manifest,
    training_records,
    update_lock,
    write_file_manifest,
)

SHA = "a" * 64


def _rec(**kw) -> FileRecord:
    base = dict(path="raw/vctk/p227/p227_001_mic1.flac", sha256=SHA, source_id="vctk-0.92", license="CC-BY-4.0",
                role="train", speaker="p227", split="train", sr=48000, num_samples=48000, band_hz=20000.0,
                codec="flac", attribution="Univ. of Edinburgh, CC BY 4.0", schema=2)
    base.update(kw)
    return FileRecord.from_dict(base)


def test_pilot_sources_load_and_ids_match_filenames():
    sources = load_sources()
    assert set(sources) == {"vctk-0.92", "librispeech-dev-clean", "noise-wikimedia-cc0", "cv27-pt",
                            "rir-synthetic-v1", "noise-synthetic-v1"}
    assert sources["vctk-0.92"].license == "CC-BY-4.0"
    assert sources["cv27-pt"].license == "CC0-1.0"
    assert "no_rehost" in sources["cv27-pt"].restrictions
    assert len(sources["noise-wikimedia-cc0"].artifacts) == 3


def test_source_missing_key_is_rejected(tmp_path: Path):
    (tmp_path / "x.yaml").write_text("id: x\nname: X\n")
    with pytest.raises(ManifestError, match="faltam"):
        load_sources(tmp_path)


def test_source_id_must_match_filename(tmp_path: Path):
    src = (Path(__file__).resolve().parents[1] / "manifests/sources/vctk-0.92.yaml").read_text(encoding="utf-8")
    (tmp_path / "other.yaml").write_text(src)
    with pytest.raises(ManifestError, match="id"):
        load_sources(tmp_path)


@pytest.mark.parametrize("bad", [dict(sha256="xyz"), dict(path="../escape.flac"), dict(path="/abs.flac"),
                                 dict(sr=0), dict(split="nope"), dict(role="nope")])
def test_file_record_validation(bad):
    with pytest.raises((ManifestError, ValueError)):
        _rec(**bad)


def test_write_read_roundtrip_is_deterministic(tmp_path: Path):
    recs = [_rec(path="b.flac"), _rec(path="a.flac")]
    h1 = write_file_manifest(recs, tmp_path / "m1.jsonl")
    h2 = write_file_manifest(list(reversed(recs)), tmp_path / "m2.jsonl")
    assert h1 == h2
    back = read_file_manifest(tmp_path / "m1.jsonl")
    assert [r.path for r in back] == ["a.flac", "b.flac"]
    assert back[0] == _rec(path="a.flac")


def test_training_records_enforce_allowlist_and_source_license():
    sources = load_sources()
    assert training_records([_rec()], sources) == [_rec()]
    with pytest.raises(ManifestError, match="licença"):
        training_records([_rec(license="CC0-1.0")], sources)
    with pytest.raises(LicenseRefused):
        training_records([_rec(role="test_only")], sources)
    with pytest.raises(ManifestError, match="fonte"):
        training_records([_rec(source_id="voxceleb")], sources)


def test_training_records_cross_check_record_role_against_source_role(tmp_path: Path):
    src = (Path(__file__).resolve().parents[1] / "manifests/sources/vctk-0.92.yaml").read_text(encoding="utf-8")
    (tmp_path / "demand-test.yaml").write_text(
        src.replace("id: vctk-0.92", "id: demand-test").replace("role: train", "role: test_only"), encoding="utf-8"
    )
    sources = load_sources(tmp_path)
    assert sources["demand-test"].role.value == "test_only"
    with pytest.raises(ManifestError, match="papel"):
        training_records([_rec(source_id="demand-test")], sources)


def test_duplicate_paths_are_rejected(tmp_path: Path):
    with pytest.raises(ManifestError, match="duplicado"):
        write_file_manifest([_rec(), _rec(sha256="b" * 64)], tmp_path / "m.jsonl")


def test_from_dict_rejects_missing_and_extra_keys():
    good = _rec().to_dict()
    with pytest.raises(ManifestError, match="chaves"):
        FileRecord.from_dict({k: v for k, v in good.items() if k != "codec"})
    with pytest.raises(ManifestError, match="chaves"):
        FileRecord.from_dict({**good, "extra": 1})


def test_invalid_role_is_manifest_error(tmp_path: Path):
    with pytest.raises(ManifestError):
        _rec(role="nope")
    src = (Path(__file__).resolve().parents[1] / "manifests/sources/vctk-0.92.yaml").read_text(encoding="utf-8")
    (tmp_path / "vctk-0.92.yaml").write_text(src.replace("role: train", "role: nope"), encoding="utf-8")
    with pytest.raises(ManifestError, match="papel"):
        load_sources(tmp_path)


@pytest.mark.parametrize("lic_line", ["license:", "license: null", "license: ''"])
def test_null_or_empty_source_license_is_rejected(tmp_path: Path, lic_line: str):
    src = (Path(__file__).resolve().parents[1] / "manifests/sources/vctk-0.92.yaml").read_text(encoding="utf-8")
    (tmp_path / "vctk-0.92.yaml").write_text(src.replace("license: CC-BY-4.0", lic_line), encoding="utf-8")
    with pytest.raises(ManifestError, match="licença"):
        load_sources(tmp_path)


def test_non_ascii_speaker_roundtrip_is_utf8(tmp_path: Path):
    rec = _rec(speaker="falante-ção")
    write_file_manifest([rec], tmp_path / "m.jsonl")
    assert "falante-ção" in (tmp_path / "m.jsonl").read_bytes().decode("utf-8")
    assert read_file_manifest(tmp_path / "m.jsonl") == [rec]


def test_lock_refuses_hash_change(tmp_path: Path):
    lock = tmp_path / "lock.json"
    update_lock("s", "a.tar.gz", SHA, 10, upstream_checksum="md5:00", lock_path=lock)
    update_lock("s", "a.tar.gz", SHA, 10, upstream_checksum="md5:00", lock_path=lock)
    assert lock_entry("s", "a.tar.gz", lock) == {"sha256": SHA, "size": 10, "upstream_checksum": "md5:00"}
    with pytest.raises(ManifestError, match="mudou"):
        update_lock("s", "a.tar.gz", "b" * 64, 10, lock_path=lock)
    assert json.loads(lock.read_text())["s"]["a.tar.gz"]["sha256"] == SHA


@pytest.mark.parametrize("split", ["test", "dev"])
def test_training_records_refuse_dev_and_test_splits(split):
    with pytest.raises(ManifestError, match="dev/teste"):
        training_records([_rec(split=split)], load_sources())
    assert training_records([_rec(split="train")], load_sources())


def test_pilot_is_not_trainable_unless_pilot_loader_opts_in():
    with pytest.raises(ManifestError, match="pilot"):
        training_records([_rec(split="pilot")], load_sources())
    assert training_records([_rec(split="pilot")], load_sources(), allow_pilot=True)


def test_splits_and_schema_constants():
    assert SPLITS == {"train", "dev", "test", "noise", "rir", "pilot"}
    assert TRAIN_SPLITS == ("noise", "rir", "train")
    assert MANIFEST_SCHEMA == 2 and _rec().schema == 2


_V1 = dict(path="raw/vctk/p227/p227_001_mic1.flac", sha256=SHA, source_id="vctk-0.92", license="CC-BY-4.0",
           role="train", speaker="p227", split="pilot", sr=48000, num_samples=48000, band_hz=20000.0, codec="flac")


def test_migrate_v1_keeps_pilot_split_and_takes_attribution_from_source():
    rec = migrate_v1(_V1)
    assert rec.split == "pilot" and rec.schema == 2
    assert rec.attribution == load_sources()["vctk-0.92"].attribution
    with pytest.raises(ManifestError, match="pilot"):
        training_records([rec], load_sources())
    with pytest.raises(ManifestError, match="fonte"):
        migrate_v1({**_V1, "source_id": "desconhecida"})
    with pytest.raises(ManifestError, match="chaves"):
        migrate_v1({**_V1, "extra": 1})


def test_read_file_manifest_reads_v1_and_v2_lines(tmp_path: Path):
    v2 = _rec(path="b.flac")
    f = tmp_path / "m.jsonl"
    f.write_text(json.dumps(_V1, sort_keys=True) + "\n" + json.dumps(v2.to_dict(), sort_keys=True) + "\n",
                 encoding="utf-8")
    back = read_file_manifest(f)
    assert [r.schema for r in back] == [2, 2] and back[0].attribution and back[1] == v2


def test_real_m0_pilot_manifests_migrate():
    from cctrain.data.manifest import FILES_DIR

    recs = read_file_manifest(FILES_DIR / "vctk-0.92-pilot.jsonl")
    assert len(recs) == 406 and {r.split for r in recs} == {"pilot", "test"}


@pytest.mark.parametrize("bad", [dict(license=""), dict(license="  "), dict(license=None), dict(attribution=""),
                                 dict(attribution=None)])
def test_record_without_license_or_attribution_is_refused(bad):
    with pytest.raises(ManifestError):
        _rec(**bad)


def test_v2_dict_without_attribution_is_refused():
    d = _rec().to_dict()
    del d["attribution"]
    with pytest.raises(ManifestError, match="chaves"):
        FileRecord.from_dict(d)
    with pytest.raises(ManifestError, match="schema"):
        _rec(schema=1)


@pytest.mark.parametrize("split", ["train", "dev", "pilot"])
def test_speech_split_requires_speaker(split):
    with pytest.raises(ManifestError, match="speaker"):
        _rec(speaker=None, split=split)
    assert _rec(speaker=None, split="noise").speaker_key is None
    assert _rec(speaker=None, split="rir").speaker_key is None


def test_speaker_key_is_namespaced_by_source():
    a = _rec(source_id="vctk-0.92", speaker="1")
    b = _rec(source_id="librispeech-dev-clean", speaker="1")
    assert a.speaker_key == "vctk-0.92:1" and b.speaker_key == "librispeech-dev-clean:1"
    assert a.speaker_key != b.speaker_key


@pytest.mark.parametrize("bad", [dict(sr=48000.0), dict(sr="48000"), dict(sr=True), dict(num_samples=1.5),
                                 dict(num_samples="10"), dict(num_samples=None), dict(sr=None)])
def test_sr_and_num_samples_validated_by_type(bad):
    with pytest.raises(ManifestError):
        _rec(**bad)


@pytest.mark.parametrize("sep", ["\u2028", "\u2029"])
def test_unicode_line_separators_in_jsonl(tmp_path: Path, sep):
    rec = _rec(speaker=f"a{sep}b")
    f = tmp_path / "m.jsonl"
    write_file_manifest([rec, _rec(path="z.flac")], f)
    raw = f.read_bytes().decode("utf-8")
    assert sep not in raw and len(raw.splitlines()) == 2
    assert read_file_manifest(f)[0] == rec
    # separador U+2028/U+2029 solto entre registros não é quebra de linha JSONL: erro controlado
    line = json.dumps(_rec().to_dict())
    f.write_text(line + sep + line + "\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        read_file_manifest(f)


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", ""])
def test_corrupt_lock_is_manifest_error(tmp_path: Path, content):
    lock = tmp_path / "lock.json"
    lock.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestError, match="lock"):
        lock_entry("s", "a", lock)
    with pytest.raises(ManifestError, match="lock"):
        update_lock("s", "a", SHA, 1, lock_path=lock)
