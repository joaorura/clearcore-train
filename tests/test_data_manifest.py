import json
from pathlib import Path

import pytest

from cctrain.data.licenses import LicenseRefused
from cctrain.data.manifest import (
    FileRecord,
    ManifestError,
    load_sources,
    lock_entry,
    read_file_manifest,
    training_records,
    update_lock,
    write_file_manifest,
)

SHA = "a" * 64


def _rec(**kw) -> FileRecord:
    base = dict(path="raw/vctk/p227/p227_001_mic1.flac", sha256=SHA, source_id="vctk-0.92", license="CC-BY-4.0",
                role="train", speaker="p227", split="pilot", sr=48000, num_samples=48000, band_hz=20000.0,
                codec="flac")
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
                                 dict(sr=0), dict(split="train"), dict(role="nope")])
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
    assert training_records([_rec(split="pilot")], load_sources())
