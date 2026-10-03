from collections import Counter

import pytest

from cctrain.data.manifest import (
    FILES_DIR,
    NON_TRAIN_SPLITS,
    ManifestError,
    load_sources,
    read_file_manifest,
    training_records,
)
from cctrain.data.pilot import PILOT_MANIFESTS, VCTK_TEST_FILES
from cctrain.hashing import sha256_file

pytestmark = pytest.mark.data


@pytest.mark.parametrize("source_id", sorted(PILOT_MANIFESTS))
def test_every_record_matches_its_file(paths, source_id):
    records = read_file_manifest(PILOT_MANIFESTS[source_id])
    assert records, source_id
    for r in records:
        assert sha256_file(paths.data / r.path) == r.sha256, r.path
    allowed = [r for r in records if r.split not in NON_TRAIN_SPLITS]
    assert training_records(allowed, load_sources()) == allowed
    for r in records:
        if r.split in NON_TRAIN_SPLITS:
            with pytest.raises(ManifestError, match="dev/teste"):
                training_records([r], load_sources())


def test_test_and_dev_material_is_labelled_as_such():
    vctk = read_file_manifest(PILOT_MANIFESTS["vctk-0.92"])
    assert {r.speaker for r in vctk if r.split == "test"} == {"p225", "p226"}
    assert not {r.speaker for r in vctk if r.split == "pilot"} & {"p225", "p226"}
    assert {r.split for r in read_file_manifest(PILOT_MANIFESTS["librispeech-dev-clean"])} == {"dev"}
    assert {r.split for r in read_file_manifest(PILOT_MANIFESTS["noise-wikimedia-cc0"])} == {"test"}


def test_pilot_amounts():
    vctk = read_file_manifest(PILOT_MANIFESTS["vctk-0.92"])
    pilot_s = sum(r.num_samples / r.sr for r in vctk if r.split == "pilot")
    assert pilot_s >= 15 * 60
    assert {r.path.rsplit("/", 1)[1] for r in vctk if r.split == "test"} == set(VCTK_TEST_FILES)
    libri = read_file_manifest(PILOT_MANIFESTS["librispeech-dev-clean"])
    assert len({r.speaker for r in libri}) >= 40
    assert sum(r.num_samples / r.sr for r in libri) >= 5 * 3600
    noise = read_file_manifest(PILOT_MANIFESTS["noise-wikimedia-cc0"])
    # ac_fan_cc0_10s (ogg) sai com 480_020 amostras mesmo no clipe do Clearcore (SHA-256 idêntico); ver sources.lock.json.
    assert len(noise) == 3
    assert all(480_000 <= r.num_samples <= 480_020 for r in noise)
    assert Counter(r.num_samples for r in noise) == Counter({480_000: 2, 480_020: 1})
    clearcore = {a["name"]: a["clearcore_sha256"] for a in load_sources()["noise-wikimedia-cc0"].artifacts}
    assert {r.path.rsplit("/", 1)[1].removesuffix(".wav"): r.sha256 for r in noise} == clearcore


def test_cv_private_manifest_matches_committed_hash_and_files(paths):
    digest_file = FILES_DIR / "cv27-pt.sha256"
    private = paths.data / "manifests" / "cv27-pt.jsonl"
    if not digest_file.is_file() or not private.is_file():
        pytest.skip("CV 27.0 não ingerido neste checkout")
    assert digest_file.read_text().split()[0] == sha256_file(private)
    records = read_file_manifest(private)
    assert records and all(r.source_id == "cv27-pt" and r.license == "CC0-1.0" for r in records)
    for r in records:
        assert sha256_file(paths.data / r.path) == r.sha256, r.path
    assert "client_id" not in digest_file.read_text()
