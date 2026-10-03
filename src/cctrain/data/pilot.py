"""Dados do piloto do M0 (spec §9.2): ~1 h para medir, nunca para treinar.

VCTK por HTTP Range (8 falantes x 50 falas mic1 + as 6 amostras de teste do Clearcore), LibriSpeech
dev-clean inteiro (stream do tar.gz; o loader usa um subconjunto), os 3 ruídos CC0 dos testes do Clearcore
(split test) e, se o dono tiver baixado, o Common Voice 27.0 pt (apenas a variante pt-BR; manifest por arquivo
com client_id fora do git).
"""

from __future__ import annotations

import csv
import hashlib
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

import numpy as np
import requests
import soundfile as sf

from cctrain.data.band import effective_band_hz
from cctrain.data.manifest import (
    FILES_DIR,
    FileRecord,
    ManifestError,
    SourceManifest,
    load_sources,
    update_lock,
    write_file_manifest,
)
from cctrain.data.sources import (
    IntegrityError,
    cut_clip,
    fetch_zip_members,
    list_zip_members,
    stream_download,
    stream_extract_tar,
)
from cctrain.hashing import sha256_file
from cctrain.paths import Paths

VCTK_PREFIX = "wav48_silence_trimmed"
VCTK_PILOT_SPEAKERS = ("p227", "p228", "p229", "p230", "p231", "p233", "p234", "p236")
VCTK_UTTERANCES_PER_SPEAKER = 50
# Amostras de teste do Clearcore (falantes p225/p226, fora do treino); SHA-256 de scripts/voice-samples.sha256.
VCTK_TEST_FILES = {
    "p225_003_mic1.flac": "877a8327608bd37b82d125adca98bdf7c92c904dff8c58fb711d7d9be364dc99",
    "p225_008_mic1.flac": "c4c4d3e1e128821c364db33fcec4b708e36edcea3bbc70f5334d6715aa4e00f9",
    "p225_011_mic1.flac": "8eaa96bb597b8ea88198b35da6f4a4e546125bc92c114e0aaee1cd87e7f08bbb",
    "p225_022_mic1.flac": "a4a88b254b17544e05fb259ded18c294b133abda8ff515ae9c72e842b64ec9d0",
    "p226_008_mic1.flac": "9570b6f075670d731ed0f0e0097db48988972b7a541e36da6455c9f1003ae3c6",
    "p226_016_mic1.flac": "4b3124fbf362cff6f20e557cf18d20f2975f8f76d42a2942612ce67921b408da",
}
# O zip do VCTK 0.92 não traz o texto da licença; ele é um bitstream irmão no mesmo registro do DataShare.
VCTK_LICENSE_URL = "https://datashare.ed.ac.uk/server/api/core/bitstreams/956a1688-0b59-428c-8a2f-10837433dde3/content"
VCTK_LICENSE_MD5 = "2946b37a07baeba80cea628909e28cae"
LIBRISPEECH_MD5_URL = "https://www.openslr.org/resources/12/md5sum.txt"
PILOT_MANIFESTS = {
    "vctk-0.92": FILES_DIR / "vctk-0.92-pilot.jsonl",
    "librispeech-dev-clean": FILES_DIR / "librispeech-dev-clean-pilot.jsonl",
    "noise-wikimedia-cc0": FILES_DIR / "noise-wikimedia-cc0-pilot.jsonl",
}

CV_SOURCE_ID = "cv27-pt"
# SHA-256 do tar.gz baixado pelo dono (Mozilla Data Collective), conferido durante a extração.
CV_ARCHIVE_SHA256 = "d5c6e70d669378ce8178b2a0f03e3281d67c4ee5ddd5200718221b28e0b93e60"
CV_VALIDATED_SUFFIX = "/pt/validated.tsv"
CV_VARIANT_PTBR = "Portuguese (Brasil)"  # valor da coluna `variant` do corpus
CV_CLIPS_PER_SPEAKER = 5


def check_license_text(text: str, what: str) -> None:
    if "Attribution 4.0" not in text:
        raise ManifestError(f"{what}: texto de licença não declara CC BY 4.0; pare e consulte o dono")


def select_vctk_members(names: Iterable[str], *, license_in_zip: bool = True) -> dict[str, str]:
    names = list(names)
    out: dict[str, str] = {}
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for n in names:
        parts = n.split("/")
        if len(parts) == 3 and parts[0] == VCTK_PREFIX and parts[2].endswith("_mic1.flac"):
            if parts[1] in VCTK_PILOT_SPEAKERS:
                by_speaker[parts[1]].append(n)
            elif parts[2] in VCTK_TEST_FILES:
                out[n] = f"raw/vctk-0.92/test/{parts[2]}"
    licenses = [n for n in names if "licen" in n.lower() and not n.endswith("/")]
    if license_in_zip and not licenses:
        raise ManifestError("VCTK: arquivo de licença ausente no zip")
    for n in licenses:
        out[n] = f"raw/vctk-0.92/{PurePosixPath(n).name}"
    for spk in VCTK_PILOT_SPEAKERS:
        members = sorted(by_speaker[spk])
        if len(members) < VCTK_UTTERANCES_PER_SPEAKER:
            raise ManifestError(f"VCTK: {spk} tem só {len(members)} falas mic1")
        for m in members[:VCTK_UTTERANCES_PER_SPEAKER]:
            out[m] = f"raw/vctk-0.92/pilot/{spk}/{PurePosixPath(m).name}"
    missing = set(VCTK_TEST_FILES) - {PurePosixPath(v).name for v in out.values()}
    if missing:
        raise ManifestError(f"VCTK: amostras de teste ausentes: {sorted(missing)}")
    return out


def librispeech_select(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if len(parts) < 2 or parts[0] != "LibriSpeech":
        return None
    rest = PurePosixPath(*parts[1:])
    if len(rest.parts) == 1 and rest.name.upper().endswith(".TXT"):
        return str(rest)
    if rest.parts[0] == "dev-clean" and (rest.name.endswith(".flac") or rest.name.endswith(".trans.txt")):
        return str(rest)
    return None


def record_for_audio(path: Path, data_root: Path, *, source: SourceManifest, speaker: str | None, split: str,
                     codec: str) -> FileRecord:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return FileRecord(
        path=str(Path(path).relative_to(data_root)), sha256=sha256_file(path), source_id=source.id,
        license=source.license, role=source.role.value, speaker=speaker, split=split, sr=int(sr),
        num_samples=int(mono.size), band_hz=effective_band_hz(mono, int(sr)), codec=codec)


def fetch_vctk(paths: Paths) -> list[FileRecord]:
    src = load_sources()["vctk-0.92"]
    names = list_zip_members(src.url)
    license_in_zip = any("licen" in n.lower() and not n.endswith("/") for n in names)
    if not license_in_zip:
        lic = paths.data / "raw" / "vctk-0.92" / "license_text.txt"
        d = stream_download(VCTK_LICENSE_URL, lic, expected_md5=VCTK_LICENSE_MD5, need_bytes=1 << 20)
        check_license_text(lic.read_text(errors="replace"), "VCTK license_text.txt (DataShare)")
        update_lock(src.id, "license_text.txt", d.sha256, d.size, upstream_checksum=f"md5:{VCTK_LICENSE_MD5}")
    selection = select_vctk_members(names, license_in_zip=license_in_zip)
    digests = fetch_zip_members(src.url, {m: paths.data / rel for m, rel in selection.items()})
    for member, rel in selection.items():
        name = PurePosixPath(rel).name
        if "licen" in name.lower():
            check_license_text((paths.data / rel).read_text(errors="replace"), f"VCTK {member}")
            update_lock(src.id, member, digests[member].sha256, digests[member].size)
        if name in VCTK_TEST_FILES and digests[member].sha256 != VCTK_TEST_FILES[name]:
            raise IntegrityError(f"{name}: SHA-256 difere do registrado pelo Clearcore")
    records = [
        record_for_audio(paths.data / rel, paths.data, source=src, speaker=PurePosixPath(rel).name.split("_")[0],
                         split="test" if "/test/" in rel else "pilot", codec="flac")
        for rel in selection.values() if rel.endswith(".flac")
    ]
    write_file_manifest(records, PILOT_MANIFESTS["vctk-0.92"])
    return records


def _librispeech_md5() -> str | None:
    try:
        r = requests.get(LIBRISPEECH_MD5_URL, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return None
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "dev-clean.tar.gz":
            return parts[0]
    return None


def fetch_librispeech(paths: Paths) -> list[FileRecord]:
    src = load_sources()["librispeech-dev-clean"]
    md5 = _librispeech_md5()
    out_dir = paths.data / "raw" / "librispeech-dev-clean"
    digest, files = stream_extract_tar(src.url, out_dir, librispeech_select, expected_md5=md5,
                                       need_bytes=600 * 1024**2)
    update_lock(src.id, "dev-clean.tar.gz", digest.sha256, digest.size,
                upstream_checksum=f"md5:{md5}" if md5 else None)
    lic = next((f for f in files if "/" not in f and "LICENSE" in f.upper()), None)
    if lic is None:
        raise ManifestError("LibriSpeech: LICENSE.TXT ausente no tar")
    check_license_text((out_dir / lic).read_text(errors="replace"), "LibriSpeech LICENSE.TXT")
    records = [record_for_audio(out_dir / f, paths.data, source=src, speaker=PurePosixPath(f).parts[1],
                                split="pilot", codec="flac") for f in files if f.endswith(".flac")]
    write_file_manifest(records, PILOT_MANIFESTS["librispeech-dev-clean"])
    return records


def fetch_noise(paths: Paths) -> list[FileRecord]:
    src = load_sources()["noise-wikimedia-cc0"]
    out_dir = paths.data / "raw" / "noise-wikimedia-cc0"
    records = []
    for art in src.artifacts:
        raw = out_dir / "_raw" / f"{art['name']}.{art['raw_ext']}"
        headers = {"Range": f"bytes=0-{art['range_bytes'] - 1}"} if art["range_bytes"] else None
        d = stream_download(art["url"], raw, headers=headers, need_bytes=art["range_bytes"] or 64 * 1024**2)
        clip = out_dir / f"{art['name']}.wav"
        cut_clip(raw, clip, start_s=float(art["start_s"]))
        raw.unlink()  # bruto transitório criado por esta função
        matches = sha256_file(clip) == art["clearcore_sha256"]
        update_lock(src.id, raw.name, d.sha256, d.size,
                    upstream_checksum="clearcore-clip:" + ("igual" if matches else "difere (versão do ffmpeg)"))
        records.append(record_for_audio(clip, paths.data, source=src, speaker=None, split="test", codec="wav"))
    write_file_manifest(records, PILOT_MANIFESTS["noise-wikimedia-cc0"])
    return records


def cv_pilot_selection(rows: Iterable[Mapping[str, str]], per_speaker: int = CV_CLIPS_PER_SPEAKER) -> list[dict]:
    """Subconjunto determinístico do piloto: só variante pt-BR, até `per_speaker` clipes por client_id."""
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("variant") == CV_VARIANT_PTBR:
            by_speaker[row["client_id"]].append(dict(row))
    if not by_speaker:
        raise ManifestError(f"CV: nenhuma linha com variant == {CV_VARIANT_PTBR!r} (filtro pt-BR)")
    out: list[dict] = []
    for spk in sorted(by_speaker):
        ordered = sorted(by_speaker[spk], key=lambda r: hashlib.sha256(r["path"].encode()).hexdigest())
        out += ordered[:per_speaker]
    return sorted(out, key=lambda r: r["path"])


def _cv_read_validated(archive: Path) -> list[dict]:
    """Lê validated.tsv em stream; os metadados vêm antes dos clipes no tar, então a leitura para cedo."""
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(CV_VALIDATED_SUFFIX):
                f = tar.extractfile(member)
                assert f is not None
                text = f.read().decode("utf-8").splitlines()
                reader = csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
                needed = {"client_id", "path", "variant"}
                if not reader.fieldnames or not needed <= set(reader.fieldnames):
                    raise ManifestError(f"CV: validated.tsv sem as colunas {sorted(needed)}")
                return list(reader)
    raise ManifestError("CV: validated.tsv ausente no tar")


def ingest_cv(paths: Paths, archive: Path, per_speaker: int = CV_CLIPS_PER_SPEAKER) -> list[FileRecord]:
    src = load_sources()[CV_SOURCE_ID]
    archive = Path(archive)
    if not archive.is_file():
        raise ManifestError(f"{archive} não existe (download manual pelo dono, após aceitar os termos)")
    selection = cv_pilot_selection(_cv_read_validated(archive), per_speaker)
    wanted = {PurePosixPath(r["path"]).name: r["client_id"] for r in selection}
    out_dir = paths.data / "raw" / CV_SOURCE_ID

    def select(name: str) -> str | None:
        p = PurePosixPath(name)
        return f"clips/{p.name}" if p.parent.name == "clips" and p.name in wanted else None

    digest, files = stream_extract_tar(archive, out_dir, select, expected_sha256=CV_ARCHIVE_SHA256,
                                       need_bytes=2 * 1024**3)
    update_lock(src.id, archive.name, digest.sha256, digest.size)
    if len(files) != len(wanted):
        raise ManifestError(f"CV: {len(files)} clipes extraídos de {len(wanted)} selecionados")
    records = [record_for_audio(out_dir / f, paths.data, source=src, speaker=wanted[PurePosixPath(f).name],
                                split="pilot", codec="mp3") for f in files]
    private = paths.data / "manifests" / f"{CV_SOURCE_ID}.jsonl"
    digest_manifest = write_file_manifest(records, private)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    (FILES_DIR / f"{CV_SOURCE_ID}.sha256").write_text(f"{digest_manifest}  data/manifests/{CV_SOURCE_ID}.jsonl\n")
    return records


def duration_hours(records: Iterable[FileRecord]) -> float:
    return float(np.sum([r.num_samples / r.sr for r in records]) / 3600.0)
