"""Ingestão em FLAC nativo dos dados do M1, fase a fase (plano M1, Tarefa 8; fases em configs/data/m1-phases-v1.yaml).

Tudo é idempotente e retomável: um download interrompido continua do offset por `Range` (entre processos também), uma
fonte cujo manifest v2 já existe é pulada, e brutos transitórios só são apagados depois do manifest gravado. Nenhum
dado vai para /tmp; o disco é conferido por fase (`phase_deficit_gb`) antes de baixar qualquer byte.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import requests
import soundfile as sf
import yaml

from cctrain.data.band import effective_band_hz
from cctrain.data.manifest import (
    FILES_DIR,
    FileRecord,
    ManifestError,
    SourceManifest,
    load_sources,
    read_file_manifest,
    update_lock,
    write_file_manifest,
)
from cctrain.data.sources import (
    CHUNK,
    USER_AGENT,
    IntegrityError,
    StreamDigest,
    _chunks,
    _redact,
    _verify,
    list_zip_members,
    stream_extract_tar,
)
from cctrain.disk import GB, DiskError, require_free
from cctrain.hashing import sha256_file
from cctrain.m1.disk_plan import PHASES_V1, Phase, load_phases_config, plan_phases
from cctrain.paths import Paths

LIBRISPEECH_SPEAKER_CAP_S = 240.0
SPEAKER_CAP_S = LIBRISPEECH_SPEAKER_CAP_S

LIBRISPEECH_MD5_URL = "https://www.openslr.org/resources/12/md5sum.txt"
DNS_BASE = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/noise_fullband"
DNS_FREESOUND_PARTS = ("datasets_fullband.noise_fullband.freesound_000.tar.bz2",)
DNS_FREESOUND_MARK = "_Freesound_validated_"
DNS_HOURS_CAP = 10.0
FSD_BASE = "https://zenodo.org/api/records/4060432/files"
FSD_HOURS_CAP = 15.0
CC0_URL = "http://creativecommons.org/publicdomain/zero/1.0/"
CC_BY_URLS = {"http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
              "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0"}
# Classes de voz humana do ontology do FSD50K (a lista é conservadora: na dúvida o clipe sai).
FSD_VOICE_LABELS = frozenset({
    "Human_voice", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking",
    "Conversation", "Speech", "Narration_and_monologue", "Speech_synthesizer", "Singing", "Male_singing",
    "Female_singing", "Child_singing", "Choir", "Yodeling", "Chant", "Humming", "Whispering", "Shout", "Yell",
    "Screaming", "Bellow", "Whoop", "Laughter", "Giggle", "Chuckle_and_chortle", "Snicker", "Belly_laugh",
    "Crying_and_sobbing", "Baby_cry_and_infant_cry", "Whimper", "Groan", "Gasp", "Sigh", "Gargling",
    "Applause", "Cheering", "Crowd", "Human_group_actions", "Rapping", "Burping_and_eructation", "Hiccup",
    "Cough", "Sneeze", "Throat_clearing", "Breathing", "Snoring", "Gurgling", "Chatter", "Bark",
})


class PhaseBlocked(RuntimeError):
    """Fase `blocked_disk` ou sem espaço: não inicia; carrega o déficit em GB."""

    def __init__(self, phase: str, deficit_gb: float):
        super().__init__(f"fase {phase} blocked_disk: faltam {deficit_gb:.2f} GB (liberar disco é decisão do dono)")
        self.phase = phase
        self.deficit_gb = deficit_gb


# ---------------------------------------------------------------------------------------------------------------------
# seleção e regras puras
# ---------------------------------------------------------------------------------------------------------------------

def dns_freesound_cc0(meta_rows: Iterable[str | Mapping[str, str]]) -> set[str]:
    """Caminhos do DNS cuja origem Freesound (CC0 por seleção do DNS) é determinável pelo nome (`*_Freesound_validated_<id>*`).

    O README do DNS-Challenge declara que do Freesound só entraram arquivos CC0; AudioSet e DEMAND (outras licenças) não
    carregam essa marca e ficam de fora. Sem origem determinável, o arquivo é excluído.
    """
    out: set[str] = set()
    for row in meta_rows:
        name = row if isinstance(row, str) else row.get("path") or row.get("name") or ""
        base = name.rsplit("/", 1)[-1]
        if DNS_FREESOUND_MARK in base and not base.lower().startswith(("audioset", "demand")):
            out.add(name)
    return out


def fsd50k_allowed(clip_meta: Mapping) -> bool:
    """CC0 ou CC BY (3.0/4.0) e nenhum rótulo de voz humana; NC, Sampling+ e sem licença ficam fora."""
    lic = clip_meta.get("license")
    if lic != CC0_URL and lic not in CC_BY_URLS:
        return False
    labels = clip_meta.get("labels") or ()
    if isinstance(labels, str):
        labels = labels.split(",")
    return not (set(labels) & FSD_VOICE_LABELS)


def fsd50k_license_id(url: str) -> str:
    return "CC0-1.0" if url == CC0_URL else CC_BY_URLS[url]


def librispeech_cap(records: list[FileRecord], max_s: float = SPEAKER_CAP_S) -> list[FileRecord]:
    """Mantém, por falante, os primeiros enunciados (ordem de `path`) cuja soma não passa de `max_s` segundos."""
    kept: list[FileRecord] = []
    used: dict[str, float] = defaultdict(float)
    for r in sorted(records, key=lambda r: r.path):
        key = r.speaker or ""
        dur = r.num_samples / r.sr
        if used[key] + dur <= max_s:
            used[key] += dur
            kept.append(r)
    return kept


def librispeech_speaker_sex(speakers_txt: str) -> dict[str, str]:
    """`SPEAKERS.TXT` do LibriSpeech (`ID | SEX | SUBSET | MINUTES | NAME`) -> {id: 'F'|'M'}."""
    out: dict[str, str] = {}
    for line in speakers_txt.splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("F", "M"):
            out[parts[0]] = parts[1]
    return out


def vctk_mic1_members(names: Iterable[str]) -> dict[str, str]:
    """{membro do zip: caminho relativo em raw/vctk-0.92/full}, só mic1 de falantes `p###`."""
    out: dict[str, str] = {}
    for n in names:
        parts = n.split("/")
        if len(parts) == 3 and parts[0] == "wav48_silence_trimmed" and parts[2].endswith("_mic1.flac") \
                and re.fullmatch(r"p\d+", parts[1]):
            out[n] = f"raw/vctk-0.92/full/{parts[1]}/{parts[2]}"
    return out


# ---------------------------------------------------------------------------------------------------------------------
# download retomável entre processos
# ---------------------------------------------------------------------------------------------------------------------

def _head(url: str, session: requests.Session) -> dict:
    r = session.get(url, stream=True, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"}, timeout=60)
    try:
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} para {_redact(url)}")
        cr = re.match(r"bytes \d+-\d+/(\d+)", r.headers.get("Content-Range", ""))
        total = int(cr.group(1)) if cr else int(r.headers.get("Content-Length", "0") or 0)
        return {"size": total, "etag": r.headers.get("ETag", ""), "last_modified": r.headers.get("Last-Modified", ""),
                "ranges": r.status_code == 206}
    finally:
        r.close()


BLOCK = 64 << 20


def download_resumable(url: str, dest: Path, *, expected_size: int | None = None, expected_md5: str | None = None,
                       expected_sha256: str | None = None, session: requests.Session | None = None,
                       log=print, connections: int = 8) -> StreamDigest:
    """Baixa `url` para `dest` em blocos de 64 MB por `connections` conexões `Range` (os servidores limitam ~1 MB/s por
    conexão), com retomada entre processos: `dest.part` pré-alocado + `dest.part.meta` com o validador (ETag/Last-Modified),
    o tamanho e os blocos concluídos. Bloco com falha repete (retry/Range por `_chunks`, Tarefa 5). Se o recurso mudou,
    recomeça. Hash e tamanho finais são conferidos contra os valores conhecidos; nada além do `.part` é apagado.
    """
    from concurrent.futures import ThreadPoolExecutor

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()
    part = dest.with_name(dest.name + ".part")
    meta_path: Path | None = dest.with_name(dest.name + ".part.meta")
    if not dest.is_file():
        head = _head(url, s)
        size = expected_size or head["size"]
        if expected_size and head["size"] and head["size"] != expected_size:
            raise IntegrityError(f"{_redact(url)}: tamanho {head['size']} != esperado {expected_size}")
        if not head["ranges"]:
            raise IntegrityError(f"{_redact(url)}: servidor sem Range; sem retomada segura")
        validator = head["etag"] or head["last_modified"]
        nblocks = -(-size // BLOCK)
        done: set[int] = set()
        if part.is_file() and meta_path.is_file():
            old = json.loads(meta_path.read_text())
            if old.get("validator") == validator and old.get("size") == size and old.get("block") == BLOCK:
                done = set(old["done"])
        if not done or not part.is_file() or part.stat().st_size != size:
            done = set()
            require_free(dest.parent, size)
            with open(part, "wb") as f:
                f.truncate(size)
        lock = threading.Lock()
        state = {"bytes": len(done) * BLOCK, "t0": time.time(), "last": time.time()}
        base_bytes = state["bytes"]

        def save() -> None:
            tmp = meta_path.with_name(meta_path.name + ".tmp")
            tmp.write_text(json.dumps({"validator": validator, "size": size, "block": BLOCK, "done": sorted(done)}))
            os.replace(tmp, meta_path)

        save()
        log(f"download {dest.name}: {size / GB:.2f} GB, {len(done)}/{nblocks} blocos prontos, {connections} conexões")
        fd = os.open(part, os.O_WRONLY)

        def fetch_block(i: int) -> None:
            a, b = i * BLOCK, min((i + 1) * BLOCK, size) - 1
            pos = a
            for chunk in _chunks(url, headers={"Range": f"bytes={a}-{b}"}, session=requests.Session()):
                os.pwrite(fd, chunk, pos)
                pos += len(chunk)
            if pos != b + 1:
                raise IntegrityError(f"{dest.name}: bloco {i} veio com {pos - a} bytes de {b + 1 - a}")
            with lock:
                done.add(i)
                state["bytes"] += b + 1 - a
                save()
                if time.time() - state["last"] > 60:
                    state["last"] = time.time()
                    rate = (state["bytes"] - base_bytes) / (state["last"] - state["t0"]) / 1e6
                    log(f"  {dest.name}: {state['bytes'] / GB:.2f}/{size / GB:.2f} GB ({rate:.1f} MB/s)")

        try:
            with ThreadPoolExecutor(max_workers=connections) as ex:
                list(ex.map(fetch_block, [i for i in range(nblocks) if i not in done]))
        finally:
            os.close(fd)
        target = part
    else:
        target = dest
        meta_path = None
    sha, md5 = hashlib.sha256(), hashlib.md5()
    total = 0
    with open(target, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            sha.update(chunk)
            md5.update(chunk)
            total += len(chunk)
    d = StreamDigest(sha.hexdigest(), md5.hexdigest(), total)
    try:
        _verify(d, dest.name, expected_sha256, expected_md5, expected_size)
    except IntegrityError:
        if meta_path is not None:
            part.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        raise
    if meta_path is not None:
        os.replace(part, dest)
        meta_path.unlink(missing_ok=True)
    return d


# ---------------------------------------------------------------------------------------------------------------------
# fatos de áudio em paralelo (sha256, sr, amostras, banda)
# ---------------------------------------------------------------------------------------------------------------------

def _audio_facts(path: str) -> tuple[str, int, int, float | None]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return sha256_file(path), int(sr), int(mono.size), effective_band_hz(mono, int(sr))


def build_records(paths_rel: list[str], data_root: Path, *, source: SourceManifest, split: str, codec: str,
                  speaker_of=lambda rel: None, license_of=None, attribution_of=None,
                  workers: int | None = None) -> list[FileRecord]:
    """Um `FileRecord` v2 por arquivo, com licença/atribuição por arquivo (`license_of`/`attribution_of` opcionais)."""
    abs_paths = [str(data_root / r) for r in paths_rel]
    workers = workers or int(os.environ.get("CCTRAIN_INGEST_WORKERS", "4"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        facts = list(pool.map(_audio_facts, abs_paths, chunksize=16))
    out = []
    for rel, (sha, sr, n, band) in zip(paths_rel, facts, strict=True):
        out.append(FileRecord(
            path=rel, sha256=sha, source_id=source.id, license=license_of(rel) if license_of else source.license,
            role=source.role.value, speaker=speaker_of(rel), split=split, sr=sr, num_samples=n, band_hz=band,
            codec=codec, attribution=attribution_of(rel) if attribution_of else source.attribution))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# fase 1: downloads brutos (em paralelo, hosts diferentes)
# ---------------------------------------------------------------------------------------------------------------------

PHASE1_DOWNLOADS = {
    "dns-freesound-000": (f"{DNS_BASE}/{DNS_FREESOUND_PARTS[0]}", DNS_FREESOUND_PARTS[0], 3470032239, None),
    "fsd50k-eval-z01": (f"{FSD_BASE}/FSD50K.eval_audio.z01/content", "FSD50K.eval_audio.z01", 3221225472,
                        "3090670eaeecc013ca1ff84fe4442aeb"),
    "fsd50k-eval-zip": (f"{FSD_BASE}/FSD50K.eval_audio.zip/content", "FSD50K.eval_audio.zip", 3037675767,
                        "6fa47636c3a3ad5c7dfeba99f2637982"),
    "fsd50k-metadata": (f"{FSD_BASE}/FSD50K.metadata.zip/content", "FSD50K.metadata.zip", 6700838,
                        "b9ea0c829a411c1d42adb9da539ed237"),
    "fsd50k-ground-truth": (f"{FSD_BASE}/FSD50K.ground_truth.zip/content", "FSD50K.ground_truth.zip", 334701,
                            "ca27382c195e37d2269c4c866dd73485"),
}


def prefetch_phase1(data_root: Path, log=print) -> dict[str, StreamDigest]:
    """Baixa (retomável) os brutos da fase 1: ordem dos mais pesados primeiro, DNS (Azure) e FSD50K (Zenodo) em paralelo."""
    from concurrent.futures import ThreadPoolExecutor

    dl = data_root / "raw" / "_dl"

    def one(item):
        key, (url, name, size, md5) = item
        return key, download_resumable(url, dl / name, expected_size=size, expected_md5=md5, log=log)

    with ThreadPoolExecutor(max_workers=3) as ex:
        return dict(ex.map(one, PHASE1_DOWNLOADS.items()))


# ---------------------------------------------------------------------------------------------------------------------
# utilidades de arquivo
# ---------------------------------------------------------------------------------------------------------------------

def _src(source_id: str) -> SourceManifest:
    return load_sources()[source_id]


def _manifest_file(source_id: str, name: str | None = None) -> Path:
    return FILES_DIR / f"{name or source_id}.jsonl"


def _cached(source_id: str, name: str | None = None) -> list[FileRecord] | None:
    """Manifest v2 já gravado e com todos os arquivos presentes: a fonte está pronta (rerun idempotente)."""
    path = _manifest_file(source_id, name)
    if not path.is_file():
        return None
    return read_file_manifest(path)


def _write_flac(audio_bytes_or_path, dest: Path) -> float:
    """WAV -> FLAC PCM_16 na taxa nativa (grava em `.part`, renomeia). Devolve a duração em s."""
    data, sr = sf.read(audio_bytes_or_path, dtype="int16", always_2d=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        sf.write(str(part), data, sr, format="FLAC", subtype="PCM_16")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return len(data) / sr


def _wav_to_flac_job(args: tuple[str, str]) -> float:
    src, dst = args
    dur = _write_flac(src, Path(dst))
    Path(src).unlink()
    return dur


def _check_license_present(text: str, what: str, needle: str = "Attribution 4.0") -> None:
    if needle not in text:
        raise ManifestError(f"{what}: texto de licença não declara {needle}; pare e consulte o dono")


def _seconds(rec: FileRecord) -> float:
    return rec.num_samples / rec.sr


# ---------------------------------------------------------------------------------------------------------------------
# fase 1: ruído DNS (Freesound CC0) e FSD50K eval
# ---------------------------------------------------------------------------------------------------------------------

def ingest_dns(paths: Paths, log=print) -> list[FileRecord]:
    src_id = "dns-noise-fullband-cc0"
    if (cached := _cached(src_id)) is not None:
        return cached
    src = _src(src_id)
    url, name, size, _ = PHASE1_DOWNLOADS["dns-freesound-000"]
    arc = paths.data / "raw" / "_dl" / name
    d = download_resumable(url, arc, expected_size=size, log=log)
    update_lock(src_id, name, d.sha256, d.size)
    hours, rels = 0.0, []
    import tarfile

    with tarfile.open(arc, mode="r|bz2") as tar:
        for member in tar:
            if not member.isfile() or not dns_freesound_cc0([member.name]):
                continue
            rel = f"raw/{src_id}/{PurePosixPath(member.name).stem}.flac"
            dest = paths.data / rel
            if dest.is_file():
                hours += sf.info(str(dest)).duration / 3600
            else:
                hours += _write_flac(io.BytesIO(tar.extractfile(member).read()), dest) / 3600
            rels.append(rel)
            if hours >= DNS_HOURS_CAP:
                break
    if not rels:
        raise ManifestError("DNS: nenhum arquivo com origem Freesound determinável")
    records = build_records(rels, paths.data, source=src, split="noise", codec="flac")
    write_file_manifest(records, _manifest_file(src_id))
    arc.unlink()  # bruto transitório baixado por esta função
    log(f"{src_id}: {len(records)} arquivos, {hours:.2f} h")
    return records


def _fsd_members(names: Iterable[str]) -> dict[str, str]:
    """{id do clipe: nome do membro no zip}."""
    return {PurePosixPath(n).stem: n for n in names if n.endswith(".wav")}


def _7z_list(archive: Path) -> list[str]:
    out = subprocess.run(["7z", "l", "-slt", str(archive)], check=True, capture_output=True, text=True).stdout
    return [line[7:] for line in out.splitlines() if line.startswith("Path = ") and line.endswith(".wav")]


def ingest_fsd50k(paths: Paths, log=print) -> dict[str, list[FileRecord]]:
    ids = ("fsd50k-eval-cc0", "fsd50k-eval-ccby")
    cached = {i: _cached(i) for i in ids}
    if all(v is not None for v in cached.values()):
        return cached  # type: ignore[return-value]
    dl = paths.data / "raw" / "_dl"
    digests = {}
    for key in ("fsd50k-eval-z01", "fsd50k-eval-zip", "fsd50k-metadata", "fsd50k-ground-truth"):
        url, name, size, md5 = PHASE1_DOWNLOADS[key]
        digests[name] = download_resumable(url, dl / name, expected_size=size, expected_md5=md5, log=log)
    for name, d in digests.items():
        update_lock("fsd50k-eval-cc0", name, d.sha256, d.size, upstream_checksum=f"md5:{d.md5}")
    meta_dir = paths.data / "raw" / "fsd50k-eval" / "_meta"
    with zipfile.ZipFile(dl / "FSD50K.metadata.zip") as z:
        info = json.loads(z.read("FSD50K.metadata/eval_clips_info_FSD50K.json"))
    with zipfile.ZipFile(dl / "FSD50K.ground_truth.zip") as z:
        gt = list(csv.DictReader(z.read("FSD50K.ground_truth/eval.csv").decode().splitlines()))
    del meta_dir
    allowed = []
    for row in gt:
        meta = info[row["fname"]]
        if fsd50k_allowed({"license": meta["license"], "labels": row["labels"]}):
            allowed.append(row["fname"])
    allowed.sort(key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    members = _fsd_members(_7z_list(dl / "FSD50K.eval_audio.zip"))
    missing = [i for i in allowed if i not in members]
    if missing:
        raise IntegrityError(f"FSD50K: {len(missing)} clipes permitidos sem membro no zip (ex.: {missing[:3]})")
    staging = paths.data / "raw" / "fsd50k-eval" / "_staging"
    done_rels: dict[str, list[str]] = {i: [] for i in ids}
    meta_of: dict[str, dict] = {}
    hours = 0.0
    for start in range(0, len(allowed), 300):
        if hours >= FSD_HOURS_CAP:
            break
        batch = allowed[start:start + 300]
        todo = []
        for cid in batch:
            lic = fsd50k_license_id(info[cid]["license"])
            src_id = ids[0] if lic == "CC0-1.0" else ids[1]
            rel = f"raw/{src_id}/{cid}.flac"
            meta_of[rel] = info[cid] | {"id": cid, "license_id": lic}
            if (paths.data / rel).is_file():
                hours += sf.info(str(paths.data / rel)).duration / 3600
                done_rels[src_id].append(rel)
            else:
                todo.append((cid, src_id, rel))
        if todo:
            staging.mkdir(parents=True, exist_ok=True)
            subprocess.run(["7z", "x", "-y", f"-o{staging}", str(dl / "FSD50K.eval_audio.zip"),
                            *[members[c] for c, _, _ in todo]], check=True, capture_output=True)
            for cid, src_id, rel in todo:
                wav = staging / members[cid]
                hours += _write_flac(str(wav), paths.data / rel) / 3600
                wav.unlink()
                done_rels[src_id].append(rel)
        if hours >= FSD_HOURS_CAP:
            break
    shutil.rmtree(staging, ignore_errors=True)
    out: dict[str, list[FileRecord]] = {}
    for src_id, rels in done_rels.items():
        src = _src(src_id)

        def attribution(rel, _m=meta_of):
            m = _m[rel]
            return (f"Freesound user {m['uploader']}: \"{m['title']}\" (Freesound clip {m['id']}), "
                    f"{m['license']} ({m['license_id']}); via FSD50K, Fonseca et al., doi:10.5281/zenodo.4060432")

        recs = build_records(sorted(rels), paths.data, source=src, split="noise", codec="flac",
                             attribution_of=attribution)
        write_file_manifest(recs, _manifest_file(src_id))
        out[src_id] = recs
    for name in ("FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip"):
        (dl / name).unlink(missing_ok=True)
    log(f"fsd50k-eval: {sum(len(v) for v in out.values())} clipes, {hours:.2f} h")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# fase 2: VCTK mic1 completo (zip por Range, conexões paralelas)
# ---------------------------------------------------------------------------------------------------------------------

def _vctk_speaker(rel: str) -> str:
    return PurePosixPath(rel).name.split("_")[0]


def ingest_vctk(paths: Paths, log=print, workers: int = 48) -> list[FileRecord]:
    from remotezip import RemoteZip

    src_id = "vctk-0.92"
    name = "vctk-0.92-mic1"
    if (cached := _cached(src_id, name)) is not None:
        return cached
    src = _src(src_id)
    url = src.url
    sizes = list_zip_members(url)
    selection = vctk_mic1_members(sizes)
    if len(selection) < 40_000:
        raise ManifestError(f"VCTK: só {len(selection)} membros mic1 no zip")
    local = threading.local()
    count = {"done": 0, "t": time.time()}
    lock = threading.Lock()

    def fetch(item: tuple[str, str]) -> None:
        member, rel = item
        dest = paths.data / rel
        if dest.is_file() and dest.stat().st_size == sizes[member]:
            return
        for attempt in range(5):
            try:
                if getattr(local, "zip", None) is None:
                    local.zip = RemoteZip(url, headers={"User-Agent": USER_AGENT})
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_name(dest.name + ".part")
                with local.zip.open(member) as zf, open(part, "wb") as f:
                    shutil.copyfileobj(zf, f, CHUNK)
                if part.stat().st_size != sizes[member]:
                    raise IntegrityError(f"{member}: tamanho {part.stat().st_size} != {sizes[member]}")
                os.replace(part, dest)
                break
            except (requests.RequestException, OSError, zipfile.BadZipFile, IntegrityError):
                local.zip = None
                dest.with_name(dest.name + ".part").unlink(missing_ok=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        with lock:
            count["done"] += 1
            if time.time() - count["t"] > 120:
                count["t"] = time.time()
                log(f"  vctk: {count['done']}/{len(selection)} arquivos")

    require_free(paths.data, int(8.0 * GB))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fetch, selection.items()))
    rels = sorted(selection.values())
    records = build_records(rels, paths.data, source=src, split="train", codec="flac", speaker_of=_vctk_speaker)
    write_file_manifest(records, _manifest_file(src_id, name))
    log(f"vctk-0.92-mic1: {len(records)} arquivos, {sum(map(_seconds, records)) / 3600:.2f} h")
    return records


# ---------------------------------------------------------------------------------------------------------------------
# fase 3: LibriSpeech train-clean-100 (teto 240 s/falante) e test-clean
# ---------------------------------------------------------------------------------------------------------------------

def _librispeech_md5(subset: str) -> str | None:
    try:
        r = requests.get(LIBRISPEECH_MD5_URL, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return None
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == f"{subset}.tar.gz":
            return parts[0]
    return None


def librispeech_subset_select(subset: str):
    def select(name: str) -> str | None:
        parts = PurePosixPath(name).parts
        if len(parts) < 2 or parts[0] != "LibriSpeech":
            return None
        rest = PurePosixPath(*parts[1:])
        if len(rest.parts) == 1 and rest.name.upper().endswith(".TXT"):
            return str(rest)
        if rest.parts[0] == subset and (rest.name.endswith(".flac") or rest.name.endswith(".trans.txt")):
            return str(rest)
        return None
    return select


def ingest_librispeech(paths: Paths, subset: str, log=print) -> list[FileRecord]:
    src_id = f"librispeech-{subset}"
    if (cached := _cached(src_id)) is not None:
        return cached
    src = _src(src_id)
    out_dir = paths.data / "raw" / src_id
    md5 = _librispeech_md5(subset)
    if not out_dir.exists():
        digest, _ = stream_extract_tar(src.url, out_dir, librispeech_subset_select(subset), expected_md5=md5,
                                       need_bytes=int(7.5 * GB), done=lambda rel: True)
        update_lock(src_id, f"{subset}.tar.gz", digest.sha256, digest.size,
                    upstream_checksum=f"md5:{md5}" if md5 else None)
    lic = out_dir / "LICENSE.TXT"
    if not lic.is_file():
        raise ManifestError(f"{src_id}: LICENSE.TXT ausente no tar")
    _check_license_present(lic.read_text(errors="replace"), f"{src_id} LICENSE.TXT")
    flacs = sorted(p.relative_to(paths.data).as_posix() for p in out_dir.rglob("*.flac"))
    split = "test" if subset == "test-clean" else "train"
    speaker_of = lambda rel: PurePosixPath(rel).parts[3]  # noqa: E731  raw/<src>/<subset>/<spk>/<chap>/f.flac
    records = build_records(flacs, paths.data, source=src, split=split, codec="flac", speaker_of=speaker_of)
    if subset == "train-clean-100":
        kept = librispeech_cap(records, LIBRISPEECH_SPEAKER_CAP_S)
        keep = {r.path for r in kept}
        for r in records:
            if r.path not in keep:
                (paths.data / r.path).unlink()  # extraído por esta função; passou do teto de 240 s/falante
        records = kept
    write_file_manifest(records, _manifest_file(src_id))
    log(f"{src_id}: {len(records)} arquivos, {len({r.speaker for r in records})} falantes, "
        f"{sum(map(_seconds, records)) / 3600:.2f} h")
    return records


# ---------------------------------------------------------------------------------------------------------------------
# fase 4: CML-TTS pt (teto 30 h, 40 min por falante)
# ---------------------------------------------------------------------------------------------------------------------

CML_PREFIX = "cml_tts_dataset_portuguese_v0.1"
CML_MD5 = "743bad054ca861688aa026e505b26aff"
CML_HOURS_CAP = 30.0
CML_SPEAKER_CAP_S = 2400.0


def cml_select_factory(hours_cap: float = CML_HOURS_CAP, speaker_cap_s: float = CML_SPEAKER_CAP_S):
    """`select` para `stream_extract_tar`: WAV 24 kHz pcm16 (duração = bytes/48000), teto por falante e global.

    Determinístico (ordem do tar); também guarda os metadados de texto (transcripts, licença) de nível baixo.
    """
    used: dict[str, float] = defaultdict(float)
    total = [0.0]

    def select_sized(name: str, size: int) -> str | None:
        parts = PurePosixPath(name).parts
        if len(parts) < 2 or parts[0] != CML_PREFIX:
            return None
        if len(parts) != 6 or parts[2] != "audio" or not parts[5].endswith(".wav"):
            return "/".join(parts[1:]) if parts[-1].endswith((".txt", ".md", ".csv")) else None
        dur = max(size - 44, 0) / 48000.0
        speaker = parts[3]
        if total[0] + dur > hours_cap * 3600 or used[speaker] + dur > speaker_cap_s:
            return None
        used[speaker] += dur
        total[0] += dur
        return "/".join(parts[1:])

    select_sized.wants_size = True  # type: ignore[attr-defined]
    return select_sized


def ingest_cml(paths: Paths, log=print) -> list[FileRecord]:
    src_id = "cml-tts-pt"
    if (cached := _cached(src_id)) is not None:
        return cached
    src = _src(src_id)
    out_dir = paths.data / "raw" / src_id
    if not out_dir.exists():
        digest, _ = stream_extract_tar(src.url, out_dir, cml_select_factory(), expected_md5=CML_MD5,
                                             need_bytes=int(9 * GB), done=lambda rel: True)
        update_lock(src_id, "cml_tts_dataset_portuguese_v0.1.tar.bz", digest.sha256, digest.size,
                    upstream_checksum=f"md5:{CML_MD5}")
    wavs = sorted(p for p in out_dir.rglob("*.wav"))
    if wavs:
        jobs = [(str(p), str(p.with_suffix(".flac"))) for p in wavs]
        with ProcessPoolExecutor(max_workers=int(os.environ.get("CCTRAIN_INGEST_WORKERS", "4"))) as pool:
            list(pool.map(_wav_to_flac_job, jobs, chunksize=8))
    flacs = sorted(p.relative_to(paths.data).as_posix() for p in out_dir.rglob("*.flac"))
    if not flacs:
        raise ManifestError("CML-TTS: nenhum áudio extraído")

    def split_of(rel: str) -> str:
        return PurePosixPath(rel).parts[2]

    by_split: dict[str, list[str]] = defaultdict(list)
    for rel in flacs:
        by_split[split_of(rel)].append(rel)
    records: list[FileRecord] = []
    for split, rels in by_split.items():
        if split not in ("train", "dev", "test"):
            raise ManifestError(f"CML-TTS: split inesperado {split!r}")
        records += build_records(rels, paths.data, source=src, split=split, codec="flac",
                                 speaker_of=lambda rel: PurePosixPath(rel).parts[4])
    write_file_manifest(records, _manifest_file(src_id))
    log(f"{src_id}: {len(records)} arquivos, {len({r.speaker for r in records})} falantes, "
        f"{sum(map(_seconds, records)) / 3600:.2f} h")
    return records


# ---------------------------------------------------------------------------------------------------------------------
# fase 5: Common Voice 27.0 pt completo (arquivo local, client_id fora do git)
# ---------------------------------------------------------------------------------------------------------------------

CV_EXCLUDED_VARIANTS = frozenset({"Portuguese (Portugal)"})


def cv_select_rows(rows: Iterable[Mapping[str, str]]) -> dict[str, str]:
    """{nome do clipe: client_id} dos clipes validados que não são da variante de Portugal.

    Decisão do M1 T8: variante vazia (corpus antigo, sem rótulo) entra junto com pt-BR; só `Portuguese (Portugal)` sai.
    """
    out: dict[str, str] = {}
    for row in rows:
        if row.get("variant", "") in CV_EXCLUDED_VARIANTS or not row.get("client_id"):
            continue
        out[PurePosixPath(row["path"]).name] = row["client_id"]
    if not out:
        raise ManifestError("CV: nenhum clipe selecionado")
    return out


def ingest_cv_full(paths: Paths, archive: Path, log=print) -> list[FileRecord]:
    src_id = "cv27-pt"
    private = paths.data / "manifests" / "cv27-pt-full.jsonl"
    if private.is_file():
        return read_file_manifest(private)
    src = _src(src_id)
    archive = Path(archive)
    if not archive.is_file():
        raise ManifestError(f"{archive} não existe (CCTRAIN_CV_ARCHIVE; download manual pelo dono)")
    from cctrain.data.pilot import CV_ARCHIVE_SHA256, _cv_read_validated

    wanted = cv_select_rows(_cv_read_validated(archive))
    out_dir = paths.data / "raw" / "cv27-pt-full"

    def select(name: str) -> str | None:
        p = PurePosixPath(name)
        return f"clips/{p.name}" if p.parent.name == "clips" and p.name in wanted else None

    if not out_dir.exists():
        digest, _ = stream_extract_tar(archive, out_dir, select, expected_sha256=CV_ARCHIVE_SHA256,
                                       need_bytes=int(4.5 * GB), done=lambda rel: True)
        update_lock(src_id, archive.name, digest.sha256, digest.size)
    rels = sorted(p.relative_to(paths.data).as_posix() for p in out_dir.rglob("*.mp3"))
    records = build_records(rels, paths.data, source=src, split="train", codec="mp3",
                            speaker_of=lambda rel: wanted[PurePosixPath(rel).name])
    digest_manifest = write_file_manifest(records, private)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    (FILES_DIR / "cv27-pt-full.sha256").write_text(f"{digest_manifest}  data/manifests/cv27-pt-full.jsonl\n")
    log(f"cv27-pt: {len(records)} clipes, {len({r.speaker for r in records})} client_id, "
        f"{sum(map(_seconds, records)) / 3600:.2f} h")
    return records


# ---------------------------------------------------------------------------------------------------------------------
# guarda de disco por fase e despacho
# ---------------------------------------------------------------------------------------------------------------------

def phase_deficit_gb(phase_id: str, done_phases: Iterable[str], free_bytes: int, reserve_gb: float,
                     config: Path = PHASES_V1, workdir: Path | None = None) -> float:
    """Déficit em GB (> 0 = `blocked_disk`) da fase, projetando só as fases ainda não concluídas com o livre atual."""
    cfg = load_phases_config(config)
    done = set(done_phases)
    remaining = [p for p in cfg["phases"] if p["id"] not in done]
    if phase_id not in {p["id"] for p in remaining}:
        return 0.0
    cfg = dict(cfg, phases=remaining)
    tmp = Path(workdir or Paths.from_env().build) / f"phases-remaining-{os.getpid()}.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    try:
        plan = plan_phases(free_bytes, reserve_gb, config=tmp)
    finally:
        tmp.unlink(missing_ok=True)
    phase = next(p for p in plan if p.id == phase_id)
    return max(-phase.cumulative_margin_gb, -phase.peak_margin_gb, 0.0)


def ingest_phase(phase: Phase, paths: Paths, *, cv_archive: Path | None = None, log=print) -> list[FileRecord]:
    """Ingere uma fase do `m1-phases-v1.yaml`; devolve os registros de todas as fontes de áudio da fase."""
    pid = phase.id
    if pid == "1-noise-rir-encoder":
        recs = ingest_dns(paths, log)
        for v in ingest_fsd50k(paths, log).values():
            recs += v
        return recs
    if pid == "2-vctk":
        return ingest_vctk(paths, log)
    if pid == "3-librispeech-100-test-clean":
        return ingest_librispeech(paths, "train-clean-100", log) + ingest_librispeech(paths, "test-clean", log)
    if pid == "4-cml-tts-pt":
        return ingest_cml(paths, log)
    if pid == "5-cv27-pt":
        archive = cv_archive or os.environ.get("CCTRAIN_CV_ARCHIVE")
        if not archive:
            raise ManifestError("CCTRAIN_CV_ARCHIVE ausente (aceite dos termos pelo dono)")
        return ingest_cv_full(paths, Path(archive), log)
    raise ManifestError(f"fase desconhecida: {pid}")


__all__ = ["phase_by_id", "ingest_phase", "DiskError", "ManifestError", "PhaseBlocked", "download_resumable", "dns_freesound_cc0", "fsd50k_allowed",
           "librispeech_cap", "librispeech_speaker_sex", "vctk_mic1_members", "build_records"]


def phase_by_id(phase_id: str, config: Path = PHASES_V1) -> Phase:
    """`Phase` (sem margens de disco, que a guarda calcula) para o despacho de `ingest_phase`."""
    for p in load_phases_config(config)["phases"]:
        if p["id"] == phase_id:
            return Phase(p["id"], tuple(s["id"] for s in p["sources"]), 0.0, 0.0, 0.0, 0.0, "ok")
    raise ManifestError(f"fase desconhecida: {phase_id}")


# ---------------------------------------------------------------------------------------------------------------------
# extra (fora do m1-phases-v1): TTS-Portuguese Corpus, 1 falante pt-BR 48 kHz
# ---------------------------------------------------------------------------------------------------------------------

TTSPT_URL = "https://www.dropbox.com/s/ohpc7epowv9ct7o/TTS-Portuguese-Corpus.zip?dl=1"
TTSPT_SIZE = 2371059638


def ingest_tts_portuguese(paths: Paths, log=print) -> list[FileRecord]:
    src_id = "tts-portuguese-corpus"
    if (cached := _cached(src_id)) is not None:
        return cached
    src = _src(src_id)
    arc = paths.data / "raw" / "_dl" / "TTS-Portuguese-Corpus.zip"
    d = download_resumable(TTSPT_URL, arc, expected_size=TTSPT_SIZE, log=log)
    update_lock(src_id, arc.name, d.sha256, d.size)
    rels = []
    with zipfile.ZipFile(arc) as z:
        wavs = sorted(n for n in z.namelist() if n.lower().endswith(".wav") and "__MACOSX" not in n)
        if not wavs:
            raise ManifestError("TTS-Portuguese: nenhum WAV no zip")
        for n in wavs:
            rel = f"raw/{src_id}/{PurePosixPath(n).stem}.flac"
            if not (paths.data / rel).is_file():
                _write_flac(io.BytesIO(z.read(n)), paths.data / rel)
            rels.append(rel)
    records = build_records(rels, paths.data, source=src, split="train", codec="flac",
                            speaker_of=lambda rel: "ttsptbr-male1")
    write_file_manifest(records, _manifest_file(src_id))
    arc.unlink()
    log(f"{src_id}: {len(records)} arquivos, {sum(map(_seconds, records)) / 3600:.2f} h")
    return records
