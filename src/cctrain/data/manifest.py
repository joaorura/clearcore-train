"""Manifests de fonte (YAML, um por fonte) e por arquivo (JSONL), e o lock de hashes dos brutos."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

import yaml

from cctrain.data.licenses import Role, check_train_allowed
from cctrain.hashing import sha256_file
from cctrain.paths import repo_root

SOURCES_DIR = repo_root() / "manifests" / "sources"
FILES_DIR = repo_root() / "manifests" / "files"
LOCK_PATH = repo_root() / "manifests" / "sources.lock.json"
SPLITS = frozenset({"pilot", "test", "noise", "rir", "dev"})
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = ("id", "name", "url", "landing_page", "license", "license_evidence", "role", "attribution",
             "native_sr", "access")


class ManifestError(ValueError):
    """Manifest inválido ou inconsistente. Falha fechado."""


@dataclass(frozen=True)
class SourceManifest:
    id: str
    name: str
    url: str
    landing_page: str
    license: str
    license_evidence: str
    role: Role
    attribution: str
    native_sr: int | None
    access: str
    artifacts: tuple[dict, ...] = field(default_factory=tuple)
    restrictions: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_yaml(cls, path: Path) -> SourceManifest:
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, Mapping):
            raise ManifestError(f"{path}: não é um mapeamento YAML")
        missing = [k for k in _REQUIRED if k not in data]
        if missing:
            raise ManifestError(f"{path}: faltam {missing}")
        if data["id"] != Path(path).stem:
            raise ManifestError(f"{path}: id {data['id']!r} difere do nome do arquivo")
        if not str(data["license"]).strip():
            raise ManifestError(f"{path}: licença vazia")
        return cls(
            id=data["id"], name=data["name"], url=data["url"], landing_page=data["landing_page"],
            license=data["license"], license_evidence=data["license_evidence"], role=Role(data["role"]),
            attribution=data["attribution"], native_sr=data["native_sr"], access=data["access"],
            artifacts=tuple(data.get("artifacts") or ()), restrictions=tuple(data.get("restrictions") or ()),
        )


def load_sources(directory: Path = SOURCES_DIR) -> dict[str, SourceManifest]:
    out: dict[str, SourceManifest] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        src = SourceManifest.from_yaml(path)
        out[src.id] = src
    return out


@dataclass(frozen=True)
class FileRecord:
    path: str
    sha256: str
    source_id: str
    license: str
    role: str
    speaker: str | None
    split: str
    sr: int
    num_samples: int
    band_hz: float | None
    codec: str

    def __post_init__(self) -> None:
        p = PurePosixPath(self.path)
        if p.is_absolute() or ".." in p.parts or not self.path:
            raise ManifestError(f"caminho inválido: {self.path!r}")
        if not _SHA_RE.match(self.sha256):
            raise ManifestError(f"{self.path}: sha256 inválido")
        if self.sr <= 0 or self.num_samples < 0:
            raise ManifestError(f"{self.path}: sr/num_samples inválidos")
        if self.split not in SPLITS:
            raise ManifestError(f"{self.path}: split {self.split!r} fora de {sorted(SPLITS)}")
        Role(self.role)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> FileRecord:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def write_file_manifest(records: Iterable[FileRecord], path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False) for r in sorted(records, key=lambda r: r.path)]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for line in lines))
    os.replace(tmp, path)
    return sha256_file(path)


def read_file_manifest(path: Path) -> list[FileRecord]:
    return [FileRecord.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line]


def training_records(records: Iterable[FileRecord], sources: Mapping[str, SourceManifest]) -> list[FileRecord]:
    out = []
    for r in records:
        src = sources.get(r.source_id)
        if src is None:
            raise ManifestError(f"{r.path}: fonte {r.source_id!r} sem manifest")
        if r.license != src.license:
            raise ManifestError(f"{r.path}: licença {r.license!r} difere da fonte ({src.license!r})")
        check_train_allowed(r.license, r.role, what=r.path)
        out.append(r)
    return out


def _read_lock(lock_path: Path) -> dict:
    return json.loads(Path(lock_path).read_text()) if Path(lock_path).is_file() else {}


def lock_entry(source_id: str, artifact: str, lock_path: Path = LOCK_PATH) -> dict | None:
    return _read_lock(lock_path).get(source_id, {}).get(artifact)


def update_lock(source_id: str, artifact: str, sha256: str, size: int, *, upstream_checksum: str | None = None,
                lock_path: Path = LOCK_PATH) -> None:
    if not _SHA_RE.match(sha256):
        raise ManifestError(f"sha256 inválido para {source_id}/{artifact}")
    lock = _read_lock(lock_path)
    current = lock.get(source_id, {}).get(artifact)
    entry = {"sha256": sha256, "size": size, "upstream_checksum": upstream_checksum}
    if current is not None and (current["sha256"] != sha256 or current["size"] != size):
        raise ManifestError(
            f"{source_id}/{artifact}: hash mudou ({current['sha256']} -> {sha256}); o bruto upstream mudou,"
            " investigue antes de aceitar"
        )
    lock.setdefault(source_id, {})[artifact] = entry
    tmp = Path(lock_path).with_name(Path(lock_path).name + ".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, lock_path)
