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
MANIFEST_SCHEMA = 2
SPLITS = frozenset({"train", "dev", "test", "noise", "rir", "pilot"})
# Dev/teste nunca entram no treino (spec §4.5); `pilot` é só de medição do M0 e também não é treinável.
NON_TRAIN_SPLITS = frozenset({"test", "dev", "pilot"})
TRAIN_SPLITS = ("noise", "rir", "train")
# Splits de fala exigem falante; `noise`/`rir` não têm, e `test` pode abrigar ruído retido (piloto do M0).
_SPEAKER_REQUIRED_SPLITS = frozenset({"train", "dev", "pilot"})
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
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ManifestError(f"{path}: YAML inválido: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ManifestError(f"{path}: não é um mapeamento YAML")
        missing = [k for k in _REQUIRED if k not in data]
        if missing:
            raise ManifestError(f"{path}: faltam {missing}")
        if data["id"] != Path(path).stem:
            raise ManifestError(f"{path}: id {data['id']!r} difere do nome do arquivo")
        if not isinstance(data["license"], str) or not data["license"].strip():
            raise ManifestError(f"{path}: licença vazia ou não textual")
        try:
            role = Role(data["role"])
        except ValueError as exc:
            raise ManifestError(f"{path}: papel inválido {data['role']!r}") from exc
        return cls(
            id=data["id"], name=data["name"], url=data["url"], landing_page=data["landing_page"],
            license=data["license"], license_evidence=data["license_evidence"], role=role,
            attribution=data["attribution"], native_sr=data["native_sr"], access=data["access"],
            artifacts=tuple(data.get("artifacts") or ()), restrictions=tuple(data.get("restrictions") or ()),
        )


def load_sources(directory: Path = SOURCES_DIR) -> dict[str, SourceManifest]:
    out: dict[str, SourceManifest] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        src = SourceManifest.from_yaml(path)
        out[src.id] = src
    return out


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _nonempty_str(v: object) -> bool:
    return isinstance(v, str) and bool(v.strip())


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
    attribution: str
    schema: int = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ManifestError(f"caminho inválido: {self.path!r}")
        p = PurePosixPath(self.path)
        if p.is_absolute() or ".." in p.parts:
            raise ManifestError(f"caminho inválido: {self.path!r}")
        if not isinstance(self.sha256, str) or not _SHA_RE.match(self.sha256):
            raise ManifestError(f"{self.path}: sha256 inválido")
        if self.schema != MANIFEST_SCHEMA or not _is_int(self.schema):
            raise ManifestError(f"{self.path}: schema {self.schema!r} não é {MANIFEST_SCHEMA}")
        for name in ("source_id", "role", "split", "codec"):
            if not isinstance(getattr(self, name), str):
                raise ManifestError(f"{self.path}: {name} não textual")
        if not _nonempty_str(self.license):
            raise ManifestError(f"{self.path}: licença ausente")
        if not _nonempty_str(self.attribution):
            raise ManifestError(f"{self.path}: atribuição ausente")
        if not _is_int(self.sr) or not _is_int(self.num_samples):
            raise ManifestError(f"{self.path}: sr/num_samples devem ser inteiros")
        if self.sr <= 0 or self.num_samples < 0:
            raise ManifestError(f"{self.path}: sr/num_samples inválidos")
        if self.speaker is not None and not _nonempty_str(self.speaker):
            raise ManifestError(f"{self.path}: speaker deve ser texto não vazio ou None")
        if self.band_hz is not None and (isinstance(self.band_hz, bool) or not isinstance(self.band_hz, int | float)):
            raise ManifestError(f"{self.path}: band_hz deve ser número ou None")
        if self.split not in SPLITS:
            raise ManifestError(f"{self.path}: split {self.split!r} fora de {sorted(SPLITS)}")
        try:
            Role(self.role)
        except ValueError as exc:
            raise ManifestError(f"{self.path}: papel inválido {self.role!r}") from exc
        if self.speaker is None and self.split in _SPEAKER_REQUIRED_SPLITS:
            raise ManifestError(f"{self.path}: split de fala {self.split!r} exige speaker")

    @property
    def speaker_key(self) -> str | None:
        """Chave de falante com namespace da fonte: dois corpora com `speaker="1"` não colidem."""
        return None if self.speaker is None else f"{self.source_id}:{self.speaker}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> FileRecord:
        fields = set(cls.__dataclass_fields__)
        keys = set(d)
        if keys != fields:
            raise ManifestError(f"registro com chaves inválidas: faltam {sorted(fields - keys)}, sobram {sorted(keys - fields)}")
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


_V1_FIELDS = frozenset(FileRecord.__dataclass_fields__) - {"attribution", "schema"}


def migrate_v1(d: Mapping, sources: Mapping[str, SourceManifest] | None = None) -> FileRecord:
    """Lê um registro v1 do M0 (sem `attribution`/`schema`); a atribuição vem do manifest da fonte."""
    keys = set(d)
    if keys != _V1_FIELDS:
        raise ManifestError(f"registro v1 com chaves inválidas: faltam {sorted(_V1_FIELDS - keys)}, "
                            f"sobram {sorted(keys - _V1_FIELDS)}")
    sources = load_sources() if sources is None else sources
    src = sources.get(d["source_id"])
    if src is None:
        raise ManifestError(f"{d['path']!r}: fonte {d['source_id']!r} sem manifest; não há atribuição para migrar")
    return FileRecord(**{k: d[k] for k in sorted(_V1_FIELDS)}, attribution=src.attribution, schema=MANIFEST_SCHEMA)


def write_file_manifest(records: Iterable[FileRecord], path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.path)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if prev.path == cur.path:
            raise ManifestError(f"path duplicado no manifest: {cur.path!r}")
    # U+2028/U+2029 ficam escapados: são quebras de linha para `str.splitlines` e para vários leitores de JSONL.
    lines = [json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False).replace("\u2028", "\\u2028")
             .replace("\u2029", "\\u2029") for r in ordered]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)


def read_file_manifest(path: Path, sources: Mapping[str, SourceManifest] | None = None) -> list[FileRecord]:
    """Lê JSONL v2, ou v1 do M0 (sem `schema`) via `migrate_v1`. Só `\\n` separa registros."""
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}:{n}: JSON inválido: {exc}") from exc
        if not isinstance(d, Mapping):
            raise ManifestError(f"{path}:{n}: registro não é um objeto JSON")
        try:
            out.append(FileRecord.from_dict(d) if "schema" in d else migrate_v1(d, sources))
        except ManifestError as exc:
            raise ManifestError(f"{path}:{n}: {exc}") from exc
    return out


def training_records(records: Iterable[FileRecord], sources: Mapping[str, SourceManifest], *,
                     allow_pilot: bool = False) -> list[FileRecord]:
    """Filtra para treino. `allow_pilot` é só do loader de piloto do M0 (medição de custo), nunca do T1."""
    out = []
    for r in records:
        src = sources.get(r.source_id)
        if src is None:
            raise ManifestError(f"{r.path}: fonte {r.source_id!r} sem manifest")
        if r.split == "pilot" and not allow_pilot:
            raise ManifestError(f"{r.path}: split 'pilot' é só de medição e não é treinável")
        if r.split in NON_TRAIN_SPLITS - {"pilot"}:
            raise ManifestError(f"{r.path}: split {r.split!r} é de dev/teste e não pode entrar no treino (spec §4.5)")
        if r.license != src.license:
            raise ManifestError(f"{r.path}: licença {r.license!r} difere da fonte ({src.license!r})")
        check_train_allowed(r.license, r.role, what=r.path)
        if Role(r.role) is not src.role:
            raise ManifestError(f"{r.path}: papel {r.role!r} difere do da fonte ({src.role.value!r})")
        out.append(r)
    return out


def _read_lock(lock_path: Path) -> dict:
    if not Path(lock_path).is_file():
        return {}
    try:
        lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{lock_path}: lock com JSON inválido: {exc}") from exc
    if not isinstance(lock, dict):
        raise ManifestError(f"{lock_path}: lock não é um objeto JSON")
    return lock


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
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, lock_path)
