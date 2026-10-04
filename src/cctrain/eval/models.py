"""Avaliadores locais com SHA-256 pinado em `manifests/models/<id>.yaml` (DNSMOS, Whisper; M2 Tarefa 3).

`verify_model` confere cada arquivo marcado `used: true` contra o manifest e falha fechado (arquivo ausente ou hash
diferente). Os modelos ficam em `$CCTRAIN_DATA_DIR/models/<local_dir>/`, fora do git, e nunca entram no pacote de entrega.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cctrain.hashing import sha256_file
from cctrain.paths import Paths, repo_root

MANIFEST_DIR = repo_root() / "manifests" / "models"


class ModelMismatch(RuntimeError):
    """Arquivo de modelo ausente ou com SHA-256 diferente do manifest."""


def load_model_manifest(model_id: str, manifest_dir: Path = MANIFEST_DIR) -> dict:
    return yaml.safe_load((Path(manifest_dir) / f"{model_id}.yaml").read_text(encoding="utf-8"))


def model_dir(model_id: str, paths: Paths | None = None, manifest_dir: Path = MANIFEST_DIR) -> Path:
    return (paths or Paths.from_env()).data / "models" / load_model_manifest(model_id, manifest_dir)["local_dir"]


def verify_model(model_id: str, paths: Paths | None = None, manifest_dir: Path = MANIFEST_DIR) -> Path:
    """Devolve o diretório do modelo depois de conferir SHA-256 e tamanho dos arquivos `used`."""
    man = load_model_manifest(model_id, manifest_dir)
    base = model_dir(model_id, paths, manifest_dir)
    for rel, meta in man["files"].items():
        if not meta.get("used"):
            continue
        f = base / rel
        if not f.is_file():
            raise ModelMismatch(f"{model_id}: {f} ausente")
        if f.stat().st_size != meta["size"] or sha256_file(f) != meta["sha256"]:
            raise ModelMismatch(f"{model_id}: {rel} não confere com o manifest (SHA-256/tamanho)")
    return base
