"""Acesso ao DeepFilterNet v0.5.6 vendorizado e ao checkpoint PyTorch do DFNet3."""

from __future__ import annotations

import importlib
import json
import os
import sys
import zipfile
from pathlib import Path
from types import ModuleType

from cctrain.disk import require_free
from cctrain.hashing import sha256_file
from cctrain.paths import ConfigError, Paths, repo_root

VENDORED_ROOT = repo_root() / "third_party" / "DeepFilterNet"
_PACKAGE_DIR = VENDORED_ROOT / "DeepFilterNet"
_ZIP_REL = "models/DeepFilterNet3.zip"
_MARKER = ".cctrain-extracted.json"


def vendored_info() -> dict:
    return json.loads((VENDORED_ROOT / "VENDORED.json").read_text())


def ensure_df_importable() -> ModuleType:
    path = str(_PACKAGE_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    module = importlib.import_module("df")
    if not str(Path(module.__file__).resolve()).startswith(str(_PACKAGE_DIR.resolve())):
        raise ConfigError(f"pacote 'df' importado de fora do vendor: {module.__file__}")
    return module


def ensure_dfnet3_checkpoint(paths: Paths | None = None) -> Path:
    paths = paths or Paths.from_env()
    expected = vendored_info()["files"][_ZIP_REL]
    target = paths.data / "models" / "DeepFilterNet3"
    marker = target / _MARKER
    if marker.is_file() and json.loads(marker.read_text()).get("zip_sha256") == expected["sha256"]:
        return _model_dir(target)
    zip_path = VENDORED_ROOT / _ZIP_REL
    if sha256_file(zip_path) != expected["sha256"]:
        raise ConfigError(f"{zip_path}: SHA-256 difere de VENDORED.json")
    require_free(paths.data, 4 * expected["size"])
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        raise ConfigError(f"{staging} existe de uma extração interrompida; inspecione antes de continuar")
    staging.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ConfigError(f"membro inseguro no zip: {info.filename}")
        z.extractall(staging)
    (staging / _MARKER).write_text(json.dumps({"zip_sha256": expected["sha256"]}) + "\n")
    if target.exists():
        raise ConfigError(f"{target} existe sem marcador válido; inspecione antes de continuar")
    os.replace(staging, target)
    return _model_dir(target)


def _model_dir(target: Path) -> Path:
    configs = sorted(p for p in target.rglob("config.ini"))
    if len(configs) != 1:
        raise ConfigError(f"esperado 1 config.ini em {target}, achados {len(configs)}")
    model_dir = configs[0].parent
    if not (model_dir / "checkpoints").is_dir():
        raise ConfigError(f"{model_dir} sem checkpoints/")
    return model_dir
