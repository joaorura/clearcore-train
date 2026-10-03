"""Port do SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) para o nosso módulo (M1 Tarefa 10).

O SpeechBrain não é dependência: carregamos o `state_dict` com `torch.load(weights_only=True)` e remapeamos os
nomes para `ECAPATDNN`. O checkpoint é fixado por revisão HF e SHA-256 em manifests/models/spkrec-ecapa-voxceleb.yaml
(licença Apache-2.0 conferida no repositório HF; proveniência VoxCeleb). Pesos derivados são CC BY-NC 4.0 + NOTICE.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import yaml
from torch import Tensor

from cctrain.data.sources import stream_download
from cctrain.export.enrollment import EnrollmentModel
from cctrain.hashing import sha256_file
from cctrain.models.ecapa import ECAPATDNN
from cctrain.models.fbank import SpeechBrainFbank
from cctrain.models.film import FilmGenerator
from cctrain.paths import repo_root

MANIFEST_PATH = repo_root() / "manifests" / "models" / "spkrec-ecapa-voxceleb.yaml"
CKPT_NAME = "embedding_model.ckpt"
FETCH_FILES = ("embedding_model.ckpt", "hyperparams.yaml", "README.md")
REFERENCE_FILES = ("classifier.ckpt", "mean_var_norm_emb.ckpt", "label_encoder.txt", "config.json",
                   "example1.wav", "example2.flac")  # só para gerar/estender os vetores de referência
CHANNELS, MFA_CHANNELS, EMBEDDING_DIM = 1024, 3072, 192


class ModelManifestError(RuntimeError):
    """Manifest do modelo ou arquivo do checkpoint fora do fixado. Falha fechado."""


def load_model_manifest(path: Path = MANIFEST_PATH) -> dict:
    m = yaml.safe_load(Path(path).read_text())
    if m.get("license") != "Apache-2.0":
        raise ModelManifestError(f"licença do checkpoint não é Apache-2.0: {m.get('license')!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(m.get("revision", ""))):
        raise ModelManifestError("revision HF não é um SHA de commit de 40 hex")
    return m


def _check_file(path: Path, manifest: dict, name: str) -> None:
    pinned = manifest["files"][name]
    got = sha256_file(path)
    if got != pinned["sha256"]:
        raise ModelManifestError(f"{name}: sha256 {got} != {pinned['sha256']}")


def fetch_checkpoint(dest: Path, names: tuple[str, ...] = FETCH_FILES) -> Path:
    """Baixa os arquivos do checkpoint na revisão fixada, conferindo SHA-256 e tamanho (única rede da Tarefa 10)."""
    m = load_model_manifest()
    dest = Path(dest)
    for name in names:
        f = m["files"][name]
        stream_download(f"{m['base_url']}/{name}", dest / name, expected_sha256=f["sha256"], expected_size=f["size"])
    return dest


def load_speechbrain_state(ckpt_dir: Path) -> dict[str, Tensor]:
    """`state_dict` do `embedding_model.ckpt`, depois de conferir o SHA-256 fixado (antes de desserializar)."""
    path = Path(ckpt_dir) / CKPT_NAME
    _check_file(path, load_model_manifest(), CKPT_NAME)
    return torch.load(path, map_location="cpu", weights_only=True)


_RULES: tuple[tuple[str, str], ...] = (
    (r"blocks\.0\.conv\.conv\.(\w+)", r"layer1.conv.\1"),
    (r"blocks\.0\.norm\.norm\.(\w+)", r"layer1.bn.\1"),
    (r"blocks\.([123])\.(tdnn[12])\.conv\.conv\.(\w+)", r"layer\1+1.\2.conv.\3"),
    (r"blocks\.([123])\.(tdnn[12])\.norm\.norm\.(\w+)", r"layer\1+1.\2.bn.\3"),
    (r"blocks\.([123])\.res2net_block\.blocks\.(\d)\.conv\.conv\.(\w+)", r"layer\1+1.res2.blocks.\2.conv.\3"),
    (r"blocks\.([123])\.res2net_block\.blocks\.(\d)\.norm\.norm\.(\w+)", r"layer\1+1.res2.blocks.\2.bn.\3"),
    (r"blocks\.([123])\.se_block\.(conv[12])\.conv\.(\w+)", r"layer\1+1.se.\2.\3"),
    (r"mfa\.conv\.conv\.(\w+)", r"mfa.conv.\1"),
    (r"mfa\.norm\.norm\.(\w+)", r"mfa.bn.\1"),
    (r"asp\.tdnn\.conv\.conv\.(\w+)", r"pool.tdnn.conv.\1"),
    (r"asp\.tdnn\.norm\.norm\.(\w+)", r"pool.tdnn.bn.\1"),
    (r"asp\.conv\.conv\.(\w+)", r"pool.conv.\1"),
    (r"asp_bn\.norm\.(\w+)", r"pool_bn.\1"),
    (r"fc\.conv\.(\w+)", r"fc.\1"),
)


def _map_key(key: str) -> str:
    for pattern, repl in _RULES:
        m = re.fullmatch(pattern, key)
        if m:
            out = m.expand(repl)
            return re.sub(r"layer(\d)\+1", lambda g: f"layer{int(g.group(1)) + 1}", out)
    raise ModelManifestError(f"tensor do checkpoint sem regra de mapeamento: {key}")


def port_ecapa(sb_state: dict[str, Tensor], context_mode: str = "expand") -> ECAPATDNN:
    """`ECAPATDNN(C=1024, mfa=3072, 192-d, reflect)` com os pesos do SpeechBrain, em modo eval."""
    model = ECAPATDNN(channels=CHANNELS, mfa_channels=MFA_CHANNELS, embedding_dim=EMBEDDING_DIM,
                      context_mode=context_mode, pad_mode="reflect")
    mapped: dict[str, Tensor] = {}
    for key, value in sb_state.items():
        new = _map_key(key)
        if new in mapped:
            raise ModelManifestError(f"mapeamento duplicado: {key} -> {new}")
        # fc é Conv1d k=1 no SpeechBrain e Linear aqui; os demais Conv1d têm o mesmo formato
        mapped[new] = value.squeeze(-1) if new == "fc.weight" else value
    model.load_state_dict(mapped, strict=True)
    return model.eval()


def build_enrollment(encoder: ECAPATDNN, generator: FilmGenerator) -> EnrollmentModel:
    """Áudio 16 kHz [1, N] -> 4 vetores FiLM [256] e embedding [192] (L2), com o front-end do SpeechBrain."""
    return EnrollmentModel(SpeechBrainFbank(), encoder, generator)
