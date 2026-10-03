"""Splits disjuntos por `speaker_key` (spec §4.5), comuns ao M1 e ao M2 (M1 Tarefa 9).

`assign_splits` é uma função pura dos registros e da configuração: o mesmo falante tem o mesmo split para sempre, e
nenhum falante cai em dois splits (a chave leva o namespace da fonte). `apply_splits` grava o split nos registros e marca
dev/teste como `test_only`, de modo que o loader recuse esses clipes por dois caminhos (split e papel).
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from cctrain.data.manifest import FileRecord
from cctrain.paths import Paths, repo_root

SPLITS_V1 = repo_root() / "configs" / "data" / "splits-v1.yaml"
RULES = frozenset({"random_holdout", "official", "train", "cv_clients"})
FINAL_SPLITS = frozenset({"train", "dev", "test"})


class SplitError(ValueError):
    """Configuração de split inválida ou atribuição inconsistente. Falha fechado."""


@dataclass(frozen=True)
class SplitConfig:
    version: int
    seed: int
    sources: Mapping[str, Mapping]


def load_split_config(path: Path = SPLITS_V1) -> SplitConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, Mapping) or d.get("version") != 1 or not isinstance(d.get("seed"), int):
        raise SplitError(f"{path}: configuração de split inválida")
    for sid, rule in (d.get("sources") or {}).items():
        if rule.get("rule") not in RULES:
            raise SplitError(f"{path}: fonte {sid!r} com regra inválida {rule.get('rule')!r}")
    return SplitConfig(version=1, seed=d["seed"], sources=d["sources"])


def _order(keys: Iterable[str], seed: int) -> list[str]:
    return sorted(keys, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())


def dedupe_by_sha256(records: Iterable[FileRecord]) -> list[FileRecord]:
    """Um registro por conteúdo (mantém o de menor `path`). O DNS e o FSD50K trazem cópias idênticas."""
    seen: dict[str, FileRecord] = {}
    for r in sorted(records, key=lambda r: r.path):
        seen.setdefault(r.sha256, r)
    return sorted(seen.values(), key=lambda r: r.path)


def assign_splits(records: Iterable[FileRecord], cfg: SplitConfig, *,
                  cv_retained: Collection[str] | None = None) -> dict[str, str]:
    """{speaker_key: 'train'|'dev'|'test'}. Registros sem falante (ruído, RIR) e do split `pilot` são ignorados.

    `cv_retained`: `speaker_key` do CV que passaram no filtro (§4.6). Sem ele o CV só ganha o dev (a calibração do
    filtro vem antes dele); com ele, o teste sai dos retidos fora do dev.
    """
    by_source: dict[str, dict[str, list[FileRecord]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.speaker_key is None or r.split == "pilot":
            continue
        by_source[r.source_id][r.speaker_key].append(r)
    out: dict[str, str] = {}
    for sid in sorted(by_source):
        rule = cfg.sources.get(sid)
        if rule is None:
            raise SplitError(f"fonte {sid!r} sem regra em splits-v1.yaml")
        spk = by_source[sid]
        kind = rule["rule"]
        if kind == "train":
            out.update({k: "train" for k in spk})
        elif kind == "official":
            for k, recs in spk.items():
                splits = {r.split for r in recs}
                if len(splits) != 1:
                    raise SplitError(f"{k}: falante em mais de um split oficial {sorted(splits)}")
                (s,) = splits
                if s not in FINAL_SPLITS:
                    raise SplitError(f"{k}: split oficial {s!r} fora de {sorted(FINAL_SPLITS)}")
                out[k] = s
        elif kind == "random_holdout":
            fixed = {f"{sid}:{p}" for p in rule["test_fixed"]}
            missing = fixed - set(spk)
            if missing:
                raise SplitError(f"{sid}: falantes fixos ausentes: {sorted(missing)}")
            pool = _order(set(spk) - fixed, cfg.seed)
            test = fixed | set(pool[: rule["test_random"]])
            dev = set(pool[rule["test_random"]: rule["test_random"] + rule["dev_random"]])
            out.update({k: "test" if k in test else "dev" if k in dev else "train" for k in spk})
        else:  # cv_clients
            cand = [k for k, recs in spk.items() if len(recs) >= rule["min_clips"]]
            dev = set(_order(cand, cfg.seed)[: rule["dev_clients"]])
            test: set[str] = set()
            if cv_retained is not None:
                pool = _order((set(cand) & set(cv_retained)) - dev, cfg.seed)
                test = set(pool[: rule["test_clients"]])
            out.update({k: "dev" if k in dev else "test" if k in test else "train" for k in spk})
    return out


def apply_splits(records: Iterable[FileRecord], assignment: Mapping[str, str]) -> list[FileRecord]:
    out = []
    for r in records:
        if r.speaker_key is None:
            out.append(r)
            continue
        if r.speaker_key not in assignment:
            raise SplitError(f"{r.path}: falante {r.speaker_key!r} sem split atribuído")
        split = assignment[r.speaker_key]
        role = "test_only" if split in ("dev", "test") else r.role
        out.append(dataclasses.replace(r, split=split, role=role))
    return out


# ---- sexo do falante (só LibriSpeech e VCTK), para o gate funcional de mesmo sexo ----

_LS_SOURCES = ("librispeech-dev-clean", "librispeech-test-clean", "librispeech-train-clean-100")


@lru_cache(maxsize=8)
def _sex_tables(data_dir: str) -> dict[str, str]:
    root = Path(data_dir) / "raw"
    out: dict[str, str] = {}
    for sid in _LS_SOURCES:
        f = root / sid / "SPEAKERS.TXT"
        if f.is_file():
            for line in f.read_text(errors="replace").splitlines():
                if line.startswith(";") or "|" not in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if parts[0].isdigit() and parts[1] in ("F", "M"):
                    out[f"{sid}:{parts[0]}"] = parts[1]
    f = root / "vctk-0.92" / "speaker-info.txt"
    if f.is_file():
        for line in f.read_text(errors="replace").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3 and parts[2] in ("F", "M"):
                out[f"vctk-0.92:{parts[0]}"] = parts[2]
    return out


def speaker_sex(key: str, data_dir: Path | None = None) -> str | None:
    """'F'|'M' para falantes do LibriSpeech e do VCTK; None para o resto (ou falante ausente da tabela)."""
    data_dir = Paths.from_env().data if data_dir is None else Path(data_dir)
    return _sex_tables(str(data_dir)).get(key)
