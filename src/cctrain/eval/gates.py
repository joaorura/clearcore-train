"""Carrega e valida os níveis pré-registrados (gates-v1) e o orçamento do M0, conferindo o congelamento."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from cctrain.hashing import sha256_file
from cctrain.paths import repo_root

GATES_V1 = repo_root() / "configs" / "eval" / "gates-v1.yaml"
BUDGET_V1 = repo_root() / "configs" / "m0" / "budget-v1.yaml"
REQUIRED_GATES = tuple(f"G{i}" for i in range(12))
OPTIONAL_GATES = ("G3b", "G5b")
CLASSES = frozenset({"B", "M"})
_PROBE_KEYS = {"condition", "sir_db", "split", "metric", "ci", "go_ci_lower_min"}
_M1_KEYS = {"eer_max", "tract_parity", "functional"}
_SCALARS = (int, float, bool, str, type(None))


class GatesError(ValueError):
    """Arquivo de níveis inválido ou alterado depois de congelado."""


@dataclass(frozen=True)
class Gate:
    id: str
    cls: str
    criteria: Mapping
    notes: str


@dataclass(frozen=True)
class GateSet:
    version: int
    gates: dict[str, Gate]
    m1: Mapping
    probe_rule: Mapping
    calibrated_in_dev: tuple[str, ...]
    statistics: Mapping
    runtime_path: Mapping
    informative: Mapping
    sha256: str


@dataclass(frozen=True)
class Budget:
    reserve_gb: float
    vram_b16_reserved_gb_max: float
    loader_rss_gb_max: float
    t3: Mapping
    sha256: str


def verify_frozen(path: Path) -> str:
    path = Path(path)
    lock = path.with_suffix(".sha256")
    if not lock.is_file():
        raise GatesError(f"{lock} ausente: o arquivo não está congelado")
    tokens = lock.read_text().split()
    if not tokens:
        raise GatesError(f"{lock} vazio: o arquivo não está congelado")
    expected = tokens[0]
    actual = sha256_file(path)
    if actual != expected:
        raise GatesError(f"{path} foi alterado depois de congelado ({actual} != {expected}); crie uma nova versão")
    return actual


def _check_leaves(obj, where: str) -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _check_leaves(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_leaves(v, f"{where}[{i}]")
    elif not isinstance(obj, _SCALARS):
        raise GatesError(f"{where}: tipo {type(obj).__name__} não permitido")


def load_gates(path: Path = GATES_V1, *, verify: bool = True) -> GateSet:
    path = Path(path)
    digest = verify_frozen(path) if verify else sha256_file(path)
    data = yaml.safe_load(path.read_text())
    if data.get("version") != 1:
        raise GatesError("version deve ser 1")
    raw_gates = data.get("gates") or {}
    missing = [g for g in REQUIRED_GATES if g not in raw_gates]
    if missing:
        raise GatesError(f"gates ausentes: {missing}")
    extra = set(raw_gates) - set(REQUIRED_GATES) - set(OPTIONAL_GATES)
    if extra:
        raise GatesError(f"gates desconhecidos: {sorted(extra)}")
    gates: dict[str, Gate] = {}
    for gid, g in raw_gates.items():
        if g.get("class") not in CLASSES:
            raise GatesError(f"{gid}: classe {g.get('class')!r} fora de {sorted(CLASSES)}")
        criteria = g.get("criteria")
        if not isinstance(criteria, Mapping) or not criteria:
            raise GatesError(f"{gid}: criteria vazio")
        _check_leaves(criteria, gid)
        gates[gid] = Gate(id=gid, cls=g["class"], criteria=criteria, notes=g.get("notes", ""))
    m1 = data.get("m1") or {}
    probe = data.get("probe_rule") or {}
    if set(m1) != _M1_KEYS:
        raise GatesError(f"m1 deve ter {sorted(_M1_KEYS)}")
    if set(probe) != _PROBE_KEYS:
        raise GatesError(f"probe_rule deve ter {sorted(_PROBE_KEYS)}")
    return GateSet(
        version=1, gates=gates, m1=m1, probe_rule=probe,
        calibrated_in_dev=tuple(data.get("calibrated_in_dev") or ()),
        statistics=data.get("statistics") or {}, runtime_path=data.get("runtime_path") or {},
        informative=data.get("informative") or {}, sha256=digest,
    )


def load_budget(path: Path = BUDGET_V1, *, verify: bool = True) -> Budget:
    path = Path(path)
    digest = verify_frozen(path) if verify else sha256_file(path)
    d = yaml.safe_load(path.read_text())
    if d.get("version") != 1:
        raise GatesError("budget: version deve ser 1")
    return Budget(
        reserve_gb=float(d["disk"]["reserve_gb"]),
        vram_b16_reserved_gb_max=float(d["vram"]["batch16_reserved_gb_max"]),
        loader_rss_gb_max=float(d["ram"]["loader_rss_gb_max"]),
        t3=d["t3"], sha256=digest,
    )
