"""Allowlist de licenças do treino (spec §4.1). Tudo o que não está na lista é recusado."""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    TRAIN = "train"
    TEST_ONLY = "test_only"
    TOOL_ONLY = "tool_only"


# `LicenseRef-cctrain-generated`: dados gerados pelo próprio repo, sem terceiros (RIR sintéticas).
# Não está na lista da spec §4.1; decisão de planejamento 6, a confirmar pelo dono no M5.
TRAIN_ALLOWED = frozenset(
    {
        "CC0-1.0",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "Apache-2.0",
        "LicenseRef-public-domain-declared",
        "LicenseRef-cctrain-generated",
    }
)


class LicenseRefused(PermissionError):
    """Arquivo fora da allowlist ou de fonte só-teste/ferramenta."""


def check_train_allowed(license_id: str, role: Role | str, *, what: str) -> None:
    role = Role(role)
    if role is not Role.TRAIN:
        raise LicenseRefused(f"{what}: papel {role.value} não pode entrar no treino")
    if license_id not in TRAIN_ALLOWED:
        raise LicenseRefused(
            f"{what}: licença {license_id!r} fora da allowlist {sorted(TRAIN_ALLOWED)}"
            " (NC, ND, SA, ODbL/DbCL, GPL, research-only e não confirmadas ficam fora)"
        )
