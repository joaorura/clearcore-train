#!/usr/bin/env python3
"""Exporta um checkpoint do pDFNet3 Pro v4-keep no CONTRATO v3 (mesmo trio enc/erb_dec/df_dec, sem `p_target`).

Diferença para `m3_export_pro_v3.py`: o `manifest.json` ganha `model` próprio e o campo `gating` (objeto) com a regra
do gating "novo" com perfil, a mesma que o treino emula (`train_pro_v4.gate_v4`) e que o v3.1 usou na avaliação:

    com speaker_present = 1: estágio = ZERO se lsnr < zero_below_db, senão ERB+DF (sem passthrough nem ERB-only)
    sem perfil:              lsnr de sempre (ZERO < -10, ruidoso > 30, só ERB em (20, 30])

O campo é escrito em duas formas equivalentes, para o runtime que lê qualquer uma delas:
  - `mode: "lsnr-profile"`, `zero_below_db: -10` (forma curta);
  - `profile` / `no_profile` com `min_db`/`max_db_erb`/`max_db_df` (`null` = sem limite), a forma de
    `docs/plan-pro-v4.md` §3.3.
Asset sem `gating` objeto mantém o comportamento v3 (gating atual também com perfil).

Uso: python scripts/m3_export_pro_v4_keep.py --ckpt <ckpt> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cctrain.export.pack import read_members  # noqa: E402
from cctrain.export.pdfnet3_pro_v3 import GRAPHS, build_manifest, export_graph, load_pro_v3, pack, sha256_file  # noqa: E402
from cctrain.paths import Paths, clearcore_root  # noqa: E402

MODEL_ID = "pdfnet3-pro-v4-keep"
ZERO_BELOW_DB = -10.0
GATING = {
    "mode": "lsnr-profile",
    "applies_when": "speaker_present == 1",
    "zero_below_db": ZERO_BELOW_DB,
    "profile": {"min_db": ZERO_BELOW_DB, "max_db_erb": None, "max_db_df": None},
    "no_profile": {"mode": "lsnr", "min_db": -10.0, "max_db_erb": 30.0, "max_db_df": 20.0},
    "decision": "com perfil: ZERO se lsnr < zero_below_db, senão ERB+DF (sem passthrough nem ERB-only); sem perfil: "
                "o gating por lsnr de sempre. null = sem limite.",
    "execution": "os três grafos rodam em todo quadro ativo (estados sempre avançam); o gating só seleciona a saída. "
                 "Atalho de silêncio digital como hoje.",
}


def export_asset(ckpt: Path, out_dir: Path, approved: Path) -> tuple[Path, str, dict]:
    out_dir = Path(out_dir)
    members = out_dir / "members"
    if members.exists():
        shutil.rmtree(members)
    members.mkdir(parents=True)
    pro = load_pro_v3(ckpt)  # strict: a arquitetura do keep é a do v3/v3.1 (155 tensores, sem cabeça de presença)
    files = {f"{g}.onnx": export_graph(pro, g, members / f"{g}.onnx") for g in GRAPHS}
    cfg = members / "config.ini"
    cfg.write_bytes(read_members(approved)["config.ini"])
    files["config.ini"] = cfg
    manifest = build_manifest(files, ckpt, sha256_file(ckpt))
    manifest["model"] = MODEL_ID
    manifest["contract"] = "pdfnet3-pro-v3 (grafos, E/S e estados idênticos)"
    manifest["gating"] = GATING
    (members / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tar = out_dir / f"{MODEL_ID}-stateful-asset-v1.tar.gz"
    return tar, pack(tar, files, manifest), manifest


def main() -> None:
    paths = Paths.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=paths.runs / "m4" / "pro-v4-keep" / "export")
    ap.add_argument("--approved", type=Path,
                    default=clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin")
    a = ap.parse_args()
    tar, sha, manifest = export_asset(a.ckpt, a.out_dir, a.approved)
    tar2, sha2, _ = export_asset(a.ckpt, a.out_dir / "repro", a.approved)
    shutil.rmtree(a.out_dir / "repro")
    print(json.dumps({"asset": str(tar), "sha256": sha, "reproducible": sha == sha2, "gating": manifest["gating"],
                      "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
