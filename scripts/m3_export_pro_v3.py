#!/usr/bin/env python3
"""Exporta o pDFNet3 Pro v3 completo (stateful, S=1) e empacota o asset com manifest.json.

Uso:
    python scripts/m3_export_pro_v3.py [--ckpt PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cctrain.export.pack import read_members
from cctrain.export.pdfnet3_pro_v3 import GRAPHS, MODEL_ID, build_manifest, export_graph, load_pro_v3, pack, sha256_file
from cctrain.paths import Paths, clearcore_root

DEFAULT_CKPT = "m2/pro-v2-mhca-k16-tse/checkpoints/ckpt-00002000.pt"


def export_asset(ckpt: Path, out_dir: Path, approved: Path) -> tuple[Path, str, dict]:
    out_dir = Path(out_dir)
    members = out_dir / "members"
    if members.exists():
        shutil.rmtree(members)
    members.mkdir(parents=True)
    pro = load_pro_v3(ckpt)
    files = {f"{g}.onnx": export_graph(pro, g, members / f"{g}.onnx") for g in GRAPHS}
    cfg = members / "config.ini"  # mesmo config.ini do DFNet3 v0.5.6 (parâmetros de DSP do runtime)
    cfg.write_bytes(read_members(approved)["config.ini"])
    files["config.ini"] = cfg
    manifest = build_manifest(files, ckpt, sha256_file(ckpt))
    (members / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tar = out_dir / f"{MODEL_ID}-stateful-asset-v1.tar.gz"
    return tar, pack(tar, files, manifest), manifest


def main() -> None:
    paths = Paths.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=paths.runs / DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=Path, default=paths.runs / "m3_deploy_pro_v3_full")
    ap.add_argument("--approved", type=Path,
                    default=clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin")
    a = ap.parse_args()
    tar, sha, manifest = export_asset(a.ckpt, a.out_dir, a.approved)
    tar2, sha2, _ = export_asset(a.ckpt, a.out_dir / "repro", a.approved)
    shutil.rmtree(a.out_dir / "repro")
    print(json.dumps({"asset": str(tar), "sha256": sha, "reproducible": sha == sha2,
                      "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
