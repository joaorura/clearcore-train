"""Grava `configs/data/splits-v1.resolved.json`: o split de todo falante que não é do Common Voice (versionado).

O CV fica fora: o `client_id` nunca vai para o git (spec §4.6); o split dele é gravado por `m1_cv_filter.py` em `data/`.
"""

from __future__ import annotations

import json
import sys

from cctrain.data.manifest import read_file_manifest
from cctrain.data.splits import SPLITS_V1, assign_splits, dedupe_by_sha256, load_split_config
from cctrain.hashing import sha256_file
from cctrain.paths import repo_root

NON_SPEECH_SUFFIX = ".params.jsonl"
OUT = repo_root() / "configs" / "data" / "splits-v1.resolved.json"


def non_cv_speech_records():
    recs = []
    for f in sorted((repo_root() / "manifests" / "files").glob("*.jsonl")):
        if f.name.endswith(NON_SPEECH_SUFFIX):
            continue
        recs += [r for r in read_file_manifest(f) if r.speaker is not None and r.split != "pilot"]
    # o manifest do piloto do VCTK repete 6 clipes do mic1 (mesmo SHA-256, outro caminho): um registro por conteúdo
    return dedupe_by_sha256(recs)


def main() -> int:
    cfg = load_split_config()
    assignment = assign_splits(non_cv_speech_records(), cfg)
    doc = {"version": 1, "seed": cfg.seed, "config_sha256": sha256_file(SPLITS_V1),
           "assignment": dict(sorted(assignment.items()))}
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    counts: dict[str, dict[str, int]] = {}
    for k, v in assignment.items():
        counts.setdefault(k.split(":")[0], {}).setdefault(v, 0)
        counts[k.split(":")[0]][v] += 1
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
