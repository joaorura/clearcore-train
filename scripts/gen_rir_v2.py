"""Gera as RIR `synthetic-v2` (cauda longa) em data/rir/synthetic-v2 e os manifests versionados (schema v2)."""

import argparse

from cctrain.data.manifest import FILES_DIR, write_file_manifest
from cctrain.data.rir import params_to_jsonl, write_rir_v2_set
from cctrain.paths import Paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=256)
    args = ap.parse_args()
    paths = Paths.from_env()
    records, params = write_rir_v2_set(paths.data / "rir" / "synthetic-v2", range(args.count), data_root=paths.data)
    digest = write_file_manifest(records, FILES_DIR / "rir-synthetic-v2.jsonl")
    (FILES_DIR / "rir-synthetic-v2.params.jsonl").write_text(params_to_jsonl(params), encoding="utf-8")
    print(f"{len(records)} RIR v2; manifest sha256 {digest}")


if __name__ == "__main__":
    main()
