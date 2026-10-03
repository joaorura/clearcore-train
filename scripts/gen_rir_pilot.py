"""Gera as 64 RIR do piloto em data/rir/synthetic-v1 e os manifests versionados."""

from cctrain.data.manifest import FILES_DIR, write_file_manifest
from cctrain.data.rir import params_to_jsonl, write_rir_set
from cctrain.paths import Paths


def main() -> None:
    paths = Paths.from_env()
    records, params = write_rir_set(paths.data / "rir" / "synthetic-v1", seeds=range(64), data_root=paths.data)
    digest = write_file_manifest(records, FILES_DIR / "rir-synthetic-v1-pilot.jsonl")
    (FILES_DIR / "rir-synthetic-v1-pilot.params.jsonl").write_text(params_to_jsonl(params))
    print(f"{len(records)} RIR; manifest sha256 {digest}")


if __name__ == "__main__":
    main()
