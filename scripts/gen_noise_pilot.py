"""Gera os 16 ruídos coloridos sintéticos do piloto em data/noise/synthetic-v1 e o manifest versionado.

Rótulo: pilot-only, cost measurement (R2 do M0); não substitui o corpus de ruído do treino.
"""

from cctrain.data.manifest import FILES_DIR, write_file_manifest
from cctrain.data.noise import write_noise_set
from cctrain.paths import Paths


def main() -> None:
    paths = Paths.from_env()
    records = write_noise_set(paths.data / "noise" / "synthetic-v1", seeds=range(16), data_root=paths.data)
    digest = write_file_manifest(records, FILES_DIR / "noise-synthetic-v1-pilot.jsonl")
    print(f"{len(records)} ruídos; manifest sha256 {digest}")


if __name__ == "__main__":
    main()
