"""Busca os dados do piloto do M0. Uso: fetch_pilot.py {vctk,librispeech,noise,cv,all}."""

import os
import sys
from pathlib import Path

from cctrain.data import pilot
from cctrain.paths import Paths


def main(which: str) -> None:
    paths = Paths.from_env()
    steps = {"vctk": pilot.fetch_vctk, "librispeech": pilot.fetch_librispeech, "noise": pilot.fetch_noise}
    for name, fn in steps.items():
        if which in (name, "all"):
            recs = fn(paths)
            print(f"{name}: {len(recs)} arquivos, {pilot.duration_hours(recs):.2f} h")
    if which in ("cv", "all"):
        archive = os.environ.get("CCTRAIN_CV_ARCHIVE")
        if not archive:
            print("cv: NÃO EXECUTADO (CCTRAIN_CV_ARCHIVE ausente; aceite dos termos pendente com o dono)")
        else:
            recs = pilot.ingest_cv(paths, Path(archive))
            print(f"cv: {len(recs)} clipes, {len({r.speaker for r in recs})} client_id, "
                  f"{pilot.duration_hours(recs):.2f} h")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
