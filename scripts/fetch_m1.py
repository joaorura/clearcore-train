"""Ingestão do M1 em background (Tarefa 8), fase a fase, idempotente e retomável.

Uso: nohup python scripts/fetch_m1.py [--phases 1 2 ...] > runs/m1/fetch.log 2>&1 &
`--prefetch-only` só baixa os brutos da fase 1. Estado em runs/m1/fetch-status.json. Fase `blocked_disk` não inicia e
encerra com código 2 e o déficit em GB; falha de uma fase não impede as seguintes (código final 1).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from cctrain.data import ingest
from cctrain.disk import GB
from cctrain.eval.gates import load_budget
from cctrain.m1.disk_plan import load_phases_config
from cctrain.paths import Paths


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_status(path: Path, status: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="*", type=int, help="números das fases (1-5); padrão: todas")
    ap.add_argument("--status", default="fetch-status.json", help="arquivo de estado em runs/m1")
    ap.add_argument("--prefetch-only", action="store_true")
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    if args.prefetch_only:
        for k, d in ingest.prefetch_phase1(Path(paths.data), log=_log).items():
            _log(f"{k}: {d.size} B sha256 {d.sha256}")
        return 0
    cfg = load_phases_config()
    ids = [p["id"] for p in cfg["phases"]]
    wanted = [ids[n - 1] for n in (args.phases or range(1, len(ids) + 1))]
    status_path = Path(paths.runs) / "m1" / args.status
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    reserve_gb = load_budget().reserve_gb
    failed = False
    for phase_id in wanted:
        done = [i for i, v in status.items() if v.get("status") == "done"]
        free = shutil.disk_usage(paths.data).free
        deficit = ingest.phase_deficit_gb(phase_id, done, free, reserve_gb)
        if deficit > 0:
            status[phase_id] = {"status": "blocked_disk", "deficit_gb": round(deficit, 2)}
            _write_status(status_path, status)
            _log(f"{phase_id}: blocked_disk, déficit {deficit:.2f} GB (livre {free / GB:.1f} GB, reserva {reserve_gb} GB)")
            return 2
        _log(f"{phase_id}: iniciando (livre {free / GB:.1f} GB)")
        t0 = time.time()
        status[phase_id] = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "free_gb_before": free / GB}
        _write_status(status_path, status)
        try:
            records = ingest.ingest_phase(ingest.phase_by_id(phase_id), paths, log=_log)
        except Exception as e:  # noqa: BLE001 - registra e segue para a próxima fase
            failed = True
            status[phase_id].update(status="failed", error=f"{type(e).__name__}: {e}"[:500],
                                    seconds=round(time.time() - t0))
            _log(f"{phase_id}: FALHOU {type(e).__name__}: {e}")
        else:
            hours = sum(r.num_samples / r.sr for r in records) / 3600
            after = shutil.disk_usage(paths.data).free
            status[phase_id].update(status="done", seconds=round(time.time() - t0), files=len(records),
                                    hours=round(hours, 2), disk_gb_used=round((free - after) / GB, 2),
                                    free_gb_after=after / GB)
            _log(f"{phase_id}: concluída, {len(records)} arquivos, {hours:.2f} h, {(free - after) / GB:.2f} GB, "
                 f"{time.time() - t0:.0f} s")
        _write_status(status_path, status)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
