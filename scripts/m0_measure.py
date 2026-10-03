"""Medições do M0. Uso: m0_measure.py {env,disk,loader,step,beta,offline,all}. Grava em $CCTRAIN_RUNS_DIR/m0/."""

import json
import platform
import subprocess
import sys

import numpy as np
import psutil
import torch

from cctrain.data.loader import PilotLoaderConfig, PilotMixDataset, pilot_records
from cctrain.data.manifest import FILES_DIR, read_file_manifest
from cctrain.data.pilot import PILOT_MANIFESTS
from cctrain.m0 import beta_probe, disk_probe, loader_probe, offline_probe, step_probe
from cctrain.m0.gpu_monitor import GpuMonitor
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmConfig, FilmGenerator
from cctrain.models.pdfnet3 import PDFNet3
from cctrain.paths import Paths, clearcore_root

PATHS = Paths.from_env()
OUT = PATHS.runs / "m0"


def _save(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str) + "\n")
    print(name, json.dumps(obj, default=str)[:400])


def _dataset() -> PilotMixDataset:
    speech, noise, rirs = pilot_records(PATHS.data)
    return PilotMixDataset(speech, noise, rirs, PATHS.data, PilotLoaderConfig())


def _batches(n_batches: int, batch_size: int) -> list[dict]:
    it = iter(_dataset())
    out = []
    for _ in range(n_batches):
        items = [next(it) for _ in range(batch_size)]
        out.append({k: torch.as_tensor(np.stack([i[k] for i in items])) for k in
                    ("spec", "spec_clean", "feat_erb", "feat_spec")})
    return out


def env() -> None:
    vm = psutil.virtual_memory()
    smi = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, check=False).stdout
    _save("env", {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "ram_total_gb": vm.total / 1024**3, "ram_available_gb": vm.available / 1024**3,
                  "swap_used_gb": psutil.swap_memory().used / 1024**3, "nvidia_smi_q_head": smi[:4000]})


def disk() -> None:
    records = [r for p in PILOT_MANIFESTS.values() for r in read_file_manifest(p)]
    records += read_file_manifest(FILES_DIR / "rir-synthetic-v1-pilot.jsonl")
    records += read_file_manifest(FILES_DIR / "noise-synthetic-v1-pilot.jsonl")
    cv = PATHS.data / "manifests" / "cv27-pt.jsonl"
    if cv.is_file():
        records += read_file_manifest(cv)
    _save("disk", disk_probe.disk_report(PATHS, records))


def loader() -> None:
    _save("loader", loader_probe.probe_loader(_dataset()))


def step() -> None:
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM livre {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")
    for name, bsz, with_teacher in (("step_b16", 16, False), ("step_b32", 32, False), ("step_b16_teacher", 16, True)):
        base, _ = load_upstream_dfnet3(device="cpu")
        model = PDFNet3(base, FilmGenerator(FilmConfig(2.302585092994046, 4.0)))
        teacher = load_upstream_dfnet3(device="cuda")[0] if with_teacher else None
        batch = _batches(1, bsz)[0]
        try:
            with GpuMonitor() as mon:
                result = step_probe.probe_step(model, batch, device="cuda", teacher=teacher)
            result["gpu"] = mon.summary()
        except torch.cuda.OutOfMemoryError as e:
            result = {"batch": bsz, "teacher": with_teacher, "oom": True, "error": str(e)[:500]}
        _save(name, result)
        del model, teacher, base
        torch.cuda.empty_cache()


def beta() -> None:
    base, _ = load_upstream_dfnet3(device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(base.parameters()).device.type
    acts = beta_probe.collect_site_activations(base, _batches(25, 8), device)
    report = beta_probe.beta_rule_report(acts)
    report["samples"] = 25 * 8
    _save("beta_rule", report)
    beta_probe.write_beta_bound(report["beta_bound"])


def offline() -> None:
    from cctrain.eval.runtime_path import StatefulOrtFrameModel, parity_mixture, run_replica

    x = parity_mixture(PATHS.data)
    replica = run_replica(x, StatefulOrtFrameModel(clearcore_root() / "models/stateful"))
    base, _ = load_upstream_dfnet3(device="cpu")
    _save("offline_vs_runtime", offline_probe.offline_vs_runtime(base, x, replica))


def main(which: str) -> None:
    steps = {"env": env, "disk": disk, "loader": loader, "step": step, "beta": beta, "offline": offline}
    for name, fn in steps.items():
        if which in (name, "all"):
            fn()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
