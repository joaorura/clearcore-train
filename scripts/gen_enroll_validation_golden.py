"""Gera tests/golden/enroll_validation_v1.json rodando a validação do Clearcore (Rust, std puro) sobre os casos
de tests/enroll_cases.py. Precisa de rustc (toolchain 1.90.0). Uso: scripts/gen_enroll_validation_golden.py"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from enroll_cases import cases  # noqa: E402


def main() -> None:
    work = ROOT / "build" / "enroll-golden"
    work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "RUSTUP_TOOLCHAIN": "1.90.0"}
    exe = work / "enroll_validate_ref"
    subprocess.run(["rustc", "-O", "-o", str(exe), str(ROOT / "scripts" / "enroll_validate_ref.rs")], check=True,
                   env=env)
    paths = []
    for name, x in cases().items():
        p = work / f"{name}.f32"
        np.asarray(x, "<f4").tofile(p)
        paths.append(str(p))
    out = subprocess.run([str(exe), *paths], check=True, capture_output=True, text=True, env=env).stdout
    golden = {}
    for line in out.splitlines():
        row = json.loads(line)
        golden[row.pop("case")] = row
    dest = ROOT / "tests" / "golden" / "enroll_validation_v1.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps({"source": "clearcore crates/model/src/enrollment.rs @ 818e7ce",
                                "cases": golden}, indent=1, sort_keys=True) + "\n")
    print(f"{len(golden)} casos -> {dest}")


if __name__ == "__main__":
    main()
