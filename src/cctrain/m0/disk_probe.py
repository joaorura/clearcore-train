"""Fator FLAC/PCM por taxa de amostragem e tamanho dos ambientes (substitui os [est.] da spec §4.2)."""

from __future__ import annotations

import io
import os
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import soundfile as sf

from cctrain.data.audio import load_audio
from cctrain.data.manifest import FileRecord
from cctrain.paths import Paths


def flac_ratio_by_sr(records: Iterable[FileRecord], data_root: Path) -> dict[str, dict]:
    acc: dict[int, dict] = defaultdict(lambda: {"files": 0, "seconds": 0.0, "pcm16_bytes": 0, "flac_bytes": 0,
                                                "stored_bytes": 0, "codecs": set()})
    for r in records:
        a = acc[r.sr]
        path = Path(data_root) / r.path
        a["files"] += 1
        a["seconds"] += r.num_samples / r.sr
        a["pcm16_bytes"] += r.num_samples * 2
        a["stored_bytes"] += path.stat().st_size
        a["codecs"].add(r.codec)
        if r.codec == "flac":
            a["flac_bytes"] += path.stat().st_size
        else:
            x, sr = load_audio(path)
            buf = io.BytesIO()
            sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
            a["flac_bytes"] += len(buf.getvalue())
    out = {}
    for sr, a in sorted(acc.items()):
        hours = a["seconds"] / 3600.0
        out[str(sr)] = {
            "files": a["files"], "hours": hours, "pcm16_bytes": a["pcm16_bytes"], "flac_bytes": a["flac_bytes"],
            "stored_bytes": a["stored_bytes"], "codecs": sorted(a["codecs"]),
            "flac_over_pcm16": a["flac_bytes"] / a["pcm16_bytes"],
            "flac_gb_per_hour": a["flac_bytes"] / 1e9 / hours if hours else float("nan"),
            "stored_gb_per_hour": a["stored_bytes"] / 1e9 / hours if hours else float("nan"),
        }
    return out


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def disk_report(paths: Paths, records: Iterable[FileRecord]) -> dict:
    records = list(records)
    env_dirs = {
        "venv": Path(os.environ.get("UV_PROJECT_ENVIRONMENT", paths.repo / ".venv")),
        "tract_check_target": paths.repo / "tools/tract-check/target",
        "rustup": Path(os.environ.get("RUSTUP_HOME", Path.home() / ".rustup")),
        "data": paths.data,
        "third_party": paths.repo / "third_party",
    }
    rir = [r for r in records if r.split == "rir"]
    return {
        "by_sr": flac_ratio_by_sr([r for r in records if r.split != "rir"], paths.data),
        "rir": {"count": len(rir), "stored_bytes": sum((paths.data / r.path).stat().st_size for r in rir)},
        "dirs_bytes": {k: dir_size(v) if v.exists() else None for k, v in env_dirs.items()},
        "free_bytes": shutil.disk_usage(paths.data).free,
    }
