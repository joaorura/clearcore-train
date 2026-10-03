"""Fator FLAC/PCM por taxa de amostragem e tamanho dos ambientes (substitui os [est.] da spec §4.2)."""

from __future__ import annotations

import io
import os
import shutil
import stat
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
    """Soma dos tamanhos dos arquivos regulares, contando cada inode uma vez (hardlinks do target do cargo)."""
    total = 0
    seen: set[tuple[int, int]] = set()
    for root, _, files in os.walk(path):
        for f in files:
            st = os.lstat(os.path.join(root, f))
            if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            total += st.st_size
    return total


def disk_report(paths: Paths, records: Iterable[FileRecord]) -> dict:
    records = list(records)
    # O target do cargo e o venv vivem no checkout principal (CCTRAIN_ROOT), não no worktree (paths.repo).
    main_root = Path(os.environ.get("CCTRAIN_ROOT", paths.repo))
    env_dirs = {
        "venv": Path(os.environ.get("UV_PROJECT_ENVIRONMENT", main_root / ".venv")),
        "tract_check_target": main_root / "tools/tract-check/target",
        "rustup": Path(os.environ.get("RUSTUP_HOME", Path.home() / ".rustup")),
        "data": paths.data,
        "third_party": paths.repo / "third_party",
    }
    rir = [r for r in records if r.split == "rir"]
    audio = [r for r in records if r.split != "rir"]
    by_source: dict[str, list[FileRecord]] = defaultdict(list)
    for r in audio:
        by_source[r.source_id].append(r)
    return {
        "by_sr": flac_ratio_by_sr(audio, paths.data),
        # fator representativo por fonte (cada fonte tem um perfil de codec/conteúdo; ex.: VCTK = fala 48 kHz em FLAC)
        "by_source": {sid: flac_ratio_by_sr(recs, paths.data) for sid, recs in sorted(by_source.items())},
        "tract_check_target_root": str(main_root),
        "rir": {"count": len(rir), "stored_bytes": sum((paths.data / r.path).stat().st_size for r in rir)},
        "dirs_bytes": {k: dir_size(v) if v.exists() else None for k, v in env_dirs.items()},
        "free_bytes": shutil.disk_usage(paths.data).free,
    }
