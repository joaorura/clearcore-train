"""RIR sintéticas por image source (pyroomacoustics), 48 kHz, RT60 alvo U[0,05; 1,0] s (spec §4.2).

Cada RIR é cortada para começar 1 ms antes do caminho direto (o alvo seco fica alinhado com a mistura),
limitada a 1,2 s e normalizada para pico 0,99. max_order é limitado a 20 (custo); o RT60 medido é registrado.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf

from cctrain.data.manifest import FileRecord
from cctrain.disk import require_free
from cctrain.hashing import sha256_file

SOURCE_ID = "rir-synthetic-v1"
LICENSE = "LicenseRef-cctrain-generated"


@dataclass(frozen=True)
class RirParams:
    seed: int
    room_dim: tuple[float, float, float]
    rt60_target: float
    rt60_measured: float
    max_order: int
    source: tuple[float, float, float]
    mic: tuple[float, float, float]
    length: int
    gain: float


def generate_rir(seed: int, fs: int = 48_000, rt60_range: tuple[float, float] = (0.05, 1.0),
                 max_order_cap: int = 20, max_len_s: float = 1.2) -> tuple[np.ndarray, RirParams]:
    rng = np.random.default_rng(seed)
    for _ in range(200):
        room = np.array([rng.uniform(3.0, 10.0), rng.uniform(3.0, 10.0), rng.uniform(2.5, 4.0)])
        rt60 = float(rng.uniform(*rt60_range))
        src = rng.uniform(0.5, room - 0.5)
        mic = rng.uniform(0.5, room - 0.5)
        if np.linalg.norm(src - mic) < 0.5:
            continue
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, room)
        except ValueError:
            continue
        order = int(min(max_order, max_order_cap))
        r = pra.ShoeBox(room, fs=fs, materials=pra.Material(e_abs), max_order=order, air_absorption=False)
        r.add_source(src)
        r.add_microphone(mic)
        r.compute_rir()
        h = np.asarray(r.rir[0][0], dtype=np.float64)
        start = max(0, int(np.argmax(np.abs(h))) - fs // 1000)
        h = h[start : start + int(max_len_s * fs)]
        gain = 0.99 / float(np.max(np.abs(h)))
        h = (h * gain).astype(np.float32)
        try:
            measured = float(pra.experimental.measure_rt60(h, fs=fs))
        except Exception:  # noqa: BLE001 - medida informativa; RIR curta pode não decair 60 dB
            measured = float("nan")
        params = RirParams(seed, tuple(map(float, room)), rt60, measured, order, tuple(map(float, src)),
                           tuple(map(float, mic)), int(h.size), gain)
        return h, params
    raise RuntimeError(f"seed {seed}: nenhuma sala viável em 200 tentativas")


def write_rir_set(out_dir: Path, seeds: Iterable[int], data_root: Path) -> tuple[list[FileRecord], list[RirParams]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, params = [], []
    for seed in seeds:
        h, p = generate_rir(seed)
        path = out_dir / f"rir_{seed:05d}.flac"
        require_free(out_dir, h.size * 4)
        sf.write(path, h, 48_000, subtype="PCM_24")
        records.append(FileRecord(
            path=str(path.relative_to(data_root)), sha256=sha256_file(path), source_id=SOURCE_ID, license=LICENSE,
            role="train", speaker=None, split="rir", sr=48_000, num_samples=int(h.size), band_hz=None,
            codec="flac"))
        params.append(p)
    return records, params


def params_to_jsonl(params: Iterable[RirParams]) -> str:
    import json

    return "".join(json.dumps(asdict(p), sort_keys=True) + "\n" for p in params)
