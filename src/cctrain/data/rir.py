"""RIR sintéticas, 48 kHz, RT60 alvo U[0,05; 1,0] s (spec §4.2). v1: image source puro; v2: ISM + cauda estocástica.

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

from cctrain.data.manifest import FileRecord, ManifestError
from cctrain.disk import require_free
from cctrain.hashing import sha256_file

SOURCE_ID = "rir-synthetic-v1"
SOURCE_ID_V2 = "rir-synthetic-v2"
LICENSE = "LicenseRef-cctrain-generated"
ATTRIBUTION = "clearcore-train"


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
            codec="flac", attribution=ATTRIBUTION))
        params.append(p)
    return records, params


def params_to_jsonl(params: Iterable[RirParams]) -> str:
    """JSONL estrito: NaN/Inf em qualquer campo levanta `ValueError` (nunca grava `NaN`, que não é JSON)."""
    import json

    return "".join(json.dumps(asdict(p), sort_keys=True, allow_nan=False) + "\n" for p in params)


# --- synthetic-v2: image source de ordem baixa (early) + cauda estocástica (late) --------------------------------

ISM_ORDER_V2 = 3  # só as primeiras reflexões; a cauda difusa vem do modelo estocástico, sem teto de ordem 20
_FADE_IN_MAX_S = 0.01


def measure_rt60_schroeder(h: np.ndarray, fs: int) -> float:
    """RT60 por integração reversa de Schroeder: reta ajustada entre -5 e -25 dB (T20), extrapolada a 60 dB.

    NaN quando a curva não cobre a faixa (RIR nula ou curta demais).
    """
    h = np.asarray(h, dtype=np.float64)
    energy = np.cumsum((h * h)[::-1])[::-1]
    if h.size < 8 or not np.isfinite(energy[0]) or energy[0] <= 0.0:
        return float("nan")
    with np.errstate(divide="ignore"):
        edc_db = 10.0 * np.log10(energy / energy[0])
    sel = (edc_db <= -5.0) & (edc_db >= -25.0)
    if int(sel.sum()) < 8:
        return float("nan")
    t = np.arange(h.size)[sel] / fs
    slope = float(np.polyfit(t, edc_db[sel], 1)[0])
    return -60.0 / slope if slope < 0.0 and np.isfinite(slope) else float("nan")


def generate_rir_v2(seed: int, fs: int = 48_000, rt60_range: tuple[float, float] = (0.05, 1.0)
                    ) -> tuple[np.ndarray, RirParams]:
    """RIR com RT60 efetivo em `rt60_range`: ISM de ordem <= 3 + cauda gaussiana de decaimento exponencial.

    A cauda tem a energia do campo difuso relativa ao som direto, (r / r_c)^2 com r_c = 0,057 sqrt(V / RT60),
    e decai 60 dB em `rt60`. Alinhada como v1: começa 1 ms antes do caminho direto, pico 0,99. Determinística
    por `seed`; `rt60_measured` (Schroeder) pode ser NaN, e o gravador de conjuntos recusa esse caso.
    """
    rng = np.random.default_rng(seed)
    for _ in range(200):
        room = np.array([rng.uniform(3.0, 10.0), rng.uniform(3.0, 10.0), rng.uniform(2.5, 4.0)])
        rt60 = float(rng.uniform(*rt60_range))
        src = rng.uniform(0.5, room - 0.5)
        mic = rng.uniform(0.5, room - 0.5)
        dist = float(np.linalg.norm(src - mic))
        if dist < 0.5:
            continue
        volume = float(np.prod(room))
        surface = 2.0 * (room[0] * room[1] + room[0] * room[2] + room[1] * room[2])
        e_abs = float(np.clip(0.161 * volume / (surface * rt60), 0.02, 0.95))
        r = pra.ShoeBox(room, fs=fs, materials=pra.Material(e_abs), max_order=ISM_ORDER_V2, air_absorption=False)
        r.add_source(src)
        r.add_microphone(mic)
        r.compute_rir()
        early = np.asarray(r.rir[0][0], dtype=np.float64)
        direct = int(np.argmax(np.abs(early)))
        n_total = int(min(1.5 * rt60 + 0.1, 2.0) * fs)
        h = np.zeros(direct + n_total)
        h[: early.size] = early[: h.size]
        a_direct = float(abs(early[direct]))
        delta = 3.0 * np.log(10.0) / rt60  # amplitude: -60 dB em rt60
        r_c = 0.057 * np.sqrt(volume / rt60)
        e_rev = (dist / r_c) ** 2  # energia reverberante / energia direta
        sigma = a_direct * np.sqrt(e_rev * 2.0 * delta / fs)
        t = np.arange(n_total) / fs
        fade = np.minimum(1.0, t / min(_FADE_IN_MAX_S, rt60 / 5.0))
        h[direct:] += sigma * rng.standard_normal(n_total) * np.exp(-delta * t) * fade
        start = max(0, direct - fs // 1000)
        h = h[start:]
        gain = 0.99 / float(np.max(np.abs(h)))
        h = (h * gain).astype(np.float32)
        measured = float(measure_rt60_schroeder(h, fs))
        params = RirParams(seed, tuple(map(float, room)), rt60, measured, ISM_ORDER_V2, tuple(map(float, src)),
                           tuple(map(float, mic)), int(h.size), gain)
        return h, params
    raise RuntimeError(f"seed {seed}: nenhuma sala viável em 200 tentativas")


def write_rir_v2_set(out_dir: Path, seeds: Iterable[int], data_root: Path
                     ) -> tuple[list[FileRecord], list[RirParams]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, params = [], []
    for seed in seeds:
        h, p = generate_rir_v2(seed)
        if not np.isfinite(p.rt60_measured):
            raise ManifestError(f"rir seed {seed}: rt60_measured não finito ({p.rt60_measured}); registro recusado")
        path = out_dir / f"rir_{seed:05d}.flac"
        require_free(out_dir, h.size * 4)
        sf.write(path, h, 48_000, subtype="PCM_24")
        records.append(FileRecord(
            path=str(path.relative_to(data_root)), sha256=sha256_file(path), source_id=SOURCE_ID_V2,
            license=LICENSE, role="train", speaker=None, split="rir", sr=48_000, num_samples=int(h.size),
            band_hz=None, codec="flac", attribution=ATTRIBUTION))
        params.append(p)
    return records, params
