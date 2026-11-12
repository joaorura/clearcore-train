"""Matriz de avaliação do pDFNet3 Pro v4 (plano `docs/plan-pro-v4.md` §8): células, receitas determinísticas e render.

Cada item é uma função pura de `(seed, célula, índice)`: a *receita* (JSON) guarda falante, trechos de fala (arquivo,
início, tamanho), ganhos, ruído e janelas temporais; `render` reconstrói as trilhas a 48 kHz. As receitas são
congeladas por SHA-256 (`recipes.jsonl` + `manifest.json`), então todos os modelos veem exatamente as mesmas misturas.

Convenções de nível (todas em dBFS de RMS):
- alvo: RMS da fala **ativa** (VAD relativo à trilha) = `level`;
- interferente k: RMS da fala ativa = `level - SIR_k` (SIR por interferente, contra o nível ativo do alvo);
- ruído `snrX`: RMS do ruído (trilha inteira) = `level - X`; `silent`: ruído rosa de piso a -70 dBFS (absoluto);
- limitador de pico só se a soma passar de -1 dBFS (escala todas as trilhas juntas; SIR e SNR ficam intactos).
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cctrain.data.audio import load_audio, resample

SR = 48_000
HOP = 480
SEG_S = 5.0
SEG = int(SEG_S * SR)

LEVELS = (-45.0, -30.0, -18.0)
SIRS = (-10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0)
NS = (0, 1, 2, 3, 4)
NOISES = ("silent", "snr20", "snr5")
SILENT_FLOOR_DBFS = -70.0
ITEMS_PER_CELL = 12
EXTRA_ITEMS = 40
VAD_REL_DB = 30.0
VAD_FLOOR_DB = -75.0
PEAK_LIMIT = 10 ** (-1.0 / 20)
SEED = 20261110


# --- células ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    axis: str  # matrix | similar | temporal | user
    n: int
    sir: float | None
    level: float
    noise: str
    pattern: str = "overlap"  # overlap | conv | blocks

    @property
    def cid(self) -> str:
        s = "na" if self.sir is None else f"{self.sir:+.0f}"
        return f"{self.axis}|n{self.n}|sir{s}|lv{self.level:.0f}|{self.noise}|{self.pattern}"

    def views(self) -> tuple[str, ...]:
        if self.axis == "temporal":
            return ("mix",)
        v = ["mix"]
        if self.n == 0:
            v.append("noprof")
        else:
            v.append("int")
            if self.axis in ("matrix", "similar") and self.sir == 0.0 and self.n == 1:
                v.append("swap")
        return tuple(v)


def matrix_cells() -> list[Cell]:
    cells = []
    for lv in LEVELS:
        for nz in NOISES:
            cells.append(Cell("matrix", 0, None, lv, nz))
    for n in NS[1:]:
        for sir in SIRS:
            for lv in LEVELS:
                for nz in NOISES:
                    cells.append(Cell("matrix", n, sir, lv, nz))
    return cells


def extra_cells() -> list[Cell]:
    out = [Cell("similar", 1, 0.0, -30.0, "silent")]
    for n in (1, 3):
        for pat in ("conv", "blocks"):
            for nz in ("silent", "snr20"):
                out.append(Cell("temporal", n, 0.0, -30.0, nz, pat))
    return out


def user_cells() -> list[Cell]:
    out = []
    for nz in ("silent", "snr5"):
        out.append(Cell("user", 0, None, -30.0, nz))
        for n in NS[1:]:
            for sir in (-5.0, 0.0, 10.0, 20.0):
                out.append(Cell("user", n, sir, -30.0, nz))
    return out


def all_cells() -> list[Cell]:
    return matrix_cells() + extra_cells() + user_cells()


def n_items(cell: Cell) -> int:
    if cell.axis == "matrix" or cell.axis == "user":
        return ITEMS_PER_CELL
    if cell.axis == "similar":
        return EXTRA_ITEMS
    return EXTRA_ITEMS // 8  # temporal: 8 células (N 1/3 x conversa/blocos x 2 ruídos) x 5 itens = 40


def rng_for(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


# --- áudio --------------------------------------------------------------------------------------------------------


class _LRU(OrderedDict):
    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap

    def fetch(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        v = fn()
        self[key] = v
        if len(self) > self.cap:
            self.popitem(last=False)
        return v


_AUDIO = _LRU(160)
DATA_ROOT: Path | None = None
USER_STREAM_TAG = "@user_stream"
USER_STREAM_PATH: Path | None = None


def set_roots(data_root: Path, user_stream: Path | None = None) -> None:
    global DATA_ROOT, USER_STREAM_PATH
    DATA_ROOT = Path(data_root)
    USER_STREAM_PATH = Path(user_stream) if user_stream else None


def load48(rel: str) -> np.ndarray:
    def _load():
        if rel == USER_STREAM_TAG:
            assert USER_STREAM_PATH is not None, "set_roots(user_stream=...) ausente"
            x, sr = load_audio(USER_STREAM_PATH)
        else:
            x, sr = load_audio(DATA_ROOT / rel)
        return resample(x, sr, SR)

    return _AUDIO.fetch(rel, _load)


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    fr = np.asarray(x[: n * HOP], np.float64).reshape(n, HOP)
    return 10 * np.log10((fr * fr).mean(axis=1) + 1e-12)


def activity(x: np.ndarray, rel_db: float = VAD_REL_DB, floor_db: float = VAD_FLOOR_DB) -> np.ndarray:
    """VAD relativo à trilha, por quadro de 10 ms: nível >= max(pico da trilha - 30 dB, -75 dBFS)."""
    lv = frame_db(x)
    if lv.size == 0 or not np.isfinite(lv).any() or lv.max() < floor_db:
        return np.zeros(lv.size, bool)
    return lv >= max(lv.max() - rel_db, floor_db)


def active_rms_db(x: np.ndarray) -> float:
    act = activity(x)
    n = len(act) * HOP
    ms = np.asarray(x[:n], np.float64).reshape(-1, HOP) ** 2
    ms = ms.mean(axis=1)
    sel = ms[act] if act.any() else ms
    return float(10 * np.log10(sel.mean() + 1e-20))


def scale_to_level(x: np.ndarray, level_db: float) -> np.ndarray:
    return np.asarray(x, np.float64) * 10 ** ((level_db - active_rms_db(x)) / 20)


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20))


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    freqs[0] = freqs[1]
    f /= np.sqrt(freqs / freqs[1])
    return np.fft.irfft(f, n)


def window_mask(n: int, windows_s: list[list[float]], fade_s: float = 0.01) -> np.ndarray:
    m = np.zeros(n)
    fade = int(fade_s * SR)
    for a, b in windows_s:
        i, j = max(int(a * SR), 0), min(int(b * SR), n)
        if j <= i:
            continue
        seg = np.ones(j - i)
        k = min(fade, (j - i) // 2)
        if k > 0:
            ramp = np.linspace(0, 1, k, endpoint=False)
            seg[:k] = ramp
            seg[-k:] = ramp[::-1]
        m[i:j] = np.maximum(m[i:j], seg)
    return m


def speech_track(segs: list[dict]) -> np.ndarray:
    parts = []
    for s in segs:
        if s["path"] is None:
            parts.append(np.zeros(int(s["n"]), np.float32))
        else:
            x = load48(s["path"])
            a = int(s["start"])
            seg = x[a : a + int(s["n"])]
            if len(seg) < s["n"]:
                seg = np.pad(seg, (0, int(s["n"]) - len(seg)))
            parts.append(seg)
    return np.concatenate(parts).astype(np.float32)[:SEG]


# --- render --------------------------------------------------------------------------------------------------------


@dataclass
class Rendered:
    target: np.ndarray  # [T] f32 (trilha do alvo já no nível e com a janela temporal)
    interf: np.ndarray  # [N, T] f32
    noise: np.ndarray  # [T] f32
    tgt_act: np.ndarray  # [frames] bool
    int_act: np.ndarray  # [frames] bool (OR dos interferentes)

    def noisy(self, with_target: bool = True) -> np.ndarray:
        x = self.noise.astype(np.float64) + self.interf.astype(np.float64).sum(axis=0)
        if with_target:
            x = x + self.target
        return x.astype(np.float32)


def render(rec: dict) -> Rendered:
    cell = rec["cell"]
    level = float(cell["level"])
    tgt = speech_track(rec["target"]["segs"]).astype(np.float64)
    if rec["target"].get("windows") is not None:
        tgt = tgt * window_mask(SEG, rec["target"]["windows"])
    tgt = scale_to_level(tgt, level)
    ints = []
    for it in rec["interferers"]:
        x = speech_track(it["segs"]).astype(np.float64)
        if it.get("windows") is not None:
            x = x * window_mask(SEG, it["windows"])
        ints.append(scale_to_level(x, level - float(it["sir_db"])))
    nz = rec["noise"]
    if nz["kind"] == "none":
        noise = np.zeros(SEG)
    elif nz["kind"] == "pink":
        noise = pink_noise(SEG, rng_for("pink", rec["id"]))
        noise *= 10 ** ((SILENT_FLOOR_DBFS - rms_db(noise)) / 20)
    else:
        x = load48(nz["path"])
        a = int(nz["start"])
        reps = int(np.ceil((a + SEG) / len(x))) + 1
        x = np.tile(x, reps)[a : a + SEG].astype(np.float64)
        noise = x * 10 ** (((level - float(nz["snr_db"])) - rms_db(x)) / 20)
    ints_a = np.stack(ints) if ints else np.zeros((0, SEG))
    mix_peak = np.max(np.abs(tgt + ints_a.sum(axis=0) + noise))
    g = min(1.0, PEAK_LIMIT / max(mix_peak, 1e-9))
    tgt, ints_a, noise = tgt * g, ints_a * g, noise * g
    tgt_act = activity(tgt)
    int_act = np.zeros_like(tgt_act)
    for x in ints_a:
        int_act = int_act | activity(x)
    return Rendered(tgt.astype(np.float32), ints_a.astype(np.float32), noise.astype(np.float32), tgt_act, int_act)


# --- banco de falantes ---------------------------------------------------------------------------------------------


@dataclass
class Bank:
    keys: list[str]
    sex: dict[str, str | None]
    corpus: dict[str, str]
    utts: dict[str, list[dict]]  # {path, n, sr}
    emb: np.ndarray  # [S, 192]
    noise: list[str]  # caminhos relativos (ruído de dev, nunca visto no treino)

    def idx(self, key: str) -> int:
        return self.keys.index(key)

    def to_json(self) -> dict:
        return {"keys": self.keys, "sex": self.sex, "corpus": self.corpus, "utts": self.utts, "noise": self.noise}

    @staticmethod
    def from_json(d: dict, emb: np.ndarray) -> "Bank":
        return Bank(d["keys"], d["sex"], d["corpus"], d["utts"], emb, d["noise"])

    def neighbors(self, key: str, k: int = 10, same_sex: bool = True) -> list[str]:
        i = self.idx(key)
        sims = self.emb @ self.emb[i]
        order = [self.keys[j] for j in np.argsort(-sims) if j != i]
        if same_sex and self.sex.get(key):
            order = [s for s in order if self.sex.get(s) == self.sex[key]]
        return order[:k]


def speech_segs(bank: Bank, spk: str, rng: np.random.Generator, total: int = SEG) -> list[dict]:
    """Trechos de fala de `spk` somando `total` amostras a 48 kHz (uma fala longa ou várias com pausas de 0,2 s)."""
    pool = bank.utts[spk]
    segs, have = [], 0
    for k in rng.permutation(len(pool)):
        u = pool[int(k)]
        n48 = int(round(u["n"] * SR / u["sr"]))
        need = total - have
        if n48 >= need:
            start = int(rng.integers(0, n48 - need + 1))
            segs.append({"path": u["path"], "start": start, "n": need})
            have = total
            break
        segs.append({"path": u["path"], "start": 0, "n": n48})
        have += n48
        pause = min(int(0.2 * SR), total - have)
        if pause > 0:
            segs.append({"path": None, "start": 0, "n": pause})
            have += pause
        if have >= total:
            break
    if have < total:
        segs.append({"path": None, "start": 0, "n": total - have})
    return segs


def temporal_windows(pattern: str, n_int: int, rng: np.random.Generator) -> tuple[list, list[list]]:
    """Janelas (s) de atividade do alvo e de cada interferente, em 5 s, para conversa e blocos."""
    tw: list = []
    iw: list[list] = [[] for _ in range(n_int)]
    if pattern == "blocks":
        # só-alvo, ambos, só-interferente em ordem aleatória, com lacuna de silêncio comum entre blocos
        kinds = list(rng.permutation(["t", "b", "i"]))
        gap, blk = 0.25, (SEG_S - 2 * 0.25) / 3
        t0 = 0.0
        for kd in kinds:
            a, b = t0, t0 + blk
            if kd in ("t", "b"):
                tw.append([a, b])
            if kd in ("i", "b"):
                for w in iw:
                    w.append([a, b])
            t0 = b + gap
        return tw, iw
    # conversa: turnos de 0,5 a 2 s, sobreposição de 0 a 30 % em cada troca
    t, who = 0.0, int(rng.integers(0, n_int + 1))
    while t < SEG_S:
        dur = float(rng.uniform(0.5, 2.0))
        end = min(t + dur, SEG_S)
        (tw if who == 0 else iw[who - 1]).append([t, end])
        if end >= SEG_S:
            break
        nxt = int(rng.integers(0, n_int + 1))
        while nxt == who:
            nxt = int(rng.integers(0, n_int + 1))
        t = end - float(rng.uniform(0.0, 0.3)) * dur
        who = nxt
    return tw, iw


def make_recipe(bank: Bank, cell: Cell, idx: int, *, user: dict | None = None, attempt: int = 0) -> dict:
    """Receita do item `idx` da célula (determinística). `user`: alvo = trilha do usuário real (ver `user_windows`)."""
    rng = rng_for("v4eval", SEED, cell.cid, idx, attempt)
    rid = f"{cell.cid}#{idx:03d}"
    if user is not None:
        tkey = user["key"]
        win = user["windows"][idx % len(user["windows"])]
        tsegs = [{"path": USER_STREAM_TAG, "start": int(win), "n": SEG}]
        enroll_row = idx % int(user["embs_n"])
        profile = {"kind": "user", "row": enroll_row}
    else:
        order = rng_for("order", SEED, cell.axis, cell.n, cell.level, cell.noise).permutation(len(bank.keys))
        tkey = bank.keys[int(order[idx % len(order)])]
        tsegs = speech_segs(bank, tkey, rng)
        profile = {"kind": "bank", "spk": tkey}
    cand = [s for s in bank.keys if s != tkey]
    if cell.axis == "similar":
        near = bank.neighbors(tkey, 10, same_sex=True) or bank.neighbors(tkey, 10, same_sex=False)
        cand = near
    ispks = [cand[int(k)] for k in rng.permutation(len(cand))[: cell.n]]
    ints = [{"spk": s, "segs": speech_segs(bank, s, rng), "sir_db": float(cell.sir)} for s in ispks]
    windows_t = None
    if cell.axis == "temporal":
        windows_t, iw = temporal_windows(cell.pattern, cell.n, rng)
        for it, w in zip(ints, iw, strict=True):
            it["windows"] = w
    if cell.noise == "silent":
        noise = {"kind": "pink"}
    else:
        snr = float(cell.noise.removeprefix("snr"))
        path = bank.noise[int(rng.integers(0, len(bank.noise)))]
        noise = {"kind": "file", "path": path, "start": int(rng.integers(0, 10 * SR)), "snr_db": snr}
    return {"id": rid, "cell": asdict(cell) | {"cid": cell.cid}, "idx": idx, "target": {"spk": tkey, "segs": tsegs,
            "windows": windows_t}, "interferers": ints, "noise": noise, "profile": profile,
            "swap_spk": ispks[0] if ispks else None}


def recipes_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_recipes(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
