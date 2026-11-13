"""Gerador de misturas do pDFNet3 Pro v4 (`docs/plan-pro-v4.md` §2.3): 1 alvo + N∈{0..4} interferentes.

Cada amostra é função pura de `(seed, índice)` (o índice global já codifica o passo). Resumo da receita:

- N com P = 0,25/0,35/0,20/0,12/0,08 (fase 1: só N∈{0,1,2});
- SIR por interferente contra o nível ATIVO do alvo: 65% U[−10, 15], 15% U[−15, −10], 20% U[15, 30] (fase 1: U[−10, 20]);
- nível do alvo (RMS de fala ativa) U[−50, −12] dBFS, 10% em [−50, −40] (metade com inclinação espectral de
  sussurro); limitador de pico só acima de −1 dBFS (nunca normaliza para cima);
- ruído: 20% silencioso (piso de microfone −80..−60 dBFS, rosa ou branco), 30% SNR U[20, 40], 35% U[5, 20],
  15% U[−5, 5] (fase 3: silencioso 30%);
- RIR com p=0,5 por mistura, uma RIR independente por falante; alvo da perda = caminho direto + 50 ms;
- padrão temporal: 35% contínuo, 30% conversa (turnos 0,5–2 s, sobreposição 0–30%), 15% blocos, 20% esparso;
- 8% alvo ausente (N≥1, perfil de quem não fala, nível do interferente pela regra do nível do alvo);
- 35% dos interferentes entre os 50 vizinhos de cosseno do centróide do alvo (fase 1: 20%, fase 3: 50%);
- rótulos por quadro com VAD relativo à trilha já escalonada (até 30 dB do máximo, piso −75 dBFS).

Desvios do plano (registrados no relatório do v4): RIR v3 multi-fonte não existe ainda, então cada falante recebe
uma RIR sorteada do pool v2 (salas diferentes); os negativos difíceis usam só os falantes do banco (centróide do
banco); o critério de mesmo sexo não é aplicado; os interferentes extras do CV são os falantes novos (≥5 clipes,
fora de dev/teste), sem os clipes extras dos 603 já usados.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, oaconvolve
from torch.utils.data import Dataset

from cctrain.data.audio import load_audio, resample
from cctrain.data.features import SR, analysis, extract_features
from cctrain.data.manifest import FileRecord, read_file_manifest
from cctrain.data.mix import crop_or_pad, loop_to_length
from cctrain.train.embed_bank import EmbeddingBankError
from cctrain.train.stage_f import TIERS, lowpass48, slot_tiers, tier_of

HOP = 480
NYQUIST_HZ = 24_000.0
N_PROBS = (0.25, 0.35, 0.20, 0.12, 0.08)
PATTERN_PROBS = (0.35, 0.30, 0.15, 0.20)
EARLY_MS = 50.0
PEAK_LIMIT = 10 ** (-1.0 / 20.0)
SILENCE_MS = 1e-7
MAX_REDRAWS = 8

# Held-out (teste do usuário): fora do treino em qualquer papel (alvo, interferente, enrollment, ruído).
HELDOUT_SPEECH_SUBSTR = ("speech-joao-messias/joao_messias_take3_", "speech-joao-messias/joao_messias_take4_")
# take3 e take4 são a mesma fala gravada ao mesmo tempo em dois canais (08-32-48 / 08-32-51): ambos ficam fora.
HELDOUT_NOISE_SUBSTR: tuple[str, ...] = ()  # a gravação de 2026-10-08 é ruído de escritório, não fala do usuário


def is_heldout_speech(path: str) -> bool:
    return any(s in path for s in HELDOUT_SPEECH_SUBSTR)


def is_heldout_noise(path: str) -> bool:
    return any(s in path for s in HELDOUT_NOISE_SUBSTR)


@dataclass(frozen=True)
class V4Params:
    seed: int = 20261110
    segment_s: float = 4.0
    batch_size: int = 16
    total_steps: int = 80_000
    phase1_frac: float = 0.15
    phase3_frac: float = 0.85
    p_drop: float = 0.15
    p_target_absent: float = 0.08
    p_rir: float = 0.5
    p_enroll_denoise: float = 0.5
    n_neighbors: int = 50
    band_quota: tuple[float, float, float] = (0.5, 0.3, 0.2)
    # P(N=0..4); pro-v4-presence sobe o N=0 (mais amostras para o L_guard). Fora do hash quando é o padrão, para os
    # checkpoints do pro-v4 continuarem retomáveis.
    n_probs: tuple[float, float, float, float, float] = N_PROBS
    # pro-v4-keep: P(contínuo, conversa, blocos, esparso) e P(o interferente abrir a conversa / o bloco só-interferente
    # vir antes do alvo), para haver mais entradas do alvo logo após o interferente. Fora do hash quando é o padrão.
    pattern_probs: tuple[float, float, float, float] = PATTERN_PROBS
    p_interf_first: float = 0.0
    # pro-v42: P(padrão "turnos" nas amostras com N>=1): alvo e interferente se revezam SEM sobreposição, com pausas
    # U[0, turn_gap_max_s] e RIR com p=p_rir_turns (caudas reverberantes do interferente sobre o início do alvo). Fora
    # do hash quando 0 (runs antigos retomáveis).
    p_turns: float = 0.0
    turn_gap_max_s: float = 0.5
    p_rir_turns: float = 0.8

    def hashable(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        if tuple(d["n_probs"]) == N_PROBS:
            d.pop("n_probs")
        if tuple(d["pattern_probs"]) == PATTERN_PROBS:
            d.pop("pattern_probs")
        if d["p_interf_first"] == 0.0:
            d.pop("p_interf_first")
        if d["p_turns"] == 0.0:
            for k in ("p_turns", "turn_gap_max_s", "p_rir_turns"):
                d.pop(k)
        return d


# --- utilidades de sinal -------------------------------------------------------------------------------------------


def _conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolução com a RIR, truncada ao comprimento de `x`. FFT em float32 (≈3x mais rápida que float64; erro relativo
    ao pico ≈2e-7, ~-134 dB, abaixo de qualquer piso de ruído da receita); a saída volta a float64 para o restante da
    mistura (ganhos, soma, VAD) manter os dtypes de sempre."""
    n = len(x)
    y = oaconvolve(np.asarray(x, np.float32), np.asarray(h, np.float32))[:n]
    return y.astype(np.float64)


def early_rir(h: np.ndarray, ms: float = EARLY_MS) -> np.ndarray:
    peak = int(np.argmax(np.abs(h)))
    return h[: peak + int(ms * SR / 1000.0) + 1]


def frame_levels(x: np.ndarray, n_frames: int) -> np.ndarray:
    f = np.asarray(x[: n_frames * HOP], np.float64).reshape(n_frames, HOP)
    return 10 * np.log10(np.mean(f**2, axis=1) + 1e-20)


def rel_activity(x: np.ndarray, n_frames: int, *, range_db: float = 30.0, floor_dbfs: float = -75.0) -> np.ndarray:
    """VAD relativo à trilha já escalonada (§2.3): a até `range_db` do máximo da trilha e acima do piso."""
    lv = frame_levels(x, n_frames)
    mx = float(lv.max())
    if mx <= floor_dbfs:
        return np.zeros(n_frames, bool)
    return lv >= max(mx - range_db, floor_dbfs)


def active_rms_db(x: np.ndarray, n_frames: int) -> float:
    act = rel_activity(x, n_frames, floor_dbfs=-200.0)
    if not act.any():
        return -200.0
    f = np.asarray(x[: n_frames * HOP], np.float64).reshape(n_frames, HOP)[act]
    return float(10 * np.log10(np.mean(f**2) + 1e-20))


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n)
    spec = np.fft.rfft(w)
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / (np.sqrt(np.mean(x**2)) + 1e-20)


def whisper_tilt(x: np.ndarray) -> np.ndarray:
    """Polo único em 1 kHz: ~−6 dB/oit acima de 1 kHz (aproximação de sussurro/longe)."""
    a = math.exp(-2 * math.pi * 1000.0 / SR)
    y = lfilter([1 - a], [1, -a], np.asarray(x, np.float64))
    return y * (np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) / (np.sqrt(np.mean(y**2)) + 1e-20))


def _ramp_env(on_frames: np.ndarray, n: int, ramp: int = 480) -> np.ndarray:
    """Envelope por amostra (0/1 por quadro) com rampas de 10 ms."""
    env = np.repeat(on_frames.astype(np.float64), HOP)[:n]
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    if ramp > 1:
        if n < ramp:  # janela maior que o sinal: o 'same' do numpy troca os papéis; caminho original
            env = np.convolve(env, np.ones(ramp) / ramp, mode="same")
        else:
            # Média móvel por soma acumulada, O(n) em vez de O(n·ramp). Mesmo alinhamento do `np.convolve(..., "same")`
            # com núcleo de `ramp` pontos: saída[i] = média de env[i-(ramp-1-(ramp-1)//2) .. i+(ramp-1)//2], fora = 0
            # (ramp=480: janela [i-240, i+239]). Os valores 0/1 somam exatamente em float64.
            right = (ramp - 1) // 2
            left = ramp - 1 - right
            c = np.zeros(n + ramp)
            np.cumsum(env, out=c[left + 1 : left + 1 + n])
            c[left + 1 + n :] = c[left + n]
            env = (c[ramp : ramp + n] - c[:n]) / ramp
    return np.clip(env, 0.0, 1.0)


PATTERNS = ("continuous", "conversation", "blocks", "sparse", "turns")


def turns_mask(rng: np.random.Generator, n_tracks: int, frames: int, p_interf_first: float,
               gap_max_s: float) -> np.ndarray:
    """pro-v42: turnos alternados alvo/interferente sem sobreposição, pausas U[0, gap_max_s]. Turno do interferente
    U[0,4, 2,0] s (um interferente sorteado por turno), do alvo U[0,6, 3,0] s (turnos longos cobrem a recuperação
    > 2 s depois do interferente)."""
    fps = SR / HOP
    on = np.zeros((n_tracks, frames), bool)
    t = 0.0
    tgt_turn = not (rng.random() < max(p_interf_first, 0.5))
    ni = n_tracks - 1
    nxt = int(rng.integers(ni))  # interferentes em rodízio a partir de um sorteado
    turns: list[tuple[int, int]] = []
    while t < frames:
        if tgt_turn:
            k, d = 0, rng.uniform(0.6, 3.0) * fps
        else:
            k, d = 1 + nxt, rng.uniform(0.4, 2.0) * fps
            nxt = (nxt + 1) % ni
        a, b = int(t), int(min(frames, t + d))
        on[k, a:b] = True
        if k:
            turns.append((a, b))
        t += d + rng.uniform(0.0, gap_max_s) * fps
        tgt_turn = not tgt_turn
    if not on[0].any():  # item curto que só teve turno do interferente: o alvo entra no fim
        on[0, max(0, frames - int(fps)):] = True
        on[1:, max(0, frames - int(fps)):] = False
    for k in range(1, n_tracks):  # interferente sem turno: fala junto com um turno de outro interferente
        if not on[k].any():
            if turns:
                a, b = turns[int(rng.integers(len(turns)))]
            else:
                a, b = 0, min(frames, int(fps))
                on[0, a:b] = False
            on[k, a:b] = True
    return on


def temporal_masks(rng: np.random.Generator, n_tracks: int, frames: int, target_present: bool,
                   pattern_probs: Sequence[float] = PATTERN_PROBS, p_interf_first: float = 0.0,
                   p_turns: float = 0.0, turn_gap_max_s: float = 0.5) -> tuple[str, np.ndarray]:
    """Padrão temporal (§2.3). Devolve (nome, on[n_tracks, frames]); a trilha 0 é o alvo.

    `p_interf_first` > 0 (pro-v4-keep): com essa probabilidade a conversa começa por um interferente e, nos blocos, o
    bloco só-interferente vem antes de um bloco com alvo (entradas do alvo logo após o interferente). Com 0 a sequência
    de sorteios é a de sempre (runs antigos retomáveis)."""
    fps = SR / HOP  # 100 quadros/s
    on = np.ones((n_tracks, frames), bool)
    if n_tracks > 1 and p_turns > 0 and rng.random() < p_turns:
        on = turns_mask(rng, n_tracks, frames, p_interf_first, turn_gap_max_s)
        if not target_present:
            on[0] = False
        return "turns", on
    if n_tracks == 1:
        pat = "continuous" if rng.random() < 0.6 else "sparse"
    else:
        pat = ["continuous", "conversation", "blocks", "sparse"][int(rng.choice(4, p=list(pattern_probs)))]
    if pat == "conversation":
        on[:] = False
        t, cur = 0.0, int(rng.integers(n_tracks))
        if p_interf_first > 0 and rng.random() < p_interf_first:
            cur = 1 + int(rng.integers(n_tracks - 1))
        while t < frames:
            d = rng.uniform(0.5, 2.0) * fps
            a, b = int(t), int(min(frames, t + d))
            on[cur, a:b] = True
            ov = rng.uniform(0.0, 0.3)
            t = t + d * (1.0 - ov)
            nxt = [k for k in range(n_tracks) if k != cur]
            cur = nxt[int(rng.integers(len(nxt)))]
    elif pat == "blocks":
        order = list(rng.permutation(3))  # 0: só alvo, 1: ambos, 2: só interferente
        if p_interf_first > 0 and rng.random() < p_interf_first and order[-1] == 2:
            order = [2] + order[:-1]  # o só-interferente deixa de ser o último: algum bloco com alvo vem depois
        cuts = np.sort(rng.uniform(0.2, 0.8, 2)) * frames
        bounds = [0, int(cuts[0]), int(cuts[1]), frames]
        on[:] = False
        for blk, (a, b) in zip(order, zip(bounds[:-1], bounds[1:])):
            if blk in (0, 1):
                on[0, a:b] = True
            if blk in (1, 2):
                on[1:, a:b] = True
    elif pat == "sparse":
        for k in range(n_tracks):
            t = 0.0
            while t < frames:
                d = rng.uniform(0.3, 1.5) * fps
                on[k, int(t) : int(min(frames, t + d))] = rng.random() < 0.55
                t += d
            if not on[k].any():
                a = int(rng.integers(0, max(frames - 50, 1)))
                on[k, a : a + 50] = True
    if not target_present:
        on[0] = False
    return pat, on


def _sir(rng: np.random.Generator, phase: int) -> float:
    if phase == 1:
        return float(rng.uniform(-10.0, 20.0))
    u = rng.random()
    if u < 0.65:
        return float(rng.uniform(-10.0, 15.0))
    if u < 0.80:
        return float(rng.uniform(-15.0, -10.0))
    return float(rng.uniform(15.0, 30.0))


def _level(rng: np.random.Generator) -> tuple[float, bool]:
    """(nível ativo em dBFS, baixo/sussurro)."""
    if rng.random() < 0.10:
        return float(rng.uniform(-50.0, -40.0)), True
    return float(rng.uniform(-40.0, -12.0)), False


def _noise_mode(rng: np.random.Generator, phase: int) -> tuple[str, float]:
    p_silent = 0.30 if phase == 3 else 0.20
    rest = 1.0 - p_silent
    u = rng.random()
    if u < p_silent:
        return "silent", float(rng.uniform(-80.0, -60.0))
    u = (u - p_silent) / rest
    if u < 0.30 / 0.80:
        return "snr", float(rng.uniform(20.0, 40.0))
    if u < 0.65 / 0.80:
        return "snr", float(rng.uniform(5.0, 20.0))
    return "snr", float(rng.uniform(-5.0, 5.0))


# --- pools ---------------------------------------------------------------------------------------------------------


@dataclass
class V4Pools:
    """Pools compactos (só strings e números, para não inflar a RAM dos workers)."""

    target_tiers: dict[str, list[str]]           # tier -> falantes-alvo (no banco)
    recs: dict[str, list[tuple[str, int, float]]]  # falante -> [(path, sr, band_hz)]
    rec_tier: dict[str, dict[str, list[int]]]     # falante -> tier -> índices em recs[falante]
    interf_keys: list[str]                         # todos os falantes possíveis como interferente
    neighbors: dict[str, list[str]]               # alvo -> vizinhos por cosseno (falantes de treino)
    noise: list[tuple[str, int, float]]
    rirs: list[str]
    heldout_enroll: tuple[str, ...] = field(default=())
    stats: dict = field(default_factory=dict)


def build_pools(speech_train: Sequence[FileRecord], noise: Sequence[FileRecord], rirs: Sequence[FileRecord], bank,
                data_dir: Path, *, n_neighbors: int = 50, cv_extra_min_clips: int = 5, cv_extra_max_clips: int = 20,
                seed: int = 0) -> V4Pools:
    rng = np.random.default_rng(seed)
    recs: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    heldout_enroll = []
    n_heldout_speech = 0
    for r in speech_train:
        if r.split != "train":
            raise ValueError(f"{r.path}: split {r.split} fora do treino")
        if is_heldout_speech(r.path):
            heldout_enroll.append(r.path)
            n_heldout_speech += 1
            continue
        recs[r.speaker_key].append((r.path, int(r.sr), float(r.band_hz) if r.band_hz else NYQUIST_HZ))
    in_bank = set(bank.speakers())
    base_keys = sorted(recs)
    # interferentes extras do CV-full: falantes novos, split train, fora do filtrado
    split = json.loads((data_dir / "manifests" / "cv27-pt-splits-v1.json").read_text())
    nontrain = {k for k, v in split.items() if v != "train"}
    full = data_dir / "manifests" / "cv27-pt-full.jsonl"
    extra: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    if full.is_file():
        for line in full.read_text().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            key = f"{d['source_id']}:{d['speaker']}"
            if key in recs or key in nontrain or d.get("split") != "train":
                continue
            extra[key].append((d["path"], int(d["sr"]), float(d["band_hz"] or NYQUIST_HZ)))
    n_extra = 0
    for k, lst in sorted(extra.items()):
        if len(lst) < cv_extra_min_clips:
            continue
        idx = rng.permutation(len(lst))[:cv_extra_max_clips]
        recs[k] = [lst[i] for i in sorted(idx)]
        n_extra += 1
    rec_tier: dict[str, dict[str, list[int]]] = {}
    for k, lst in recs.items():
        d: dict[str, list[int]] = defaultdict(list)
        for i, (_p, sr, _b) in enumerate(lst):
            d[tier_of(sr)].append(i)
        rec_tier[k] = dict(d)
    targets = [k for k in base_keys if k in in_bank]
    target_tiers = {t: [k for k in targets if rec_tier[k].get(t)] for t in TIERS}
    # vizinhos por cosseno do centróide do banco
    cents = []
    for k in targets:
        v = np.asarray(bank.all(k), np.float64)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        c = v.mean(0)
        cents.append(c / (np.linalg.norm(c) + 1e-12))
    C = np.stack(cents)
    sim = C @ C.T
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argsort(-sim, axis=1)[:, :n_neighbors]
    neighbors = {k: [targets[j] for j in nn_idx[i]] for i, k in enumerate(targets)}
    noise_l, n_heldout_noise = [], 0
    for r in noise:
        if is_heldout_noise(r.path):
            n_heldout_noise += 1
            continue
        noise_l.append((r.path, int(r.sr), float(r.band_hz) if r.band_hz else NYQUIST_HZ))
    stats = {"targets": len(targets), "train_speakers": len(base_keys), "cv_extra_speakers": n_extra,
             "interferer_speakers": len(recs), "noise": len(noise_l), "rirs": len(rirs),
             "heldout_speech_files": n_heldout_speech,
             "joao_messias_train_files": len(recs.get("speech-joao-messias:joao_messias", [])), "heldout_noise_files": n_heldout_noise,
             "target_tiers": {t: len(v) for t, v in target_tiers.items()},
             "neighbor_cos_median": float(np.median(np.take_along_axis(sim, nn_idx[:, :1], 1)))}
    return V4Pools(target_tiers=target_tiers, recs=dict(recs), rec_tier=rec_tier, interf_keys=sorted(recs),
                   neighbors=neighbors, noise=noise_l, rirs=[r.path for r in rirs],
                   heldout_enroll=tuple(heldout_enroll), stats=stats)


# --- dataset -------------------------------------------------------------------------------------------------------


class StageV4Dataset(Dataset):
    """Amostra `i` -> dict; o índice global `i` = passo·B + posição. `start_step` permite retomada exata."""

    def __init__(self, pools: V4Pools, bank, data_root: Path, p: V4Params, *, length: int | None = None):
        self.pools, self.bank, self.data_root, self.p = pools, bank, Path(data_root), p
        self.slots = slot_tiers(p.batch_size, p.band_quota)
        self.length = length if length is not None else p.total_steps * p.batch_size
        self.provenance = bool(getattr(bank, "has_provenance", False))
        self._rir_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def __getstate__(self) -> dict:
        # o cache de RIRs (≈43 MB f32) é por processo: nunca viaja no pickle (spawn/forkserver), cada worker monta o seu
        return {**self.__dict__, "_rir_cache": None}

    def __len__(self) -> int:
        return self.length

    def _load48(self, rel: str) -> np.ndarray:
        x, sr = load_audio(self.data_root / rel)
        return resample(x, sr, SR)

    def _rirs(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Todas as RIRs (48 kHz, float32) e suas partes "early" (caminho direto + 50 ms, vista sem cópia), carregadas
        uma vez por processo/worker: antes cada amostra relia, decodificava e reamostrava 1–5 RIRs do disco."""
        if self._rir_cache is None:
            cache = {}
            for rel in self.pools.rirs:
                h = np.ascontiguousarray(self._load48(rel), dtype=np.float32)
                cache[rel] = (h, early_rir(h))
            self._rir_cache = cache
        return self._rir_cache

    def phase(self, i: int) -> int:
        frac = (i // self.p.batch_size) / max(self.p.total_steps, 1)
        return 1 if frac < self.p.phase1_frac else (3 if frac >= self.p.phase3_frac else 2)

    def __getitem__(self, i: int) -> dict:
        for attempt in range(MAX_REDRAWS):
            rng = np.random.default_rng([int(self.p.seed), int(i), int(attempt)])
            try:
                return self._build(int(i), rng)
            except (EmbeddingBankError, ValueError):
                continue
        raise RuntimeError(f"amostra {i}: {MAX_REDRAWS} sorteios falharam")

    def _pick_interferer(self, rng, tkey: str, taken: set[str], p_hard: float) -> str:
        nb = [k for k in self.pools.neighbors.get(tkey, []) if k not in taken]
        if nb and rng.random() < p_hard:
            return nb[int(rng.integers(len(nb)))]
        keys = self.pools.interf_keys
        for _ in range(20):
            k = keys[int(rng.integers(len(keys)))]
            if k not in taken:
                return k
        raise ValueError("sem interferente")

    def _speech(self, rng, key: str, n: int, tier: str | None = None) -> tuple[np.ndarray, tuple[str, int, float]]:
        lst = self.pools.recs[key]
        idx = self.pools.rec_tier[key].get(tier) if tier else None
        j = idx[int(rng.integers(len(idx)))] if idx else int(rng.integers(len(lst)))
        rec = lst[j]
        return crop_or_pad(self._load48(rec[0]), n, rng), rec

    def _build(self, i: int, rng: np.random.Generator) -> dict:
        p = self.p
        n = int(p.segment_s * SR)
        frames = n // HOP
        phase = self.phase(i)
        tier = self.slots[i % p.batch_size]
        npr = tuple(p.n_probs)
        probs = np.array(npr[:3]) / sum(npr[:3]) if phase == 1 else np.array(npr)
        n_int = int(rng.choice(len(probs), p=probs))
        p_hard = {1: 0.20, 2: 0.35, 3: 0.50}[phase]
        target_present = not (n_int >= 1 and rng.random() < p.p_target_absent / (1 - npr[0]))
        tkeys = self.pools.target_tiers[tier]
        tkey = tkeys[int(rng.integers(len(tkeys)))]
        tdry, trec = self._speech(rng, tkey, n, tier)
        lvl_db, low = _level(rng)
        if target_present and low and rng.random() < 0.5:
            tdry = whisper_tilt(tdry)
        taken = {tkey}
        idry, irecs = [], []
        for _ in range(n_int):
            k = self._pick_interferer(rng, tkey, taken, p_hard)
            taken.add(k)
            x, r = self._speech(rng, k, n)
            idry.append(x)
            irecs.append(r)
        pat, on = temporal_masks(rng, 1 + n_int, frames, target_present, p.pattern_probs, p.p_interf_first,
                                 p.p_turns, p.turn_gap_max_s)
        tracks = [np.asarray(tdry, np.float64) * _ramp_env(on[0], n)] + \
                 [np.asarray(x, np.float64) * _ramp_env(on[k + 1], n) for k, x in enumerate(idry)]
        reverb = bool(self.pools.rirs) and rng.random() < (p.p_rir_turns if pat == "turns" else p.p_rir)
        wet, early = [], []
        for x in tracks:
            if reverb:
                h, h_early = self._rirs()[self.pools.rirs[int(rng.integers(len(self.pools.rirs)))]]
                wet.append(_conv(x, h))
                early.append(_conv(x, h_early))
            else:
                # cópias distintas: os ganhos abaixo (`wet[k] *= g; early[k] *= g`) são in-place e, com o mesmo array
                # nas duas listas, o interferente seco recebia g² (SIR efetivo 2t−2s−i em vez de s; bug até 3b36625)
                wet.append(x)
                early.append(x.copy())
        # níveis relativos: SIR por interferente contra o nível ativo do alvo
        sirs = []
        if target_present:
            t_db = active_rms_db(wet[0], frames)
            if t_db < -150:
                raise ValueError("alvo sem fala ativa")
            for k in range(n_int):
                s = _sir(rng, phase)
                i_db = active_rms_db(wet[k + 1], frames)
                if i_db < -150:
                    raise ValueError("interferente sem fala ativa")
                g = 10 ** ((t_db - s - i_db) / 20.0)
                wet[k + 1] *= g
                early[k + 1] *= g
                sirs.append(s)
        else:
            wet[0][:] = 0.0
            early[0][:] = 0.0
            for k in range(n_int):  # interferentes entre si: ±6 dB
                if k:
                    rel = float(rng.uniform(-6.0, 6.0))
                    i_db = active_rms_db(wet[k + 1], frames)
                    r_db = active_rms_db(wet[1], frames)
                    if i_db < -150 or r_db < -150:
                        raise ValueError("interferente sem fala ativa")
                    g = 10 ** ((r_db + rel - i_db) / 20.0)
                    wet[k + 1] *= g
                    early[k + 1] *= g
        speech = np.sum(wet, axis=0)
        interf_sum = np.sum(wet[1:], axis=0) if n_int else np.zeros(n)
        ref_sig = wet[0] if target_present else interf_sum
        ref_db = active_rms_db(ref_sig, frames)
        if ref_db < -150:
            raise ValueError("referência sem fala ativa")
        # ruído (SNR contra o alvo, ou contra os interferentes com alvo ausente)
        mode, val = _noise_mode(rng, phase)
        band = trec[2] if target_present else irecs[0][2]
        src_sr = trec[1] if target_present else irecs[0][1]
        noise = np.zeros(n)
        floor_db = None
        if mode == "snr":
            nrec = self.pools.noise[int(rng.integers(len(self.pools.noise)))]
            nz = loop_to_length(self._load48(nrec[0]), n, rng).astype(np.float64)
            if src_sr < 2 * NYQUIST_HZ:
                nz = lowpass48(nz, src_sr / 2.0 * 0.98).astype(np.float64)
            nz_db = 10 * np.log10(np.mean(nz**2) + 1e-20)
            noise = nz * 10 ** ((ref_db - val - nz_db) / 20.0)
        else:
            floor_db = val
        # nível absoluto: o sinal de referência (alvo ou interferentes) vai para o nível sorteado
        g = 10 ** ((lvl_db - ref_db) / 20.0)
        noisy = (speech + noise) * g
        early = [e * g for e in early]
        if floor_db is not None:
            fl = pink_noise(n, rng) if rng.random() < 0.5 else rng.standard_normal(n)
            noisy = noisy + fl * 10 ** (floor_db / 20.0)
        pk = float(np.max(np.abs(noisy)))
        lim = min(1.0, PEAK_LIMIT / pk) if pk > 0 else 1.0
        noisy *= lim
        early = [e * lim for e in early]
        tgt = early[0]
        int_ref = np.sum(early[1:], axis=0) if n_int else np.zeros(n)
        clean_all = tgt + int_ref
        mask0 = bool(rng.random() < p.p_drop)
        exclude = (trec[0],) + self.pools.heldout_enroll
        if self.provenance:
            emb = np.asarray(self.bank.sample(tkey, rng, exclude_utterances=exclude,
                                              p_denoised=p.p_enroll_denoise), np.float32)
        else:
            emb = np.asarray(self.bank.sample(tkey, rng), np.float32)
        noisy32 = noisy.astype(np.float32)
        feats = extract_features(noisy32)
        t_act = rel_activity(tgt * 1.0, frames) if target_present else np.zeros(frames, bool)
        i_act = np.zeros(frames, bool)
        for e in early[1:]:
            i_act |= rel_activity(e, frames)
        # pro-v42: interferente com reverberação (trilha molhada), VAD a 45 dB do máximo (caudas)
        i_act45 = np.zeros(frames, bool)
        for k in range(n_int):
            i_act45 |= rel_activity(wet[k + 1] * g * lim, frames, range_db=45.0)
        fms = np.mean(noisy32[: frames * HOP].reshape(frames, HOP).astype(np.float64) ** 2, axis=1)
        dsil = fms < SILENCE_MS
        max_freq = float(min(band, NYQUIST_HZ))
        return {
            "spec": feats.spec, "feat_erb": feats.feat_erb, "feat_spec": feats.feat_spec,
            "spec_clean": analysis((clean_all if mask0 else tgt).astype(np.float32)),
            "spec_interf": analysis(int_ref.astype(np.float32)),
            "embedding": emb, "mask": np.full(1, 0.0 if mask0 else 1.0, np.float32),
            "target_active": t_act, "interf_active": i_act, "interf_active_wet45": i_act45, "digital_silence": dsil,
            "n_interf": np.int64(n_int), "has_target": np.bool_(target_present),
            "max_freq": np.float32(max_freq), "level_db": np.float32(lvl_db),
            "sir_min_db": np.float32(min(sirs) if sirs else np.nan),
            "noise_silent": np.bool_(mode == "silent"), "noise_val": np.float32(val), "reverb": np.bool_(reverb),
            "pattern": PATTERNS.index(pat), "phase": np.int64(phase),
            "target_key": tkey, "target_path": trec[0],
        }
