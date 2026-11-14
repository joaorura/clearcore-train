"""Receita das amostras do mix_v4 na CPU (workers do DataLoader); a mistura é feita em lote na GPU (`mixer.py`).

O worker repete EXATAMENTE a sequência de sorteios de `StageV4Dataset._build` (mesmo `rng` por `(seed, i, tentativa)`)
e para antes da álgebra de sinais: entrega o áudio cru recortado a 48 kHz (decodificação + soxr continuam aqui), as
máscaras de turno, os índices de RIR, SIR/±6 dB, modo de ruído, o ruído branco do piso e o embedding. Como os sorteios
do `_build` não dependem dos valores do áudio, só das exceções (que descartam a tentativa inteira), sortear tudo antes e
misturar depois produz a mesma amostra.

Exceções do `_build` que dependem do áudio (`active_rms_db < -150` de alguma trilha presente): a receita testa a
condição suficiente "trilha seca nula nos quadros ligados (±1 quadro de rampa)", que é o caso real (máscara sem turno
ou recorte digitalmente mudo); o resto (sinal > 0 mas abaixo de −150 dB) a GPU detecta em `valid` e o lote cai no
caminho CPU para aquela amostra (`GpuMixer.fill_invalid`).
"""

from __future__ import annotations

import numpy as np
import torch

from cctrain.data.features import SR
from cctrain.data.mix import loop_to_length
from cctrain.data.mix_v4 import (
    HOP,
    MAX_REDRAWS,
    NYQUIST_HZ,
    PATTERNS,
    StageV4Dataset,
    _level,
    _noise_mode,
    _sir,
    temporal_masks,
    whisper_tilt,
)
from cctrain.train.embed_bank import EmbeddingBankError

MAX_TRACKS = 5


def _dead(x: np.ndarray, on: np.ndarray, frames: int) -> bool:
    """Trilha seca sem nenhuma amostra ≠ 0 nos quadros ligados (dilatados 1 quadro: a rampa de 10 ms vaza ±240)."""
    if not on.any():
        return True
    dil = on.copy()
    dil[1:] |= on[:-1]
    dil[:-1] |= on[1:]
    f = np.asarray(x[: frames * HOP], np.float32).reshape(frames, HOP)
    return not np.any(f[dil])


class StageV4RecipeDataset(StageV4Dataset):
    """`i -> receita` (dict de numpy), mesma semântica de índice/retomada do `StageV4Dataset`."""

    def __getitem__(self, i: int) -> dict:
        for attempt in range(MAX_REDRAWS):
            rng = np.random.default_rng([int(self.p.seed), int(i), int(attempt)])
            try:
                r = self._recipe(int(i), rng)
                r["attempt"] = np.int64(attempt)
                return r
            except (EmbeddingBankError, ValueError):
                continue
        raise RuntimeError(f"amostra {i}: {MAX_REDRAWS} sorteios falharam")

    def cpu_sample(self, i: int) -> dict:
        """A amostra pelo caminho CPU de referência (fallback de `valid=False` e testes)."""
        return StageV4Dataset.__getitem__(self, i)

    def _recipe(self, i: int, rng: np.random.Generator) -> dict:
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
        reverb = bool(self.pools.rirs) and rng.random() < (p.p_rir_turns if pat == "turns" else p.p_rir)
        rir_idx = np.full(MAX_TRACKS, -1, np.int64)
        if reverb:
            for k in range(1 + n_int):
                rir_idx[k] = int(rng.integers(len(self.pools.rirs)))
        # exceções dependentes do áudio (ver docstring do módulo): todas as trilhas presentes precisam de fala
        if target_present and _dead(tdry, on[0], frames):
            raise ValueError("alvo sem fala ativa")
        for k, x in enumerate(idry):
            if _dead(x, on[k + 1], frames):
                raise ValueError("interferente sem fala ativa")
        sirs = np.full(MAX_TRACKS - 1, np.nan)
        rels = np.full(MAX_TRACKS - 1, np.nan)
        if target_present:
            for k in range(n_int):
                sirs[k] = _sir(rng, phase)
        else:
            for k in range(1, n_int):
                rels[k] = float(rng.uniform(-6.0, 6.0))
        mode, val = _noise_mode(rng, phase)
        band = trec[2] if target_present else irecs[0][2]
        src_sr = trec[1] if target_present else irecs[0][1]
        noise = np.zeros(0, np.float32)
        lp_cut = NYQUIST_HZ  # sem passa-baixa
        if mode == "snr":
            nrec = self.pools.noise[int(rng.integers(len(self.pools.noise)))]
            noise = loop_to_length(self._load48(nrec[0]), n, rng)
            if src_sr < 2 * NYQUIST_HZ:
                lp_cut = src_sr / 2.0 * 0.98
        floor_white = np.zeros(0, np.float32)
        pink = False
        if mode != "snr":
            pink = bool(rng.random() < 0.5)
            floor_white = rng.standard_normal(n).astype(np.float32)  # pink_noise() sorteia o mesmo branco
        mask0 = self._mask0(i, bool(rng.random() < p.p_drop))
        exclude = (trec[0],) + self.pools.heldout_enroll
        if self.provenance:
            emb = np.asarray(self.bank.sample(tkey, rng, exclude_utterances=exclude,
                                              p_denoised=p.p_enroll_denoise), np.float32)
        else:
            emb = np.asarray(self.bank.sample(tkey, rng), np.float32)
        tracks = ([np.asarray(tdry, np.float32)] if target_present else []) + [np.asarray(x, np.float32) for x in idry]
        track_k = ([0] if target_present else []) + list(range(1, n_int + 1))
        on_full = np.zeros((MAX_TRACKS, frames), bool)
        on_full[: 1 + n_int] = on
        return {
            "index": np.int64(i), "dry": np.stack(tracks), "track_k": np.asarray(track_k, np.int64),
            "on": on_full, "rir_idx": rir_idx, "reverb": np.bool_(reverb), "sirs": sirs, "rels": rels,
            "n_interf": np.int64(n_int), "has_target": np.bool_(target_present), "level_db": np.float64(lvl_db),
            "noise_silent": np.bool_(mode == "silent"), "noise_val": np.float64(val), "noise": noise,
            "lp_cut": np.float64(lp_cut), "floor_white": floor_white, "floor_pink": np.bool_(pink),
            "mask0": np.bool_(mask0), "embedding": emb, "max_freq": np.float32(min(band, NYQUIST_HZ)),
            "pattern": PATTERNS.index(pat), "phase": np.int64(phase), "target_key": tkey, "target_path": trec[0],
        }


_STACK = ("index", "on", "rir_idx", "reverb", "sirs", "rels", "n_interf", "has_target", "level_db", "noise_silent",
          "noise_val", "lp_cut", "floor_pink", "mask0", "embedding", "max_freq", "pattern", "phase", "attempt")


def collate_recipes(items: list[dict]) -> dict:
    """Lote compacto: só as trilhas presentes (`dry[M, n]` + `track_b`/`track_k`), ruído só das amostras com SNR e
    branco do piso só das silenciosas (≈3,5 MB/amostra em 6 s, contra 8 MB com tudo preenchido)."""
    out: dict = {k: torch.from_numpy(np.stack([np.asarray(it[k]) for it in items])) for k in _STACK}
    out["dry"] = torch.from_numpy(np.concatenate([it["dry"] for it in items]))
    out["track_b"] = torch.cat([torch.full((len(it["track_k"]),), b, dtype=torch.long) for b, it in enumerate(items)])
    out["track_k"] = torch.from_numpy(np.concatenate([it["track_k"] for it in items]))
    for key, flag in (("noise", lambda it: len(it["noise"]) > 0), ("floor_white", lambda it: len(it["floor_white"]) > 0)):
        sel = [b for b, it in enumerate(items) if flag(it)]
        n = items[0]["dry"].shape[1]
        out[key] = torch.from_numpy(np.stack([items[b][key] for b in sel])) if sel else torch.zeros(0, n)
        out[key + "_b"] = torch.as_tensor(sel, dtype=torch.long)
    out["target_key"] = [it["target_key"] for it in items]
    out["target_path"] = [it["target_path"] for it in items]
    return out
