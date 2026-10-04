"""Métricas dos gates G1-G4 (spec §7.1; M2 Tarefa 3): SI-SDR, STOI/ESTOI, LSD > 8 kHz, TSOS e bootstrap pareado.

Convenção: `f(est, ref, sr, **cfg) -> np.ndarray` com um valor por enunciado. `est` e `ref` são 1-D (um enunciado),
2-D (N x T) ou uma lista de arrays; `est` já vem **alinhado** a `ref` (o atraso de 1.440 amostras do runtime é
compensado por quem chama); comprimentos diferentes são cortados no menor. SI-SDR, LSD e TSOS rodam na taxa de entrada
(48 kHz nos gates); STOI usa o caminho fixo e registrado até 16 kHz (`cctrain.data.resample.to_16k`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cctrain.data.resample import to_16k
from cctrain.eval.functional import si_sdr as _si_sdr_one
from cctrain.eval.functional import speaker_bootstrap

STOI_SR = 16_000


def as_utterances(x) -> list[np.ndarray]:
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return [x]
        if x.ndim == 2:
            return [row for row in x]
        raise ValueError(f"esperado 1-D, 2-D ou lista de 1-D, veio {x.ndim}-D")
    return [np.asarray(u) for u in x]


def _pairs(est, ref) -> list[tuple[np.ndarray, np.ndarray]]:
    e, r = as_utterances(est), as_utterances(ref)
    if len(e) != len(r):
        raise ValueError(f"{len(e)} estimativas para {len(r)} referências")
    out = []
    for a, b in zip(e, r, strict=True):
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("enunciados mono 1-D")
        n = min(len(a), len(b))
        out.append((np.asarray(a[:n], np.float64), np.asarray(b[:n], np.float64)))
    return out


# ------------------------------------------------------------------------------------------------ SI-SDR
def si_sdr(est, ref, sr: int, *, eps: float = 1e-8) -> np.ndarray:
    """SI-SDR em dB, a mesma definição do gate funcional do M1 (`eval.functional.si_sdr`)."""
    del sr  # independe da taxa
    return np.array([_si_sdr_one(a, b, eps) for a, b in _pairs(est, ref)], np.float64)


# ------------------------------------------------------------------------------------------------ STOI / ESTOI
def stoi(est, ref, sr: int, *, extended: bool = False) -> np.ndarray:
    """STOI (`extended=False`) ou ESTOI (Taal et al. 2011; Jensen e Taal 2016) por `pystoi` (MIT), a 16 kHz."""
    from pystoi import stoi as _stoi

    out = []
    for a, b in _pairs(est, ref):
        if sr != STOI_SR:
            a, b = to_16k(a.astype(np.float32), sr), to_16k(b.astype(np.float32), sr)
        out.append(_stoi(b, a, STOI_SR, extended=extended))  # pystoi: (limpo, degradado)
    return np.array(out, np.float64)


# ------------------------------------------------------------------------------------------------ STFT comum
def _frames(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    if len(x) < n:
        return np.zeros((0, n))
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    return x[idx]


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)  # periódica


def _stft_mag(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    """|STFT| sem normalização (janela Hann periódica, sem preenchimento): [quadros, n//2 + 1]."""
    return np.abs(np.fft.rfft(_frames(x, n, hop) * _hann(n), axis=1))


def _win_hop(sr: int, win_ms: float, hop_ms: float) -> tuple[int, int]:
    return int(round(sr * win_ms / 1000.0)), int(round(sr * hop_ms / 1000.0))


# ------------------------------------------------------------------------------------------------ LSD > 8 kHz
def lsd_above_8k(est, ref, sr: int, *, cutoff_hz: float = 8000.0, win_ms: float = 20.0, hop_ms: float = 10.0,
                 floor_db: float = -100.0) -> np.ndarray:
    """Distância log-espectral (dB) só nos bins acima de `cutoff_hz`, média sobre quadros.

    Por quadro: sqrt(média_k (10 log10(P_ref + piso) - 10 log10(P_est + piso))^2), com P = potência da STFT normalizada
    pela soma da janela (seno de escala plena -> 0,25) e piso `floor_db` (re escala plena)."""
    n, hop = _win_hop(sr, win_ms, hop_ms)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    sel = freqs > cutoff_hz
    if not sel.any():
        raise ValueError(f"sem bins acima de {cutoff_hz / 1000:g} kHz com sr={sr}: a taxa precisa passar de "
                         f"{2 * cutoff_hz / 1000:g} kHz")
    w_sum = _hann(n).sum()
    floor = 10.0 ** (floor_db / 10.0)
    out = []
    for a, b in _pairs(est, ref):
        pa = (_stft_mag(a, n, hop)[:, sel] / w_sum) ** 2
        pb = (_stft_mag(b, n, hop)[:, sel] / w_sum) ** 2
        if len(pa) == 0:
            out.append(np.nan)
            continue
        d = 10.0 * np.log10(pb + floor) - 10.0 * np.log10(pa + floor)
        out.append(float(np.sqrt((d**2).mean(axis=1)).mean()))
    return np.array(out, np.float64)


# ------------------------------------------------------------------------------------------------ TSOS
@dataclass(frozen=True)
class TsosConfig:
    """Parâmetros de medida do TSOS (`calibrated_in_dev: tsos_config` no gates-v2; os níveis G3 não mudam).

    `p` e `gamma` são os do paper (0,3 e 0,1); STFT de 20/10 ms (a do DeepFilterNet; o paper não a fixa nesta parte).
    A eq. 3 NÃO é invariante à escala (o lado esquerdo é quadrático em |S|^p; o direito, linear): `amplitude_scale`
    multiplica os sinais antes da STFT (1,0 = float em [-1, 1]; 32768 = escala int16) e vai para o relatório."""

    p: float = 0.3
    gamma: float = 0.1
    win_ms: float = 20.0
    hop_ms: float = 10.0
    amplitude_scale: float = 1.0
    active_db: float = 40.0  # só para `active_only`: quadros da referência até 40 dB abaixo do quadro mais forte


def tsos_frames(est: np.ndarray, ref: np.ndarray, sr: int, cfg: TsosConfig | None = None, *,
                active_only: bool = False) -> np.ndarray:
    """Máscara booleana por quadro de sobre-supressão (Eskimez et al. 2021, eq. 2 e 3, arXiv 2110.09625, conferidas no PDF).

    Com S o espectro do alvo limpo (`ref`), Ŝ o da saída (`est`):
        L_OS(t, f) = h(|S(t, f)|^p - |Ŝ(t, f)|^p)^2,   h(x) = x se x > 0, senão 0                      (eq. 2)
        TSOS(t)    = 1 se  Σ_f L_OS(t, f) > γ Σ_f |S(t, f)|^p,  senão 0                                 (eq. 3)
    Com `active_only`, os quadros da referência sem atividade ficam de fora da máscara (devolve só os ativos)."""
    cfg = cfg or TsosConfig()
    (a, b), = _pairs(est, ref)
    n, hop = _win_hop(sr, cfg.win_ms, cfg.hop_ms)
    s = _stft_mag(b * cfg.amplitude_scale, n, hop) ** cfg.p
    e = _stft_mag(a * cfg.amplitude_scale, n, hop) ** cfg.p
    los = np.maximum(s - e, 0.0) ** 2
    flags = los.sum(axis=1) > cfg.gamma * s.sum(axis=1)
    if active_only:
        fr = (_frames(b, n, hop) ** 2).mean(axis=1)
        if fr.size == 0 or fr.max() <= 0.0:
            return flags[:0]
        flags = flags[fr > fr.max() * 10.0 ** (-cfg.active_db / 10.0)]
    return flags


def tsos(est, ref, sr: int, cfg: TsosConfig | None = None, *, active_only: bool = False) -> np.ndarray:
    """TSOS em % dos quadros, por enunciado (NaN sem quadros)."""
    out = []
    for a, b in _pairs(est, ref):
        f = tsos_frames(a, b, sr, cfg, active_only=active_only)
        out.append(100.0 * float(f.mean()) if f.size else np.nan)
    return np.array(out, np.float64)


def tsos_summary(frames: np.ndarray, hop_s: float) -> dict:
    """Percentual, segundos por 30 min (E3Net, arXiv 2204.00771) e a maior corrida contínua de quadros OS."""
    f = np.asarray(frames, bool)
    n = int(f.size)
    longest = run = 0
    for v in f:
        run = run + 1 if v else 0
        longest = max(longest, run)
    pct = 100.0 * float(f.mean()) if n else float("nan")
    return {"pct": pct, "n_frames": n, "n_os": int(f.sum()), "longest_run_s": longest * hop_s,
            "s30min": (pct / 100.0) * 1800.0 if n else float("nan")}


# ------------------------------------------------------------------------------------------------ bootstrap pareado
def paired_bootstrap(per_speaker_a, per_speaker_b, *, resamples: int = 10_000, ci: float = 0.95,
                     seed: int = 0) -> tuple[float, float, float]:
    """Média e IC percentil de (a - b) com unidade de reamostragem = falante (`gates-v2.statistics`).

    Reaproveita `eval.functional.speaker_bootstrap` do M1. Aceita {falante: valor} (as chaves têm de coincidir) ou duas
    sequências do mesmo tamanho (um valor por falante)."""
    if isinstance(per_speaker_a, Mapping) and isinstance(per_speaker_b, Mapping):
        if set(per_speaker_a) != set(per_speaker_b):
            raise ValueError("os dois braços precisam ter os mesmos falantes")
        keys = sorted(per_speaker_a)
        a = [float(per_speaker_a[k]) for k in keys]
        b = [float(per_speaker_b[k]) for k in keys]
    else:
        a, b = list(per_speaker_a), list(per_speaker_b)
        if len(a) != len(b):
            raise ValueError("os dois braços precisam ter os mesmos falantes (mesmo tamanho)")
        keys = [str(i) for i in range(len(a))]
    deltas: Sequence[float] = [x - y for x, y in zip(a, b, strict=True)]
    return speaker_bootstrap(deltas, keys, resamples=resamples, ci=ci, seed=seed)
