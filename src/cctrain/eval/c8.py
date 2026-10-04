"""C8: enrollment inválido não pode zerar o alvo (M2 Tarefa 3).

Critério (R06 §AB2; spec §7.3): TSOS do alvo real com enrollment inválido <= TSOS no modo sem perfil + 1 p.p. Aqui: a
geração dos enrollments inválidos (silêncio, ruído branco e rosa, "música" sintética, fala de outro, trecho curto) e o
veredito por tipo, pareado por falante (`paired_bootstrap`). O embedding e o TSOS vêm de quem chama (Tarefa 4). Informativo
no M2 (G5 e C8 não bloqueiam), mas entra na decisão da AB2. Leitura padrão: estimativa pontual, como escrito no critério;
`reading="ci_upper"` aplica o limite superior do IC 95% (mais estrito).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cctrain.eval.metrics import paired_bootstrap

SR = 16_000
INVALID_KINDS = ("silence", "white_noise", "pink_noise", "music_synthetic", "other_speaker", "short")
MARGIN_PP = 1.0
_SCALE = (0, 2, 4, 5, 7, 9, 11, 12)  # escala maior, em semitons


def _normalize(x: np.ndarray, rms_dbfs: float = -25.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return x if rms == 0.0 else (x * (10 ** (rms_dbfs / 20.0) / rms)).astype(np.float32)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.arange(spec.size), 1)
    return np.fft.irfft(spec / np.sqrt(f), n)


def _music(seconds: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(int(seconds * SR))
    note_n, pos = int(0.45 * SR), 0
    root = 220.0 * 2 ** (int(rng.integers(0, 12)) / 12)
    while pos < out.size:
        f0 = root * 2 ** (int(rng.choice(_SCALE)) / 12)
        n = min(note_n, out.size - pos)
        t = np.arange(n) / SR
        env = np.exp(-4.0 * t / 0.45) * (1 - np.exp(-t / 0.01))
        out[pos:pos + n] = env * sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3, 4))
        pos += note_n
    return out


def make_invalid_enrollment(kind: str, rng: np.random.Generator, *, seconds: float = 8.0,
                            other_speech: np.ndarray | None = None) -> np.ndarray:
    """Áudio mono float32 a 16 kHz, determinístico por `rng`. `other_speech` (16 kHz) é a fala de outro falante."""
    n = int(seconds * SR)
    if kind == "silence":
        return np.zeros(n, np.float32)
    if kind == "white_noise":
        return _normalize(rng.standard_normal(n).astype(np.float32))
    if kind == "pink_noise":
        return _normalize(_pink(n, rng).astype(np.float32))
    if kind == "music_synthetic":
        return _normalize(_music(seconds, rng).astype(np.float32))
    if kind in ("other_speaker", "short"):
        if other_speech is None or len(other_speech) == 0:
            raise ValueError(f"{kind}: informe other_speech (fala de outro falante, 16 kHz)")
        x = np.asarray(other_speech, np.float32)
        return x[:n].copy() if kind == "other_speaker" else x[: int(0.8 * SR)].copy()
    raise ValueError(f"tipo de enrollment inválido desconhecido: {kind!r} (esperado um de {INVALID_KINDS})")


def c8_verdict(tsos_invalid: Mapping[str, Mapping[str, float]], tsos_noprofile: Mapping[str, float], *,
               margin_pp: float = MARGIN_PP, reading: str = "point_estimate", resamples: int = 10_000,
               seed: int = 0) -> dict:
    """Por tipo: delta = TSOS(inválido) - TSOS(sem perfil), em p.p., pareado por falante; passa se o delta (média, ou limite
    superior do IC 95% com `reading="ci_upper"`) <= `margin_pp`. O veredito geral exige todos os tipos."""
    if reading not in ("point_estimate", "ci_upper"):
        raise ValueError("reading: point_estimate ou ci_upper")
    kinds = {}
    for kind, per_spk in tsos_invalid.items():
        mean, lo, hi = paired_bootstrap(per_spk, tsos_noprofile, resamples=resamples, seed=seed)
        value = mean if reading == "point_estimate" else hi
        kinds[kind] = {"delta_pp": {"mean": mean, "ci95": [lo, hi]}, "pass": bool(value <= margin_pp)}
    return {"margin_pp": margin_pp, "reading": reading, "kinds": kinds,
            "pass": bool(kinds) and all(v["pass"] for v in kinds.values())}
