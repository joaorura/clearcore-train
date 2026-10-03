"""Indicador barato da AB1 (spec §7.3, R06 §5): quanto o ruído no enrollment afasta o embedding do limpo e quanto o
denoise pelo DFNet3 (sem perfil) o devolve. Por falante de dev: cos(emb(cru), emb(limpo)) e cos(emb(denoisado),
emb(limpo)) em cada SNR; o delta é emparelhado por falante, com IC 95% por bootstrap sobre falantes.
Informativo: não é nível do M1; o resultado completo da AB1 é do M4."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from cctrain.data.mix import loop_to_length, scale_for_ratio
from cctrain.data.resample import to_16k
from cctrain.eval.functional import speaker_bootstrap
from cctrain.train.embed_bank import SR48, _load48, concat_utterances48


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ab1_indicator(encoder: Callable[[np.ndarray], np.ndarray], utterances: Mapping[str, Sequence[str]],
                  data_root: Path, noise: Sequence[str], *, denoise: Callable[[np.ndarray], np.ndarray],
                  snrs_db: Sequence[float] = (20.0, 10.0, 5.0, 0.0, -5.0), seed: int = 0, resamples: int = 10_000,
                  _capture_clean: dict | None = None) -> dict:
    rows: dict[str, dict[str, list[float]]] = {f"{s}": {"raw": [], "den": [], "spk": []} for s in snrs_db}
    for si, spk in enumerate(sorted(utterances)):
        rng = np.random.default_rng([seed, si])
        clean48 = concat_utterances48(rng, list(utterances[spk]), data_root)
        if _capture_clean is not None:
            _capture_clean["x"] = clean48
        ref = encoder(to_16k(clean48, SR48))
        nz = loop_to_length(_load48(data_root, noise[int(rng.integers(len(noise)))]), clean48.size, rng)
        for snr in snrs_db:
            noisy = (clean48.astype(np.float64) + scale_for_ratio(clean48, nz, float(snr))).astype(np.float32)
            den = np.asarray(denoise(noisy), np.float32)
            r = rows[f"{snr}"]
            r["raw"].append(_cos(encoder(to_16k(noisy, SR48)), ref))
            r["den"].append(_cos(encoder(to_16k(den, SR48)), ref))
            r["spk"].append(spk)
    out: dict = {"n_speakers": len(utterances), "snrs_db": list(snrs_db), "by_snr_db": {}}
    for key, r in rows.items():
        raw, den = np.array(r["raw"]), np.array(r["den"])
        m, lo, hi = speaker_bootstrap(den - raw, r["spk"], resamples=resamples, seed=seed)
        out["by_snr_db"][key] = {
            "n_speakers": len(raw), "cos_raw_clean": {"mean": float(raw.mean())},
            "cos_denoised_clean": {"mean": float(den.mean())},
            "delta_denoised_minus_raw": {"mean": m, "ci95": [lo, hi]}}
    return out
