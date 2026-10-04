"""Indicador barato da AB1 (spec §7.3, R06 §5): quanto o ruído no enrollment afasta o embedding do limpo e quanto o
denoise pelo DFNet3 (sem perfil) o devolve. Por falante de dev: cos(emb(cru), emb(limpo)) e cos(emb(denoisado),
emb(limpo)) em cada SNR; o delta é emparelhado por falante, com IC 95% por bootstrap sobre falantes.
Informativo: não é nível do M1; o resultado completo da AB1 é do M4.

Controle (M2 Tarefa 1): a referência "limpa" do indicador original não passava pelo DFNet3, então não separava a
distorção do próprio denoise da perda de informação do ruído. O controle repete as medidas contra a referência
`emb(denoise(limpo))` e reporta `cos(emb(denoise(limpo)), emb(limpo))` (distorção só do denoise, sem ruído)."""

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
                  _capture_clean: dict | None = None,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    rows: dict[str, dict[str, list[float]]] = {
        f"{s}": {"raw": [], "den": [], "raw_c": [], "den_c": [], "spk": []} for s in snrs_db}
    clean_den_cos: list[float] = []
    clean_spk: list[str] = []
    for si, spk in enumerate(sorted(utterances)):
        if progress is not None:
            progress(si, len(utterances))
        rng = np.random.default_rng([seed, si])
        clean48 = concat_utterances48(rng, list(utterances[spk]), data_root)
        if _capture_clean is not None:
            _capture_clean["x"] = clean48
        ref = encoder(to_16k(clean48, SR48))
        ref_den = encoder(to_16k(np.asarray(denoise(clean48), np.float32), SR48))  # controle: limpo pelo denoise
        clean_den_cos.append(_cos(ref_den, ref))
        clean_spk.append(spk)
        nz = loop_to_length(_load48(data_root, noise[int(rng.integers(len(noise)))]), clean48.size, rng)
        for snr in snrs_db:
            noisy = (clean48.astype(np.float64) + scale_for_ratio(clean48, nz, float(snr))).astype(np.float32)
            den = np.asarray(denoise(noisy), np.float32)
            r = rows[f"{snr}"]
            e_raw, e_den = encoder(to_16k(noisy, SR48)), encoder(to_16k(den, SR48))
            r["raw"].append(_cos(e_raw, ref))
            r["den"].append(_cos(e_den, ref))
            r["raw_c"].append(_cos(e_raw, ref_den))
            r["den_c"].append(_cos(e_den, ref_den))
            r["spk"].append(spk)
    cm, clo, chi = speaker_bootstrap(np.array(clean_den_cos), clean_spk, resamples=resamples, seed=seed)
    out: dict = {"n_speakers": len(utterances), "snrs_db": list(snrs_db), "by_snr_db": {},
                 "clean_through_denoiser_vs_clean": {"cos": {"mean": cm, "ci95": [clo, chi]}}}
    for key, r in rows.items():
        raw, den = np.array(r["raw"]), np.array(r["den"])
        m, lo, hi = speaker_bootstrap(den - raw, r["spk"], resamples=resamples, seed=seed)
        raw_c, den_c = np.array(r["raw_c"]), np.array(r["den_c"])
        mc, loc, hic = speaker_bootstrap(den_c - raw_c, r["spk"], resamples=resamples, seed=seed)
        out["by_snr_db"][key] = {
            "n_speakers": len(raw), "cos_raw_clean": {"mean": float(raw.mean())},
            "cos_denoised_clean": {"mean": float(den.mean())},
            "delta_denoised_minus_raw": {"mean": m, "ci95": [lo, hi]},
            "control_clean_denoised_ref": {
                "cos_raw_vs_ref": {"mean": float(raw_c.mean())}, "cos_denoised_vs_ref": {"mean": float(den_c.mean())},
                "delta_denoised_minus_raw": {"mean": mc, "ci95": [loc, hic]}}}
    return out
