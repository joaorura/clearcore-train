"""Indicador barato da AB1 (spec §7.3): cosseno do embedding cru/denoisado com o do enrollment limpo (M1 Tarefa 12)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from cctrain.m1.ab1 import ab1_indicator


def _setup(tmp_path):
    sr = 48_000
    t = np.arange(sr * 3) / sr
    utt: dict[str, list[str]] = {}
    for i, spk in enumerate(("s1", "s2", "s3")):
        for k in range(3):
            x = (0.2 * np.sin(2 * np.pi * (120 + 50 * i) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * (2 + k) * t) ** 2)).astype("f4")
            p = tmp_path / f"{spk}_{k}.flac"
            sf.write(p, x, sr, subtype="PCM_16")
            utt.setdefault(spk, []).append(p.name)
    nz = (0.2 * np.random.default_rng(0).standard_normal(sr * 4)).astype("f4")
    sf.write(tmp_path / "noise.flac", nz, sr, subtype="PCM_16")
    return utt, ["noise.flac"]


def _encoder(x16k):
    spec = np.abs(np.fft.rfft(x16k[:16_000]))[100:292]  # embedding espectral: o ruído branco o afasta do limpo
    return (spec / (np.linalg.norm(spec) + 1e-9)).astype(np.float32)


def test_oracle_denoiser_recovers_the_clean_embedding_and_identity_does_not(tmp_path):
    utt, noise = _setup(tmp_path)
    snrs = (20.0, 0.0)
    ident = ab1_indicator(_encoder, utt, tmp_path, noise, denoise=lambda x: x, snrs_db=snrs, seed=1, resamples=200)
    assert set(ident["by_snr_db"]) == {"20.0", "0.0"}
    for row in ident["by_snr_db"].values():
        assert abs(row["delta_denoised_minus_raw"]["mean"]) < 1e-9  # denoise identidade: nenhum ganho
        assert row["n_speakers"] == 3
    assert ident["by_snr_db"]["0.0"]["cos_raw_clean"]["mean"] < ident["by_snr_db"]["20.0"]["cos_raw_clean"]["mean"]


def test_denoiser_that_removes_the_noise_gives_a_positive_ci(tmp_path):
    utt, noise = _setup(tmp_path)
    clean_by_len = {}

    def oracle(x48):  # devolve a "fala limpa" guardada no registro da chamada anterior: aqui, o sinal sem ruído
        return clean_by_len["x"][: len(x48)]

    res = ab1_indicator(_encoder, utt, tmp_path, noise, denoise=oracle, snrs_db=(0.0,), seed=1, resamples=200,
                        _capture_clean=clean_by_len)
    row = res["by_snr_db"]["0.0"]
    assert row["cos_denoised_clean"]["mean"] > row["cos_raw_clean"]["mean"]
    assert row["delta_denoised_minus_raw"]["ci95"][0] > 0


# ---- controle (M2 Tarefa 1): o enrollment limpo também passa pelo DFNet3 ----

def test_control_identity_denoiser_changes_nothing(tmp_path):
    utt, noise = _setup(tmp_path)
    res = ab1_indicator(_encoder, utt, tmp_path, noise, denoise=lambda x: x, snrs_db=(20.0, 0.0), seed=1, resamples=200)
    assert res["clean_through_denoiser_vs_clean"]["cos"]["mean"] > 1 - 1e-6  # denoise identidade: sem distorção
    for row in res["by_snr_db"].values():
        c = row["control_clean_denoised_ref"]
        assert abs(c["cos_raw_vs_ref"]["mean"] - row["cos_raw_clean"]["mean"]) < 1e-6
        assert abs(c["cos_denoised_vs_ref"]["mean"] - row["cos_denoised_clean"]["mean"]) < 1e-6
        assert abs(c["delta_denoised_minus_raw"]["mean"]) < 1e-9


def test_control_separates_denoise_distortion_from_noise_removal(tmp_path):
    utt, noise = _setup(tmp_path)

    def distorting(x48):  # adiciona um tom fixo de 250 Hz: muda o embedding espectral mesmo sem ruído
        t = np.arange(x48.size) / 48_000
        return (x48 + 0.1 * np.sin(2 * np.pi * 250 * t)).astype("f4")

    res = ab1_indicator(_encoder, utt, tmp_path, noise, denoise=distorting, snrs_db=(10.0,), seed=1, resamples=200)
    assert res["clean_through_denoiser_vs_clean"]["cos"]["mean"] < 0.999  # só o denoise já afasta do limpo
    row = res["by_snr_db"]["10.0"]
    ctl = row["control_clean_denoised_ref"]
    # com a referência também denoisada, o denoisado se aproxima mais dela do que do limpo cru
    assert ctl["cos_denoised_vs_ref"]["mean"] > row["cos_denoised_clean"]["mean"]
    assert set(ctl["delta_denoised_minus_raw"]) == {"mean", "ci95"} and res["n_speakers"] == 3
