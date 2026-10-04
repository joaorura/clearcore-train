"""Métricas dos gates G1-G4 (M2 Tarefa 3): SI-SDR, STOI/ESTOI, LSD > 8 kHz, TSOS, bootstrap pareado."""

from __future__ import annotations

import numpy as np
import pytest

from cctrain.eval import metrics as M

SR = 48_000


def speechlike(seconds: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Sinal modulado em amplitude com harmônicos e ruído de banda larga fraco (ativo em todos os quadros)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 40))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t) ** 2
    x = 0.1 * x * env + 0.002 * rng.standard_normal(t.size)
    return x.astype(np.float32)


def test_as_utterances_accepts_1d_2d_and_lists():
    a = np.zeros(10)
    assert len(M.as_utterances(a)) == 1
    assert len(M.as_utterances(np.zeros((3, 10)))) == 3
    assert [len(u) for u in M.as_utterances([np.zeros(5), np.zeros(7)])] == [5, 7]


# ---- SI-SDR ----
def test_si_sdr_known_snr_and_scale_invariance():
    ref = speechlike(3.0)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(ref.size).astype(np.float32)
    noise *= np.sqrt(np.mean(ref**2) / np.mean(noise**2)) * 10 ** (-10 / 20)  # 10 dB
    est = ref + noise
    v = M.si_sdr(est, ref, SR)
    assert v.shape == (1,) and abs(v[0] - 10.0) < 0.3
    assert abs(M.si_sdr(3.0 * est, ref, SR)[0] - v[0]) < 1e-3  # invariante à escala
    assert M.si_sdr(ref, ref, SR)[0] > 60.0  # identidade


def test_si_sdr_per_utterance_array():
    refs = [speechlike(2.0, seed=0), speechlike(2.5, seed=1)]
    out = M.si_sdr(refs, refs, SR)
    assert out.shape == (2,)


# ---- STOI / ESTOI ----
@pytest.mark.parametrize("extended", [False, True])
def test_stoi_identity_one_and_noise_lowers(extended):
    ref = speechlike(3.0)
    noisy = ref + 0.05 * np.random.default_rng(2).standard_normal(ref.size).astype(np.float32)
    assert abs(M.stoi(ref, ref, SR, extended=extended)[0] - 1.0) < 1e-3
    assert M.stoi(noisy, ref, SR, extended=extended)[0] < 0.95


def test_stoi_resamples_to_16k_fixed_path():
    ref = speechlike(3.0)
    v48 = M.stoi(ref, ref, SR)[0]
    ref16 = speechlike(3.0, sr=16_000)
    v16 = M.stoi(ref16, ref16, 16_000)[0]
    assert abs(v48 - 1.0) < 1e-3 and abs(v16 - 1.0) < 1e-3
    assert M.STOI_SR == 16_000


# ---- LSD > 8 kHz ----
def test_lsd_identity_zero_and_band_restriction():
    from scipy.signal import butter, sosfiltfilt

    ref = speechlike(2.0)
    assert M.lsd_above_8k(ref, ref, SR)[0] == 0.0
    hi = ref - sosfiltfilt(butter(8, 9000, "low", fs=SR, output="sos"), ref).astype(np.float32)  # só > 9 kHz
    assert M.lsd_above_8k(ref - hi, ref, SR)[0] > 5.0  # remover a banda alta pesa muito
    # distorção apenas abaixo de 6 kHz não conta
    low = sosfiltfilt(butter(8, 6000, "low", fs=SR, output="sos"), 0.5 * ref).astype(np.float32)
    assert M.lsd_above_8k(ref + low, ref, SR)[0] < 0.5


def test_lsd_gain_change_is_20log10():
    ref = speechlike(2.0) + 0.02 * np.random.default_rng(5).standard_normal(2 * SR).astype(np.float32)  # banda alta bem acima do piso
    v = M.lsd_above_8k(0.5 * ref, ref, SR)[0]  # −6,02 dB em todos os bins
    assert abs(v - 6.0206) < 0.05


def test_lsd_needs_a_band_above_cutoff():
    ref = speechlike(1.0, sr=16_000)
    with pytest.raises(ValueError, match="8"):
        M.lsd_above_8k(ref, ref, 16_000)


# ---- TSOS (Eskimez et al. 2021, eq. 2-3) ----
def _tsos_bruteforce(est, ref, sr, p, gamma, win_ms, hop_ms):
    """Implementação direta e independente das equações 2 e 3 (laço por quadro)."""
    n, h = int(round(sr * win_ms / 1000)), int(round(sr * hop_ms / 1000))
    w = np.hanning(n + 1)[:-1]  # Hann periódica
    out = []
    for s in range(0, len(ref) - n + 1, h):
        S = np.abs(np.fft.rfft(w * ref[s:s + n].astype(np.float64)))
        E = np.abs(np.fft.rfft(w * est[s:s + n].astype(np.float64)))
        los = np.maximum(S**p - E**p, 0.0) ** 2  # eq. 2 (h(x) = max(x, 0))
        out.append(los.sum() > gamma * (S**p).sum())  # eq. 3
    return np.array(out)


def test_tsos_equation_3_matches_bruteforce():
    rng = np.random.default_rng(3)
    ref = speechlike(2.0, sr=16_000) * 40.0  # amplitude alta o bastante para a escala da eq. 3 produzir os dois casos
    gains = np.where((np.arange(ref.size) // 3200) % 2 == 0, 1.0, 0.0)  # metade dos trechos zerada
    est = (ref * gains + 0.01 * rng.standard_normal(ref.size)).astype(np.float32)
    cfg = M.TsosConfig(p=0.3, gamma=0.1, win_ms=20.0, hop_ms=10.0)
    got = M.tsos_frames(est, ref, 16_000, cfg)
    want = _tsos_bruteforce(est, ref, 16_000, 0.3, 0.1, 20.0, 10.0)
    assert got.dtype == bool and got.shape == want.shape
    assert (got == want).all()
    assert 0.2 < got.mean() < 0.8


def test_tsos_identity_zero_and_silence_is_total_over_active_frames():
    ref = speechlike(2.0) * 40.0
    assert M.tsos(ref, ref, SR)[0] == 0.0
    out = M.tsos(np.zeros_like(ref), ref, SR)[0]
    assert out == 100.0  # saída muda: todos os quadros com fala são sobre-suprimidos
    # alvo silencioso e saída com energia: nada a sobre-suprimir
    assert M.tsos(ref, np.zeros_like(ref), SR)[0] == 0.0


def test_tsos_only_counts_removal_not_addition():
    ref = speechlike(2.0) * 40.0
    assert M.tsos(2.0 * ref, ref, SR)[0] == 0.0  # amplificar não é sobre-supressão (h só passa o removido)
    assert M.tsos(0.0 * ref, ref, SR)[0] > M.tsos(0.9 * ref, ref, SR)[0] >= 0.0


def test_tsos_config_scale_is_explicit():
    ref = speechlike(2.0)  # escala de float [-1, 1]
    est = 0.2 * ref
    small = M.tsos(est, ref, SR, cfg=M.TsosConfig(amplitude_scale=1.0))[0]
    big = M.tsos(est, ref, SR, cfg=M.TsosConfig(amplitude_scale=32768.0))[0]
    assert big >= small  # a eq. 3 não é invariante à escala: o parâmetro é explícito e vai para o relatório


def test_tsos_summary_reports_percent_s30min_and_longest_run():
    frames = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    s = M.tsos_summary(frames, hop_s=0.01)
    assert s["pct"] == 40.0 and s["n_frames"] == 10
    assert abs(s["longest_run_s"] - 0.03) < 1e-9
    assert abs(s["s30min"] - 0.04 / 0.10 * 1800.0) < 1e-6


def test_tsos_active_only_variant_ignores_silent_reference_frames():
    ref = np.concatenate([speechlike(1.0) * 40.0, np.zeros(SR)])
    est = np.zeros_like(ref)
    allf = M.tsos(est, ref, SR)[0]
    act = M.tsos(est, ref, SR, active_only=True)[0]
    assert act == 100.0 and allf < 100.0


# ---- bootstrap pareado ----
def test_paired_bootstrap_by_speaker():
    a = {"s1": 3.0, "s2": 4.0, "s3": 5.0, "s4": 4.0}
    b = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 2.0}
    mean, lo, hi = M.paired_bootstrap(a, b, resamples=500, seed=0)
    assert mean == 2.0 and lo == 2.0 == hi  # delta constante por falante: IC degenerado
    with pytest.raises(ValueError, match="falantes"):
        M.paired_bootstrap({"s1": 1.0}, {"s2": 1.0})
    m2, lo2, hi2 = M.paired_bootstrap({"s1": 1, "s2": 5, "s3": 2, "s4": 6}, {"s1": 0, "s2": 0, "s3": 0, "s4": 0},
                                      resamples=2000, seed=1)
    assert lo2 < m2 < hi2 and abs(m2 - 3.5) < 1e-9
