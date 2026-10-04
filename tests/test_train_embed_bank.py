"""Banco de embeddings do M2 (spec §4.4): ~20 variantes de enrollment por falante de treino, com proveniência (v2)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from cctrain.data.enroll import check_enrollment_audio
from cctrain.data.manifest import FileRecord, ManifestError
from cctrain.hashing import sha256_file
from cctrain.train.embed_bank import (
    ArrayEmbeddingBank,
    EmbeddingBank,
    EmbeddingBankError,
    build_bank,
    denoise_schedule,
    make_enrollment_audio,
    make_enrollment_variant,
)


def _build(*a, **kw):
    kw.setdefault("p_denoise", 0.0)  # sem denoise real nos testes, salvo onde o teste o fornece
    return build_bank(*a, **kw)


def _identity_denoiser():
    return lambda x: x


def _speech(seconds: float, f0: float, seed: int, sr: int = 48_000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(seed)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
    return (0.2 * env * np.sin(2 * np.pi * f0 * t) + 0.002 * rng.standard_normal(t.size)).astype(np.float32)


def _rec(root, rel, x, sr, source, lic, speaker, split):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, x, sr, subtype="PCM_16")
    return FileRecord(path=rel, sha256=sha256_file(p), source_id=source, license=lic, role="train", speaker=speaker,
                      split=split, sr=sr, num_samples=x.size, band_hz=None, codec="flac", attribution="test")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("bank")
    speech = []
    for i, spk in enumerate(("a", "b", "c")):
        for k in range(4):  # clipes curtos de 3 s a 48 kHz: o enrollment concatena 2-4 deles
            speech.append(_rec(root, f"s/{spk}_{k}.flac", _speech(3.0, 120 + 40 * i, 10 * i + k), 48_000,
                               "vctk-0.92", "CC-BY-4.0", spk, "train"))
    # ruído passa-baixa (média móvel, corte ~2 kHz): fica na banda do filtro de 16 kHz, então o SNR é conservado
    white = np.random.default_rng(0).standard_normal(48_000 * 4)
    lowpass = (0.3 * np.convolve(white, np.ones(24) / 24, "same")).astype("f4")
    noise = [_rec(root, "n/n.flac", lowpass, 48_000, "noise-wikimedia-cc0", "CC0-1.0", None, "noise")]
    h = np.zeros(4800, np.float32)
    h[30], h[500] = 0.9, 0.2
    rirs = [_rec(root, "r/r.flac", h, 48_000, "rir-synthetic-v1", "LicenseRef-cctrain-generated", None, "rir")]
    return root, speech, noise, rirs


def _fake_encoder(calls=None):
    def enc(x16k: np.ndarray) -> np.ndarray:
        if calls is not None:
            calls.append(len(x16k))
        v = np.zeros(192, np.float32)
        v[: 8] = [float(np.mean(x16k)), float(np.std(x16k)), float(np.abs(x16k).max()), len(x16k) / 16_000, 1, 2, 3, 4]
        return v / np.linalg.norm(v)
    return enc


def test_the_bank_is_an_EmbeddingBank():
    bank = ArrayEmbeddingBank({"k": np.ones((3, 192), np.float32)})
    b: EmbeddingBank = bank
    assert b.speakers() == ["k"]
    assert b.sample("k", np.random.default_rng(0)).shape == (192,)
    assert bank.n_variants("k") == 3


def test_make_enrollment_audio_is_valid_16k_of_6_to_12_s_and_deterministic(corpus):
    root, speech, noise, rirs = corpus
    kw = dict(utterances=[r.path for r in speech if r.speaker == "a"], data_root=root, noise=[n.path for n in noise],
              rirs=[r.path for r in rirs], denoise=None)
    for v in range(8):
        a = make_enrollment_audio(np.random.default_rng([1, v]), **kw)
        assert a.dtype == np.float32 and 96_000 <= a.size <= 192_000
        check_enrollment_audio(a)
    x = make_enrollment_audio(np.random.default_rng([1, 0]), **kw)
    assert np.array_equal(x, make_enrollment_audio(np.random.default_rng([1, 0]), **kw))
    assert not np.array_equal(x, make_enrollment_audio(np.random.default_rng([1, 1]), **kw))


def test_augmentation_probabilities_are_honoured(corpus):
    root, speech, noise, rirs = corpus
    utt = [r.path for r in speech if r.speaker == "a"]
    seen = []

    def denoise(x48):
        seen.append(len(x48))
        return x48

    def run(**p):
        seen.clear()
        for v in range(60):
            make_enrollment_audio(np.random.default_rng([2, v]), utterances=utt, data_root=root,
                                  noise=[n.path for n in noise], rirs=[r.path for r in rirs], denoise=denoise, **p)
        return len(seen)

    assert run(p_denoise=0.0, p_noise=0.0, p_rir=0.0) == 0
    assert run(p_denoise=1.0, p_noise=0.0, p_rir=0.0) == 60
    assert 18 <= run(p_denoise=0.5, p_noise=0.0, p_rir=0.0) <= 42


def test_noise_is_added_at_the_drawn_snr_and_skipped_when_p_is_zero(corpus):
    root, speech, noise, rirs = corpus
    utt = [r.path for r in speech if r.speaker == "b"]
    kw = dict(utterances=utt, data_root=root, noise=[n.path for n in noise], rirs=[], denoise=None, p_rir=0.0,
              p_denoise=0.0)
    clean = make_enrollment_audio(np.random.default_rng([3, 0]), p_noise=0.0, snr_db=(10.0, 10.0), **kw)
    noisy = make_enrollment_audio(np.random.default_rng([3, 0]), p_noise=1.0, snr_db=(10.0, 10.0), **kw)
    assert clean.shape == noisy.shape
    resid = noisy.astype(np.float64) - clean
    snr = 10 * np.log10(np.mean(clean.astype(np.float64) ** 2) / np.mean(resid**2))
    assert 9.0 < snr < 11.0


def test_noise_pool_required_when_p_noise_positive(corpus):
    root, speech, _, rirs = corpus
    with pytest.raises(EmbeddingBankError, match="ruído"):
        make_enrollment_audio(np.random.default_rng(0), utterances=[r.path for r in speech[:4]], data_root=root,
                              noise=[], rirs=[], denoise=None, p_noise=0.7)


def test_build_bank_has_20_variants_per_training_speaker_and_only_embeddings(corpus):
    root, speech, noise, rirs = corpus
    calls: list[int] = []
    bank = _build(_fake_encoder(calls), speech, data_root=root, variants=20, noise=noise, rirs=rirs, seed=5)
    assert bank.speakers() == ["vctk-0.92:a", "vctk-0.92:b", "vctk-0.92:c"]
    for k in bank.speakers():
        assert bank.n_variants(k) == 20
        assert bank.sample(k, np.random.default_rng(0)).shape == (192,)
    assert len(calls) == 60 and all(96_000 <= n <= 192_000 for n in calls)
    assert bank.meta["variants_per_speaker"] == 20 and bank.meta["skipped_speakers"] == []


def test_build_bank_is_deterministic_independent_of_input_order(corpus):
    root, speech, noise, rirs = corpus
    a = _build(_fake_encoder(), speech, data_root=root, variants=4, noise=noise, rirs=rirs, seed=9)
    b = _build(_fake_encoder(), list(reversed(speech)), data_root=root, variants=4, noise=noise, rirs=rirs, seed=9)
    for k in a.speakers():
        assert np.array_equal(a.all(k), b.all(k))
    c = _build(_fake_encoder(), speech, data_root=root, variants=4, noise=noise, rirs=rirs, seed=10)
    assert not np.array_equal(a.all("vctk-0.92:a"), c.all("vctk-0.92:a"))


def test_build_bank_resumes_from_the_cache_without_re_encoding(corpus, tmp_path):
    root, speech, noise, rirs = corpus
    calls: list[int] = []
    kw = dict(data_root=root, variants=3, noise=noise, rirs=rirs, seed=1, cache_dir=tmp_path / "cache")
    first = _build(_fake_encoder(calls), [r for r in speech if r.speaker != "c"], **kw)
    assert len(calls) == 6
    calls.clear()
    second = _build(_fake_encoder(calls), speech, **kw)
    assert len(calls) == 3  # só o falante novo
    for k in first.speakers():
        assert np.array_equal(first.all(k), second.all(k))


def test_worker_processes_give_the_same_bank_as_the_inline_run(corpus):
    root, speech, noise, rirs = corpus
    kw = dict(data_root=root, variants=3, noise=noise, rirs=rirs, seed=4)
    inline = _build(_fake_encoder(), speech, workers=0, **kw)
    pooled = _build(_fake_encoder(), speech, workers=2, **kw)
    for k in inline.speakers():
        assert np.array_equal(inline.all(k), pooled.all(k))


def test_cache_from_another_configuration_is_refused(corpus, tmp_path):
    root, speech, noise, rirs = corpus
    kw = dict(data_root=root, noise=noise, rirs=rirs, cache_dir=tmp_path / "cache")
    _build(_fake_encoder(), speech, variants=2, seed=1, **kw)
    with pytest.raises(EmbeddingBankError, match="cache"):
        _build(_fake_encoder(), speech, variants=2, seed=2, **kw)


def test_dev_and_test_speakers_never_enter_the_bank(corpus):
    root, speech, noise, rirs = corpus
    bad = [replace(r, split="dev") if r.speaker == "a" else r for r in speech]
    with pytest.raises(ManifestError):
        _build(_fake_encoder(), bad, data_root=root, variants=2, noise=noise, rirs=rirs)


def test_speaker_without_enough_valid_variants_is_skipped_and_recorded(corpus):
    root, speech, noise, rirs = corpus
    quiet = [replace(r, path=r.path) for r in speech]
    # codificador que nunca é chamado se o áudio é inválido: aqui forçamos recusa pelo ruído a SNR muito baixo
    # não é determinístico; em vez disso, falante com um único clipe silencioso (sem fala ativa) é pulado
    sil = _rec(root, "s/zz_0.flac", np.zeros(48_000 * 3, np.float32), 48_000, "vctk-0.92", "CC-BY-4.0", "zz", "train")
    bank = _build(_fake_encoder(), [*quiet, sil], data_root=root, variants=3, noise=noise, rirs=rirs, seed=1)
    assert "vctk-0.92:zz" not in bank.speakers()
    assert bank.meta["skipped_speakers"] == ["vctk-0.92:zz"]


def test_save_load_roundtrip_and_tampering_is_detected(corpus, tmp_path):
    root, speech, noise, rirs = corpus
    bank = _build(_fake_encoder(), speech, data_root=root, variants=3, noise=noise, rirs=rirs, seed=1)
    path = tmp_path / "bank.npz"
    bank.save(path, extra_meta={"encoder_sha256": "abc"})
    meta = json.loads((tmp_path / "bank.meta.json").read_text())
    assert meta["sha256"] == sha256_file(path) and meta["encoder_sha256"] == "abc" and meta["n_speakers"] == 3
    again = ArrayEmbeddingBank.load(path)
    assert again.speakers() == bank.speakers() and np.array_equal(again.all("vctk-0.92:b"), bank.all("vctk-0.92:b"))
    path.write_bytes(path.read_bytes()[:-8] + b"\0" * 8)
    with pytest.raises(EmbeddingBankError, match="sha256"):
        ArrayEmbeddingBank.load(path)


def test_trim_silence_strips_edges_keeps_a_margin_and_leaves_silence_alone():
    from cctrain.train.embed_bank import trim_silence

    sr = 48_000
    core = _speech(2.0, 150, 1)
    x = np.concatenate([np.zeros(sr, np.float32), core, np.zeros(sr, np.float32)])
    y = trim_silence(x)
    assert 2.0 * sr <= y.size <= 2.0 * sr + 0.25 * sr
    assert np.array_equal(trim_silence(np.zeros(sr, np.float32)), np.zeros(sr, np.float32))
    assert np.array_equal(trim_silence(core), core[: trim_silence(core).size]) or trim_silence(core).size <= core.size


def test_clips_with_long_silences_still_give_valid_enrollments_thanks_to_trimming(tmp_path):
    root = tmp_path
    sr = 48_000
    utt = []
    for k in range(5):
        x = np.concatenate([np.zeros(sr, np.float32), _speech(1.5, 130 + k, k), np.zeros(sr, np.float32)])
        utt.append(_rec(root, f"p/{k}.flac", x, sr, "vctk-0.92", "CC-BY-4.0", "p", "train").path)
    kw = dict(utterances=utt, data_root=root, noise=[], rirs=[], denoise=None, p_noise=0.0, p_rir=0.0, max_tries=1)
    ok = sum(make_enrollment_audio(np.random.default_rng([6, v]), **kw) is not None for v in range(20))
    assert ok == 20


# --- v2: proveniência, exclusão da fala da mistura e controle de p_denoised (M2 Tarefa 2) ---------------------------


def _bank_v2(corpus, variants=20, p_denoise=0.5, **kw):
    root, speech, noise, rirs = corpus
    return build_bank(_fake_encoder(), speech, data_root=root, variants=variants, noise=noise, rirs=rirs, seed=3,
                      p_denoise=p_denoise, denoise_factory=_identity_denoiser, **kw)


def test_variant_provenance_records_utterances_denoise_snr_and_rir(corpus):
    root, speech, noise, rirs = corpus
    utt = [r.path for r in speech if r.speaker == "a"]
    seen = set()
    for v in range(30):
        x = make_enrollment_variant(np.random.default_rng([7, v]), utterances=utt, data_root=root,
                                    noise=[n.path for n in noise], rirs=[r.path for r in rirs], denoise=lambda y: y,
                                    denoise_flag=bool(v % 2))
        assert set(x.utterance_ids) <= set(utt) and len(x.utterance_ids) >= 2  # clipes de 3 s: 6-12 s exigem 2 ou mais
        assert x.denoised == bool(v % 2)
        assert x.snr_db is None or 0.0 <= x.snr_db <= 30.0
        seen.add((x.snr_db is None, x.rir))
    assert len(seen) >= 3  # com p_noise=0,7 e p_rir=0,3, as quatro combinações aparecem; 3 já provam que é registrado


def test_denoise_schedule_is_balanced_deterministic_and_respects_extremes():
    for key in ("k1", "k2", "k3"):
        flags = [denoise_schedule(0, key, v, 0.5) for v in range(20)]
        assert sum(flags) == 10 and flags == [denoise_schedule(0, key, v, 0.5) for v in range(20)]
        assert not any(denoise_schedule(0, key, v, 0.0) for v in range(20))
        assert all(denoise_schedule(0, key, v, 1.0) for v in range(20))
        assert sum(denoise_schedule(0, key, v, 0.25) for v in range(20)) == 5


def test_v2_bank_has_half_denoised_variants_per_speaker_and_stats(corpus):
    bank = _bank_v2(corpus)
    assert bank.has_provenance
    for k in bank.speakers():
        info = bank.variants(k)
        assert len(info) == 20 and int(info.denoised.sum()) == 10
        assert all(len(u) >= 2 for u in info.utterance_ids)
    st = bank.meta["stats"]
    assert st["denoised_fraction_max_abs_dev"] == 0.0 and st["speakers_lt10_raw"] == 0
    assert st["denoised_fraction_mean"] == 0.5


def test_sample_never_returns_a_variant_containing_an_excluded_utterance(corpus):
    bank = _bank_v2(corpus)
    k = "vctk-0.92:a"
    info = bank.variants(k)
    for u in sorted({u for ids in info.utterance_ids for u in ids}):
        ok = info.eligible([u])
        assert ok.any() and not ok.all()  # com 4 clipes de 3 s há variantes com e sem cada fala
        allowed = {bank.all(k)[i].tobytes() for i in np.flatnonzero(ok)}
        banned = {bank.all(k)[i].tobytes() for i in np.flatnonzero(~ok)} - allowed
        rng = np.random.default_rng(0)
        for _ in range(200):
            e = bank.sample(k, rng, exclude_utterances=[u], p_denoised=0.5).tobytes()
            assert e in allowed and e not in banned


def test_sample_p_denoised_zero_gives_only_raw_and_one_only_denoised(corpus):
    bank = _bank_v2(corpus)
    k = "vctk-0.92:b"
    info = bank.variants(k)
    raw = {bank.all(k)[i].tobytes() for i in np.flatnonzero(~info.denoised)} - \
        {bank.all(k)[i].tobytes() for i in np.flatnonzero(info.denoised)}
    den = {bank.all(k)[i].tobytes() for i in np.flatnonzero(info.denoised)} - raw
    rng = np.random.default_rng(1)
    assert all(bank.sample(k, rng, p_denoised=0.0).tobytes() in raw for _ in range(100))
    assert all(bank.sample(k, rng, p_denoised=1.0).tobytes() in den for _ in range(100))
    frac = np.mean([bank.sample(k, rng, p_denoised=0.5).tobytes() in den for _ in range(400)])
    assert 0.4 < frac < 0.6


def test_sample_fails_closed_without_eligible_variants_or_provenance(corpus):
    bank = _bank_v2(corpus, variants=4)
    k = "vctk-0.92:c"
    every = sorted({u for ids in bank.variants(k).utterance_ids for u in ids})
    with pytest.raises(EmbeddingBankError, match="elegível"):
        bank.sample(k, np.random.default_rng(0), exclude_utterances=every)
    v1 = ArrayEmbeddingBank({"k": np.ones((3, 192), np.float32)})
    assert v1.sample("k", np.random.default_rng(0)).shape == (192,)  # sem filtros, como o v1
    with pytest.raises(EmbeddingBankError, match="v2"):
        v1.sample("k", np.random.default_rng(0), exclude_utterances=["x"])
    with pytest.raises(EmbeddingBankError, match="v2"):
        v1.sample("k", np.random.default_rng(0), p_denoised=0.5)


def test_v2_save_load_roundtrip_keeps_provenance(corpus, tmp_path):
    bank = _bank_v2(corpus, variants=6)
    path = tmp_path / "bank-v2.npz"
    meta = bank.save(path)
    assert meta["format"] == 2
    again = ArrayEmbeddingBank.load(path)
    assert again.has_provenance and again.speakers() == bank.speakers()
    for k in bank.speakers():
        a, b = bank.variants(k), again.variants(k)
        assert np.array_equal(bank.all(k), again.all(k)) and a.utterance_ids == b.utterance_ids
        assert np.array_equal(a.denoised, b.denoised) and np.array_equal(a.rir, b.rir)
        assert np.array_equal(a.snr_db, b.snr_db, equal_nan=True)


def test_v2_cache_resume_keeps_provenance_and_is_identical(corpus, tmp_path):
    root, speech, noise, rirs = corpus
    kw = dict(data_root=root, variants=4, noise=noise, rirs=rirs, seed=2, p_denoise=0.5,
              denoise_factory=_identity_denoiser, cache_dir=tmp_path / "c")
    first = build_bank(_fake_encoder(), [r for r in speech if r.speaker != "c"], **kw)
    calls: list[int] = []
    second = build_bank(_fake_encoder(calls), speech, **kw)
    assert len(calls) == 4  # só o falante novo
    full = build_bank(_fake_encoder(), speech, **{**kw, "cache_dir": None})
    for k in full.speakers():
        assert np.array_equal(second.all(k), full.all(k))
        assert second.variants(k).utterance_ids == full.variants(k).utterance_ids
        assert np.array_equal(second.variants(k).denoised, full.variants(k).denoised)
    assert first.speakers() == ["vctk-0.92:a", "vctk-0.92:b"]


def test_denoise_probability_without_a_denoiser_is_refused(corpus):
    root, speech, noise, rirs = corpus
    with pytest.raises(EmbeddingBankError, match="denoise"):
        build_bank(_fake_encoder(), speech, data_root=root, variants=2, noise=noise, rirs=rirs, p_denoise=0.5)
