import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from cctrain.data.manifest import FileRecord
from cctrain.data.mix import power
from cctrain.eval.functional import (
    DEVIATION_V1,
    OFFLINE_DELAY,
    RUNTIME_DELAY,
    DevMixture,
    build_dev_mixtures,
    deviation_gate,
    deviation_metrics,
    enhance_offline,
    functional_gate,
    load_deviation_levels,
    pool_deviation,
    read_librispeech_sex,
    si_sdr,
    speaker_bootstrap,
)
from cctrain.eval.gates import GatesError
from cctrain.eval.runtime_path import HOP, ReplicaResult
from cctrain.hashing import sha256_file

# --- congelamento do gate do desvio ---


def test_deviation_levels_are_frozen_with_the_literal_digest():
    assert sha256_file(DEVIATION_V1) == "2803f83135f94f7967dc7ca01ae182abbaabf0ad8b20e59508ff97b0a0be554f"
    lv = load_deviation_levels()
    assert lv.error_to_signal_db_max == -15.0 and lv.stage_agreement_min == 0.90 and lv.blocking is False


def test_editing_the_levels_breaks_the_freeze(tmp_path: Path):
    p = tmp_path / DEVIATION_V1.name
    shutil.copy(DEVIATION_V1, p)
    shutil.copy(DEVIATION_V1.with_suffix(".sha256"), p.with_suffix(".sha256"))
    p.write_text(p.read_text().replace("-15.0", "-10.0"))
    with pytest.raises(GatesError):
        load_deviation_levels(p)


def test_deviation_gate_picks_offline_only_when_both_levels_pass():
    assert deviation_gate({"error_to_signal_db": -17.54, "stage_agreement": 0.926}) == "offline"  # medida do M0
    assert deviation_gate({"error_to_signal_db": -15.0, "stage_agreement": 0.90}) == "offline"  # nos limites
    assert deviation_gate({"error_to_signal_db": -14.9, "stage_agreement": 0.99}) == "replica"
    assert deviation_gate({"error_to_signal_db": -30.0, "stage_agreement": 0.899}) == "replica"
    with pytest.raises(KeyError):
        deviation_gate({"error_to_signal_db": -30.0})


def test_deviation_gate_never_raises_on_a_bad_measurement():  # não bloqueante: NaN cai na réplica
    assert deviation_gate({"error_to_signal_db": float("nan"), "stage_agreement": 0.99}) == "replica"


def test_deviation_metrics_on_identical_runtime_and_offline():
    rng = np.random.default_rng(0)
    n_frames = 50
    off = (0.1 * rng.standard_normal(n_frames * HOP + OFFLINE_DELAY)).astype(np.float32)
    rt_audio = np.concatenate([np.zeros(RUNTIME_DELAY - OFFLINE_DELAY, np.float32), off])[: n_frames * HOP]
    stages = np.full(n_frames, 2, np.uint8)
    rt = ReplicaResult(rt_audio, np.zeros(n_frames, np.float32), stages.copy())
    m = deviation_metrics(off, torch.from_numpy(stages.astype(np.int64))[None], rt)
    assert m["stage_agreement"] == 1.0 and m["error_to_signal_db"] < -100


def test_pool_deviation_weights_by_energy_and_frames():
    a = {"diff_energy": 1.0, "signal_energy": 100.0, "stage_agreement": 1.0, "stage_agreement_frames": 100,
         "max_abs_diff": 0.1}
    b = {"diff_energy": 9.0, "signal_energy": 100.0, "stage_agreement": 0.5, "stage_agreement_frames": 300,
         "max_abs_diff": 0.2}
    p = pool_deviation([a, b])
    assert abs(p["error_to_signal_db"] - 10 * math.log10(10.0 / 200.0)) < 1e-9
    assert abs(p["stage_agreement"] - (100 * 1.0 + 300 * 0.5) / 400) < 1e-12 and p["max_abs_diff"] == 0.2


# --- métricas ---


def test_si_sdr_is_scale_invariant_and_matches_a_known_ratio():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(48_000)
    noise = rng.standard_normal(48_000)
    est = ref + noise * math.sqrt(power(ref) / power(noise) / 10 ** 1.0)  # 10 dB
    assert abs(si_sdr(est, ref) - 10.0) < 0.2
    assert abs(si_sdr(3.0 * est, ref) - si_sdr(est, ref)) < 1e-6
    assert si_sdr(ref, ref) > 60


def test_bootstrap_unit_is_the_speaker_and_is_reproducible():
    deltas = [10.0] + [0.0] * 99
    speakers = ["a"] + ["b"] * 99
    mean, lo, hi = speaker_bootstrap(deltas, speakers, resamples=2000, seed=0)
    assert mean == 5.0  # média de médias por falante, não por mistura (que daria 0,1)
    assert (mean, lo, hi) == speaker_bootstrap(deltas, speakers, resamples=2000, seed=0)
    assert 0.0 <= lo <= mean <= hi <= 10.0


def test_bootstrap_ci_excludes_zero_for_a_consistent_gain_and_contains_it_for_noise():
    spk = [f"s{i}" for i in range(20) for _ in range(3)]
    gain = [1.0 + 0.1 * (i % 3) for i in range(60)]
    mean, lo, hi = speaker_bootstrap(gain, spk, resamples=2000, seed=1)
    assert lo > 0.0 and hi >= lo
    rng = np.random.default_rng(2)
    noise = list(rng.standard_normal(60))
    _, lo, hi = speaker_bootstrap(noise, spk, resamples=2000, seed=1)
    assert lo < 0.0 < hi


# --- misturas de dev ---


def _write(root, rel, x, sr, speaker, split="dev"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, x.astype("f4"), sr, subtype="PCM_16")
    return FileRecord(path=rel, sha256=sha256_file(p), source_id="librispeech-dev-clean", license="CC-BY-4.0",
                      role="train", speaker=speaker, split=split, sr=sr, num_samples=x.size, band_hz=None,
                      codec="flac", attribution="test")


@pytest.fixture()
def dev_corpus(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(16_000 * 14) / 16_000
    spk = {"f1": "F", "f2": "F", "f3": "F", "m1": "M", "m2": "M"}
    recs = []
    for i, s in enumerate(spk):
        for k in range(4):
            x = 0.1 * np.sin(2 * np.pi * (120 + 40 * i) * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
            recs.append(_write(tmp_path, f"d/{s}_{k}.flac", x + 0.01 * rng.standard_normal(t.size), 16_000, s))
    noise = [_write(tmp_path, "n/n.flac", 0.05 * rng.standard_normal(48_000 * 4), 48_000, None, split="noise")]
    sex = {f"librispeech-dev-clean:{s}": g for s, g in spk.items()}
    return tmp_path, recs, noise, sex


def _embed(a16k):
    v = np.zeros(192, np.float32)
    v[: 4] = [float(np.mean(a16k)), float(np.std(a16k)), 1.0, 0.0]
    return v / np.linalg.norm(v)


def test_dev_mixtures_are_same_sex_sir_zero_and_disjoint_enrollment(dev_corpus):
    root, recs, noise, sex = dev_corpus
    mixes = build_dev_mixtures(recs, noise, root, sex, _embed, n_mixtures=12, seed=0)
    assert len(mixes) == 12
    for m in mixes:
        assert isinstance(m, DevMixture)
        assert m.noisy.shape == m.clean.shape == (3 * 48_000,)
        assert sex[m.speaker] == sex[m.interferer_speaker] and m.speaker != m.interferer_speaker
        assert m.sir_db == 0.0 and m.embedding.shape == (192,)
        assert m.enrollment_paths and m.mixture_path not in m.enrollment_paths
    assert len({m.speaker for m in mixes}) >= 3
    again = build_dev_mixtures(recs, noise, root, sex, _embed, n_mixtures=12, seed=0)
    assert all(np.array_equal(a.noisy, b.noisy) for a, b in zip(mixes, again))


def test_c3_has_no_noise_and_c4_has_noise_at_the_requested_snr(dev_corpus):
    root, recs, noise, sex = dev_corpus
    c3 = build_dev_mixtures(recs, [], root, sex, _embed, n_mixtures=4, seed=0)  # C3 é o padrão e dispensa ruído
    c4 = build_dev_mixtures(recs, noise, root, sex, _embed, n_mixtures=4, condition="C4", snr_db=10.0, seed=0)
    for m in c3:
        assert m.snr_db is None and m.sir_db == 0.0
        # sem ruído: noisy = alvo + interferente; a SIR medida contra o alvo seco (sem RIR) é 0 dB
        interf_part = m.noisy - m.clean
        assert abs(10 * np.log10(power(m.clean) / power(interf_part))) < 0.05
    assert all(m.snr_db == 10.0 for m in c4)
    # o ruído é medido contra o alvo: com o interferente e o ruído somados, o resíduo supera o de C3
    assert np.mean([power(m.noisy - m.clean) for m in c4]) > np.mean([power(m.noisy - m.clean) for m in c3])
    with pytest.raises(ValueError, match="ruído"):
        build_dev_mixtures(recs, [], root, sex, _embed, n_mixtures=2, condition="C4", seed=0)
    with pytest.raises(ValueError, match="condition"):
        build_dev_mixtures(recs, noise, root, sex, _embed, n_mixtures=2, condition="C5", seed=0)


def test_dev_mixtures_refuse_test_and_train_splits(dev_corpus):
    root, recs, noise, sex = dev_corpus
    bad = [FileRecord(**{**r.to_dict(), "split": "test"}) for r in recs]
    with pytest.raises(ValueError, match="split"):
        build_dev_mixtures(bad, noise, root, sex, _embed, n_mixtures=2, seed=0)


def test_librispeech_sex_parser(tmp_path):
    p = tmp_path / "SPEAKERS.TXT"
    p.write_text(";ID  |SEX| SUBSET           |MINUTES| NAME\n14   | F | train-clean-360  | 25.03 | Kristin\n"
                 "17   | M | dev-clean        |  8.0  | Gord\n")
    assert read_librispeech_sex(p) == {"librispeech-dev-clean:14": "F", "librispeech-dev-clean:17": "M"}


# --- gate funcional ---


@pytest.fixture(scope="module")
def base_model():
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3

    torch.set_num_threads(2)
    return load_upstream_dfnet3(device="cpu")


def _mixes(n=4, seconds=1.0):
    rng = np.random.default_rng(3)
    out = []
    for i in range(n):
        t = np.arange(int(48_000 * seconds)) / 48_000.0
        clean = (0.1 * np.sin(2 * np.pi * (150 + 30 * i) * t)).astype(np.float32)
        noisy = (clean + 0.03 * rng.standard_normal(t.size)).astype(np.float32)
        emb = np.eye(192, dtype=np.float32)[i]
        out.append(DevMixture(noisy=noisy, clean=clean, speaker=f"s{i % 2}", interferer_speaker="x", embedding=emb,
                              sir_db=0.0, snr_db=10.0, mixture_path="m", enrollment_paths=("e",)))
    return out


def _ckpt(tmp_path, out_scale):
    from cctrain.models.film import FilmGenerator, load_film_config
    from cctrain.train.checkpoint import save_checkpoint

    gen = FilmGenerator(load_film_config())
    if out_scale:
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            gen.out.weight.copy_(torch.randn(gen.out.weight.shape, generator=g) * out_scale)
    opt = torch.optim.AdamW(gen.parameters())
    return save_checkpoint(tmp_path, step=1, model=gen.state_dict(), optimizer=opt, sampler_state={},
                           cfg_hash="x", commit=None)


def test_identity_generator_gives_exactly_zero_delta_offline(base_model, tmp_path):
    base, _ = base_model
    r = functional_gate(_ckpt(tmp_path, 0.0), _mixes(), base=base, path="offline", resamples=200)
    assert r.path == "offline" and r.n_mixtures == 4 and r.n_speakers == 2
    assert r.delta_si_sdr_db == 0.0 and r.ci_low == 0.0 and r.ci_high == 0.0
    assert r.passed is False  # o gate exige limite inferior > 0, não >= 0
    assert abs(sum(r.stage_fractions.values()) - 1.0) < 1e-9


def test_non_identity_generator_changes_the_output(base_model, tmp_path):
    base, _ = base_model
    r = functional_gate(_ckpt(tmp_path, 0.3), _mixes(), base=base, path="offline", resamples=200)
    assert r.delta_si_sdr_db != 0.0 and math.isfinite(r.delta_si_sdr_db)


def test_replica_path_uses_the_runtime_for_candidate_and_baseline(base_model, tmp_path):
    base, _ = base_model
    calls = []

    def fake_runtime(audio, film):
        calls.append(film is None)
        n = len(audio) // HOP
        out = np.concatenate([np.zeros(RUNTIME_DELAY, np.float32), audio])[: n * HOP]
        return ReplicaResult(out, np.zeros(n, np.float32), np.full(n, 3, np.uint8))

    r = functional_gate(_ckpt(tmp_path, 0.3), _mixes(), base=base, path="replica", runtime=fake_runtime,
                        resamples=200)
    assert r.path == "replica" and calls.count(True) == 4 and calls.count(False) == 4  # base sem film, candidato com
    assert r.delta_si_sdr_db == 0.0  # o runtime falso devolve a mesma coisa nos dois lados
    assert r.stage_fractions["none"] == 1.0


def test_replica_path_with_worker_threads_gives_the_same_result(base_model, tmp_path):
    base, _ = base_model

    def runtime(audio, film):
        n = len(audio) // HOP
        gain = 1.0 if film is None else 0.5 + float(film[0])
        out = np.concatenate([np.zeros(RUNTIME_DELAY, np.float32), gain * audio])[: n * HOP]
        return ReplicaResult(out, np.zeros(n, np.float32), np.full(n, 3, np.uint8))

    kw = dict(base=base, path="replica", runtime=runtime, resamples=200)
    a = functional_gate(_ckpt(tmp_path, 0.3), _mixes(), **kw)
    b = functional_gate(_ckpt(tmp_path, 0.3), _mixes(), workers=3, **kw)
    assert a.n_mixtures == b.n_mixtures and a.stage_fractions == b.stage_fractions
    for f in ("delta_si_sdr_db", "ci_low", "ci_high", "candidate_si_sdr_db", "baseline_si_sdr_db"):
        assert abs(getattr(a, f) - getattr(b, f)) < 1e-6, f


def test_replica_path_requires_a_runtime(base_model, tmp_path):
    base, _ = base_model
    with pytest.raises(ValueError, match="runtime"):
        functional_gate(_ckpt(tmp_path, 0.0), _mixes(1), base=base, path="replica")


@pytest.mark.data
@pytest.mark.clearcore
def test_deviation_metrics_reproduce_the_m0_measurement_and_pick_offline(base_model, paths, m1_runs):
    """Mesma mistura e mesmo modelo do M0: `deviation_metrics` bate com `offline_vs_runtime` e o gate escolhe offline."""
    import json

    from cctrain.eval.runtime_path import StatefulOrtFrameModel, parity_mixture, run_replica
    from cctrain.m0.offline_probe import offline_vs_runtime
    from cctrain.paths import clearcore_root

    base, _ = base_model
    x = parity_mixture(paths.data)
    rt = run_replica(x, StatefulOrtFrameModel(clearcore_root() / "models/stateful"))
    ref = offline_vs_runtime(base, x, rt)
    y, stages = enhance_offline(base, x)
    mine = deviation_metrics(y, stages, rt)
    for k in ("max_abs_diff", "error_to_signal_db", "stage_agreement", "stage_agreement_frames"):
        assert mine[k] == pytest.approx(ref[k], rel=1e-9), k
    assert deviation_gate(mine) == "offline"
    (m1_runs / "t11_deviation_base.json").write_text(json.dumps(
        {**mine, "gate": deviation_gate(mine), "levels_sha256": load_deviation_levels().sha256}, indent=2) + "\n")
