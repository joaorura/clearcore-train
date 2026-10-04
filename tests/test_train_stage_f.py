import json
import math
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
import torch

from cctrain.data.manifest import FileRecord, ManifestError
from cctrain.data.mix import MixKind
from cctrain.hashing import sha256_file
from cctrain.m2.config import load_selection
from cctrain.models.state_hash import state_dict_sha256
from cctrain.train.checkpoint import (
    CheckpointError,
    list_checkpoints,
    load_checkpoint,
    load_latest,
    pinned_steps,
    save_checkpoint,
)
from cctrain.train.embed_bank import EmbeddingBankError
from cctrain.train.stage_c import StageCData
from cctrain.train.stage_f import (
    TIERS,
    BatchNormDriftError,
    InitError,
    StageFDataset,
    StageFHParams,
    bn_state_sha256,
    branch,
    build_models,
    load_init,
    lowpass48,
    n_batchnorm,
    prepare_student,
    run_stage_f,
    slot_tiers,
    stage_f_step,
    tier_of,
)

SPEAKERS = {  # speaker -> (sr, source)
    "f1": 48_000, "f2": 48_000, "m1": 48_000, "m2": 48_000,
    "a1": 24_000, "a2": 24_000, "b1": 16_000, "b2": 16_000,
}


def _write(root, rel, x, sr, source, lic, speaker, split, band=None):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, x.astype("f4"), sr, subtype="PCM_16")
    return FileRecord(path=rel, sha256=sha256_file(p), source_id=source, license=lic, role="train", speaker=speaker,
                      split=split, sr=sr, num_samples=x.size, band_hz=band, codec="flac", attribution="test")


class ProvBank:
    """Banco v2 falso: 6 variantes por falante; a variante `v` do falante `k` contém a fala `k_v` (v < 3) ou `k_(v-3)`
    e `k_(v-2)`; o embedding identifica a variante (one-hot em 192-d)."""

    has_provenance = True

    def __init__(self, keys, paths_of):
        self.keys = sorted(keys)
        self.paths_of = paths_of
        self.calls: list[dict] = []

    def speakers(self):
        return self.keys

    def variant_paths(self, key, v):
        ps = self.paths_of[key]
        return {ps[v % len(ps)], ps[(v + 1) % len(ps)]}

    def sample(self, key, rng, *, exclude_utterances=(), p_denoised=None):
        ex = set(exclude_utterances)
        ok = [v for v in range(6) if not (self.variant_paths(key, v) & ex)
              and (p_denoised is None or (v % 2 == 1) == (rng.random() < p_denoised) or p_denoised in (0.0, 1.0))]
        if p_denoised == 0.0:
            ok = [v for v in ok if v % 2 == 0]
        if p_denoised == 1.0:
            ok = [v for v in ok if v % 2 == 1]
        self.calls.append({"key": key, "exclude": ex, "p": p_denoised})
        if not ok:
            raise EmbeddingBankError(f"{key}: nenhuma variante elegível")
        v = ok[int(rng.integers(len(ok)))]
        e = np.zeros(192, np.float32)
        e[(self.keys.index(key) * 6 + v) % 192] = 1.0
        return e


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("stagef")
    rng = np.random.default_rng(0)
    speech, paths_of = [], {}
    for i, (spk, sr) in enumerate(SPEAKERS.items()):
        t = np.arange(sr * 3) / sr
        paths_of[f"vctk-0.92:{spk}"] = []
        for k in range(3):
            env = 0.5 + 0.5 * np.sin(2 * np.pi * (2 + k) * t) ** 2
            x = 0.1 * env * np.sin(2 * np.pi * (110 + 35 * i) * t) + 0.003 * rng.standard_normal(t.size)
            r = _write(root, f"s/{spk}_{k}.flac", x, sr, "vctk-0.92", "CC-BY-4.0", spk, "pilot", band=min(7_000.0, sr / 2))
            speech.append(r)
            paths_of[f"vctk-0.92:{spk}"].append(r.path)
    noise = [_write(root, "n/n.flac", 0.05 * rng.standard_normal(48_000 * 4), 48_000, "noise-wikimedia-cc0",
                    "CC0-1.0", None, "noise")]
    h = np.zeros(4800)
    h[40], h[400] = 0.9, 0.2
    rirs = [_write(root, "r/r.flac", h, 48_000, "rir-synthetic-v1", "LicenseRef-cctrain-generated", None, "rir")]
    return root, StageCData(speech, noise, rirs, None), ProvBank(paths_of.keys(), paths_of)


def _hp(tmp_path, **kw):
    base = StageFHParams(run_id="t", seed=3, batch_size=8, segment_s=1.0, steps_per_epoch=3, epochs=2, warmup_steps=3,
                         device="cpu", num_workers=0, checkpoint_every_min=1e9, runs_dir=tmp_path, pin_epoch=None,
                         gpu_log_interval_s=1e9)
    return replace(base, **kw)


# --- amostragem -----------------------------------------------------------------------------------------------------


def test_slot_tiers_follow_the_band_quota_for_the_batch_sizes_in_use():
    q = (0.5, 0.3, 0.2)
    for b in (2, 4, 8, 16, 32):
        s = slot_tiers(b, q)
        assert len(s) == b and s.count("full") >= 0.5 * b and s.count("khz16") <= 0.2 * b + 1e-9
    s16 = slot_tiers(16, q)
    assert (s16.count("full"), s16.count("khz24"), s16.count("khz16")) == (8, 5, 3)
    assert (tier_of(48_000), tier_of(32_000), tier_of(24_000), tier_of(16_000)) == ("full", "khz24", "khz24", "khz16")


def test_every_batch_meets_the_band_quota_and_tier_follows_the_native_rate(corpus, tmp_path):
    root, data, bank = corpus
    hp = _hp(tmp_path, s=0.0, p_drop=0.0)
    ds = StageFDataset(data, bank, root, hp, epoch=0)
    for first in (0, 8, 16):
        tiers = [TIERS[int(ds[first + k]["tier"])] for k in range(8)]
        assert tiers.count("full") >= 4 and tiers.count("khz16") <= 1 and tiers.count("khz24") >= 2
    it = ds[0]
    assert TIERS[int(it["tier"])] == "full" and tier_of(48_000) == "full"
    rate = {r.path: r.sr for r in data.speech}
    assert rate[it["target_path"]] == 48_000


def test_dataset_item_contract_and_determinism(corpus, tmp_path):
    root, data, bank = corpus
    hp = _hp(tmp_path)
    ds = StageFDataset(data, bank, root, hp, epoch=0)
    assert len(ds) == 24
    it = ds[0]
    assert it["spec"].shape == (1, 100, 481, 2) and it["spec_clean"].shape == (1, 100, 481, 2)
    assert it["embedding"].shape == (192,) and it["mask"].shape == (1,) and it["lsnr_mask"].shape == (100,)
    assert np.isfinite(it["spec"]).all() and 0 < it["max_freq"] <= 24_000
    again = StageFDataset(data, bank, root, hp, epoch=0)[5]
    assert np.array_equal(ds[5]["spec"], again["spec"]) and np.array_equal(ds[5]["embedding"], again["embedding"])
    assert not np.array_equal(ds[5]["spec"], StageFDataset(data, bank, root, hp, epoch=1)[5]["spec"])
    assert not np.array_equal(ds[5]["spec"], StageFDataset(data, bank, root, replace(hp, seed=4), epoch=0)[5]["spec"])


def test_embedding_dropout_decides_mask_target_and_gate_frames(corpus, tmp_path):
    root, data, bank = corpus
    none = StageFDataset(data, bank, root, _hp(tmp_path, p_drop=0.0, s=0.0), epoch=0)
    allm = StageFDataset(data, bank, root, _hp(tmp_path, p_drop=1.0, s=0.0), epoch=0)
    for i in range(10):
        assert float(none[i]["mask"][0]) == 1.0 and float(allm[i]["mask"][0]) == 0.0
        assert not allm[i]["interferer_frames"].any() and allm[i]["lsnr_mask"].all()  # sem perfil: sem L_gate
    # mesma amostra, com e sem perfil: o alvo sem perfil inclui o interferente (soma das falas secas)
    seen_with_interferer, ratios = 0, []
    for i in range(40):
        a, b = none[i], allm[i]
        if MixKind(int(a["kind"])) in (MixKind.TARGET_INTERFERER, MixKind.TARGET_INTERFERER_NOISE):
            seen_with_interferer += 1
            ratios.append(np.sum(b["spec_clean"] ** 2) / np.sum(a["spec_clean"] ** 2))
        else:
            assert np.allclose(a["spec_clean"], b["spec_clean"], atol=1e-6)
    assert seen_with_interferer >= 3 and min(ratios) > 1.0 and max(ratios) > 1.2


def test_no_target_class_has_silent_target_with_profile_and_the_sum_of_speakers_without(corpus, tmp_path):
    root, data, bank = corpus
    with_p = StageFDataset(data, bank, root, _hp(tmp_path, p_drop=0.0, s=0.999), epoch=0)
    without = StageFDataset(data, bank, root, _hp(tmp_path, p_drop=1.0, s=0.999), epoch=0)
    kinds = set()
    for i in range(12):
        a, b = with_p[i], without[i]
        if MixKind(int(a["kind"])) is MixKind.INTERFERER_ONLY:
            kinds.add("no_target")
            assert not a["spec_clean"].any() and not a["lsnr_mask"].any() and a["interferer_frames"].any()
            assert a["target_path"] == ""
            assert b["spec_clean"].any() and b["lsnr_mask"].all()
    assert kinds == {"no_target"}


def test_enrollment_never_comes_from_the_mixture_utterance_and_p_denoised_is_forwarded(corpus, tmp_path):
    root, data, bank = corpus
    bank.calls.clear()
    ds = StageFDataset(data, bank, root, _hp(tmp_path, s=0.0, p_enroll_denoise=0.25), epoch=0)
    for i in range(16):
        it = ds[i]
        k = bank.keys.index(it["target_key"])
        hot = int(np.argmax(it["embedding"]))
        v = (hot - k * 6) % 192
        assert it["target_path"] not in bank.variant_paths(it["target_key"], v)
    assert bank.calls and all(c["p"] == 0.25 for c in bank.calls)
    assert all(c["exclude"] for c in bank.calls)


def test_a_speaker_with_no_eligible_variant_is_redrawn_and_a_v1_bank_is_still_accepted(corpus, tmp_path):
    root, data, bank = corpus

    class Stingy(ProvBank):
        def sample(self, key, rng, **kw):
            if key == "vctk-0.92:f1":
                raise EmbeddingBankError("sem variante")
            return super().sample(key, rng, **kw)

    st = Stingy(bank.keys, bank.paths_of)
    ds = StageFDataset(data, st, root, _hp(tmp_path, s=0.0), epoch=0)
    assert all(ds[i]["target_key"] != "vctk-0.92:f1" for i in range(8))

    class V1:
        def speakers(self):
            return bank.keys

        def sample(self, key, rng):
            return np.ones(192, np.float32)

    assert StageFDataset(data, V1(), root, _hp(tmp_path), epoch=0)[0]["embedding"].shape == (192,)


def test_dev_records_and_missing_band_tiers_fail_closed(corpus, tmp_path):
    root, data, bank = corpus
    bad = [FileRecord(**{**r.to_dict(), "split": "dev"}) for r in data.speech[:4]] + list(data.speech[4:])
    with pytest.raises(ManifestError):
        StageFDataset(StageCData(bad, data.noise, data.rirs), bank, root, _hp(tmp_path), epoch=0)
    only48 = [r for r in data.speech if r.sr == 48_000]
    with pytest.raises(ValueError, match="cota de banda"):
        StageFDataset(StageCData(only48, data.noise, data.rirs), bank, root, _hp(tmp_path), epoch=0)


def test_noise_is_lowpassed_to_the_native_band_of_limited_speech():
    x = np.random.default_rng(0).standard_normal(48_000).astype(np.float32)
    y = lowpass48(x, 8_000.0 * 0.98)
    spec = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(y.size, 1 / 48_000)
    assert spec[f > 10_000].sum() < 1e-4 * spec[f < 6_000].sum()
    assert lowpass48(x, 24_000.0) is x


# --- modelo e perda ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models():
    return build_models(StageFHParams(device="cpu"), "cpu")


@pytest.fixture(scope="module")
def batch(corpus, tmp_path_factory):
    root, data, bank = corpus
    ds = StageFDataset(data, bank, root, _hp(tmp_path_factory.mktemp("b"), batch_size=4, p_drop=0.0, s=0.0), epoch=0)
    items = [ds[i] for i in range(4)]
    return {k: torch.as_tensor(np.stack([it[k] for it in items])) for k in
            ("spec", "feat_erb", "feat_spec", "spec_clean", "embedding", "mask", "interferer_frames", "lsnr_mask",
             "max_freq")}


def _forward(model, b, mask):
    return model(b["spec"].clone(), b["feat_erb"], b["feat_spec"], b["embedding"], mask)


def test_step0_output_with_zero_film_or_without_profile_is_bit_identical_to_upstream_dfnet3(models, batch):
    model, teacher, _ = models
    prepare_student(model)
    b = batch
    ref = teacher(b["spec"].clone(), b["feat_erb"], b["feat_spec"])
    with torch.no_grad():
        # gerador zerado (inicial), com perfil
        out = _forward(model, b, torch.ones(4, 1))
        for x, y in zip(out[:3], ref[:3], strict=True):
            assert torch.equal(x, y)
        # gerador qualquer, sem perfil
        saved = {k: v.clone() for k, v in model.film.state_dict().items()}
        g = torch.Generator().manual_seed(0)
        for p in model.film.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
        assert float(model.film(b["embedding"], torch.ones(4, 1)).gamma_enc.sub(1.0).abs().max()) > 0.1  # gerador vivo
        out0 = _forward(model, b, torch.zeros(4, 1))
        for x, y in zip(out0[:3], ref[:3], strict=True):
            assert torch.equal(x, y)
        model.film.load_state_dict(saved)


def test_distillation_term_is_zero_at_step0_and_only_counts_mask0_rows(models, batch):
    from df.loss import Istft
    from df.model import ModelParams

    model, teacher, df_state = models
    istft = Istft(960, 480, torch.as_tensor(df_state.fft_window().copy()))
    hp = StageFHParams(device="cpu")
    prepare_student(model)
    mixed = {**batch, "mask": torch.tensor([[1.0], [0.0], [1.0], [0.0]])}
    mixed["interferer_frames"] = batch["interferer_frames"] & (mixed["mask"] > 0)
    parts = stage_f_step(model, teacher, istft, ModelParams(), hp, mixed)
    # aluno == professor; o resíduo (~1e-12 na escala 500) vem do tamanho de lote diferente nos kernels (4 x 2 linhas)
    assert float(parts["t"].detach()) < 1e-8 and float(parts["n0"]) == 2
    allp = stage_f_step(model, teacher, istft, ModelParams(), hp, {**batch, "mask": torch.ones(4, 1)})
    assert float(allp["t"].detach()) == 0.0 and float(allp["n0"]) == 0
    base_sha = state_dict_sha256(model.base)
    with torch.no_grad():
        for p in model.base.parameters():
            p.add_(1e-3 * torch.randn_like(p))
    try:
        moved = stage_f_step(model, teacher, istft, ModelParams(), hp, mixed)
        assert float(moved["t"].detach()) > 1e-6
        other = {**mixed, "spec": mixed["spec"].clone(), "feat_erb": mixed["feat_erb"].clone(),
                 "feat_spec": mixed["feat_spec"].clone()}
        for k in ("spec", "feat_erb", "feat_spec"):
            other[k][[0, 2]] = other[k][[0, 2]].flip(-2) * 0.5  # troca só as linhas com perfil
        moved2 = stage_f_step(model, teacher, istft, ModelParams(), hp, other)
        assert float(moved2["t"].detach()) == pytest.approx(float(moved["t"].detach()), rel=1e-4)
        moved["total"].backward()
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.base.parameters())
    finally:
        model.zero_grad(set_to_none=True)
    assert state_dict_sha256(model.base) != base_sha


def test_prepare_student_freezes_only_the_batchnorm_and_keeps_the_rest_trainable(models):
    model, _, _ = models
    prepare_student(model)
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns and n_batchnorm(model) == len(bns)
    assert not any(m.training or any(p.requires_grad for p in m.parameters()) for m in bns)
    assert sum(p.requires_grad for p in model.base.parameters()) > 0 and all(p.requires_grad for p in model.film.parameters())
    assert all(m.training for m in model.modules() if isinstance(m, torch.nn.RNNBase))


def test_init_checks_the_generator_checkpoint_sha_and_accepts_stage_c_and_f_formats(models, tmp_path):
    model, _, _ = models
    gen = {k: torch.randn_like(v) for k, v in model.film.state_dict().items()}
    opt = torch.optim.AdamW(model.film.parameters())
    p = save_checkpoint(tmp_path / "c", step=1, model=gen, optimizer=opt, sampler_state={}, cfg_hash="x", commit=None)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    digest = load_init(model, p, expected_sha256=sha256_file(p))
    assert digest == sha256_file(p) and torch.equal(model.film.out.weight, gen["out.weight"])
    with pytest.raises(InitError):
        load_init(model, p, expected_sha256="0" * 64)
    pf = save_checkpoint(tmp_path / "f", step=1, model=saved, optimizer=opt, sampler_state={}, cfg_hash="x", commit=None)
    load_init(model, pf)
    assert torch.equal(model.film.out.weight, saved["film.out.weight"])


# --- rodada: BN congelada, retomada, ramo, avaliação -----------------------------------------------------------------


def _run(corpus, hp, **kw):
    root, data, bank = corpus
    return run_stage_f(replace(hp, data_root=root), bank, data, max_hours=1.0, init=kw.pop("init", None), **kw)


@pytest.fixture(scope="module")
def generator_init(models, tmp_path_factory):
    model, _, _ = models
    g = torch.Generator().manual_seed(11)
    state = {k: 0.05 * torch.randn(v.shape, generator=g) for k, v in model.film.state_dict().items()}
    opt = torch.optim.AdamW(model.film.parameters())
    return save_checkpoint(tmp_path_factory.mktemp("init"), step=1, model=state, optimizer=opt, sampler_state={},
                           cfg_hash="x", commit=None)


def test_run_trains_the_base_but_keeps_batchnorm_statistics_and_weights_identical(corpus, tmp_path, generator_init):
    hp = _hp(tmp_path, max_steps=4)
    ckpt = _run(corpus, hp, init=generator_init)
    summary = json.loads((tmp_path / "m2" / "t" / "summary.json").read_text())
    assert summary["steps"] == 4 and summary["n_batchnorm_modules"] > 0
    assert summary["bn_sha256_before"] == summary["bn_sha256_after"]
    ck = load_checkpoint(ckpt)
    fresh, _, _ = build_models(StageFHParams(device="cpu"), "cpu")
    fresh_sd, now = fresh.state_dict(), ck.model
    bn_keys = [k for k in now if any(t in k for t in ("running_mean", "running_var", "num_batches_tracked"))]
    assert bn_keys and all(torch.equal(now[k], fresh_sd[k]) for k in bn_keys)
    moved = [k for k in now if k.startswith("base.") and k not in bn_keys and not torch.equal(now[k], fresh_sd[k])]
    assert len(moved) > 10  # a base treinou
    log = [json.loads(line) for line in (tmp_path / "m2" / "t" / "loss-log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3, 4] and all(math.isfinite(r["total"]) for r in log)
    assert {"mr", "os", "gate", "lsnr", "t", "n0", "lr"} <= set(log[0])
    assert summary["trainable_params"] > summary["config"]["batch_size"]


def test_batchnorm_hash_detects_drift(models):
    model, _, _ = models
    before = bn_state_sha256(model)
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    saved = bn.running_mean.clone()
    with torch.no_grad():
        bn.running_mean.add_(1e-3)
    assert bn_state_sha256(model) != before
    with torch.no_grad():
        bn.running_mean.copy_(saved)
    assert bn_state_sha256(model) == before and issubclass(BatchNormDriftError, RuntimeError)


def test_resume_reproduces_the_loss_and_the_weights_bit_for_bit_for_20_steps_on_cpu(corpus, tmp_path, generator_init):
    base = _hp(tmp_path / "a", run_id="straight", max_steps=20, steps_per_epoch=3, epochs=10, batch_size=4, p_drop=0.5)
    _run(corpus, base, init=generator_init)
    first = replace(base, runs_dir=tmp_path / "b", run_id="resumed", max_steps=10)
    _run(corpus, first, init=generator_init)
    second = replace(first, max_steps=20)
    last = _run(corpus, second)  # sem init: retoma do checkpoint

    def losses(hp):
        lines = (hp.runs_dir / "m2" / hp.run_id / "loss-log.jsonl").read_text().splitlines()
        return {json.loads(x)["step"]: json.loads(x) for x in lines}

    a, b = losses(base), losses(second)
    assert sorted(a) == sorted(b) == list(range(1, 21))
    for s in range(1, 21):
        assert a[s]["total"] == b[s]["total"] and a[s]["t"] == b[s]["t"], s
    ca = load_latest(tmp_path / "a" / "m2" / "straight" / "checkpoints")
    cb = load_checkpoint(last)
    assert ca.step == cb.step == 20
    assert all(torch.equal(ca.model[k], cb.model[k]) for k in ca.model)
    assert sum(float(r["n0"]) for r in a.values()) > 0  # a destilação foi exercida


def test_resume_with_another_config_is_refused_and_no_resume_refuses_existing_checkpoints(corpus, tmp_path,
                                                                                         generator_init):
    hp = _hp(tmp_path, max_steps=2)
    _run(corpus, hp, init=generator_init)
    with pytest.raises(CheckpointError):
        _run(corpus, replace(hp, max_steps=4, lr=2e-4))
    root, data, bank = corpus
    with pytest.raises(CheckpointError, match="resume=False"):
        run_stage_f(replace(hp, data_root=root), bank, data, max_hours=1.0, resume=False)


def test_time_budget_counts_the_time_before_a_resume(corpus, tmp_path, generator_init):
    hp = _hp(tmp_path, max_steps=2, steps_per_epoch=10)
    first = _run(corpus, hp, init=generator_init)
    spent = load_checkpoint(first).extra["elapsed_s"]
    assert spent > 0
    root, data, bank = corpus
    last = run_stage_f(replace(hp, max_steps=100, data_root=root), bank, data, max_hours=spent / 7200.0)
    assert load_checkpoint(last).step == 3


def test_epoch_checkpoints_pin_the_chosen_epoch_and_a_branch_continues_the_cosine(corpus, tmp_path, generator_init):
    root, data, bank = corpus
    hp = _hp(tmp_path, run_id="main", epochs=4, steps_per_epoch=2, batch_size=4, keep_last=1, pin_epoch=2,
             data_root=root)
    run_stage_f(replace(hp, stop_epoch=3), bank, data, max_hours=1.0, init=generator_init)
    ckdir = tmp_path / "m2" / "main" / "checkpoints"
    assert pinned_steps(ckdir) == {4} and (ckdir / "ckpt-00000004.pt").is_file()  # a época 2 sobrevive à poda
    pinned = ckdir / "ckpt-00000004.pt"
    out = branch(pinned, {"p_drop": 0.5, "s": 0.1}, 1, bank=bank, data=data, run_id="ab2-x", max_hours=1.0,
                 runs_dir=tmp_path, data_root=root, device="cpu", num_workers=0, gpu_log_interval_s=1e9)
    ck = load_checkpoint(out)
    assert ck.step == 6 and ck.sampler_state == {"epoch": 3, "batch": 0}
    assert ck.extra["hp"]["p_drop"] == 0.5 and ck.extra["hp"]["s"] == 0.1 and ck.extra["init_sha256"] == sha256_file(pinned)
    log = [json.loads(x) for x in (tmp_path / "m2" / "ab2-x" / "loss-log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [5, 6]
    assert log[0]["lr"] == pytest.approx(1e-4 * 0.5 * (1 + math.cos(math.pi * 5 / 8)), rel=1e-6)  # cosseno do pai
    with pytest.raises(ValueError, match="override"):
        branch(pinned, {"lr": 1.0}, 1, bank=bank, data=data, run_id="ab2-bad", max_hours=1.0, runs_dir=tmp_path)
    with pytest.raises(CheckpointError):  # ramo parte de fim de época, não de checkpoint de meio de época
        mid = save_checkpoint(tmp_path / "mid", step=3, model=ck.model, optimizer=torch.optim.AdamW(
            [torch.nn.Parameter(torch.zeros(1))]), sampler_state={"epoch": 1, "batch": 1}, cfg_hash="x", commit=None,
            extra={"hp": ck.extra["hp"]})
        branch(mid, {"seed": 1}, 1, bank=bank, data=data, run_id="ab2-mid", max_hours=1.0, runs_dir=tmp_path)


def test_evaluation_hook_runs_before_the_epoch_checkpoint_and_drives_best_and_early_stopping(corpus, tmp_path,
                                                                                            generator_init):
    root, data, bank = corpus
    rule = load_selection()
    rule = replace(rule, early_stopping=replace(rule.early_stopping, from_epoch=1, patience_evaluations=2))
    scores = {1: 1.0, 2: 3.0, 3: 3.0, 4: 3.0, 5: 3.0}
    calls = []

    def evaluator(model, epoch):
        calls.append(epoch)
        return {"c0": {"delta_si_sdr_db": 0.0, "delta_stoi": 0.0}, "c3": {"delta_si_sdr_db": scores[epoch]}}

    hp = _hp(tmp_path, run_id="ev", epochs=5, steps_per_epoch=2, batch_size=4, keep_last=1, data_root=root)
    last = run_stage_f(hp, bank, data, max_hours=1.0, init=generator_init, evaluator=evaluator, rule=rule)
    summary = json.loads((tmp_path / "m2" / "ev" / "summary.json").read_text())
    assert calls == [1, 2, 3, 4] and summary["stop_reason"] == "early_stopping" and summary["epoch"] == 4
    assert sorted(p.name for p in (tmp_path / "m2" / "ev" / "eval").glob("*.json")) == [f"epoch-0{e}.json" for e in (1, 2, 3, 4)]
    best = json.loads((tmp_path / "m2" / "ev" / "checkpoints" / "best.json").read_text())
    assert best["step"] == 4 and best["metric"] == -3.0  # época 2 (empate com 3 e 4: a mais antiga)
    assert (tmp_path / "m2" / "ev" / "checkpoints" / "ckpt-00000004.pt").is_file()
    assert {c.step for c in list_checkpoints(tmp_path / "m2" / "ev" / "checkpoints")} == {4, 8}
    assert load_checkpoint(last).step == 8
