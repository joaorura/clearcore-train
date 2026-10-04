"""Réplica Python do runtime com FiLM (M2 Tarefa 4): `runtime_forward` × `tract-check gating`, no modelo do estágio C.

O estágio C é base DFNet3 congelada + gerador FiLM treinado: o tract roda o asset aprovado com as entradas FiLM
(`build_film_identity_asset`) e o vetor do gerador real (`ckpt-00020160` do M1, SHA-256 em `stage-f-v1.yaml`) para
perfis reais; a réplica roda o mesmo vetor pelo PyTorch. Identidade = sem `film.f32` no tract, `film=None` na réplica.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from cctrain.eval.functional import film_vector
from cctrain.eval.runtime_path import (
    HOP,
    RUNTIME_DELAY,
    Stage,
    TorchFrameModel,
    align_runtime,
    identity_film,
    parity_mixture,
    run_replica,
    runtime_forward,
    stage_fractions,
    tract_gating,
)
from cctrain.hashing import sha256_file
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmGenerator, load_film_config
from cctrain.models.pdfnet3 import PDFNet3
from cctrain.paths import clearcore_root, repo_root
from cctrain.tractcheck import APPROVED_ASSET_SHA256

ABS, REL = 1e-4, 1e-4
FRAMES = 1000


@pytest.fixture(scope="module")
def base():
    torch.set_num_threads(1)
    model, _ = load_upstream_dfnet3(device="cpu")
    return model


def _tone_noise(frames: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(frames * HOP) / 48_000.0
    voiced = sum(np.sin(2 * np.pi * 140 * h * t) / h for h in range(1, 12))
    return (0.08 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)) * voiced + 0.01 * rng.standard_normal(t.size)).astype(
        np.float32)


def test_runtime_forward_shapes_and_alignment(base):
    x = _tone_noise(60)
    enh, stages = runtime_forward(base, None, x)
    assert enh.shape == x.shape and enh.dtype == np.float32 and np.isfinite(enh).all()
    assert stages.shape == (60,) and stages.dtype == np.uint8
    raw = run_replica(x, TorchFrameModel(base, None, x))
    assert np.array_equal(enh, align_runtime(raw.audio, x.size)) and np.array_equal(stages, raw.stages)
    assert RUNTIME_DELAY == 3 * HOP


def test_runtime_forward_accepts_length_not_multiple_of_hop(base):
    x = _tone_noise(50)
    x = np.concatenate([x, x[:123]])
    enh, stages = runtime_forward(base, None, x)
    assert enh.shape == x.shape and stages.shape == (50,)
    ref, _ = runtime_forward(base, None, x[: 50 * HOP])
    assert np.array_equal(enh[: 50 * HOP], ref)


def test_runtime_forward_neutral_film_equals_none(base):
    x = _tone_noise(40)
    a, sa = runtime_forward(base, None, x)
    b, sb = runtime_forward(base, identity_film(), x)
    assert np.array_equal(a, b) and np.array_equal(sa, sb)


def test_runtime_forward_accepts_pdfnet3_and_moves_with_film(base):
    gen = FilmGenerator(load_film_config())
    torch.manual_seed(1)
    with torch.no_grad():
        gen.out.weight.normal_(std=0.02)
        gen.out.bias.normal_(std=0.02)
    emb = np.random.default_rng(0).standard_normal(192).astype(np.float32)
    emb /= np.linalg.norm(emb)
    x = _tone_noise(80)
    neutral, _ = runtime_forward(PDFNet3(base, gen), None, x)
    filmed, _ = runtime_forward(PDFNet3(base, gen), film_vector(gen, emb), x)
    assert np.max(np.abs(filmed - neutral)) > 1e-5


def test_runtime_forward_silence_is_all_silent_and_zero(base):
    enh, stages = runtime_forward(base, None, np.zeros(12 * HOP, np.float32))
    assert not enh.any() and np.all(stages == int(Stage.SILENT))


def test_generator_mask_zero_is_identity():
    gen = FilmGenerator(load_film_config())
    torch.manual_seed(2)
    with torch.no_grad():
        gen.out.weight.normal_(std=0.3)
        gen.out.bias.normal_(std=0.3)
        v = gen(torch.randn(1, 192), torch.zeros(1, 1))
    got = torch.cat([t.reshape(-1) for t in (v.gamma_enc, v.beta_enc, v.gamma_df, v.beta_df)]).numpy()
    assert np.array_equal(got, identity_film())


# --- paridade contra o tract (dados reais + checkpoint do estágio C) -----------------------------------------------



@pytest.mark.tract
@pytest.mark.clearcore
@pytest.mark.data
def test_replica_matches_tract_with_stage_c_profiles_and_identity(base, paths, tmp_path):
    from cctrain.export.film import build_film_identity_asset
    from cctrain.train.checkpoint import load_checkpoint
    from cctrain.train.embed_bank import ArrayEmbeddingBank

    cfg = yaml.safe_load((repo_root() / "configs/train/stage-f-v1.yaml").read_text())["init"]
    ckpt = paths.runs / "m1" / "stage-c-m1" / "checkpoints" / "ckpt-00020160.pt"
    if not ckpt.is_file():
        pytest.skip(f"checkpoint do estágio C ausente: {ckpt}")
    assert sha256_file(ckpt) == cfg["generator_checkpoint_sha256"]
    bank_path = paths.data / "embeddings" / "bank-m1-sbecapa-v1.npz"
    if not bank_path.is_file():
        pytest.skip(f"banco de embeddings ausente: {bank_path}")
    gen = FilmGenerator(load_film_config())
    gen.load_state_dict(load_checkpoint(ckpt).model)
    gen.eval()
    bank = ArrayEmbeddingBank.load(bank_path)
    spk = bank.speakers()
    profiles = {f"profile{k}": film_vector(gen, bank.all(spk[i])[0]) for k, i in enumerate((0, len(spk) // 2, -1))}

    approved = clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin"
    assert sha256_file(approved) == APPROVED_ASSET_SHA256
    asset = build_film_identity_asset(approved, tmp_path / "asset").path
    x = parity_mixture(paths.data)
    assert x.size == FRAMES * HOP
    identity_rt = tract_gating(asset, x, tmp_path / "rt_identity")

    report: dict = {"frames": FRAMES, "abs_tol": ABS, "rel_tol": REL, "checkpoint_sha256": sha256_file(ckpt),
                    "asset_sha256": sha256_file(asset), "fractions_identity_tract": stage_fractions(identity_rt.stages)}
    failures = []
    for name, vec in {"identity": None, **profiles}.items():
        rt = identity_rt if vec is None else tract_gating(asset, x, tmp_path / f"rt_{name}", film=vec)
        py = run_replica(x, TorchFrameModel(base, vec, x))
        excess = np.abs(py.audio - rt.audio) - (ABS + REL * np.abs(rt.audio))
        lsnr_excess = np.abs(py.lsnr - rt.lsnr) - (ABS + REL * np.abs(rt.lsnr))
        enh, st = runtime_forward(base, vec, x)
        entry = {
            "max_abs_diff_audio": float(np.max(np.abs(py.audio - rt.audio))),
            "max_abs_diff_lsnr": float(np.max(np.abs(py.lsnr - rt.lsnr))),
            "worst_excess_audio": float(excess.max()), "worst_excess_lsnr": float(lsnr_excess.max()),
            "stage_mismatches": int((py.stages != rt.stages).sum()),
            "film_vs_identity_max_abs_audio": float(np.max(np.abs(rt.audio - identity_rt.audio))),
            "film_vs_identity_max_abs_lsnr": float(np.max(np.abs(rt.lsnr - identity_rt.lsnr))),
            "stage_counts_tract": {s.name.lower(): int((rt.stages == int(s)).sum()) for s in Stage},
            "runtime_forward_equals_aligned_replica": bool(np.array_equal(enh, align_runtime(py.audio, x.size))
                                                           and np.array_equal(st, py.stages)),
        }
        if vec is not None:
            entry["film_gamma_range"] = [float(vec[[*range(256), *range(512, 768)]].min()),
                                         float(vec[[*range(256), *range(512, 768)]].max())]
            entry["film_beta_abs_max"] = float(np.abs(vec[[*range(256, 512), *range(768, 1024)]]).max())
        report[name] = entry
        if not (excess.max() <= 0 and lsnr_excess.max() <= 0 and entry["stage_mismatches"] == 0
                and entry["runtime_forward_equals_aligned_replica"]):
            failures.append(name)
    out = paths.runs / "m2"
    out.mkdir(parents=True, exist_ok=True)
    report["pass"] = not failures
    (out / "t4_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not failures, json.dumps(report, indent=2)
    # a identidade tem de exercitar os cinco estágios (como o teste do M0)
    assert all(v >= 20 for v in report["identity"]["stage_counts_tract"].values()), report["identity"]
