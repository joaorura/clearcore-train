"""Testes do Marco M3: Exportação Reproduzível dos Assets ONNX (pDFNet3 e Enrollment) e Paridade Tract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from cctrain import tractcheck
from cctrain.eval.runtime_path import (
    HOP,
    TorchFrameModel,
    parity_mixture,
    run_replica,
    tract_gating,
)
from cctrain.export.enrollment import (
    OUTPUT_NAMES,
    build_stage_f_enrollment_asset,
    build_stage_f_enrollment_model,
    scale_to_rms_dbfs,
    speechlike_16k,
)
from cctrain.export.film import (
    CONFIG_INI_SHA256,
    HIDDEN,
    SITES,
)
from cctrain.export.pack import DFNET_MEMBERS, read_members
from cctrain.export.pdfnet3 import (
    build_pdfnet3_stage_f_asset,
    build_pdfnet3_stage_f_members,
)
from cctrain.hashing import sha256_bytes, sha256_file
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.paths import Paths, clearcore_root
from scripts.m3_export import edge_signals, film_vectors

pytestmark = [pytest.mark.clearcore, pytest.mark.tract]


@pytest.fixture(scope="module")
def approved_asset() -> Path:
    p = clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin"
    assert p.is_file()
    return p


@pytest.fixture(scope="module")
def ckpt_path(paths: Paths) -> Path:
    p = paths.runs / "m2/t3/checkpoints/ckpt-00022500.pt"
    if not p.is_file():
        pytest.skip(f"Checkpoint eleito ausente: {p}")
    return p


@pytest.fixture(scope="module")
def sb_ckpt_dir(paths: Paths) -> Path:
    p = paths.data / "models/spkrec-ecapa-voxceleb"
    if not (p / "embedding_model.ckpt").is_file():
        pytest.skip(f"Checkpoint SpeechBrain ausente: {p}")
    return p


@pytest.fixture(scope="module")
def pdfnet3_asset_path(paths: Paths) -> Path:
    p = paths.runs / "m3/pdfnet3-release-asset-v1.tar.gz"
    if not p.is_file():
        pytest.skip(f"Asset pDFNet3 ausente em {p} (rode scripts/m3_export.py)")
    return p


@pytest.fixture(scope="module")
def enrollment_asset_path(paths: Paths) -> Path:
    p = paths.runs / "m3/voice-enrollment-asset-v1.tar.gz"
    if not p.is_file():
        pytest.skip(f"Asset Enrollment ausente em {p} (rode scripts/m3_export.py)")
    return p


def test_pdfnet3_members_contract_and_structure(pdfnet3_asset_path: Path):
    members = read_members(pdfnet3_asset_path)
    assert sorted(members.keys()) == sorted(DFNET_MEMBERS)
    assert sha256_bytes(members["config.ini"]) == CONFIG_INI_SHA256

    for name in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx"):
        m = onnx.load_from_string(members[name])
        onnx.checker.check_model(m)

    enc_m = onnx.load_from_string(members["enc.onnx"])
    assert [i.name for i in enc_m.graph.input][-2:] == ["gamma", "beta"]
    df_m = onnx.load_from_string(members["df_dec.onnx"])
    assert [i.name for i in df_m.graph.input][-2:] == ["gamma", "beta"]


def test_pdfnet3_reproducibility(approved_asset: Path, ckpt_path: Path, paths: Paths):
    out_dir = paths.runs / "m3/test_repro_pdfnet3"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        a1 = build_pdfnet3_stage_f_asset(approved_asset, ckpt_path, out_dir / "p1.tar.gz", out_dir / "m1")
        a2 = build_pdfnet3_stage_f_asset(approved_asset, ckpt_path, out_dir / "p2.tar.gz", out_dir / "m2")
        assert a1.sha256 == a2.sha256
        assert a1.size == a2.size
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


def test_voice_enrollment_reproducibility(sb_ckpt_dir: Path, ckpt_path: Path, paths: Paths):
    out_dir = paths.runs / "m3/test_repro_enroll"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        e1 = build_stage_f_enrollment_asset(sb_ckpt_dir, ckpt_path, out_dir / "e1.tar.gz", out_dir / "e1.onnx")
        e2 = build_stage_f_enrollment_asset(sb_ckpt_dir, ckpt_path, out_dir / "e2.tar.gz", out_dir / "e2.onnx")
        assert e1.sha256 == e2.sha256
        assert e1.size == e2.size
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


def test_g0_tract_gating_parity(pdfnet3_asset_path: Path, ckpt_path: Path, paths: Paths):
    mix = parity_mixture(paths.data)
    assert mix.size == 1000 * HOP

    py_base, _ = load_upstream_dfnet3()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_base = {k[len("base."):]: v for k, v in ckpt["model"].items() if k.startswith("base.")}
    py_base.load_state_dict(ckpt_base)
    py_base.eval()

    work_dir = paths.runs / "m3/test_g0_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        for profile in ("neutral", "random", "extreme_enc_up_df_down", "extreme_enc_down_df_up"):
            vec = film_vectors(profile)
            rt = tract_gating(pdfnet3_asset_path, mix, work_dir / f"rt_{profile}", film=vec)
            py = run_replica(mix, TorchFrameModel(py_base, vec, mix))

            audio_diff = np.abs(py.audio - rt.audio)
            excess = audio_diff - (1e-4 + 1e-4 * np.abs(rt.audio))
            mismatches = int((py.stages != rt.stages).sum())

            assert mismatches == 0, f"Stage mismatches em {profile}: {mismatches}"
            assert excess.max() <= 0, f"Excesso de áudio em {profile}: {excess.max()}"
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


def test_g11_enrollment_contract_and_tract_parity(sb_ckpt_dir: Path, ckpt_path: Path, paths: Paths):
    enroll_model = build_stage_f_enrollment_model(sb_ckpt_dir, ckpt_path)
    onnx_path = paths.runs / "m3/enrollment.onnx"
    assert onnx_path.is_file()

    work_dir = paths.runs / "m3/test_g11_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        for sec in (2, 6, 8, 10, 12):
            x = scale_to_rms_dbfs(speechlike_16k(16_000 * sec, seed=sec), -20.0).astype("<f4")
            inp = work_dir / f"in_{sec}s.f32"
            out = work_dir / f"out_{sec}s.json"
            x.tofile(inp)

            proc = tractcheck.run("enroll", onnx_path, inp, out)
            assert proc.returncode == 0
            rep = json.loads(out.read_text())
            assert rep["ok"]

            with torch.no_grad():
                ref = [t.numpy() for t in enroll_model(torch.from_numpy(x)[None])]

            for name, r in zip(OUTPUT_NAMES, ref):
                got = np.asarray(rep["outputs"][name], np.float32)
                err = np.abs(got - r).max()
                assert err <= 1e-4, f"{name} {sec}s err {err} > 1e-4"

            emb = np.asarray(rep["outputs"]["embedding"], np.float32)
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-5, f"Norma {norm} fora de 1.0"

        for name, x in edge_signals().items():
            inp = work_dir / f"edge_{name}.f32"
            out = work_dir / f"edge_{name}.json"
            x.astype("<f4").tofile(inp)

            proc = tractcheck.run("enroll", onnx_path, inp, out)
            assert proc.returncode == 0
            rep = json.loads(out.read_text())
            assert rep["ok"]

            with torch.no_grad():
                ref = [t.numpy() for t in enroll_model(torch.from_numpy(x.astype(np.float32))[None])]

            for oname, r in zip(OUTPUT_NAMES, ref):
                got = np.asarray(rep["outputs"][oname], np.float32)
                assert np.isfinite(got).all(), f"Não-finito em {name}/{oname}"
                excess = np.abs(got - r) - (1e-4 + 1e-4 * np.abs(r))
                assert excess.max() <= 0, f"Excesso em {name}/{oname}: {excess.max()}"
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
