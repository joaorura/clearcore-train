#!/usr/bin/env python3
"""Marco M3: Exportação Reproduzível dos Assets ONNX (pDFNet3 e Enrollment) e Paridade com Tract.

Uso:
    python scripts/m3_export.py [--ckpt PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cctrain import tractcheck
from cctrain.disk import require_free
from cctrain.eval.runtime_path import (
    HOP,
    TorchFrameModel,
    parity_mixture,
    run_replica,
    split_film,
    tract_gating,
)
from cctrain.export.enrollment import (
    OUTPUT_NAMES,
    build_stage_f_enrollment_asset,
    build_stage_f_enrollment_model,
    export_enrollment_onnx,
    scale_to_peak,
    scale_to_rms_dbfs,
    speechlike_16k,
)
from cctrain.export.film import CONFIG_INI_SHA256
from cctrain.export.pack import DFNET_MEMBERS, PackResult, pack_asset, read_members
from cctrain.export.pdfnet3 import (
    build_pdfnet3_stage_f_asset,
    build_pdfnet3_stage_f_members,
)
from cctrain.hashing import sha256_bytes, sha256_file
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmGenerator, load_film_config
from cctrain.paths import Paths, clearcore_root, repo_root
from cctrain.tractcheck import APPROVED_ASSET_SHA256


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def film_vectors(profile: str) -> np.ndarray:
    """Gera vetores FiLM [gamma_enc, beta_enc, gamma_df, beta_df] para testes."""
    import math

    cfg = load_film_config()
    gamma_max, beta_max = math.exp(cfg.log_gamma_bound), cfg.beta_bound
    if profile == "neutral":
        v = np.ones(1024, dtype=np.float32)
        v[256:512] = 0.0
        v[768:1024] = 0.0
        return v
    if profile == "random":
        torch.manual_seed(5)
        gen = FilmGenerator(cfg)
        with torch.no_grad():
            gen.out.weight.normal_(std=0.02)
            gen.out.bias.normal_(std=0.02)
            emb = torch.randn(1, cfg.embedding_dim)
            emb = emb / emb.norm()
            out = gen(emb, torch.ones(1, 1))
        return np.concatenate([t.reshape(-1).numpy() for t in (out.gamma_enc, out.beta_enc, out.gamma_df, out.beta_df)])
    up, down = (gamma_max, beta_max), (1.0 / gamma_max, -beta_max)
    (ge, be), (gd, bd) = (up, down) if profile == "extreme_enc_up_df_down" else (down, up)
    return np.concatenate([np.full(256, val, np.float32) for val in (ge, be, gd, bd)]).astype(np.float32)


def edge_signals(n: int = 96_000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    impulse = np.zeros(n, np.float32)
    impulse[n // 2] = 0.99
    return {
        "silence": np.zeros(n, np.float32),
        "dc_0.1": np.full(n, 0.1, np.float32),
        "full_scale_square": np.where((np.arange(n) // 50) % 2 == 0, 1.0, -1.0).astype(np.float32),
        "white_noise_peak0.99": scale_to_peak(rng.standard_normal(n).astype(np.float32), 0.99),
        "noise_-90dBFS": scale_to_rms_dbfs(rng.standard_normal(n).astype(np.float32), -90.0),
        "single_impulse": impulse,
    }


def run_m3_export(
    ckpt_path: Path,
    approved_asset: Path,
    sb_ckpt_dir: Path,
    out_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    require_free(out_dir, 500 * 1024 * 1024)

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "approved_asset": str(approved_asset),
        "approved_asset_sha256": sha256_file(approved_asset),
    }

    # -------------------------------------------------------------------------------------------------
    # M3-T1: Exportação dos membros e empacotamento do pDFNet3 (Estágio F)
    # -------------------------------------------------------------------------------------------------
    log("Iniciando M3-T1: exportação do pDFNet3 com pesos retreinados do Estágio F...")
    members_dir = out_dir / "members"
    pdfnet3_asset_path = out_dir / "pdfnet3-release-asset-v1.tar.gz"

    pdfnet3_res = build_pdfnet3_stage_f_asset(
        approved_asset=approved_asset,
        ckpt_path=ckpt_path,
        out_path=pdfnet3_asset_path,
        out_members_dir=members_dir,
    )
    log(f"pDFNet3 empacotado: {pdfnet3_res.path} (SHA-256: {pdfnet3_res.sha256}, {pdfnet3_res.size} bytes)")

    pdfnet3_members_info = {}
    for name in DFNET_MEMBERS:
        p = members_dir / name
        pdfnet3_members_info[name] = {
            "size": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        log(f"  Membro {name}: {pdfnet3_members_info[name]['size']} bytes, sha256: {pdfnet3_members_info[name]['sha256']}")

    report["pdfnet3_asset"] = {
        "path": str(pdfnet3_asset_path),
        "sha256": pdfnet3_res.sha256,
        "size": pdfnet3_res.size,
        "members": pdfnet3_members_info,
    }

    # -------------------------------------------------------------------------------------------------
    # M3-T2: Validação de Reprodutibilidade Determinística (Segundo Export pDFNet3)
    # -------------------------------------------------------------------------------------------------
    log("Iniciando M3-T2: teste de reprodutibilidade determinística do pDFNet3 (segundo export)...")
    repro_dir = out_dir / "repro_pdfnet3"
    repro_asset_path = repro_dir / "pdfnet3-repro.tar.gz"
    repro_members_dir = repro_dir / "members"

    repro_res = build_pdfnet3_stage_f_asset(
        approved_asset=approved_asset,
        ckpt_path=ckpt_path,
        out_path=repro_asset_path,
        out_members_dir=repro_members_dir,
    )

    pdfnet3_repro_identical = (pdfnet3_res.sha256 == repro_res.sha256)
    log(f"Segundo export pDFNet3 SHA-256: {repro_res.sha256}")
    log(f"Reprodutibilidade pDFNet3: {'PASS (100% idêntico bit-a-bit)' if pdfnet3_repro_identical else 'FAIL'}")
    assert pdfnet3_repro_identical, f"Segundo export diferiu: {pdfnet3_res.sha256} != {repro_res.sha256}"

    report["pdfnet3_reproducibility"] = {
        "first_export_sha256": pdfnet3_res.sha256,
        "second_export_sha256": repro_res.sha256,
        "identical": pdfnet3_repro_identical,
    }
    shutil.rmtree(repro_dir, ignore_errors=True)

    # -------------------------------------------------------------------------------------------------
    # M3-T3: Exportação do Modelo de Enrollment (SpeechBrain ECAPA + FiLM Generator)
    # -------------------------------------------------------------------------------------------------
    log("Iniciando M3-T3: exportação do modelo de voice enrollment...")
    enroll_onnx_path = out_dir / "enrollment.onnx"
    enroll_asset_path = out_dir / "voice-enrollment-asset-v1.tar.gz"

    enroll_res = build_stage_f_enrollment_asset(
        sb_ckpt_dir=sb_ckpt_dir,
        ckpt_path=ckpt_path,
        out_tar_gz=enroll_asset_path,
        out_onnx_path=enroll_onnx_path,
    )
    log(f"Enrollment empacotado: {enroll_res.path} (SHA-256: {enroll_res.sha256}, {enroll_res.size} bytes)")

    # Reprodutibilidade de enrollment
    log("Teste de reprodutibilidade determinística do Enrollment (segundo export)...")
    repro_enroll_dir = out_dir / "repro_enroll"
    repro_enroll_dir.mkdir(parents=True, exist_ok=True)
    repro_enroll_asset_path = repro_enroll_dir / "voice-enrollment-repro.tar.gz"
    repro_enroll_onnx_path = repro_enroll_dir / "enrollment.onnx"

    repro_enroll_res = build_stage_f_enrollment_asset(
        sb_ckpt_dir=sb_ckpt_dir,
        ckpt_path=ckpt_path,
        out_tar_gz=repro_enroll_asset_path,
        out_onnx_path=repro_enroll_onnx_path,
    )

    enroll_repro_identical = (enroll_res.sha256 == repro_enroll_res.sha256)
    log(f"Segundo export Enrollment SHA-256: {repro_enroll_res.sha256}")
    log(f"Reprodutibilidade Enrollment: {'PASS (100% idêntico bit-a-bit)' if enroll_repro_identical else 'FAIL'}")
    assert enroll_repro_identical, f"Segundo export diferiu: {enroll_res.sha256} != {repro_enroll_res.sha256}"

    report["enrollment_asset"] = {
        "path": str(enroll_asset_path),
        "sha256": enroll_res.sha256,
        "size": enroll_res.size,
        "onnx_size": enroll_onnx_path.stat().st_size,
        "onnx_sha256": sha256_file(enroll_onnx_path),
    }
    report["enrollment_reproducibility"] = {
        "first_export_sha256": enroll_res.sha256,
        "second_export_sha256": repro_enroll_res.sha256,
        "identical": enroll_repro_identical,
    }
    shutil.rmtree(repro_enroll_dir, ignore_errors=True)

    # -------------------------------------------------------------------------------------------------
    # M3-T4 / G0: Validação de Contrato e Paridade Tract do pDFNet3
    # -------------------------------------------------------------------------------------------------
    log("Iniciando M3-T4 / Gate G0: validação de contrato e paridade Tract do pDFNet3...")
    readback = read_members(pdfnet3_asset_path)
    assert sorted(readback.keys()) == sorted(DFNET_MEMBERS), f"Membros lidos incorretos: {sorted(readback.keys())}"
    config_sha = sha256_bytes(readback["config.ini"])
    assert config_sha == CONFIG_INI_SHA256, f"config.ini inválido: {config_sha} != {CONFIG_INI_SHA256}"
    log(f"  Contrato dos 4 membros e hash do config.ini: PASS ({config_sha})")

    # Carga de teste Tract sem erros
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    dummy_audio = np.zeros(48000, dtype="<f4")
    dummy_in = work_dir / "dummy_48k.f32"
    dummy_audio.tofile(dummy_in)
    tract_proc = tractcheck.run("gating", pdfnet3_asset_path, dummy_in, work_dir / "tract_smoke")
    assert tract_proc.returncode == 0, f"Falha na carga do pDFNet3 no Tract: {tract_proc.stderr}"
    log("  Carga e execução básica do pDFNet3 no Tract 0.19.16: PASS")

    # Paridade Numérica P2 (Tract vs PyTorch replica) em 1.000 quadros de mistura real
    log("Executando paridade numérica P2 (Tract vs Réplica PyTorch) em 1.000 quadros...")
    mix = parity_mixture(data_dir)
    assert mix.size == 1000 * HOP

    py_base, _ = load_upstream_dfnet3()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_base = {k[len("base."):]: v for k, v in ckpt["model"].items() if k.startswith("base.")}
    py_base.load_state_dict(ckpt_base)
    py_base.eval()

    p2_results = {}
    for profile in ("neutral", "random", "extreme_enc_up_df_down", "extreme_enc_down_df_up"):
        vec = film_vectors(profile)
        film_file = work_dir / f"film_{profile}.f32"
        vec.tofile(film_file)

        rt = tract_gating(pdfnet3_asset_path, mix, work_dir / f"p2_rt_{profile}", film=vec)
        py = run_replica(mix, TorchFrameModel(py_base, vec, mix))

        audio_diff = np.abs(py.audio - rt.audio)
        excess = audio_diff - (1e-4 + 1e-4 * np.abs(rt.audio))
        mismatches = int((py.stages != rt.stages).sum())

        p2_pass = (excess.max() <= 0) and (mismatches == 0)
        p2_results[profile] = {
            "max_abs_audio": float(audio_diff.max()),
            "worst_excess_audio": float(excess.max()),
            "stage_mismatches": mismatches,
            "pass": p2_pass,
        }
        log(f"  P2 [{profile:22s}]: max_abs={audio_diff.max():.2e}, excess={excess.max():.2e}, mismatches={mismatches} -> {'PASS' if p2_pass else 'FAIL'}")
        assert p2_pass, f"Falha em P2 para profile {profile}: {p2_results[profile]}"

    report["gate_g0"] = {
        "status": "PASS",
        "config_ini_sha256": config_sha,
        "config_ini_verified": (config_sha == CONFIG_INI_SHA256),
        "tract_smoke_pass": (tract_proc.returncode == 0),
        "p2_profiles": p2_results,
    }

    # -------------------------------------------------------------------------------------------------
    # M3-T4 / G11: Validação de Contrato do Enrollment (SpeechBrain + FiLM)
    # -------------------------------------------------------------------------------------------------
    log("Iniciando Gate G11: validação de contrato e paridade do modelo de voice enrollment...")
    enroll_model = build_stage_f_enrollment_model(sb_ckpt_dir, ckpt_path)

    g11_durations = {}
    for sec in (2, 6, 8, 10, 12):
        x = scale_to_rms_dbfs(speechlike_16k(16_000 * sec, seed=sec), -20.0).astype("<f4")
        inp_file = work_dir / f"enroll_{sec}s.f32"
        out_file = work_dir / f"enroll_{sec}s.json"
        x.tofile(inp_file)

        proc = tractcheck.run("enroll", enroll_onnx_path, inp_file, out_file)
        assert proc.returncode == 0, f"Falha no tractcheck enroll para {sec}s: {proc.stderr}"
        rep = json.loads(out_file.read_text())
        assert rep["ok"], f"Relatório do Tract não ok para {sec}s: {rep}"

        with torch.no_grad():
            ref = [t.numpy() for t in enroll_model(torch.from_numpy(x)[None])]

        worst_err = 0.0
        for name, r in zip(OUTPUT_NAMES, ref):
            got = np.asarray(rep["outputs"][name], np.float32)
            worst_err = max(worst_err, float(np.abs(got - r).max()))

        emb = np.asarray(rep["outputs"]["embedding"], np.float32)
        norm = float(np.linalg.norm(emb))
        ge = np.asarray(rep["outputs"]["gamma_enc"], np.float32)
        be = np.asarray(rep["outputs"]["beta_enc"], np.float32)
        gd = np.asarray(rep["outputs"]["gamma_df"], np.float32)
        bd = np.asarray(rep["outputs"]["beta_df"], np.float32)

        dur_pass = (
            worst_err <= 1e-4
            and abs(norm - 1.0) <= 1e-5
            and float(ge.min()) > 0.0
            and float(gd.min()) > 0.0
            and float(np.abs(be).max()) <= 4.0
            and float(np.abs(bd).max()) <= 4.0
        )
        g11_durations[f"{sec}s"] = {
            "max_abs_err": worst_err,
            "embedding_l2_norm": norm,
            "gamma_enc_range": [float(ge.min()), float(ge.max())],
            "beta_enc_range": [float(be.min()), float(be.max())],
            "gamma_df_range": [float(gd.min()), float(gd.max())],
            "beta_df_range": [float(bd.min()), float(bd.max())],
            "pass": dur_pass,
        }
        log(f"  G11 [{sec:2d}s]: max_abs={worst_err:.2e}, ||emb||_2={norm:.6f}, γ_enc=[{ge.min():.2f}, {ge.max():.2f}], β_enc=[{be.min():.2f}, {be.max():.2f}] -> {'PASS' if dur_pass else 'FAIL'}")
        assert dur_pass, f"Falha no teste de duração {sec}s: {g11_durations[f'{sec}s']}"

    # Casos de borda do G11
    log("Executando casos de borda do G11 (silêncio, DC, onda quadrada, ruído branco, -90 dBFS, impulso)...")
    g11_edges = {}
    for name, x in edge_signals().items():
        inp_file = work_dir / f"edge_{name}.f32"
        out_file = work_dir / f"edge_{name}.json"
        x.astype("<f4").tofile(inp_file)

        proc = tractcheck.run("enroll", enroll_onnx_path, inp_file, out_file)
        assert proc.returncode == 0, f"Falha no tractcheck enroll para borda {name}: {proc.stderr}"
        rep = json.loads(out_file.read_text())
        assert rep["ok"], f"Relatório do Tract não ok para borda {name}: {rep}"

        with torch.no_grad():
            ref = [t.numpy() for t in enroll_model(torch.from_numpy(x.astype(np.float32))[None])]

        worst_err = 0.0
        worst_excess = -1.0
        finite = True
        for oname, r in zip(OUTPUT_NAMES, ref):
            got = np.asarray(rep["outputs"][oname], np.float32)
            if not np.isfinite(got).all():
                finite = False
            err = np.abs(got - r)
            worst_err = max(worst_err, float(err.max()))
            excess_rel = float((err - (1e-4 + 1e-4 * np.abs(r))).max())
            worst_excess = max(worst_excess, excess_rel)

        edge_pass = finite and (worst_excess <= 0.0)
        g11_edges[name] = {
            "finite": finite,
            "max_abs_err": worst_err,
            "worst_excess_rel": worst_excess,
            "pass": edge_pass,
        }
        log(f"  G11 borda [{name:20s}]: finite={finite}, max_abs={worst_err:.2e}, excess_rel={worst_excess:.2e} -> {'PASS' if edge_pass else 'FAIL'}")
        assert edge_pass, f"Falha no caso de borda {name}: {g11_edges[name]}"

    report["gate_g11"] = {
        "status": "PASS",
        "durations": g11_durations,
        "edges": g11_edges,
    }

    # Salva o relatório JSON final
    report_file = out_dir / "m3_report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    log(f"Relatório M3 salvo com sucesso em: {report_file}")

    return report


def main() -> None:
    paths = Paths.from_env()
    parser = argparse.ArgumentParser(description="Exportação Reproduzível dos Assets do Marco M3")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=paths.runs / "m2/t3/checkpoints/ckpt-00022500.pt",
        help="Caminho para o checkpoint eleito do Estágio F",
    )
    parser.add_argument(
        "--approved",
        type=Path,
        default=clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin",
        help="Caminho para o asset aprovado base",
    )
    parser.add_argument(
        "--sb-ckpt-dir",
        type=Path,
        default=paths.data / "models/spkrec-ecapa-voxceleb",
        help="Diretório do checkpoint SpeechBrain ECAPA",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.runs / "m3",
        help="Diretório de saída para os assets do M3",
    )
    args = parser.parse_args()

    report = run_m3_export(
        ckpt_path=args.ckpt,
        approved_asset=args.approved,
        sb_ckpt_dir=args.sb_ckpt_dir,
        out_dir=args.out_dir,
        data_dir=paths.data,
    )

    print("\n" + "=" * 80)
    print("                     MARCO M3 — RELATÓRIO DE EXPORTAÇÃO")
    print("=" * 80)
    print(f"Asset pDFNet3: {report['pdfnet3_asset']['path']}")
    print(f"  SHA-256: {report['pdfnet3_asset']['sha256']}")
    print(f"  Tamanho: {report['pdfnet3_asset']['size']} bytes")
    print(f"  Reprodutibilidade 100% bit-a-bit: {report['pdfnet3_reproducibility']['identical']}")
    print("-" * 80)
    print(f"Asset Voice Enrollment: {report['enrollment_asset']['path']}")
    print(f"  SHA-256: {report['enrollment_asset']['sha256']}")
    print(f"  Tamanho: {report['enrollment_asset']['size']} bytes")
    print(f"  Reprodutibilidade 100% bit-a-bit: {report['enrollment_reproducibility']['identical']}")
    print("-" * 80)
    print(f"Status Gate G0 (Contrato & Paridade Tract): {report['gate_g0']['status']}")
    print(f"Status Gate G11 (Contrato & Bordas Enrollment): {report['gate_g11']['status']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
