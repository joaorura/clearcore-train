"""Testes unitários para o script da Tarefa 8 (scripts/m2_t8.py).

Cobre todo o fluxo de medição e veredito sem tocar no áudio do dev-gate (usando mocks/fixtures).
Garante que data/dev-m2/dev-gate/opened.log NUNCA seja criado ou alterado por estes testes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import m2_t8  # noqa: E402

from cctrain.m2.report import m2_verdict  # noqa: E402

DEV_GATE_OPENED_LOG = ROOT / "data" / "dev-m2" / "dev-gate" / "opened.log"


@dataclass
class MockDevItem:
    cond: str
    spk: str
    samples: int = 48000
    sir: float | None = 0.0
    pair: str | None = "same"
    target_utt: str = "raw/librispeech-dev-clean/1/1/1.flac"

    def __getitem__(self, k: str):
        if k == "condition":
            return self.cond
        if k == "target_speaker":
            return self.spk
        if k == "sir_db":
            return self.sir
        if k == "sex_pair":
            return self.pair
        if k == "target_utt":
            return self.target_utt
        return None

    def noisy(self) -> np.ndarray:
        t = np.arange(self.samples) / 48000.0
        return (0.1 * np.sin(2 * np.pi * 300 * t) + 0.01 * np.cos(2 * np.pi * 600 * t)).astype(np.float32)

    def clean(self) -> np.ndarray | None:
        if self.cond == "C7":
            return None
        t = np.arange(self.samples) / 48000.0
        return (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


class MockDevHalf:
    def __init__(self, items: list[MockDevItem]):
        self.items = items

    def select(self, condition: str | None = None, **where) -> list[MockDevItem]:
        out = [i for i in self.items if condition is None or i.cond == condition]
        return [i for i in out if all(i[k] == v for k, v in where.items())]

    def embedding(self, _item: MockDevItem) -> np.ndarray:
        return np.ones(192, dtype=np.float32) / np.sqrt(192)


@pytest.fixture(autouse=True)
def guard_dev_gate_unopened():
    """Garante de forma estrita que nenhum teste abra ou crie o opened.log do dev-gate real."""
    existed_before = DEV_GATE_OPENED_LOG.exists()
    assert not existed_before, "opened.log do dev-gate já existe antes dos testes! Ele deve permanecer lacrado."
    yield
    existed_after = DEV_GATE_OPENED_LOG.exists()
    assert not existed_after, "VIOLAÇÃO: Um teste abriu o dev-gate real e criou opened.log!"


def test_verify_prerequisites_checks_frozen_hashes():
    hashes = m2_t8.verify_prerequisites()
    assert "gates-v2" in hashes
    assert "m2-selection-v1" in hashes
    assert "runtime-deviation-v1" in hashes
    assert "tsos-config-v1" in hashes
    assert "devsets_freeze" in hashes
    for v in hashes.values():
        assert len(v) == 64


def test_read_ab2_candidate_missing(tmp_path: Path):
    non_existent = tmp_path / "missing_candidate.json"
    with pytest.raises(FileNotFoundError, match="Arquivo de candidato da AB2 não encontrado"):
        m2_t8.read_ab2_candidate(non_existent)


def test_read_ab2_candidate_success(tmp_path: Path):
    ckpt_file = tmp_path / "dummy_ckpt.pt"
    ckpt_file.write_bytes(b"dummy_weights_12345")
    candidate_json = tmp_path / "ab2_candidate.json"
    candidate_data = {
        "candidate_run_id": "ab2-pdrop01",
        "epoch": 6,
        "step": 45000,
        "score_c3_delta_si_sdr_db": 3.82,
        "c0_delta_si_sdr_db": 6.10,
        "c0_delta_stoi": 0.045,
        "checkpoint_path": str(ckpt_file),
    }
    candidate_json.write_text(json.dumps(candidate_data))

    read_data = m2_t8.read_ab2_candidate(candidate_json)
    assert read_data["candidate_run_id"] == "ab2-pdrop01"
    assert read_data["epoch"] == 6
    assert read_data["step"] == 45000
    assert "checkpoint_sha256" in read_data
    assert len(read_data["checkpoint_sha256"]) == 64


def test_make_enhancers():
    mock_model = MagicMock()
    mock_teacher = MagicMock()

    # offline path
    with patch("m2_t8.enhance_offline") as mock_off, patch("m2_t8.align_runtime") as mock_align:
        mock_off.return_value = (np.zeros(48000, dtype=np.float32), None)
        mock_align.return_value = np.zeros(48000, dtype=np.float32)
        ec, et = m2_t8.make_enhancers(mock_model, mock_teacher, "offline")
        res_c = ec(np.zeros(48000, dtype=np.float32))
        res_t = et(np.zeros(48000, dtype=np.float32))
        assert len(res_c) == 48000
        assert len(res_t) == 48000

    # replica path
    with patch("m2_t8.runtime_forward") as mock_rf:
        mock_rf.return_value = (np.zeros(48000, dtype=np.float32), np.zeros(100, dtype=np.uint8))
        ec, et = m2_t8.make_enhancers(mock_model, mock_teacher, "replica")
        res_c = ec(np.zeros(48000, dtype=np.float32))
        res_t = et(np.zeros(48000, dtype=np.float32))
        assert len(res_c) == 48000
        assert len(res_t) == 48000

    with pytest.raises(ValueError, match="Caminho inválido"):
        m2_t8.make_enhancers(mock_model, mock_teacher, "invalid_path")


def test_measure_runtime_deviation():
    items = [MockDevItem("C3", f"spk_{i}") for i in range(5)]
    mock_dev = MockDevHalf(items)
    mock_model = MagicMock()

    with patch("m2_t8.enhance_offline") as mock_off, \
         patch("m2_t8.film_vector", return_value=np.zeros(1024, dtype=np.float32)), \
         patch("m2_t8.run_replica") as mock_rep, \
         patch("m2_t8.TorchFrameModel"):
        mock_off.return_value = (np.zeros(48000, dtype=np.float32), np.zeros(100, dtype=np.uint8))
        res_rep = MagicMock()
        res_rep.audio = np.zeros(48000, dtype=np.float32)
        res_rep.stages = np.full(100, 2, dtype=np.uint8)
        mock_rep.return_value = res_rep

        result = m2_t8.measure_runtime_deviation(mock_model, mock_dev, n_items=3)
        assert "error_to_signal_db" in result
        assert "stage_agreement" in result
        assert result["path"] in ("offline", "replica")


def test_evaluate_g1():
    items = [MockDevItem("C0", f"spk_{i}") for i in range(4)]
    mock_dev = MockDevHalf(items)

    def enh_c(x, _emb=None):
        return x

    def enh_t(x):
        return x

    mock_dnsmos = MagicMock()
    mock_dnsmos.score.return_value = {"ovrl": 3.5, "sig": 3.8, "bak": 3.2}

    res = m2_t8.evaluate_g1(enh_c, enh_t, mock_dev, dnsmos_scorer=mock_dnsmos)
    assert "delta_stoi" in res
    assert "delta_si_sdr_db" in res
    assert "delta_dnsmos_ovrl" in res
    assert "delta_dnsmos_sig" in res
    assert "delta_dnsmos_bak" in res
    assert "delta_lsd_above_8k_db" in res
    assert res["n_items"] == 4
    assert res["delta_stoi"]["ci_low"] >= -0.003
    assert res["delta_si_sdr_db"]["ci_low"] >= -0.5


def test_evaluate_g2():
    items = [MockDevItem("C3", f"spk_{i}", sir=0.0) for i in range(2)] + \
            [MockDevItem("C4", f"spk_{i}", sir=5.0) for i in range(2)]
    mock_dev = MockDevHalf(items)

    def enh_c(x, _emb=None):
        # leve ganho para testar
        return 1.1 * x

    def enh_t(x):
        return x

    mock_dnsmos_p = MagicMock()
    mock_dnsmos_p.score.return_value = {"ovrl": 3.7, "sig": 3.9, "bak": 3.5}

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = "hello world"

    transcripts = {"raw/librispeech-dev-clean/1/1/1.flac": "hello world"}

    res = m2_t8.evaluate_g2(enh_c, enh_t, mock_dev, dnsmos_p_scorer=mock_dnsmos_p,
                            asr_scorer=mock_asr, transcripts=transcripts)
    assert "delta_si_sdr_db" in res
    assert "delta_dnsmos_p_ovrl" in res
    assert "wer_ratio_to_dfnet3" in res
    assert "wer_not_above_noisy" in res
    assert res["n_items"] == 4


def test_evaluate_g3():
    c0c_items = [MockDevItem("C0c", f"spk_{i}") for i in range(3)]
    c0_items = [MockDevItem("C0", f"spk_{i}") for i in range(3)]
    mock_dev = MockDevHalf(c0c_items + c0_items)

    def enh_c(x, _emb=None):
        return x

    def enh_t(x):
        return x

    res = m2_t8.evaluate_g3(enh_c, enh_t, mock_dev)
    assert "delta_tsos_over_dfnet3_pp" in res
    assert "tsos_c0_pct" in res
    assert abs(res["delta_tsos_over_dfnet3_pp"]) < 1e-4  # idênticos
    assert res["tsos_c0_pct"] <= 1.0


def test_evaluate_g4():
    items = [MockDevItem("C7", f"spk_{i}", pair="different") for i in range(2)] + \
            [MockDevItem("C7", f"spk_{i}", pair="same") for i in range(2)]
    mock_dev = MockDevHalf(items)

    def enh_c(x, emb=None):
        # se tiver perfil, atenua forte (simula rejeição do falante errado)
        return 0.05 * x if emb is not None else x

    res = m2_t8.evaluate_g4(enh_c, mock_dev)
    assert "trr_diff_sex" in res
    assert "trr_same_sex" in res
    assert "mode_sep_db" in res
    assert "trr_curves" in res
    assert res["trr_diff_sex"] >= 0.0
    assert res["trr_same_sex"] >= 0.0
    assert res["mode_sep_db"] >= 0.0


def test_full_pipeline_verdict_and_report_generation(tmp_path: Path):
    candidate = {
        "candidate_run_id": "ab2-pdrop01",
        "epoch": 6,
        "step": 45000,
        "score_c3_delta_si_sdr_db": 3.85,
        "c0_delta_si_sdr_db": 6.20,
        "c0_delta_stoi": 0.048,
        "checkpoint_path": str(tmp_path / "ckpt.pt"),
        "checkpoint_sha256": "0" * 64,
    }

    dev_res = {"error_to_signal_db": -16.8, "stage_agreement": 0.95, "path": "replica"}
    g1 = {
        "delta_stoi": {"point": 0.02, "ci_low": 0.005, "ci_high": 0.035},
        "delta_si_sdr_db": {"point": 5.5, "ci_low": 4.8, "ci_high": 6.2},
        "delta_dnsmos_ovrl": {"point": 0.12, "ci_low": 0.08, "ci_high": 0.16},
        "delta_dnsmos_sig": {"point": 0.10, "ci_low": 0.06, "ci_high": 0.14},
        "delta_dnsmos_bak": {"point": 0.14, "ci_low": 0.10, "ci_high": 0.18},
        "delta_lsd_above_8k_db": {"point": 0.15, "ci_low": 0.05, "ci_high": 0.25},
    }
    g2 = {
        "delta_si_sdr_db": {"point": 3.85, "ci_low": 2.1, "ci_high": 5.6},
        "delta_dnsmos_p_ovrl": 0.22,
        "wer_ratio_to_dfnet3": 0.81,
        "wer_not_above_noisy": True,
        "wer_candidate": 0.16,
        "wer_dfnet3": 0.20,
        "wer_noisy": 0.25,
    }
    g3 = {
        "delta_tsos_over_dfnet3_pp": 0.12,
        "tsos_c0_pct": 0.35,
        "candidate_c0c_tsos_pct": 2.12,
        "dfnet3_c0c_tsos_pct": 2.00,
    }
    g4 = {
        "trr_diff_sex": 0.94,
        "trr_same_sex": 0.76,
        "mode_sep_db": 17.5,
        "trr_curves": {
            "different": {6: 0.99, 10: 0.97, 15: 0.94, 20: 0.88},
            "same": {6: 0.95, 10: 0.88, 15: 0.76, 20: 0.65},
        },
    }
    informatives = (
        {"spk_sim_delta_ci_lower": 0.01},
        {"dnsmos_p_ovrl": 3.65},
        {"tanh_saturation_pct": 0.005, "gating_fractions": {"erb_df": 0.85, "zero": 0.10}},
    )

    inp = m2_t8.M2Inputs(
        runtime_deviation=dev_res,
        g1=g1,
        g2=g2,
        g3=g3,
        g4=g4,
        g5=informatives[0],
        g8=informatives[1],
        c8=informatives[2],
    )

    v = m2_verdict(inp)
    assert v.go
    assert len(v.blocking_failures) == 0

    hashes = m2_t8.verify_prerequisites()
    report_file = tmp_path / "m2-report.md"
    rendered = m2_t8.render_m2_report(
        verdict=v,
        candidate=candidate,
        g1=g1,
        g2=g2,
        g3=g3,
        g4=g4,
        dev_res=dev_res,
        informatives=informatives,
        hashes=hashes,
        out_path=report_file,
    )

    assert "# Relatório Final do M2" in rendered
    assert "**GO**" in rendered
    assert "ab2-pdrop01" in rendered
    assert "G1_delta_stoi_ci_low" in rendered
    assert "G2_delta_si_sdr_point" in rendered
    assert "G3_tsos_max_over_dfnet3_pp" in rendered
    assert "G4_trr_different_sex" in rendered
    assert report_file.is_file()


def test_cmd_status_runs_without_opening_dev_gate():
    args = MagicMock()
    # Deve rodar e não lançar exceções nem criar opened.log
    m2_t8.cmd_status(args)
    assert not DEV_GATE_OPENED_LOG.exists()
