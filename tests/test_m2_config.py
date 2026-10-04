"""Config do estágio F e regra de seleção de checkpoint, congeladas antes da Tarefa 6 (M2 Tarefa 1)."""

from __future__ import annotations

import copy
import hashlib
import math
import shutil
from pathlib import Path

import pytest
import yaml

from cctrain.eval.gates import GatesError
from cctrain.m2.config import (
    SELECTION_V1,
    STAGE_F_V1,
    M2ConfigError,
    StageFConfig,
    load_selection,
    load_stage_f_config,
    parse_stage_f_config,
)

# SHA-256 literais (Tarefa 1): qualquer edição depois do congelamento falha aqui, mesmo que o .sha256 seja refeito.
STAGE_F_V1_SHA256 = "195ac97bc044e24138d56f1c2bdc877c2ede1d9b67ed22966efbfc0eaf75e20b"
SELECTION_V1_SHA256 = "4d843ff2d9eb8e33644078673f4c93c42e991cad62dd7ad9db3af69a3372f2d3"


def _raw() -> dict:
    return copy.deepcopy(yaml.safe_load(STAGE_F_V1.read_text()))


def test_frozen_files_have_literal_sha256_and_load():
    assert hashlib.sha256(STAGE_F_V1.read_bytes()).hexdigest() == STAGE_F_V1_SHA256
    assert hashlib.sha256(SELECTION_V1.read_bytes()).hexdigest() == SELECTION_V1_SHA256
    assert load_stage_f_config().sha256 == STAGE_F_V1_SHA256
    assert load_selection().sha256 == SELECTION_V1_SHA256


def test_stage_f_matches_the_plan():
    c = load_stage_f_config()
    assert (c.optimizer.name, c.optimizer.lr, c.optimizer.epochs, c.optimizer.warmup_steps) == ("adamw", 1e-4, 20, 1000)
    assert c.optimizer.schedule == "cosine"
    assert (c.data.batch_size, c.data.segment_s, c.seed) == (16, 3.0, 0)
    assert (c.loss.lambda_t, c.loss.lambda_os, c.loss.lambda_gate) == (500.0, 500.0, 1e-3)
    assert c.loss.os_average == "time_frequency"
    assert c.film.p_drop == 0.2 and c.mixing.s == 0.05 and c.enrollment.p_enroll_denoise == 0.5
    assert math.isclose(c.film.log_gamma_bound, math.log(10.0)) and c.film.beta_bound == 4.0
    assert c.init.generator_checkpoint == "stage-c-m1/checkpoints/ckpt-00020160.pt"
    assert c.init.base == "dfnet3-upstream-v0.5.6"
    assert c.freeze.encoder and c.freeze.batchnorm == "eval_frozen"
    assert c.mixing.band_quota.full_band_min == 0.5 and c.mixing.band_quota.khz16_max == 0.2
    assert c.mixing.snr_grid_db == (-5.0, 0.0, 5.0, 10.0, 20.0, 40.0) and c.mixing.sir_db == (-5.0, 25.0)


def test_mixture_fractions_follow_spec_4_4():
    f = load_stage_f_config().mixing.fractions()
    assert math.isclose(sum(f.values()), 1.0)
    assert math.isclose(f["no_target"], 0.05)
    assert math.isclose(f["target_noise"], 0.35 * 0.95 / 0.9)
    assert math.isclose(f["target_interferer"], 0.20 * 0.95 / 0.9)
    s10 = parse_stage_f_config({**_raw(), "mixing": {**_raw()["mixing"], "s": 0.10}}).mixing.fractions()
    assert [round(s10[k], 6) for k in ("target_noise", "target_interferer", "target_interferer_noise", "no_target")] == [
        0.35, 0.20, 0.35, 0.10]


def test_ab1_is_one_full_seed_per_arm_without_inherited_branches():
    ab1 = load_stage_f_config().ablations.ab1
    assert ab1.mode == "full_run" and ab1.inherited_branches is False
    assert ab1.seeds_per_arm == 1 and ab1.arms_p_enroll_denoise == (0.0, 0.5)


@pytest.mark.parametrize("path,value,match", [
    (("film", "log_gamma_bound"), math.log(100.0) + 0.01, "Γ"),
    (("film", "beta_bound"), 60.0, "B"),
    (("freeze", "batchnorm"), "train", "BatchNorm"),
    (("freeze", "encoder"), False, "encoder"),
    (("optimizer", "lr"), 1e-3, "lr"),
    (("optimizer", "epochs"), 40, "épocas"),
    (("data", "batch_size"), 32, "batch"),
    (("loss", "lambda_t"), 0.0, "λ_T"),
    (("film", "p_drop"), 1.0, "p_drop"),
    (("mixing", "s"), 0.5, "s"),
    (("mixing", "band_quota", "full_band_min"), 0.3, "cota"),
    (("enrollment", "p_enroll_denoise"), 1.5, "p_enroll_denoise"),
    (("ablations", "ab1", "inherited_branches"), True, "AB1"),
    (("ablations", "ab1", "seeds_per_arm"), 3, "AB1"),
])
def test_validator_refuses_out_of_spec(path, value, match):
    raw = _raw()
    node = raw
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(M2ConfigError, match=match):
        parse_stage_f_config(raw)


def test_validator_refuses_unknown_and_missing_keys():
    raw = _raw()
    raw["surprise"] = 1
    with pytest.raises(M2ConfigError, match="chaves"):
        parse_stage_f_config(raw)
    raw = _raw()
    del raw["loss"]
    with pytest.raises(M2ConfigError, match="chaves"):
        parse_stage_f_config(raw)


def test_config_hash_is_stable_and_changes_with_content():
    a = load_stage_f_config()
    assert a.config_hash() == load_stage_f_config().config_hash()
    raw = _raw()
    raw["seed"] = 1
    assert parse_stage_f_config(raw).config_hash() != a.config_hash()


def test_stage_f_is_a_stagefconfig_and_p_drop_agrees_with_film_v1():
    from cctrain.models.film import load_film_config

    c = load_stage_f_config()
    assert isinstance(c, StageFConfig)
    film = load_film_config()
    assert math.isclose(c.film.log_gamma_bound, film.log_gamma_bound) and c.film.beta_bound == film.beta_bound
    assert yaml.safe_load((STAGE_F_V1.parents[2] / "configs/pdfnet3/film-v1.yaml").read_text())["p_drop"]["stage_f"] == 0.2


def test_selection_rule_is_literal():
    s = load_selection()
    assert s.evaluation.split == "dev-select" and s.evaluation.path == "offline_gating_emulated"
    assert (s.eligibility.condition, s.eligibility.delta_si_sdr_db_min, s.eligibility.delta_stoi_min) == (
        "C0", -0.5, -0.003)
    assert s.eligibility.reading == "point_estimate"
    assert (s.score.condition, s.score.sir_db, s.score.aggregation) == ("C3", 0.0, "mean_per_speaker")
    assert s.tie.tolerance_db == 0.05 and s.tie.prefer == "oldest"
    assert (s.early_stopping.from_epoch, s.early_stopping.patience_evaluations,
            s.early_stopping.min_improvement_db) == (8, 4, 0.1)
    assert s.no_eligible_action == "no_go_t3"
    assert set(s.forbidden_inputs) == {"dev-gate", "test"}


def test_owner_decisions_are_frozen_in_the_selection_file():
    s = load_selection()
    assert s.g1_dev_reading == "ci95_bound_toward_failure"  # mínimos pelo limite inferior, máximos pelo superior
    assert s.on_stage_f_fail.gates == ("G1", "G2", "G4")
    assert s.on_stage_f_fail.action == "stop_and_diagnose"
    assert s.on_stage_f_fail.export is False and s.on_stage_f_fail.gates_v3 is False
    assert s.speaker_sex_source == "declared_metadata_only"


def test_selection_validator_refuses_a_relaxed_rule(tmp_path):
    raw = yaml.safe_load(SELECTION_V1.read_text())
    for edit in (lambda r: r["forbidden_inputs"].remove("test"),
                 lambda r: r.__setitem__("g1_dev_reading", "point_estimate"),
                 lambda r: r["on_stage_f_fail"].__setitem__("gates_v3", True),
                 lambda r: r["eligibility"].__setitem__("delta_si_sdr_db_min", -1.0),
                 lambda r: r.__setitem__("speaker_sex_source", "model_inferred")):
        bad = copy.deepcopy(raw)
        edit(bad)
        p = tmp_path / "m2-selection-v1.yaml"
        p.write_text(yaml.safe_dump(bad))
        p.with_suffix(".sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "  m2-selection-v1.yaml\n")
        with pytest.raises(M2ConfigError):
            load_selection(p)


def test_any_edit_breaks_the_freeze(tmp_path: Path):
    for src, loader in ((STAGE_F_V1, load_stage_f_config), (SELECTION_V1, load_selection)):
        dst = tmp_path / src.name
        shutil.copy(src, dst)
        shutil.copy(src.with_suffix(".sha256"), dst.with_suffix(".sha256"))
        dst.write_text(dst.read_text() + "\n# edit\n")
        with pytest.raises(GatesError):
            loader(dst)
