from dataclasses import replace

import numpy as np
import pytest

from cctrain.m2.config import load_selection
from cctrain.m2.select import (
    EpochEval,
    SelectItem,
    best_epoch,
    evaluate_select,
    is_eligible,
    make_epoch_eval,
    read_history,
    should_stop,
    write_eval,
)

RULE = load_selection()


def _m(si_c0=0.0, stoi_c0=0.0, c3=1.0):
    return {"c0": {"delta_si_sdr_db": si_c0, "delta_stoi": stoi_c0}, "c3": {"delta_si_sdr_db": c3}}


def _ev(epoch, score, eligible=True):
    return EpochEval(epoch, epoch * 10, eligible, score if eligible else None, {})


def test_eligibility_margins_are_point_estimates_on_c0():
    assert is_eligible(_m(-0.5, -0.003), RULE)
    assert not is_eligible(_m(-0.51, 0.0), RULE)
    assert not is_eligible(_m(0.0, -0.0031), RULE)
    assert make_epoch_eval(3, 30, _m(-1.0, 0.0, 9.0), RULE).score is None
    assert make_epoch_eval(3, 30, _m(0.0, 0.0, 2.5), RULE).score == 2.5


def test_best_is_the_highest_eligible_score_and_ties_go_to_the_oldest():
    h = [_ev(1, 1.0), _ev(2, 2.00), _ev(3, 2.04), _ev(4, 9.0, eligible=False), _ev(5, 1.9)]
    assert best_epoch(h, RULE).epoch == 2  # 2,04 e 2,00 empatam (< 0,05 dB): o mais antigo
    assert best_epoch([_ev(1, 1.0), _ev(2, 2.06)], RULE).epoch == 2
    assert best_epoch([_ev(1, 0, eligible=False)], RULE) is None


def test_early_stopping_counts_from_epoch_8_with_patience_4_and_min_improvement():
    gains = [_ev(e, 0.5 * e) for e in range(1, 8)]  # sobe 0,5 dB por época até a 7
    assert not should_stop(gains + [_ev(8, 3.55)], RULE)  # sem melhora na 8 (conta 1)
    flat = gains + [_ev(e, 3.5 + 0.005 * e) for e in range(8, 12)]  # 8, 9, 10, 11: 4 sem melhora >= 0,1
    assert should_stop(flat, RULE)
    assert not should_stop(flat[:-1], RULE)
    # melhora >= 0,1 dB zera a contagem
    assert not should_stop(flat[:-2] + [_ev(10, 4.0), _ev(11, 4.0)], RULE)
    # antes da época 8 não conta
    assert not should_stop([_ev(1, 1.0)] + [_ev(e, 1.0) for e in range(2, 8)], RULE)
    short = replace(RULE, early_stopping=replace(RULE.early_stopping, from_epoch=1, patience_evaluations=2))
    assert should_stop([_ev(1, 1.0), _ev(2, 1.0), _ev(3, 1.0)], short)


def test_evaluation_files_are_written_atomically_and_read_back_in_order(tmp_path):
    for e in (2, 10, 1):
        write_eval(tmp_path, make_epoch_eval(e, e * 7, _m(c3=float(e)), RULE))
    assert [h.epoch for h in read_history(tmp_path)] == [1, 2, 10]
    assert (tmp_path / "eval" / "epoch-02.json").is_file() and not list(tmp_path.glob("eval/*.tmp"))


@pytest.fixture(scope="module")
def models():
    from cctrain.train.stage_f import StageFHParams, build_models

    return build_models(StageFHParams(device="cpu"), "cpu")


def _items(n=2):
    rng = np.random.default_rng(0)
    t = np.arange(24_000) / 48_000
    out = []
    for i in range(n):
        clean = (0.1 * np.sin(2 * np.pi * (200 + 40 * i) * t)).astype(np.float32)
        noisy = clean + 0.02 * rng.standard_normal(t.size).astype(np.float32)
        out.append(SelectItem("C0", noisy, clean, f"s{i}"))
        out.append(SelectItem("C3", noisy, clean, f"s{i}", rng.standard_normal(192).astype(np.float32)))
    return out


def test_evaluate_select_is_zero_for_the_untrained_student_and_needs_both_conditions(models):
    model, teacher, _ = models
    r = evaluate_select(model, _items(), teacher=teacher)
    assert r["c0"]["delta_si_sdr_db"] == pytest.approx(0.0, abs=1e-4) and r["c3"]["delta_si_sdr_db"] == pytest.approx(
        0.0, abs=1e-4)
    assert r["c0"]["delta_stoi"] == pytest.approx(0.0, abs=1e-6) and r["c0"]["n_speakers"] == 2
    with pytest.raises(ValueError, match="C3"):
        evaluate_select(model, [i for i in _items() if i.condition == "C0"], teacher=teacher)
