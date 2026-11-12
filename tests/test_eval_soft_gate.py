"""Gating suave do harness de avaliação (`tse_eval.soft_gain`, lookahead, janela fixa, presença dura)."""

from __future__ import annotations

import numpy as np

from cctrain.eval import tse_eval as te


def test_soft_gain_limits_and_dynamics():
    floor_db = -20.0
    g = te.soft_gain(np.zeros(50), floor_db, 15.0, 200.0)
    assert np.allclose(g, 10 ** (floor_db / 20))  # p=0 -> piso
    g = te.soft_gain(np.ones(200), floor_db, 15.0, 200.0)
    assert g[-1] > 0.999 and np.all(np.diff(g) >= 0)  # p=1 -> sobe até 1
    # ataque rápido (15 ms) sobe bem mais depressa que a liberação (200 ms) desce
    p = np.concatenate([np.ones(30), np.zeros(30)])
    g = te.soft_gain(p, floor_db, 15.0, 200.0)
    up = (g[3] - g[0]) / (1 - g[0])
    down = (g[30] - g[33]) / (g[30] - 10 ** (floor_db / 20))
    assert up > 0.6 > down
    assert g.min() >= 10 ** (floor_db / 20) - 1e-6 and g.max() <= 1.0 + 1e-6


def test_floor_zero_db_is_open():
    p = np.random.default_rng(0).uniform(size=100)
    assert np.allclose(te.soft_gain(p, 0.0, 15.0, 150.0), 1.0)


def test_lookahead_max_opens_before_onset():
    p = np.zeros(40)
    p[20:] = 1.0
    g0 = te.soft_gain(p, -20.0, 15.0, 200.0, lookahead_ms=0.0)
    g4 = te.soft_gain(p, -20.0, 15.0, 200.0, lookahead_ms=40.0)
    assert np.array_equal(te.lookahead_max(p, 0), p)
    assert te.lookahead_max(p, 4)[16] == 1.0 and te.lookahead_max(p, 4)[15] == 0.0
    assert g4[20] > g0[20] and g4[19] > g0[19]  # a porta já está aberta quando a palavra começa


def test_window_gain_median_over_window():
    p = np.concatenate([np.ones(10), np.zeros(10)])
    g = te.window_gain(p, -20.0, 100.0)
    assert np.allclose(g[:10], 1.0) and np.allclose(g[10:], 0.1)
    p = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], float)  # mediana 0 numa janela de 100 ms
    assert np.allclose(te.window_gain(p, -20.0, 100.0), 0.1)


def test_presence_pass_hysteresis_and_hangover():
    p = np.array([0.0, 0.5, 0.2, 0.05, 0.0, 0.0, 0.0])
    out = te.presence_pass(p, 0.3, 0.1, 2)
    assert out.tolist() == [False, True, True, True, False, False, False]
