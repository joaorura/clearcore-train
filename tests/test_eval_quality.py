"""Métricas de qualidade do alvo (`tse_eval.quality_metrics`): saída-sombra da réplica e detecção de dropouts."""

from __future__ import annotations

import numpy as np

from cctrain.eval import tse_eval as te
from cctrain.eval.runtime_path import TorchFrameModel, run_replica
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3


def test_dropouts_counts_only_20_to_300_ms_runs():
    act = np.ones(400, bool)
    g = np.zeros(400)
    g[10:11] = -20  # 10 ms: ignorado
    g[50:55] = -20  # 50 ms: dropout
    g[100:130] = -15  # 300 ms: dropout
    g[200:240] = -30  # 400 ms: supressão longa
    act[300] = False
    g[299:302] = -20  # 1 + 1 quadros separados por um inativo: ignorados
    d = te.dropouts(g, act)
    assert [(a, b) for a, b, _ in d["events"]] == [(50, 55), (100, 130)]
    assert d["drop_frames"] == 35 and d["long_frames"] == 40
    assert d["thr"] == -10.0


def test_dropout_threshold_relative_to_median():
    act = np.ones(100, bool)
    g = np.full(100, 6.0)
    g[40:45] = -3.0  # 9 dB abaixo da mediana: não cai (limiar = max(6-10, -10) = -4)
    g[60:65] = -5.0
    d = te.dropouts(g, act)
    assert [(a, b) for a, b, _ in d["events"]] == [(60, 65)]


def test_shadow_outputs_sum_to_output():
    rng = np.random.default_rng(0)
    n = 480 * 300
    t = (rng.standard_normal(n) * 0.05 * (np.sin(np.arange(n) / 4000) > 0)).astype(np.float32)
    z = (rng.standard_normal(n) * 0.01).astype(np.float32)
    x = t + z
    base, _ = load_upstream_dfnet3()
    thr = (np.float32(-np.inf), np.float32(np.inf), np.float32(np.inf))  # ERB+DF em todo quadro
    res = run_replica(x, TorchFrameModel(base, None, x, thr), thr, shadow=[t, z])
    assert np.sum(res.audio**2) > 0
    err = res.audio - res.shadow[0] - res.shadow[1]
    assert 10 * np.log10(np.sum(err**2) / np.sum(res.audio**2)) < -100
