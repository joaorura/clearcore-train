#!/usr/bin/env python3
"""Métricas de troca de turno sobre os scores de presença do dev (saída do `ceiling_native.py` / `ceiling_train_heads.py`).

Pedido após o O2 (oráculo com 150 ms de atraso + 10% de erro em rajadas caiu para 4/9): o que conta é latência e precisão
nas TROCAS, não o AUC médio. Para cada score [V,T] (vista mix + int, como `ceiling_common.pos_neg`):
  auc_sw300   AUC alvo × só-interferente restrito aos quadros a até 300 ms de uma troca (início/fim do alvo ou do interf.)
  err_sw300   erro (alvo zerado + só-interf. passado) nesses quadros, no ponto de operação (gate varrido, ≤2% alvo zerado)
  lag_corr    deslocamento (quadros de 10 ms) que maximiza a correlação score × rótulo do alvo (score atrasado = lag > 0)
  t_on/t_off  mediana do tempo (ms) até a decisão abrir após o alvo voltar depois do interferente / fechar após o alvo
              parar com o interferente ativo (limite 1 s; "nunca" conta 1 s)
  err_pct     fração de quadros alvo/só-interf. com decisão errada; burst_ms = duração média das rajadas de erro
O O2 do oráculo, por construção: 150 ms de atraso, ~10% dos quadros em rajadas de 200 ms (O2': 30 ms, 5%).

Uso: ceiling_switch_metrics.py --dev-cache <npz> --scores NOME=arquivo.npy [...] [--gates curve.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ceiling_common as C
from train_pro_v4_presence import presence_state_machine

WIN = 30
CAP = 100


def near_switch(c, win=WIN):
    ta, ia = c["ta"], c["ia"]
    ev = np.zeros_like(ta)
    for m in (ta, ia):
        d = np.zeros_like(m)
        d[:, 1:] = m[:, 1:] != m[:, :-1]
        ev |= d
    out = ev.copy()
    for k in range(1, win + 1):
        out[:, k:] |= ev[:, :-k]
        out[:, :-k] |= ev[:, k:]
    return out


def auc_on(s, pos, neg):
    if not pos.any() or not neg.any():
        return float("nan")
    return C.auc(np.concatenate([s[pos], s[neg]]), np.concatenate([np.ones(pos.sum(), bool), np.zeros(neg.sum(), bool)]))


def lag_corr(c, s, max_lag=40):
    mix = c["view"] == "mix"
    v = c["valid"][mix]
    y = c["ta"][mix].astype(np.float64)
    x = s[mix].astype(np.float64)
    best = (-2.0, 0)
    for lag in range(0, max_lag + 1):
        xs, ys, vs = x[:, lag:], y[:, : y.shape[1] - lag], v[:, lag:] & v[:, : v.shape[1] - lag]
        a, b = xs[vs], ys[vs]
        r = np.corrcoef(a, b)[0, 1]
        best = max(best, (r, lag))
    return best[1], best[0]


def turn_times(c, dec):
    """dec [V,T] bool (True = passa). t_on: alvo começa (antes só-interf. nos 300 ms) -> quadros até passar.
    t_off: alvo para com interf. ativo -> quadros até zerar."""
    ta, ia = c["ta"], c["ia"]
    mix = c["view"] == "mix"
    ton, toff = [], []
    for v in np.flatnonzero(mix):
        t, i, d = ta[v], ia[v], dec[v]
        for f in range(1, len(t)):
            if t[f] and not t[f - 1] and i[max(0, f - WIN):f].any():
                g = np.flatnonzero(d[f:f + CAP])
                ton.append(g[0] if g.size else CAP)
            if not t[f] and t[f - 1] and i[f]:
                g = np.flatnonzero(~d[f:f + CAP])
                toff.append(g[0] if g.size else CAP)
    return (10 * float(np.median(ton)) if ton else float("nan"), len(ton),
            10 * float(np.median(toff)) if toff else float("nan"), len(toff))


def bursts(err):
    lens = []
    for row in err:
        r = np.concatenate([[0], row.astype(np.int8), [0]])
        d = np.diff(r)
        lens += list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    return 10 * float(np.mean(lens)) if lens else float("nan")


def analyse(c, s, gate=None):
    pos, neg = C.pos_neg(c)
    sw = near_switch(c)
    if gate == "ba":  # ponto de máxima acurácia balanceada, sem histerese (compara com o O2 binário)
        sc = np.concatenate([s[pos], s[neg]])
        y = np.concatenate([np.ones(pos.sum(), bool), np.zeros(neg.sum(), bool)])
        qs = np.quantile(sc, np.linspace(0.01, 0.99, 99))
        tau = float(max(qs, key=lambda q: C.bal_acc(sc >= q, y)))
        gate = {"tau_on": tau, "tau_off": tau, "hangover": 0}
        zero = s < tau
    elif gate is None:
        gate, zero = C.sweep_gate(s, pos, neg, c["valid"])
    else:
        zero = ~presence_state_machine(s, gate["tau_on"], gate["tau_off"], gate["hangover"])
    dec = ~zero
    err = (pos & zero) | (neg & dec)
    lag, r = lag_corr(c, s)
    t_on, n_on, t_off, n_off = turn_times(c, dec)
    lab = pos | neg
    return {"auc": round(auc_on(s, pos, neg), 4), "auc_sw300": round(auc_on(s, pos & sw, neg & sw), 4),
            "err_sw300_pct": round(100 * err[lab & sw].mean(), 1), "err_pct": round(100 * err[lab].mean(), 1),
            "tgt_zero_pct": round(100 * (pos & zero).sum() / pos.sum(), 2),
            "oi_pass_pct": round(100 * (neg & dec).sum() / neg.sum(), 1), "burst_ms": round(bursts(err & lab), 0),
            "lag_corr_ms": 10 * lag, "r_at_lag": round(r, 3), "t_on_ms": t_on, "n_on": n_on, "t_off_ms": t_off,
            "n_off": n_off, "gate": {k: gate[k] for k in ("tau_on", "tau_off", "hangover")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-cache", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(a.dev_cache)
    c = {k: z[k] for k in z.files if k not in ("emb31",)}
    res = {}
    for spec in a.scores:
        name, path = spec.split("=", 1)
        sc = np.load(path).astype(np.float32)
        res[name] = analyse(c, sc)
        print(name, json.dumps(res[name]), flush=True)
        res[name + "@BA"] = analyse(c, sc, gate="ba")
        print(name + "@BA", json.dumps(res[name + "@BA"]), flush=True)
    # oráculos sintéticos (rótulo do alvo com atraso + rajadas), para comparação direta com O2/O2'
    from cctrain.models.oracle import OracleConfig, corrupt_np

    for nm, d, e in (("oracle_O2_150ms_10pct", 15, 0.10), ("oracle_O2b_30ms_5pct", 3, 0.05), ("oracle_clean", 0, 0.0)):
        cfg = OracleConfig("noisy" if d or e else "clean", delay_frames=d, err_frac=e, burst_frames=20)
        rng = np.random.default_rng(0)
        s = np.stack([corrupt_np(t, cfg, rng) for t in c["ta"]]).astype(np.float32)
        res[nm] = analyse(c, s, gate={"tau_on": 0.5, "tau_off": 0.5, "hangover": 0})
        print(nm, json.dumps(res[nm]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
