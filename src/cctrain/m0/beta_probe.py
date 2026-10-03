"""Regra de B (spec §3.2): p99 das ativações pós-ReLU nos dois sites FiLM do DFNet3 upstream, dados do M0."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from cctrain.models.film import FILM_CONFIG_V1, beta_bound_rule


def collect_site_activations(base: nn.Module, batches: Iterable[dict], device: str) -> dict[str, np.ndarray]:
    sites = {"enc": base.enc.emb_gru.linear_in, "df_dec": base.df_dec.df_gru.linear_in}
    store: dict[str, list[np.ndarray]] = {k: [] for k in sites}
    hooks = [m.register_forward_hook(
        lambda _m, _i, out, k=k: store[k].append(out.detach().float().cpu().reshape(-1).numpy()))
        for k, m in sites.items()]
    try:
        with torch.no_grad():
            for b in batches:
                base(*(torch.as_tensor(b[k]).to(device) for k in ("spec", "feat_erb", "feat_spec")))
    finally:
        for h in hooks:
            h.remove()
    return {k: np.concatenate(v) for k, v in store.items()}


def limitation_note(max_observed: float, beta_bound: float) -> str:
    rel = ">" if max_observed > beta_bound else "<="
    return (f"ativações medidas com o ruído sintético do piloto (pilot-only, cost measurement) e fala VCTK/CV27; "
            f"máximo observado {max_observed:.1f} {rel} B={beta_bound:g}")


def current_beta_bound(path: Path = FILM_CONFIG_V1) -> float | None:
    value = yaml.safe_load(Path(path).read_text())["beta_bound"]
    return None if value is None else float(value)


def beta_rule_report(acts: dict[str, np.ndarray]) -> dict:
    pooled = np.concatenate(list(acts.values()))
    p99 = float(np.percentile(pooled, 99))
    beta_bound = beta_bound_rule(p99)
    return {
        "p99_by_site": {k: float(np.percentile(v, 99)) for k, v in acts.items()},
        "max_by_site": {k: float(v.max()) for k, v in acts.items()},
        "values_by_site": {k: int(v.size) for k, v in acts.items()},
        "p99_pooled": p99,
        "rule": "B = min(50, max(4, 2*p99_pooled))",
        "beta_bound": beta_bound,
        "limitation": limitation_note(float(pooled.max()), beta_bound),
    }


def write_beta_bound(value: float, path: Path = FILM_CONFIG_V1) -> None:
    path = Path(path)
    text = path.read_text()
    if not re.search(r"^beta_bound: null$", text, flags=re.M):
        raise ValueError(f"{path}: beta_bound já fixado; mudar exige nova versão da config")
    path.write_text(re.sub(r"^beta_bound: null$", f"beta_bound: {value}  # fixado no M0 (regra da spec §3.2)",
                           text, count=1, flags=re.M))
