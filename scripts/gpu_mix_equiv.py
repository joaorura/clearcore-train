#!/usr/bin/env python
"""Equivalência fim a fim CPU (`StageV4Dataset._build`) vs GPU (receita + `GpuMixer`) com as mesmas receitas/seeds.

Uso: PYTHONPATH=src python scripts/gpu_mix_equiv.py [--n 96] [--batch 12] [--device cuda] [--json out.json]
Parâmetros do v4.2 (os mesmos padrões do train_pro_v42.py). Imprime: max|diff| de áudio (mistura, alvo, interferente
molhado e de referência, ruído) absoluto e relativo ao pico, max|diff| das features/espectros, % de quadros divergentes
por rótulo, igualdade dos metadados e estatísticas do lote (nível ativo da referência e SNR efetivo).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cctrain.data import mix_v4 as M  # noqa: E402
from cctrain.data.gpu.mixer import GpuMixer  # noqa: E402
from cctrain.data.gpu.recipe import StageV4RecipeDataset, collate_recipes  # noqa: E402

AUDIO = ("noisy", "tgt", "int_ref", "interf_wet", "noise")
FEATS = ("spec", "feat_erb", "feat_spec", "spec_clean", "spec_interf")
LABELS = ("target_active", "interf_active", "interf_active_wet45", "digital_silence")
META = ("n_interf", "has_target", "pattern", "reverb", "mask", "level_db", "sir_min_db", "noise_val", "noise_silent",
        "max_freq", "phase", "embedding")


def cpu_reference(ds: StageV4RecipeDataset, i: int) -> tuple[dict, dict]:
    """Amostra CPU + sinais internos do `_build` (capturados no `extract_features`, já no fim da mistura)."""
    cap: dict = {}
    orig = M.extract_features

    def spy(x):
        cap.clear()
        cap.update(sys._getframe(1).f_locals)
        return orig(x)

    M.extract_features = spy
    try:
        smp = ds.cpu_sample(i)
    finally:
        M.extract_features = orig
    g, lim, n = cap["g"], cap["lim"], len(cap["noisy"])
    fl = cap.get("fl") if cap["floor_db"] is not None else None
    noise = (cap["noise"] * g + (fl * 10 ** (cap["floor_db"] / 20.0) if fl is not None else 0.0)) * lim
    early, wet, n_int = cap["early"], cap["wet"], cap["n_int"]
    sig = {"noisy": cap["noisy"], "tgt": early[0], "int_ref": np.sum(early[1:], axis=0) if n_int else np.zeros(n),
           "interf_wet": (np.sum(wet[1:], axis=0) * g * lim) if n_int else np.zeros(n), "noise": noise,
           "ref_db": cap["ref_db"], "g": g, "lim": lim}
    return smp, sig


def _stats(ref_db, g, lim, noise, frames):
    lvl = ref_db + 20 * np.log10(g * lim)
    p = float(np.mean(np.asarray(noise, np.float64) ** 2))
    return lvl, (lvl - 10 * np.log10(p) if p > 0 else np.nan)


def run(ds: StageV4RecipeDataset, mixer: GpuMixer, idx: list[int], batch: int) -> dict:
    worst_a = {k: 0.0 for k in AUDIO}
    worst_r = {k: 0.0 for k in AUDIO}
    worst_f = {k: 0.0 for k in FEATS}
    worst_fr = {k: 0.0 for k in FEATS}
    lab_diff = {k: 0 for k in LABELS}
    lab_tot = 0
    meta_bad: dict[str, int] = {}
    lvl_d, snr_d, invalid, refill, n_rev, n_dry_int = [], [], 0, 0, 0, 0
    lv_cpu, snr_cpu = [], []
    for s in range(0, len(idx), batch):
        ids = idx[s : s + batch]
        rb = collate_recipes([ds[i] for i in ids])
        out = mixer(rb, debug=True)
        invalid += int((~out["valid"]).sum())
        dbg = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in out["debug"].items()}
        for b, i in enumerate(ids):
            smp, sig = cpu_reference(ds, i)
            if not bool(out["valid"][b]):
                continue
            n_rev += bool(smp["reverb"])
            n_dry_int += (not bool(smp["reverb"])) and int(smp["n_interf"]) > 0
            for k in AUDIO:
                d = float(np.max(np.abs(dbg[k][b] - sig[k])))
                worst_a[k] = max(worst_a[k], d)
                pk = float(np.max(np.abs(sig[k])))
                if pk > 0:
                    worst_r[k] = max(worst_r[k], d / pk)
            for k in FEATS:
                a = out[k][b].cpu().numpy().astype(np.float64)
                c = smp[k].astype(np.float64)
                d = float(np.max(np.abs(a - c)))
                worst_f[k] = max(worst_f[k], d)
                worst_fr[k] = max(worst_fr[k], d / max(float(np.max(np.abs(c))), 1e-30))
            for k in LABELS:
                lab_diff[k] += int(np.sum(out[k][b].cpu().numpy() != smp[k]))
            lab_tot += len(smp["target_active"])
            for k in META:
                a = out[k][b].cpu().numpy()
                c = np.asarray(smp[k])
                ok = np.allclose(a, c, rtol=1e-6, atol=1e-6, equal_nan=True) if a.dtype.kind == "f" else \
                    np.array_equal(a, c)
                if not ok:
                    meta_bad[k] = meta_bad.get(k, 0) + 1
            for k in ("target_key", "target_path"):
                if out[k][b] != smp[k]:
                    meta_bad[k] = meta_bad.get(k, 0) + 1
            frames = len(smp["target_active"])
            lc, sc = _stats(sig["ref_db"], sig["g"], sig["lim"], sig["noise"], frames)
            lg, sg = _stats(float(dbg["ref_db"][b]), float(dbg["g"][b]), float(dbg["lim"][b]), dbg["noise"][b], frames)
            lv_cpu.append(lc)
            snr_cpu.append(sc)
            lvl_d.append(abs(lg - lc))
            if np.isfinite(sc):
                snr_d.append(abs(sg - sc))
        refill += mixer.fill_invalid(out, ds)
    return {
        "n": len(idx), "reverb": n_rev, "dry_with_interf": n_dry_int, "gpu_invalid": invalid, "refilled": refill,
        "audio_maxabs": worst_a, "audio_rel_peak": worst_r, "feat_maxabs": worst_f, "feat_rel_peak": worst_fr,
        "label_frames_diff_pct": {k: 100.0 * v / max(lab_tot, 1) for k, v in lab_diff.items()},
        "label_frames": lab_tot, "meta_mismatch": meta_bad,
        "level_db_maxdiff": float(np.max(lvl_d)), "snr_eff_db_maxdiff": float(np.max(snr_d)) if snr_d else 0.0,
        "level_db_cpu_mean": float(np.mean(lv_cpu)),
        "snr_eff_db_cpu_median": float(np.nanmedian(snr_cpu)),
    }


def make(device: str, **kw):
    import train_pro_v42 as T

    from cctrain.paths import Paths

    a = T.parse_args([f"--{k.replace('_', '-')}={v}" for k, v in kw.items()])
    paths = Paths.from_env()
    vp = T.v42_params(a)
    base, pl = T.make_dataset(paths, vp)
    ds = StageV4RecipeDataset(base.pools, base.bank, base.data_root, vp)
    return ds, GpuMixer.from_dataset(ds, device), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ds, mixer, _ = make(a.device, batch_size=a.batch)
    res = run(ds, mixer, list(range(a.start, a.start + a.n)), a.batch)
    txt = json.dumps(res, indent=1, default=float)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt)


if __name__ == "__main__":
    main()
