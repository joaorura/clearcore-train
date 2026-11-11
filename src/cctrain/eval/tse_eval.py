"""Harness de avaliação do pDFNet3 Pro (plano v4 §8): roda um checkpoint quadro a quadro com o gating do runtime, mede
as métricas por item, agrega por célula e confere os 11 critérios de aceite.

Gating (`GATINGS`): `current` = estágios de hoje (ZERO < -10 < ERB+DF <= 20 < ERB-only <= 30 < passthrough); `new` =
plano §3.3: com perfil (`speaker_present=1`) só ZERO se lsnr < -10, senão ERB+DF (sem ERB-only nem passthrough). Sem
perfil o gating é sempre o atual (comportamento DFNet3).
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cctrain.eval import runtime_path as rp
from cctrain.eval import tse_matrix as tm
from cctrain.eval.functional import si_sdr

CURRENT = (np.float32(-10.0), np.float32(30.0), np.float32(20.0))
NEW_PROFILE = (np.float32(-10.0), np.float32("inf"), np.float32("inf"))
GATINGS = {"current": CURRENT, "new": NEW_PROFILE}

ARCHS = {  # num_tokens, token_dim, proj_dim (contrato v2 = K8/D64; v3 e v4 = K16/D128)
    "v2": dict(query_dim=512, embedding_dim=192, proj_dim=64, num_heads=4, num_tokens=8, token_dim=64),
    "v3": dict(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4, num_tokens=16, token_dim=128),
    "v4": dict(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4, num_tokens=16, token_dim=128),
}


# --- modelo -------------------------------------------------------------------------------------------------------


def build_model(ckpt: Path, arch: str, device: str):
    from cctrain.m2.config import load_stage_f_config
    from cctrain.models.cross_attention import SubbandCrossAttentionConfig
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.models.film import FilmConfig, FilmGenerator
    from cctrain.models.pdfnet3_pro import PDFNet3Pro
    from cctrain.train.checkpoint import load_checkpoint
    from cctrain.train.stage_f import StageFHParams

    base, _ = load_upstream_dfnet3(device=device)
    hp = StageFHParams.from_config(load_stage_f_config())
    gen = FilmGenerator(FilmConfig(log_gamma_bound=hp.log_gamma_bound, beta_bound=hp.beta_bound)).to(device)
    m = PDFNet3Pro(base, gen, cross_attention_config=SubbandCrossAttentionConfig(**ARCHS[arch]),
                   enable_cross_attention=True).to(device)
    m.load_state_dict(load_checkpoint(Path(ckpt)).model, strict=True)
    return m.eval()


class ProfileFrameModel(rp.TorchFrameModel):
    """`TorchFrameModel` com o forward completo do PDFNet3Pro (FiLM + freq_mixer + cross-attention), mask=1."""

    def __init__(self, model, embedding, audio, thresholds):
        _, self.feat_erb, self.feat_spec = rp.replica_features(audio)
        self.lsnr = np.zeros(0, np.float32)
        self.gains, self.coefs, self.t = {}, {}, 0
        if not len(self.feat_erb):
            return
        dev = next(model.parameters()).device
        base = model.base
        emb_spk = torch.from_numpy(np.asarray(embedding, np.float32))[None].to(dev)
        m = torch.ones((1, 1), device=dev)
        ferb = torch.from_numpy(np.ascontiguousarray(self.feat_erb))[None, None].to(dev)
        fspec = torch.from_numpy(np.stack([self.feat_spec.real, self.feat_spec.imag], 0).astype(np.float32))[None].to(dev)
        with torch.no_grad():
            v = model.film(emb_spk, m)
            ferb = ferb + (model.freq_mixer(ferb) - ferb) * m.view(-1, 1, 1, 1)
            e0, e1, e2, e3, emb, c0, lsnr = base.enc(ferb, fspec, film=(v.gamma_enc, v.beta_enc))
            a = model.cross_attn(emb, emb_spk, mask=m)
            emb_d, sg = a.output, a.subband_gains[0, 0]
            self.lsnr = lsnr.reshape(-1).cpu().numpy().astype(np.float32)
            flags = [rp.apply_stages(x, thresholds) for x in self.lsnr]
            erb_idx = [t for t, f in enumerate(flags) if f[0]]
            df_idx = [t for t, f in enumerate(flags) if f[2]]
            if erb_idx:
                i = torch.tensor(erb_idx, device=dev)
                mm = base.erb_dec(emb_d[:, i], e3[:, :, i], e2[:, :, i], e1[:, :, i], e0[:, :, i]) * sg[i][None, None]
                mm = mm.cpu().numpy()
                self.gains = {t: mm[0, 0, k].astype(np.float32) for k, t in enumerate(erb_idx)}
            if df_idx:
                i = torch.tensor(df_idx, device=dev)
                c = base.df_dec(emb_d[:, i], c0[:, :, i], film=(v.gamma_df, v.beta_df))
                c = c[0].cpu().numpy().reshape(len(df_idx), rp.NB_DF, rp.DF_ORDER, 2)
                self.coefs = {t: (c[k, ..., 0] + 1j * c[k, ..., 1]).astype(np.complex64) for k, t in enumerate(df_idx)}


class Runner:
    """Roda uma mistura pelo caminho do runtime. `profile=None` = sem perfil (DFNet3 base, gating atual)."""

    def __init__(self, ckpt: Path, arch: str, gating: str, device: str):
        self.model = build_model(ckpt, arch, device)
        self.base_cpu = copy.deepcopy(self.model.base).to("cpu").eval()
        self.thr = GATINGS[gating]

    def run(self, noisy: np.ndarray, profile: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        n = np.asarray(noisy).size
        # o runtime atrasa 1.440 amostras: sem cauda de silêncio os últimos 3 hops da saída se perderiam e o SI-SDR
        # do passthrough puro cairia para ~14 dB. A cauda é ruído a -55 dBFS (silêncio digital seria pulado pelo atalho
        # do runtime e não empurraria a saída) e sai do cômputo (estágios cortados em n // HOP).
        tail = np.random.default_rng(0).standard_normal(4 * rp.HOP).astype(np.float32) * np.float32(10 ** (-55 / 20))
        audio = np.concatenate([np.asarray(noisy, np.float32).reshape(-1), tail])
        if profile is None:
            res = rp.run_replica(audio, rp.TorchFrameModel(self.base_cpu, None, audio, CURRENT), CURRENT)
        else:
            res = rp.run_replica(audio, ProfileFrameModel(self.model, profile, audio, self.thr), self.thr)
        return rp.align_runtime(res.audio, n), res.stages[: n // rp.HOP]


# --- métricas ------------------------------------------------------------------------------------------------------

ZERO_STAGES = (int(rp.Stage.ZERO), int(rp.Stage.SILENT))


def _db(a: float, b: float) -> float:
    return float(10 * np.log10((a + 1e-20) / (b + 1e-20)))


def lsq_coefs(est: np.ndarray, tracks: list[np.ndarray]) -> np.ndarray:
    A = np.stack(tracks, 1).astype(np.float64)
    co, *_ = np.linalg.lstsq(A, np.asarray(est, np.float64), rcond=None)
    return co


def interf_suppression_db(est: np.ndarray, noisy: np.ndarray, tgt: np.ndarray, ints: np.ndarray) -> tuple[float, float]:
    """(supressão do interferente dB, ganho do alvo dB) por projeção de mínimos quadrados da saída sobre as trilhas.
    Supressão = energia ponderada dos coeficientes dos interferentes na entrada contra a da saída."""
    tracks = [tgt] + list(ints)
    en = np.array([float(np.sum(x.astype(np.float64) ** 2)) for x in ints])
    co_o, co_i = lsq_coefs(est, tracks), lsq_coefs(noisy, tracks)

    def leak(co):
        return float(np.sum(co[1:] ** 2 * en) / np.sum(en))

    return _db(leak(co_i), leak(co_o)), float(20 * np.log10(max(abs(co_o[0]), 1e-4)))


def frame_mask_samples(mask: np.ndarray, n: int) -> np.ndarray:
    m = np.repeat(mask, tm.HOP)
    return np.pad(m, (0, max(0, n - m.size)))[:n]


def item_metrics(view: str, rec: dict, r: tm.Rendered, noisy: np.ndarray, enh: np.ndarray, stages: np.ndarray,
                 base: dict) -> dict:
    """Métricas de uma execução. `base`: STOI/ESTOI/SI-SDR da entrada (calculados uma vez por item)."""
    from cctrain.eval.metrics import stoi

    out: dict = {}
    st = np.asarray(stages)
    zero = np.isin(st, ZERO_STAGES)
    zero_only = st == int(rp.Stage.ZERO)
    nf = min(len(st), len(r.tgt_act))
    tact, iact = r.tgt_act[:nf], r.int_act[:nf]
    n_int = r.interf.shape[0]
    out["stage_zero_pct"] = float(zero.mean() * 100)
    out["stage_zero_only_pct"] = float(zero_only.mean() * 100)
    out["stage_none_pct"] = float((st == int(rp.Stage.NONE)).mean() * 100)
    out["stage_erb_only_pct"] = float((st == int(rp.Stage.ERB_ONLY)).mean() * 100)
    if view == "int":
        e_in, e_out = float(np.sum(noisy.astype(np.float64) ** 2)), float(np.sum(enh.astype(np.float64) ** 2))
        out["leak_db"] = _db(e_out, e_in)
        return out
    t = r.target
    out["tgt_zero_pct"] = float(zero[:nf][tact].mean() * 100) if tact.any() else float("nan")
    out["tgt_zero_only_pct"] = float(zero_only[:nf][tact].mean() * 100) if tact.any() else float("nan")
    out["sisdr_out"] = si_sdr(enh, t)
    out["d_sisdr"] = out["sisdr_out"] - base["sisdr_in"]
    ms = frame_mask_samples(tact, len(t))
    if ms.sum() > 4800:
        mb = ms.astype(bool)
        out["d_sisdr_act"] = si_sdr(enh[mb], t[mb]) - si_sdr(noisy[mb], t[mb])
    out["d_stoi"] = float(stoi(enh, t, tm.SR)[0]) - base["stoi_in"]
    if view == "mix":
        out["d_estoi"] = float(stoi(enh, t, tm.SR, extended=True)[0]) - base["estoi_in"]
    if n_int:
        supp, gain = interf_suppression_db(enh, noisy, t, r.interf)
        out["interf_supp_db"], out["target_gain_db"] = supp, gain
        only_i = iact & ~tact
        if view == "mix" and only_i.sum() > 5:
            fm = frame_mask_samples(only_i, len(t)).astype(bool)
            out["leak_intonly_frames_db"] = _db(float(np.sum(enh[fm].astype(np.float64) ** 2)),
                                                float(np.sum(noisy[fm].astype(np.float64) ** 2)))
            out["int_only_zero_pct"] = float(zero[:nf][only_i].mean() * 100)
        if view == "mix" and rec["cell"]["axis"] == "temporal":
            for name, m in (("t_only", tact & ~iact), ("both", tact & iact), ("sil", ~tact & ~iact)):
                if m.sum() > 5:
                    out[f"{name}_zero_pct"] = float(zero[:nf][m].mean() * 100)
    else:
        out["target_gain_db"] = float(20 * np.log10(max(abs(lsq_coefs(enh, [t])[0]), 1e-4)))
    return out


def input_baseline(r: tm.Rendered, noisy: np.ndarray) -> dict:
    from cctrain.eval.metrics import stoi

    t = r.target
    return {"sisdr_in": si_sdr(noisy, t), "stoi_in": float(stoi(noisy, t, tm.SR)[0]),
            "estoi_in": float(stoi(noisy, t, tm.SR, extended=True)[0])}


def run_item(runner: Runner, rec: dict, emb_of: Callable[[dict], np.ndarray],
             swap_emb_of: Callable[[dict], np.ndarray | None]) -> list[dict]:
    """Todas as vistas da célula do item. Devolve uma linha por vista."""
    cell = tm.Cell(**{k: v for k, v in rec["cell"].items() if k != "cid"})
    r = tm.render(rec)
    rows = []
    noisy_mix = r.noisy(True)
    base = input_baseline(r, noisy_mix)
    prof = emb_of(rec)
    for view in cell.views():
        if view == "mix":
            enh, st = runner.run(noisy_mix, prof)
            noisy = noisy_mix
        elif view == "noprof":
            enh, st = runner.run(noisy_mix, None)
            noisy = noisy_mix
        elif view == "swap":
            enh, st = runner.run(noisy_mix, swap_emb_of(rec))
            noisy = noisy_mix
        else:  # int
            noisy = r.noisy(False)
            enh, st = runner.run(noisy, prof)
        row = {"id": rec["id"], "cid": rec["cell"]["cid"], "view": view, "target": rec["target"]["spk"],
               **item_metrics(view, rec, r, noisy, enh, st, base)}
        rows.append(row)
    return rows


# --- agregação -------------------------------------------------------------------------------------------------------


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cell_table(rows: Iterable[dict]) -> dict[tuple[str, str], dict]:
    g: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        g[(r["cid"], r["view"])].append(r)
    out = {}
    for k, rs in g.items():
        keys = {m for r in rs for m in r if m not in ("id", "cid", "view", "target")}
        agg = {"n": len(rs)}
        for m in sorted(keys):
            vals = np.array([r[m] for r in rs if m in r and r[m] is not None], float)
            vals = vals[np.isfinite(vals)]
            agg[m] = float(vals.mean()) if vals.size else float("nan")
            if m in ("leak_db", "leak_intonly_frames_db") and vals.size:
                agg[m + "_median"] = float(np.median(vals))
        out[k] = agg
    return out


def parse_cid(cid: str) -> dict:
    axis, n, sir, lv, noise, pat = cid.split("|")
    return {"axis": axis, "n": int(n[1:]), "sir": None if sir[3:] == "na" else float(sir[3:]),
            "level": float(lv[2:]), "noise": noise, "pattern": pat}


def bootstrap(vals: np.ndarray, B: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, v.size, (B, v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


@dataclass
class Table:
    cells: dict[tuple[str, str], dict]
    rows: list[dict]

    def sel(self, view: str, axis: str = "matrix", **kw) -> list[tuple[dict, dict]]:
        out = []
        for (cid, v), agg in self.cells.items():
            if v != view:
                continue
            p = parse_cid(cid)
            if p["axis"] != axis:
                continue
            if all((p[k] in val if isinstance(val, (tuple, list, set)) else p[k] == val) for k, val in kw.items()):
                out.append((p, agg))
        return sorted(out, key=lambda x: (x[0]["n"], x[0]["sir"] or 0, x[0]["level"], x[0]["noise"]))

    def pooled(self, view: str, metric: str, axis: str = "matrix", **kw) -> tuple[float, float, float]:
        cids = {tuple(sorted(p.items(), key=lambda kv: kv[0])) for p, _ in self.sel(view, axis, **kw)}
        vals = []
        for r in self.rows:
            if r["view"] != view or metric not in r or r[metric] is None:
                continue
            p = parse_cid(r["cid"])
            if p["axis"] == axis and tuple(sorted(p.items(), key=lambda kv: kv[0])) in cids:
                vals.append(r[metric])
        return bootstrap(np.array(vals, float))


def make_table(rows: list[dict]) -> Table:
    return Table(cell_table(rows), rows)


# --- critérios de aceite ---------------------------------------------------------------------------------------------

NOMINAL_LEVEL = -30.0


@dataclass
class CriterionResult:
    num: int
    name: str
    target: str
    status: str  # PASS | FAIL | N/A
    nominal: str  # valor/estado no nível nominal (-30 dBFS)
    all_levels: str
    detail: str = ""


def _check(cells: list[tuple[dict, dict]], fn: Callable[[dict], tuple[bool, float]], fmt: str = "{:+.2f}",
           worst: str = "min") -> tuple[str, str]:
    """(estado, pior valor). `fn(agg) -> (ok, valor)`; cél. sem dado = N/A."""
    oks, vals = [], []
    for _, a in cells:
        ok, v = fn(a)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        oks.append(ok)
        vals.append(v)
    if not oks:
        return "N/A", "-"
    w = min(vals) if worst == "min" else max(vals)
    return ("PASS" if all(oks) else "FAIL") + f" ({sum(oks)}/{len(oks)})", fmt.format(w)


def evaluate_criteria(t: Table, ref: Table | None) -> list[CriterionResult]:
    res: list[CriterionResult] = []

    def both(num, name, target, fn_cells, fn, fmt="{:+.2f}", worst="min", ref_needed=False):
        if ref_needed and ref is None:
            res.append(CriterionResult(num, name, target, "N/A", "sem referência (--ref)", "-"))
            return
        out = []
        for levels in ((NOMINAL_LEVEL,), tm.LEVELS):
            cells = fn_cells(levels)
            out.append(_check(cells, fn, fmt, worst))
        status = "PASS" if out[0][0].startswith("PASS") else ("FAIL" if out[0][0].startswith("FAIL") else "N/A")
        res.append(CriterionResult(num, name, target, status, f"{out[0][0]}; pior {out[0][1]}",
                                   f"{out[1][0]}; pior {out[1][1]}"))

    # 1: N=0, SNR 5: perfil >= sem perfil - 0.5 dB e dSTOI >= 0
    def c1_cells(levels):
        out = []
        for p, a in t.sel("mix", n=0, noise="snr5", level=tuple(levels)):
            npf = t.cells.get((tm.Cell("matrix", 0, None, p["level"], "snr5").cid, "noprof"), {})
            out.append((p, a | {"_noprof_d_sisdr": npf.get("d_sisdr", float("nan"))}))
        return out

    both(1, "N=0 (SNR 5): perfil não piora", "dSI-SDR(perfil) >= sem perfil - 0,5 dB e dSTOI >= 0",
         c1_cells, lambda a: (a["d_sisdr"] >= a["_noprof_d_sisdr"] - 0.5 and a["d_stoi"] >= 0,
                              min(a["d_sisdr"] - a["_noprof_d_sisdr"] + 0.5, a["d_stoi"] * 100)),
         fmt="{:+.2f} (margem; dSTOI em centésimos)")

    # 2: N=1, SIR 0, silencioso
    both(2, "N=1, SIR 0, silencioso", "dSI-SDR >= +4, dSTOI >= +0,03, ganho do alvo >= -3 dB, supressão >= 15 dB",
         lambda lv: t.sel("mix", n=1, sir=0.0, noise="silent", level=tuple(lv)),
         lambda a: (a["d_sisdr"] >= 4 and a["d_stoi"] >= 0.03 and a["target_gain_db"] >= -3 and a["interf_supp_db"] >= 15,
                    min(a["d_sisdr"] - 4, (a["d_stoi"] - 0.03) * 100, a["target_gain_db"] + 3, a["interf_supp_db"] - 15)),
         fmt="{:+.2f} (pior margem)")

    # 3: N=2,3, SIR 0 (todos os ruídos)
    both(3, "N=2 e N=3, SIR 0 por interferente", "dSI-SDR >= +3 dB e dSTOI >= 0",
         lambda lv: t.sel("mix", n=(2, 3), sir=0.0, level=tuple(lv)),
         lambda a: (a["d_sisdr"] >= 3 and a["d_stoi"] >= 0, min(a["d_sisdr"] - 3, a["d_stoi"] * 100)),
         fmt="{:+.2f} (pior margem)")

    # 4: só-interferente (o limiar de %ZERO depende do ruído)
    oks_nom, oks_all = [], []
    for levels, store in (((NOMINAL_LEVEL,), oks_nom), (tm.LEVELS, oks_all)):
        for nz, thr in (("silent", 90.0), ("snr5", 80.0)):
            for p, a in t.sel("int", n=(1, 2, 3), noise=nz, level=tuple(levels)):
                store.append((a["stage_zero_pct"] >= thr and a.get("leak_db_median", 0) <= -30, a["stage_zero_pct"],
                              a.get("leak_db_median", float("nan"))))
    def _s(store):
        if not store:
            return "N/A"
        ok = [x[0] for x in store]
        return (("PASS" if all(ok) else "FAIL") + f" ({sum(ok)}/{len(ok)}); pior %ZERO {min(x[1] for x in store):.1f}%, "
                f"pior vazamento {max(x[2] for x in store):+.1f} dB")
    res.append(CriterionResult(4, "Só-interferente (N=1..3, perfil)",
                               "%ZERO >= 90% (silencioso) / >= 80% (SNR 5); vazamento mediano <= -30 dB",
                               "PASS" if _s(oks_nom).startswith("PASS") else ("FAIL" if _s(oks_nom).startswith("FAIL") else "N/A"),
                               _s(oks_nom), _s(oks_all)))

    # 5: interferente fraco
    both(5, "Interferente fraco (SIR +20/+30, silencioso)", "supressão do interferente >= 15 dB",
         lambda lv: t.sel("mix", n=(1, 2, 3, 4), sir=(20.0, 30.0), noise="silent", level=tuple(lv)),
         lambda a: (a["interf_supp_db"] >= 15, a["interf_supp_db"] - 15), fmt="{:+.2f} dB (pior margem)")

    # 6: alvo zerado indevidamente (N=0, perfil), só decisão ZERO do modelo
    both(6, "Alvo zerado indevidamente (N=0)", "<= 2% dos quadros do alvo em ZERO, todo nível >= -45",
         lambda lv: t.sel("mix", n=0, level=tuple(lv)),
         lambda a: (a["tgt_zero_only_pct"] <= 2.0, a["tgt_zero_only_pct"]), fmt="{:.2f}% (pior)", worst="max")

    # 7: sem perfil vs v2 sem perfil (N=0, nível nominal, SNR 20 e 5)
    if ref is None:
        res.append(CriterionResult(7, "Sem perfil ~ v2 sem perfil", "dSI-SDR <= 0,2 dB e dSTOI <= 0,005 de diferença",
                                   "N/A", "sem referência (--ref)", "-"))
    else:
        oks, w = [], []
        for nz in ("snr20", "snr5"):
            cid = tm.Cell("matrix", 0, None, NOMINAL_LEVEL, nz).cid
            a, b = t.cells.get((cid, "noprof")), ref.cells.get((cid, "noprof"))
            if a and b:
                dd, ds = abs(a["d_sisdr"] - b["d_sisdr"]), abs(a["d_stoi"] - b["d_stoi"])
                oks.append(dd <= 0.2 and ds <= 0.005)
                w.append((dd, ds))
        status = "N/A" if not oks else ("PASS" if all(oks) else "FAIL")
        res.append(CriterionResult(7, "Sem perfil ~ v2 sem perfil", "|ddSI-SDR| <= 0,2 dB e |ddSTOI| <= 0,005 (N=0, SNR 20/5)",
                                   status, f"{status}; pior {max(x[0] for x in w):.3f} dB / {max(x[1] for x in w):.4f}" if w else "-",
                                   "-"))

    # 8: robustez de nível: dSI-SDR(-45) >= dSI-SDR(-18) - 1.5 (N=1, SIR 0)
    oks, drops = [], []
    for nz in tm.NOISES:
        lo = t.cells.get((tm.Cell("matrix", 1, 0.0, -45.0, nz).cid, "mix"))
        hi = t.cells.get((tm.Cell("matrix", 1, 0.0, -18.0, nz).cid, "mix"))
        if lo and hi:
            drops.append(hi["d_sisdr"] - lo["d_sisdr"])
            oks.append(drops[-1] <= 1.5)
    st8 = "N/A" if not oks else ("PASS" if all(oks) else "FAIL")
    res.append(CriterionResult(8, "Robustez de nível (N=1, SIR 0)", "dSI-SDR(-45) a até 1,5 dB de dSI-SDR(-18)",
                               st8, f"{st8} ({sum(oks)}/{len(oks)}); pior queda {max(drops):+.2f} dB" if drops else "-", "-"))

    # 9: pareado com o v2 (ref)
    if ref is None:
        res.append(CriterionResult(9, "Pareado com v2", ">= v2 em >= 80% das células N>=1; nunca pior que 0,5 dB em N=0",
                                   "N/A", "sem referência (--ref)", "-"))
    else:
        wins, tot, worst0 = 0, 0, 0.0
        for p, a in t.sel("mix"):
            b = ref.cells.get((tm.Cell("matrix", p["n"], p["sir"], p["level"], p["noise"]).cid, "mix"))
            if not b:
                continue
            d = a["d_sisdr"] - b["d_sisdr"]
            if p["n"] == 0:
                worst0 = min(worst0, d)
            else:
                tot += 1
                wins += d >= 0
        ok = tot > 0 and wins / tot >= 0.8 and worst0 >= -0.5
        res.append(CriterionResult(9, "Pareado com v2", ">= v2 em >= 80% das células N>=1; nunca pior que 0,5 dB em N=0",
                                   "PASS" if ok else "FAIL",
                                   f"{'PASS' if ok else 'FAIL'}; {wins}/{tot} células N>=1 ({100*wins/max(tot,1):.0f}%), pior N=0 {worst0:+.2f} dB",
                                   "-"))

    # 10: especificidade: perfil trocado => ganho do alvo <= -15 dB
    both(10, "Especificidade (perfil trocado)", "ganho do alvo <= -15 dB",
         lambda lv: t.sel("swap", n=1, sir=0.0, level=tuple(lv)),
         lambda a: (a["target_gain_db"] <= -15, a["target_gain_db"]), fmt="{:+.1f} dB (pior)", worst="max")

    res.append(CriterionResult(11, "Runtime", "paridade >= 60 dB, < 10 ms/hop, MACs inalterados", "N/A",
                               "fora do harness (gate de export/paridade)", "-"))
    return res


def fmt_ci(t: Table, view: str, metric: str, axis="matrix", **kw) -> str:
    m, lo, hi = t.pooled(view, metric, axis, **kw)
    return "-" if not math.isfinite(m) else f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
