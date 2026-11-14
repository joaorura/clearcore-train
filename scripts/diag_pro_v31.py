#!/usr/bin/env python3
"""Diagnóstico do L_zero do pro-v4 (pDFNet3 Pro v3.1): normas e cossenos de gradiente por termo e por grupo de
parâmetros, distribuição do lsnr nos quadros só-interferente e sonda linear (separabilidade alvo × interferente)
antes e depois da cross-attention. Só leitura: não grava checkpoint.

Uso: python scripts/diag_pro_v31.py --ckpt <ckpt> [--ckpt <ckpt> ...] --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import train_pro_v4 as T  # noqa: E402
from cctrain.m1 import pools  # noqa: E402
from cctrain.m2.config import load_stage_f_config  # noqa: E402
from cctrain.models.cross_attention import SubbandCrossAttentionConfig  # noqa: E402
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3  # noqa: E402
from cctrain.models.film import FilmConfig, FilmGenerator  # noqa: E402
from cctrain.models.pdfnet3_pro import PDFNet3Pro  # noqa: E402
from cctrain.paths import Paths  # noqa: E402
from cctrain.train.checkpoint import load_checkpoint  # noqa: E402
from cctrain.train.embed_bank import ArrayEmbeddingBank  # noqa: E402
from cctrain.train.stage_c import _lsnr_target, _to_device  # noqa: E402
from cctrain.train.stage_f import HOP, StageFHParams, prepare_student  # noqa: E402
from cctrain.data.mix_v4 import StageV4Dataset, V4Params, build_pools  # noqa: E402
from df.loss import Istft  # noqa: E402
from df.model import ModelParams  # noqa: E402

PCT = (5, 25, 50, 75, 95)


def build(device):
    base, df_state = load_upstream_dfnet3(device=device)
    teacher, _ = load_upstream_dfnet3(device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    hp = StageFHParams.from_config(load_stage_f_config(), run_id="diag", num_workers=0, batch_size=8)
    gen = FilmGenerator(FilmConfig(log_gamma_bound=hp.log_gamma_bound, beta_bound=hp.beta_bound)).to(device)
    attn = SubbandCrossAttentionConfig(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4, num_tokens=16,
                                       token_dim=128)
    model = PDFNet3Pro(base, gen, cross_attention_config=attn, enable_cross_attention=True).to(device)
    istft = Istft(960, HOP, torch.as_tensor(df_state.fft_window().copy())).to(device)
    return model, teacher, istft, hp


def load_into(model, path):
    ck = load_checkpoint(Path(path))
    cur = model.state_dict()
    compat = {k: v for k, v in ck.model.items() if k in cur and v.shape == cur[k].shape}
    model.load_state_dict(compat, strict=False)
    prepare_student(model)
    return len(compat), len(cur)


def groups(model):
    b = model.base
    g = {
        "lsnr_fc": list(b.enc.lsnr_fc.parameters()),
        "erb_mask": list(b.erb_dec.parameters()) + list(model.cross_attn.parameters()),
        "emb_gru": list(b.enc.emb_gru.parameters()),
        "film": list(model.film.parameters()),
    }
    g["all"] = [p for p in model.parameters() if p.requires_grad]
    return {k: [p for p in v if p.requires_grad] for k, v in g.items()}


def flat(gs, ps):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(gs, ps)])


def cos(a, b):
    return float(F.cosine_similarity(a[None], b[None]).item()) if a.norm() > 0 and b.norm() > 0 else float("nan")


@torch.no_grad()
def probe_feats(model, b):
    """emb antes da cross-attention (entrada do lsnr_fc) e saída da cross-attention, por quadro."""
    cap = {}
    h = model.base.enc.register_forward_hook(lambda m, i, o: cap.__setitem__("emb", o[4].detach()))
    (_, _, lsnr, _), attn = model(b["spec"].clone(), b["feat_erb"], b["feat_spec"], b["embedding"], b["mask"],
                                  return_attention_details=True)
    h.remove()
    return cap["emb"], attn.output.detach(), lsnr.detach()


def fit_probe(xtr, ytr, xte, yte, steps=400):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    w = torch.zeros(xtr.shape[1], device=xtr.device, requires_grad=True)
    c = torch.zeros(1, device=xtr.device, requires_grad=True)
    pw = (1 - ytr).sum() / ytr.sum().clamp(min=1)
    opt = torch.optim.LBFGS([w, c], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lo = F.binary_cross_entropy_with_logits(xtr @ w + c, ytr, pos_weight=pw) + 1e-3 * (w * w).sum()
        lo.backward()
        return lo

    opt.step(closure)
    with torch.no_grad():
        s = xte @ w + c
    return bal_acc(s, yte), auc(s, yte)


def bal_acc(s, y, thr=0.0):
    p = s > thr
    tpr = (p & (y > 0.5)).float().sum() / (y > 0.5).sum().clamp(min=1)
    tnr = (~p & (y < 0.5)).float().sum() / (y < 0.5).sum().clamp(min=1)
    return float((tpr + tnr) / 2)


def auc(s, y):
    s, y = s.double().cpu().numpy(), y.cpu().numpy() > 0.5
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--steps", default="40000,40001,80500,80501", help="passos de treino cujos lotes são reproduzidos")
    ap.add_argument("--half", type=int, default=8, help="amostras por forward (o lote de 16 vira 2 metades)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    paths = Paths.from_env()
    model, teacher, istft, hp = build(dev)
    df_params = ModelParams()

    vp = V4Params(batch_size=16, total_steps=85500)
    bank = ArrayEmbeddingBank.load(paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz")
    speech = pools.by_split(pools.speech_records(paths), "train")
    noise_train, _ = pools.noise_pools()
    pl = build_pools(speech, noise_train, pools.rir_records(), bank, paths.data, seed=vp.seed)
    ds = StageV4Dataset(pl, bank, paths.data, vp)
    idx = [s * 16 + k for s in map(int, a.steps.split(",")) for k in range(16)]
    batches = [_to_device(b, dev) for b in DataLoader(ds, batch_size=a.half, sampler=idx, num_workers=0)]
    print(f"{len(batches)} lotes de {a.half}", flush=True)

    report = {}
    for ck in a.ckpt:
        n, tot = load_into(model, ck)
        cka = load_checkpoint(Path(ck)).extra.get("args") or {}
        args = SimpleNamespace(lambda_leak=cka.get("lambda_leak", 2.0), lambda_ovl=cka.get("lambda_ovl", 14.0),
                               lambda_guard=cka.get("lambda_guard", 1.6), lambda_zero=cka.get("lambda_zero", 1.3e-2),
                               lambda_keep=cka.get("lambda_keep", 0.11), lsnr_factor=cka.get("lsnr_factor", 1e-3),
                               formant_weight=cka.get("formant_weight", 2.5))
        G = groups(model)
        terms = ("mr", "zero", "keep", "lsnr", "guard", "leak", "total")
        norms = {t: {g: [] for g in G} for t in terms}
        cosr = {k: {g: [] for g in ("lsnr_fc", "emb_gru", "all")} for k in
                ("zero_keep", "zero_mr", "zero_lsnr", "keep_lsnr", "zero_total", "zero_guard")}
        sums = {t: {g: None for g in ("lsnr_fc", "emb_gru", "all")} for t in terms}
        vals = {t: [] for t in terms}
        lz_oi, lt_oi, lz_t, lt_t, zfrac = [], [], [], [], []
        pe, pa, pl_, py = [], [], [], []
        for b in batches:
            prepare_student(model)
            model.zero_grad(set_to_none=True)
            parts = T.step_v4(model, teacher, istft, df_params, hp, b, args)
            raw = {"mr": parts["mr"], "zero": parts["zero"] / args.lambda_zero,
                   "keep": parts["keep"] / args.lambda_keep, "lsnr": parts["lsnr"],
                   "guard": parts["guard"] / args.lambda_guard, "leak": parts["leak"] / args.lambda_leak,
                   "total": parts["total"]}
            fl = {}
            for t, v in raw.items():
                vals[t].append(float(v.detach()))
                if not v.requires_grad:
                    for g in G:
                        norms[t][g].append(0.0)
                    continue
                gs = torch.autograd.grad(v, G["all"], retain_graph=True, allow_unused=True)
                gmap = {id(p): gg for p, gg in zip(G["all"], gs)}
                for g, ps in G.items():
                    vec = flat([gmap.get(id(p)) for p in ps], ps)
                    norms[t][g].append(float(vec.norm()))
                    if g in ("lsnr_fc", "emb_gru", "all"):
                        fl[(t, g)] = vec
                        sums[t][g] = vec.clone() if sums[t][g] is None else sums[t][g] + vec
            for k in cosr:
                x, y = k.split("_")
                for g in cosr[k]:
                    if (x, g) in fl and (y, g) in fl:
                        cosr[k][g].append(cos(fl[(x, g)], fl[(y, g)]))
            del fl, parts
            # lsnr nos quadros
            with torch.no_grad():
                emb, embc, lsnr = probe_feats(model, b)
                nb, Tn = b["spec"].shape[0], b["spec"].shape[2]
                lv = lsnr.reshape(nb, -1)
                lt = _lsnr_target(df_params, b["spec_clean"], b["spec"])
                prof = (b["mask"].reshape(nb) > 0)[:, None]
                ta, ia, dsil = b["target_active"][:, :Tn], b["interf_active"][:, :Tn], b["digital_silence"][:, :Tn]
                oi = prof & ~ta & ia & ~dsil
                kt = prof & ta & ~dsil
                lz_oi.append(lv[oi].cpu()); lt_oi.append(lt[oi].cpu())
                lz_t.append(lv[kt].cpu()); lt_t.append(lt[kt].cpu())
                sel = oi | kt
                pe.append(emb[sel].float()); pa.append(embc[sel].float()); pl_.append(lv[sel].float())
                py.append(kt[sel].float())
                torch.cuda.empty_cache()
        lz_oi, lt_oi, lz_t, lt_t = map(torch.cat, (lz_oi, lt_oi, lz_t, lt_t))
        E, A, L, Y = map(torch.cat, (pe, pa, pl_, py))
        h = len(Y) // 2  # metade inicial (primeiros lotes) treina, metade final testa
        res_probe = {
            "n_frames": int(len(Y)), "frac_target": float(Y.mean()),
            "lsnr_raw_auc": auc(L[h:], Y[h:]), "lsnr_thr-10_balacc": bal_acc(L[h:], Y[h:], -10.0),
            "emb_pre_ca": fit_probe(E[:h], Y[:h], E[h:], Y[h:]),
            "emb_post_ca": fit_probe(A[:h], Y[:h], A[h:], Y[h:]),
        }
        cos_sum = {}
        for k in cosr:
            x, y = k.split("_")
            cos_sum[k] = {g: cos(sums[x][g], sums[y][g]) for g in ("lsnr_fc", "emb_gru", "all")
                          if sums[x][g] is not None and sums[y][g] is not None}
        pct = lambda t: [round(float(v), 2) for v in np.percentile(t.numpy(), PCT)]  # noqa: E731
        report[ck] = {
            "loaded": f"{n}/{tot}", "lambdas": vars(args),
            "loss_mean": {t: float(np.mean(v)) for t, v in vals.items()},
            "grad_norm_mean": {t: {g: float(np.mean(v)) for g, v in d.items()} for t, d in norms.items()},
            "grad_norm_weighted_lsnr_fc": {
                "zero": float(np.mean(norms["zero"]["lsnr_fc"])) * args.lambda_zero,
                "keep": float(np.mean(norms["keep"]["lsnr_fc"])) * args.lambda_keep,
                "lsnr": float(np.mean(norms["lsnr"]["lsnr_fc"])),
                "mr": float(np.mean(norms["mr"]["lsnr_fc"]))},
            "cos_per_batch_mean": {k: {g: float(np.nanmean(v)) if v else None for g, v in d.items()}
                                   for k, d in cosr.items()},
            "cos_summed": cos_sum,
            "lsnr_only_interf": {"n": int(len(lz_oi)), "pred_pct": pct(lz_oi), "target_pct": pct(lt_oi),
                                 "frac_lt_-10": float((lz_oi < -10).float().mean()),
                                 "frac_lt_-12": float((lz_oi < -12).float().mean()),
                                 "frac_target_lt_-10": float((lt_oi < -10).float().mean()),
                                 "mean_relu_sq_+12": float((F.relu(lz_oi + 12) ** 2).mean())},
            "lsnr_target_frames": {"n": int(len(lz_t)), "pred_pct": pct(lz_t), "target_pct": pct(lt_t),
                                   "frac_lt_-10": float((lz_t < -10).float().mean()),
                                   "frac_target_lt_-10": float((lt_t < -10).float().mean()),
                                   "frac_target_lt_-8": float((lt_t < -8).float().mean())},
            "probe": res_probe,
        }
        print(json.dumps(report[ck], indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
