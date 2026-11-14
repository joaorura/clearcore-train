#!/usr/bin/env python3
"""Perfil e paridade das perdas do passo pro-v5 (P3): `step_v5_ref` (antiga) vs `step_v5` (fast).

    python scripts/profile_v5_loss.py --cache /caminho/lotes.pt --dump 4 --batch 48   # gera lotes reais (gpu-mix)
    python scripts/profile_v5_loss.py --cache /caminho/lotes.pt --iters 10            # paridade + tempos

Mede, com eventos CUDA: (1) o passo inteiro (forward, perdas, backward de `total`) de cada implementação; (2) por
perda, o forward da seção e o backward isolado da perda até as saídas do modelo (autograd.grad com retain_graph).
Paridade: com o MESMO forward do modelo, valor de cada perda e gradientes dos parâmetros treináveis, ref vs fast."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pro_v5 as tp  # noqa: E402
from train_pro_v4 import gate_v4, leak_loss, ovl_loss  # noqa: E402
from train_pro_v42 import since_interferer_s, target_shadow, turn_loss  # noqa: E402

from cctrain.losses.dfnet import asymmetric_formant_loss, df_coef_temporal_regularization_loss, dfnet_loss, local_snr_loss  # noqa: E402
from cctrain.train.stage_c import _lsnr_target  # noqa: E402

LOSS_KEYS = ("mr", "os", "leak", "ovl", "lsnr", "turn", "n0g", "tail", "leak_abs", "df_reg")


class Timer:
    def __init__(self):
        self.ms = defaultdict(list)

    @contextmanager
    def __call__(self, name):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        yield
        e.record()
        self._pending.append((name, s, e))

    def start(self):
        self._pending = []

    def flush(self):
        torch.cuda.synchronize()
        for n, s, e in self._pending:
            self.ms[n].append(s.elapsed_time(e))
        self._pending = []


def ref_sections(model, istft, df_params, hp, b, a, amp, tm, fwd):
    """Cópia de `step_v5_ref` com seções medidas (mesma ordem e mesmas chamadas)."""
    spec, spec_clean, mask = b["spec"], b["spec_clean"], b["mask"]
    nb, T = spec.shape[0], spec.shape[2]
    spec_e, m, lsnr, df_coefs, profile = fwd
    with tm("spectral+shadow"):
        spec_m = model.base.mask(spec.clone(), m)
        if amp:
            base = model.base
            spec_e = base.df_op(spec.clone(), df_coefs)
            spec_e = torch.cat([spec_e[..., : base.nb_df, :], spec_m[..., base.nb_df:, :]], dim=-2)
        out, zero = gate_v4(spec, spec_m, spec_e, lsnr, profile)
    with tm("istft"):
        est, tgt = istft(out), istft(spec_clean)
    kw = dict(n_ffts=hp.fft_sizes, gamma=hp.loss_gamma)
    ta = b["target_active"][:, :T]
    ia = b["interf_active"][:, :T]
    ds = b["digital_silence"][:, :T]
    prof = profile[:, None]
    f_leak = (prof & ~ta & ia & ~ds).float()
    f_ovl = (prof & ta & ia).float()
    has_int = (b["n_interf"] >= 1)[:, None]
    p = {}
    with tm("mr+os"):
        p["mr"] = dfnet_loss(est, tgt, b["max_freq"], factor=hp.loss_factor, factor_complex=hp.loss_factor_complex, **kw)
        os_ = asymmetric_formant_loss(est, tgt, b["max_freq"], factor=hp.lambda_os, formant_weight=a.formant_weight,
                                      extra_formant_penalty=1.0, **kw)
        idx_n1 = (b["n_interf"] >= 1).nonzero(as_tuple=True)[0]
        if idx_n1.numel():
            os_ = os_ + (idx_n1.numel() / nb) * asymmetric_formant_loss(
                est[idx_n1], tgt[idx_n1], b["max_freq"][idx_n1], factor=hp.lambda_os, formant_weight=a.formant_weight,
                extra_formant_penalty=1.0, **kw)
        p["os"] = os_
    with tm("leak+tail+leak_abs+ovl"):
        p["leak"] = a.lambda_leak * leak_loss(out, spec, f_leak)
        p["ovl"] = a.lambda_ovl * ovl_loss(out, spec_clean, b["spec_interf"], f_ovl)
        iw = b["interf_active_wet45"][:, :T]
        since_i = since_interferer_s(ia)
        f_tail = (prof & ~ta & ~ia & iw & ~ds & (since_i <= a.tail_s)).float()
        p["tail"] = a.lambda_tail * leak_loss(out, spec, f_tail)
        p["leak_abs"] = a.lambda_leak_absent * leak_loss(out, spec, f_leak * (~b["has_target"]).float()[:, None])
    with tm("lsnr"):
        lsnr_t = _lsnr_target(df_params, spec_clean, spec)
        p["lsnr"] = local_snr_loss(lsnr, lsnr_t, ~ds, factor=a.lsnr_factor)
    with tm("turn+n0g"):
        shadow = target_shadow(model, spec_clean, m, df_coefs, zero)
        valid = prof & has_int & b["has_target"][:, None] & ta & ~ia & ~ds
        turn, _ = turn_loss(shadow, spec_clean, valid, ta, ia, hinge_db=a.turn_hinge_db, boost=a.turn_boost,
                            decay_s=a.turn_decay_s)
        v0 = prof & (b["n_interf"] == 0)[:, None] & b["has_target"][:, None] & ta & ~ds
        n0g, _ = turn_loss(shadow, spec_clean, v0, ta, torch.zeros_like(ia), hinge_db=a.n0g_hinge_db, boost=1.0,
                           decay_s=1.0)
        p["turn"], p["n0g"] = a.lambda_turn * turn, a.lambda_n0g * n0g
    with tm("df_reg"):
        p["df_reg"] = df_coef_temporal_regularization_loss(df_coefs, time_dim=1) / (df_coefs.numel() / df_coefs.shape[1]) * 0.05
    return p


class Frozen(torch.nn.Module):
    """Devolve saídas do modelo já calculadas (o mesmo forward para ref e fast)."""

    def __init__(self, model, outs):
        super().__init__()
        self.base = model.base
        self._outs = outs

    def forward(self, *_a, **_k):
        return self._outs


def setup(argv):
    a = tp.parse_args(argv)
    device = "cuda"
    amp = a.amp == "bf16"
    from cctrain.m2.config import load_stage_f_config
    from cctrain.paths import Paths

    paths = Paths.from_env()
    hp = tp.StageFHParams.from_config(load_stage_f_config(), run_id="prof", num_workers=0, batch_size=a.global_batch)
    model, acfg, sites, base_kind, train_cond, _ = tp.build_v5(a, device, hp, paths)
    trainable = tp.prepare(model, train_cond)
    if amp:
        tp.fp32_spectral(model)
    istft = tp.Istft(960, tp.HOP, torch.as_tensor(tp._fft_window())).to(device)
    return a, device, amp, paths, hp, model, trainable, istft, tp.ModelParams()


def dump(a, paths, device, n, path):
    from torch.utils.data import DataLoader

    vp = tp.V4Params(seed=a.seed, segment_s=a.segment_s, batch_size=a.global_batch, total_steps=10_000, phase1_frac=0.0,
                     n_probs=tp._np(a.n_probs, 5, tp.N_PROBS), pattern_probs=tp._np(a.pattern_probs, 4, tp.PATTERN_PROBS),
                     p_target_absent=a.p_target_absent, p_interf_first=a.p_interf_first, p_turns=a.p_turns,
                     turn_gap_max_s=a.turn_gap_max_s, p_rir_turns=a.p_rir_turns)
    ds, _ = tp.make_dataset(paths, vp)
    ds = tp.StageV4RecipeDataset(ds.pools, ds.bank, ds.data_root, vp)
    mixer = tp.GpuMixer.from_dataset(ds, device)
    dl = DataLoader(ds, batch_size=a.global_batch, sampler=list(range(1000 * a.global_batch, (1000 + n) * a.global_batch)),
                    num_workers=a.workers or 3, worker_init_fn=tp.init_worker, collate_fn=tp.collate_recipes)
    out = []
    for batch in dl:
        batch = mixer(batch)
        mixer.fill_invalid(batch, ds)
        batch.pop("valid")
        batch.pop("index")
        out.append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()})
    torch.save(out, path)
    print(f"{len(out)} lotes -> {path}")


def rel(x, y):
    return float((x - y).abs().max() / y.abs().max().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--parity-batch", type=int, default=12, help="amostras por lote na paridade (memória)")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-args", default="", help="argumentos extras do train_pro_v5")
    o = ap.parse_args()
    argv = ["--variant", "P3", "--gpu-mix", "--amp", "bf16", "--global-batch", str(o.batch)] + o.train_args.split()
    a, device, amp, paths, hp, model, trainable, istft, params = setup(argv)
    if o.dump:
        dump(a, paths, device, o.dump, o.cache)
    batches = [tp._to_device_nb(b, device) for b in torch.load(o.cache)]
    batches = [{k: (v[: o.batch] if torch.is_tensor(v) and v.ndim and v.shape[0] >= o.batch else v) for k, v in b.items()}
               for b in batches]
    # pesos treináveis não nulos (no passo 0 os ramos zerados tornariam vários gradientes triviais)
    torch.manual_seed(0)
    with torch.no_grad():
        for p_ in trainable:
            if p_.abs().max() == 0:
                p_.normal_(0, 1e-3)

    report = {"parity": [], "step_ms": {}, "fwd_ms": {}, "bwd_ms": {}}
    # --- paridade (mesmo forward; algoritmos determinísticos) --------------------------------------------------------
    def cmp(g1, g2):
        n_, d_ = 0.0, 0.0
        for x, y in zip(g1, g2):
            if y is None or x is None:
                continue
            n_ += float(((x.double() - y.double()) ** 2).sum())
            d_ += float((y.double() ** 2).sum())
        return (n_ / d_) ** 0.5 if d_ > 0 else (0.0 if n_ == 0 else float("inf"))

    prev = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)
    for p_amp in (False, amp):
        worst = defaultdict(float)
        for b in batches:
            b = {k: (v[: o.parity_batch] if torch.is_tensor(v) and v.ndim and v.shape[0] == o.batch else v)
                 for k, v in b.items()}
            fwd = tp._forward_v5(model, b, p_amp)
            leaves = list(fwd[:4])
            pr = tp.step_v5_ref(Frozen(model, fwd[:4]), istft, params, hp, b, a, p_amp)
            pr2 = tp.step_v5_ref(Frozen(model, fwd[:4]), istft, params, hp, b, a, p_amp)
            pf = tp.step_v5(model, istft, params, hp, b, a, p_amp, _fwd=fwd)
            for k in ("total",) + LOSS_KEYS + ("g_turn_db", "g_turn_early_db", "g_n0_db", "zero_frac_only_interf",
                                               "zero_frac_target"):
                x, y = pf[k].detach().double(), pr[k].detach().double()
                d = 0.0 if (torch.isnan(x) & torch.isnan(y)).all() else float((x - y).abs() / y.abs().clamp_min(1e-30))
                worst[f"val:{k}"] = max(worst[f"val:{k}"], d)
            for k in ("total",) + LOSS_KEYS:
                def g(parts, wrt):
                    return torch.autograd.grad(parts[k], wrt, retain_graph=True, allow_unused=True)
                for tag, wrt in (("params", trainable), ("outs", leaves)):
                    gr = g(pr, wrt)
                    worst[f"grad_{tag}:{k}"] = max(worst[f"grad_{tag}:{k}"], cmp(g(pf, wrt), gr))
                    worst[f"noise_{tag}:{k}"] = max(worst[f"noise_{tag}:{k}"], cmp(g(pr2, wrt), gr))
            del fwd, leaves, pr, pr2, pf
            torch.cuda.empty_cache()
        report["parity"].append({"amp": bool(p_amp), **worst})
        print(f"paridade amp={p_amp} (máx. sobre {len(batches)} lotes; rel. L2; noise = ref vs ref):",
              json.dumps(worst, indent=0))
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev

    # --- passo inteiro ----------------------------------------------------------------------------------------------
    def full(impl, b, tm):
        tm.start()
        with tm("forward"):
            fwd = tp._forward_v5(model, b, amp)
        with tm("losses"):
            if impl == "ref":
                parts = tp.step_v5_ref(Frozen(model, fwd[:4]), istft, params, hp, b, a, amp)
            else:
                parts = tp.step_v5(model, istft, params, hp, b, a, amp, _fwd=fwd)
        with tm("backward"):
            parts["total"].backward()
        with tm("step"):
            pass
        tm.flush()
        for p_ in trainable:
            p_.grad = None

    for impl in ("ref", "fast"):
        tm = Timer()
        for it in range(o.warm + o.iters):
            for b in batches:
                if it == o.warm and b is batches[0]:
                    tm.ms.clear()
                torch.cuda.synchronize()
                full(impl, b, tm)
        report["step_ms"][impl] = {k: statistics.median(v) for k, v in tm.ms.items() if k != "step"}
        report["step_ms"][impl]["sum"] = sum(report["step_ms"][impl].values())
    print("passo:", json.dumps(report["step_ms"], indent=1))

    # --- por perda: forward da seção e backward isolado até as saídas do modelo ------------------------------------
    for impl in ("ref", "fast"):
        tf, tb = Timer(), Timer()
        for it in range(o.warm + o.iters):
            for b in batches:
                if it == o.warm and b is batches[0]:
                    tf.ms.clear()
                    tb.ms.clear()
                fwd = tp._forward_v5(model, b, amp)
                leaves = [t for t in fwd[:4]]
                torch.cuda.synchronize()
                tf.start()
                if impl == "ref":
                    parts = ref_sections(model, istft, params, hp, b, a, amp, tf, fwd)
                else:
                    parts = tp.step_v5(model, istft, params, hp, b, a, amp, timer=tf, _fwd=fwd)
                tf.flush()
                tb.start()
                for k in LOSS_KEYS:
                    with tb(k):
                        torch.autograd.grad(parts[k], leaves, retain_graph=True, allow_unused=True)
                with tb("ALL(total)"):
                    torch.autograd.grad(sum(parts[k] for k in LOSS_KEYS), leaves, allow_unused=True)
                tb.flush()
                del parts, fwd, leaves
        report["fwd_ms"][impl] = {k: statistics.median(v) for k, v in tf.ms.items()}
        report["bwd_ms"][impl] = {k: statistics.median(v) for k, v in tb.ms.items()}
    print("forward por seção:", json.dumps(report["fwd_ms"], indent=1))
    print("backward por perda (até as saídas do modelo):", json.dumps(report["bwd_ms"], indent=1))
    print("vram pico GiB:", torch.cuda.max_memory_allocated() / 2**30)
    if o.out:
        Path(o.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
