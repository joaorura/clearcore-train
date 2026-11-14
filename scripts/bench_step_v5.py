#!/usr/bin/env python3
"""Bench + paridade do passo do pro-v5 (forward + perdas + backward + clip + AdamW), com lotes reais em cache na GPU.

Pega K lotes reais do loader (--gpu-mix), força cada um homogêneo (todo com perfil, ou todo sem com --off-every), e
roda cada variante a partir do MESMO estado inicial: --parity-steps passos registrando a perda total e o vetor de
gradientes (antes do clip) para comparar com a referência (caminho atual de train_pro_v5), e depois --warm + --steps
passos cronometrados (sync só no fim; a variante "ref" faz os syncs do laço atual a cada passo).

Variantes (--variants, separadas por vírgula; '+' combina opções):
  ref                 step_v5_ref (perdas antigas) + syncs por passo do laço (finite.item, synchronize, float), AdamW padrão
  ref-nolog           idem, sem os syncs do laço (perdas lidas só no fim)
  fl / fl-nolog       step_v5 (perdas fast do perf/v5-loss, gate dinâmico) com / sem os syncs do laço
  +fl / +lc           (sf/graph/c-*) perdas fast do perf/v5-loss no StepRunner / --loss-compile
  sf                  step_v5_sf (sem sync) + gate estático
  bench / fused / foreach / cl / zg  cudnn.benchmark / AdamW fused / foreach / channels_last / zero_grad(set_to_none=False)
  graph               CUDA Graph manual de forward+perdas+backward (implica sf)
  c-default / c-ro / c-mat   torch.compile(mode=default / reduce-overhead / max-autotune-no-cudagraphs) (implica sf)
Ex.: --variants ref,ref-nolog,sf,sf+bench,sf+fused,graph+fused,c-ro+fused
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import train_pro_v5 as T  # noqa: E402
from cctrain.train.v5_fast import StepRunner  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--global-batch", type=int, default=48)
    p.add_argument("--n-batches", type=int, default=4)
    p.add_argument("--off-every", type=int, default=0, help="1 a cada N lotes sem perfil (0 = todos com perfil)")
    p.add_argument("--parity-steps", type=int, default=20)
    p.add_argument("--parity-batch", type=int, default=16,
                   help="paridade num recorte do lote (referência + variante no mesmo passo não cabem em 24 GB com B48 "
                        "e um CUDA Graph); 0 = lote inteiro")
    p.add_argument("--warm", type=int, default=5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--variants", default="ref,ref-nolog,sf")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default=None, help="JSON com os resultados")
    p.add_argument("--deterministic", action="store_true",
                   help="algoritmos determinísticos (cuDNN/cuBLAS) para a paridade sem o ruído de atomics")
    a, rest = p.parse_known_args()
    ta = T.parse_args(["--global-batch", str(a.global_batch), "--gpu-mix"] + rest)
    return a, ta


def load_batches(ta, n: int, off_every: int, device: str):
    paths = T.Paths.from_env()
    vp = T.V4Params(seed=ta.seed, segment_s=ta.segment_s, batch_size=ta.global_batch, total_steps=1000,
                    phase1_frac=0.0, n_probs=T._np(ta.n_probs, 5, T.N_PROBS),
                    pattern_probs=T._np(ta.pattern_probs, 4, T.PATTERN_PROBS), p_interf_first=ta.p_interf_first,
                    p_turns=ta.p_turns, turn_gap_max_s=ta.turn_gap_max_s, p_rir_turns=ta.p_rir_turns,
                    p_target_absent=ta.p_target_absent)
    ds, _ = T.make_dataset(paths, vp)
    ds = T.StageV4RecipeDataset(ds.pools, ds.bank, ds.data_root, vp)
    mixer = T.GpuMixer.from_dataset(ds, device)
    bl = ta.global_batch
    dl = DataLoader(ds, batch_size=bl, sampler=list(range(n * bl)), num_workers=ta.workers or 0,
                    collate_fn=T.collate_recipes, worker_init_fn=T.init_worker, pin_memory=True)
    out = []
    for i, rb in enumerate(dl):
        b = mixer(rb)
        mixer.fill_invalid(b, ds)
        b.pop("valid")
        b.pop("index")
        on = not (off_every and i % off_every == off_every - 1)
        b["mask"] = torch.full_like(b["mask"], 1.0 if on else 0.0)
        out.append(({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}, on))
    return out


def flat_grads(params):
    return torch.cat([(p.grad.float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)).reshape(-1)
                      for p in params]).cpu()


def main():
    a, ta = parse()
    device = "cuda"
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    paths = T.Paths.from_env()
    cfg_f = T.load_stage_f_config()
    hp = T.StageFHParams.from_config(cfg_f, run_id="bench", num_workers=0, batch_size=ta.global_batch)
    clip = ta.grad_clip if ta.grad_clip is not None else hp.grad_clip
    model, acfg, sites, base_kind, train_cond, _ = T.build_v5(ta, device, hp, paths)
    trainable = T.prepare(model, train_cond)
    amp = ta.amp == "bf16"
    if amp:
        T.fp32_spectral(model)
    istft = T.Istft(960, T.HOP, torch.as_tensor(T._fft_window())).to(device)
    params = T.ModelParams()
    lr = ta.lr * math.sqrt(ta.global_batch / T.REF_BATCH)
    init = [p.detach().clone() for p in trainable]
    print(f"treináveis {sum(p.numel() for p in trainable)} | lotes {a.n_batches}×{ta.global_batch} | torch "
          f"{torch.__version__} | {torch.cuda.get_device_name()}", flush=True)
    t0 = time.time()
    batches = load_batches(ta, a.n_batches, a.off_every, device)
    print(f"lotes prontos em {time.time() - t0:.0f}s (com perfil: {[on for _, on in batches]})", flush=True)

    def ref_grads(b):
        """Referência no estado atual: step_v5 eager, AdamW padrão, cudnn.benchmark off, layout contíguo."""
        for p in trainable:
            p.grad = None
        parts = T.step_v5_ref(model, istft, params, hp, b, ta, amp)
        parts["total"].backward()
        return ({k: float(v.detach()) for k, v in parts.items()},
                [None if p.grad is None else p.grad.detach().clone() for p in trainable])

    def flat(gs):
        return torch.cat([(g if g is not None else torch.zeros_like(p)).float().reshape(-1)
                          for g, p in zip(gs, trainable)])

    def run_variant(name: str) -> dict:
        opts = set(name.split("+"))
        base = next(iter(opts & {"ref", "ref-nolog", "fl", "fl-nolog", "sf", "graph", "c-default", "c-ro", "c-mat"}),
                    "sf")
        import cctrain.losses.fast_v5 as fv5

        fv5.set_compile("lc" in opts)
        with torch.no_grad():
            for p, x in zip(trainable, init):
                p.copy_(x)
        okw = {"fused": True} if "fused" in opts else ({"foreach": True} if "foreach" in opts else {})
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=ta.weight_decay, **okw)
        eager_fn = T.step_v5 if base.startswith("fl") else T.step_v5_ref

        def make_runner():
            if base in ("ref", "ref-nolog", "fl", "fl-nolog"):
                return None
            mode = {"sf": "eager", "graph": "graph"}.get(base, "compile")
            cm = {"c-default": "default", "c-ro": "reduce-overhead", "c-mat": "max-autotune-no-cudagraphs"}.get(base)
            return StepRunner(model, istft, params, hp, ta, amp, trainable, mode=mode, compile_mode=cm,
                              loss_fn=T.step_v5 if "fl" in opts else None)

        runner = make_runner()
        pb = a.parity_batch

        def cut(b):
            if not pb:
                return b
            return {k: (v[:pb] if torch.is_tensor(v) and v.dim() and v.shape[0] == ta.global_batch else v)
                    for k, v in b.items()}
        per_step_sync = base in ("ref", "fl")

        def setup(on_variant: bool):
            torch.backends.cudnn.benchmark = on_variant and "bench" in opts
            if "cl" in opts:
                model.to(memory_format=torch.channels_last if on_variant else torch.contiguous_format)

        def one(b, on):
            if runner is None:
                opt.zero_grad(set_to_none="zg" not in opts)
                parts = eager_fn(model, istft, params, hp, b, ta, amp)
                tot = parts["total"]
                if per_step_sync:
                    if torch.isfinite(tot).float().item() < 1:
                        return parts
                tot.backward()
            else:
                parts = runner(b, on)
            return parts

        def finish(parts):
            torch.nn.utils.clip_grad_norm_(trainable, clip)
            opt.step()
            if per_step_sync:
                torch.cuda.synchronize()
                _ = {k: float(v.detach()) for k, v in parts.items()}

        # paridade em passo travado: a cada passo, referência e variante no MESMO estado; o estado avança com os
        # gradientes da referência (o otimizador da variante: AdamW padrão × o da variante, comparados à parte)
        res = {"variant": name}
        lrel, grel = [], []
        opt_ref = torch.optim.AdamW([torch.nn.Parameter(p.detach().clone()) for p in trainable], lr=lr,
                                    weight_decay=ta.weight_decay)
        shadow = opt_ref.param_groups[0]["params"]
        prel = []
        part_rel: dict = {}
        tc = time.time()
        for i in range(a.parity_steps):
            b, on = batches[i % len(batches)]
            b = cut(b)
            setup(False)
            lr_, gr_ = ref_grads(b)
            setup(True)
            parts = one(b, on)
            pv = {k: float(v.detach()) for k, v in parts.items()}
            gv = [None if p.grad is None else p.grad.detach() for p in trainable]
            fr, fv = flat(gr_), flat(gv)
            lrel.append(abs(pv["total"] - lr_["total"]) / max(abs(lr_["total"]), 1e-12))
            for k in lr_:
                if k in pv and math.isfinite(lr_[k]) and math.isfinite(pv[k]):
                    d = abs(pv[k] - lr_[k]) / max(abs(lr_[k]), 1e-6)
                    part_rel[k] = max(part_rel.get(k, 0.0), d)
                elif k in pv and math.isnan(lr_[k]) != math.isnan(pv[k]):
                    part_rel[k] = float("inf")
            grel.append(float((fv - fr).norm() / fr.norm().clamp_min(1e-30)))
            for p, g, sp in zip(trainable, gr_, shadow):  # avança com os gradientes da referência
                p.grad = None if g is None else g.clone()
                sp.grad = None if g is None else g.clone()
            torch.nn.utils.clip_grad_norm_(trainable, clip)
            torch.nn.utils.clip_grad_norm_(shadow, clip)
            opt.step()
            opt_ref.step()
            num = sum(float((p.detach() - sp.detach()).float().norm() ** 2) for p, sp in zip(trainable, shadow))
            den = sum(float((sp.detach() - x).float().norm() ** 2) for sp, x in zip(shadow, init))
            prel.append(math.sqrt(num / max(den, 1e-30)))
        res["first_steps_s"] = time.time() - tc  # inclui compilação/captura
        res.update({"loss_rel_max": max(lrel), "grad_rel_max": max(grel), "grad_rel_med": statistics.median(grel),
                    "upd_rel_max": max(prel), "part_rel_max": {k: float(f"{v:.1e}") for k, v in part_rel.items()}, "loss_rel": [float(f"{x:.2e}") for x in lrel],
                    "grad_rel": [float(f"{x:.2e}") for x in grel]})
        del opt_ref, shadow, parts, gv, gr_
        for p in trainable:
            p.grad = None
        if base.startswith("c-") or "lc" in opts:
            torch._dynamo.reset()
            fv5.set_compile("lc" in opts)
        runner = None
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        runner = make_runner()
        # tempo: a partir do estado inicial, trajetória própria
        setup(True)
        with torch.no_grad():
            for p, x in zip(trainable, init):
                p.copy_(x)
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=ta.weight_decay, **okw)
        for i in range(a.warm):
            b, on = batches[i % len(batches)]
            finish(one(b, on))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        for i in range(a.steps):
            b, on = batches[i % len(batches)]
            finish(one(b, on))
        torch.cuda.synchronize()
        res["s_per_step"] = (time.perf_counter() - t1) / a.steps
        res["vram_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        setup(False)
        if base.startswith("c-") or "lc" in opts:
            torch._dynamo.reset()
        fv5.set_compile(False)
        del runner, opt
        torch.cuda.empty_cache()
        return res

    results = []
    ref_s = None
    for v in ["ref"] + [x for x in a.variants.split(",") if x != "ref"]:
        try:
            r = run_variant(v)
            ref_s = ref_s or r["s_per_step"]
            r["gain"] = ref_s / r["s_per_step"]
            print(f"[{v}] {r['s_per_step']:.4f} s/passo (×{r['gain']:.2f}) vram {r['vram_gb']:.1f} GiB | "
                  f"{a.parity_steps} passos travados {r['first_steps_s']:.0f}s | perda rel máx {r['loss_rel_max']:.1e} | "
                  f"grad rel máx {r['grad_rel_max']:.1e} med {r['grad_rel_med']:.1e} | atualização rel máx "
                  f"{r['upd_rel_max']:.1e}\n    partes {r['part_rel_max']}", flush=True)
        except Exception as e:  # noqa: BLE001
            r = {"variant": v, "error": f"{type(e).__name__}: {str(e)[:600]}"}
            print(f"[{v}] FALHOU: {r['error']}", flush=True)
            traceback.print_exc(limit=8)
            torch._dynamo.reset()
            torch.cuda.empty_cache()
        results.append(r)
        if a.out:
            Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
