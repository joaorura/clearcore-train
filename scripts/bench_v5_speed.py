#!/usr/bin/env python3
"""Bench de velocidade do pro-v5 (P3) por componente, numa GPU. Tempos com CUDA events (mediana de N iterações após
aquecimento); VRAM = pico do alocador.

  gru   GRUs do DFNet3 isoladas (dimensões lidas do modelo: enc.emb_gru, erb_dec.emb_gru, df_dec.df_gru), forward +
        backward, T quadros × batch B, em fp32 / autocast bf16 / autocast fp16 / bf16 nativo (pesos bf16, sem autocast),
        com e sem LoRA (LoRAGRU, perfil ligado em todas as amostras, e lote metade com/metade sem perfil). Lista os
        kernels CUDA da GRU (profiler) e o dtype da saída: mostra se o cuDNN roda e em que precisão.
  step  passo de treino inteiro com dados reais (--gpu-mix): mistura de lotes reais na GPU (f64 e f32), depois o passo
        sobre lotes já misturados (cacheados), dividido em fases: forward do modelo | perdas (espectral fp32, iSTFT,
        perdas) | backward | clip+AdamW. Precisões: --amp none|bf16|fp16 (fp16 com GradScaler), --gru-fp32.

Uso (rig): srun --gres=gpu:1 -c 4 --mem=32G -t 60 env/bin/python scripts/bench_v5_speed.py step --amp bf16
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import train_pro_v5 as T  # noqa: E402

GIB = 1024**3


def _ev():
    return torch.cuda.Event(enable_timing=True)


def _kernels(fn, filt=None, top=12) -> list[tuple[str, float, int]]:
    """Kernels CUDA de uma chamada (nome, ms total, nº de chamadas), ordenados por tempo."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        fn()
        torch.cuda.synchronize()
    agg: dict[str, list[float]] = {}
    for e in p.events():
        if e.device_type.name != "CUDA":
            continue
        if filt and not filt(e.name):
            continue
        a = agg.setdefault(e.name, [0.0, 0])
        a[0] += e.device_time_total / 1e3 if hasattr(e, "device_time_total") else e.cuda_time_total / 1e3
        a[1] += 1
    rows = sorted(((k, v[0], v[1]) for k, v in agg.items()), key=lambda r: -r[1])
    return rows[:top]


def _time(fn, warm=3, n=10) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = _ev(), _ev()
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / 1e3)
    return statistics.median(ts)


# --- GRU isolada ----------------------------------------------------------------------------------------------------


def gru_bench(a) -> dict:
    from cctrain.models.adapters import GRU_SITES, AdapterGate, LoRAGRU, _get

    dev = "cuda"
    args = T.parse_args(["--variant", "P3"])
    hp = T.StageFHParams.from_config(T.load_stage_f_config(), run_id="bench", num_workers=0, batch_size=a.batch)
    model = T.build_model(dev, hp, presence=False)
    out = {"B": a.batch, "T": a.frames, "cudnn": torch.backends.cudnn.version(), "cudnn_enabled":
           torch.backends.cudnn.enabled, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "sites": {}}
    for site in GRU_SITES:
        base_gru = _get(model.base, site)
        dims = (base_gru.input_size, base_gru.hidden_size, base_gru.num_layers)
        res = {"dims(in,hidden,layers)": dims}
        for mode in ("fp32", "bf16-autocast", "fp16-autocast", "bf16-native", "fp16-native"):
            for lora in ("none", "lora-all", "lora-half"):
                gru = torch.nn.GRU(*dims[:2], num_layers=dims[2], batch_first=True).to(dev)
                gru.load_state_dict(base_gru.state_dict())
                for p in gru.parameters():
                    p.requires_grad_(False)  # base congelada, como no treino
                gate = AdapterGate()
                mod = gru if lora == "none" else LoRAGRU(gru, 8, 1.0, gate).to(dev)
                if lora != "none":
                    with torch.no_grad():
                        for p in mod.lora.values():
                            p.normal_(0, 0.02)
                    g = torch.ones(a.batch, device=dev)
                    if lora == "lora-half":
                        g[: a.batch // 2] = 0
                    gate.g = g
                mod.train()
                native = mode.endswith("native")
                dt = {"fp32": torch.float32, "bf16-autocast": torch.bfloat16, "fp16-autocast": torch.float16,
                      "bf16-native": torch.bfloat16, "fp16-native": torch.float16}[mode]
                if native:
                    mod.to(dt)
                x = torch.randn(a.batch, a.frames, dims[0], device=dev, dtype=dt if native else torch.float32,
                                requires_grad=True)
                ctx = torch.autocast("cuda", dtype=dt) if mode.endswith("autocast") else _null()

                def run():
                    with ctx:
                        y, _ = mod(x)
                    y.float().square().mean().backward()

                try:
                    torch.cuda.reset_peak_memory_stats()
                    t = _time(run, warm=3, n=a.iters)
                    with ctx:
                        ydt = str(mod(x)[0].dtype)
                    ks = _kernels(run, top=6)
                    res[f"{mode}/{lora}"] = {"ms": round(t * 1e3, 2), "out_dtype": ydt,
                                             "vram_gb": round(torch.cuda.max_memory_allocated() / GIB, 2),
                                             "kernels": [(k[:70], round(ms, 2), c) for k, ms, c in ks]}
                except Exception as e:  # noqa: BLE001
                    res[f"{mode}/{lora}"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(site, mode, lora, json.dumps(res[f"{mode}/{lora}"])[:300], flush=True)
        out["sites"][site] = res
    return out


class _Split:
    """Fachada do modelo para o bench: devolve as saídas como folhas destacadas (o backward das perdas para nelas) e
    guarda as saídas reais para o backward do modelo em separado."""

    def __init__(self, model):
        self.model, self.base = model, model.base

    def __bool__(self):
        return True

    def __call__(self, *args, **kw):
        self.outs = [o for o in self.model(*args, **kw)]
        self.leaves = [o.detach().requires_grad_(o.requires_grad) for o in self.outs]
        return tuple(self.leaves)

    def backward(self):
        pairs = [(o, lf.grad) for o, lf in zip(self.outs, self.leaves) if o.requires_grad and lf.grad is not None]
        torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --- passo inteiro --------------------------------------------------------------------------------------------------


def load_batches(a, device: str, args=None):
    """`a.nbatch` lotes de receita reais (DataLoader com workers), na memória pinned do host."""
    from torch.utils.data import DataLoader

    from cctrain.data.gpu.recipe import StageV4RecipeDataset, collate_recipes
    from cctrain.data.mix_v4 import V4Params
    from cctrain.data.workers import init_worker
    from cctrain.paths import Paths

    paths = Paths.from_env()
    vp = V4Params(seed=20261114, segment_s=6.0, batch_size=a.batch, total_steps=40_000, phase1_frac=0.0,
                  p_interf_first=0.7, p_turns=0.4, turn_gap_max_s=0.5, p_rir_turns=0.8, p_target_absent=0.20,
                  pattern_probs=(0.25, 0.40, 0.20, 0.15))
    ds0, _ = T.make_dataset(paths, vp)
    ds = StageV4RecipeDataset(ds0.pools, ds0.bank, ds0.data_root, vp)
    if args is not None and args.profile_drop_mode == "batch":
        ds.drop_block = a.batch
    dl = DataLoader(ds, batch_size=a.batch, sampler=list(range(a.batch * a.nbatch)), num_workers=a.workers,
                    pin_memory=True, worker_init_fn=init_worker, collate_fn=collate_recipes,
                    prefetch_factor=2 if a.workers else None)
    t0 = time.perf_counter()
    rbs = list(dl)
    print(f"{len(rbs)} lotes de receita em {time.perf_counter() - t0:.1f}s ({a.workers} workers)", flush=True)
    return ds, rbs


def mix_bench(a, ds, rbs, device) -> tuple[dict, list[dict]]:
    from cctrain.data.gpu.mixer import GpuMixer

    res = {}
    mixed = None
    nbytes = sum(v.numel() * v.element_size() for v in rbs[0].values() if torch.is_tensor(v))
    res["recipe_batch_MB"] = round(nbytes / 1e6, 1)

    def h2d():
        for rb in rbs:
            for v in rb.values():
                if torch.is_tensor(v):
                    v.to(device, non_blocking=True)

    res["h2d_s_per_batch"] = round(_time(h2d, warm=1, n=3) / len(rbs), 4)
    for dt in a.mix_dtypes.split(","):
        kw = {} if dt == "f64" else {"dtype": torch.float32}
        try:
            mixer = GpuMixer.from_dataset(ds, device, **kw)
        except TypeError:
            res[dt] = "GpuMixer sem parâmetro dtype"
            continue
        outs = []

        def mix_all():
            outs.clear()
            for rb in rbs:
                o = mixer(rb)
                mixer.fill_invalid(o, ds)
                outs.append(o)

        t = _time(mix_all, warm=1, n=3) / len(rbs)
        res[dt] = {"s_per_batch": round(t, 4)}
        if a.profile:
            res[dt]["kernels"] = [(k[:60], round(ms / len(rbs), 2), c) for k, ms, c in _kernels(mix_all, top=8)]
        print("mix", dt, json.dumps(res[dt])[:400], flush=True)
        if mixed is None:
            mixed = []
            for o in outs:
                o = dict(o)
                o.pop("valid"); o.pop("index")
                mixed.append(o)
    return res, mixed


def step_bench(a) -> dict:
    device = "cuda"
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    argv = ["--variant", "P3", "--global-batch", str(a.batch), "--amp", a.amp, "--gpu-mix"] + a.extra.split()
    args = T.parse_args(argv)
    hp = T.StageFHParams.from_config(T.load_stage_f_config(), run_id="bench", num_workers=0, batch_size=a.batch)
    paths = T.Paths.from_env()
    model, acfg, sites, *_ = T.build_v5(args, device, hp, paths)
    trainable = T.prepare(model, True)
    amp_dtype = T.amp_dtype_of(args, device)
    T.setup_precision(model, args, amp_dtype)
    istft = T.Istft(960, T.HOP, torch.as_tensor(T._fft_window())).to(device)
    params = T.ModelParams()
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    T.setup_speed(model, args)
    ds, rbs = load_batches(a, device, args)
    out = {"amp": a.amp, "extra": a.extra, "B": a.batch, "gpu": torch.cuda.get_device_name(0)}
    out["profile_frac_per_batch"] = [round(1 - float(rb["mask0"].float().mean()), 3) for rb in rbs]
    out["mix"], mixed = mix_bench(a, ds, rbs, device)
    del rbs
    torch.cuda.empty_cache()
    T.maybe_compile(model, args)

    marks: dict[str, torch.cuda.Event] = {}

    def mark(name):
        e = _ev()
        e.record()
        marks[name] = e

    ph = {k: [] for k in ("fwd", "loss", "bwd", "opt", "total", "bwd_loss", "bwd_model")}
    split = _Split(model) if a.split_bwd else None
    nonfinite = 0
    scales = []
    torch.cuda.reset_peak_memory_stats()
    n_tot = a.warm + a.steps
    wall = []
    for i in range(n_tot):
        b = mixed[i % len(mixed)]
        tw = time.perf_counter()
        marks.clear()
        mark("start")
        opt.zero_grad(set_to_none=True)
        parts = T.step_v5(split if split else model, istft, params, hp, b, args, amp_dtype, mark=mark)
        mark("loss")
        total = parts["total"]
        if split:  # backward das perdas até as saídas do modelo | backward do modelo
            scaler.scale(total).backward()
            mark("bwd_loss")
            split.backward()
        else:
            scaler.scale(total).backward()
        mark("bwd")
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, hp.grad_clip)
        scaler.step(opt)
        scaler.update()
        mark("opt")
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - tw)
        if not torch.isfinite(total):
            nonfinite += 1
        if scaler.is_enabled():
            scales.append(scaler.get_scale())
        if i >= a.warm:
            ph["fwd"].append(marks["start"].elapsed_time(marks["fwd"]) / 1e3)
            ph["loss"].append(marks["fwd"].elapsed_time(marks["loss"]) / 1e3)
            ph["bwd"].append(marks["loss"].elapsed_time(marks["bwd"]) / 1e3)
            if split:
                ph["bwd_loss"].append(marks["loss"].elapsed_time(marks["bwd_loss"]) / 1e3)
                ph["bwd_model"].append(marks["bwd_loss"].elapsed_time(marks["bwd"]) / 1e3)
            ph["opt"].append(marks["bwd"].elapsed_time(marks["opt"]) / 1e3)
            ph["total"].append(marks["start"].elapsed_time(marks["opt"]) / 1e3)
    med = {k: statistics.median(v) for k, v in ph.items() if v}
    out["step"] = {k: round(v, 4) for k, v in med.items()}
    out["step_pct"] = {k: round(100 * med[k] / med["total"], 1) for k in ("fwd", "loss", "bwd", "opt")}
    out["wall_s_per_step"] = round(statistics.median(wall[a.warm:]), 4)
    out["samples_per_s(step only)"] = round(a.batch / med["total"], 1)
    out["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
    out["nonfinite_losses"] = nonfinite
    if scales:
        out["grad_scale"] = {"first": scales[0], "last": scales[-1], "min": min(scales), "max": max(scales),
                             "n_changes": sum(1 for x, y in zip(scales, scales[1:]) if x != y)}
    if a.profile_bwd:  # backward do modelo por operação (autograd), só a parte do modelo
        sp = split or _Split(model)
        b = mixed[0]
        opt.zero_grad(set_to_none=True)
        parts = T.step_v5(sp, istft, params, hp, b, args, amp_dtype)
        scaler.scale(parts["total"]).backward()
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            sp.backward()
            torch.cuda.synchronize()
        rows = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
        out["bwd_model_ops"] = [(e.key[:90], round(e.device_time_total / 1e3, 2), e.count) for e in rows
                                if e.key.startswith("autograd::engine")][: a.top]
        out["bwd_model_kernels"] = [(e.key[:90], round(e.device_time_total / 1e3, 2), e.count) for e in rows
                                    if not e.key.startswith(("autograd", "aten", "cuda"))][: a.top]
    if a.profile:
        def one():
            b = mixed[0]
            opt.zero_grad(set_to_none=True)
            parts = T.step_v5(model, istft, params, hp, b, args, amp_dtype)
            scaler.scale(parts["total"]).backward()

        out["kernels_step"] = [(k[:80], round(ms, 2), c) for k, ms, c in _kernels(one, top=a.top)]
        out["kernels_gru"] = [(k[:80], round(ms, 2), c) for k, ms, c in _kernels(
            one, filt=lambda n: any(s in n.lower() for s in ("rnn", "gru", "lstm", "persist", "recurr")), top=10)]
    print(json.dumps(out, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("gru", "step"))
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--extra", default="", help="flags extras do train_pro_v5 (ex. '--gru-fp32')")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--nbatch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mix-dtypes", default="f64", help="f64[,f32] (f32 só com GpuMixer(dtype=...))")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--split-bwd", action="store_true", help="separa o backward: perdas | modelo")
    ap.add_argument("--profile-bwd", action="store_true", help="profiler do backward do modelo por op do autograd")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = gru_bench(a) if a.what == "gru" else step_bench(a)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
