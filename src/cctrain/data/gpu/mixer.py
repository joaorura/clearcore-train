"""Mistura do mix_v4 em lote na GPU, a partir das receitas de `recipe.StageV4RecipeDataset` / `collate_recipes`.

Reproduz a álgebra de `StageV4Dataset._build` (já com a correção do ganho seco, 0145934) em torch, float64 nos sinais
como no CPU (a convolução de RIR é float32, como o `_conv`; features/espectros float32, como o libdf):
envelope de turno -> RIR molhada/early -> SIR (ou ±6 dB com alvo ausente) -> ruído com SNR (passa-baixa da banda da
fala) -> nível absoluto -> piso de microfone -> limitador de pico -> rótulos por quadro e features do modelo.

Trilhas empacotadas: `dry[M, n]` com `track_b[M]` (amostra) e `track_k[M]` (0 = alvo, 1..4 = interferente), só as
trilhas presentes; somas por amostra com `index_add_`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from cctrain.data.gpu import dsp, features, labels
from cctrain.data.mix_v4 import HOP, PEAK_LIMIT

NEG = -200.0


def _db(p: Tensor) -> Tensor:
    return 10.0 * torch.log10(p + 1e-20)


def _scatter_or(mask: Tensor, idx: Tensor, B: int) -> Tensor:
    out = torch.zeros(B, mask.shape[1], dtype=torch.int32, device=mask.device)
    out.index_add_(0, idx, mask.to(torch.int32))
    return out > 0


class GpuMixer:
    def __init__(self, rirs: list[np.ndarray], early_lens: list[int], device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float64, lp_dtype: torch.dtype | None = None,
                 label_dtype: torch.dtype | None = None):
        """`dtype`: sinais da mistura (float64 = paridade com o CPU; float32 é mais rápido, ~1e-6 relativo).
        `lp_dtype`/`label_dtype`: dtype do passa-baixa IIR do ruído e dos rótulos/níveis por quadro (padrão: `dtype`)."""
        self.device = torch.device(device)
        self.dtype = dtype
        self.lp_dtype = lp_dtype or dtype
        self.label_dtype = label_dtype or dtype
        L = max((len(h) for h in rirs), default=1)
        H = np.zeros((max(len(rirs), 1), L), np.float32)
        for j, h in enumerate(rirs):
            H[j, : len(h)] = h
        self.H = torch.from_numpy(H).to(self.device)
        self.hlen = torch.as_tensor([len(h) for h in rirs] or [1], dtype=torch.long, device=self.device)
        self.elen = torch.as_tensor(list(early_lens) or [1], dtype=torch.long, device=self.device)

    @classmethod
    def from_dataset(cls, ds, device="cuda", **kw) -> GpuMixer:
        """RIRs (48 kHz f32) e comprimentos "early" exatamente como o CPU (`early_rir`: pico + 50 ms)."""
        cache = ds._rirs()
        hs = [cache[rel][0] for rel in ds.pools.rirs]
        return cls(hs, [len(cache[rel][1]) for rel in ds.pools.rirs], device, **kw)

    # entradas de áudio do lote (f32 na GPU); `PoolMixer` as monta a partir do pool residente
    def _dry(self, rb: dict) -> Tensor:
        dev = self.device
        if "dry_packed" in rb:  # collate_recipes(sparse=True): reexpande os quadros enviados, o resto é zero
            M_, n_ = (int(v) for v in rb["dry_shape"])
            dry = torch.zeros(M_, n_ // HOP, HOP, dtype=torch.float32, device=dev)
            dry[rb["dry_m"].to(dev, non_blocking=True), rb["dry_f"].to(dev, non_blocking=True)] = \
                rb["dry_packed"].to(dev, non_blocking=True)
            return dry.reshape(M_, n_)
        return rb["dry"].to(dev, non_blocking=True)

    def _noise(self, rb: dict, n: int) -> Tensor:
        return rb["noise"].to(self.device, non_blocking=True).float()

    def _floor_white(self, rb: dict, n: int) -> Tensor:
        return rb["floor_white"].to(self.device, non_blocking=True)

    @torch.no_grad()
    def __call__(self, rb: dict, debug: bool = False) -> dict:
        dev, dt, ldt = self.device, self.dtype, self.label_dtype
        g_ = lambda k: rb[k].to(dev, non_blocking=True)  # noqa: E731
        dry = self._dry(rb)
        M, n = dry.shape
        F = n // HOP
        B = rb["index"].shape[0]
        tb, tk = g_("track_b"), g_("track_k")
        is_t, is_i = tk == 0, tk > 0
        present = g_("has_target")
        pres_t = present[tb]

        # envelope de turno e RIR (molhada / caminho direto + 50 ms)
        x = dry.to(dt) * dsp.ramp_env(g_("on")[tb, tk], n, dtype=dt)
        wet, early = x.clone(), x.clone()
        rv = g_("reverb")[tb]
        if bool(rv.any()):
            sel = rv.nonzero().squeeze(1)
            r = g_("rir_idx")[tb[sel], tk[sel]]
            h = self.H[r]
            wet[sel] = dsp.fft_conv(x[sel], h, self.hlen[r])
            early[sel] = dsp.fft_conv(x[sel], h, self.elen[r])

        # SIR contra o nível ativo do alvo; alvo ausente: interferentes 2.. a ±6 dB do primeiro
        lv = labels.active_rms_db(wet, F, ldt).double()  # ganhos/níveis por trilha sempre em f64 (vetores pequenos)
        t_db = torch.full((B,), NEG, dtype=torch.float64, device=dev)
        t_db[tb[is_t]] = lv[is_t]
        r_db = torch.full((B,), NEG, dtype=torch.float64, device=dev)
        r_db[tb[tk == 1]] = lv[tk == 1]
        ki = (tk - 1).clamp(min=0)
        sir = g_("sirs")[tb, ki]
        rel = g_("rels")[tb, ki]
        g_pres = torch.pow(10.0, (t_db[tb] - sir - lv) / 20.0)
        g_abs = torch.where(tk >= 2, torch.pow(10.0, (r_db[tb] + rel - lv) / 20.0), torch.ones_like(lv))
        g_tr = torch.where(is_i, torch.where(pres_t, g_pres, g_abs), torch.ones_like(lv))
        wet *= g_tr[:, None].to(dt)
        early *= g_tr[:, None].to(dt)
        bad = torch.zeros(B, dtype=torch.int32, device=dev)
        bad.index_add_(0, tb, (lv < -150).to(torch.int32))

        speech = torch.zeros(B, n, dtype=dt, device=dev).index_add_(0, tb, wet)
        interf_sum = torch.zeros(B, n, dtype=dt, device=dev).index_add_(0, tb[is_i], wet[is_i])
        tgt_wet = torch.zeros(B, n, dtype=dt, device=dev)
        tgt_wet[tb[is_t]] = wet[is_t]
        ref_db = labels.active_rms_db(torch.where(present[:, None], tgt_wet, interf_sum), F, ldt).double()
        valid = (bad == 0) & (ref_db >= -150)

        # ruído: SNR contra a referência (alvo, ou interferentes com alvo ausente)
        val = g_("noise_val")
        noise = torch.zeros(B, n, dtype=dt, device=dev)
        nb = g_("noise_b")
        if nb.numel():
            nz = dsp.lowpass48(self._noise(rb, n), g_("lp_cut")[nb], dtype=self.lp_dtype).to(dt)
            nz_db = _db(torch.mean(nz.double() ** 2, dim=1)) if dt != torch.float64 else _db(torch.mean(nz**2, dim=1))
            noise[nb] = nz * torch.pow(10.0, (ref_db[nb] - val[nb] - nz_db) / 20.0)[:, None].to(dt)
        g = torch.pow(10.0, (g_("level_db") - ref_db) / 20.0)
        noisy = (speech + noise) * g[:, None].to(dt)
        early *= g[tb][:, None].to(dt)
        floor = torch.zeros(B, n, dtype=dt, device=dev) if debug else None
        fb = g_("floor_white_b")
        if fb.numel():
            w = self._floor_white(rb, n).to(dt)
            pk_ = g_("floor_pink")[fb]
            if bool(pk_.any()):
                w[pk_] = dsp.pink_from_white(w[pk_])
            fl = w * torch.pow(10.0, val[fb] / 20.0)[:, None].to(dt)
            noisy[fb] += fl
            if debug:
                floor[fb] = fl
        pk = noisy.abs().amax(dim=1).double()
        lim = torch.where(pk > 0, torch.clamp(PEAK_LIMIT / pk.clamp(min=1e-300), max=1.0), torch.ones_like(pk))
        noisy *= lim[:, None].to(dt)
        early *= lim[tb][:, None].to(dt)
        tgt = torch.zeros(B, n, dtype=dt, device=dev)
        tgt[tb[is_t]] = early[is_t]
        int_ref = torch.zeros(B, n, dtype=dt, device=dev).index_add_(0, tb[is_i], early[is_i])
        mask0 = g_("mask0")
        noisy32 = noisy.float()

        spec, feat_erb, feat_spec = features.extract_features(noisy32)
        spec_clean = features.analysis(torch.where(mask0[:, None], tgt + int_ref, tgt).float())
        spec_interf = features.analysis(int_ref.float())

        t_act = labels.rel_activity(tgt, F, dtype=ldt) & present[:, None]
        i_act = _scatter_or(labels.rel_activity(early[is_i], F, dtype=ldt), tb[is_i], B)
        wet_i = wet[is_i] * (g * lim)[tb[is_i]][:, None].to(dt)
        i_act45 = _scatter_or(labels.rel_activity(wet_i, F, range_db=45.0, dtype=ldt), tb[is_i], B)
        dsil = labels.digital_silence(noisy32, F, ldt)

        s = g_("sirs").clone()
        s[torch.isnan(s)] = float("inf")
        sir_min = s.min(dim=1).values
        sir_min[torch.isinf(sir_min)] = float("nan")
        out = {
            "spec": spec, "feat_erb": feat_erb, "feat_spec": feat_spec, "spec_clean": spec_clean,
            "spec_interf": spec_interf, "embedding": g_("embedding"),
            "mask": torch.where(mask0, 0.0, 1.0).float()[:, None],
            "target_active": t_act, "interf_active": i_act, "interf_active_wet45": i_act45, "digital_silence": dsil,
            "n_interf": g_("n_interf"), "has_target": present, "max_freq": g_("max_freq"),
            "level_db": g_("level_db").float(), "sir_min_db": sir_min.float(), "noise_silent": g_("noise_silent"),
            "noise_val": val.float(), "reverb": g_("reverb"), "pattern": g_("pattern"), "phase": g_("phase"),
            "target_key": list(rb["target_key"]), "target_path": list(rb["target_path"]),
            "valid": valid, "index": g_("index"),
        }
        if debug:
            wsum = torch.zeros(B, n, dtype=dt, device=dev).index_add_(0, tb[is_i], wet_i)
            out["debug"] = {"noisy": noisy, "tgt": tgt, "int_ref": int_ref, "interf_wet": wsum,
                            "noise": (noise * g[:, None] + floor) * lim[:, None], "ref_db": ref_db, "lim": lim,
                            "g": g, "track_lv": lv, "track_g": g_tr}
        return out

    def fill_invalid(self, out: dict, ds) -> int:
        """Amostras com `valid=False` (trilha presente abaixo de −150 dB que a receita não pegou) são refeitas pelo
        caminho CPU, que sorteia de novo com a tentativa seguinte, como no `StageV4Dataset`."""
        bad = (~out["valid"]).nonzero().squeeze(1).tolist()
        for b in bad:
            smp = ds.cpu_sample(int(out["index"][b]))
            for k, v in smp.items():
                if isinstance(out.get(k), list):
                    out[k][b] = v
                elif k in out:
                    out[k][b] = torch.as_tensor(np.asarray(v)).to(out[k].device, out[k].dtype)
            out["valid"][b] = True
        return len(bad)


class PoolMixer(GpuMixer):
    """`GpuMixer` alimentado pelo pool residente (`cctrain.data.gpu.pool`): o lote traz só referências; a GPU recorta,
    converte (int16 -> f32) e aplica o sussurro; o resto da mistura é o mesmo código."""

    def __init__(self, *a, pool, **kw):
        super().__init__(*a, **kw)
        self.pool = pool

    @classmethod
    def from_dataset(cls, ds, device="cuda", pool=None, **kw) -> PoolMixer:
        cache = ds._rirs()
        hs = [cache[rel][0] for rel in ds.pools.rirs]
        return cls(hs, [len(cache[rel][1]) for rel in ds.pools.rirs], device, pool=pool, **kw)

    def _dry(self, rb: dict) -> Tensor:
        ref = rb["dry_ref"]
        dry = self.pool.gather(ref, int(rb["n_samples"]), wrap=False, gid_col=5)
        tilt = (ref[:, 4] == 1).nonzero().squeeze(1)
        if tilt.numel():
            td = tilt.to(self.device)
            dry[td] = dsp.whisper_tilt(dry[td].double()).float()
        return dry

    def _noise(self, rb: dict, n: int) -> Tensor:
        return self.pool.gather(rb["noise_ref"], n, wrap=True, gid_col=4)

    def _floor_white(self, rb: dict, n: int) -> Tensor:
        return self.pool.floor_white(rb["floor_seed"], n)

    @torch.no_grad()
    def __call__(self, rb: dict, debug: bool = False, count: bool = True) -> dict:
        out = super().__call__(rb, debug)
        out["attempt"] = rb["attempt"]
        if count:
            self.pool.note_batch()
        return out

    def fill_invalid(self, out: dict, ds) -> int:
        """`valid=False`: refaz a receita com a tentativa seguinte (como o `_build`), ainda do pool, e mistura na GPU."""
        from cctrain.data.gpu.recipe import collate_recipes

        bad = (~out["valid"]).nonzero().squeeze(1).tolist()
        B = len(out["valid"])
        todo = [(b, int(out["index"][b]), int(out["attempt"][b]) + 1) for b in bad]
        while todo:  # todas as refações do lote numa mistura só (por rodada)
            sub = self(collate_recipes([ds.recipe_from(i, att) for _, i, att in todo]), count=False)
            nxt = []
            for j, (b, i, _) in enumerate(todo):
                if not bool(sub["valid"][j]):
                    nxt.append((b, i, int(sub["attempt"][j]) + 1))
                    continue
                for k, v in sub.items():
                    if k in ("debug", "valid", "attempt"):
                        continue
                    if isinstance(v, list):
                        out[k][b] = v[j]
                    elif torch.is_tensor(v) and v.dim() and out[k].shape[0] == B:
                        out[k][b] = v[j].to(out[k].device, out[k].dtype)
                out["valid"][b] = True
            todo = nxt
        out.pop("attempt", None)
        return len(bad)
