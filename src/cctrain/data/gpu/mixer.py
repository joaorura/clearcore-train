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
    def __init__(self, rirs: list[np.ndarray], early_lens: list[int], device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        L = max((len(h) for h in rirs), default=1)
        H = np.zeros((max(len(rirs), 1), L), np.float32)
        for j, h in enumerate(rirs):
            H[j, : len(h)] = h
        self.H = torch.from_numpy(H).to(self.device)
        self.hlen = torch.as_tensor([len(h) for h in rirs] or [1], dtype=torch.long, device=self.device)
        self.elen = torch.as_tensor(list(early_lens) or [1], dtype=torch.long, device=self.device)

    @classmethod
    def from_dataset(cls, ds, device="cuda") -> GpuMixer:
        """RIRs (48 kHz f32) e comprimentos "early" exatamente como o CPU (`early_rir`: pico + 50 ms)."""
        cache = ds._rirs()
        hs = [cache[rel][0] for rel in ds.pools.rirs]
        return cls(hs, [len(cache[rel][1]) for rel in ds.pools.rirs], device)

    @torch.no_grad()
    def __call__(self, rb: dict, debug: bool = False) -> dict:
        dev = self.device
        g_ = lambda k: rb[k].to(dev, non_blocking=True)  # noqa: E731
        dry = g_("dry")
        M, n = dry.shape
        F = n // HOP
        B = rb["index"].shape[0]
        tb, tk = g_("track_b"), g_("track_k")
        is_t, is_i = tk == 0, tk > 0
        present = g_("has_target")
        pres_t = present[tb]

        # envelope de turno e RIR (molhada / caminho direto + 50 ms)
        x = dry.double() * dsp.ramp_env(g_("on")[tb, tk], n)
        wet, early = x.clone(), x.clone()
        rv = g_("reverb")[tb]
        if bool(rv.any()):
            sel = rv.nonzero().squeeze(1)
            r = g_("rir_idx")[tb[sel], tk[sel]]
            h = self.H[r]
            wet[sel] = dsp.fft_conv(x[sel], h, self.hlen[r])
            early[sel] = dsp.fft_conv(x[sel], h, self.elen[r])

        # SIR contra o nível ativo do alvo; alvo ausente: interferentes 2.. a ±6 dB do primeiro
        lv = labels.active_rms_db(wet, F)
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
        wet *= g_tr[:, None]
        early *= g_tr[:, None]
        bad = torch.zeros(B, dtype=torch.int32, device=dev)
        bad.index_add_(0, tb, (lv < -150).to(torch.int32))

        speech = torch.zeros(B, n, dtype=torch.float64, device=dev).index_add_(0, tb, wet)
        interf_sum = torch.zeros(B, n, dtype=torch.float64, device=dev).index_add_(0, tb[is_i], wet[is_i])
        tgt_wet = torch.zeros(B, n, dtype=torch.float64, device=dev)
        tgt_wet[tb[is_t]] = wet[is_t]
        ref_db = labels.active_rms_db(torch.where(present[:, None], tgt_wet, interf_sum), F)
        valid = (bad == 0) & (ref_db >= -150)

        # ruído: SNR contra a referência (alvo, ou interferentes com alvo ausente)
        val = g_("noise_val")
        noise = torch.zeros(B, n, dtype=torch.float64, device=dev)
        nb = g_("noise_b")
        if nb.numel():
            nz = dsp.lowpass48(g_("noise").float(), g_("lp_cut")[nb]).double()
            nz_db = _db(torch.mean(nz**2, dim=1))
            noise[nb] = nz * torch.pow(10.0, (ref_db[nb] - val[nb] - nz_db) / 20.0)[:, None]
        g = torch.pow(10.0, (g_("level_db") - ref_db) / 20.0)
        noisy = (speech + noise) * g[:, None]
        early *= g[tb][:, None]
        floor = torch.zeros(B, n, dtype=torch.float64, device=dev) if debug else None
        fb = g_("floor_white_b")
        if fb.numel():
            w = g_("floor_white").double()
            pk_ = g_("floor_pink")[fb]
            if bool(pk_.any()):
                w[pk_] = dsp.pink_from_white(w[pk_])
            fl = w * torch.pow(10.0, val[fb] / 20.0)[:, None]
            noisy[fb] += fl
            if debug:
                floor[fb] = fl
        pk = noisy.abs().amax(dim=1)
        lim = torch.where(pk > 0, torch.clamp(PEAK_LIMIT / pk.clamp(min=1e-300), max=1.0), torch.ones_like(pk))
        noisy *= lim[:, None]
        early *= lim[tb][:, None]
        tgt = torch.zeros(B, n, dtype=torch.float64, device=dev)
        tgt[tb[is_t]] = early[is_t]
        int_ref = torch.zeros(B, n, dtype=torch.float64, device=dev).index_add_(0, tb[is_i], early[is_i])
        mask0 = g_("mask0")
        noisy32 = noisy.float()

        spec, feat_erb, feat_spec = features.extract_features(noisy32)
        spec_clean = features.analysis(torch.where(mask0[:, None], tgt + int_ref, tgt).float())
        spec_interf = features.analysis(int_ref.float())

        t_act = labels.rel_activity(tgt, F) & present[:, None]
        i_act = _scatter_or(labels.rel_activity(early[is_i], F), tb[is_i], B)
        wet_i = wet[is_i] * (g * lim)[tb[is_i]][:, None]
        i_act45 = _scatter_or(labels.rel_activity(wet_i, F, range_db=45.0), tb[is_i], B)
        dsil = labels.digital_silence(noisy32, F)

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
            wsum = torch.zeros(B, n, dtype=torch.float64, device=dev).index_add_(0, tb[is_i], wet_i)
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
