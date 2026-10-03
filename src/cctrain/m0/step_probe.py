"""Passo de treino do pDFNet3 (forward + backward) para medir s/passo e VRAM (spec §5.5).

A perda é um proxy de custo da multi-resolução do upstream (FFT 256/512/1024/2048, gamma 0,3, fatores 500/500)
sobre uma iSTFT com janela Vorbis; a perda real (paridade com df/loss.py) é do M2.
"""

from __future__ import annotations

import math
import statistics
import time

import torch
from torch import Tensor, nn

from cctrain.m0.loader_probe import total_rss

GIB = 1024**3


def vorbis_window(n: int = 960) -> Tensor:
    k = torch.arange(n, dtype=torch.float64)
    return torch.sin(0.5 * math.pi * torch.sin(math.pi * (k + 0.5) / n) ** 2).float()


def istft_proxy(spec: Tensor) -> Tensor:
    c = torch.view_as_complex(spec[:, 0].permute(0, 2, 1, 3).contiguous())  # [B, F, T]
    return torch.istft(c, n_fft=960, hop_length=480, window=vorbis_window().to(spec.device), center=True)


class MultiResSpecLossProxy(nn.Module):
    def __init__(self, n_ffts=(256, 512, 1024, 2048), gamma: float = 0.3, factor: float = 500.0,
                 factor_complex: float = 500.0):
        super().__init__()
        self.n_ffts, self.gamma, self.factor, self.factor_complex = n_ffts, gamma, factor, factor_complex

    def forward(self, est: Tensor, ref: Tensor) -> Tensor:
        loss = est.new_zeros(())
        for n in self.n_ffts:
            w = torch.hann_window(n, device=est.device)
            y = torch.stft(est, n, n // 4, window=w, return_complex=True)
            s = torch.stft(ref, n, n // 4, window=w, return_complex=True)
            ya, sa = y.abs().clamp_min(1e-12).pow(self.gamma), s.abs().clamp_min(1e-12).pow(self.gamma)
            loss = loss + self.factor * nn.functional.mse_loss(ya, sa)
            yc = torch.view_as_real(y / y.abs().clamp_min(1e-12) * ya)
            sc = torch.view_as_real(s / s.abs().clamp_min(1e-12) * sa)
            loss = loss + self.factor_complex * nn.functional.mse_loss(yc, sc)
        return loss


def probe_step(model: nn.Module, batch: dict, *, device: str, teacher: nn.Module | None = None,
               warmup: int = 5, steps: int = 20) -> dict:
    model.to(device).train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    loss_fn = MultiResSpecLossProxy().to(device)
    spec, erb, specf, clean = (torch.as_tensor(batch[k]).to(device) for k in ("spec", "feat_erb", "feat_spec",
                                                                              "spec_clean"))
    b = spec.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    emb = nn.functional.normalize(torch.randn(b, 192, generator=g), dim=-1).to(device)
    mask = (torch.rand(b, 1, generator=g) > 0.2).float().to(device)
    ref = istft_proxy(clean)
    torch.cuda.reset_peak_memory_stats(device)
    times, finite = [], True
    for i in range(warmup + steps):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        spec_e = model(spec.clone(), erb, specf, emb, mask)[0]
        est = istft_proxy(spec_e)
        loss = loss_fn(est, ref)
        if teacher is not None:
            with torch.no_grad():
                t_spec = teacher(spec.clone(), erb, specf)[0]
            loss = loss + loss_fn(est, istft_proxy(t_spec))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize(device)
        finite = finite and bool(torch.isfinite(loss))
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return {
        "batch": b, "teacher": teacher is not None, "steps": steps,
        "s_per_step_median": statistics.median(times), "s_per_step_max": max(times),
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / GIB,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / GIB,
        "rss_gb": total_rss() / GIB, "loss_finite": finite,
    }
