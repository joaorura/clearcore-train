"""Rótulos/VAD do gerador `mix_v4` em lote (torch, qualquer device), com paridade exata com a versão CPU.

Contrato: trilhas `[M, n]`, `F = n // HOP` quadros, float64. Sem loop por amostra. As definições seguem
`cctrain.data.mix_v4` ao pé da letra:

* `frame_levels`: 10*log10(mean(x_quadro**2) + 1e-20), sobre os primeiros `F * HOP` amostras (dBFS de potência).
* `rel_activity`: o máximo é o **máximo global da trilha** (não há janela local); ativo se nível >=
  max(max_nivel - range_db, floor_dbfs), e tudo inativo quando max_nivel <= floor_dbfs.
* `active_rms_db`: RMS em dB (piso -200) sobre as amostras dos quadros ativos de `rel_activity(floor=-200)`.
* `digital_silence`: média por quadro (float64) de `noisy32**2` < SILENCE_MS.

Só os rótulos: mistura e ganhos ficam fora deste módulo.
"""

from __future__ import annotations

import torch

HOP = 480  # = cctrain.data.features.HOP / mix_v4.HOP (replicado para não importar o extrator)
SILENCE_MS = 1e-7  # = mix_v4.SILENCE_MS
_EPS = 1e-20


def _frames(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """`[M, n]` -> `[M, F, HOP]` em float64 (como `x[: F*HOP].reshape(F, HOP)` do CPU)."""
    return x[:, : n_frames * HOP].to(torch.float64).reshape(x.shape[0], n_frames, HOP)


def _frame_ms(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    return torch.mean(_frames(x, n_frames) ** 2, dim=2)


def frame_levels(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Nível por quadro em dB (potência média + 1e-20): `f64[M, F]`."""
    return 10 * torch.log10(_frame_ms(x, n_frames) + _EPS)


def _active_from_levels(lv: torch.Tensor, range_db: float, floor_dbfs: float) -> torch.Tensor:
    mx = lv.amax(dim=1, keepdim=True)
    thr = torch.clamp(mx - range_db, min=floor_dbfs)  # max(mx - range_db, floor_dbfs)
    return (lv >= thr) & (mx > floor_dbfs)


def rel_activity(x: torch.Tensor, n_frames: int, range_db: float = 30.0, floor_dbfs: float = -75.0) -> torch.Tensor:
    """VAD relativo por trilha: `bool[M, F]`. Funciona igual com `range_db=45.0` (interf_active_wet45)."""
    return _active_from_levels(frame_levels(x, n_frames), range_db, floor_dbfs)


def active_rms_db(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """RMS (dB) nos quadros ativos (`floor=-200`); -200 se nenhum quadro ativo: `f64[M]`."""
    ms = _frame_ms(x, n_frames)
    lv = 10 * torch.log10(ms + _EPS)
    act = _active_from_levels(lv, 30.0, -200.0)
    cnt = act.sum(dim=1)
    mean = (ms * act).sum(dim=1) / cnt.clamp(min=1)
    out = 10 * torch.log10(mean + _EPS)
    return torch.where(cnt > 0, out, torch.full_like(out, -200.0))


def digital_silence(noisy32: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Quadros de silêncio digital: média por quadro de `noisy32**2` em float64 < 1e-7: `bool[B, F]`."""
    return _frame_ms(noisy32, n_frames) < SILENCE_MS
