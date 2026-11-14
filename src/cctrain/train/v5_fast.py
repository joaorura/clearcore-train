"""Passo do pro-v5 sem sincronização do host (para CUDA Graphs / torch.compile reduce-overhead) e o executor do passo.

`step_v5_sf` calcula exatamente as mesmas perdas de `train_pro_v5.step_v5`, trocando só o que lia a GPU no host:

- `leak_loss`/`ovl_loss`/`local_snr_loss`/`turn_loss`: `if float(n) == 0` vira um denominador seguro (com n = 0 o
  numerador já é 0 e a perda é 0, com gradiente 0, como no original);
- `L_os` do subconjunto n ≥ 1 (`nonzero` + indexação dinâmica + 2ª passada de STFT): a perda é média por amostra,
  então `L_os(lote) + (n1/nb)·L_os(n1) = média(per · (1 + [n ≥ 1]))`, com uma STFT só;
- `_bin_weights` sem as verificações `bool(...)` (os pesos de bins com max_freq ≥ sr/2 são todos 1, o que dá a mesma
  média do caminho `err.mean()`); pesos de formantes calculados uma vez por (n_fft, device);
- estatísticas `g[v].median()` viram `nanmedian` sobre o tensor mascarado (mesmo valor: a mediana inferior).

O lote tem de ser homogêneo (todo com perfil ou todo sem): `AdapterGate.static` evita o `unique`/`nonzero` do LoRAGRU.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor

from cctrain.losses.dfnet import GAMMA, N_FFTS, SR, _EPS, _angle, formant_frequency_weights

FRAME_S = 480 / 48_000  # HOP / SR (igual a train_pro_v42.FRAME_S)

_FW_CACHE: dict = {}


def _formant_w(n_fft: int, device, formant_weight: float) -> tuple[Tensor, Tensor]:
    key = (n_fft, str(device), formant_weight)
    if key not in _FW_CACHE:
        fw, fm = formant_frequency_weights(n_fft, SR, torch.device("cpu"), formant_weight=formant_weight)
        _FW_CACHE[key] = (fw.to(device), fm.to(device))
    return _FW_CACHE[key]


_FREQ_CACHE: dict = {}


def _bin_w(max_freq: Tensor, n_fft: int) -> Tensor:
    key = (n_fft, str(max_freq.device))
    if key not in _FREQ_CACHE:
        _FREQ_CACHE[key] = torch.arange(n_fft // 2 + 1, dtype=torch.float32, device=max_freq.device) * (SR / n_fft)
    return (_FREQ_CACHE[key][None, :] <= max_freq.reshape(-1, 1).float()).to(torch.float32).unsqueeze(-1)


def _per_sample(err: Tensor, w: Tensor) -> Tensor:
    if err.ndim == 4:
        w = w.unsqueeze(-1)
    return (err * w).flatten(1).sum(dim=1) / (w.expand_as(err).flatten(1).sum(dim=1).clamp_min(1.0))


def _stft(x: Tensor, n_fft: int, win: Tensor) -> Tensor:
    return torch.stft(x.reshape(-1, x.shape[-1]), n_fft=n_fft, hop_length=n_fft // 4, window=win, normalized=True,
                      return_complex=True)


_WIN: dict = {}


def _win(n_fft: int, x: Tensor) -> Tensor:
    key = (n_fft, str(x.device), x.dtype)
    if key not in _WIN:
        _WIN[key] = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return _WIN[key]


def mr_and_os(est: Tensor, tgt: Tensor, max_freq: Tensor, os_weight: Tensor, *, n_ffts=N_FFTS, gamma=GAMMA,
              factor: float, factor_complex: float, os_factor: float, formant_weight: float,
              extra_formant_penalty: float = 1.0) -> tuple[Tensor, Tensor]:
    """`dfnet_loss` e `asymmetric_formant_loss` (ponderada por amostra com `os_weight`) com as STFTs compartilhadas."""
    angle = _angle()
    mr = est.new_zeros(())
    os_ = est.new_zeros(())
    for n_fft in n_ffts:
        win = _win(n_fft, est)
        y, s = _stft(est, n_fft, win), _stft(tgt, n_fft, win)
        y_abs, s_abs = y.abs().clamp_min(_EPS).pow(gamma), s.abs().clamp_min(_EPS).pow(gamma)
        w = _bin_w(max_freq, n_fft)
        mr = mr + _per_sample((y_abs - s_abs) ** 2, w).mean() * factor
        if factor_complex:  # view_as_real(|·|^γ · e^{jφ}) em reais (o inductor não lida com o escalar complexo 1j)
            ay, as_ = angle.apply(y), angle.apply(s)
            yc = torch.stack([y_abs * torch.cos(ay), y_abs * torch.sin(ay)], -1)
            sc = torch.stack([s_abs * torch.cos(as_), s_abs * torch.sin(as_)], -1)
            mr = mr + _per_sample((yc - sc) ** 2, w).mean() * factor_complex
        fw, fm = _formant_w(n_fft, est.device, formant_weight)
        err = F.relu(s_abs - y_abs) ** 2
        if extra_formant_penalty > 0.0:
            err = err * (1.0 + fm * extra_formant_penalty * err)
        err = err * fw
        os_ = os_ + (_per_sample(err, w) * os_weight).mean()
    return mr, os_ / len(n_ffts) * os_factor


def _pow(spec: Tensor) -> Tensor:
    return spec[:, 0, ..., 0] ** 2 + spec[:, 0, ..., 1] ** 2


def _safe_div(num: Tensor, den: Tensor) -> Tensor:
    """num/den com den = 0 -> 0 (o num já é 0 nesse caso); gradiente finito."""
    return num / torch.where(den > 0, den, torch.ones_like(den))


def leak_loss(out, spec, fmask):
    eo, em = _pow(out).sum(-1), _pow(spec).sum(-1)
    eps = 1e-10
    r = F.relu(10 * torch.log10((eo + eps) / (em + eps)) + 40.0) / 40.0
    return _safe_div((r * fmask).sum(), fmask.sum())


def ovl_loss(out, spec_clean, spec_interf, fmask):
    y = (_pow(out) + 1e-12).sqrt()
    s = (_pow(spec_clean) + 1e-12).sqrt()
    i = (_pow(spec_interf) + 1e-12).sqrt()
    ii = (i * i).sum(-1) + 1e-8
    v = F.relu(((y * i).sum(-1) - (s * i).sum(-1)) / ii)
    return _safe_div((v * fmask).sum(), fmask.sum())


def local_snr_loss(lsnr, target_lsnr, keep, *, factor: float):
    x = lsnr.reshape(lsnr.shape[0], -1)
    k = keep.to(x.dtype)
    return factor * _safe_div((((x - target_lsnr) ** 2) * k).sum(), k.sum())


def since_interferer_s(ia: Tensor) -> Tensor:
    T = ia.shape[1]
    idx = torch.arange(T, device=ia.device)[None].expand_as(ia)
    last = torch.where(ia, idx, torch.full_like(idx, -1)).cummax(1).values
    s = (idx - last).float() * FRAME_S
    return torch.where(last >= 0, s, torch.full_like(s, float("inf")))


def _masked_median(x: Tensor, m: Tensor) -> Tensor:
    return torch.where(m, x, torch.full_like(x, float("nan"))).reshape(-1).nanmedian()


def turn_loss(shadow, spec_clean, valid, ta, since, *, hinge_db: float, boost: float, decay_s: float):
    e_sh, e_s = _pow(shadow.float()), _pow(spec_clean.float())
    E_sh, E_s = e_sh.sum(-1), e_s.sum(-1)
    g = torch.clamp(10 * torch.log10((E_sh + 1e-12) / (E_s + 1e-12)), min=-40.0)
    l = F.relu(hinge_db - g) / 10.0
    taf = ta.float()
    mean_act = (E_s * taf).sum(1, keepdim=True) / taf.sum(1, keepdim=True).clamp(min=1)
    w_e = torch.sqrt(E_s / mean_act.clamp(min=1e-12)).clamp(max=3.0)
    w_t = 1.0 + (boost - 1.0) * torch.clamp(1.0 - since / decay_s, min=0.0, max=1.0)
    w_t = torch.where(torch.isfinite(since), w_t, torch.ones_like(w_t))
    w = w_e * w_t * valid.float()
    loss = _safe_div((l * w).sum(), w.sum())
    with torch.no_grad():
        stats = {"g_turn_db": _masked_median(g, valid), "g_turn_early_db": _masked_median(g, valid & (since <= 0.5))}
    return loss, stats


_LSNR_T = {}


def _lsnr_target(df_params, spec_clean, spec):
    from df.modules import LocalSnrTarget

    key = id(df_params)
    if key not in _LSNR_T:
        _LSNR_T[key] = LocalSnrTarget(ws=20, target_snr_range=[df_params.lsnr_min - 1, df_params.lsnr_max + 1])
    return _LSNR_T[key](spec_clean, noise=spec - spec_clean)


def gate_v4(spec, spec_m, spec_e, lsnr, profile):
    level = lsnr.detach().reshape(spec.shape[0], -1)
    prof = profile.reshape(-1, 1).bool()
    zero = level < -10.0
    none = (~zero) & (level > 30.0) & (~prof)
    erb_only = (~zero) & (~none) & (level > 20.0) & (~prof)

    def w(m):
        return m[:, None, :, None, None]

    out = torch.where(w(zero), torch.zeros_like(spec_e), spec_e)
    out = torch.where(w(erb_only), spec_m, out)
    out = torch.where(w(none), spec, out)
    return out, zero


def target_shadow(base, spec_clean, m, df_coefs, zero):
    sm = base.mask(spec_clean.clone(), m)
    se = base.df_op(spec_clean.clone(), df_coefs)
    se = torch.cat([se[..., : base.nb_df, :], sm[..., base.nb_df:, :]], dim=-2)
    return torch.where(zero[:, None, :, None, None], torch.zeros_like(se), se)


def step_v5_sf(model, istft, df_params, hp, b: dict, a, amp: bool) -> dict:
    """Mesmas perdas de `train_pro_v5.step_v5`, sem sync do host (ver docstring do módulo)."""
    spec, spec_clean, mask = b["spec"], b["spec_clean"], b["mask"]
    nb, T = spec.shape[0], spec.shape[2]
    profile = mask.reshape(nb) > 0
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    with ctx:
        spec_e, m, lsnr, df_coefs = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], mask)
    spec_e, m, lsnr, df_coefs = spec_e.float(), m.float(), lsnr.float(), df_coefs.float()
    base = model.base
    spec_m = base.mask(spec.clone(), m)
    if amp:
        spec_e = base.df_op(spec.clone(), df_coefs)
        spec_e = torch.cat([spec_e[..., : base.nb_df, :], spec_m[..., base.nb_df:, :]], dim=-2)
    out, zero = gate_v4(spec, spec_m, spec_e, lsnr, profile)
    est, tgt = istft(out), istft(spec_clean)

    ta = b["target_active"][:, :T]
    ia = b["interf_active"][:, :T]
    ds = b["digital_silence"][:, :T]
    prof = profile[:, None]
    f_leak = (prof & ~ta & ia & ~ds).float()
    f_ovl = (prof & ta & ia).float()
    f_keep = prof & ta & ~ds
    n1 = b["n_interf"] >= 1
    has_int = n1[:, None]

    mr, os_ = mr_and_os(est, tgt, b["max_freq"], 1.0 + n1.float(), n_ffts=hp.fft_sizes, gamma=hp.loss_gamma,
                        factor=hp.loss_factor, factor_complex=hp.loss_factor_complex, os_factor=hp.lambda_os,
                        formant_weight=a.formant_weight)
    leak = leak_loss(out, spec, f_leak)
    ovl = ovl_loss(out, spec_clean, b["spec_interf"], f_ovl)
    lsnr_t = _lsnr_target(df_params, spec_clean, spec)
    ls = local_snr_loss(lsnr, lsnr_t, ~ds, factor=a.lsnr_factor)

    shadow = target_shadow(base, spec_clean, m, df_coefs, zero)
    since_i = since_interferer_s(ia)
    valid = prof & has_int & b["has_target"][:, None] & ta & ~ia & ~ds
    turn, tst = turn_loss(shadow, spec_clean, valid, ta, since_i, hinge_db=a.turn_hinge_db, boost=a.turn_boost,
                          decay_s=a.turn_decay_s)
    v0 = prof & (b["n_interf"] == 0)[:, None] & b["has_target"][:, None] & ta & ~ds
    since_0 = torch.full_like(since_i, float("inf"))  # since_interferer_s(zeros) = inf em todo quadro
    n0g, n0st = turn_loss(shadow, spec_clean, v0, ta, since_0, hinge_db=a.n0g_hinge_db, boost=1.0, decay_s=1.0)
    iw = b["interf_active_wet45"][:, :T]
    f_tail = (prof & ~ta & ~ia & iw & ~ds & (since_i <= a.tail_s)).float()
    tail = leak_loss(out, spec, f_tail)
    leak_abs = leak_loss(out, spec, f_leak * (~b["has_target"]).float()[:, None])
    d = df_coefs[:, 1:] - df_coefs[:, :-1]
    l_df = d.abs().sum() / (df_coefs.numel() / df_coefs.shape[1]) * 0.05

    parts = {"mr": mr, "os": os_, "leak": a.lambda_leak * leak, "ovl": a.lambda_ovl * ovl, "lsnr": ls,
             "turn": a.lambda_turn * turn, "n0g": a.lambda_n0g * n0g, "tail": a.lambda_tail * tail,
             "leak_abs": a.lambda_leak_absent * leak_abs, "df_reg": l_df}
    total = sum(parts.values())
    with torch.no_grad():
        oi = f_leak.bool()
        stats = {"raw_leak": leak.detach(), "raw_turn": turn.detach(), "raw_n0g": n0g.detach(),
                 "raw_tail": tail.detach(), "g_turn_db": tst["g_turn_db"], "g_turn_early_db": tst["g_turn_early_db"],
                 "g_n0_db": n0st["g_turn_db"],
                 "zero_frac_only_interf": (zero & oi).sum() / oi.sum().clamp(min=1),
                 "zero_frac_target": (zero & f_keep).sum() / f_keep.sum().clamp(min=1)}
    return {"total": total, **parts, **stats}


class IstftSF(torch.nn.Module):
    """`df.loss.Istft` sem o teste NOLA de `torch.istft` (que lê a GPU no host e quebra a captura de CUDA Graph):
    irfft (norm ortho = normalized=True) · janela, overlap-add por `F.fold`, ÷ envelope Σ janela², corte center."""

    def __init__(self, istft):
        super().__init__()
        self.n_fft, self.hop = istft.n_fft_inv, istft.hop_inv
        self.register_buffer("w", istft.w_inv.detach().clone())
        self._env: dict = {}

    def _envelope(self, n_frames: int, device, dtype) -> Tensor:
        key = (n_frames, str(device), dtype)
        if key not in self._env:
            n = self.n_fft + self.hop * (n_frames - 1)
            w2 = (self.w.to(device, dtype) ** 2)[None, :, None].expand(1, -1, n_frames)
            env = F.fold(w2, output_size=(1, n), kernel_size=(1, self.n_fft), stride=(1, self.hop)).reshape(n)
            c = self.n_fft // 2
            self._env[key] = env[c:n - c]
        return self._env[key]

    def forward(self, x: Tensor) -> Tensor:
        xc = torch.view_as_complex(x.contiguous()) if not x.is_complex() else x  # [B, 1, T, F]
        t, f = xc.shape[-2:]
        sh = xc.shape[:-2]
        spec = F.pad(xc.reshape(-1, t, f).transpose(1, 2), (0, 1))  # [N, F, T+1]
        nfr = spec.shape[-1]
        fr = torch.fft.irfft(spec, n=self.n_fft, dim=1, norm="ortho") * self.w.to(spec.real.dtype)[None, :, None]
        n = self.n_fft + self.hop * (nfr - 1)
        y = F.fold(fr, output_size=(1, n), kernel_size=(1, self.n_fft), stride=(1, self.hop)).reshape(fr.shape[0], n)
        c = self.n_fft // 2
        y = y[:, c:n - c] / self._envelope(nfr, y.device, y.dtype)
        return y.view(*sh, y.shape[-1]) if xc.ndim > 2 else y


# --- executor do passo (forward + perdas + backward) -----------------------------------------------------------------


def batch_profile_state(mask0) -> bool | None:
    """Pelo `mask0` da receita (CPU, antes da mistura na GPU): True = lote todo com perfil, False = todo sem, None =
    misto (caminho dinâmico)."""
    if mask0 is None:
        return None
    m = torch.as_tensor(mask0).reshape(-1).bool()
    if bool(m.all()):
        return False
    if not bool(m.any()):
        return True
    return None


class StepRunner:
    """forward + perdas + backward de um passo, com os gradientes deixados em `p.grad` dos treináveis.

    mode: "eager" (step_v5_sf, sem sync), "compile" (torch.compile da função forward+perdas, com `compile_mode`) ou
    "graph" (CUDA Graph manual de forward+perdas+backward, um grafo por estado de perfil; entradas estáticas).
    O otimizador, o clip e o scheduler ficam fora (o chamador)."""

    def __init__(self, model, istft, df_params, hp, a, amp, trainable: list, mode: str = "eager",
                 compile_mode: str | None = None, warmup: int = 3, loss_fn=None):
        self.model, self.istft, self.df_params, self.hp, self.a, self.amp = model, istft, df_params, hp, a, amp
        self.trainable = trainable
        self.mode = mode
        # torch.istft lê a GPU no host (teste NOLA): quebra grafo/captura
        self.istft = IstftSF(istft).to(next(model.parameters()).device)
        self.warmup = warmup
        self.gate = getattr(model, "adapter_gate", None)
        # loss_fn(model, istft, df_params, hp, b, a, amp) -> parts: padrão step_v5_sf; o train passa o step_v5 do
        # perf/v5-loss (perdas fast). Precisa ser sem sync para graph/compile reduce-overhead.
        self.loss_fn = loss_fn or step_v5_sf
        self.compile_mode = compile_mode
        self._fn = self._loss
        if mode == "compile":
            kw = {"mode": compile_mode} if compile_mode and compile_mode != "default" else {}
            self._fn = torch.compile(self._loss, dynamic=False, **kw)
        self.graphs: dict = {}

    def _loss(self, b: dict) -> dict:
        return self.loss_fn(self.model, self.istft, self.df_params, self.hp, b, self.a, self.amp)

    def _set_static(self, on: bool | None) -> None:
        if self.gate is not None:
            self.gate.static = on

    def __call__(self, b: dict, on: bool | None) -> dict:
        if self.mode == "graph" and on is not None:
            return self._graph_step(b, on)
        self._set_static(on)
        try:
            for p in self.trainable:
                p.grad = None
            if self.mode == "compile" and self.compile_mode in ("reduce-overhead", "max-autotune"):
                torch.compiler.cudagraph_mark_step_begin()
            parts = self._fn(b)
            parts["total"].backward()
        finally:
            self._set_static(None)
        # sem referência ao grafo do autograd fora daqui: um AccumulateGrad vivo guarda a stream em que foi criado e,
        # se for a stream padrão (legada), a captura seguinte falha ("legacy stream depend on a capturing stream")
        return {k: v.detach() for k, v in parts.items()}

    def _graph_step(self, b: dict, on: bool) -> dict:
        st = self.graphs.get(on)
        if st is None:
            st = self.graphs[on] = self._capture(b, on)
        static_b, graph, parts, grads = st
        for k, v in b.items():
            if torch.is_tensor(v):
                static_b[k].copy_(v, non_blocking=True)
        graph.replay()
        for p, g in zip(self.trainable, grads):
            p.grad = g
        return parts

    def _capture(self, b: dict, on: bool):
        static_b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        self._set_static(on)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(self.warmup):
                    for p in self.trainable:
                        p.grad = None
                    self._loss(static_b)["total"].backward()
            torch.cuda.current_stream().wait_stream(s)
            for p in self.trainable:
                p.grad = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # devolve o cache do caminho eager: o pool do grafo guarda todas as ativações
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):  # thread do pin_memory
                parts = self._loss(static_b)
                parts["total"].backward()
            grads = [p.grad for p in self.trainable]
            parts = {k: v.detach() for k, v in parts.items()}
        finally:
            self._set_static(None)
        return static_b, graph, parts, grads
