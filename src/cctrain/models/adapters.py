"""Adaptadores do pDFNet3 Pro v5: LoRA nas GRUs / convs 1×1 do encoder e FiLM opcional na saída das GRUs.

A rede base (DFNet3 publicado, ou o v3.1 no controle) fica congelada; os adaptadores são módulos SEPARADOS, enrolados em
volta das camadas originais (o original fica em `.orig`, com os pesos intocados), ativos só com o perfil ligado:

    LoRA (linear/conv 1×1):  y = W x + g · s · B(A x)                  s = alpha / rank, B = 0 no início
    LoRA (GRU, por camada):  W_ih' = W_ih + g·s·B_ih A_ih,  W_hh' = W_hh + g·s·B_hh A_hh   (entrada e recorrente)
    FiLM (saída da GRU):     y' = y ⊙ (1 + g·γ(e)) + g·β(e)              γ, β = Linear(e) com peso e viés zerados

g é a máscara de perfil por amostra (0 = desligado, 1 = ligado) e e o embedding do perfil. Com g = 0 o caminho é o
módulo original, sem nenhuma operação a mais (`torch.where`/ramo separado): saída bit-idêntica à base. No passo 0
(B = 0, FiLM zerado) também com g = 1 a saída é bit-idêntica (W + 0 = W).

O g e o e chegam por um `AdapterGate` compartilhado, preenchido por um pre-hook no `forward` do PDFNet3Pro (a partir dos
argumentos `embedding` e `mask`) e limpo por um post-hook. Quem chama os submódulos da base diretamente (o eval quadro a
quadro em `cctrain.eval.tse_eval`) usa `set_gate`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

GRU_SITES = ("enc.emb_gru.gru", "erb_dec.emb_gru.gru", "df_dec.df_gru.gru")
# convs pontuais (1×1, groups=1) do encoder: o par depthwise+pointwise das convs separáveis
ENC_CONV_SITES = ("enc.erb_conv1.1", "enc.erb_conv2.1", "enc.erb_conv3.1", "enc.df_conv0.2", "enc.df_conv1.1")
ENC_LINEAR_SITES = ("enc.df_fc_emb.0",)  # GroupedLinearEinsum 3072 -> 512


@dataclass(frozen=True)
class AdapterConfig:
    gru_rank: int = 0  # 0 = sem LoRA nas GRUs
    enc_rank: int = 0  # 0 = sem LoRA no encoder
    alpha_over_rank: float = 1.0  # s = alpha / rank
    gru_film: bool = False  # FiLM do embedding na saída das GRUs
    emb_dim: int = 192

    @property
    def any(self) -> bool:
        return self.gru_rank > 0 or self.enc_rank > 0 or self.gru_film

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> AdapterConfig:
        return cls(**(d or {}))


class AdapterGate:
    """Estado compartilhado (não é Module nem buffer): g [B] float ou None (= desligado) e o embedding [B, E].

    `static` (padrão None = decide pelo conteúdo de g, com sync do host): True = lote homogêneo COM perfil e g ≡ 1,
    False = lote homogêneo SEM perfil. Quem chama garante a homogeneidade (o host sabe pela receita, sem ler a GPU);
    sem sync o forward pode ser capturado num CUDA Graph / torch.compile sem quebra de grafo."""

    __slots__ = ("g", "emb", "static")

    def __init__(self):
        self.g: Tensor | None = None
        self.emb: Tensor | None = None
        self.static: bool | None = None


class _Adapter(nn.Module):
    def __init__(self, gate: AdapterGate):
        super().__init__()
        object.__setattr__(self, "_gate", gate)  # fora do registro de submódulos/estado

    def _on(self) -> Tensor | None:
        g = self._gate.g
        if g is None:
            return None
        if self._gate.static is not None:
            return g if self._gate.static else None
        if not bool((g > 0).any()):
            return None
        return g


def _lora_pair(out_f: int, in_f: int, rank: int) -> tuple[nn.Parameter, nn.Parameter]:
    a = nn.Parameter(torch.empty(rank, in_f))
    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
    return a, nn.Parameter(torch.zeros(out_f, rank))


class LoRAConv1x1(_Adapter):
    def __init__(self, conv: nn.Conv2d, rank: int, scale: float, gate: AdapterGate):
        super().__init__(gate)
        assert conv.kernel_size == (1, 1) and conv.groups == 1, conv
        self.orig = conv
        self.scale = scale
        self.lora_A, self.lora_B = _lora_pair(conv.out_channels, conv.in_channels, rank)

    def forward(self, x: Tensor) -> Tensor:
        y = self.orig(x)
        g = self._on()
        if g is None:
            return y
        a = F.conv2d(x, self.lora_A[:, :, None, None], stride=self.orig.stride)
        d = F.conv2d(a, self.lora_B[:, :, None, None]) * self.scale
        gb = g.to(y.dtype).view(-1, 1, 1, 1)
        return torch.where(gb > 0, y + gb * d, y)


class LoRALinear(_Adapter):
    """Sobre um linear qualquer de [B, T, I] -> [B, T, O] (aqui o GroupedLinearEinsum do encoder)."""

    def __init__(self, lin: nn.Module, in_f: int, out_f: int, rank: int, scale: float, gate: AdapterGate):
        super().__init__(gate)
        self.orig = lin
        self.scale = scale
        self.lora_A, self.lora_B = _lora_pair(out_f, in_f, rank)

    def forward(self, x: Tensor) -> Tensor:
        y = self.orig(x)
        g = self._on()
        if g is None:
            return y
        d = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        gb = g.to(y.dtype).view(-1, *([1] * (y.dim() - 1)))
        return torch.where(gb > 0, y + gb * d, y)


class LoRAGRU(_Adapter):
    """LoRA nas projeções de entrada e recorrente de cada camada de um `nn.GRU` (batch_first), FiLM opcional na saída.

    As amostras com perfil rodam a GRU com os pesos somados (W + g·s·BA) pelo mesmo kernel do `nn.GRU`
    (`torch._VF.gru`, cuDNN na GPU); as sem perfil rodam o módulo original. Os dois grupos são remontados na ordem.

    `fused = True` (treino, gate binário g ∈ {0, 1}): em vez de duas chamadas do cuDNN (uma por grupo, cada uma com os
    T passos sequenciais: o custo da GRU dobra num lote misto), UMA chamada com hidden 2H em que as duas GRUs (original
    e com LoRA) correm lado a lado com pesos bloco-diagonais por porta (r, z, n); cada amostra fica com a metade do seu
    grupo. Matematicamente idêntico (as portas da GRU são elemento a elemento); numericamente igual a ~1 ulp (outro
    formato de GEMM), por isso só no treino: o padrão (False) mantém a saída bit-idêntica da base nas linhas sem perfil.
    """

    fused: bool = False

    def __init__(self, gru: nn.GRU, rank: int, scale: float, gate: AdapterGate, film_emb_dim: int | None = None):
        super().__init__(gate)
        assert gru.batch_first and not gru.bidirectional and gru.bias and gru.proj_size == 0
        self.orig = gru
        self.rank = rank
        self.scale = scale
        h = gru.hidden_size
        if rank > 0:
            self.lora = nn.ParameterDict()
            for layer in range(gru.num_layers):
                in_f = gru.input_size if layer == 0 else h
                self.lora[f"ih_l{layer}_A"], self.lora[f"ih_l{layer}_B"] = _lora_pair(3 * h, in_f, rank)
                self.lora[f"hh_l{layer}_A"], self.lora[f"hh_l{layer}_B"] = _lora_pair(3 * h, h, rank)
        else:
            self.lora = None
        if film_emb_dim:
            self.film = nn.Linear(film_emb_dim, 2 * h)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            self.film = None

    def _params(self, gv: float) -> list[Tensor]:
        g = self.orig
        out = []
        for layer in range(g.num_layers):
            w_ih, w_hh = getattr(g, f"weight_ih_l{layer}"), getattr(g, f"weight_hh_l{layer}")
            if self.lora is not None:
                s = gv * self.scale
                w_ih = w_ih + s * (self.lora[f"ih_l{layer}_B"] @ self.lora[f"ih_l{layer}_A"])
                w_hh = w_hh + s * (self.lora[f"hh_l{layer}_B"] @ self.lora[f"hh_l{layer}_A"])
            out += [w_ih, w_hh, getattr(g, f"bias_ih_l{layer}"), getattr(g, f"bias_hh_l{layer}")]
        return out

    def _run_on(self, x: Tensor, h: Tensor | None, gv: float, emb: Tensor | None) -> tuple[Tensor, Tensor]:
        g = self.orig
        if h is None:
            h = torch.zeros(g.num_layers, x.shape[0], g.hidden_size, dtype=x.dtype, device=x.device)
        y, hn = torch._VF.gru(x, h, self._params(gv), True, g.num_layers, float(g.dropout), g.training, False, True)
        if self.film is not None:
            gamma, beta = self.film(emb).chunk(2, dim=-1)
            y = y * (1.0 + gv * gamma[:, None]) + gv * beta[:, None]
        return y, hn

    def _fused_params(self) -> list[Tensor]:
        """Pesos da GRU de hidden 2H = [original | com LoRA (g = 1)], bloco-diagonais por porta."""
        g = self.orig
        H = g.hidden_size
        out = []
        both = list(zip(self._params(0.0) if self.lora is None else self._params_raw(), self._params(1.0)))
        for layer in range(g.num_layers):
            (wi0, wi1), (wh0, wh1), (bi0, bi1), (bh0, bh1) = both[4 * layer: 4 * layer + 4]
            if layer == 0:  # entrada compartilhada: só empilha as linhas por porta
                w_ih = torch.cat([t for k in range(3) for t in (wi0[k * H:(k + 1) * H], wi1[k * H:(k + 1) * H])], 0)
            else:
                w_ih = _blockdiag_gates(wi0, wi1, H)
            w_hh = _blockdiag_gates(wh0, wh1, H)
            b_ih = torch.cat([t for k in range(3) for t in (bi0[k * H:(k + 1) * H], bi1[k * H:(k + 1) * H])], 0)
            b_hh = torch.cat([t for k in range(3) for t in (bh0[k * H:(k + 1) * H], bh1[k * H:(k + 1) * H])], 0)
            out += [w_ih, w_hh, b_ih, b_hh]
        return out

    def _params_raw(self) -> list[Tensor]:
        g = self.orig
        out = []
        for layer in range(g.num_layers):
            out += [getattr(g, f"weight_ih_l{layer}"), getattr(g, f"weight_hh_l{layer}"),
                    getattr(g, f"bias_ih_l{layer}"), getattr(g, f"bias_hh_l{layer}")]
        return out

    def _forward_fused(self, x: Tensor, h: Tensor | None, g: Tensor, emb: Tensor | None) -> tuple[Tensor, Tensor]:
        gr = self.orig
        H = gr.hidden_size
        if h is None:
            h2 = torch.zeros(gr.num_layers, x.shape[0], 2 * H, dtype=x.dtype, device=x.device)
        else:
            h2 = torch.cat([h, h], dim=-1)
        y2, hn2 = torch._VF.gru(x, h2, self._fused_params(), True, gr.num_layers, float(gr.dropout), gr.training,
                                False, True)
        on = g > 0
        y1 = y2[..., H:]
        if self.film is not None:
            gamma, beta = self.film(emb).chunk(2, dim=-1)
            gv = g.to(y1.dtype)[:, None, None]
            y1 = y1 * (1.0 + gv * gamma[:, None]) + gv * beta[:, None]
        y = torch.where(on[:, None, None], y1, y2[..., :H])
        hn = torch.where(on[None, :, None], hn2[..., H:], hn2[..., :H])
        return y, hn

    def forward(self, x: Tensor, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        g = self._on()
        if g is None:
            return self.orig(x, h)
        emb = self._gate.emb
        if self._gate.static:  # lote homogêneo com g ≡ 1: um só grupo, sem unique/nonzero (sem sync)
            return self._run_on(x, h, 1.0, emb)
        if self.fused:
            return self._forward_fused(x, h, g, emb)
        b = x.shape[0]
        groups = []  # (índices, valor de g)
        for gv in torch.unique(g[g > 0]).tolist():
            groups.append(((g == gv).nonzero(as_tuple=True)[0], float(gv)))
        off = (g <= 0).nonzero(as_tuple=True)[0]
        if not off.numel() and len(groups) == 1:
            return self._run_on(x, h, groups[0][1], emb)
        ys, hs, idxs = [], [], []
        for idx, gv in groups:
            y, hn = self._run_on(x[idx], None if h is None else h[:, idx], gv, None if emb is None else emb[idx])
            ys.append(y); hs.append(hn); idxs.append(idx)
        if off.numel():
            y, hn = self.orig(x[off], None if h is None else h[:, off])
            ys.append(y); hs.append(hn); idxs.append(off)
        order = torch.cat(idxs)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(b, device=order.device)
        return torch.cat(ys, 0)[inv], torch.cat(hs, 1)[:, inv]


def _blockdiag_gates(w0: Tensor, w1: Tensor, H: int) -> Tensor:
    """[3H, H] × 2 -> [6H, 2H]: por porta k, [[w0_k, 0], [0, w1_k]] (ordem das portas do cuDNN/torch: r, z, n)."""
    z = w0.new_zeros(H, w1.shape[1])
    rows = []
    for k in range(3):
        rows.append(torch.cat([w0[k * H:(k + 1) * H], z], 1))
        rows.append(torch.cat([z, w1[k * H:(k + 1) * H]], 1))
    return torch.cat(rows, 0)


def set_lora_gru_fused(root: nn.Module, on: bool = True) -> int:
    """Liga o caminho fundido (uma chamada do cuDNN com hidden 2H) em todas as LoRAGRU. Só para gate binário (treino)."""
    n = 0
    for m in root.modules():
        if isinstance(m, LoRAGRU):
            m.fused = on
            n += 1
    return n


# --- montagem ------------------------------------------------------------------------------------------------------


def _get(root: nn.Module, path: str) -> nn.Module:
    m = root
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def _set(root: nn.Module, path: str, new: nn.Module) -> None:
    parent, last = path.rsplit(".", 1)
    par = _get(root, parent)
    if last.isdigit():
        par[int(last)] = new
    else:
        setattr(par, last, new)


def _pre_hook(module, args, kwargs):
    names = ("spec", "feat_erb", "feat_spec", "embedding", "mask")
    bound = dict(zip(names, args))
    bound.update({k: v for k, v in kwargs.items() if k in names})
    mask, emb = bound.get("mask"), bound.get("embedding")
    gate = module.adapter_gate
    gate.g = None if mask is None else mask.reshape(-1).to(torch.float32)
    gate.emb = emb.reshape(emb.shape[0], -1) if emb is not None else None


def _post_hook(module, args, kwargs, output):
    module.adapter_gate.g = None
    module.adapter_gate.emb = None


def attach_adapters(model: nn.Module, cfg: AdapterConfig) -> dict[str, list[str]]:
    """Enrola os adaptadores em `model.base` (PDFNet3Pro) e liga os hooks. Chame DEPOIS de carregar os pesos da base e
    ANTES de carregar um checkpoint que já tenha adaptadores. Devolve os sítios por tipo."""
    if getattr(model, "adapter_cfg", None) is not None:
        raise RuntimeError("adaptadores já anexados")
    gate = AdapterGate()
    object.__setattr__(model, "adapter_gate", gate)
    model.adapter_cfg = cfg
    sites: dict[str, list[str]] = {"gru": [], "enc_conv": [], "enc_linear": []}
    s = cfg.alpha_over_rank
    if cfg.gru_rank > 0 or cfg.gru_film:
        for p in GRU_SITES:
            gru = _get(model.base, p)
            _set(model.base, p, LoRAGRU(gru, cfg.gru_rank, s, gate, cfg.emb_dim if cfg.gru_film else None)
                 .to(next(gru.parameters()).device))
            sites["gru"].append(p)
    if cfg.enc_rank > 0:
        for p in ENC_CONV_SITES:
            conv = _get(model.base, p)
            _set(model.base, p, LoRAConv1x1(conv, cfg.enc_rank, s, gate).to(conv.weight.device))
            sites["enc_conv"].append(p)
        for p in ENC_LINEAR_SITES:
            lin = _get(model.base, p)
            _set(model.base, p, LoRALinear(lin, lin.input_size, lin.hidden_size, cfg.enc_rank, s, gate)
                 .to(lin.weight.device))
            sites["enc_linear"].append(p)
    model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    model.register_forward_hook(_post_hook, with_kwargs=True)
    return sites


def adapter_modules(root: nn.Module) -> list[_Adapter]:
    return [m for m in root.modules() if isinstance(m, _Adapter)]


def adapter_param_names(model: nn.Module) -> list[str]:
    """Nomes dos parâmetros NOVOS dos adaptadores (exclui o `.orig.*`, que é a base)."""
    out = []
    for n, _ in model.named_parameters():
        if ".orig." in n:
            continue
        if any(k in n for k in (".lora_A", ".lora_B", ".lora.", ".film.")) and n.startswith("base."):
            out.append(n)
    return out


def set_gate(root: nn.Module, g: Tensor | None, emb: Tensor | None = None) -> None:
    """Liga/desliga os adaptadores em quem chama os submódulos da base sem passar pelo forward do PDFNet3Pro."""
    for m in adapter_modules(root):
        m._gate.g = None if g is None else g.reshape(-1).to(torch.float32)
        m._gate.emb = emb


# --- congelamento --------------------------------------------------------------------------------------------------

# camadas que o Pro adiciona em cima do DFNet3 (o resto de `base.*` é o DFNet3 publicado)
COND_PREFIXES = ("film.", "cross_attn.", "freq_mixer.", "presence.", "lsnr_corr.", "oracle_cond.")


def param_groups(model: nn.Module) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"base": [], "cond": [], "adapter": []}
    ad = set(adapter_param_names(model))
    for n, _ in model.named_parameters():
        if n in ad:
            groups["adapter"].append(n)
        elif n.startswith(COND_PREFIXES):
            groups["cond"].append(n)
        else:
            groups["base"].append(n)
    return groups


def apply_trainable(model: nn.Module, train_cond: bool, train_adapters: bool) -> list[nn.Parameter]:
    """requires_grad só nos grupos pedidos (BatchNorm continua congelada). Devolve os parâmetros treináveis."""
    g = param_groups(model)
    on = set((g["cond"] if train_cond else []) + (g["adapter"] if train_adapters else []))
    bn = set()
    for mn, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            bn |= {f"{mn}.{pn}" for pn, _ in m.named_parameters(recurse=False)}
    out = []
    for n, p in model.named_parameters():
        req = n in on and n not in bn
        p.requires_grad_(req)
        if req:
            out.append(p)
    return out
