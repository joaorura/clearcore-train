"""Oráculo de presença (exp/oracle-ceiling, teste de teto da `docs/proposta-arquitetura-tse.md` §3.1).

O modelo recebe, como entrada extra por quadro, a atividade VERDADEIRA do alvo e do interferente (rótulos do mix: alvo
`target_active`, VAD pico−30; interferente `interf_active_wet45`, VAD pico−45 da trilha molhada). Pergunta: com presença
perfeita, o modelo passa do teto C8 × (C1, C6)? Não é um modelo para produção.

Entradas por quadro: o = [alvo, interf, alvo·interf] ∈ {0,1}³ (o produto dá à camada afim o termo de sobreposição).
Injeção (todas zero-init: no passo 0 o modelo é o de antes; tudo × máscara de perfil):
  - antes da GRU do encoder: β_enc' = β_enc + W_e o           (soma na entrada do `enc.emb_gru.gru`, depois do linear_in)
  - decoder DF: FiLM γ_df' = γ_df ⊙ (1 + W_dg o), β_df' = β_df + W_db o       (entrada do `df_dec.df_gru.gru`)
  - decoder ERB: emb' = emb ⊙ (1 + W_rg o) + W_rb o                          (entrada do `erb_dec`)
  - gate ZERO: lsnr' = lsnr + escala · W_l o                                  (mesma escala do `lsnr_corr`)

Alinhamento: o rótulo é do quadro de SAÍDA (o mesmo índice `t` de `spec`/`target_active` no treino). No treino o encoder
roda sobre features adiantadas por `pad_feat` (o índice t do encoder já é o quadro de saída t). No eval quadro a quadro
(`tse_eval.ProfileFrameModel`) o encoder no índice t produz a saída do quadro t − conv_lookahead (2): lá o oráculo é
deslocado de 2 quadros (`eval_oracle_frames`).

Braço O2 (oráculo imperfeito): atraso de `delay` quadros + erros em rajadas (inversão do bit em trechos de `burst`
quadros, início com prob. err/burst por quadro, ~err da duração), independentes por canal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn

N_FEAT = 3
CONV_LOOKAHEAD = 2


@dataclass(frozen=True)
class OracleConfig:
    mode: str = "clean"  # clean | noisy
    delay_frames: int = 0
    err_frac: float = 0.0
    burst_frames: int = 20

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "OracleConfig | None":
        return cls(**d) if d else None


class OracleCond(nn.Module):
    def __init__(self, enc_hidden: int, df_hidden: int, emb_dim: int):
        super().__init__()
        self.enc_b = nn.Linear(N_FEAT, enc_hidden)
        self.df_g = nn.Linear(N_FEAT, df_hidden)
        self.df_b = nn.Linear(N_FEAT, df_hidden)
        self.erb_g = nn.Linear(N_FEAT, emb_dim)
        self.erb_b = nn.Linear(N_FEAT, emb_dim)
        self.lsnr = nn.Linear(N_FEAT, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, o: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """o [B,T,3] (já em float), mask [B,1] -> deltas [B,T,·] multiplicados pela máscara de perfil."""
        g = mask.reshape(-1, 1, 1).to(o.dtype)
        return {k: getattr(self, k)(o) * g for k in ("enc_b", "df_g", "df_b", "erb_g", "erb_b", "lsnr")}


def attach_oracle(model: nn.Module, cfg: OracleConfig) -> OracleCond:
    dev = next(model.parameters()).device
    base = model.base
    mod = OracleCond(base.enc.emb_gru.hidden_size, base.df_dec.df_gru.hidden_size, model.cross_attn.cfg.query_dim)
    model.oracle_cond = mod.to(dev)
    model.oracle_cfg = cfg
    return mod


def features(ta: Tensor, ia: Tensor) -> Tensor:
    """ta, ia [B,T] (bool/float) -> [B,T,3] float."""
    ta, ia = ta.float(), ia.float()
    return torch.stack([ta, ia, ta * ia], -1)


# --- corrupção (braço O2) -------------------------------------------------------------------------------------------


def corrupt_torch(lab: Tensor, cfg: OracleConfig, gen: torch.Generator | None = None) -> Tensor:
    """lab [B,T] bool -> bool atrasado e com rajadas de erro (O2). `clean` devolve o próprio rótulo."""
    if cfg.mode == "clean":
        return lab
    x = lab
    d = int(cfg.delay_frames)
    if d > 0:
        x = torch.cat([torch.zeros_like(x[:, :d]), x[:, :-d]], 1)
    if cfg.err_frac > 0:
        L = max(1, int(cfg.burst_frames))
        p = cfg.err_frac / L
        st = (torch.rand(x.shape, device=x.device, generator=gen) < p).float()
        # rajada causal de L quadros a partir de cada início
        flip = torch.nn.functional.max_pool1d(torch.nn.functional.pad(st[:, None], (L - 1, 0)), L, stride=1)[:, 0] > 0
        x = x ^ flip
    return x


def corrupt_np(lab: np.ndarray, cfg: OracleConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.mode == "clean":
        return lab
    x = lab.astype(bool).copy()
    d = int(cfg.delay_frames)
    if d > 0:
        x = np.concatenate([np.zeros(d, bool), x[:-d]])[: lab.size]
    if cfg.err_frac > 0:
        L = max(1, int(cfg.burst_frames))
        st = rng.random(x.size) < cfg.err_frac / L
        flip = np.zeros(x.size, bool)
        for i in np.flatnonzero(st):
            flip[i:i + L] = True
        x = x ^ flip
    return x


# --- eval -----------------------------------------------------------------------------------------------------------


def eval_labels(target: np.ndarray, interf: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Rótulos do eval pela MESMA regra do mix_v4: alvo pico−30, interferentes pico−45 (OR por trilha)."""
    from cctrain.data.mix_v4 import rel_activity

    ta = rel_activity(np.asarray(target, np.float64), n_frames) if np.any(target) else np.zeros(n_frames, bool)
    ia = np.zeros(n_frames, bool)
    for x in np.atleast_2d(interf):
        if np.any(x):
            ia |= rel_activity(np.asarray(x, np.float64), n_frames, range_db=45.0)
    return ta, ia


def eval_oracle_frames(target: np.ndarray, interf: np.ndarray, active: np.ndarray, cfg: OracleConfig,
                       seed: int) -> np.ndarray:
    """Oráculo por quadro do ENCODER do eval (índice dos quadros fora do silêncio digital): [T_active, 3] float32.

    `target`/`interf` com o mesmo comprimento do áudio (já com a cauda). O encoder no índice t decide a saída do quadro
    ativo t − CONV_LOOKAHEAD."""
    hop = 480
    nf = len(target) // hop
    ta, ia = eval_labels(target, interf, nf)
    rng = np.random.default_rng(seed)
    ta = corrupt_np(ta, cfg, rng)
    ia = corrupt_np(ia, cfg, rng)
    ta, ia = ta[active], ia[active]
    k = CONV_LOOKAHEAD
    ta = np.concatenate([np.zeros(k, bool), ta[:-k]])[: ta.size] if ta.size else ta
    ia = np.concatenate([np.zeros(k, bool), ia[:-k]])[: ia.size] if ia.size else ia
    t, i = ta.astype(np.float32), ia.astype(np.float32)
    return np.stack([t, i, t * i], -1)
