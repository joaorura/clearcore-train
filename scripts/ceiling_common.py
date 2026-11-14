"""Experimento de teto da presença (exp/presence-ceiling): métricas, ECAPA em janelas deslizantes e cabeças.

Métricas no mesmo conjunto de dev do `eval_presence.py` (304 receitas, 520 vistas): AUC e acurácia balanceada (BA)
alvo × só-interferente; % ZERO no só-interferente e no alvo com a máquina de estados do runtime (histerese + hangover),
escolhendo o ponto de operação de maior % ZERO só-interf. com % ZERO alvo ≤ 2% (agregado), como o `--sweep`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from eval_presence import auc, bal_acc  # noqa: E402,F401
from train_pro_v4_presence import presence_state_machine  # noqa: E402

ECAPA_SR = 16_000
ECAPA_HOP_S = 0.1
FRAME_S = 0.01


# --- métricas --------------------------------------------------------------------------------------------------------


def pos_neg(c: dict, pos_mask: np.ndarray | None = None):
    """Máscaras [V,T] de positivos (alvo ativo na vista mix) e negativos (só interferente ativo)."""
    mix = c["view"] == "mix"
    pos = c["ta"] & c["valid"] & mix[:, None]
    if pos_mask is not None:
        pos = pos & pos_mask
    # como o eval_presence: só-interferente = quadros com interferente ativo e alvo inativo, nas duas vistas
    # (vista só-interf. e trechos sem alvo da mistura, inclusive o eixo temporal)
    neg = ~c["ta"] & c["ia"] & c["valid"]
    return pos, neg


def zero_rates(zero: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    return 100 * (zero & pos).sum() / max(pos.sum(), 1), 100 * (zero & neg).sum() / max(neg.sum(), 1)


def sweep_gate(score: np.ndarray, pos: np.ndarray, neg: np.ndarray, valid: np.ndarray, max_tgt: float = 2.0,
               prob: bool = True):
    """Varre (τ_on, τ_off, H). Limiar em quantis do score (vale para cosseno) e, se `prob`, também a grade do
    `eval_presence --sweep`. Devolve (melhor ponto, decisão ZERO [V,T])."""
    vs = score[valid]
    grid = []
    for q in np.arange(0.05, 0.96, 0.05):
        for d in (0.0, 0.05, 0.1, 0.2):
            if q - d <= 0.0:
                continue
            grid.append((float(np.quantile(vs, q)), float(np.quantile(vs, q - d))))
    if prob:
        for t_on in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for d in (0.0, 0.1, 0.2):
                if t_on - d > 0:
                    grid.append((t_on, round(t_on - d, 2)))
    best, best_z = None, None
    for t_on, t_off in grid:
        for h in (0, 3, 5, 8, 12, 20):
            z = ~presence_state_machine(score, t_on, t_off, h)
            tz, oz = zero_rates(z, pos, neg)
            row = {"tau_on": t_on, "tau_off": t_off, "hangover": h, "tgt_zero_pct": tz, "only_interf_zero_pct": oz}
            if tz <= max_tgt:
                if best is None or best["tgt_zero_pct"] > max_tgt or oz > best["only_interf_zero_pct"]:
                    best, best_z = row, z
            elif best is None or (best["tgt_zero_pct"] > max_tgt and tz < best["tgt_zero_pct"]):
                best, best_z = row, z
    return best, best_z


def by_group(c: dict, zero: np.ndarray, score: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> dict:
    """% ZERO e AUC por N e por ruído (o N do só-interf. é o da receita)."""
    out = {}
    n = c["n"]
    noise = c["noise"]
    for name, sel in [(f"N{k}", n == k) for k in range(5)] + \
                     [(f"noise:{z}", noise == z) for z in ("silent", "snr5", "snr20")]:
        p, q = pos & sel[:, None], neg & sel[:, None]
        if p.sum() == 0 and q.sum() == 0:
            continue
        row = {}
        if p.sum():
            row["tgt_zero_pct"] = round(100 * (zero & p).sum() / p.sum(), 2)
        if q.sum():
            row["only_interf_zero_pct"] = round(100 * (zero & q).sum() / q.sum(), 2)
        if p.sum() and q.sum():
            s = np.concatenate([score[p], score[q]])
            y = np.concatenate([np.ones(p.sum(), bool), np.zeros(q.sum(), bool)])
            row["auc"] = round(auc(s, y), 4)
        out[name] = row
    return out


def evaluate_score(c: dict, score: np.ndarray, pos_mask: np.ndarray | None = None, prob: bool = True,
                   gate: dict | None = None, groups: bool = True) -> dict:
    """Métricas completas de um score por quadro [V,T] (maior = alvo). `gate` fixo pula a varredura."""
    pos, neg = pos_neg(c, pos_mask)
    s = np.concatenate([score[pos], score[neg]])
    y = np.concatenate([np.ones(pos.sum(), bool), np.zeros(neg.sum(), bool)])
    a = auc(s, y)
    # BA no melhor limiar (independe de calibração) e no τ_on do ponto de operação
    qs = np.quantile(s, np.linspace(0.01, 0.99, 99))
    ba_max = max(bal_acc(s >= t, y) for t in qs)
    if gate is None:
        gate, zero = sweep_gate(score, pos, neg, c["valid"], prob=prob)
    else:
        zero = ~presence_state_machine(score, gate["tau_on"], gate["tau_off"], gate["hangover"])
        tz, oz = zero_rates(zero, pos, neg)
        gate = {**gate, "tgt_zero_pct": tz, "only_interf_zero_pct": oz}
    res = {"auc": a, "ba_max": ba_max, "ba_at_tau_on": bal_acc(s >= gate["tau_on"], y), "gate": gate,
           "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
    if groups:
        res["groups"] = by_group(c, zero, score, pos, neg)
    return res


# --- ECAPA em janelas deslizantes ------------------------------------------------------------------------------------


def load_ecapa(data_dir: Path, device: str):
    from cctrain.m1.encoder import load_encoder

    enc = load_encoder(data_dir / "models" / "spkrec-ecapa-voxceleb", device=device)
    return enc.fbank, enc.ecapa


@torch.no_grad()
def window_embeddings(fbank, ecapa, x16: torch.Tensor, win_s: float, hop_s: float = ECAPA_HOP_S,
                      chunk: int = 192, amp: bool = True) -> torch.Tensor:
    """x16 [B,N] (16 kHz) -> embeddings L2 [B,K,192] das janelas causais que terminam em t_k = k·hop (k=1..K),
    cobrindo [max(0, t_k − W), t_k] (janelas parciais no início)."""
    b, n = x16.shape
    hop = int(round(hop_s * ECAPA_SR))
    w = int(round(win_s * ECAPA_SR))
    ends = list(range(hop, n + 1, hop))
    out = torch.zeros(b, len(ends), 192, device=x16.device)
    by_len: dict[int, list[int]] = {}
    for k, e in enumerate(ends):
        by_len.setdefault(min(e, w), []).append(k)
    for ln, ks in by_len.items():
        segs = torch.stack([x16[:, ends[k] - ln: ends[k]] for k in ks], dim=1).reshape(-1, ln)  # [B*len(ks), ln]
        embs = []
        for i in range(0, segs.shape[0], chunk):
            feats = fbank(segs[i: i + chunk].float())  # front-end em fp32 (log de silêncio digital em fp16 dá NaN)
            with torch.autocast(x16.device.type, dtype=torch.float16, enabled=amp and x16.is_cuda):
                e = ecapa(feats)
            e = e.float()
            embs.append(e / torch.sqrt((e * e).sum(-1, keepdim=True) + 1e-12))
        embs = torch.cat(embs).reshape(b, len(ks), -1)
        out[:, ks] = embs
    return out


def cos_per_frame(emb_win: torch.Tensor, profile: torch.Tensor, frames: int, hop_s: float = ECAPA_HOP_S):
    """Cosseno janela×perfil [B,K] -> por quadro de 10 ms [B,frames], causal: a janela que termina em t_k vale para os
    quadros [t_k, t_(k+1)); os quadros antes da 1ª janela usam a 1ª."""
    cos = (emb_win * profile[:, None, :]).sum(-1)  # [B,K]
    per = int(round(hop_s / FRAME_S))
    idx = torch.clamp(torch.arange(frames, device=cos.device) // per - 1, min=0, max=cos.shape[1] - 1)
    return cos[:, idx]


def energy_weight(frame_db: np.ndarray | torch.Tensor):
    """Peso de energia do quadro da mistura em [0,1]: 0 a −75 dBFS, 1 a −30 dBFS (absoluto, causal)."""
    if isinstance(frame_db, torch.Tensor):
        return torch.clamp((frame_db + 75.0) / 45.0, 0.0, 1.0)
    return np.clip((frame_db + 75.0) / 45.0, 0.0, 1.0)


def frame_db_t(x48: torch.Tensor, frames: int, hop: int = 480) -> torch.Tensor:
    fr = x48[:, : frames * hop].reshape(x48.shape[0], frames, hop).double()
    return (10 * torch.log10((fr * fr).mean(-1) + 1e-12)).float()


# --- cabeças ---------------------------------------------------------------------------------------------------------


class CausalTCN(nn.Module):
    """Contexto causal ~1,3 s: convs dilatadas (k=3, d=1..32, campo receptivo 127 quadros) com resíduo."""

    def __init__(self, dim: int, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, 3, dilation=d) for d in dilations)
        self.dil = dilations

    def forward(self, x):  # [B,T,D]
        y = x.transpose(1, 2)
        for conv, d in zip(self.convs, self.dil):
            y = y + torch.relu(conv(F.pad(y, (2 * d, 0))))
        return y.transpose(1, 2)


class Head(nn.Module):
    """Generalização da `PresenceHead`: z = relu(W·x), s = W_s·e, [z, z⊙s, extra] -> (TCN) -> GRU -> logit."""

    def __init__(self, in_dim=512, emb_dim=192, proj=64, hidden=32, layers=1, ctx=False, extra=0, init_p=0.9,
                 use_trunk=True):
        super().__init__()
        self.use_trunk = use_trunk
        d = 0
        if use_trunk:
            self.inp = nn.Linear(in_dim, proj)
            self.spk = nn.Linear(emb_dim, proj)
            d = 2 * proj
        d += extra
        self.tcn = CausalTCN(d) if ctx else None
        self.gru = nn.GRU(d, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, math.log(init_p / (1 - init_p)))

    def forward(self, x, emb, extra=None):
        parts = []
        if self.use_trunk:
            z = torch.relu(self.inp(x))
            parts += [z, z * self.spk(emb).unsqueeze(1)]
        if extra is not None:
            parts.append(extra)
        h = torch.cat(parts, -1)
        if self.tcn is not None:
            h = self.tcn(h)
        y, _ = self.gru(h)
        return self.out(y).squeeze(-1)


def head_specs() -> dict[str, dict]:
    """Variantes treinadas juntas (mesmos lotes, tronco v3.1 congelado)."""
    return {
        "h32": dict(proj=64, hidden=32),
        "h128x2": dict(proj=128, hidden=128, layers=2),
        "h128x2_ctx1s": dict(proj=128, hidden=128, layers=2, ctx=True),
        "comb_h32": dict(proj=64, hidden=32, extra=3),
        "comb_h128x2_ctx1s": dict(proj=128, hidden=128, layers=2, ctx=True, extra=3),
        "ecapa_only_gru16": dict(hidden=16, extra=3, use_trunk=False),
    }


def extra_feats(cos1: torch.Tensor, mix_db: torch.Tensor) -> torch.Tensor:
    """[cos 1 s, cos·peso de energia, energia normalizada] [B,T,3]."""
    w = energy_weight(mix_db)
    return torch.stack([cos1, cos1 * w, (mix_db + 50.0) / 25.0], -1)
