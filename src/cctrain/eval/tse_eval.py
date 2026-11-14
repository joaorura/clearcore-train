"""Harness de avaliação do pDFNet3 Pro (plano v4 §8): roda um checkpoint quadro a quadro com o gating do runtime, mede
as métricas por item, agrega por célula e confere os 11 critérios de aceite.

Gating (`GATINGS`): `current` = estágios de hoje (ZERO < -10 < ERB+DF <= 20 < ERB-only <= 30 < passthrough); `new` =
plano §3.3: com perfil (`speaker_present=1`) só ZERO se lsnr < -10, senão ERB+DF (sem ERB-only nem passthrough). Sem
perfil o gating é sempre o atual (comportamento DFNet3).
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cctrain.eval import runtime_path as rp
from cctrain.eval import tse_matrix as tm
from cctrain.eval.functional import si_sdr

CURRENT = (np.float32(-10.0), np.float32(30.0), np.float32(20.0))
NEW_PROFILE = (np.float32(-10.0), np.float32("inf"), np.float32("inf"))
GATINGS = {"current": CURRENT, "new": NEW_PROFILE}

ARCHS = {  # num_tokens, token_dim, proj_dim (contrato v2 = K8/D64; v3 e v4 = K16/D128; v4p = v4 + cabeça p_target)
    "v2": dict(query_dim=512, embedding_dim=192, proj_dim=64, num_heads=4, num_tokens=8, token_dim=64),
    "v3": dict(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4, num_tokens=16, token_dim=128),
    "v4": dict(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4, num_tokens=16, token_dim=128),
    "v4p": dict(query_dim=512, embedding_dim=192, proj_dim=128, num_heads=4, num_tokens=16, token_dim=128),
}


@dataclass(frozen=True)
class Gating:
    """Gating COM perfil. `lsnr`: estágios por lsnr (`thr`, como GATINGS); `hard`: presença p_target pela máquina de
    estados do runtime (τ_on/τ_off/hangover) -> ERB+DF ou ZERO; `soft`: sempre ERB+DF (sem ZERO, sem passthrough) e
    ganho de saída por quadro g = floor + (1-floor)·s, s = p_target suavizado (ataque/liberação de um polo)."""

    kind: str = "lsnr"
    thr: tuple = NEW_PROFILE
    tau_on: float = 0.3
    tau_off: float = 0.1
    hangover: int = 12
    floor_db: float = -20.0
    attack_ms: float = 15.0
    release_ms: float = 200.0
    lookahead_ms: float = 0.0  # soft: p_target := máximo em [t, t+L] antes da suavização (saída atrasada L ms)
    window_ms: float = 0.0  # kind `window`: mediana de p_target em janelas fixas (alinhadas ao início), saída atrasada W

    @staticmethod
    def of(name: str) -> "Gating":
        return Gating("lsnr", GATINGS[name])


def lookahead_max(p: np.ndarray, frames: int) -> np.ndarray:
    """p'[t] = max(p[t .. t+frames]) (não causal: exige `frames` quadros de atraso na saída)."""
    p = np.asarray(p, np.float64)
    if frames <= 0:
        return p
    pad = np.concatenate([p, np.full(frames, p[-1] if len(p) else 0.0)])
    return np.max(np.lib.stride_tricks.sliding_window_view(pad, frames + 1), axis=1)[: len(p)]


def window_gain(p: np.ndarray, floor_db: float, window_ms: float, hop_ms: float = 10.0) -> np.ndarray:
    """Decisão por janela fixa: g = floor + (1-floor)·mediana(p) da janela, aplicado à janela inteira."""
    floor = 10.0 ** (floor_db / 20.0)
    w = max(int(round(window_ms / hop_ms)), 1)
    out = np.empty(len(p), np.float32)
    for a in range(0, len(p), w):
        out[a : a + w] = floor + (1.0 - floor) * float(np.median(p[a : a + w]))
    return out


def soft_gain(p: np.ndarray, floor_db: float, attack_ms: float, release_ms: float, hop_ms: float = 10.0,
              lookahead_ms: float = 0.0) -> np.ndarray:
    """g[t] = floor + (1 - floor)·s[t]; s = p suavizado causalmente (sobe com `attack_ms`, desce com `release_ms`;
    constantes de tempo de um polo). Estado inicial s=0 (o runtime parte de 'ausente')."""
    floor = 10.0 ** (floor_db / 20.0)
    a_up, a_dn = (1.0 - math.exp(-hop_ms / max(ms, 1e-3)) for ms in (attack_ms, release_ms))
    s, out = 0.0, np.empty(len(p), np.float32)
    for t, v in enumerate(lookahead_max(p, int(round(lookahead_ms / hop_ms)))):
        s += (a_up if v > s else a_dn) * (v - s)
        out[t] = floor + (1.0 - floor) * s
    return out


def presence_pass(p: np.ndarray, tau_on: float, tau_off: float, hangover: int) -> np.ndarray:
    """Mesma máquina de estados de `train_pro_v4_presence.presence_state_machine` (histerese + hangover)."""
    active, hang, out = False, 0, np.zeros(len(p), bool)
    for i, v in enumerate(np.asarray(p, np.float64)):
        active = v >= (tau_off if active else tau_on)
        hang = hangover if active else max(hang - 1, 0)
        out[i] = active or hang > 0
    return out


# --- modelo -------------------------------------------------------------------------------------------------------


def build_model(ckpt: Path, arch: str, device: str):
    from cctrain.m2.config import load_stage_f_config
    from cctrain.models.cross_attention import SubbandCrossAttentionConfig
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.models.film import FilmConfig, FilmGenerator
    from cctrain.models.pdfnet3_pro import PDFNet3Pro
    from cctrain.train.checkpoint import load_checkpoint
    from cctrain.train.stage_f import StageFHParams

    base, _ = load_upstream_dfnet3(device=device)
    hp = StageFHParams.from_config(load_stage_f_config())
    gen = FilmGenerator(FilmConfig(log_gamma_bound=hp.log_gamma_bound, beta_bound=hp.beta_bound)).to(device)
    m = PDFNet3Pro(base, gen, cross_attention_config=SubbandCrossAttentionConfig(**ARCHS[arch]),
                   enable_cross_attention=True, presence_head=(arch == "v4p")).to(device)
    ck = load_checkpoint(Path(ckpt))
    if ck.extra.get("adapter"):  # pro-v5: adaptadores LoRA/FiLM enrolados na base (ativos só com perfil)
        from cctrain.models.adapters import AdapterConfig, attach_adapters

        attach_adapters(m, AdapterConfig.from_dict(ck.extra["adapter"]))
    if ck.extra.get("lsnr_corr"):  # pro-v5-keep: correção do lsnr pelo perfil
        from cctrain.models.pdfnet3_pro import attach_lsnr_correction

        attach_lsnr_correction(m, float(ck.extra["lsnr_corr"]))
    m.load_state_dict(ck.model, strict=True)
    return m.eval()


class ProfileFrameModel(rp.TorchFrameModel):
    """`TorchFrameModel` com o forward completo do PDFNet3Pro (FiLM + freq_mixer + cross-attention), mask=1.

    O encoder roda uma vez (`self.p` = p_target por quadro, se o modelo tem a cabeça); os decoders rodam em
    `set_gating`, só nos quadros que o gating manda decodificar (como o runtime, que não avança o decoder parado).
    Para presença/soft o lsnr entregue à réplica é sintético (0 = ERB+DF, -100 = ZERO) com limiares (-10, inf, inf).
    """

    def __init__(self, model, embedding, audio, thresholds=None):
        _, self.feat_erb, self.feat_spec = rp.replica_features(audio)
        self.lsnr = np.zeros(0, np.float32)
        self.lsnr_true = self.lsnr
        self.p = np.zeros(0, np.float32)
        self.gains, self.coefs, self.t = {}, {}, 0
        self.n = len(self.feat_erb)
        if not self.n:
            return
        dev = next(model.parameters()).device
        base = model.base
        self._model = model
        emb_spk = torch.from_numpy(np.asarray(embedding, np.float32))[None].to(dev)
        m = torch.ones((1, 1), device=dev)
        ferb = torch.from_numpy(np.ascontiguousarray(self.feat_erb))[None, None].to(dev)
        fspec = torch.from_numpy(np.stack([self.feat_spec.real, self.feat_spec.imag], 0).astype(np.float32))[None].to(dev)
        if getattr(model, "adapter_cfg", None) is not None:  # pro-v5: liga os adaptadores (perfil ligado)
            from cctrain.models.adapters import set_gate

            set_gate(model, m.reshape(-1), emb_spk)
        with torch.no_grad():
            self._v = model.film(emb_spk, m)
            ferb = ferb + (model.freq_mixer(ferb) - ferb) * m.view(-1, 1, 1, 1)
            e0, e1, e2, e3, emb, c0, lsnr = base.enc(ferb, fspec, film=(self._v.gamma_enc, self._v.beta_enc))
            a = model.cross_attn(emb, emb_spk, mask=m)
            if getattr(model, "lsnr_corr", None) is not None:  # pro-v5-keep
                lsnr = lsnr + model.lsnr_corr(a.output) * (model.lsnr_corr_scale * m.reshape(-1, 1, 1))
            self._enc = (e0, e1, e2, e3, c0, a.output, a.subband_gains[0, 0])
            self.lsnr_true = lsnr.reshape(-1).cpu().numpy().astype(np.float32)
            if getattr(model, "presence", None) is not None:
                logit, _ = model.presence(a.output, emb_spk)
                self.p = torch.sigmoid(logit.float()).reshape(-1).cpu().numpy().astype(np.float32)

    def set_gating(self, g: "Gating") -> tuple:
        """Calcula os decoders para o gating `g`. Devolve `(thresholds para run_replica, frame_gain ou None)`."""
        self.gains, self.coefs, self.t = {}, {}, 0
        if not self.n:
            return g.thr if g.kind == "lsnr" else NEW_PROFILE, None
        frame_gain = None
        if g.kind == "lsnr":
            thr, self.lsnr = g.thr, self.lsnr_true
            flags = [rp.apply_stages(x, thr) for x in self.lsnr]
        else:
            if not len(self.p):
                raise ValueError("gating de presença exige modelo com cabeça p_target (arch v4p)")
            thr = NEW_PROFILE
            if g.kind == "hard":
                ok = presence_pass(self.p, g.tau_on, g.tau_off, g.hangover)
            else:  # soft / window: nunca ZERO
                ok = np.ones(self.n, bool)
                frame_gain = window_gain(self.p, g.floor_db, g.window_ms) if g.kind == "window" else \
                    soft_gain(self.p, g.floor_db, g.attack_ms, g.release_ms, lookahead_ms=g.lookahead_ms)
            self.lsnr = np.where(ok, 0.0, -100.0).astype(np.float32)
            flags = [rp.apply_stages(x, thr) for x in self.lsnr]
        dev = next(self._model.parameters()).device
        base = self._model.base
        e0, e1, e2, e3, c0, emb_d, sg = self._enc
        erb_idx = [t for t, f in enumerate(flags) if f[0]]
        df_idx = [t for t, f in enumerate(flags) if f[2]]
        with torch.no_grad():
            if erb_idx:
                i = torch.tensor(erb_idx, device=dev)
                mm = base.erb_dec(emb_d[:, i], e3[:, :, i], e2[:, :, i], e1[:, :, i], e0[:, :, i]) * sg[i][None, None]
                mm = mm.cpu().numpy()
                self.gains = {t: mm[0, 0, k].astype(np.float32) for k, t in enumerate(erb_idx)}
            if df_idx:
                i = torch.tensor(df_idx, device=dev)
                c = base.df_dec(emb_d[:, i], c0[:, :, i], film=(self._v.gamma_df, self._v.beta_df))
                c = c[0].cpu().numpy().reshape(len(df_idx), rp.NB_DF, rp.DF_ORDER, 2)
                self.coefs = {t: (c[k, ..., 0] + 1j * c[k, ..., 1]).astype(np.complex64) for k, t in enumerate(df_idx)}
        return thr, frame_gain


class Runner:
    """Roda uma mistura pelo caminho do runtime. `profile=None` = sem perfil (DFNet3 base, gating atual).
    `gating`: nome de GATINGS ('current'/'new') ou um `Gating` (hard/soft de presença exigem arch v4p)."""

    def __init__(self, ckpt: Path, arch: str, gating: "str | Gating", device: str):
        self.model = build_model(ckpt, arch, device)
        self.base_cpu = copy.deepcopy(self.model.base).to("cpu").eval()
        if getattr(self.model, "adapter_cfg", None) is not None:  # sem perfil = adaptadores desligados
            from cctrain.models.adapters import set_gate

            set_gate(self.base_cpu, None)
        self.gating = Gating.of(gating) if isinstance(gating, str) else gating
        self.thr = self.gating.thr

    @staticmethod
    def _padded(noisy: np.ndarray) -> tuple[np.ndarray, int]:
        n = np.asarray(noisy).size
        # o runtime atrasa 1.440 amostras: sem cauda de silêncio os últimos 3 hops da saída se perderiam e o SI-SDR
        # do passthrough puro cairia para ~14 dB. A cauda é ruído a -55 dBFS (silêncio digital seria pulado pelo atalho
        # do runtime e não empurraria a saída) e sai do cômputo (estágios cortados em n // HOP).
        tail = np.random.default_rng(0).standard_normal(4 * rp.HOP).astype(np.float32) * np.float32(10 ** (-55 / 20))
        return np.concatenate([np.asarray(noisy, np.float32).reshape(-1), tail]), n

    def run(self, noisy: np.ndarray, profile: np.ndarray | None, shadow: list[np.ndarray] | None = None) -> tuple:
        """-> (saída, estágios) ou, com `shadow` (componentes da entrada), (saída, estágios, [saída-sombra de cada um])."""
        audio, n = self._padded(noisy)
        if profile is not None:
            return self.run_multi(noisy, profile, {"_": self.gating}, shadow=shadow)["_"]
        res = rp.run_replica(audio, rp.TorchFrameModel(self.base_cpu, None, audio, CURRENT), CURRENT,
                             shadow=_pad_shadow(shadow, audio.size))
        return _result(res, n, shadow is not None)

    def run_multi(self, noisy: np.ndarray, profile: np.ndarray, gatings: dict[str, Gating],
                  shadow: list[np.ndarray] | None = None) -> dict[str, tuple]:
        """Vários gatings sobre o MESMO encoder (o forward do encoder é o custo; os decoders rodam por gating)."""
        audio, n = self._padded(noisy)
        pm = ProfileFrameModel(self.model, profile, audio)
        out = {}
        for name, g in gatings.items():
            thr, fg = pm.set_gating(g)
            res = rp.run_replica(audio, pm, thr, frame_gain=fg, shadow=_pad_shadow(shadow, audio.size))
            out[name] = _result(res, n, shadow is not None)
        return out


def _pad_shadow(shadow: list[np.ndarray] | None, size: int) -> list[np.ndarray] | None:
    if shadow is None:
        return None
    return [np.pad(np.asarray(x, np.float32).reshape(-1), (0, max(0, size - np.asarray(x).size)))[:size] for x in shadow]


def _result(res: rp.ReplicaResult, n: int, with_shadow: bool) -> tuple:
    out = (rp.align_runtime(res.audio, n), res.stages[: n // rp.HOP])
    if with_shadow:
        out = out + ([rp.align_runtime(x, n) for x in res.shadow],)
    return out


class BaseRunner(Runner):
    """DFNet3 publicado (upstream, sem perfil e sem isolamento), gating atual: referência de remoção de ruído."""

    def __init__(self, device: str = "cpu"):
        from cctrain.models.dfnet3_upstream import load_upstream_dfnet3

        base, _ = load_upstream_dfnet3(device="cpu")
        self.model = None
        self.base_cpu = base.eval()
        self.gating = Gating.of("current")
        self.thr = self.gating.thr


# --- métricas ------------------------------------------------------------------------------------------------------

ZERO_STAGES = (int(rp.Stage.ZERO), int(rp.Stage.SILENT))


def _db(a: float, b: float) -> float:
    return float(10 * np.log10((a + 1e-20) / (b + 1e-20)))


def lsq_coefs(est: np.ndarray, tracks: list[np.ndarray]) -> np.ndarray:
    A = np.stack(tracks, 1).astype(np.float64)
    co, *_ = np.linalg.lstsq(A, np.asarray(est, np.float64), rcond=None)
    return co


def interf_suppression_db(est: np.ndarray, noisy: np.ndarray, tgt: np.ndarray, ints: np.ndarray) -> tuple[float, float]:
    """(supressão do interferente dB, ganho do alvo dB) por projeção de mínimos quadrados da saída sobre as trilhas.
    Supressão = energia ponderada dos coeficientes dos interferentes na entrada contra a da saída."""
    tracks = [tgt] + list(ints)
    en = np.array([float(np.sum(x.astype(np.float64) ** 2)) for x in ints])
    co_o, co_i = lsq_coefs(est, tracks), lsq_coefs(noisy, tracks)

    def leak(co):
        return float(np.sum(co[1:] ** 2 * en) / np.sum(en))

    return _db(leak(co_i), leak(co_o)), float(20 * np.log10(max(abs(co_o[0]), 1e-4)))


def frame_mask_samples(mask: np.ndarray, n: int) -> np.ndarray:
    m = np.repeat(mask, tm.HOP)
    return np.pad(m, (0, max(0, n - m.size)))[:n]



# --- métricas por quadro/segmento (gating suave): atenuação em dB por quadros de 10 ms e segmentos de 300 ms ------------

SEG_FRAMES = 30  # 300 ms
ATT_CLIP_DB = 60.0
HI_REL_DB = 20.0  # "quadros com energia do alvo >= -20 dB do máximo"


def frame_energy(x: np.ndarray, nf: int) -> np.ndarray:
    return (np.asarray(x[: nf * tm.HOP], np.float64).reshape(nf, tm.HOP) ** 2).sum(axis=1)


def att_db(e_in: np.ndarray | float, e_out: np.ndarray | float):
    """Atenuação em dB (entrada/saída), limitada a ±60 dB (saída zerada não vira +inf)."""
    return np.clip(10 * np.log10((np.asarray(e_in) + 1e-12) / (np.asarray(e_out) + 1e-12)), -ATT_CLIP_DB, ATT_CLIP_DB)


def masked_target_gain_db(est: np.ndarray, t: np.ndarray, ints: np.ndarray, m: np.ndarray) -> float:
    """Ganho do alvo (dB) por projeção de mínimos quadrados SÓ nas amostras de `m` (trilhas: alvo e interferentes)."""
    tracks = [t[m]] + [x[m] for x in ints]
    return float(20 * np.log10(max(abs(lsq_coefs(est[m], tracks)[0]), 1e-4)))


def frame_group_metrics(prefix: str, att: np.ndarray, m: np.ndarray, out: dict) -> None:
    """Atenuação por quadro nos quadros de `m`: contagem, soma (para média ponderada) e ≥10/≥20 dB."""
    a = att[m]
    out[f"{prefix}_n"] = int(a.size)
    out[f"{prefix}_att_sum"] = float(a.sum())
    out[f"{prefix}_ge10"] = int((a >= 10).sum())
    out[f"{prefix}_ge20"] = int((a >= 20).sum())


ONSET_FRAMES = 10  # 100 ms


def onset_metrics(nf: int, enh: np.ndarray, t: np.ndarray, ints: np.ndarray, tact: np.ndarray, out: dict) -> None:
    """Início de trecho do alvo: quadro com alvo ativo e nenhum alvo ativo nos 100 ms anteriores (silêncio ou só-
    interferente antes; o início do item não conta). Ganho do alvo por mínimos quadrados nos 100 ms seguintes (alvo e
    interferentes como regressores), por início; `onset_gt3` = inícios atenuados > 3 dB; `onset_gain_sum` soma dos ganhos
    (limitados a [-40, +10] dB)."""
    n = hit = 0
    tot = 0.0
    for f in range(ONSET_FRAMES, nf - ONSET_FRAMES):
        if tact[f] and not tact[f - ONSET_FRAMES : f].any():
            m = np.zeros(len(t), bool)
            m[f * tm.HOP : (f + ONSET_FRAMES) * tm.HOP] = True
            g = masked_target_gain_db(enh, t, ints, m)
            n += 1
            hit += int(g < -3.0)
            tot += float(np.clip(g, -40.0, 10.0))
    out["onset_n"], out["onset_gt3"], out["onset_gain_sum"] = n, hit, tot


def segment_metrics(view: str, nf: int, noisy: np.ndarray, enh: np.ndarray, t: np.ndarray, ints: np.ndarray,
                    tact: np.ndarray, iact: np.ndarray, out: dict) -> None:
    """Segmentos de 300 ms sem sobreposição. Só-interferente: sem alvo ativo e interferente ativo em >= 50% dos quadros
    (na vista `int`, sem alvo na mistura, basta o interferente); atenuação = energia de entrada/saída do segmento
    (conta `seg_oi_ge15`). Segmento com alvo: alvo ativo em >= 50% dos quadros; atenuação do alvo pelo ganho de mínimos
    quadrados no segmento (alvo e interferentes como regressores), contado `seg_t_gt3` quando > 3 dB."""
    e_in, e_out = frame_energy(noisy, nf), frame_energy(enh, nf)
    n_oi = hit_oi = n_t = hit_t = 0
    for k in range(nf // SEG_FRAMES):
        sl = slice(k * SEG_FRAMES, (k + 1) * SEG_FRAMES)
        has_t = view != "int" and tact[sl].mean() >= 0.5
        if (view == "int" or not tact[sl].any()) and iact[sl].mean() >= 0.5:
            n_oi += 1
            hit_oi += int(att_db(e_in[sl].sum(), e_out[sl].sum()) >= 15.0)
        elif has_t:
            a, b = k * SEG_FRAMES * tm.HOP, (k + 1) * SEG_FRAMES * tm.HOP
            m = np.zeros(len(t), bool)
            m[a:b] = True
            n_t += 1
            hit_t += int(-masked_target_gain_db(enh, t, ints, m) > 3.0)
    key = "io" if view == "int" else "oi"
    out[f"seg_{key}_n"], out[f"seg_{key}_ge15"] = n_oi, hit_oi
    if view != "int":
        out["seg_t_n"], out["seg_t_gt3"] = n_t, hit_t


# --- qualidade do alvo: picote (dropouts), instabilidade do ganho, ESTOI ativo, ruído ---------------------------------
#
# Todas sobre a SAÍDA-SOMBRA do alvo (`run_replica(shadow=...)`): o alvo limpo processado com as mesmas decisões que o
# modelo tomou sobre a mistura. A soma das sombras reproduz a saída (-136 dB de erro), então o ganho do alvo por quadro é
# exato e não depende de projeção (sem o viés de fase/coloração dos mínimos quadrados).

Q_LOCAL_FRAMES = 50  # máximo local do nível do alvo em ±500 ms (101 quadros)
Q_ACT_REL_DB = 30.0  # quadro ativo: nível do alvo >= máximo local - 30 dB (e > VAD_FLOOR_DB)
Q_DROP_REL_DB = 10.0  # quadro caído: ganho <= mediana do item - 10 dB ...
Q_DROP_ABS_DB = -10.0  # ... ou <= -10 dB absoluto (limiar = max(mediana - 10, -10))
Q_DROP_MIN, Q_DROP_MAX = 2, 30  # dropout = 20..300 ms contíguos de quadros ativos caídos; > 300 ms = supressão longa
Q_GAIN_CLIP = (-60.0, 20.0)


def target_active_local(t: np.ndarray, nf: int) -> np.ndarray:
    """Quadros (10 ms) com alvo ativo pelo critério LOCAL: nível >= máximo em ±500 ms - 30 dB e acima do piso do VAD.
    O critério local (e não o pico do item) não descarta frases mais baixas do mesmo item."""
    lv = tm.frame_db(t)[:nf]
    if not lv.size:
        return np.zeros(0, bool)
    pad = np.pad(lv, Q_LOCAL_FRAMES, constant_values=-np.inf)
    loc = np.max(np.lib.stride_tricks.sliding_window_view(pad, 2 * Q_LOCAL_FRAMES + 1), axis=1)
    return (lv >= loc - Q_ACT_REL_DB) & (lv > tm.VAD_FLOOR_DB)


def frame_gain_db(src: np.ndarray, out: np.ndarray, nf: int) -> np.ndarray:
    return np.clip(10 * np.log10((frame_energy(out, nf) + 1e-14) / (frame_energy(src, nf) + 1e-14)), *Q_GAIN_CLIP)


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Trechos contíguos [a, b) de `True`."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[0::2].tolist(), d[1::2].tolist()))


def dropouts(g: np.ndarray, act: np.ndarray) -> dict:
    """Dropouts do alvo. `g` ganho por quadro (dB), `act` quadros ativos. Quadro caído = ativo e g <= max(mediana(g[act])
    - 10, -10). Trechos contíguos de quadros caídos (um quadro inativo interrompe o trecho): 2..30 quadros (20..300 ms) =
    dropout; > 30 = supressão longa; 1 quadro = ignorado (abaixo da resolução da STFT de 20 ms)."""
    ga = g[act]
    med = float(np.median(ga))
    thr = max(med - Q_DROP_REL_DB, Q_DROP_ABS_DB)
    drop = act & (g <= thr)
    ev, long_f = [], 0
    for a, b in runs_of(drop):
        if Q_DROP_MIN <= b - a <= Q_DROP_MAX:
            ev.append((a, b, float(g[a:b].min())))
        elif b - a > Q_DROP_MAX:
            long_f += b - a
    pairs = act[1:] & act[:-1]
    return {"med": med, "thr": thr, "events": ev, "drop_frames": sum(b - a for a, b, _ in ev), "long_frames": long_f,
            "p10": float(np.percentile(ga, 10)), "std": float(np.std(ga)),
            "dstep": float(np.mean(np.abs(np.diff(g))[pairs])) if pairs.any() else float("nan")}


def quality_metrics(view: str, r: tm.Rendered, noisy: np.ndarray, enh: np.ndarray, shadow: list[np.ndarray],
                    base: dict, out: dict, scorer=None) -> None:
    """Picote/instabilidade/ESTOI ativo do alvo e, com ruído, atenuação do ruído (sombras: [alvo, ruído])."""
    from cctrain.eval.metrics import stoi

    t = r.target
    nf = len(t) // tm.HOP
    act = target_active_local(t, nf)
    if act.sum() < 20:
        return
    g = frame_gain_db(t, shadow[0], nf)
    d = dropouts(g, act)
    out["q_act_frames"] = int(act.sum())
    out["q_drop_n"] = len(d["events"])
    out["q_drop_frames"] = d["drop_frames"]
    out["q_long_frames"] = d["long_frames"]
    for k in ("med", "p10", "std", "dstep"):
        out[f"q_g_{k}"] = d[k]
    if r.interf.shape[0]:
        # quadros ativos do alvo SEM interferente ativo (alvo sozinho) e COM interferente (sobreposição)
        drop_f = np.zeros(nf, bool)
        for a, b, _ in d["events"]:
            drop_f[a:b] = True
        for key, m in (("tonly", act & ~r.int_act[:nf]), ("both", act & r.int_act[:nf])):
            if m.sum() >= 10:
                out[f"q_{key}_frames"] = int(m.sum())
                out[f"q_{key}_g_med"] = float(np.median(g[m]))
                out[f"q_{key}_g_p10"] = float(np.percentile(g[m], 10))
                out[f"q_{key}_drop_frames"] = int((drop_f & m).sum())
                out[f"q_{key}_low10_frames"] = int((g[m] <= -10.0).sum())
    ms = frame_mask_samples(act, len(t)).astype(bool)
    if ms.sum() >= tm.SR:  # >= 1 s de fala ativa concatenada
        e_out = float(stoi(enh[ms], t[ms], tm.SR, extended=True)[0])
        out["estoi_act"] = e_out
        if "estoi_in_act" in base:
            out["d_estoi_act"] = e_out - base["estoi_in_act"]
    if "estoi_in" in base and "d_estoi" not in out:
        out["d_estoi"] = float(stoi(enh, t, tm.SR, extended=True)[0]) - base["estoi_in"]
    if scorer is not None:
        from cctrain.data.resample import to_16k

        out.update({f"dnsmos_{k}": v for k, v in scorer.score(to_16k(enh.astype(np.float32), tm.SR)).items()})
    nz = r.noise
    if len(shadow) > 1 and float(np.sum(nz.astype(np.float64) ** 2)) > 0:
        en_in, en_out = frame_energy(nz, nf), frame_energy(shadow[1], nf)
        ok = en_in > 1e-12
        sil = ~tm.activity(t)[:nf] & ~act & ok
        if r.interf.shape[0]:
            sil &= ~r.int_act[:nf]
        a_n = np.clip(10 * np.log10((en_in + 1e-14) / (en_out + 1e-14)), -ATT_CLIP_DB, ATT_CLIP_DB)
        if sil.sum() >= 10:
            out["n_att_sil_mean"] = float(a_n[sil].mean())
            out["n_att_sil_p10"] = float(np.percentile(a_n[sil], 10))
            out["n_sil_frames"] = int(sil.sum())
        sp = act & ok
        if sp.sum() >= 10:
            # ruído residual durante a fala: atenuação do ruído (sombra do ruído) e SNR de saída nesses quadros
            out["n_att_speech"] = _db(float(en_in[sp].sum()), float(en_out[sp].sum()))
            et = frame_energy(shadow[0], nf)
            out["snr_out_speech"] = _db(float(et[sp].sum()), float(en_out[sp].sum()))
            out["snr_in_speech"] = _db(float(frame_energy(t, nf)[sp].sum()), float(en_in[sp].sum()))


def quality_baseline(r: tm.Rendered, noisy: np.ndarray, base: dict) -> dict:
    """Acrescenta a `base` o ESTOI da entrada só nas regiões com alvo ativo (mesma máscara de `quality_metrics`)."""
    from cctrain.eval.metrics import stoi

    t = r.target
    act = target_active_local(t, len(t) // tm.HOP)
    ms = frame_mask_samples(act, len(t)).astype(bool)
    if ms.sum() >= tm.SR:
        base["estoi_in_act"] = float(stoi(noisy[ms], t[ms], tm.SR, extended=True)[0])
    return base


def item_metrics(view: str, rec: dict, r: tm.Rendered, noisy: np.ndarray, enh: np.ndarray, stages: np.ndarray,
                 base: dict, shadow: list[np.ndarray] | None = None, scorer=None) -> dict:
    """Métricas de uma execução. `base`: STOI/ESTOI/SI-SDR da entrada (calculados uma vez por item). `shadow`: saídas-
    sombra [alvo, ruído] (habilita `quality_metrics`)."""
    from cctrain.eval.metrics import stoi

    out: dict = {}
    st = np.asarray(stages)
    zero = np.isin(st, ZERO_STAGES)
    zero_only = st == int(rp.Stage.ZERO)
    nf = min(len(st), len(r.tgt_act))
    tact, iact = r.tgt_act[:nf], r.int_act[:nf]
    n_int = r.interf.shape[0]
    out["stage_zero_pct"] = float(zero.mean() * 100)
    out["stage_zero_only_pct"] = float(zero_only.mean() * 100)
    out["stage_none_pct"] = float((st == int(rp.Stage.NONE)).mean() * 100)
    out["stage_erb_only_pct"] = float((st == int(rp.Stage.ERB_ONLY)).mean() * 100)
    if view == "int":
        e_in, e_out = float(np.sum(noisy.astype(np.float64) ** 2)), float(np.sum(enh.astype(np.float64) ** 2))
        out["leak_db"] = _db(e_out, e_in)
        if n_int:
            nfi = min(len(iact), len(noisy) // tm.HOP)
            frame_group_metrics("io", att_db(frame_energy(noisy, nfi), frame_energy(enh, nfi)), iact[:nfi], out)
            segment_metrics("int", nfi, noisy, enh, r.target, r.interf, tact, iact, out)
        return out
    t = r.target
    out["tgt_zero_pct"] = float(zero[:nf][tact].mean() * 100) if tact.any() else float("nan")
    out["tgt_zero_only_pct"] = float(zero_only[:nf][tact].mean() * 100) if tact.any() else float("nan")
    out["sisdr_out"] = si_sdr(enh, t)
    out["d_sisdr"] = out["sisdr_out"] - base["sisdr_in"]
    ms = frame_mask_samples(tact, len(t))
    if ms.sum() > 4800:
        mb = ms.astype(bool)
        out["d_sisdr_act"] = si_sdr(enh[mb], t[mb]) - si_sdr(noisy[mb], t[mb])
    out["d_stoi"] = float(stoi(enh, t, tm.SR)[0]) - base["stoi_in"]
    if view == "mix":
        out["d_estoi"] = float(stoi(enh, t, tm.SR, extended=True)[0]) - base["estoi_in"]
    if n_int:
        supp, gain = interf_suppression_db(enh, noisy, t, r.interf)
        out["interf_supp_db"], out["target_gain_db"] = supp, gain
        only_i = iact & ~tact
        if view == "mix" and only_i.sum() > 5:
            fm = frame_mask_samples(only_i, len(t)).astype(bool)
            out["leak_intonly_frames_db"] = _db(float(np.sum(enh[fm].astype(np.float64) ** 2)),
                                                float(np.sum(noisy[fm].astype(np.float64) ** 2)))
            out["int_only_zero_pct"] = float(zero[:nf][only_i].mean() * 100)
        if view == "mix" and rec["cell"]["axis"] == "temporal":
            for name, m in (("t_only", tact & ~iact), ("both", tact & iact), ("sil", ~tact & ~iact)):
                if m.sum() > 5:
                    out[f"{name}_zero_pct"] = float(zero[:nf][m].mean() * 100)
    else:
        out["target_gain_db"] = float(20 * np.log10(max(abs(lsq_coefs(enh, [t])[0]), 1e-4)))
    # --- gating suave: quadros de alvo forte (>= -20 dB do máximo), sobreposição, só-interferente, segmentos de 300 ms
    nf = min(len(st), len(r.tgt_act), len(t) // tm.HOP)
    ints = r.interf
    lv_t = tm.frame_db(t)[:nf]
    hi = lv_t >= lv_t.max() - HI_REL_DB if nf else np.zeros(0, bool)
    hi = hi & (lv_t > tm.VAD_FLOOR_DB) if nf else hi
    if hi.sum() >= 5:
        mh = frame_mask_samples(hi, len(t)).astype(bool)
        out["target_gain_hi_db"] = masked_target_gain_db(enh, t, ints, mh)
        out["d_sisdr_hi"] = si_sdr(enh[mh], t[mh]) - si_sdr(noisy[mh], t[mh])
    if n_int:
        both = tact[:nf] & iact[:nf]
        if view == "mix" and both.sum() >= 20:
            mb = frame_mask_samples(both, len(t)).astype(bool)
            out["supp_overlap_db"], out["tgt_gain_overlap_db"] = interf_suppression_db(enh[mb], noisy[mb], t[mb], ints[:, mb])
        if view == "mix":
            e_in, e_out = frame_energy(noisy, nf), frame_energy(enh, nf)
            only_i = iact[:nf] & ~tact[:nf]
            if only_i.sum() >= 5:
                frame_group_metrics("oi", att_db(e_in, e_out), only_i, out)
    segment_metrics(view, nf, noisy, enh, t, ints, tact, iact, out)
    onset_metrics(nf, enh, t, ints, tact, out)
    if shadow is not None:
        quality_metrics(view, r, noisy, enh, shadow, base, out, scorer)
    return out


def input_baseline(r: tm.Rendered, noisy: np.ndarray) -> dict:
    from cctrain.eval.metrics import stoi

    t = r.target
    return {"sisdr_in": si_sdr(noisy, t), "stoi_in": float(stoi(noisy, t, tm.SR)[0]),
            "estoi_in": float(stoi(noisy, t, tm.SR, extended=True)[0])}


def run_item(runner: Runner, rec: dict, emb_of: Callable[[dict], np.ndarray],
             swap_emb_of: Callable[[dict], np.ndarray | None]) -> list[dict]:
    """Todas as vistas da célula do item. Devolve uma linha por vista."""
    cell = tm.Cell(**{k: v for k, v in rec["cell"].items() if k != "cid"})
    r = tm.render(rec)
    rows = []
    noisy_mix = r.noisy(True)
    base = input_baseline(r, noisy_mix)
    prof = emb_of(rec)
    for view in cell.views():
        if view == "mix":
            enh, st = runner.run(noisy_mix, prof)
            noisy = noisy_mix
        elif view == "noprof":
            enh, st = runner.run(noisy_mix, None)
            noisy = noisy_mix
        elif view == "swap":
            enh, st = runner.run(noisy_mix, swap_emb_of(rec))
            noisy = noisy_mix
        else:  # int
            noisy = r.noisy(False)
            enh, st = runner.run(noisy, prof)
        row = {"id": rec["id"], "cid": rec["cell"]["cid"], "view": view, "target": rec["target"]["spk"],
               **item_metrics(view, rec, r, noisy, enh, st, base)}
        rows.append(row)
    return rows


# --- agregação -------------------------------------------------------------------------------------------------------


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cell_table(rows: Iterable[dict]) -> dict[tuple[str, str], dict]:
    g: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        g[(r["cid"], r["view"])].append(r)
    out = {}
    for k, rs in g.items():
        keys = {m for r in rs for m in r if m not in ("id", "cid", "view", "target")}
        agg = {"n": len(rs)}
        for m in sorted(keys):
            vals = np.array([r[m] for r in rs if m in r and r[m] is not None], float)
            vals = vals[np.isfinite(vals)]
            agg[m] = float(vals.mean()) if vals.size else float("nan")
            if m in ("leak_db", "leak_intonly_frames_db") and vals.size:
                agg[m + "_median"] = float(np.median(vals))
        out[k] = agg
    return out


def parse_cid(cid: str) -> dict:
    axis, n, sir, lv, noise, pat = cid.split("|")
    return {"axis": axis, "n": int(n[1:]), "sir": None if sir[3:] == "na" else float(sir[3:]),
            "level": float(lv[2:]), "noise": noise, "pattern": pat}


def bootstrap(vals: np.ndarray, B: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, v.size, (B, v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


@dataclass
class Table:
    cells: dict[tuple[str, str], dict]
    rows: list[dict]

    def sel(self, view: str, axis: str = "matrix", **kw) -> list[tuple[dict, dict]]:
        out = []
        for (cid, v), agg in self.cells.items():
            if v != view:
                continue
            p = parse_cid(cid)
            if p["axis"] != axis:
                continue
            if all((p[k] in val if isinstance(val, (tuple, list, set)) else p[k] == val) for k, val in kw.items()):
                out.append((p, agg))
        return sorted(out, key=lambda x: (x[0]["n"], x[0]["sir"] or 0, x[0]["level"], x[0]["noise"]))

    def pooled(self, view: str, metric: str, axis: str = "matrix", **kw) -> tuple[float, float, float]:
        cids = {tuple(sorted(p.items(), key=lambda kv: kv[0])) for p, _ in self.sel(view, axis, **kw)}
        vals = []
        for r in self.rows:
            if r["view"] != view or metric not in r or r[metric] is None:
                continue
            p = parse_cid(r["cid"])
            if p["axis"] == axis and tuple(sorted(p.items(), key=lambda kv: kv[0])) in cids:
                vals.append(r[metric])
        return bootstrap(np.array(vals, float))


def make_table(rows: list[dict]) -> Table:
    return Table(cell_table(rows), rows)


# --- critérios de aceite ---------------------------------------------------------------------------------------------

NOMINAL_LEVEL = -30.0


@dataclass
class CriterionResult:
    num: int
    name: str
    target: str
    status: str  # PASS | FAIL | N/A
    nominal: str  # valor/estado no nível nominal (-30 dBFS)
    all_levels: str
    detail: str = ""


def _check(cells: list[tuple[dict, dict]], fn: Callable[[dict], tuple[bool, float]], fmt: str = "{:+.2f}",
           worst: str = "min") -> tuple[str, str]:
    """(estado, pior valor). `fn(agg) -> (ok, valor)`; cél. sem dado = N/A."""
    oks, vals = [], []
    for _, a in cells:
        ok, v = fn(a)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        oks.append(ok)
        vals.append(v)
    if not oks:
        return "N/A", "-"
    w = min(vals) if worst == "min" else max(vals)
    return ("PASS" if all(oks) else "FAIL") + f" ({sum(oks)}/{len(oks)})", fmt.format(w)


def evaluate_criteria(t: Table, ref: Table | None) -> list[CriterionResult]:
    res: list[CriterionResult] = []

    def both(num, name, target, fn_cells, fn, fmt="{:+.2f}", worst="min", ref_needed=False):
        if ref_needed and ref is None:
            res.append(CriterionResult(num, name, target, "N/A", "sem referência (--ref)", "-"))
            return
        out = []
        for levels in ((NOMINAL_LEVEL,), tm.LEVELS):
            cells = fn_cells(levels)
            out.append(_check(cells, fn, fmt, worst))
        status = "PASS" if out[0][0].startswith("PASS") else ("FAIL" if out[0][0].startswith("FAIL") else "N/A")
        res.append(CriterionResult(num, name, target, status, f"{out[0][0]}; pior {out[0][1]}",
                                   f"{out[1][0]}; pior {out[1][1]}"))

    # 1: N=0, SNR 5: perfil >= sem perfil - 0.5 dB e dSTOI >= 0
    def c1_cells(levels):
        out = []
        for p, a in t.sel("mix", n=0, noise="snr5", level=tuple(levels)):
            npf = t.cells.get((tm.Cell("matrix", 0, None, p["level"], "snr5").cid, "noprof"), {})
            out.append((p, a | {"_noprof_d_sisdr": npf.get("d_sisdr", float("nan"))}))
        return out

    both(1, "N=0 (SNR 5): perfil não piora", "dSI-SDR(perfil) >= sem perfil - 0,5 dB e dSTOI >= 0",
         c1_cells, lambda a: (a["d_sisdr"] >= a["_noprof_d_sisdr"] - 0.5 and a["d_stoi"] >= 0,
                              min(a["d_sisdr"] - a["_noprof_d_sisdr"] + 0.5, a["d_stoi"] * 100)),
         fmt="{:+.2f} (margem; dSTOI em centésimos)")

    # 2: N=1, SIR 0, silencioso
    both(2, "N=1, SIR 0, silencioso", "dSI-SDR >= +4, dSTOI >= +0,03, ganho do alvo >= -3 dB, supressão >= 15 dB",
         lambda lv: t.sel("mix", n=1, sir=0.0, noise="silent", level=tuple(lv)),
         lambda a: (a["d_sisdr"] >= 4 and a["d_stoi"] >= 0.03 and a["target_gain_db"] >= -3 and a["interf_supp_db"] >= 15,
                    min(a["d_sisdr"] - 4, (a["d_stoi"] - 0.03) * 100, a["target_gain_db"] + 3, a["interf_supp_db"] - 15)),
         fmt="{:+.2f} (pior margem)")

    # 3: N=2,3, SIR 0 (todos os ruídos)
    both(3, "N=2 e N=3, SIR 0 por interferente", "dSI-SDR >= +3 dB e dSTOI >= 0",
         lambda lv: t.sel("mix", n=(2, 3), sir=0.0, level=tuple(lv)),
         lambda a: (a["d_sisdr"] >= 3 and a["d_stoi"] >= 0, min(a["d_sisdr"] - 3, a["d_stoi"] * 100)),
         fmt="{:+.2f} (pior margem)")

    # 4: só-interferente (o limiar de %ZERO depende do ruído)
    oks_nom, oks_all = [], []
    for levels, store in (((NOMINAL_LEVEL,), oks_nom), (tm.LEVELS, oks_all)):
        for nz, thr in (("silent", 90.0), ("snr5", 80.0)):
            for p, a in t.sel("int", n=(1, 2, 3), noise=nz, level=tuple(levels)):
                store.append((a["stage_zero_pct"] >= thr and a.get("leak_db_median", 0) <= -30, a["stage_zero_pct"],
                              a.get("leak_db_median", float("nan"))))
    def _s(store):
        if not store:
            return "N/A"
        ok = [x[0] for x in store]
        return (("PASS" if all(ok) else "FAIL") + f" ({sum(ok)}/{len(ok)}); pior %ZERO {min(x[1] for x in store):.1f}%, "
                f"pior vazamento {max(x[2] for x in store):+.1f} dB")
    res.append(CriterionResult(4, "Só-interferente (N=1..3, perfil)",
                               "%ZERO >= 90% (silencioso) / >= 80% (SNR 5); vazamento mediano <= -30 dB",
                               "PASS" if _s(oks_nom).startswith("PASS") else ("FAIL" if _s(oks_nom).startswith("FAIL") else "N/A"),
                               _s(oks_nom), _s(oks_all)))

    # 5: interferente fraco
    both(5, "Interferente fraco (SIR +20/+30, silencioso)", "supressão do interferente >= 15 dB",
         lambda lv: t.sel("mix", n=(1, 2, 3, 4), sir=(20.0, 30.0), noise="silent", level=tuple(lv)),
         lambda a: (a["interf_supp_db"] >= 15, a["interf_supp_db"] - 15), fmt="{:+.2f} dB (pior margem)")

    # 6: alvo zerado indevidamente (N=0, perfil), só decisão ZERO do modelo
    both(6, "Alvo zerado indevidamente (N=0)", "<= 2% dos quadros do alvo em ZERO, todo nível >= -45",
         lambda lv: t.sel("mix", n=0, level=tuple(lv)),
         lambda a: (a["tgt_zero_only_pct"] <= 2.0, a["tgt_zero_only_pct"]), fmt="{:.2f}% (pior)", worst="max")

    # 7: sem perfil vs v2 sem perfil (N=0, nível nominal, SNR 20 e 5)
    if ref is None:
        res.append(CriterionResult(7, "Sem perfil ~ v2 sem perfil", "dSI-SDR <= 0,2 dB e dSTOI <= 0,005 de diferença",
                                   "N/A", "sem referência (--ref)", "-"))
    else:
        oks, w = [], []
        for nz in ("snr20", "snr5"):
            cid = tm.Cell("matrix", 0, None, NOMINAL_LEVEL, nz).cid
            a, b = t.cells.get((cid, "noprof")), ref.cells.get((cid, "noprof"))
            if a and b:
                dd, ds = abs(a["d_sisdr"] - b["d_sisdr"]), abs(a["d_stoi"] - b["d_stoi"])
                oks.append(dd <= 0.2 and ds <= 0.005)
                w.append((dd, ds))
        status = "N/A" if not oks else ("PASS" if all(oks) else "FAIL")
        res.append(CriterionResult(7, "Sem perfil ~ v2 sem perfil", "|ddSI-SDR| <= 0,2 dB e |ddSTOI| <= 0,005 (N=0, SNR 20/5)",
                                   status, f"{status}; pior {max(x[0] for x in w):.3f} dB / {max(x[1] for x in w):.4f}" if w else "-",
                                   "-"))

    # 8: robustez de nível: dSI-SDR(-45) >= dSI-SDR(-18) - 1.5 (N=1, SIR 0)
    oks, drops = [], []
    for nz in tm.NOISES:
        lo = t.cells.get((tm.Cell("matrix", 1, 0.0, -45.0, nz).cid, "mix"))
        hi = t.cells.get((tm.Cell("matrix", 1, 0.0, -18.0, nz).cid, "mix"))
        if lo and hi:
            drops.append(hi["d_sisdr"] - lo["d_sisdr"])
            oks.append(drops[-1] <= 1.5)
    st8 = "N/A" if not oks else ("PASS" if all(oks) else "FAIL")
    res.append(CriterionResult(8, "Robustez de nível (N=1, SIR 0)", "dSI-SDR(-45) a até 1,5 dB de dSI-SDR(-18)",
                               st8, f"{st8} ({sum(oks)}/{len(oks)}); pior queda {max(drops):+.2f} dB" if drops else "-", "-"))

    # 9: pareado com o v2 (ref)
    if ref is None:
        res.append(CriterionResult(9, "Pareado com v2", ">= v2 em >= 80% das células N>=1; nunca pior que 0,5 dB em N=0",
                                   "N/A", "sem referência (--ref)", "-"))
    else:
        wins, tot, worst0 = 0, 0, 0.0
        for p, a in t.sel("mix"):
            b = ref.cells.get((tm.Cell("matrix", p["n"], p["sir"], p["level"], p["noise"]).cid, "mix"))
            if not b:
                continue
            d = a["d_sisdr"] - b["d_sisdr"]
            if p["n"] == 0:
                worst0 = min(worst0, d)
            else:
                tot += 1
                wins += d >= 0
        ok = tot > 0 and wins / tot >= 0.8 and worst0 >= -0.5
        res.append(CriterionResult(9, "Pareado com v2", ">= v2 em >= 80% das células N>=1; nunca pior que 0,5 dB em N=0",
                                   "PASS" if ok else "FAIL",
                                   f"{'PASS' if ok else 'FAIL'}; {wins}/{tot} células N>=1 ({100*wins/max(tot,1):.0f}%), pior N=0 {worst0:+.2f} dB",
                                   "-"))

    # 10: especificidade: perfil trocado => ganho do alvo <= -15 dB
    both(10, "Especificidade (perfil trocado)", "ganho do alvo <= -15 dB",
         lambda lv: t.sel("swap", n=1, sir=0.0, level=tuple(lv)),
         lambda a: (a["target_gain_db"] <= -15, a["target_gain_db"]), fmt="{:+.1f} dB (pior)", worst="max")

    res.append(CriterionResult(11, "Runtime", "paridade >= 60 dB, < 10 ms/hop, MACs inalterados", "N/A",
                               "fora do harness (gate de export/paridade)", "-"))
    return res


def fmt_ci(t: Table, view: str, metric: str, axis="matrix", **kw) -> str:
    m, lo, hi = t.pooled(view, metric, axis, **kw)
    return "-" if not math.isfinite(m) else f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
