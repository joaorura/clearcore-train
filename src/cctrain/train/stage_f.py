"""Estágio F (T3, spec §5.3-5.6): base DFNet3 treinável com BatchNorm congelada + gerador FiLM, com destilação.

Aluno = `PDFNet3(base treinável, gerador)`, iniciado do gerador do estágio C do M1 (SHA-256 conferido) sobre o DFNet3
upstream. BatchNorm em `eval()` com parâmetros congelados (estatísticas e pesos da BN idênticos, por hash, ao fim da
rodada). Professor = DFNet3 upstream congelado, que só roda nas amostras `mask=0`.

Perda por amostra (`stage_f_step`), toda sobre a saída com gating emulado (§3.5) e com bins acima de `max_freq` fora:
- base MR (DFNet3, 500/500) e `L_OS` (média tempo-frequência, 500) contra o alvo da amostra: com perfil, a fala-alvo
  seca (zeros na classe "sem alvo", fração `s`); sem perfil (`mask=0`), a soma das falas secas de todos os presentes;
- `L_gate` (1e-3) nos quadros com interferente ativo, só com perfil;
- `LocalSnrLoss` (1e-3), mascarada na classe "sem alvo" com perfil; sem perfil, toda a fala contra o ruído;
- `L_T` (500) = MR(aluno, DFNet3 congelado) só nas amostras `mask=0` (`losses.distill`).

Dados (§4.3-4.4, estágio F): misturas 35/20/35 · (1−s)/0,9 + `s` sem alvo, SIR U[−5, 25] dB, 1-2 interferentes
(p=0,5), interferente de qualquer outro falante (sem restrição de sexo), RIR p=0,3, ruído passa-baixa na banda nativa
da fala limitada; cota de banda por batch pela taxa nativa da fala-alvo (≥ 50% 48 kHz, ~30% 24-32 kHz, ≤ 20% 16 kHz);
embedding dropout `p_drop` (mask=0); enrollment do banco v2, nunca a fala da mistura, com fração `p_enroll_denoise`
de variantes denoisadas. A amostra `i` da época `e` é função pura de `(seed, e, i)`: a retomada reproduz a sequência.

Checkpoints atômicos (spec §5.6) a cada `checkpoint_every_min` e no fim de cada época; mantém os 3 últimos, o melhor
pela regra de seleção e a época fixada (`pin_epoch`). A avaliação por época (`evaluator`, no `dev-select`) roda ANTES
de gravar o checkpoint do fim de época, para que o `best` seja decidido com a métrica dele. Tudo em `runs/m2/<run_id>/`.
"""

from __future__ import annotations

import contextlib
import json
import math
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cctrain.data.audio import load_audio, resample
from cctrain.data.features import SR, analysis, extract_features
from cctrain.data.manifest import FileRecord, load_sources, training_records
from cctrain.data.mix import MixKind, crop_or_pad, loop_to_length, make_mixture, mixture_fractions
from cctrain.eval.gates import load_budget
from cctrain.eval.runtime_path import emulate_gating_offline
from cctrain.gpu import gpu_lock
from cctrain.hashing import sha256_file
from cctrain.losses.dfnet import dfnet_loss, gate_loss, local_snr_loss, over_suppression_loss
from cctrain.losses.distill import distill_loss
from cctrain.m2.config import SelectionRule, StageFConfig
from cctrain.m2.select import best_epoch, make_epoch_eval, read_history, should_stop, write_eval
from cctrain.models.state_hash import state_dict_sha256
from cctrain.paths import Paths
from cctrain.train.checkpoint import (
    Checkpoint,
    CheckpointError,
    config_hash,
    git_commit,
    load_checkpoint,
    load_latest,
    restore_rng,
    save_checkpoint,
)
from cctrain.train.embed_bank import EmbeddingBank, EmbeddingBankError
from cctrain.train.stage_c import (
    GIB,
    HOP,
    NYQUIST_HZ,
    StageCData,
    _atomic_json,
    _lsnr_target,
    _to_device,
    _truncate_log,
    frame_activity,
    total_pss,
)

TIERS = ("full", "khz24", "khz16")
MAX_REDRAWS = 8
_RUNTIME_FIELDS = frozenset({"run_id", "device", "num_workers", "prefetch_factor", "checkpoint_every_min", "probe_every",
                             "gpu_lock_timeout_s", "max_steps", "runs_dir", "data_root", "gpu_log_interval_s"})
_BRANCH_OVERRIDES = frozenset({"p_drop", "s", "p_enroll_denoise", "seed"})


class BatchNormDriftError(RuntimeError):
    """Estatísticas ou parâmetros da BatchNorm mudaram durante o estágio F (spec §5.3)."""


class InitError(RuntimeError):
    """Gerador ou base de partida diferente do congelado em `stage-f-v1.yaml`."""


# --- hiperparâmetros -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StageFHParams:
    """Parâmetros efetivos de uma rodada. Os valores de produção vêm de `configs/train/stage-f-v1.yaml`
    (`from_config`); os testes usam valores pequenos diretamente."""

    run_id: str = "t3"
    seed: int = 0
    # otimização
    lr: float = 1e-4
    epochs: int = 20  # tamanho do cosseno
    warmup_steps: int = 1000
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    batch_size: int = 16
    segment_s: float = 3.0
    steps_per_epoch: int = 7500
    # perdas
    lambda_t: float = 500.0
    lambda_os: float = 500.0
    lambda_gate: float = 1e-3
    loss_factor: float = 500.0
    loss_factor_complex: float = 500.0
    loss_gamma: float = 0.3
    fft_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
    # FiLM e misturas
    log_gamma_bound: float = math.log(10.0)
    beta_bound: float = 4.0
    p_drop: float = 0.2
    s: float = 0.05
    snr_grid_db: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 20.0, 40.0)
    sir_db: tuple[float, float] = (-5.0, 25.0)
    p_two_interferers: float = 0.5
    p_rir: float = 0.3
    band_quota: tuple[float, float, float] = (0.5, 0.3, 0.2)  # full_band_min, khz24_target, khz16_max
    p_enroll_denoise: float = 0.5
    # checkpoints
    keep_last: int = 3
    pin_epoch: int | None = 14
    stop_epoch: int | None = None  # ramo: para após esta época (contada do início do cosseno)
    # integridade
    base_state_sha256: str = ""
    generator_checkpoint: str = ""
    generator_checkpoint_sha256: str = ""
    config_sha256: str = ""
    branched_from: str = ""  # SHA-256 do checkpoint de partida de um ramo
    # fora do hash de retomada
    device: str = "cuda"
    num_workers: int = 4
    prefetch_factor: int = 2
    checkpoint_every_min: float = 30.0
    probe_every: int = 20
    gpu_lock_timeout_s: float | None = None
    gpu_log_interval_s: float = 60.0
    max_steps: int | None = None
    runs_dir: Path | None = None
    data_root: Path | None = None

    def hashable(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in _RUNTIME_FIELDS}

    @classmethod
    def from_config(cls, cfg: StageFConfig, **kw) -> StageFHParams:
        """Parâmetros do T3 a partir do `stage-f-v1.yaml` congelado; `kw` troca campos de execução ou de ablação."""
        q = cfg.mixing.band_quota
        base = cls(
            seed=cfg.seed, lr=cfg.optimizer.lr, epochs=cfg.optimizer.epochs, warmup_steps=cfg.optimizer.warmup_steps,
            weight_decay=cfg.optimizer.weight_decay, grad_clip=cfg.optimizer.grad_clip,
            batch_size=cfg.data.batch_size, segment_s=cfg.data.segment_s, steps_per_epoch=cfg.data.steps_per_epoch,
            lambda_t=cfg.loss.lambda_t, lambda_os=cfg.loss.lambda_os, lambda_gate=cfg.loss.lambda_gate,
            loss_factor=cfg.loss.base.factor, loss_factor_complex=cfg.loss.base.factor_complex,
            loss_gamma=cfg.loss.base.gamma, fft_sizes=cfg.loss.base.fft_sizes,
            log_gamma_bound=cfg.film.log_gamma_bound, beta_bound=cfg.film.beta_bound, p_drop=cfg.film.p_drop,
            s=cfg.mixing.s, snr_grid_db=cfg.mixing.snr_grid_db, sir_db=cfg.mixing.sir_db,
            p_two_interferers=cfg.mixing.p_two_interferers, p_rir=cfg.mixing.p_rir,
            band_quota=(q.full_band_min, q.khz24_target, q.khz16_max),
            p_enroll_denoise=cfg.enrollment.p_enroll_denoise, keep_last=cfg.checkpoint.keep_last,
            pin_epoch=cfg.checkpoint.pin_epoch, checkpoint_every_min=float(cfg.checkpoint.every_min),
            base_state_sha256=cfg.init.base_state_sha256, generator_checkpoint=cfg.init.generator_checkpoint,
            generator_checkpoint_sha256=cfg.init.generator_checkpoint_sha256, config_sha256=cfg.sha256)
        return replace(base, **kw)


# --- amostragem: cota de banda e passa-baixa -------------------------------------------------------------------------


def tier_of(sr: int) -> str:
    """Classe de banda pela taxa nativa da fala (§4.3): 48 kHz, 24-32 kHz ou 16 kHz."""
    return "full" if sr >= 44_100 else ("khz24" if sr >= 22_050 else "khz16")


def slot_tiers(batch_size: int, quota: tuple[float, float, float]) -> tuple[str, ...]:
    """Classe de banda da fala-alvo de cada posição do batch (a posição é `i % batch_size`): ≥ `full_band_min` em
    48 kHz (teto), `khz16` no máximo `floor(khz16_max · B)`, o resto em 24-32 kHz (alvo ~`khz24_target`)."""
    full_min, khz24, khz16_max = quota
    n_full = math.ceil(full_min * batch_size - 1e-9)
    rest = batch_size - n_full
    n16 = min(math.floor(khz16_max * batch_size + 1e-9), round(rest * khz16_max / max(khz24 + khz16_max, 1e-9)))
    return ("full",) * n_full + ("khz24",) * (rest - n16) + ("khz16",) * n16


def lowpass48(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Passa-baixa Butterworth de ordem 8, fase zero, para 48 kHz (ruído com a banda da fala limitada, §4.3)."""
    if cutoff_hz >= NYQUIST_HZ * 0.999:
        return x
    sos = butter(8, cutoff_hz / NYQUIST_HZ, output="sos")
    return sosfiltfilt(sos, np.asarray(x, np.float64)).astype(np.float32)


def _sample_rng(seed: int, epoch: int, index: int, attempt: int = 0) -> np.random.Generator:
    key = [int(seed), int(epoch), int(index)] + ([int(attempt)] if attempt else [])
    return np.random.default_rng(key)


class StageFDataset(Dataset):
    """Mapa `i -> amostra` do estágio F, função pura de `(hp.seed, epoch, i)`. Dev/teste são recusados."""

    def __init__(self, data: StageCData, bank: EmbeddingBank, data_root: Path, hp: StageFHParams, *, epoch: int):
        sources = data.sources or load_sources()
        self.speech = training_records(data.speech, sources, allow_pilot=True)
        self.noise = training_records(data.noise, sources, allow_pilot=True)
        self.rirs = training_records(data.rirs, sources, allow_pilot=True)
        self.data_root, self.hp, self.epoch, self.bank = Path(data_root), hp, int(epoch), bank
        self.provenance = bool(getattr(bank, "has_provenance", False))
        by: dict[str, dict[str, list[FileRecord]]] = defaultdict(lambda: defaultdict(list))
        for r in self.speech:
            if r.speaker_key is None:
                raise ValueError(f"{r.path}: registro de fala sem falante")
            by[tier_of(r.sr)][r.speaker_key].append(r)
        self.by_tier = {t: {k: v for k, v in sorted(by[t].items())} for t in TIERS}
        self.speakers = sorted({k for t in TIERS for k in self.by_tier[t]})
        in_bank = set(bank.speakers())
        self.targets = [k for k in self.speakers if k in in_bank]
        if not self.targets:
            raise ValueError("nenhum falante de treino está no banco de embeddings")
        if len(self.speakers) < 2:
            raise ValueError("a mistura exige ao menos 2 falantes de treino")
        if not self.noise:
            raise ValueError("sem ruído de treino")
        self.slots = slot_tiers(hp.batch_size, hp.band_quota)
        self.tier_targets = {t: [k for k in self.by_tier[t] if k in in_bank] for t in TIERS}
        for t in set(self.slots):
            if not self.tier_targets[t]:
                raise ValueError(f"cota de banda: nenhuma fala de treino da classe {t} com falante no banco")
        fr = mixture_fractions(hp.s)
        self.kinds, self.kind_p = list(fr), np.array(list(fr.values()))

    def __len__(self) -> int:
        return self.hp.steps_per_epoch * self.hp.batch_size

    def _load48(self, r: FileRecord) -> np.ndarray:
        x, sr = load_audio(self.data_root / r.path)
        return resample(x, sr, SR)

    def _pick(self, rng, keys: Sequence[str], tier: str | None, exclude: Sequence[str] = ()) -> FileRecord:
        pool = [k for k in keys if k not in exclude]
        key = pool[int(rng.integers(len(pool)))]
        recs = self.by_tier[tier][key] if tier else [r for t in TIERS for r in self.by_tier[t].get(key, [])]
        return recs[int(rng.integers(len(recs)))]

    def _embedding(self, key: str, rng, exclude: tuple[str, ...]) -> np.ndarray:
        if self.provenance:
            return np.asarray(self.bank.sample(key, rng, exclude_utterances=exclude,
                                               p_denoised=self.hp.p_enroll_denoise), np.float32)
        return np.asarray(self.bank.sample(key, rng), np.float32)

    def __getitem__(self, i: int) -> dict:
        for attempt in range(MAX_REDRAWS):
            try:
                return self._build(int(i), _sample_rng(self.hp.seed, self.epoch, int(i), attempt))
            except EmbeddingBankError:  # falante sem variante elegível (toda variante contém a fala): novo sorteio
                continue
        raise EmbeddingBankError(f"amostra {i}: {MAX_REDRAWS} sorteios sem variante de enrollment elegível")

    def _build(self, i: int, rng: np.random.Generator) -> dict:
        hp = self.hp
        n = int(hp.segment_s * SR)
        frames = n // HOP
        tier = self.slots[i % hp.batch_size]
        kind = self.kinds[int(rng.choice(len(self.kinds), p=self.kind_p))]
        has_target = kind is not MixKind.INTERFERER_ONLY
        interferers: list[np.ndarray] = []
        if has_target:
            tkey = self.tier_targets[tier][int(rng.integers(len(self.tier_targets[tier])))]
            trec = self.by_tier[tier][tkey][int(rng.integers(len(self.by_tier[tier][tkey])))]
            target = crop_or_pad(self._load48(trec), n, rng)
            band_src, exclude = trec, (trec.path,)
            if kind is MixKind.TARGET_NOISE:
                count, first = 0, None
            else:
                count = 2 if rng.random() < hp.p_two_interferers else 1
                first = None
        else:  # "locutor trocado": o perfil é de quem não fala; a banda da cota é a do primeiro interferente
            tkey = self.targets[int(rng.integers(len(self.targets)))]
            trec, exclude = None, ()
            target = np.zeros(n, np.float32)
            count = 2 if rng.random() < hp.p_two_interferers else 1
            first = self._pick(rng, [k for k in self.by_tier[tier] if k != tkey], tier)
            band_src = first
        irec = None
        if count:
            chosen = [first] if first is not None else []
            while len(chosen) < count:
                taken = {r.speaker_key for r in chosen} | {tkey}
                chosen.append(self._pick(rng, [k for k in self.speakers if k not in taken] or
                                         [k for k in self.speakers if k != tkey], None))
            for r in chosen:
                interferers.append(crop_or_pad(self._load48(r), n, rng))
            irec = chosen[0]
        noise = loop_to_length(self._load48(self.noise[int(rng.integers(len(self.noise)))]), n, rng)
        if kind is MixKind.INTERFERER_ONLY and rng.random() < 0.5:
            noise = None
        if noise is not None and band_src.sr < 2 * NYQUIST_HZ:
            noise = lowpass48(noise, band_src.sr / 2.0 * 0.98)

        def rir():
            if self.rirs and rng.random() < hp.p_rir:
                return self._load48(self.rirs[int(rng.integers(len(self.rirs)))])
            return None

        snr = float(hp.snr_grid_db[int(rng.integers(len(hp.snr_grid_db)))])
        sir = float("nan")
        if interferers and has_target:
            sir = float(rng.uniform(*hp.sir_db))
        mix = make_mixture(target, interferers, noise, kind=kind, snr_db=snr, sir_db=sir, rir_target=rir(),
                           rir_interferer=rir())
        mask0 = bool(rng.random() < hp.p_drop)  # embedding dropout: sem perfil
        emb = self._embedding(tkey, rng, exclude)
        feats = extract_features(mix.noisy)
        active = np.zeros(frames, bool)
        if not mask0:
            for x in interferers:
                active |= frame_activity(x, frames)
        band = band_src.band_hz if band_src.band_hz else NYQUIST_HZ
        return {
            "spec": feats.spec, "feat_erb": feats.feat_erb, "feat_spec": feats.feat_spec,
            "spec_clean": analysis(mix.clean_all if mask0 else mix.clean), "embedding": emb,
            "mask": np.full(1, 0.0 if mask0 else 1.0, np.float32), "interferer_frames": active,
            "lsnr_mask": np.full(frames, mask0 or has_target, bool), "max_freq": np.float32(min(band, NYQUIST_HZ)),
            "kind": int(kind), "snr_db": np.float32(snr), "sir_db": np.float32(sir), "tier": TIERS.index(tier),
            "target_key": tkey, "interferer_key": irec.speaker_key if irec is not None else "",
            "target_path": trec.path if trec is not None else "",
        }


# --- modelo ----------------------------------------------------------------------------------------------------------


def prepare_student(model: nn.Module) -> None:
    """BatchNorm em `eval()` e parâmetros congelados (estatísticas e pesos não mudam); o resto treina. Os RNNs ficam em
    modo de treino (o backward do cuDNN exige); sem dropout nos GRUs do DFNet3 o forward é o mesmo do `eval()`."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        elif isinstance(m, nn.RNNBase):
            assert m.dropout == 0.0
            m.train()


def bn_state_sha256(model: nn.Module) -> str:
    """SHA-256 dos parâmetros e buffers (running_mean/var, num_batches_tracked) de todas as BatchNorm."""
    import hashlib

    h = hashlib.sha256()
    for name, m in sorted(model.named_modules(), key=lambda kv: kv[0]):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            for key, t in sorted({**dict(m.named_parameters(recurse=False)),
                                  **dict(m.named_buffers(recurse=False))}.items()):
                h.update(f"{name}.{key}".encode())
                h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def n_batchnorm(model: nn.Module) -> int:
    return sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


def build_models(hp: StageFHParams, device: str):
    """(aluno PDFNet3, professor DFNet3 congelado, df_state). O gerador começa zerado na saída (identidade)."""
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.models.film import FilmConfig, FilmGenerator
    from cctrain.models.pdfnet3 import PDFNet3

    base, df_state = load_upstream_dfnet3(device=device)
    if hp.base_state_sha256 and state_dict_sha256(base) != hp.base_state_sha256:
        raise InitError(f"base DFNet3 {state_dict_sha256(base)[:12]} != congelada {hp.base_state_sha256[:12]}")
    teacher, _ = load_upstream_dfnet3(device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    gen = FilmGenerator(FilmConfig(log_gamma_bound=hp.log_gamma_bound, beta_bound=hp.beta_bound)).to(device)
    model = PDFNet3(base, gen)
    prepare_student(model)
    return model, teacher, df_state


def load_init(model: nn.Module, path: Path, *, expected_sha256: str = "") -> str:
    """Carrega a partida do aluno: checkpoint do gerador do estágio C (só o gerador) ou do estágio F (aluno inteiro).
    Devolve o SHA-256 do arquivo."""
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise InitError(f"{path}: SHA-256 {digest[:12]} != congelado {expected_sha256[:12]}")
    state = load_checkpoint(path).model
    if any(k.startswith("film.") for k in state):
        model.load_state_dict(state)
    else:
        model.film.load_state_dict(state)
    return digest


def _lr_factor(step: int, warmup: int, total: int) -> float:
    warm = min(1.0, (step + 1) / max(warmup, 1))
    return warm * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(total, 1))))


# --- passo de treino -------------------------------------------------------------------------------------------------


def _gated(base: nn.Module, spec, spec_e, m, lsnr):
    spec_m = base.mask(spec.clone(), m)
    return emulate_gating_offline(spec, spec_m, spec_e, lsnr.reshape(spec.shape[0], -1))


def stage_f_step(model: nn.Module, teacher: nn.Module, istft: nn.Module, df_params, hp: StageFHParams, b: dict
                 ) -> dict[str, torch.Tensor]:
    """Forward + perdas do estágio F sobre um lote já no dispositivo. Devolve `total` e os termos (`mr`, `os`,
    `gate`, `lsnr`, `t`) e `n0` (amostras sem perfil)."""
    spec, spec_clean, mask = b["spec"], b["spec_clean"], b["mask"]
    nb = spec.shape[0]
    spec_e, m, lsnr, _ = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], mask)
    out, _stages = _gated(model.base, spec, spec_e, m, lsnr)
    est, tgt = istft(out), istft(spec_clean)
    lsnr_t = _lsnr_target(df_params, spec_clean, spec)
    kw = dict(n_ffts=hp.fft_sizes, gamma=hp.loss_gamma)
    mr = dfnet_loss(est, tgt, b["max_freq"], factor=hp.loss_factor, factor_complex=hp.loss_factor_complex, **kw)
    os_ = over_suppression_loss(est, tgt, b["max_freq"], factor=hp.lambda_os, **kw)
    gate = gate_loss(lsnr, b["interferer_frames"] & (mask.reshape(nb, 1) > 0), factor=hp.lambda_gate)
    ls = local_snr_loss(lsnr, lsnr_t, b["lsnr_mask"])
    idx = (mask.reshape(nb) == 0).nonzero(as_tuple=True)[0]
    if idx.numel():
        with torch.no_grad():
            s0 = spec[idx]
            t_e, t_m, t_lsnr, _ = teacher(s0.clone(), b["feat_erb"][idx], b["feat_spec"][idx])
            t_out, _ = _gated(teacher, s0, t_e, t_m, t_lsnr)
            t_est = istft(t_out)
    else:
        t_est = est[:0]
    t = distill_loss(est, t_est, mask, b["max_freq"], lambda_t=hp.lambda_t, n_ffts=hp.fft_sizes, gamma=hp.loss_gamma)
    return {"total": mr + os_ + gate + ls + t, "mr": mr, "os": os_, "gate": gate, "lsnr": ls, "t": t,
            "n0": torch.tensor(float(idx.numel()))}


# --- RAM, GPU e vazão ------------------------------------------------------------------------------------------------

_GPU_FIELDS = ("timestamp,temperature.gpu,power.draw,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active,"
               "memory.used,utilization.gpu")


@contextlib.contextmanager
def gpu_logger(path: Path, interval_s: float):
    """`nvidia-smi` a cada `interval_s` em `path` (CSV): temperatura, potência, clock, throttle (spec §5.5)."""
    stop = threading.Event()
    path.parent.mkdir(parents=True, exist_ok=True)

    def loop():
        header_done = path.exists() and path.stat().st_size > 0
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={_GPU_FIELDS}", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=20)
                with open(path, "a") as fh:
                    if not header_done:
                        fh.write(_GPU_FIELDS + "\n")
                        header_done = True
                    fh.write(out.stdout)
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(interval_s)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=30)


def measure_loader_throughput(hp: StageFHParams, bank: EmbeddingBank, data: StageCData, *, n_batches: int = 60,
                              warmup_batches: int = 3, epoch: int = 0) -> dict:
    """Vazão do loader sozinho (batches/s), com o `num_workers` e o `prefetch_factor` da rodada."""
    data_root = Path(hp.data_root) if hp.data_root else Paths.from_env().data
    ds = StageFDataset(data, bank, data_root, hp, epoch=epoch)
    dl = DataLoader(ds, batch_size=hp.batch_size, sampler=list(range((n_batches + warmup_batches) * hp.batch_size)),
                    num_workers=hp.num_workers, prefetch_factor=hp.prefetch_factor if hp.num_workers else None)
    t0, seen, pss = None, 0, 0
    for k, _ in enumerate(dl):
        if k == warmup_batches - 1:
            t0 = time.perf_counter()
        elif k >= warmup_batches:
            seen += 1
        if k % 10 == 0:
            pss = max(pss, total_pss())
    dt = time.perf_counter() - t0
    return {"batches_per_s": seen / dt, "n_batches": seen, "seconds": dt, "pss_gb_max": pss / GIB,
            "num_workers": hp.num_workers}


# --- rodada ----------------------------------------------------------------------------------------------------------

Evaluator = Callable[[nn.Module, int], dict]


def _run_dir(hp: StageFHParams) -> Path:
    base = Path(hp.runs_dir) if hp.runs_dir else Paths.from_env().runs
    return base / "m2" / hp.run_id


def run_stage_f(cfg: StageFConfig | StageFHParams, bank: EmbeddingBank, data: StageCData, *, max_hours: float,
                resume: bool = True, init: Path | None = None, evaluator: Evaluator | None = None,
                rule: SelectionRule | None = None) -> Path:
    """Treina o estágio F até `max_hours` (do run inteiro, contando o tempo antes das retomadas), `max_steps`, o fim
    das `epochs` (ou `stop_epoch`) ou o early stopping; devolve o último checkpoint.

    `resume`: retoma do último checkpoint válido do `run_id` (hash da config conferido); sem retomada, recusa um
    diretório de checkpoints já preenchido. `init`: checkpoint de partida (gerador do estágio C ou aluno do F); sem
    ele, o gerador congelado em `init.generator_checkpoint`. Em CUDA, segura o `gpu_lock`."""
    hp = StageFHParams.from_config(cfg) if isinstance(cfg, StageFConfig) else cfg
    lock = gpu_lock(hp.gpu_lock_timeout_s) if hp.device.startswith("cuda") else contextlib.nullcontext()
    with lock:
        return _run(hp, bank, data, max_hours, resume, init, evaluator, rule, None)


def branch(from_ckpt: Path, overrides: dict, epochs: int, *, bank: EmbeddingBank, data: StageCData, run_id: str,
           max_hours: float, evaluator: Evaluator | None = None, rule: SelectionRule | None = None,
           **runtime) -> Path:
    """Ramo da AB2: continua o cosseno a partir de um checkpoint do estágio F (ex.: a época 14 fixada) por `epochs`
    épocas, trocando só `p_drop`, `s`, `p_enroll_denoise` ou `seed`. O novo run tem o próprio `run_id` e hash."""
    bad = set(overrides) - _BRANCH_OVERRIDES
    if bad:
        raise ValueError(f"override(s) não permitido(s) em ramo: {sorted(bad)}; permitidos {sorted(_BRANCH_OVERRIDES)}")
    ck = load_checkpoint(from_ckpt)
    if "hp" not in ck.extra:
        raise CheckpointError(f"{from_ckpt}: não é checkpoint do estágio F (sem hp)")
    start_epoch = int(ck.sampler_state["epoch"])
    if int(ck.sampler_state["batch"]) != 0:
        raise CheckpointError(f"{from_ckpt}: ramo parte de fim de época (batch=0), veio batch={ck.sampler_state['batch']}")
    old = _hp_from_dict(ck.extra["hp"])
    hp = replace(old, run_id=run_id, stop_epoch=start_epoch + epochs, branched_from=sha256_file(from_ckpt),
                 **overrides, **runtime)
    lock = gpu_lock(hp.gpu_lock_timeout_s) if hp.device.startswith("cuda") else contextlib.nullcontext()
    with lock:
        return _run(hp, bank, data, max_hours, True, None, evaluator, rule, from_ckpt)


def _hp_from_dict(d: dict) -> StageFHParams:
    d = dict(d)
    for k in ("fft_sizes", "snr_grid_db", "sir_db", "band_quota"):
        d[k] = tuple(d[k])
    return StageFHParams(**d)


def _run(hp: StageFHParams, bank: EmbeddingBank, data: StageCData, max_hours: float, resume: bool,
         init: Path | None, evaluator: Evaluator | None, rule: SelectionRule | None, start_ckpt: Path | None) -> Path:
    from cctrain.dfvendor import ensure_df_importable

    ensure_df_importable()
    from df.loss import Istft
    from df.model import ModelParams

    paths = Paths.from_env()
    data_root = Path(hp.data_root) if hp.data_root else paths.data
    run_dir = _run_dir(hp)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = hp.device
    total_steps = hp.epochs * hp.steps_per_epoch

    model, teacher, df_state = build_models(hp, device)
    istft = Istft(960, HOP, torch.as_tensor(df_state.fft_window().copy())).to(device)
    params = ModelParams()
    torch.manual_seed(hp.seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp.lr,
                            weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, hp.warmup_steps, total_steps))
    chash = config_hash({**hp.hashable(), "frozen_config_sha256": hp.config_sha256})
    commit = git_commit()
    existing = load_latest(ckpt_dir, expected_cfg_hash=chash if resume else None)
    if existing is not None and not resume:
        raise CheckpointError(f"{ckpt_dir} já tem checkpoints e resume=False; escolha outro run_id")
    step = epoch = batch_idx = 0
    spent_before = 0.0
    init_sha = ""
    log_path = run_dir / "loss-log.jsonl"

    def restore(ck: Checkpoint) -> None:
        nonlocal step, epoch, batch_idx
        model.load_state_dict(ck.model)
        opt.load_state_dict(ck.optimizer)
        sched.load_state_dict(ck.scheduler)
        step, epoch, batch_idx = ck.step, int(ck.sampler_state["epoch"]), int(ck.sampler_state["batch"])
        restore_rng(ck)

    if existing is not None:
        restore(existing)
        spent_before, init_sha = float(existing.extra.get("elapsed_s", 0.0)), existing.extra.get("init_sha256", "")
        _truncate_log(log_path, step)
    elif start_ckpt is not None:
        ck0 = load_checkpoint(start_ckpt)
        restore(ck0)
        init_sha = sha256_file(start_ckpt)
        if hp.branched_from and init_sha != hp.branched_from:
            raise InitError("checkpoint de partida do ramo difere do registrado")
        if log_path.exists():
            log_path.unlink()
    else:
        path = Path(init) if init else paths.runs / "m1" / hp.generator_checkpoint
        if not init and not hp.generator_checkpoint:
            raise InitError("sem gerador de partida: `init` ou `init.generator_checkpoint` da config")
        init_sha = load_init(model, path, expected_sha256="" if init else hp.generator_checkpoint_sha256)
        if log_path.exists():
            log_path.unlink()
    prepare_student(model)
    bn_before = bn_state_sha256(model)
    n_bn = n_batchnorm(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    history = read_history(run_dir)
    stop_reason = ""
    t_start = time.monotonic()
    last_ckpt = t_start
    step_times: list[float] = []
    wait_times: list[float] = []
    sums: dict[str, float] = defaultdict(float)
    pss_max = vram_peak = vram_reserved = 0
    last_saved_step = existing.step if existing is not None else -1
    last_path = existing.path if existing is not None else None
    end_epoch = hp.stop_epoch if hp.stop_epoch is not None else hp.epochs
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def elapsed() -> float:
        return spent_before + (time.monotonic() - t_start)

    def save(*, final: bool = False, metric: float | None = None, pin: bool = False) -> Path:
        return save_checkpoint(
            ckpt_dir, step=step, model=model.state_dict(), optimizer=opt, scheduler=sched, keep=hp.keep_last,
            sampler_state={"epoch": epoch, "batch": batch_idx}, cfg_hash=chash, commit=commit, metric=metric, pin=pin,
            extra={"hp": asdict(hp) | {"runs_dir": None, "data_root": None}, "final": final, "elapsed_s": elapsed(),
                   "init_sha256": init_sha, "bn_sha256": bn_before, "base_state_sha256": hp.base_state_sha256,
                   "config_sha256": hp.config_sha256})

    done = (hp.max_steps is not None and step >= hp.max_steps) or epoch >= end_epoch
    log_ctx = gpu_logger(run_dir / "nvidia-smi.csv", hp.gpu_log_interval_s) if device.startswith("cuda") \
        else contextlib.nullcontext()
    with log_ctx, open(log_path, "a") as log:
        while not done:
            ds = StageFDataset(data, bank, data_root, hp, epoch=epoch)
            sampler = list(range(batch_idx * hp.batch_size, len(ds)))
            dl = DataLoader(ds, batch_size=hp.batch_size, sampler=sampler, num_workers=hp.num_workers,
                            prefetch_factor=hp.prefetch_factor if hp.num_workers else None)
            it = iter(dl)
            while True:
                tw = time.perf_counter()
                try:
                    batch = next(it)
                except StopIteration:
                    break
                t0 = time.perf_counter()
                wait_times.append(t0 - tw)
                b = _to_device(batch, device)
                prepare_student(model)
                opt.zero_grad(set_to_none=True)
                parts = stage_f_step(model, teacher, istft, params, hp, b)
                parts["total"].backward()
                nn.utils.clip_grad_norm_(trainable, hp.grad_clip)
                opt.step()
                sched.step()
                step += 1
                batch_idx += 1
                step_times.append(time.perf_counter() - t0)
                row = {k: float(v.detach()) for k, v in parts.items()}
                for k, v in row.items():
                    sums[k] += v
                log.write(json.dumps({"step": step, "epoch": epoch, "lr": sched.get_last_lr()[0], **row}) + "\n")
                log.flush()
                if step % max(hp.probe_every, 1) == 0 or step == 1:
                    pss_max = max(pss_max, total_pss())
                if (hp.max_steps is not None and step >= hp.max_steps) or elapsed() >= max_hours * 3600.0:
                    done = True
                    stop_reason = "max_steps" if (hp.max_steps is not None and step >= hp.max_steps) else "max_hours"
                    break
                if time.monotonic() - last_ckpt >= hp.checkpoint_every_min * 60.0:
                    last_path, last_saved_step, last_ckpt = save(), step, time.monotonic()
            if done:
                break
            # fim de época (o sampler esgotou)
            epoch, batch_idx = epoch + 1, 0
            metric = None
            if evaluator is not None and rule is not None:
                ev = make_epoch_eval(epoch, step, evaluator(model, epoch), rule)
                write_eval(run_dir, ev)
                history = read_history(run_dir)
                best = best_epoch(history, rule)
                if ev.eligible and best is not None and best.epoch == epoch:
                    metric = -float(ev.score)
                prepare_student(model)
            last_path = save(metric=metric, pin=hp.pin_epoch is not None and epoch == hp.pin_epoch)
            last_saved_step, last_ckpt = step, time.monotonic()
            if evaluator is not None and rule is not None and should_stop(history, rule):
                stop_reason, done = "early_stopping", True
            elif epoch >= end_epoch:
                stop_reason, done = "epochs", True
    pss_max = max(pss_max, total_pss())
    if device.startswith("cuda"):
        vram_peak, vram_reserved = (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
    bn_after = bn_state_sha256(model)
    if bn_after != bn_before:
        raise BatchNormDriftError(f"BatchNorm mudou: {bn_before[:12]} -> {bn_after[:12]}")
    if step != last_saved_step:
        last_path = save(final=True)
    n = max(len(step_times), 1)
    limit = load_budget().loader_rss_gb_max
    summary = {
        "run_id": hp.run_id, "steps": step, "epoch": epoch, "stop_reason": stop_reason, "config_hash": chash,
        "commit": commit, "init_sha256": init_sha, "bn_sha256_before": bn_before, "bn_sha256_after": bn_after,
        "n_batchnorm_modules": n_bn, "trainable_params": sum(p.numel() for p in trainable),
        "s_per_step_median": statistics.median(step_times) if step_times else None,
        "s_per_step_mean": sum(step_times) / n if step_times else None,
        "loader_wait_s_mean": sum(wait_times) / max(len(wait_times), 1),
        "loss_mean": {k: v / n for k, v in sums.items()}, "pss_gb_max": pss_max / GIB, "pss_limit_gb": limit,
        "vram_peak_allocated_gb": vram_peak / GIB if vram_peak else None,
        "vram_peak_reserved_gb": vram_reserved / GIB if vram_reserved else None,
        "elapsed_s_total": elapsed(), "last_checkpoint": last_path.name, "num_workers": hp.num_workers,
        "config": asdict(hp) | {"runs_dir": None, "data_root": None},
    }
    _atomic_json(run_dir / "summary.json", summary)
    return last_path
