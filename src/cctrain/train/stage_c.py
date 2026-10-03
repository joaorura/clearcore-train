"""Estágio C curto (spec §5.3, T2 "sonda"): treina só o gerador FiLM com a base DFNet3 congelada em `eval()`.

Dados: misturas do §4.4 (alvo + ruído / + interferente / + ambos / só interferente), com interferente do **mesmo
sexo** a SIR fixo (padrão 0 dB: o cenário do gate funcional do M1) e o embedding do alvo vindo do banco
pré-computado (o encoder não roda aqui). Perdas do §5.4 sem destilação (`losses.dfnet.stage_c_loss`), calculadas
sobre a saída com o gating emulado do §3.5 (`emulate_gating_offline`). A amostra `i` da época `e` é função pura de
`(seed, e, i)` (`mix_rng`): a retomada reproduz a sequência sem guardar estado do loader.

Integridade: SHA-256 do `state_dict` da base antes e depois (`BaseDriftError` se mudar). Tudo em `runs/m1/<run_id>/`:
`checkpoints/` (spec §5.6), `loss-log.jsonl`, `summary.json`; a sonda de RAM (PSS do trainer + workers) em
`runs/m1/stage-c-probe.json`.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cctrain.data.audio import load_audio, resample
from cctrain.data.features import SR, analysis, extract_features
from cctrain.data.manifest import FileRecord, SourceManifest, load_sources, training_records
from cctrain.data.mix import (
    SIR_RANGE_DB,
    SNRS_DB,
    MixKind,
    crop_or_pad,
    loop_to_length,
    make_mixture,
    mix_rng,
    mixture_fractions,
)
from cctrain.eval.gates import load_budget
from cctrain.eval.runtime_path import emulate_gating_offline
from cctrain.gpu import gpu_lock
from cctrain.losses.dfnet import stage_c_loss
from cctrain.models.state_hash import state_dict_sha256
from cctrain.paths import Paths
from cctrain.train.checkpoint import (
    Checkpoint,
    config_hash,
    git_commit,
    load_latest,
    restore_rng,
    save_checkpoint,
)
from cctrain.train.embed_bank import EmbeddingBank

HOP = 480
NYQUIST_HZ = 24_000.0
PROBE_NAME = "stage-c-probe.json"
# Campos que não mudam a sequência de amostras nem a otimização: ficam fora do hash que guarda a retomada.
_HASH_EXCLUDE = frozenset({"run_id", "max_steps", "device", "num_workers", "prefetch_factor", "runs_dir",
                           "data_root", "checkpoint_every_min", "probe_every", "gpu_lock_timeout_s"})
GIB = 1024**3
_TRAIN_SPLITS = ("noise", "rir", "train", "pilot")


class BaseDriftError(RuntimeError):
    """O `state_dict` da base mudou durante o estágio C (spec §5.3: hash antes == depois)."""


@dataclass(frozen=True)
class StageCConfig:
    run_id: str = "stage-c"
    seed: int = 20261003
    batch_size: int = 16
    segment_s: float = 3.0
    lr: float = 1e-3
    warmup_steps: int = 100
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    silence_fraction: float = 0.05
    sir_db: float | None = 0.0  # None: U[-5, 25]
    same_sex: bool = True
    p_rir: float = 0.3
    p_two_interferers: float = 0.0
    use_band_limit: bool = True
    steps_per_epoch: int = 500
    # fora do hash de retomada
    max_steps: int | None = None
    device: str = "cuda"
    num_workers: int = 4
    prefetch_factor: int = 2
    checkpoint_every_min: float = 30.0
    probe_every: int = 10
    gpu_lock_timeout_s: float | None = None
    runs_dir: Path | None = None
    data_root: Path | None = None

    def hashable(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in _HASH_EXCLUDE}


@dataclass(frozen=True)
class StageCData:
    speech: Sequence[FileRecord]
    noise: Sequence[FileRecord]
    rirs: Sequence[FileRecord]
    sex: Mapping[str, str] | None = None  # speaker_key -> 'F' | 'M'
    sources: Mapping[str, SourceManifest] | None = field(default=None, compare=False)


def frame_activity(x: np.ndarray, n_frames: int, *, range_db: float = 30.0, floor_dbfs: float = -60.0) -> np.ndarray:
    """Quadros (hop 480) com fala ativa: nível a até `range_db` do máximo e acima do piso absoluto."""
    f = np.asarray(x[: n_frames * HOP], np.float64).reshape(n_frames, HOP)
    level = 10 * np.log10(np.mean(f**2, axis=1) + 1e-12)
    return (level >= max(level.max() - range_db, floor_dbfs)) & (level.max() > floor_dbfs)


class StageCDataset(Dataset):
    """Mapa `i -> amostra`, função pura de `(cfg.seed, epoch, i)`. Registros de dev/teste são recusados."""

    def __init__(self, data: StageCData, bank: EmbeddingBank, data_root: Path, cfg: StageCConfig, *, epoch: int):
        sources = data.sources or load_sources()
        self.speech = training_records(data.speech, sources, allow_pilot=True)
        self.noise = training_records(data.noise, sources, allow_pilot=True)
        self.rirs = training_records(data.rirs, sources, allow_pilot=True)
        self.data_root, self.cfg, self.epoch, self.bank = Path(data_root), cfg, int(epoch), bank
        if cfg.same_sex and data.sex is None:
            raise ValueError("same_sex=True exige o mapa de sex por falante (speaker_key -> F/M); "
                             "para misturas sem restrição, same_sex=False explícito")
        self.sex = data.sex
        by: dict[str, list[FileRecord]] = defaultdict(list)
        for r in self.speech:
            if r.speaker_key is None:
                raise ValueError(f"{r.path}: registro de fala sem falante")
            if cfg.same_sex and r.speaker_key not in self.sex:
                continue
            by[r.speaker_key].append(r)
        self.by_speaker = dict(by)
        in_bank = set(bank.speakers())
        self.targets = sorted(k for k in self.by_speaker if k in in_bank)
        if not self.targets:
            raise ValueError("nenhum falante de treino está no banco de embeddings (ou no mapa de sex)")
        if not self.noise:
            raise ValueError("sem ruído de treino")
        for k in self.targets:
            if not self._peers(k):
                raise ValueError(f"falante {k} sem interferente elegível (same_sex={cfg.same_sex})")
        fr = mixture_fractions(cfg.silence_fraction)
        self.kinds = list(fr)
        self.kind_p = np.array([fr[k] for k in self.kinds])

    def __len__(self) -> int:
        return self.cfg.steps_per_epoch * self.cfg.batch_size

    def _peers(self, key: str) -> list[str]:
        return [s for s in sorted(self.by_speaker)
                if s != key and (not self.cfg.same_sex or self.sex[s] == self.sex[key])]

    def _load48(self, r: FileRecord) -> np.ndarray:
        x, sr = load_audio(self.data_root / r.path)
        return resample(x, sr, SR)

    def __getitem__(self, i: int) -> dict:
        cfg = self.cfg
        rng = mix_rng(cfg.seed, self.epoch, int(i))
        n = int(cfg.segment_s * SR)
        frames = n // HOP
        kind = self.kinds[int(rng.choice(len(self.kinds), p=self.kind_p))]
        tkey = self.targets[int(rng.integers(len(self.targets)))]
        trec = self.by_speaker[tkey][int(rng.integers(len(self.by_speaker[tkey])))]
        target = crop_or_pad(self._load48(trec), n, rng)
        interferers, ikey, irec = [], "", None
        if kind is not MixKind.TARGET_NOISE:
            peers = self._peers(tkey)
            count = 2 if (rng.random() < cfg.p_two_interferers and len(peers) > 1) else 1
            for s in rng.choice(peers, size=count, replace=False):
                recs = self.by_speaker[str(s)]
                irec = recs[int(rng.integers(len(recs)))]
                interferers.append(crop_or_pad(self._load48(irec), n, rng))
                ikey = ikey or str(s)
        noise = loop_to_length(self._load48(self.noise[int(rng.integers(len(self.noise)))]), n, rng)
        if kind is MixKind.INTERFERER_ONLY and rng.random() < 0.5:
            noise = None

        def rir():
            if self.rirs and rng.random() < cfg.p_rir:
                return self._load48(self.rirs[int(rng.integers(len(self.rirs)))])
            return None

        snr = float(SNRS_DB[int(rng.integers(len(SNRS_DB)))])
        sir = float("nan")
        if interferers and kind is not MixKind.INTERFERER_ONLY:
            sir = float(cfg.sir_db) if cfg.sir_db is not None else float(rng.uniform(*SIR_RANGE_DB))
        mix = make_mixture(target, interferers, noise, kind=kind, snr_db=snr, sir_db=sir, rir_target=rir(),
                           rir_interferer=rir())
        feats = extract_features(mix.noisy)
        active = np.zeros(frames, bool)
        for x in interferers:
            active |= frame_activity(x, frames)
        has_target = kind is not MixKind.INTERFERER_ONLY
        band_src = trec if has_target else irec
        band = band_src.band_hz if (cfg.use_band_limit and band_src is not None and band_src.band_hz) else NYQUIST_HZ
        return {
            "spec": feats.spec, "feat_erb": feats.feat_erb, "feat_spec": feats.feat_spec,
            "spec_clean": analysis(mix.clean), "embedding": np.asarray(self.bank.sample(tkey, rng), np.float32),
            "mask": np.ones(1, np.float32), "interferer_frames": active,
            "lsnr_mask": np.full(frames, has_target, bool), "max_freq": np.float32(min(band, NYQUIST_HZ)),
            "kind": int(kind), "snr_db": np.float32(snr), "sir_db": np.float32(sir),
            "target_key": tkey, "interferer_key": ikey,
        }


# --- RAM -----------------------------------------------------------------------------------------------------------


def total_pss() -> int:
    """PSS (bytes) do processo e de todos os filhos (workers do loader). PSS reparte as páginas compartilhadas."""
    procs = [psutil.Process(), *psutil.Process().children(recursive=True)]
    total = 0
    for p in procs:
        try:
            total += p.memory_full_info().pss
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass
    return total


# --- treino ----------------------------------------------------------------------------------------------------------


def freeze_base(base: nn.Module) -> None:
    """Base em `eval()` (sem atualizar estatísticas), mas com os módulos RNN em modo de treino: o backward do cuDNN
    RNN exige. Sem dropout nos GRUs do DFNet3 e sem parâmetros alterados, o forward é o mesmo (hash conferido)."""
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    for m in base.modules():
        if isinstance(m, nn.RNNBase):
            assert m.dropout == 0.0
            m.train()


def assert_base_unchanged(base: nn.Module, sha_before: str) -> str:
    after = state_dict_sha256(base)
    if after != sha_before:
        raise BaseDriftError(f"state_dict da base mudou: {sha_before[:12]} -> {after[:12]}")
    return after


def _lsnr_target(df_params, spec_clean: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
    from cctrain.dfvendor import ensure_df_importable

    ensure_df_importable()
    from df.modules import LocalSnrTarget

    tgt = LocalSnrTarget(ws=20, target_snr_range=[df_params.lsnr_min - 1, df_params.lsnr_max + 1])
    return tgt(spec_clean, noise=spec - spec_clean)


def _to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_stage_c(cfg: StageCConfig, bank: EmbeddingBank, max_minutes: float, *, data: StageCData | None = None
                ) -> Path:
    """Treina o gerador FiLM até `max_minutes` ou `cfg.max_steps`; devolve o caminho do último checkpoint.

    Retoma do último checkpoint válido do `run_id` (hash da config conferido). Em CUDA, segura o `gpu_lock`."""
    lock = gpu_lock(cfg.gpu_lock_timeout_s) if cfg.device.startswith("cuda") else contextlib.nullcontext()
    with lock:
        return _run(cfg, bank, max_minutes, data)


def _run(cfg: StageCConfig, bank: EmbeddingBank, max_minutes: float, data: StageCData | None) -> Path:
    from cctrain.dfvendor import ensure_df_importable

    ensure_df_importable()
    from df.loss import Istft
    from df.model import ModelParams

    from cctrain.data.loader import pilot_records
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.models.film import FilmGenerator, load_film_config
    from cctrain.models.pdfnet3 import PDFNet3

    paths = Paths.from_env()
    runs = Path(cfg.runs_dir) if cfg.runs_dir else paths.runs
    data_root = Path(cfg.data_root) if cfg.data_root else paths.data
    run_dir = runs / "m1" / cfg.run_id
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    if data is None:
        speech, noise, rirs = pilot_records(data_root)
        data = StageCData(speech, noise, rirs, None)

    device = cfg.device
    base, df_state = load_upstream_dfnet3(device=device)
    freeze_base(base)
    sha_before = state_dict_sha256(base)
    torch.manual_seed(cfg.seed)  # inicialização do gerador reprodutível (a camada de saída é zero; a oculta não)
    gen = FilmGenerator(load_film_config()).to(device)
    model = PDFNet3(base, gen)
    istft = Istft(960, HOP, torch.as_tensor(df_state.fft_window().copy())).to(device)
    params = ModelParams()
    trainable = [p for p in gen.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup_steps, 1)))

    chash, commit = config_hash(cfg.hashable()), git_commit()
    step, epoch, batch_idx = 0, 0, 0
    ck: Checkpoint | None = load_latest(ckpt_dir, expected_cfg_hash=chash)
    log_path = run_dir / "loss-log.jsonl"
    if ck is not None:
        gen.load_state_dict(ck.model)
        opt.load_state_dict(ck.optimizer)
        sched.load_state_dict(ck.scheduler)
        step, epoch, batch_idx = ck.step, int(ck.sampler_state["epoch"]), int(ck.sampler_state["batch"])
        restore_rng(ck)
        _truncate_log(log_path, step)
    elif log_path.exists():
        log_path.unlink()

    # o orçamento de tempo é do run inteiro: o tempo gasto antes de uma retomada conta (guardado no checkpoint)
    spent_before = float(ck.extra.get("elapsed_s", 0.0)) if ck is not None else 0.0
    t_start, last_ckpt, step_times, pss_max, vram_peak, vram_reserved = (
        time.monotonic(), time.monotonic(), [], 0, 0, 0)

    def elapsed() -> float:
        return spent_before + (time.monotonic() - t_start)
    last_path = ck.path if ck else None
    done = cfg.max_steps is not None and step >= cfg.max_steps
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def save(final: bool = False) -> Path:
        return save_checkpoint(ckpt_dir, step=step, model=gen.state_dict(), optimizer=opt, scheduler=sched,
                               sampler_state={"epoch": epoch, "batch": batch_idx}, cfg_hash=chash, commit=commit,
                               extra={"base_sha256": sha_before, "final": final, "elapsed_s": elapsed()})

    with open(log_path, "a") as log:
        while not done:
            ds = StageCDataset(data, bank, data_root, cfg, epoch=epoch)
            sampler = list(range(batch_idx * cfg.batch_size, len(ds)))
            dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers,
                            prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None)
            for batch in dl:
                t0 = time.perf_counter()
                b = _to_device(batch, device)
                gen.train()
                opt.zero_grad(set_to_none=True)
                spec, spec_clean = b["spec"], b["spec_clean"]
                spec_e, m, lsnr, _ = model(spec.clone(), b["feat_erb"], b["feat_spec"], b["embedding"], b["mask"])
                spec_m = base.mask(spec.clone(), m)
                out, _stages = emulate_gating_offline(spec, spec_m, spec_e, lsnr.reshape(spec.shape[0], -1))
                est, tgt = istft(out), istft(spec_clean)
                lsnr_t = _lsnr_target(params, spec_clean, spec)
                parts = stage_c_loss(est, tgt, lsnr, lsnr_t, lsnr_mask=b["lsnr_mask"],
                                     interferer_frames=b["interferer_frames"], max_freq=b["max_freq"])
                parts["total"].backward()
                nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                opt.step()
                sched.step()
                freeze_base(base)
                step += 1
                batch_idx += 1
                step_times.append(time.perf_counter() - t0)
                log.write(json.dumps({"step": step, "epoch": epoch, **{k: float(v.detach()) for k, v in parts.items()}})
                          + "\n")
                log.flush()
                if step % max(cfg.probe_every, 1) == 0 or step == 1:
                    pss_max = max(pss_max, total_pss())
                if (cfg.max_steps is not None and step >= cfg.max_steps) or elapsed() >= max_minutes * 60.0:
                    done = True
                    break
                if time.monotonic() - last_ckpt >= cfg.checkpoint_every_min * 60.0:
                    last_path, last_ckpt = save(), time.monotonic()
            else:
                epoch, batch_idx = epoch + 1, 0
                last_path, last_ckpt = save(), time.monotonic()
    pss_max = max(pss_max, total_pss())
    if device.startswith("cuda"):
        vram_peak = torch.cuda.max_memory_allocated(device)
        vram_reserved = torch.cuda.max_memory_reserved(device)
    sha_after = assert_base_unchanged(base, sha_before)
    last_path = save(final=True)
    limit = load_budget().loader_rss_gb_max
    probe = {"pss_gb_max": pss_max / GIB, "limit_gb": limit, "pass": pss_max / GIB <= limit,
             "num_workers": cfg.num_workers, "prefetch_factor": cfg.prefetch_factor, "batch_size": cfg.batch_size,
             "segment_s": cfg.segment_s, "steps": step, "device": device,
             "s_per_step_median": statistics.median(step_times) if step_times else None,
             "vram_peak_allocated_gb": vram_peak / GIB if vram_peak else None,
             "vram_peak_reserved_gb": vram_reserved / GIB if vram_reserved else None,
             "note": "PSS somado do trainer e dos workers do loader (psutil), amostrado a cada "
                     f"{cfg.probe_every} passos"}
    _atomic_json(runs / "m1" / PROBE_NAME, probe)
    _atomic_json(run_dir / "summary.json", {
        "run_id": cfg.run_id, "steps": step, "epoch": epoch, "base_sha256_before": sha_before,
        "base_sha256_after": sha_after, "trainable_params": sum(p.numel() for p in trainable), "config_hash": chash,
        "commit": commit, "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "elapsed_s": time.monotonic() - t_start, "elapsed_s_total": elapsed(), "last_checkpoint": last_path.name,
    })
    return last_path


def _truncate_log(path: Path, step: int) -> None:
    """Descarta do log os passos além do checkpoint retomado (uma queda deixa passos sem checkpoint)."""
    if not path.exists():
        return
    kept = [ln for ln in path.read_text().splitlines() if ln and json.loads(ln)["step"] <= step]
    path.write_text("".join(ln + "\n" for ln in kept))


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def enrollment_embedder(model: nn.Module, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """Adapta um `EnrollmentModel` (áudio 16 kHz [1, N] -> ..., embedding) a `audio -> embedding [192]`."""
    model = model.to(device).eval()

    def embed(audio16k: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.from_numpy(np.asarray(audio16k, np.float32))[None].to(device))
        return out[4].reshape(-1).cpu().numpy()

    return embed
