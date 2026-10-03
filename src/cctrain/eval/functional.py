"""Gate funcional do M1 (spec §3.1) e gate do desvio offline × runtime (spec §3.5).

`functional_gate`: ΔSI-SDR do estágio C contra o DFNet3 upstream, em misturas de mesmo sexo a SIR 0 dB do **dev**,
com IC 95% por bootstrap emparelhado por falante (a unidade de reamostragem é o falante). Os dois lados passam
pelo mesmo caminho, o de `path`: `offline` (forward PyTorch + máscara por estágio + síntese) ou `replica` (o
runtime: a função `runtime(audio, film)` devolve um `ReplicaResult`; com FiLM para o candidato e sem FiLM para o
DFNet3). `deviation_gate` escolhe o caminho a partir dos níveis congelados em `runtime-deviation-v1.yaml`; ele é
não bloqueante: nunca reprova, só escolhe.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from libdf import DF

from cctrain.data.audio import load_audio, resample
from cctrain.data.enroll import EnrollmentRejected, check_enrollment_audio
from cctrain.data.features import SR, extract_features
from cctrain.data.manifest import FileRecord
from cctrain.data.mix import MixKind, crop_or_pad, loop_to_length, make_mixture, mix_rng
from cctrain.data.resample import to_16k
from cctrain.eval.gates import GatesError, load_gates, verify_frozen
from cctrain.eval.runtime_path import (
    HOP,
    LOOKAHEAD,
    ReplicaResult,
    Stage,
    emulate_gating_offline,
)
from cctrain.paths import repo_root

DEVIATION_V1 = repo_root() / "configs" / "eval" / "runtime-deviation-v1.yaml"
OFFLINE_DELAY = HOP  # a síntese do libdf atrasa 1 hop em relação à entrada (medido no M0)
RUNTIME_DELAY = 1440  # latência do runtime (G0: latency_samples)
DEV_SPLITS = ("dev",)
PathName = Literal["offline", "replica"]
Runtime = Callable[[np.ndarray, "np.ndarray | None"], ReplicaResult]


# --- gate do desvio ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviationLevels:
    error_to_signal_db_max: float
    stage_agreement_min: float
    blocking: bool
    sha256: str


def load_deviation_levels(path: Path = DEVIATION_V1) -> DeviationLevels:
    digest = verify_frozen(Path(path))
    d = yaml.safe_load(Path(path).read_text())
    if d.get("version") != 1:
        raise GatesError("runtime-deviation: version deve ser 1")
    lv = d["levels"]
    return DeviationLevels(float(lv["error_to_signal_db_max"]), float(lv["stage_agreement_min"]),
                           bool(d["blocking"]), digest)


def deviation_gate(result: Mapping) -> Literal["offline", "replica"]:
    """`offline` se erro/sinal <= nível e concordância de estágio >= nível; senão `replica`. Valor não finito
    (NaN) não passa. Não bloqueante: devolve só a escolha de caminho do dev."""
    lv = load_deviation_levels()
    err, agree = float(result["error_to_signal_db"]), float(result["stage_agreement"])
    ok = (math.isfinite(err) and math.isfinite(agree) and err <= lv.error_to_signal_db_max
          and agree >= lv.stage_agreement_min)
    return "offline" if ok else "replica"


def deviation_metrics(offline_audio: np.ndarray, offline_stages, runtime: ReplicaResult) -> dict:
    """Mesmas medidas de `m0.offline_probe.offline_vs_runtime`, sem depender do modelo: o áudio offline sai da
    síntese do libdf (1 hop de atraso) e o do runtime tem `LOOKAHEAD` hops a mais; silêncio fora da conta."""
    stages = np.asarray(offline_stages.numpy() if hasattr(offline_stages, "numpy") else offline_stages)
    stages = stages.reshape(-1) if stages.ndim == 1 else stages[0]
    shift = LOOKAHEAD * HOP
    rt, off = runtime.audio[shift:], np.asarray(offline_audio, np.float32)[: len(offline_audio) - shift]
    n = min(rt.size, off.size)
    diff = rt[:n] - off[:n]
    valid = runtime.stages[LOOKAHEAD:] != int(Stage.SILENT)
    agree = stages[: valid.size][valid] == runtime.stages[LOOKAHEAD:][valid]
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "error_to_signal_db": float(10 * np.log10(max(np.mean(diff.astype(np.float64) ** 2), 1e-30)
                                                   / max(np.mean(rt[:n].astype(np.float64) ** 2), 1e-20))),
        "stage_agreement": float(np.mean(agree)),
        "stage_agreement_frames": int(valid.sum()),
        "silent_fraction_excluded": float(1.0 - np.mean(valid)),
    }


# --- métricas ----------------------------------------------------------------------------------------------------


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    """SI-SDR em dB (como `df.loss.SiSdr`: média removida, projeção do estimado no alvo)."""
    est = np.asarray(est, np.float64) - np.mean(est)
    ref = np.asarray(ref, np.float64) - np.mean(ref)
    alpha = (est @ ref + eps) / (ref @ ref + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((target @ target + eps) / (noise @ noise + eps)))


def speaker_bootstrap(deltas: Sequence[float], speakers: Sequence[str], *, resamples: int = 10_000,
                      ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Média e IC percentil do ΔSI-SDR por bootstrap emparelhado (o delta já é por mistura, emparelhado) com
    unidade de reamostragem = falante: média por falante, reamostra falantes com reposição (`gates-v1.statistics`)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for d, s in zip(deltas, speakers, strict=True):
        groups[s].append(float(d))
    per = np.array([np.mean(v) for _, v in sorted(groups.items())])
    mean = float(per.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per.size, size=(resamples, per.size))
    boots = per[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return mean, float(lo), float(hi)


# --- misturas de dev ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DevMixture:
    noisy: np.ndarray
    clean: np.ndarray
    speaker: str
    interferer_speaker: str
    embedding: np.ndarray
    sir_db: float
    snr_db: float
    mixture_path: str
    enrollment_paths: tuple[str, ...]


def read_librispeech_sex(path: Path, source_id: str = "librispeech-dev-clean") -> dict[str, str]:
    """`SPEAKERS.TXT` do LibriSpeech -> {`source_id:leitor`: 'F'|'M'} (a chave é a `speaker_key` do manifest)."""
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[1] in ("F", "M"):
            out[f"{source_id}:{parts[0]}"] = parts[1]
    return out


def _load48(root: Path, r: FileRecord) -> np.ndarray:
    x, sr = load_audio(Path(root) / r.path)
    return resample(x, sr, SR)


def _enrollment(root: Path, pool: Sequence[FileRecord], exclude: str, rng: np.random.Generator, embed,
                tries: int = 8) -> tuple[np.ndarray, tuple[str, ...]] | None:
    """Enrollment de 6-12 s a 16 kHz: outras falas do mesmo falante (nunca a da mistura), com pausas de 150-400 ms."""
    cands = [r for r in pool if r.path != exclude]
    if not cands:
        return None
    for _ in range(tries):
        want = float(rng.uniform(6.0, 12.0))
        parts, used, total = [], [], 0.0
        for k in rng.permutation(len(cands)):
            r = cands[int(k)]
            x, sr = load_audio(Path(root) / r.path)
            x16 = to_16k(x, sr)
            parts += [x16, np.zeros(int(rng.uniform(0.15, 0.40) * 16_000), np.float32)]
            used.append(r.path)
            total += (len(x16) / 16_000) + 0.3
            if total >= want:
                break
        audio = np.concatenate(parts)[: int(want * 16_000)].astype(np.float32)
        try:
            check_enrollment_audio(audio)
        except EnrollmentRejected:
            continue
        return embed(audio), tuple(used)
    return None


def build_dev_mixtures(speech: Sequence[FileRecord], noise: Sequence[FileRecord], data_root: Path,
                       sex: Mapping[str, str], embed: Callable[[np.ndarray], np.ndarray], *, n_mixtures: int,
                       segment_s: float = 3.0, sir_db: float = 0.0, snr_db: float = 10.0, seed: int = 0,
                       allowed_splits: Sequence[str] = DEV_SPLITS) -> list[DevMixture]:
    """Misturas alvo + interferente + ruído do mesmo sexo, SIR `sir_db`, só de registros do dev (nunca treino/teste)."""
    bad = sorted({r.split for r in speech if r.split not in allowed_splits})
    if bad:
        raise ValueError(f"split(s) {bad} fora de {list(allowed_splits)}: o gate funcional mede só no dev")
    data_root = Path(data_root)
    n = int(segment_s * SR)
    by_spk: dict[str, list[FileRecord]] = defaultdict(list)
    for r in speech:
        if r.speaker_key is None or r.speaker_key not in sex:
            continue
        by_spk[r.speaker_key].append(r)
    speakers = sorted(by_spk)
    noise = list(noise)
    if len(speakers) < 2 or not noise:
        raise ValueError("o dev precisa de pelo menos 2 falantes com sexo conhecido e 1 ruído")
    out: list[DevMixture] = []
    attempt = 0
    while len(out) < n_mixtures:
        if attempt > 20 * max(n_mixtures, 1):
            raise ValueError(f"só {len(out)} de {n_mixtures} misturas possíveis (falantes do mesmo sexo/enrollment)")
        rng = mix_rng(seed, 0, attempt)
        spk = speakers[attempt % len(speakers)]
        attempt += 1
        peers = [s for s in speakers if s != spk and sex[s] == sex[spk]]
        if not peers:
            continue
        ispk = peers[int(rng.integers(len(peers)))]
        t_rec = by_spk[spk][int(rng.integers(len(by_spk[spk])))]
        i_rec = by_spk[ispk][int(rng.integers(len(by_spk[ispk])))]
        enroll = _enrollment(data_root, by_spk[spk], t_rec.path, rng, embed)
        if enroll is None:
            continue
        emb, used = enroll
        target = crop_or_pad(_load48(data_root, t_rec), n, rng)
        interf = crop_or_pad(_load48(data_root, i_rec), n, rng)
        nz = loop_to_length(_load48(data_root, noise[int(rng.integers(len(noise)))]), n, rng)
        m = make_mixture(target, [interf], nz, kind=MixKind.TARGET_INTERFERER_NOISE, snr_db=snr_db, sir_db=sir_db)
        out.append(DevMixture(m.noisy, m.clean, spk, ispk, np.asarray(emb, np.float32), sir_db, snr_db,
                              t_rec.path, used))
    return out


# --- caminhos de avaliação ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    path: str
    n_mixtures: int
    n_speakers: int
    delta_si_sdr_db: float
    ci_low: float
    ci_high: float
    baseline_si_sdr_db: float
    candidate_si_sdr_db: float
    passed: bool
    ci_lower_gt: float
    stage_fractions: dict


def _synthesis(spec: torch.Tensor) -> np.ndarray:
    d = DF(SR, 960, HOP, 32, 2)
    c = torch.view_as_complex(spec[:, 0].contiguous()).cpu().numpy()
    return np.asarray(d.synthesis(np.ascontiguousarray(c)), np.float32).reshape(-1)


def enhance_offline(base, audio: np.ndarray, *, film_model=None, embedding: np.ndarray | None = None
                    ) -> tuple[np.ndarray, torch.Tensor]:
    """Forward + gating emulado + síntese. Com `film_model` (PDFNet3) e `embedding`, usa o perfil (mask=1)."""
    device = next(base.parameters()).device
    f = extract_features(audio)
    spec, erb, specf = (torch.from_numpy(a)[None].to(device) for a in (f.spec, f.feat_erb, f.feat_spec))
    with torch.no_grad():
        if film_model is None:
            spec_e, m, lsnr, _ = base(spec.clone(), erb, specf)
        else:
            emb = torch.from_numpy(np.asarray(embedding, np.float32))[None].to(device)
            spec_e, m, lsnr, _ = film_model(spec.clone(), erb, specf, emb, torch.ones(1, 1, device=device))
        spec_m = base.mask(spec.clone(), m)
        out, stages = emulate_gating_offline(spec, spec_m, spec_e, lsnr.reshape(1, -1))
    return _synthesis(out), stages.cpu()


def film_vector(generator, embedding: np.ndarray) -> np.ndarray:
    """`[gamma_enc, beta_enc, gamma_df, beta_df]`, 4 x 256 f32: o formato de `tract-check gating ... film.f32`."""
    with torch.no_grad():
        v = generator(torch.from_numpy(np.asarray(embedding, np.float32))[None], torch.ones(1, 1))
    return torch.cat([t.reshape(-1) for t in (v.gamma_enc, v.beta_enc, v.gamma_df, v.beta_df)]).numpy().astype(
        np.float32)


def _aligned(est: np.ndarray, delay: int, n: int) -> np.ndarray:
    est = np.asarray(est, np.float32)[delay:]
    return np.concatenate([est, np.zeros(max(0, n - est.size), np.float32)])[:n]


def tract_runtime(asset: Path, workdir: Path) -> Runtime:
    """Runtime pelo `tract-check gating` (o próprio runtime do Clearcore) sobre o asset com FiLM de identidade."""
    from cctrain.eval.runtime_path import tract_gating

    counter = iter(range(10**9))

    def run(audio: np.ndarray, film: np.ndarray | None) -> ReplicaResult:
        return tract_gating(Path(asset), audio, Path(workdir) / f"rt{next(counter)}", film=film)

    return run


def functional_gate(ckpt: Path, dev_mixtures: Sequence[DevMixture], *, path: PathName = "offline",
                    runtime: Runtime | None = None, base=None, resamples: int = 10_000, seed: int = 0
                    ) -> GateResult:
    """ΔSI-SDR (estágio C − DFNet3) por mistura, no caminho `path`, com IC 95% por falante. `passed` é o nível do
    M1 (`gates-v1.m1.functional`): limite inferior do IC > `delta_si_sdr_ci_lower_gt`."""
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.models.film import FilmGenerator, load_film_config
    from cctrain.models.pdfnet3 import PDFNet3
    from cctrain.train.checkpoint import load_checkpoint

    if path not in ("offline", "replica"):
        raise ValueError(f"path inválido: {path!r}")
    if path == "replica" and runtime is None:
        raise ValueError("path='replica' exige `runtime` (função áudio, film -> ReplicaResult)")
    if not dev_mixtures:
        raise ValueError("sem misturas de dev")
    if base is None:
        base, _ = load_upstream_dfnet3(device="cpu")
    base.eval()
    gen = FilmGenerator(load_film_config())
    gen.load_state_dict(load_checkpoint(Path(ckpt)).model)
    gen.eval()
    model = PDFNet3(base, gen).eval()
    level = float(load_gates().m1["functional"]["delta_si_sdr_ci_lower_gt"])
    deltas, spk, b_all, c_all = [], [], [], []
    stage_counts = np.zeros(len(Stage))
    for m in dev_mixtures:
        n = len(m.clean)
        if path == "offline":
            yb, _ = enhance_offline(base, m.noisy)
            yc, stages = enhance_offline(base, m.noisy, film_model=model, embedding=m.embedding)
            yb, yc = _aligned(yb, OFFLINE_DELAY, n), _aligned(yc, OFFLINE_DELAY, n)
            st = stages.numpy().reshape(-1)
        else:
            rb = runtime(m.noisy, None)
            rc = runtime(m.noisy, film_vector(gen, m.embedding))
            yb, yc = _aligned(rb.audio, RUNTIME_DELAY, n), _aligned(rc.audio, RUNTIME_DELAY, n)
            st = rc.stages
        sb, sc = si_sdr(yb, m.clean), si_sdr(yc, m.clean)
        b_all.append(sb), c_all.append(sc), deltas.append(sc - sb), spk.append(m.speaker)
        stage_counts += np.bincount(st.astype(np.int64), minlength=len(Stage))[: len(Stage)]
    mean, lo, hi = speaker_bootstrap(deltas, spk, resamples=resamples, seed=seed)
    fractions = {s.name.lower(): float(stage_counts[int(s)] / stage_counts.sum()) for s in Stage}
    return GateResult(path=path, n_mixtures=len(deltas), n_speakers=len(set(spk)), delta_si_sdr_db=mean, ci_low=lo,
                      ci_high=hi, baseline_si_sdr_db=float(np.mean(b_all)), candidate_si_sdr_db=float(np.mean(c_all)),
                      passed=bool(lo > level), ci_lower_gt=level, stage_fractions=fractions)

