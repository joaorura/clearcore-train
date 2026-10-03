"""Banco de embeddings do M2 (spec §4.4): ~20 variantes de enrollment por falante de treino, guardadas só como 192-d.

Cada variante é um enrollment de 6-12 s montado como o do produto: falas curtas do falante concatenadas com pausas de
150-400 ms (48 kHz), aumentação de canal (RIR com p=0,3; ruído com SNR U[0,30] dB e p=0,7; denoise pelo DFNet3
upstream sem perfil com p=0,5), `resampler-48k-16k-v1` até 16 kHz e a validação do Clearcore
(`check_enrollment_audio`); variante recusada é sorteada de novo. O áudio nunca é guardado (dado biométrico): só o
embedding do encoder congelado.

Determinismo: a variante `v` do falante `k` depende só de `(seed, k, v)`, nunca da ordem dos falantes. O trabalho de CPU
(carga, aumentação, DFNet3, reamostragem) roda em processos; o encoder roda no processo principal (GPU, se houver).
Retomável: `cache_dir` guarda um arquivo por falante e a assinatura da configuração; assinatura diferente é recusada.

Aviso (spec §4.4 pede "outro trecho, nunca o da mistura"): como o banco é pré-computado, o enrollment de um falante pode
conter a fala que depois aparece na mistura de treino. O banco não tem como evitar isso; está nos riscos do relatório.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from cctrain.data.audio import load_audio, resample
from cctrain.data.enroll import EnrollmentRejected, check_enrollment_audio
from cctrain.data.features import SR as SR48
from cctrain.data.manifest import FileRecord, SourceManifest, load_sources, training_records
from cctrain.data.mix import apply_rir, loop_to_length, scale_for_ratio
from cctrain.data.resample import to_16k
from cctrain.hashing import sha256_file

EMB_DIM = 192
OFFLINE_DELAY = 480  # a síntese do libdf atrasa 1 hop (medido no M0; igual a eval.functional.OFFLINE_DELAY)
PIPELINE = ("trim de silêncio das bordas -> concat 6-12 s (pausas 150-400 ms) -> RIR -> ruído -> denoise DFNet3 -> resampler-48k-16k-v1 -> "
            "check_enrollment_audio")


class EmbeddingBankError(RuntimeError):
    """Banco ou cache de banco inconsistente. Falha fechado."""


class EmbeddingBank(Protocol):
    """Banco de embeddings pré-computado: variantes de enrollment por falante de treino, só os 192-d."""

    def speakers(self) -> Sequence[str]: ...

    def sample(self, speaker_key: str, rng: np.random.Generator) -> np.ndarray: ...


class ArrayEmbeddingBank:
    """`EmbeddingBank` em memória, persistido em `.npz` com um sidecar `<nome>.meta.json` (SHA-256 do `.npz`)."""

    def __init__(self, variants: Mapping[str, np.ndarray], meta: Mapping | None = None):
        self._v = {k: np.asarray(v, np.float32).reshape(-1, EMB_DIM) for k, v in variants.items()}
        self.meta: dict = dict(meta or {})

    def speakers(self) -> list[str]:
        return sorted(self._v)

    def n_variants(self, speaker_key: str) -> int:
        return len(self._v[speaker_key])

    def all(self, speaker_key: str) -> np.ndarray:
        return self._v[speaker_key]

    def sample(self, speaker_key: str, rng: np.random.Generator) -> np.ndarray:
        v = self._v[speaker_key]
        return v[int(rng.integers(len(v)))]

    def save(self, path: Path, extra_meta: Mapping | None = None) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **self._v)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        meta = {**self.meta, **(extra_meta or {}), "sha256": sha256_file(path), "n_speakers": len(self._v),
                "n_embeddings": int(sum(len(v) for v in self._v.values()))}
        mpath = path.with_name(path.stem + ".meta.json")
        mtmp = mpath.with_name(mpath.name + ".tmp")
        mtmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        os.replace(mtmp, mpath)
        self.meta = meta
        return meta

    @classmethod
    def load(cls, path: Path) -> ArrayEmbeddingBank:
        """Sem sidecar, carrega sem conferir (bancos antigos); com sidecar, o SHA-256 tem de bater."""
        path = Path(path)
        mpath = path.with_name(path.stem + ".meta.json")
        meta = json.loads(mpath.read_text()) if mpath.is_file() else {}
        if meta.get("sha256") and sha256_file(path) != meta["sha256"]:
            raise EmbeddingBankError(f"{path}: sha256 não confere com {mpath.name}")
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files}, meta)


# --- um enrollment -------------------------------------------------------------------------------------------------


def _load48(root: Path, rel: str) -> np.ndarray:
    x, sr = load_audio(Path(root) / rel)
    return resample(x, sr, SR48)


def trim_silence(x: np.ndarray, *, frame: int = 960, range_db: float = 35.0, floor_dbfs: float = -60.0,
                 margin: int = 2_400) -> np.ndarray:
    """Tira o silêncio das bordas (quadros de 20 ms abaixo de `range_db` do máximo ou do piso), com 50 ms de margem.
    Gravações como o VCTK mic1 têm ~1 s de silêncio por fala: sem isso o enrollment reprova na fala ativa (< 60%).
    Sinal sem quadro ativo volta intacto (a validação o recusa)."""
    x = np.asarray(x, np.float32)
    n = x.size // frame
    if n == 0:
        return x
    level = 10 * np.log10(np.mean(x[: n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1) + 1e-12)
    active = np.flatnonzero((level >= level.max() - range_db) & (level > floor_dbfs))
    if active.size == 0:
        return x
    return x[max(0, active[0] * frame - margin) : min(x.size, (active[-1] + 1) * frame + margin)]


def concat_utterances48(rng: np.random.Generator, utterances: Sequence[str], data_root: Path, *,
                        trim: bool = True) -> np.ndarray:
    """Falas do falante concatenadas com pausas de 150-400 ms até U[6, 12] s (48 kHz, comprimento múltiplo de 3)."""
    want = float(rng.uniform(6.0, 12.0))
    n_want = int(want * 16_000) * 3
    parts, total, guard = [], 0, 0
    while total < n_want and guard < 4 * len(utterances) + 8:
        for k in rng.permutation(len(utterances)):
            x = _load48(data_root, utterances[int(k)])
            x = trim_silence(x) if trim else x
            gap = np.zeros(int(rng.uniform(0.15, 0.40) * SR48), np.float32)
            parts += [x, gap]
            total += x.size + gap.size
            guard += 1
            if total >= n_want:
                break
    return np.concatenate(parts)[:n_want].astype(np.float32)


def make_enrollment_audio(rng: np.random.Generator, *, utterances: Sequence[str], data_root: Path,
                          noise: Sequence[str], rirs: Sequence[str], denoise: Callable[[np.ndarray], np.ndarray] | None,
                          p_noise: float = 0.7, snr_db: tuple[float, float] = (0.0, 30.0), p_rir: float = 0.3,
                          p_denoise: float = 0.5, max_tries: int = 8, trim: bool = True) -> np.ndarray | None:
    """Enrollment de 6-12 s a 16 kHz (float32) que passa em `check_enrollment_audio`, ou `None` se as `max_tries`
    tentativas forem recusadas. `denoise`: áudio 48 kHz -> 48 kHz alinhado (None desliga)."""
    if p_noise > 0 and not noise:
        raise EmbeddingBankError("p_noise > 0 exige um pool de ruído")
    if p_rir > 0 and not rirs:
        raise EmbeddingBankError("p_rir > 0 exige um pool de RIR")
    if not utterances:
        raise EmbeddingBankError("falante sem falas")
    for _ in range(max_tries):
        x = concat_utterances48(rng, utterances, data_root, trim=trim)
        if rirs and rng.random() < p_rir:
            x = apply_rir(x, _load48(data_root, rirs[int(rng.integers(len(rirs)))]))
        if noise and rng.random() < p_noise:
            nz = loop_to_length(_load48(data_root, noise[int(rng.integers(len(noise)))]), x.size, rng)
            x = (x.astype(np.float64) + scale_for_ratio(x, nz, float(rng.uniform(*snr_db)))).astype(np.float32)
        if denoise is not None and rng.random() < p_denoise:
            x = np.asarray(denoise(x), np.float32)
        audio = to_16k(x, SR48)
        try:
            check_enrollment_audio(audio)
        except EnrollmentRejected:
            continue
        return audio.astype(np.float32)
    return None


def dfnet3_denoiser(device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """Denoise pelo DFNet3 upstream, sem perfil, 48 kHz -> 48 kHz do mesmo tamanho (atraso da síntese removido)."""
    from cctrain.eval.functional import enhance_offline
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3

    base, _ = load_upstream_dfnet3(device=device)

    def denoise(x48: np.ndarray) -> np.ndarray:
        y, _ = enhance_offline(base, x48)
        y = np.asarray(y, np.float32)[OFFLINE_DELAY:]
        return np.concatenate([y, np.zeros(max(0, len(x48) - y.size), np.float32)])[: len(x48)]

    return denoise


# --- o banco -----------------------------------------------------------------------------------------------------


_STATE: dict = {}


def _speaker_rng(seed: int, key: str, v: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), int(v)])


def _init_worker(cfg: dict) -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass
    factory = cfg["denoise_factory"]
    _STATE.clear()
    _STATE.update(cfg, denoise=factory() if factory is not None else None)


def _speaker_job(key: str) -> tuple[str, list[np.ndarray]]:
    s = _STATE
    out = []
    for v in range(s["variants"]):
        a = make_enrollment_audio(_speaker_rng(s["seed"], key, v), utterances=s["utterances"][key],
                                  data_root=s["data_root"], noise=s["noise"], rirs=s["rirs"], denoise=s["denoise"],
                                  p_noise=s["p_noise"], snr_db=s["snr_db"], p_rir=s["p_rir"],
                                  p_denoise=s["p_denoise"])
        if a is not None:
            out.append(a)
    return key, out


def _shard(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.npz"


def _signature(**kw) -> dict:
    return json.loads(json.dumps(kw, sort_keys=True))


def build_bank(encoder: Callable[[np.ndarray], np.ndarray], records: Sequence[FileRecord], *, data_root: Path,
               variants: int = 20, seed: int = 0, noise: Sequence[FileRecord] = (), rirs: Sequence[FileRecord] = (),
               denoise_factory: Callable[[], Callable[[np.ndarray], np.ndarray]] | None = None,
               p_noise: float = 0.7, snr_db: tuple[float, float] = (0.0, 30.0), p_rir: float = 0.3,
               p_denoise: float = 0.5, workers: int = 0, cache_dir: Path | None = None,
               encoder_tag: str = "", denoise_tag: str = "", sources: Mapping[str, SourceManifest] | None = None,
               allow_pilot: bool = False, log: Callable[[str], None] | None = None) -> ArrayEmbeddingBank:
    """`variants` enrollments por falante de treino -> embeddings (`encoder`: áudio 16 kHz -> 192-d).

    Só registros de treino entram (`training_records` recusa dev/teste). `workers=0` roda tudo no processo."""
    sources = sources or load_sources()
    speech = training_records(records, sources, allow_pilot=allow_pilot)
    by: dict[str, list[str]] = {}
    for r in speech:
        if r.speaker_key is None:
            raise EmbeddingBankError(f"{r.path}: registro de fala sem falante")
        by.setdefault(r.speaker_key, []).append(r.path)
    by = {k: sorted(set(v)) for k, v in sorted(by.items())}
    noise_p, rir_p = sorted(r.path for r in training_records(noise, sources)), sorted(
        r.path for r in training_records(rirs, sources))
    sig = _signature(variants=variants, seed=seed, p_noise=p_noise, snr_db=list(snr_db), p_rir=p_rir,
                     p_denoise=p_denoise, pipeline=PIPELINE, encoder_tag=encoder_tag, denoise_tag=denoise_tag,
                     noise_sha=hashlib.sha256("\n".join(noise_p).encode()).hexdigest(),
                     rir_sha=hashlib.sha256("\n".join(rir_p).encode()).hexdigest())
    done: dict[str, np.ndarray] = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cfile = cache_dir / "config.json"
        if cfile.is_file():
            if json.loads(cfile.read_text()) != sig:
                raise EmbeddingBankError(f"cache em {cache_dir} é de outra configuração (assinatura difere)")
        else:
            cfile.write_text(json.dumps(sig, indent=2, sort_keys=True) + "\n")
        for k in by:
            sp = _shard(cache_dir, k)
            if sp.is_file():
                try:
                    with np.load(sp) as z:
                        done[k] = np.asarray(z["emb"], np.float32).reshape(-1, EMB_DIM)
                except (OSError, ValueError, KeyError):
                    sp.unlink()  # shard truncado por uma queda: refaz
    todo = [k for k in by if k not in done]
    say = log or (lambda _m: None)
    say(f"banco: {len(by)} falantes, {len(done)} no cache, {len(todo)} a gerar, {variants} variantes cada")
    cfg = {"utterances": by, "data_root": Path(data_root), "noise": noise_p, "rirs": rir_p,
           "denoise_factory": denoise_factory, "variants": variants, "seed": seed, "p_noise": p_noise,
           "snr_db": tuple(snr_db), "p_rir": p_rir, "p_denoise": p_denoise}
    t0 = time.time()

    def take(res: tuple[str, list[np.ndarray]], n: int) -> None:
        key, audios = res
        emb = np.stack([np.asarray(encoder(a), np.float32).reshape(EMB_DIM) for a in audios]) if audios else \
            np.zeros((0, EMB_DIM), np.float32)
        done[key] = emb
        if cache_dir is not None:
            sp = _shard(cache_dir, key)
            tmp = sp.with_name(sp.name + ".tmp")
            with open(tmp, "wb") as f:
                np.savez(f, emb=emb)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, sp)
        say(f"[{n}/{len(todo)}] {len(emb)} variantes ({time.time() - t0:.0f} s)")

    if workers > 0 and todo:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for n, res in enumerate(pool.imap_unordered(_speaker_job, todo, chunksize=1), 1):
                take(res, n)
    else:
        _init_worker(cfg)
        for n, k in enumerate(todo, 1):
            take(_speaker_job(k), n)
    skipped = sorted(k for k in by if len(done[k]) == 0)
    short = {k: len(done[k]) for k in by if 0 < len(done[k]) < variants}
    meta = {"variants_per_speaker": variants, "seed": seed, "p_noise": p_noise, "snr_db": list(snr_db),
            "p_rir": p_rir, "p_denoise": p_denoise, "n_noise": len(noise_p), "n_rir": len(rir_p),
            "encoder_tag": encoder_tag, "denoise_tag": denoise_tag, "pipeline": PIPELINE,
            "skipped_speakers": skipped, "short_speakers": short}
    return ArrayEmbeddingBank({k: v for k, v in done.items() if len(v)}, meta)
