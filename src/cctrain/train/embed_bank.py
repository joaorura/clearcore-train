"""Banco de embeddings do M2 (spec §4.4): ~20 variantes de enrollment por falante de treino, guardadas só como 192-d.

Cada variante é um enrollment de 6-12 s montado como o do produto: falas curtas do falante concatenadas com pausas de
150-400 ms (48 kHz), aumentação de canal (RIR com p=0,3; ruído com SNR U[0,30] dB e p=0,7; denoise pelo DFNet3
upstream sem perfil com p=0,5), `resampler-48k-16k-v1` até 16 kHz e a validação do Clearcore
(`check_enrollment_audio`); variante recusada é sorteada de novo. O áudio nunca é guardado (dado biométrico): só o
embedding do encoder congelado.

Determinismo: a variante `v` do falante `k` depende só de `(seed, k, v)`, nunca da ordem dos falantes. O trabalho de CPU
(carga, aumentação, DFNet3, reamostragem) roda em processos; o encoder roda no processo principal (GPU, se houver).
Retomável: `cache_dir` guarda um arquivo por falante e a assinatura da configuração; assinatura diferente é recusada.

Formato v2 (M2 Tarefa 2): cada variante guarda a proveniência (`utterance_ids` das falas que a compõem, `denoised`,
`snr_db`, `rir`), e `EmbeddingBank.sample(..., exclude_utterances=, p_denoised=)` só devolve variante sem a fala da
mistura (spec §4.4: "outro trecho, nunca o da mistura") e com a fração de denoisadas pedida (controle exato da AB1).
O flag `denoised` de cada variante é balanceado por falante (agenda de Bresenham com fase por falante), não sorteado:
a fração por falante é `p_denoise` a menos de 1 variante, mesmo com variantes recusadas. Bancos v1 (sem proveniência)
continuam carregando, mas `sample` com `exclude_utterances`/`p_denoised` falha fechado neles.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
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
BANK_FORMAT = 2
PIPELINE = ("[trim de silêncio das bordas, opcional] -> concat 6-12 s (pausas 150-400 ms) -> RIR -> ruído -> denoise DFNet3 -> resampler-48k-16k-v1 -> "
            "check_enrollment_audio")


class EmbeddingBankError(RuntimeError):
    """Banco ou cache de banco inconsistente. Falha fechado."""


class EmbeddingBank(Protocol):
    """Banco de embeddings pré-computado: variantes de enrollment por falante de treino, só os 192-d."""

    def speakers(self) -> Sequence[str]: ...

    def sample(self, speaker_key: str, rng: np.random.Generator, *, exclude_utterances: Iterable[str] = (),
               p_denoised: float | None = None) -> np.ndarray: ...


@dataclass(frozen=True)
class SpeakerVariants:
    """Proveniência das variantes de um falante, alinhada às linhas de `emb`."""

    utterance_ids: tuple[tuple[str, ...], ...]  # por variante: caminhos (ids de registro) das falas usadas
    denoised: np.ndarray  # bool (n,)
    snr_db: np.ndarray  # float32 (n,), NaN = sem ruído
    rir: np.ndarray  # bool (n,)

    def __post_init__(self) -> None:
        n = len(self.utterance_ids)
        for name in ("denoised", "snr_db", "rir"):
            if len(getattr(self, name)) != n:
                raise EmbeddingBankError(f"proveniência inconsistente: {name} tem {len(getattr(self, name))} != {n}")
        object.__setattr__(self, "_sets", tuple(frozenset(u) for u in self.utterance_ids))

    def __len__(self) -> int:
        return len(self.utterance_ids)

    def eligible(self, exclude: Iterable[str]) -> np.ndarray:
        ex = frozenset(exclude)
        return np.array([not (u & ex) for u in self._sets], bool)  # type: ignore[attr-defined]


class ArrayEmbeddingBank:
    """`EmbeddingBank` em memória, persistido em `.npz` com um sidecar `<nome>.meta.json` (SHA-256 do `.npz`)."""

    def __init__(self, variants: Mapping[str, np.ndarray], meta: Mapping | None = None,
                 info: Mapping[str, SpeakerVariants] | None = None):
        self._v = {k: np.asarray(v, np.float32).reshape(-1, EMB_DIM) for k, v in variants.items()}
        self._info = dict(info or {})
        for k, i in self._info.items():
            if k not in self._v or len(i) != len(self._v[k]):
                raise EmbeddingBankError(f"{k}: proveniência não alinha com as variantes")
        self.meta: dict = dict(meta or {})

    @property
    def has_provenance(self) -> bool:
        return bool(self._info) and set(self._info) == set(self._v)

    def speakers(self) -> list[str]:
        return sorted(self._v)

    def n_variants(self, speaker_key: str) -> int:
        return len(self._v[speaker_key])

    def all(self, speaker_key: str) -> np.ndarray:
        return self._v[speaker_key]

    def variants(self, speaker_key: str) -> SpeakerVariants:
        if speaker_key not in self._info:
            raise EmbeddingBankError(f"{speaker_key}: banco sem proveniência por variante (v1)")
        return self._info[speaker_key]

    def sample(self, speaker_key: str, rng: np.random.Generator, *, exclude_utterances: Iterable[str] = (),
               p_denoised: float | None = None) -> np.ndarray:
        """Uma variante do falante. Sem filtros, uniforme (compatível com o v1). `exclude_utterances`: nunca devolve
        variante que contenha alguma dessas falas. `p_denoised`: classe (denoisada ou crua) sorteada com essa
        probabilidade, depois uniforme na classe; em 0 e 1 só uma classe, sem recuo; em (0,1), se a classe sorteada
        não tem variante elegível, usa a outra. Sem variante elegível: erro (falha fechado)."""
        v = self._v[speaker_key]
        exclude = frozenset(exclude_utterances)
        if not exclude and p_denoised is None:
            return v[int(rng.integers(len(v)))]
        if speaker_key not in self._info:
            raise EmbeddingBankError(f"{speaker_key}: banco sem proveniência por variante; "
                                     "exclude_utterances/p_denoised exigem o formato v2")
        info = self._info[speaker_key]
        ok = info.eligible(exclude)
        if p_denoised is None:
            pool = np.flatnonzero(ok)
        else:
            if not 0.0 <= p_denoised <= 1.0:
                raise EmbeddingBankError(f"p_denoised fora de [0,1]: {p_denoised}")
            want = bool(rng.random() < p_denoised)
            pool = np.flatnonzero(ok & (info.denoised == want))
            if pool.size == 0 and 0.0 < p_denoised < 1.0:
                pool = np.flatnonzero(ok & (info.denoised != want))
        if pool.size == 0:
            raise EmbeddingBankError(f"{speaker_key}: nenhuma variante elegível (exclude={len(exclude)} falas, "
                                     f"p_denoised={p_denoised})")
        return v[int(pool[int(rng.integers(pool.size))])]

    def stats(self) -> dict:
        """Estatísticas de proveniência (só contagens; nenhum id de falante)."""
        if not self.has_provenance:
            return {}
        fr, n_raw, n_den = [], [], []
        exposed_all = exposed_raw = exposed_den = 0
        for k in self.speakers():
            i = self._info[k]
            fr.append(float(np.mean(i.denoised)))
            n_raw.append(int((~i.denoised).sum()))
            n_den.append(int(i.denoised.sum()))
            exp_all = exp_raw = exp_den = False
            for u in sorted({u for ids in i.utterance_ids for u in ids}):
                ok = i.eligible([u])
                exp_all |= not ok.any()
                exp_raw |= not (ok & ~i.denoised).any()
                exp_den |= not (ok & i.denoised).any()
            exposed_all, exposed_raw, exposed_den = exposed_all + exp_all, exposed_raw + exp_raw, exposed_den + exp_den
        fr_a = np.array(fr)
        dev = np.abs(fr_a - 0.5)
        return {
            "denoised_fraction_mean": float(fr_a.mean()), "denoised_fraction_min": float(fr_a.min()),
            "denoised_fraction_max": float(fr_a.max()), "denoised_fraction_max_abs_dev": float(dev.max()),
            "speakers_denoised_fraction_off_by_more_than_0.1": int((dev > 0.1).sum()),
            "speakers_lt10_raw": int(sum(n < 10 for n in n_raw)), "speakers_lt10_denoised": int(sum(n < 10 for n in n_den)),
            "speakers_lt20_variants": int(sum(len(self._v[k]) < 20 for k in self.speakers())),
            "speakers_zero_raw": int(sum(n == 0 for n in n_raw)),
            "speakers_with_an_utterance_in_every_variant": int(exposed_all),
            "speakers_with_an_utterance_in_every_raw_variant": int(exposed_raw),
            "speakers_with_an_utterance_in_every_denoised_variant": int(exposed_den),
        }

    def save(self, path: Path, extra_meta: Mapping | None = None) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        arrays: dict[str, np.ndarray] = {}
        if self.has_provenance:
            table = sorted({u for i in self._info.values() for ids in i.utterance_ids for u in ids})
            index = {u: n for n, u in enumerate(table)}
            arrays["__format__"] = np.array(BANK_FORMAT)
            arrays["__utterances__"] = np.array(table, dtype=str) if table else np.zeros(0, dtype="<U1")
            for k, v in self._v.items():
                i = self._info[k]
                width = max((len(ids) for ids in i.utterance_ids), default=0)
                pad = np.full((len(i), width), -1, np.int32)
                for r, ids in enumerate(i.utterance_ids):
                    pad[r, : len(ids)] = [index[u] for u in ids]
                arrays.update({f"emb/{k}": v, f"utt/{k}": pad, f"den/{k}": i.denoised.astype(bool),
                               f"snr/{k}": i.snr_db.astype(np.float32), f"rir/{k}": i.rir.astype(bool)})
        else:
            arrays = dict(self._v)
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        meta = {**self.meta, **(extra_meta or {}), "sha256": sha256_file(path), "n_speakers": len(self._v),
                "n_embeddings": int(sum(len(v) for v in self._v.values())),
                "format": BANK_FORMAT if self.has_provenance else 1}
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
            if "__format__" not in z.files:
                return cls({k: z[k] for k in z.files}, meta)
            if int(z["__format__"]) != BANK_FORMAT:
                raise EmbeddingBankError(f"{path}: formato {int(z['__format__'])} desconhecido")
            table = [str(u) for u in z["__utterances__"]]
            emb, info = {}, {}
            for name in z.files:
                if not name.startswith("emb/"):
                    continue
                k = name[4:]
                emb[k] = z[name]
                info[k] = SpeakerVariants(
                    tuple(tuple(table[j] for j in row if j >= 0) for row in z[f"utt/{k}"]),
                    z[f"den/{k}"].astype(bool), z[f"snr/{k}"].astype(np.float32), z[f"rir/{k}"].astype(bool))
            return cls(emb, meta, info)


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


def concat_utterances48_ids(rng: np.random.Generator, utterances: Sequence[str], data_root: Path, *,
                            trim: bool = True) -> tuple[np.ndarray, tuple[str, ...]]:
    """Falas do falante concatenadas com pausas de 150-400 ms até U[6, 12] s (48 kHz, comprimento múltiplo de 3), e os
    ids (caminhos) das falas usadas. Toda fala carregada contribui com amostras: o laço para quando o total alcança o
    alvo, e a última fala começa antes dele."""
    want = float(rng.uniform(6.0, 12.0))
    n_want = int(want * 16_000) * 3
    parts, used, total, guard = [], [], 0, 0
    while total < n_want and guard < 4 * len(utterances) + 8:
        for k in rng.permutation(len(utterances)):
            x = _load48(data_root, utterances[int(k)])
            x = trim_silence(x) if trim else x
            gap = np.zeros(int(rng.uniform(0.15, 0.40) * SR48), np.float32)
            parts += [x, gap]
            used.append(utterances[int(k)])
            total += x.size + gap.size
            guard += 1
            if total >= n_want:
                break
    return np.concatenate(parts)[:n_want].astype(np.float32), tuple(sorted(set(used)))


def concat_utterances48(rng: np.random.Generator, utterances: Sequence[str], data_root: Path, *,
                        trim: bool = True) -> np.ndarray:
    return concat_utterances48_ids(rng, utterances, data_root, trim=trim)[0]


@dataclass(frozen=True)
class EnrollmentVariant:
    """Um enrollment aceito (16 kHz) e a sua proveniência. O áudio é biométrico: nunca é guardado."""

    audio: np.ndarray
    utterance_ids: tuple[str, ...]
    denoised: bool
    snr_db: float | None  # None: sem ruído
    rir: bool


def make_enrollment_variant(rng: np.random.Generator, *, utterances: Sequence[str], data_root: Path,
                            noise: Sequence[str], rirs: Sequence[str],
                            denoise: Callable[[np.ndarray], np.ndarray] | None, p_noise: float = 0.7,
                            snr_db: tuple[float, float] = (0.0, 30.0), p_rir: float = 0.3, p_denoise: float = 0.5,
                            denoise_flag: bool | None = None, max_tries: int = 16, trim: bool = True
                            ) -> EnrollmentVariant | None:
    """Enrollment de 6-12 s a 16 kHz que passa em `check_enrollment_audio`, ou `None` se as `max_tries` tentativas
    forem recusadas. `denoise`: áudio 48 kHz -> 48 kHz alinhado (None desliga). `denoise_flag`: força denoisar (ou
    não); None sorteia com `p_denoise`. Cada tentativa recusada é sorteada de novo, mas o flag fica o mesmo."""
    if p_noise > 0 and not noise:
        raise EmbeddingBankError("p_noise > 0 exige um pool de ruído")
    if p_rir > 0 and not rirs:
        raise EmbeddingBankError("p_rir > 0 exige um pool de RIR")
    if not utterances:
        raise EmbeddingBankError("falante sem falas")
    if denoise_flag and denoise is None:
        raise EmbeddingBankError("denoise_flag=True exige a função de denoise")
    for _ in range(max_tries):
        x, used = concat_utterances48_ids(rng, utterances, data_root, trim=trim)
        has_rir = bool(rirs) and bool(rng.random() < p_rir)
        if has_rir:
            x = apply_rir(x, _load48(data_root, rirs[int(rng.integers(len(rirs)))]))
        snr = None
        if noise and rng.random() < p_noise:
            nz = loop_to_length(_load48(data_root, noise[int(rng.integers(len(noise)))]), x.size, rng)
            snr = float(rng.uniform(*snr_db))
            x = (x.astype(np.float64) + scale_for_ratio(x, nz, snr)).astype(np.float32)
        den = bool(denoise_flag) if denoise_flag is not None else (denoise is not None and bool(rng.random() < p_denoise))
        if den:
            x = np.asarray(denoise(x), np.float32)
        audio = to_16k(x, SR48)
        try:
            check_enrollment_audio(audio)
        except EnrollmentRejected:
            continue
        return EnrollmentVariant(audio.astype(np.float32), used, den, snr, has_rir)
    return None


def make_enrollment_audio(rng: np.random.Generator, **kw) -> np.ndarray | None:
    """Só o áudio de `make_enrollment_variant` (ou `None`)."""
    v = make_enrollment_variant(rng, **kw)
    return None if v is None else v.audio


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


def denoise_schedule(seed: int, key: str, v: int, p: float) -> bool:
    """A variante `v` do falante `key` é denoisada? Agenda de Bresenham com fase por falante: em qualquer janela de
    variantes a fração é `p` a menos de 1 variante, e só depende de `(seed, key, v)` (determinístico, sem sorteio)."""
    phase = float(_speaker_rng(seed, key, 10**6).random())
    return int(np.floor((v + 1) * p + phase)) - int(np.floor(v * p + phase)) == 1


def _init_worker(cfg: dict) -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass
    factory = cfg["denoise_factory"]
    _STATE.clear()
    _STATE.update(cfg, denoise=factory() if factory is not None else None)


def _speaker_job(key: str) -> tuple[str, list[EnrollmentVariant]]:
    s = _STATE
    out = []
    for v in range(s["variants"]):
        a = make_enrollment_variant(
            _speaker_rng(s["seed"], key, v), utterances=s["utterances"][key], data_root=s["data_root"],
            noise=s["noise"], rirs=s["rirs"], denoise=s["denoise"], p_noise=s["p_noise"], snr_db=s["snr_db"],
            p_rir=s["p_rir"], denoise_flag=denoise_schedule(s["seed"], key, v, s["p_denoise"]),
            max_tries=s["max_tries"], trim=s["trim"])
        if a is not None:
            out.append(a)
    return key, out


def _shard(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.npz"


def _write_shard(sp: Path, emb: np.ndarray, info: SpeakerVariants) -> None:
    table = sorted({u for ids in info.utterance_ids for u in ids})
    index = {u: n for n, u in enumerate(table)}
    width = max((len(ids) for ids in info.utterance_ids), default=0)
    pad = np.full((len(info), width), -1, np.int32)
    for r, ids in enumerate(info.utterance_ids):
        pad[r, : len(ids)] = [index[u] for u in ids]
    tmp = sp.with_name(sp.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, emb=emb, table=np.array(table, dtype=str) if table else np.zeros(0, dtype="<U1"), utt=pad,
                 den=info.denoised, snr=info.snr_db, rir=info.rir)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, sp)


def _read_shard(sp: Path) -> tuple[np.ndarray, SpeakerVariants]:
    with np.load(sp) as z:
        table = [str(u) for u in z["table"]]
        emb = np.asarray(z["emb"], np.float32).reshape(-1, EMB_DIM)
        info = SpeakerVariants(tuple(tuple(table[j] for j in row if j >= 0) for row in z["utt"]),
                               z["den"].astype(bool), z["snr"].astype(np.float32), z["rir"].astype(bool))
    if len(info) != len(emb):
        raise ValueError("shard inconsistente")
    return emb, info


def _signature(**kw) -> dict:
    return json.loads(json.dumps(kw, sort_keys=True))


def build_bank(encoder: Callable[[np.ndarray], np.ndarray], records: Sequence[FileRecord], *, data_root: Path,
               variants: int = 20, seed: int = 0, noise: Sequence[FileRecord] = (), rirs: Sequence[FileRecord] = (),
               denoise_factory: Callable[[], Callable[[np.ndarray], np.ndarray]] | None = None,
               p_noise: float = 0.7, snr_db: tuple[float, float] = (0.0, 30.0), p_rir: float = 0.3,
               p_denoise: float = 0.5, max_tries: int = 16, trim: bool = False, workers: int = 0,
               cache_dir: Path | None = None, encoder_tag: str = "", denoise_tag: str = "",
               sources: Mapping[str, SourceManifest] | None = None, allow_pilot: bool = False,
               log: Callable[[str], None] | None = None) -> ArrayEmbeddingBank:
    """`variants` enrollments por falante de treino -> embeddings (`encoder`: áudio 16 kHz -> 192-d) e proveniência
    por variante (falas usadas, denoisada, SNR, RIR).

    Só registros de treino entram (`training_records` recusa dev/teste). `workers=0` roda tudo no processo.
    `p_denoise` > 0 exige `denoise_factory`."""
    if p_denoise > 0 and denoise_factory is None:
        raise EmbeddingBankError("p_denoise > 0 exige denoise_factory")
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
    sig = _signature(bank_format=BANK_FORMAT, variants=variants, seed=seed, p_noise=p_noise, snr_db=list(snr_db),
                     p_rir=p_rir, p_denoise=p_denoise, max_tries=max_tries, trim=trim, pipeline=PIPELINE,
                     encoder_tag=encoder_tag, denoise_tag=denoise_tag,
                     noise_sha=hashlib.sha256("\n".join(noise_p).encode()).hexdigest(),
                     rir_sha=hashlib.sha256("\n".join(rir_p).encode()).hexdigest())
    done: dict[str, tuple[np.ndarray, SpeakerVariants]] = {}
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
                    done[k] = _read_shard(sp)
                except (OSError, ValueError, KeyError):
                    sp.unlink()  # shard truncado por uma queda: refaz
    todo = [k for k in by if k not in done]
    say = log or (lambda _m: None)
    say(f"banco: {len(by)} falantes, {len(done)} no cache, {len(todo)} a gerar, {variants} variantes cada")
    cfg = {"utterances": by, "data_root": Path(data_root), "noise": noise_p, "rirs": rir_p,
           "denoise_factory": denoise_factory, "variants": variants, "seed": seed, "p_noise": p_noise,
           "snr_db": tuple(snr_db), "p_rir": p_rir, "p_denoise": p_denoise, "max_tries": max_tries, "trim": trim}
    t0 = time.time()

    def take(res: tuple[str, list[EnrollmentVariant]], n: int) -> None:
        key, vs = res
        emb = np.stack([np.asarray(encoder(v.audio), np.float32).reshape(EMB_DIM) for v in vs]) if vs else \
            np.zeros((0, EMB_DIM), np.float32)
        info = SpeakerVariants(tuple(v.utterance_ids for v in vs), np.array([v.denoised for v in vs], bool),
                               np.array([np.nan if v.snr_db is None else v.snr_db for v in vs], np.float32),
                               np.array([v.rir for v in vs], bool))
        done[key] = (emb, info)
        if cache_dir is not None:
            _write_shard(_shard(cache_dir, key), emb, info)
        say(f"[{n}/{len(todo)}] {len(emb)} variantes ({time.time() - t0:.0f} s)")

    if workers > 0 and todo:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for n, res in enumerate(pool.imap_unordered(_speaker_job, todo, chunksize=1), 1):
                take(res, n)
    else:
        _init_worker(cfg)
        for n, k in enumerate(todo, 1):
            take(_speaker_job(k), n)
    skipped = sorted(k for k in by if len(done[k][0]) == 0)
    short = {k: len(done[k][0]) for k in by if 0 < len(done[k][0]) < variants}
    meta = {"variants_per_speaker": variants, "seed": seed, "p_noise": p_noise, "snr_db": list(snr_db),
            "p_rir": p_rir, "p_denoise": p_denoise, "max_tries": max_tries, "trim_silence": trim,
            "n_noise": len(noise_p), "n_rir": len(rir_p), "encoder_tag": encoder_tag, "denoise_tag": denoise_tag,
            "pipeline": PIPELINE, "skipped_speakers": skipped, "short_speakers": short}
    kept = {k: v for k, v in done.items() if len(v[0])}
    bank = ArrayEmbeddingBank({k: v[0] for k, v in kept.items()}, meta, {k: v[1] for k, v in kept.items()})
    bank.meta["stats"] = bank.stats()
    return bank
