"""Dev do M2 pré-renderizado em duas metades disjuntas por fala (`dev-select`, `dev-gate`) e congelado por SHA-256.

`build_dev` só aceita registros do split `dev` (recusa `train` e `test`). As falas de cada falante são ordenadas por hash e
repartidas alternadamente entre as metades; dentro de uma metade, o enrollment é montado com falas do pool (as que
entram nele nunca entram em mistura) e as demais, com duração adequada, formam o pool de misturas. O enrollment usa falas com o silêncio das pontas aparado (VAD de energia, `trim_edge_silence`); as duas metades têm EXATAMENTE os
mesmos falantes (quem não tem enrollment válido ou fala para mistura em alguma das duas sai das duas; o relatório por corpus e
sexo vai em `speakers.json`). Condições (config `m2-devsets-v1.yaml`): C0 (alvo + ruído, sem perfil),
C0c (alvo limpo sozinho), C3 (concorrente sem ruído, mesmo sexo), C4 (concorrente + ruído, mesmo sexo), C7 (locutor trocado,
mesmo/diferente sexo). Sexo só declarado: falante sem sexo declarado fica fora de C3/C4/C7 (e de interferente/perfil).

O `dev-gate` só abre com `final_gate=True` (Tarefa 8) e cada abertura fica registrada em `dev-gate/opened.log`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from cctrain.data.audio import load_audio, resample
from cctrain.data.enroll import (
    VAD_ABSOLUTE_FLOOR_DBFS,
    VAD_FRAME_SAMPLES,
    VAD_RELATIVE_RANGE_DB,
    EnrollmentRejected,
    check_enrollment_audio,
)
from cctrain.data.manifest import FileRecord
from cctrain.data.mix import MixKind, crop_or_pad, loop_to_length, make_mixture
from cctrain.data.resample import to_16k
from cctrain.eval.gates import verify_frozen
from cctrain.hashing import sha256_file
from cctrain.paths import repo_root

DEVSETS_V1 = repo_root() / "configs" / "eval" / "m2-devsets-v1.yaml"
FREEZE_V1 = repo_root() / "configs" / "eval" / "m2-devsets-v1.freeze.json"
HALVES = ("dev-select", "dev-gate")
SR = 48_000
ENROLL_SR = 16_000
EMB_DIM = 192
CONDITIONS = ("C0", "C0c", "C3", "C4", "C7")
_INT16 = 32768.0


class DevSetsError(ValueError):
    """Dev do M2 inválido, adulterado ou usado fora da regra."""


class DevGateLocked(DevSetsError):
    """O `dev-gate` só abre com `final_gate=True` (Tarefa 8)."""


# --- config ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DevSetsConfig:
    seed: int
    source_split: str
    halves: tuple[str, ...]
    target_min_s: float
    target_max_s: float
    enroll_min_s: float
    enroll_max_s: float
    pause_s: tuple[float, float]
    conditions: Mapping[str, Mapping]
    sha256: str

    def per(self, condition: str, key: str, corpus: str) -> int:
        table = self.conditions[condition][key]
        return int(table.get(corpus, table["default"]))


def load_devsets_config(path: Path = DEVSETS_V1) -> DevSetsConfig:
    digest = verify_frozen(Path(path))
    d = yaml.safe_load(Path(path).read_text())
    if d.get("version") != 1 or tuple(d["halves"]) != HALVES:
        raise DevSetsError("m2-devsets: version 1 e halves [dev-select, dev-gate] são obrigatórios")
    if d["source_split"] != "dev":
        raise DevSetsError("m2-devsets: source_split deve ser dev")
    if not d["interferer"]["same_sex_only"]:
        raise DevSetsError("m2-devsets: C3/C4 são de mesmo sexo (decisão do dono)")
    if tuple(d["conditions"]) != CONDITIONS:
        raise DevSetsError(f"m2-devsets: condições devem ser {CONDITIONS}")
    return DevSetsConfig(
        seed=int(d["seed"]), source_split="dev", halves=HALVES, target_min_s=float(d["target"]["min_s"]),
        target_max_s=float(d["target"]["max_s"]), enroll_min_s=float(d["enrollment"]["min_s"]),
        enroll_max_s=float(d["enrollment"]["max_s"]),
        pause_s=(float(d["enrollment"]["pause_s"][0]), float(d["enrollment"]["pause_s"][1])),
        conditions=d["conditions"], sha256=digest)


# --- partição por fala ---------------------------------------------------------------------------------------------


def _h(seed: int, *parts: object) -> bytes:
    return hashlib.sha256("|".join(str(p) for p in (seed, *parts)).encode()).digest()


def _rng(seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(_h(seed, *parts)[:8], "big"))


def _dur(r: FileRecord) -> float:
    return r.num_samples / r.sr


@dataclass(frozen=True)
class HalfUtterances:
    """Falas de um falante numa metade, em ordem de hash. O enrollment é escolhido entre elas na construção
    (`choose_enrollment`); o pool de misturas é o resto, com duração em [min_s, max_s]."""

    pool: tuple[FileRecord, ...]

    def mix_pool(self, enroll: Iterable[FileRecord], cfg: DevSetsConfig) -> tuple[FileRecord, ...]:
        used = {r.path for r in enroll}
        return tuple(r for r in self.pool if r.path not in used and cfg.target_min_s <= _dur(r) <= cfg.target_max_s)


def partition_utterances(records: Iterable[FileRecord], cfg: DevSetsConfig) -> dict[str, dict[str, HalfUtterances]]:
    """{speaker_key: {half: HalfUtterances}}: falas ordenadas por hash e alternadas entre as metades (disjuntas por fala)."""
    by_spk: dict[str, list[FileRecord]] = defaultdict(list)
    for r in records:
        if r.split != cfg.source_split:
            raise DevSetsError(f"{r.path}: split {r.split!r}; o dev só aceita {cfg.source_split!r}")
        if r.speaker_key is not None:
            by_spk[r.speaker_key].append(r)
    out: dict[str, dict[str, HalfUtterances]] = {}
    for spk, recs in sorted(by_spk.items()):
        ordered = sorted(recs, key=lambda r: _h(cfg.seed, "part", r.path))
        out[spk] = {half: HalfUtterances(tuple(ordered[k::len(cfg.halves)])) for k, half in enumerate(cfg.halves)}
    return out


def split_noise_halves(noise: Iterable[FileRecord], cfg: DevSetsConfig) -> dict[str, list[FileRecord]]:
    out: dict[str, list[FileRecord]] = {h: [] for h in cfg.halves}
    for r in sorted(noise, key=lambda r: r.path):
        out[cfg.halves[_h(cfg.seed, "noise", r.sha256)[0] % len(cfg.halves)]].append(r)
    return out


# --- áudio ---------------------------------------------------------------------------------------------------------


def _load48(root: Path, r: FileRecord) -> np.ndarray:
    x, sr = load_audio(Path(root) / r.path)
    return resample(x, sr, SR)


def _to_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(x, np.float64) * _INT16), -32768, 32767).astype(np.int16)


def _from_i16(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / np.float32(_INT16)).astype(np.float32)


def _save_npy(path: Path, arr: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {"file": path.name, "sha256": sha256_file(path), "samples": int(arr.size)}


def _slug(speaker_key: str) -> str:
    return hashlib.sha256(speaker_key.encode()).hexdigest()[:16]


TRIM_MARGIN_FRAMES = 2  # 40 ms de margem de cada lado da fala ativa mantida pelo corte de pontas


def trim_edge_silence(x16k: np.ndarray) -> np.ndarray:
    """Apara o silêncio das pontas de uma fala (VAD de energia: quadros de 20 ms; ativo = nível >= max(piso -50 dBFS, pico
    do quadro - 30 dB), a mesma regra do `check_enrollment_audio`, mas só nas pontas; o meio da fala não é tocado),
    com 40 ms de margem. Fala sem nenhum quadro ativo volta inalterada (a validação do produto decide)."""
    x = np.asarray(x16k, np.float32)
    n_frames = len(x) // VAD_FRAME_SAMPLES
    if n_frames == 0:
        return x
    fr = x[: n_frames * VAD_FRAME_SAMPLES].astype(np.float64).reshape(n_frames, VAD_FRAME_SAMPLES)
    level = 10.0 * np.log10((fr * fr).mean(axis=1) + 1e-12)
    active = np.flatnonzero(level >= max(float(VAD_ABSOLUTE_FLOOR_DBFS), float(level.max()) - float(VAD_RELATIVE_RANGE_DB)))
    if active.size == 0:
        return x
    a = max(int(active[0]) - TRIM_MARGIN_FRAMES, 0) * VAD_FRAME_SAMPLES
    b = len(x) if int(active[-1]) + 1 + TRIM_MARGIN_FRAMES >= n_frames else (int(active[-1]) + 1 + TRIM_MARGIN_FRAMES) * VAD_FRAME_SAMPLES
    return x[a:b]


def choose_enrollment(root: Path, pool: Sequence[FileRecord], rng: np.random.Generator, cfg: DevSetsConfig,
                      tries: int = 12) -> tuple[np.ndarray, tuple[FileRecord, ...]] | None:
    """Enrollment de `enroll_min_s`..`enroll_max_s` (16 kHz) só de falas do `pool` da metade, com o silêncio das pontas de cada
    fala aparado (`trim_edge_silence`, antes de concatenar) e pausas curtas SÓ entre falas, válido pelo `check_enrollment_audio`
    do produto (sem afrouxá-lo; como no M1: várias tentativas com ordens diferentes de falas). Devolve o áudio e as falas que
    de fato entram nele (só elas saem do pool de misturas); None se nenhuma tentativa valida."""
    cache: dict[str, np.ndarray] = {}
    for _ in range(tries):
        want = float(rng.uniform(cfg.enroll_min_s, cfg.enroll_max_s))
        n_want = int(want * ENROLL_SR)
        parts, used, bounds, total = [], [], [], 0
        for k in rng.permutation(len(pool)):
            r = pool[int(k)]
            if r.path not in cache:
                x, sr = load_audio(Path(root) / r.path)
                cache[r.path] = trim_edge_silence(to_16k(x, sr))
            if parts:
                pause = np.zeros(int(rng.uniform(*cfg.pause_s) * ENROLL_SR), np.float32)
                parts.append(pause)
                total += len(pause)
            used.append(r)
            bounds.append((total, total + len(cache[r.path])))
            parts.append(cache[r.path])
            total += len(cache[r.path])
            if total >= n_want:
                break
        # só as falas que começam antes do corte entram; o corte cai na fala (nunca numa pausa): termina no fim da última fala
        # se ela acaba antes de `n_want`. Se o enrollment ficar menor que o mínimo, a validação do produto o recusa e há retry.
        keep = [(r, b) for r, b in zip(used, bounds, strict=True) if b[0] < n_want]
        if not keep:
            return None  # pool vazio
        used = [r for r, _ in keep]
        audio = np.concatenate(parts)[: min(n_want, keep[-1][1][1])].astype(np.float32)
        try:
            check_enrollment_audio(audio)
        except EnrollmentRejected:
            continue
        return audio, tuple(used)
    return None


# --- construção ----------------------------------------------------------------------------------------------------


def build_dev(records: Sequence[FileRecord], noise: Sequence[FileRecord], data_root: Path, out_dir: Path,
              sex: Mapping[str, str], embed: Callable[[np.ndarray], np.ndarray], cfg: DevSetsConfig | None = None, *,
              split: str = "dev", log: Callable[[str], None] = lambda s: None) -> dict:
    """Renderiza as duas metades em `out_dir/<metade>/` e devolve o resumo. Recusa qualquer split que não seja `dev`."""
    cfg = cfg or load_devsets_config()
    if split != cfg.source_split:
        raise DevSetsError(f"build_dev recusa o split {split!r}: só {cfg.source_split!r}")
    bad = sorted({r.split for r in records if r.split != cfg.source_split})
    if bad:
        raise DevSetsError(f"build_dev recusa registros dos splits {bad}: só {cfg.source_split!r}")
    bad_noise = sorted({r.path for r in noise if r.split in ("train", "test")})
    if bad_noise:
        raise DevSetsError("build_dev: ruído de train/test não entra no dev")
    data_root, out_dir = Path(data_root), Path(out_dir)
    parts = partition_utterances(records, cfg)
    noise_by_half = split_noise_halves(noise, cfg)
    # Enrollments das DUAS metades antes de qualquer item: o conjunto de falantes tem de ser o mesmo nas duas (senão o
    # dev-select escolheria checkpoint num conjunto de falantes diferente do que o dev-gate confirma: viés de seleção).
    chosen: dict[str, dict[str, tuple[np.ndarray, tuple[FileRecord, ...]]]] = {h: {} for h in cfg.halves}
    mix_pools: dict[str, dict[str, tuple[FileRecord, ...]]] = {h: {} for h in cfg.halves}
    why: dict[str, list[str]] = {}
    for spk in sorted(parts):
        for half in cfg.halves:
            hu = parts[spk][half]
            got = choose_enrollment(data_root, hu.pool, _rng(cfg.seed, half, "enroll", spk), cfg)
            if got is None:
                why.setdefault(spk, []).append(f"{half}: sem enrollment válido")
                continue
            mp = hu.mix_pool(got[1], cfg)
            if not mp:
                why.setdefault(spk, []).append(f"{half}: sem fala de 3-10 s para misturas depois do enrollment")
                continue
            chosen[half][spk], mix_pools[half][spk] = got, mp
    keep = {s for s in parts if all(s in chosen[h] for h in cfg.halves)}
    speakers_report = _speakers_report(parts, keep, why, sex)
    for spk in sorted(why):
        log(f"[dev] {spk.split(':', 1)[0]}:{_slug(spk)[:6]}: fora das duas metades ({'; '.join(why[spk])})")
    log(f"[dev] falantes: {len(keep)}/{len(parts)} nas duas metades; fora por corpus {speakers_report['excluded_by_corpus']}, "
        f"por sexo {speakers_report['excluded_by_sex']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "speakers.json").write_text(json.dumps(speakers_report, indent=2, sort_keys=True) + "\n")
    summary: dict = {"speakers": speakers_report}
    for half in cfg.halves:
        summary[half] = _build_half(half, {s: chosen[half][s] for s in sorted(keep)},
                                    {s: mix_pools[half][s] for s in keep}, noise_by_half[half], data_root, out_dir / half,
                                    sex, embed, cfg, log)
    return summary


def _speakers_report(parts: Mapping, keep: set[str], why: Mapping[str, list[str]], sex: Mapping[str, str]) -> dict:
    """Quem ficou e quem saiu (das duas metades), por corpus e sexo declarado, com o motivo de cada exclusão."""
    corp = lambda k: k.split(":", 1)[0]  # noqa: E731
    sx = lambda k: sex.get(k) or "undeclared"  # noqa: E731
    out_: dict = {"total": len(parts), "kept": len(keep), "excluded": len(parts) - len(keep)}
    for name, groups in (("kept", keep), ("excluded", set(parts) - keep)):
        for dim, fn in (("corpus", corp), ("sex", sx), ("corpus_sex", lambda k: f"{corp(k)}/{sx(k)}")):
            cnt: dict[str, int] = defaultdict(int)
            for k in groups:
                cnt[fn(k)] += 1
            out_[f"{name}_by_{dim}"] = dict(sorted(cnt.items()))
    out_["total_by_corpus"] = dict(sorted({c: sum(1 for k in parts if corp(k) == c) for c in {corp(k) for k in parts}}.items()))
    out_["excluded_speakers"] = {k: why[k] for k in sorted(set(parts) - keep)}
    return out_


def _build_half(half: str, enrolls: Mapping[str, tuple[np.ndarray, tuple[FileRecord, ...]]],
                mix_pool: Mapping[str, tuple[FileRecord, ...]], noise: Sequence[FileRecord],
                data_root: Path, out: Path, sex: Mapping[str, str], embed, cfg: DevSetsConfig, log) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.seed
    corpus = lambda k: k.split(":", 1)[0]  # noqa: E731
    # enrollments já escolhidos (no pool da metade; as falas usadas saem do pool de misturas)
    enroll_rows, embs = [], []
    enrolled: dict[str, int] = {}
    for spk in sorted(enrolls):
        audio, used_utts = enrolls[spk]
        meta = _save_npy(out / "enroll" / f"{_slug(spk)}.npy", _to_i16(audio))
        emb = np.asarray(embed(_from_i16(_to_i16(audio))), np.float32).reshape(EMB_DIM)
        enrolled[spk] = len(embs)
        embs.append(emb)
        enroll_rows.append({"speaker": spk, "corpus": corpus(spk), "sex": sex.get(spk), "audio": meta,
                            "seconds": len(audio) / ENROLL_SR, "utterances": [r.path for r in used_utts],
                            "embedding_row": enrolled[spk]})
    np.save(out / "enroll_emb.npy", np.stack(embs).astype(np.float32), allow_pickle=False)
    _write_jsonl(out / "enroll.jsonl", enroll_rows)

    speakers = sorted(enrolled)
    cursor: dict[str, int] = defaultdict(int)
    perms = {s: _rng(seed, half, "perm", s).permutation(len(mix_pool[s])) for s in speakers}

    def next_target(spk: str) -> FileRecord:
        pool = mix_pool[spk]
        r = pool[int(perms[spk][cursor[spk] % len(pool)])]
        cursor[spk] += 1
        return r

    def pick_interferer(spk: str, peers: Sequence[str], n: int, rng) -> tuple[str, FileRecord]:
        ispk = peers[int(rng.integers(len(peers)))]
        pool = mix_pool[ispk]
        long_enough = [r for r in pool if r.num_samples / r.sr * SR >= n] or list(pool)
        return ispk, long_enough[int(rng.integers(len(long_enough)))]

    def pick_noise(rng) -> FileRecord:
        return noise[int(rng.integers(len(noise)))]

    if not noise:
        raise DevSetsError(f"[{half}] sem ruído de dev")
    items: list[dict] = []
    declared = [s for s in speakers if s in sex]

    def emit(cond: str, spk: str, k: int, *, target: FileRecord, noisy, clean, sir=None, snr=None, interferer=None,
             ispk=None, noise_rec=None, profile=None, pair=None) -> None:
        iid = f"{half}-{cond}-{len(items):05d}"
        n = noisy.size
        entry = {"id": iid, "half": half, "condition": cond, "corpus": corpus(spk), "target_speaker": spk,
                 "target_sex": sex.get(spk), "target_utt": target.path, "interferer_speaker": ispk,
                 "interferer_sex": sex.get(ispk) if ispk else None,
                 "interferer_utt": interferer.path if interferer else None, "noise": noise_rec.path if noise_rec else None,
                 "noise_sha256": noise_rec.sha256 if noise_rec else None, "profile_speaker": profile,
                 "sex_pair": pair, "sir_db": sir, "snr_db": snr, "samples": int(n), "k": k}
        entry["noisy"] = _save_npy(out / "audio" / f"{iid}.noisy.npy", _to_i16(noisy))
        entry["clean"] = (entry["noisy"] if clean is noisy else
                          None if clean is None else _save_npy(out / "audio" / f"{iid}.clean.npy", _to_i16(clean)))
        items.append(entry)

    n_of = lambda r: int(round(_dur(r) * SR))  # noqa: E731
    for ci, spk in enumerate(speakers):  # C0: alvo + ruído, sem perfil
        cr = corpus(spk)
        grid = cfg.conditions["C0"]["snr_db"]
        for k in range(cfg.per("C0", "items_per_speaker", cr)):
            t = next_target(spk)
            rng = _rng(seed, half, "C0", spk, k)
            tgt = crop_or_pad(_load48(data_root, t), n_of(t), rng)
            nz_rec = pick_noise(rng)
            nz = loop_to_length(_load48(data_root, nz_rec), tgt.size, rng)
            snr = float(grid[(ci + k) % len(grid)])
            m = make_mixture(tgt, [], nz, kind=MixKind.TARGET_NOISE, snr_db=snr, sir_db=float("nan"))
            emit("C0", spk, k, target=t, noisy=m.noisy, clean=m.clean, snr=snr, noise_rec=nz_rec)
    for spk in speakers:  # C0c: alvo limpo sozinho, perfil do próprio alvo
        for k in range(cfg.per("C0c", "items_per_speaker", corpus(spk))):
            t = next_target(spk)
            rng = _rng(seed, half, "C0c", spk, k)
            tgt = crop_or_pad(_load48(data_root, t), n_of(t), rng)
            m = make_mixture(tgt, [], None, kind=MixKind.TARGET_NOISE, snr_db=float("nan"), sir_db=float("nan"))
            emit("C0c", spk, k, target=t, noisy=m.clean, clean=m.clean, profile=spk)
    for cond in ("C3", "C4"):  # concorrente (sem / com ruído), mesmo sexo
        sirs = cfg.conditions[cond]["sir_db"]
        for spk in declared:
            peers = [s for s in declared if s != spk and sex[s] == sex[spk]]
            if not peers:
                continue
            for sir in sirs:
                for k in range(cfg.per(cond, "items_per_speaker_per_sir", corpus(spk))):
                    t = next_target(spk)
                    rng = _rng(seed, half, cond, spk, sir, k)
                    tgt = crop_or_pad(_load48(data_root, t), n_of(t), rng)
                    ispk, irec = pick_interferer(spk, peers, tgt.size, rng)
                    interf = crop_or_pad(_load48(data_root, irec), tgt.size, rng)
                    if cond == "C4":
                        nz_rec = pick_noise(rng)
                        nz = loop_to_length(_load48(data_root, nz_rec), tgt.size, rng)
                        snr = float(cfg.conditions["C4"]["snr_db"])
                        m = make_mixture(tgt, [interf], nz, kind=MixKind.TARGET_INTERFERER_NOISE, snr_db=snr,
                                         sir_db=float(sir))
                    else:
                        nz_rec, snr = None, None
                        m = make_mixture(tgt, [interf], None, kind=MixKind.TARGET_INTERFERER, snr_db=float("nan"),
                                         sir_db=float(sir))
                    emit(cond, spk, k, target=t, noisy=m.noisy, clean=m.clean, sir=float(sir), snr=snr, interferer=irec,
                         ispk=ispk, noise_rec=nz_rec, profile=spk, pair="same")
    for spk in declared:  # C7: só o falante B fala; perfil de A
        for pair in cfg.conditions["C7"]["sex_pair"]:
            peers = [s for s in declared if s != spk and ((sex[s] == sex[spk]) == (pair == "same"))]
            if not peers:
                continue
            for k in range(cfg.per("C7", "items_per_speaker_per_pair", corpus(spk))):
                t = next_target(spk)
                rng = _rng(seed, half, "C7", spk, pair, k)
                tgt = crop_or_pad(_load48(data_root, t), n_of(t), rng)
                m = make_mixture(tgt, [], None, kind=MixKind.TARGET_NOISE, snr_db=float("nan"), sir_db=float("nan"))
                prof = peers[int(rng.integers(len(peers)))]
                emit("C7", spk, k, target=t, noisy=m.noisy, clean=None, profile=prof, pair=pair)
    _write_jsonl(out / "items.jsonl", items)
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[it["condition"]] += 1
    log(f"[{half}] {len(items)} itens {dict(counts)}; {len(enrolled)} enrollments")
    return {"items": len(items), "by_condition": dict(counts), "enrollments": len(enrolled),
            "speakers": len(speakers)}


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --- leitura -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DevItem:
    meta: Mapping
    root: Path | None  # None: metadados apenas (verificação de integridade do dev-gate); o áudio fica trancado

    def __getitem__(self, k: str):
        return self.meta[k]

    @property
    def id(self) -> str:
        return self.meta["id"]

    def noisy(self) -> np.ndarray:
        if self.root is None:
            raise DevGateLocked("metade aberta só para metadados (integridade); o áudio exige final_gate=True")
        return _from_i16(np.load(self.root / "audio" / self.meta["noisy"]["file"]))

    def clean(self) -> np.ndarray | None:
        if self.root is None:
            raise DevGateLocked("metade aberta só para metadados (integridade); o áudio exige final_gate=True")
        c = self.meta["clean"]
        return None if c is None else _from_i16(np.load(self.root / "audio" / c["file"]))


@dataclass(frozen=True)
class DevEnrollment:
    speaker: str
    corpus: str
    sex: str | None
    seconds: float
    utterances: tuple[str, ...]
    audio_file: Path | None
    embedding: np.ndarray

    def audio16k(self) -> np.ndarray:
        if self.audio_file is None:
            raise DevGateLocked("metade aberta só para metadados (integridade); o áudio exige final_gate=True")
        return _from_i16(np.load(self.audio_file))


class DevHalf:
    def __init__(self, name: str, root: Path, *, meta_only: bool = False):
        self.name, self.root = name, Path(root)
        audio_root = None if meta_only else self.root
        self.items = [DevItem(json.loads(line), audio_root) for line in (self.root / "items.jsonl").read_text().splitlines()]
        emb = np.load(self.root / "enroll_emb.npy")
        self.enrollments: dict[str, DevEnrollment] = {}
        for line in (self.root / "enroll.jsonl").read_text().splitlines():
            d = json.loads(line)
            self.enrollments[d["speaker"]] = DevEnrollment(
                d["speaker"], d["corpus"], d["sex"], d["seconds"], tuple(d["utterances"]),
                None if meta_only else self.root / "enroll" / d["audio"]["file"], emb[d["embedding_row"]])

    def select(self, condition: str | None = None, **where) -> list[DevItem]:
        out = [i for i in self.items if condition is None or i["condition"] == condition]
        return [i for i in out if all(i[k] == v for k, v in where.items())]

    def embedding(self, item: DevItem) -> np.ndarray | None:
        p = item["profile_speaker"]
        return None if p is None else self.enrollments[p].embedding


def open_half(data_dir: Path, half: str, *, final_gate: bool = False, who: str = "",
              meta_only: bool = False) -> DevHalf:
    """Abre uma metade. O `dev-gate` exige `final_gate=True` e registra a abertura (uso único, Tarefa 8).
    `meta_only=True` lê só listas e metadados (disjunção, congelamento): sem áudio, sem trava e sem registro."""
    if half not in HALVES:
        raise DevSetsError(f"metade desconhecida: {half!r}")
    root = Path(data_dir) / half
    if meta_only:
        return DevHalf(half, root, meta_only=True)
    if half == "dev-gate":
        if not final_gate:
            raise DevGateLocked("dev-gate só abre com final_gate=True (Tarefa 8); nenhuma escolha o usa")
        with open(root / "opened.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {who or 'unknown'}\n")
    return DevHalf(half, root)


# --- congelamento e verificação ------------------------------------------------------------------------------------


def utterances_of(half: DevHalf) -> set[str]:
    """Toda fala que aparece na metade (enrollments, alvos e interferentes)."""
    used: set[str] = set()
    for e in half.enrollments.values():
        used.update(e.utterances)
    for i in half.items:
        used.add(i["target_utt"])
        if i["interferer_utt"]:
            used.add(i["interferer_utt"])
    return used


def check_disjoint_and_enrollment_rule(select: DevHalf, gate: DevHalf) -> dict:
    """Levanta `DevSetsError` se as metades não têm exatamente os mesmos falantes, se uma fala (ou ruído) aparece nas duas metades, ou se alguma mistura contém uma fala do
    enrollment do seu perfil. Devolve contagens."""
    if set(select.enrollments) != set(gate.enrollments):
        diff = sorted(set(select.enrollments) ^ set(gate.enrollments))
        raise DevSetsError(f"as metades têm falantes diferentes ({len(diff)} só numa delas, p.ex. {diff[:3]})")
    if {i["target_speaker"] for i in select.items} != {i["target_speaker"] for i in gate.items}:
        raise DevSetsError("os falantes-alvo dos itens diferem entre as metades")
    both = utterances_of(select) & utterances_of(gate)
    if both:
        raise DevSetsError(f"{len(both)} falas nas duas metades, p.ex. {sorted(both)[:3]}")
    noise_both = ({i["noise"] for i in select.items if i["noise"]} & {i["noise"] for i in gate.items if i["noise"]})
    if noise_both:
        raise DevSetsError(f"{len(noise_both)} ruídos nas duas metades")
    for half in (select, gate):
        for i in half.items:
            prof = i["profile_speaker"]
            if prof is None:
                continue
            clash = set(half.enrollments[prof].utterances) & {i["target_utt"], i["interferer_utt"]}
            if clash:
                raise DevSetsError(f"{i.id}: enrollment contém a fala da mistura {sorted(clash)}")
    return {"utterances_select": len(utterances_of(select)), "utterances_gate": len(utterances_of(gate)),
            "speakers": len(select.enrollments)}


def half_digest(root: Path) -> dict:
    root = Path(root)
    return {"items_sha256": sha256_file(root / "items.jsonl"), "enroll_sha256": sha256_file(root / "enroll.jsonl"),
            "enroll_emb_sha256": sha256_file(root / "enroll_emb.npy"),
            "n_items": sum(1 for _ in open(root / "items.jsonl")),
            "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and p.name != "opened.log")}


def make_freeze(data_dir: Path, cfg: DevSetsConfig | None = None) -> dict:
    cfg = cfg or load_devsets_config()
    halves = {h: half_digest(Path(data_dir) / h) for h in HALVES}
    return {"version": 1, "config": "configs/eval/m2-devsets-v1.yaml", "config_sha256": cfg.sha256, "halves": halves}


def verify_files(root: Path) -> int:
    """SHA-256 de cada .npy contra `items.jsonl`/`enroll.jsonl`; devolve quantos arquivos conferiu."""
    root, n = Path(root), 0
    for line in (root / "items.jsonl").read_text().splitlines():
        d = json.loads(line)
        for key in ("noisy", "clean"):
            m = d[key]
            if m is None or (key == "clean" and m == d["noisy"]):
                continue
            if sha256_file(root / "audio" / m["file"]) != m["sha256"]:
                raise DevSetsError(f"{root / 'audio' / m['file']}: SHA-256 não confere")
            n += 1
    for line in (root / "enroll.jsonl").read_text().splitlines():
        m = json.loads(line)["audio"]
        if sha256_file(root / "enroll" / m["file"]) != m["sha256"]:
            raise DevSetsError(f"{root / 'enroll' / m['file']}: SHA-256 não confere")
        n += 1
    return n


def verify_freeze(data_dir: Path, freeze_path: Path = FREEZE_V1, *, deep: bool = False) -> dict:
    """Confere os resumos (e, com `deep`, o SHA-256 de cada arquivo) contra o congelamento versionado no git."""
    freeze = json.loads(Path(freeze_path).read_text())
    cfg = load_devsets_config()
    if freeze["config_sha256"] != cfg.sha256:
        raise DevSetsError("o congelamento aponta para outra versão do m2-devsets-v1.yaml")
    for h in HALVES:
        now = half_digest(Path(data_dir) / h)
        for key in ("items_sha256", "enroll_sha256", "enroll_emb_sha256", "n_items", "bytes"):
            if now[key] != freeze["halves"][h][key]:
                raise DevSetsError(f"{h}: {key} difere do congelamento ({now[key]} != {freeze['halves'][h][key]})")
        if deep:
            verify_files(Path(data_dir) / h)
    return freeze


def load_select_items(data_dir: Path | None = None) -> list:
    """Carrega os itens do dev-select necessários para a avaliação/seleção por época (C0 e C3 com SIR 0 dB)."""
    from cctrain.m2.select import SelectItem
    from cctrain.paths import Paths

    root = Path(data_dir) if data_dir is not None else Paths.from_env().data / "dev-m2"
    half = open_half(root, "dev-select")
    out = []
    for it in half.select("C0"):
        out.append(SelectItem("C0", it.noisy(), it.clean(), it["target_speaker"], None))
    for it in half.select("C3", sir_db=0.0):
        emb = half.embedding(it)
        if emb is None:
            raise DevSetsError(f"item {it.id} em C3 sem embedding de perfil")
        out.append(SelectItem("C3", it.noisy(), it.clean(), it["target_speaker"], emb))
    return out

