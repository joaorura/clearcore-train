"""Sexo por falante, só de metadado **declarado** (M2 Tarefa 1, decisão do dono de 2026-10-03).

Nada é inferido por modelo. Fontes: `SPEAKERS.TXT` do LibriSpeech, `speaker-info.txt` do VCTK, coluna `gender` do
`validated.tsv` do Common Voice e a documentação do TTS-Portuguese (1 falante masculino). Falante sem sexo declarado
(ou com valor fora de F/M, ou com declarações conflitantes) fica fora do mapa: não entra nas condições de mesmo/diferente
sexo, mas continua no treino (spec §4.4 não filtra sexo no treino).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from cctrain.hashing import sha256_file

Sex = Literal["F", "M"]
_VALID = frozenset({"F", "M"})
CV_SOURCE_ID = "cv27-pt"
_CV_GENDER = {"male_masculine": "M", "female_feminine": "F", "male": "M", "female": "F"}
# Documentação do TTS-Portuguese Corpus (README: um falante masculino pt-BR); id do falante no manifest.
TTS_PT_DECLARED: dict[str, Sex] = {"tts-portuguese-corpus:ttsptbr-male1": "M"}


class SexError(ValueError):
    """Mapa de sexo ausente, inválido ou conflitante."""


def librispeech_declared_sex(speakers_txt: str, source_id: str) -> dict[str, Sex]:
    out: dict[str, Sex] = {}
    for line in speakers_txt.splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if parts[0].isdigit() and parts[1] in _VALID:
            out[f"{source_id}:{parts[0]}"] = parts[1]  # type: ignore[assignment]
    return out


def vctk_declared_sex(speaker_info: str, source_id: str = "vctk-0.92") -> dict[str, Sex]:
    out: dict[str, Sex] = {}
    for line in speaker_info.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[2] in _VALID:
            out[f"{source_id}:{parts[0]}"] = parts[2]  # type: ignore[assignment]
    return out


def cv_declared_sex(rows: Iterable[Mapping[str, str]]) -> tuple[dict[str, Sex], dict[str, int]]:
    """Por `client_id`: F/M quando algum clipe declara e as declarações não se contradizem."""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        cid = r.get("client_id")
        if not cid:
            continue
        seen[cid]  # noqa: B018  (registra o client mesmo sem declaração)
        g = (r.get("gender") or "").strip()
        if g:
            seen[cid].add(g)
    out: dict[str, Sex] = {}
    stats = Counter(clients=len(seen), declared=0, undeclared=0, non_binary_or_other=0, conflicting=0)
    for cid, gs in seen.items():
        mapped = {_CV_GENDER.get(g) for g in gs}
        if not gs:
            stats["undeclared"] += 1
        elif None in mapped and len(gs) == 1:
            stats["non_binary_or_other"] += 1
        elif len(mapped - {None}) != 1 or None in mapped:
            stats["conflicting"] += 1
        else:
            out[f"{CV_SOURCE_ID}:{cid}"] = mapped.pop()  # type: ignore[assignment]
            stats["declared"] += 1
    return out, dict(stats)


def write_sex_rows(sex: Mapping[str, str], path: Path, *, source: str) -> str:
    """JSONL ordenado e determinístico (`speaker_key`, `sex`, `source`); devolve o SHA-256 do arquivo."""
    bad = {k: v for k, v in sex.items() if v not in _VALID}
    if bad:
        raise SexError(f"sexo fora de F/M: {sorted(bad.items())[:3]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps({"speaker_key": k, "sex": sex[k], "source": source}, sort_keys=True) + "\n"
                   for k in sorted(sex))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body)
    tmp.replace(path)
    return sha256_file(path)


def load_speaker_sex(path: Path, *more: Path) -> dict[str, Sex]:
    """Lê um ou mais JSONL (p.ex. o público no git e o do Common Voice fora dele). Conflito entre arquivos recusa."""
    out: dict[str, Sex] = {}
    for p in (path, *more):
        p = Path(p)
        if not p.is_file():
            raise SexError(f"{p} ausente")
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("sex") not in _VALID:
                raise SexError(f"{p}:{n}: sexo {d.get('sex')!r} fora de F/M")
            k = d["speaker_key"]
            if k in out and out[k] != d["sex"]:
                raise SexError(f"{p}:{n}: conflito de sexo para {k}")
            out[k] = d["sex"]
    return out


def coverage_by_corpus(speakers_by_split: Mapping[str, Sequence[str]], sex: Mapping[str, str]) -> dict:
    """{corpus: {split: {speakers, F, M, undeclared}}}; o corpus é o prefixo da `speaker_key` (`corpus:id`)."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for split, keys in speakers_by_split.items():
        for k in set(keys):
            corpus = k.split(":", 1)[0]
            row = out.setdefault(corpus, {}).setdefault(split, {"speakers": 0, "F": 0, "M": 0, "undeclared": 0})
            row["speakers"] += 1
            row[sex.get(k, "undeclared")] += 1
    return out
