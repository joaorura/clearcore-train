"""Sexo por falante só de metadado declarado (M2 Tarefa 1): nada inferido por modelo."""

from __future__ import annotations

import json

import pytest

from cctrain.m2.sex import (
    SexError,
    coverage_by_corpus,
    cv_declared_sex,
    librispeech_declared_sex,
    load_speaker_sex,
    vctk_declared_sex,
    write_sex_rows,
)

LS = """; comentario
;ID  |SEX| SUBSET           |MINUTES| NAME
  14 | F | train-clean-100  | 25.03 | Alguem
  16 | M | train-clean-100  | 25.11 | Outro
  17 | X | train-clean-100  | 25.11 | Invalido
"""
VCTK = """ID  AGE  GENDER  ACCENTS  REGION
p225  23  F    English    Southern England
p226  22  M    English    Surrey
p280  22  F    Unknown    Unknown
p999  22  -    English    Surrey
"""


def test_librispeech_only_declared_f_or_m():
    assert librispeech_declared_sex(LS, "librispeech-train-clean-100") == {
        "librispeech-train-clean-100:14": "F", "librispeech-train-clean-100:16": "M"}


def test_vctk_only_declared_f_or_m():
    assert vctk_declared_sex(VCTK) == {"vctk-0.92:p225": "F", "vctk-0.92:p226": "M", "vctk-0.92:p280": "F"}


def test_cv_maps_declared_gender_and_drops_blank_nonbinary_and_conflicts():
    rows = [
        {"client_id": "a", "gender": "female_feminine"}, {"client_id": "a", "gender": "female_feminine"},
        {"client_id": "b", "gender": "male_masculine"},
        {"client_id": "c", "gender": ""},                       # não declarado
        {"client_id": "d", "gender": "non-binary"},             # fora de F/M: não vira rótulo
        {"client_id": "e", "gender": "male_masculine"}, {"client_id": "e", "gender": "female_feminine"},  # ambíguo
        {"client_id": "f", "gender": ""}, {"client_id": "f", "gender": "female_feminine"},  # declarado em algum clipe
    ]
    out, stats = cv_declared_sex(rows)
    assert out == {"cv27-pt:a": "F", "cv27-pt:b": "M", "cv27-pt:f": "F"}
    assert stats == {"clients": 6, "declared": 3, "undeclared": 1, "non_binary_or_other": 1, "conflicting": 1}


def test_roundtrip_and_hash(tmp_path):
    p = tmp_path / "s.jsonl"
    digest = write_sex_rows({"a:1": "F", "a:2": "M"}, p, source="unit")
    assert len(digest) == 64
    assert load_speaker_sex(p) == {"a:1": "F", "a:2": "M"}
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    assert lines[0] == {"sex": "F", "source": "unit", "speaker_key": "a:1"}  # ordenado, determinístico
    assert write_sex_rows({"a:2": "M", "a:1": "F"}, tmp_path / "t.jsonl", source="unit") == digest


def test_load_merges_files_and_refuses_conflict_and_bad_values(tmp_path):
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    write_sex_rows({"x:1": "F"}, a, source="s")
    write_sex_rows({"y:1": "M"}, b, source="s")
    write_sex_rows({"x:1": "M"}, c, source="s")
    assert load_speaker_sex(a, b) == {"x:1": "F", "y:1": "M"}
    with pytest.raises(SexError, match="conflito"):
        load_speaker_sex(a, c)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"speaker_key": "z:1", "sex": "U", "source": "s"}) + "\n")
    with pytest.raises(SexError, match="sexo"):
        load_speaker_sex(bad)
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(SexError, match="ausente"):
        load_speaker_sex(missing)


def test_write_refuses_values_other_than_f_m(tmp_path):
    with pytest.raises(SexError):
        write_sex_rows({"a:1": "U"}, tmp_path / "x.jsonl", source="s")


def test_coverage_counts_by_corpus_and_split_without_inference():
    speakers = {"dev": ["libri:1", "libri:2", "cml:9"], "train": ["libri:3", "cml:7", "cml:8"]}
    sex = {"libri:1": "F", "libri:3": "M", "libri:2": "M"}
    cov = coverage_by_corpus(speakers, sex)
    assert cov["libri"]["dev"] == {"speakers": 2, "F": 1, "M": 1, "undeclared": 0}
    assert cov["libri"]["train"] == {"speakers": 1, "F": 0, "M": 1, "undeclared": 0}
    assert cov["cml"]["dev"] == {"speakers": 1, "F": 0, "M": 0, "undeclared": 1}
    assert cov["cml"]["train"]["undeclared"] == 2


def test_committed_public_sex_manifest_is_locked_and_has_no_common_voice_ids():
    from cctrain.hashing import sha256_file
    from cctrain.paths import repo_root

    p = repo_root() / "manifests" / "speakers" / "sex-v1.jsonl"
    assert p.with_suffix(".sha256").read_text().split()[0] == sha256_file(p)
    sex = load_speaker_sex(p)
    assert sex and not any(k.startswith("cv27-pt:") for k in sex)  # client_id do CV nunca vai para o git
    assert {k.split(":", 1)[0] for k in sex} == {"librispeech-dev-clean", "librispeech-test-clean",
                                                 "librispeech-train-clean-100", "vctk-0.92", "tts-portuguese-corpus"}
    assert sex["tts-portuguese-corpus:ttsptbr-male1"] == "M"
    assert not any(json.loads(line)["source"].startswith(("model", "infer")) for line in p.read_text().splitlines())
