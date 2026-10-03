"""Artefatos congelados da Tarefa 9: trials-v1 por SHA-256, nenhum `client_id` no git, filtro do CV (se houver dados)."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from cctrain.data.manifest import read_file_manifest
from cctrain.data.splits import apply_splits, assign_splits, load_split_config
from cctrain.eval.trials import read_trials, verify_trials_hashes
from cctrain.paths import repo_root

ROOT = repo_root()
sys.path.insert(0, str(ROOT / "scripts"))
LOCK = ROOT / "configs" / "eval" / "trials-v1.sha256"
GIT_LISTS = {"librispeech_test_clean.jsonl": ROOT / "configs" / "eval" / "trials-v1" / "librispeech_test_clean.jsonl",
             "vctk_heldout.jsonl": ROOT / "configs" / "eval" / "trials-v1" / "vctk_heldout.jsonl"}


def test_versioned_trial_lists_match_the_frozen_hashes():
    verify_trials_hashes(LOCK, GIT_LISTS)


def test_cv_trial_list_matches_hash_when_present(paths):
    p = paths.data / "trials-v1" / "cv_pt_heldout.jsonl"
    if not p.is_file():
        pytest.skip("lista de trials do CV ausente (fica em data/, fora do git)")
    verify_trials_hashes(LOCK, {"cv_pt_heldout.jsonl": p})


@pytest.mark.parametrize("name", sorted(GIT_LISTS))
def test_trials_only_use_held_out_speakers_and_are_balanced(name):
    ts = read_trials(GIT_LISTS[name])
    assert sum(t.target for t in ts) == 5000 and sum(not t.target for t in ts) == 5000
    spk = {t.spk_a for t in ts} | {t.spk_b for t in ts}
    cfg = load_split_config()
    from build_splits import non_cv_speech_records  # noqa: PLC0415

    recs = non_cv_speech_records()
    assign = assign_splits(recs, cfg)
    assert {assign[s] for s in spk} == {"test"}  # só falantes de teste
    train_spk = {k for k, v in assign.items() if v == "train"}
    assert not spk & train_spk
    assert all(t.a != t.b for t in ts)
    by_path = {r.path: r.sha256 for r in recs}
    assert all(by_path[t.a] != by_path[t.b] for t in ts)  # nenhum par com o mesmo conteúdo
    assert len({r.sha256 for r in recs}) == len(recs)  # um registro por conteúdo
    paths_in_manifest = {r.path for r in apply_splits(recs, assign) if r.split == "test"}
    assert {t.a for t in ts} | {t.b for t in ts} <= paths_in_manifest


def test_cv_trials_use_only_retained_test_clients(paths):
    kept = paths.data / "manifests" / "cv27-pt-filtered.jsonl"
    p = paths.data / "trials-v1" / "cv_pt_heldout.jsonl"
    if not (kept.is_file() and p.is_file()):
        pytest.skip("dados do CV ausentes")
    test = {r.path: r for r in read_file_manifest(kept) if r.split == "test"}
    ts = read_trials(p)
    assert {t.a for t in ts} | {t.b for t in ts} <= set(test)
    assert len({r.speaker_key for r in test.values()}) >= 40
    assert all(r.role == "test_only" for r in test.values())


def test_no_cv_client_id_in_tracked_files():
    """`client_id` do CV tem 128 hex; nada com esse formato pode estar no git (spec §4.6)."""
    files = subprocess.run(["git", "ls-files", "manifests", "configs", "docs", "src", "tests", "scripts"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    rx = re.compile(rb"\b[0-9a-f]{128}\b")
    bad = [f for f in files if (ROOT / f).is_file() and rx.search((ROOT / f).read_bytes())]
    assert not bad, bad
    assert not [f for f in files if "cv27-pt-full" in f and f.endswith(".jsonl")]
