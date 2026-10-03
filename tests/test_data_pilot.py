import pytest

from cctrain.data.manifest import ManifestError
from cctrain.data.pilot import (
    CV_VARIANT_PTBR,
    VCTK_PILOT_SPEAKERS,
    VCTK_TEST_FILES,
    check_license_text,
    cv_pilot_selection,
    librispeech_select,
    select_vctk_members,
)


def _names(n_per_speaker: int = 60) -> list[str]:
    names = ["license_text", "README.txt"]
    for spk in VCTK_PILOT_SPEAKERS + ("p240",):
        for i in range(1, n_per_speaker + 1):
            names += [f"wav48_silence_trimmed/{spk}/{spk}_{i:03d}_mic1.flac",
                      f"wav48_silence_trimmed/{spk}/{spk}_{i:03d}_mic2.flac"]
    names += [f"wav48_silence_trimmed/{n.split('_')[0]}/{n}" for n in VCTK_TEST_FILES]
    return names


def test_vctk_selection_is_mic1_first_50_and_test_files():
    sel = select_vctk_members(_names())
    pilot = [v for v in sel.values() if "/pilot/" in v]
    assert len(pilot) == 50 * len(VCTK_PILOT_SPEAKERS)
    assert all(v.endswith("_mic1.flac") for v in pilot)
    assert "raw/vctk-0.92/pilot/p227/p227_050_mic1.flac" in sel.values()
    assert "raw/vctk-0.92/pilot/p227/p227_051_mic1.flac" not in sel.values()
    assert sum("/test/" in v for v in sel.values()) == len(VCTK_TEST_FILES)
    assert sel["license_text"] == "raw/vctk-0.92/license_text"
    assert not any("p240" in v for v in sel.values())


def test_vctk_selection_fails_closed():
    with pytest.raises(ManifestError, match="licença"):
        select_vctk_members([n for n in _names() if n != "license_text"])
    with pytest.raises(ManifestError, match="p227"):
        select_vctk_members(_names(n_per_speaker=10))


def test_license_text_must_be_cc_by_4():
    check_license_text("This corpus is licensed under the Creative Commons License: Attribution 4.0 International", "x")
    with pytest.raises(ManifestError):
        check_license_text("Open Data Commons Attribution License (ODC-By) v1.0", "x")


def test_librispeech_selection():
    assert librispeech_select("LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac") == \
        "dev-clean/1272/128104/1272-128104-0000.flac"
    assert librispeech_select("LibriSpeech/dev-clean/1272/128104/1272-128104.trans.txt") is not None
    assert librispeech_select("LibriSpeech/LICENSE.TXT") == "LICENSE.TXT"
    assert librispeech_select("LibriSpeech/dev-clean/1272/") is None


def _cv_rows() -> list[dict]:
    rows = []
    for spk, variant, n in (("a", CV_VARIANT_PTBR, 8), ("b", CV_VARIANT_PTBR, 2), ("c", "Portuguese (Portugal)", 6),
                            ("d", "", 6)):
        rows += [{"client_id": spk, "path": f"{spk}{i}.mp3", "variant": variant} for i in range(n)]
    return rows


def test_cv_selection_keeps_only_ptbr_with_cap_per_speaker_and_is_deterministic():
    sel = cv_pilot_selection(_cv_rows(), per_speaker=5)
    assert {r["client_id"] for r in sel} == {"a", "b"}
    counts = {s: sum(r["client_id"] == s for r in sel) for s in ("a", "b")}
    assert counts == {"a": 5, "b": 2}
    assert [r["path"] for r in sel] == [r["path"] for r in cv_pilot_selection(list(reversed(_cv_rows())), 5)]


def test_cv_selection_fails_closed_without_ptbr_rows():
    with pytest.raises(ManifestError, match="pt-BR"):
        cv_pilot_selection([r for r in _cv_rows() if r["variant"] != CV_VARIANT_PTBR], per_speaker=5)


def test_vctk_selection_without_license_in_zip_is_explicit():
    names = [n for n in _names() if n != "license_text"]
    sel = select_vctk_members(names, license_in_zip=False)
    assert "license_text" not in sel
    assert sum("/pilot/" in v for v in sel.values()) == 50 * len(VCTK_PILOT_SPEAKERS)
