import json
import math
from pathlib import Path

import pytest
import yaml

from cctrain.disk import GB
from cctrain.m1 import disk_plan
from cctrain.m1.disk_plan import Phase, PlanError, load_phases_config, plan_phases, write_disk_plan

FREE_62 = 62 * GB

# fatores medidos no M0 (docs/m0-report.md, seção Disco)
F_LIBRI, F_VCTK, F_NOISE48 = 0.06663824839304526, 0.1632062417379781, 0.16719104090598327
F_CML24, F_FSD441 = 2.723407401878354 / 30.0, 2.184950010122239 / 15.0


def test_phase_order_and_contents_are_fixed():
    cfg = load_phases_config()
    assert [p["id"] for p in cfg["phases"]] == [
        "1-noise-rir-encoder", "2-vctk", "3-librispeech-100-test-clean", "4-cml-tts-pt", "5-cv27-pt"]
    by_phase = {p["id"]: [s["id"] for s in p["sources"]] for p in cfg["phases"]}
    assert by_phase["1-noise-rir-encoder"] == [
        "dns-noise-fullband-cc0", "fsd50k-eval-cc0-ccby", "rir-synthetic-v2", "speechbrain-ecapa-voxceleb",
        "wespeaker-resnet34"]
    assert by_phase["2-vctk"] == ["vctk-0.92-mic1"]
    assert by_phase["3-librispeech-100-test-clean"] == ["librispeech-train-clean-100", "librispeech-test-clean"]
    assert by_phase["4-cml-tts-pt"] == ["cml-tts-pt-cap30h"]
    assert by_phase["5-cv27-pt"] == ["cv27-pt-complete"]


def test_owner_exclusions_are_not_scheduled_but_are_recorded():
    cfg = load_phases_config()
    scheduled = {s["id"] for p in cfg["phases"] for s in p["sources"]}
    excluded = {e["id"] for e in cfg["excluded_by_owner"]}
    assert excluded == {"librispeech-train-clean-360", "librispeech-train-other-500", "mls-pt"}
    assert not scheduled & excluded
    assert "dns5-dev-g8" in {e["id"] for e in cfg["out_of_scope"]}


def test_phase_sizes_follow_measured_factors():
    phases = {p.id: p for p in plan_phases(FREE_62, 10.0)}
    assert all(isinstance(p, Phase) for p in phases.values())
    p1 = 10.0 * F_NOISE48 + 15.0 * F_FSD441 + 0.664 + 0.08 + 0.1
    assert phases["1-noise-rir-encoder"].gb_final == pytest.approx(p1, rel=1e-9)
    assert phases["2-vctk"].gb_final == pytest.approx(44.0 * F_VCTK, rel=1e-9)
    assert phases["3-librispeech-100-test-clean"].gb_final == pytest.approx((100.6 + 5.4) * F_LIBRI, rel=1e-9)
    assert phases["4-cml-tts-pt"].gb_final == pytest.approx(30.0 * F_CML24, rel=1e-9)
    assert phases["5-cv27-pt"].gb_final == pytest.approx(5_209_825_417 / GB, rel=1e-9)


def test_peaks_count_the_largest_transient_in_the_phase():
    phases = {p.id: p for p in plan_phases(FREE_62, 10.0)}
    p1 = phases["1-noise-rir-encoder"]
    assert p1.gb_peak == pytest.approx(p1.gb_final + 6.2)  # zip do FSD50K eval (não dá para ler em stream)
    for pid in ("2-vctk", "3-librispeech-100-test-clean", "4-cml-tts-pt", "5-cv27-pt"):
        assert phases[pid].gb_peak == pytest.approx(phases[pid].gb_final)  # tar/zip lidos em stream: pico ~ 0


def test_cumulative_margin_with_62_gb_free():
    baseline = 3.0 + 3.0 + 3.0 + 2.0 + 2.0  # M0: avaliadores, test sets, pré-renderizado, runs; M1: ambiente export
    phases = plan_phases(FREE_62, 10.0)
    cumulative = 0.0
    for p in phases:
        cumulative += p.gb_final
        assert p.cumulative_margin_gb == pytest.approx(62.0 - 10.0 - baseline - cumulative, rel=1e-9)
        assert p.status == "ok"
    assert phases[-1].cumulative_margin_gb > 5.0


def test_phases_with_negative_cumulative_margin_are_blocked_and_stay_blocked():
    phases = plan_phases(40 * GB, 10.0)  # 40 - 10 - 13 = 17 GB para dados
    assert [p.status for p in phases] == ["ok", "ok", "blocked_disk", "blocked_disk", "blocked_disk"]
    assert phases[1].cumulative_margin_gb > 0 > phases[2].cumulative_margin_gb
    assert phases[3].cumulative_margin_gb < phases[2].cumulative_margin_gb  # acumulado inclui as fases bloqueadas


def test_peak_that_does_not_fit_blocks_even_if_final_fits(tmp_path: Path):
    cfg = yaml.safe_load(disk_plan.PHASES_V1.read_text())
    cfg["baseline"] = []
    cfg["phases"] = [{"id": "x", "sources": [
        {"id": "small-final-big-raw", "gb_final": 1.0, "transient_gb": 30.0}]}]
    path = tmp_path / "phases.yaml"
    path.write_text(yaml.safe_dump(cfg))
    (p,) = plan_phases(30 * GB, 10.0, config=path)  # final: 20 - 1 = 19 >= 0; pico: 20 - 31 < 0
    assert p.cumulative_margin_gb == pytest.approx(19.0) and p.status == "blocked_disk"
    assert p.peak_margin_gb == pytest.approx(-11.0)


def _cfg_with(tmp_path: Path, mutate) -> Path:
    cfg = yaml.safe_load(disk_plan.PHASES_V1.read_text())
    mutate(cfg)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_missing_baseline_or_ids_and_duplicate_source_are_plan_errors(tmp_path: Path):
    cases = {
        "baseline ausente": lambda c: c.pop("baseline"),
        "baseline sem id": lambda c: c["baseline"][0].pop("id"),
        "baseline sem gb": lambda c: c["baseline"][0].pop("gb"),
        "baseline gb negativo": lambda c: c["baseline"][0].update(gb=-1.0),
        "fase sem id": lambda c: c["phases"][0].pop("id"),
        "fonte sem id": lambda c: c["phases"][0]["sources"][0].pop("id"),
        "id de fase repetido": lambda c: c["phases"][1].update(id=c["phases"][0]["id"]),
        "fonte duplicada": lambda c: c["phases"][1]["sources"].append(dict(c["phases"][0]["sources"][0])),
    }
    for name, mutate in cases.items():
        with pytest.raises(PlanError):
            plan_phases(FREE_62, 10.0, config=_cfg_with(tmp_path, mutate))
        assert name


def test_phase_requires_peak_margin():
    with pytest.raises(TypeError):
        Phase("x", ("a",), 1.0, 1.0, 1.0)  # peak_margin_gb e status são obrigatórios


def test_inputs_are_validated(tmp_path: Path):
    with pytest.raises(PlanError):
        plan_phases(-1, 10.0)
    with pytest.raises(PlanError):
        plan_phases(FREE_62, -1.0)
    with pytest.raises(PlanError):
        plan_phases(math.nan, 10.0)
    bad = yaml.safe_load(disk_plan.PHASES_V1.read_text())
    bad["phases"][0]["sources"][0]["surprise"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(PlanError, match="surprise"):
        plan_phases(FREE_62, 10.0, config=path)
    bad = yaml.safe_load(disk_plan.PHASES_V1.read_text())
    bad["phases"][0]["sources"][0].pop("hours")
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(PlanError):
        plan_phases(FREE_62, 10.0, config=path)


def test_write_disk_plan_json(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CCTRAIN_RUNS_DIR", str(tmp_path / "runs"))
    out = write_disk_plan(FREE_62, 10.0)
    assert out == tmp_path / "runs" / "m1" / "disk-plan.json"
    d = json.loads(out.read_text())
    assert d["free_gb"] == pytest.approx(62.0) and d["reserve_gb"] == 10.0
    assert d["baseline_gb"] == pytest.approx(13.0)
    assert [p["id"] for p in d["phases"]][0] == "1-noise-rir-encoder"
    assert all({"gb_final", "gb_peak", "cumulative_margin_gb", "peak_margin_gb", "status"} <= set(p)
               for p in d["phases"])
    assert d["total_gb"] == pytest.approx(sum(p["gb_final"] for p in d["phases"]) + 13.0)
    assert d["final_margin_gb"] == pytest.approx(d["phases"][-1]["cumulative_margin_gb"])
    assert d["min_peak_margin_gb"] == pytest.approx(min(p["peak_margin_gb"] for p in d["phases"]))
    assert d["blocked"] == []
    # o que o dono tirou do escopo custaria mais do que a margem: registrado para rastreio
    assert {e["id"] for e in d["excluded_by_owner"]} == {
        "librispeech-train-clean-360", "librispeech-train-other-500", "mls-pt"}
    assert d["what_if_with_excluded_margin_gb"] < 0 < d["final_margin_gb"]
    json.dumps(d, allow_nan=False)  # JSON sempre válido


def test_cli_uses_given_free_bytes(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("CCTRAIN_RUNS_DIR", str(tmp_path / "runs"))
    assert disk_plan.main(["--free-bytes", str(FREE_62), "--reserve-gb", "10"]) == 0
    assert (tmp_path / "runs" / "m1" / "disk-plan.json").is_file()
    assert "blocked_disk" not in capsys.readouterr().out
