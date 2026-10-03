import pytest

from cctrain.eval.gates import load_budget
from cctrain.m0.report import go_no_go, project_disk, render_markdown, t3_hours

DISK = {"by_sr": {"16000": {"flac_gb_per_hour": 0.06}, "48000": {"flac_gb_per_hour": 0.17}},
        "rir": {"count": 64, "stored_bytes": 64 * 100_000},
        "dirs_bytes": {"venv": 7e9, "tract_check_target": 3e9, "rustup": 1.5e9, "data": 1e9, "third_party": 1e7}}
MINIMAL = {"speech": [{"id": "a", "hours": 10.0, "sr": 48000}, {"id": "b", "hours": 10.0, "sr": 24000},
                      {"id": "c", "fixed_gb": 1.0}],
           "noise": [{"id": "n", "hours": 1.0, "sr": 16000}], "rir": {"count": 1000},
           "estimated_fixed_gb": {"x": 2.0}}


def test_disk_projection_uses_measured_factors_and_flags_interpolation():
    p = project_disk(MINIMAL, DISK)
    rows = {r["id"]: r for r in p["rows"]}
    assert rows["a"]["gb"] == pytest.approx(1.7)
    assert rows["b"]["factor_source"] == "interpolado"
    assert rows["c"]["gb"] == 1.0
    assert rows["rir"]["gb"] == pytest.approx(0.1)
    assert p["measured_gb"]["venv"] == pytest.approx(7.0)
    assert p["total_gb"] == pytest.approx(sum(r["gb"] for r in p["rows"]) + 7.0 + 3.0 + 1.5 + 2.0)


def test_t3_projection_formula():
    t3 = {"epochs": 20, "hours_audio_per_epoch": 100.0, "segment_s": 3.0, "batch": 16}
    assert t3_hours(t3, 0.5) == pytest.approx(20 * (100 * 3600 / 3) / 16 * 0.5 / 3600)


def test_go_no_go_levels_come_from_frozen_budget():
    b = load_budget()
    rows = go_no_go(b, disk_projection_gb=40.0, free_gb=55.0,
                    step16={"peak_reserved_gb": 6.5, "s_per_step_median": 0.2}, loader={"rss_max_gb": 5.0})
    verdict = {r["item"]: r["verdict"] for r in rows}
    assert verdict == {"disco": "go", "vram_b16": "go", "ram_loader": "go", "tempo_t3": "go"}
    rows = go_no_go(b, disk_projection_gb=50.0, free_gb=55.0,
                    step16={"peak_reserved_gb": 7.5, "s_per_step_median": 2.0}, loader={"rss_max_gb": 7.0})
    verdict = {r["item"]: r["verdict"] for r in rows}
    assert verdict["disco"] == "no-go" and verdict["vram_b16"] == "no-go" and verdict["ram_loader"] == "no-go"
    assert verdict["tempo_t3"] == "no-go"


def test_markdown_lists_every_section():
    md = render_markdown({"title": "M0", "sections": [("Identidade", "I1 ok"), ("Disco", "x")],
                          "go_no_go": [{"item": "disco", "value": 1, "limit": 2, "verdict": "go"}]})
    assert "## Identidade" in md and "## Disco" in md and "| disco |" in md
