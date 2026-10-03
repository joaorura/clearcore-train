from pathlib import Path

import pytest

from cctrain.paths import ConfigError, Paths, clearcore_root, repo_root


def test_repo_root_contains_pyproject():
    assert (repo_root() / "pyproject.toml").is_file()


def test_defaults_live_inside_repo():
    p = Paths.from_env({})
    assert p.data == repo_root() / "data"
    assert p.runs == repo_root() / "runs"
    assert p.delivery == repo_root() / "delivery"


def test_env_overrides_data_and_runs(tmp_path: Path):
    p = Paths.from_env({"CCTRAIN_DATA_DIR": str(tmp_path / "d"), "CCTRAIN_RUNS_DIR": str(tmp_path / "r")})
    assert p.data == (tmp_path / "d").resolve()
    assert p.runs == (tmp_path / "r").resolve()


def test_clearcore_root_is_required():
    with pytest.raises(ConfigError, match="CCTRAIN_CLEARCORE_ROOT"):
        clearcore_root({})


def test_clearcore_root_must_have_approved_asset(tmp_path: Path):
    with pytest.raises(ConfigError, match="df-compatible-release-asset-v1.bin"):
        clearcore_root({"CCTRAIN_CLEARCORE_ROOT": str(tmp_path)})


def test_no_personal_path_in_sources():
    home = str(Path.home())
    for f in (repo_root() / "src").rglob("*.py"):
        assert home not in f.read_text(), f"caminho pessoal em {f}"
