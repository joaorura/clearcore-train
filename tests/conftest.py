import pytest

from cctrain.paths import Paths


@pytest.fixture(scope="session")
def paths() -> Paths:
    return Paths.from_env()


@pytest.fixture(scope="session")
def m0_runs(paths: Paths):
    out = paths.runs / "m0"
    out.mkdir(parents=True, exist_ok=True)
    return out
