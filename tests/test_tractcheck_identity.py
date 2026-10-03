import json

import pytest

from cctrain import tractcheck
from cctrain.hashing import sha256_file
from cctrain.paths import clearcore_root

pytestmark = [pytest.mark.tract, pytest.mark.clearcore]


def test_i2_on_clearcore_film_identity_fixture(m0_runs):
    root = clearcore_root()
    film = root / "fixtures/film/film-identity-asset.tar.gz"
    approved = root / "vendor/approved/df-compatible-release-asset-v1.bin"
    golden = root / "fixtures/film/upstream-golden-1000f.f32"
    assert sha256_file(film) == tractcheck.CLEARCORE_FILM_IDENTITY_SHA256
    assert sha256_file(approved) == tractcheck.APPROVED_ASSET_SHA256
    out = m0_runs / "i2_clearcore_fixture.json"
    proc = tractcheck.run("identity", film, approved, golden, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text())
    assert report["pass"] is True
    assert report["film_hidden"] == 256
    assert report["max_abs_diff_explicit_vs_golden"] == 0.0
    assert report["max_abs_diff_explicit_vs_approved"] == 0.0
