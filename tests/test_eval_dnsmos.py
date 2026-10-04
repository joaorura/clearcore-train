"""DNSMOS P.835 e personalizado (M2 Tarefa 3): janelamento e polinômios do `dnsmos_local.py` do DNS-Challenge."""

from __future__ import annotations

import numpy as np
import pytest

from cctrain.eval import dnsmos as D
from cctrain.paths import Paths


class _Sess:
    """Sessão ONNX falsa: devolve sempre os mesmos 3 valores brutos e registra as janelas recebidas."""

    def __init__(self, raw=(3.0, 3.5, 2.5)):
        self.raw, self.calls = np.array([[raw]], np.float32), []

    def run(self, _outputs, feed):
        self.calls.append(feed["input_1"].shape)
        return [self.raw]


def test_polynomial_constants_match_the_reference_script():
    assert D.POLY["p835"]["ovrl"] == (-0.06766283, 1.11546468, 0.04602535)
    assert D.POLY["personalized"]["bak"] == (-0.04976499, 0.44276479, -0.1644611, 0.96883132)
    assert D.INPUT_LENGTH_S == 9.01 and D.SAMPLING_RATE == 16_000


@pytest.mark.parametrize("seconds,hops", [(5.0, 1), (9.5, 1), (12.0, 3), (20.3, 7)])
def test_hop_count_follows_the_reference(seconds, hops):
    # 20,3 s: int(floor(20,3) - 9,01) + 1 = 11 janelas, mas o script (e esta cópia) descarta a janela cujo fim
    # `int((idx + 9,01) * fs)` trunca 1 amostra abaixo de 144.160 por aritmética de ponto flutuante: restam 7.
    sess = _Sess()
    s = D.DnsmosScorer(sess)
    s.score(np.zeros(int(seconds * 16_000), np.float32) + 0.01)
    assert len(sess.calls) == hops
    assert all(c == (1, int(9.01 * 16_000)) for c in sess.calls)


def test_polyfit_is_applied_per_segment_then_averaged():
    raw = (3.0, 3.5, 2.5)
    out = D.DnsmosScorer(_Sess(raw)).score(np.zeros(16_000 * 10, np.float32))
    assert out["sig"] == pytest.approx(np.polyval(D.POLY["p835"]["sig"], raw[0]))
    assert out["bak"] == pytest.approx(np.polyval(D.POLY["p835"]["bak"], raw[1]))
    assert out["ovrl"] == pytest.approx(np.polyval(D.POLY["p835"]["ovrl"], raw[2]))
    outp = D.DnsmosScorer(_Sess(raw), personalized=True).score(np.zeros(16_000 * 10, np.float32))
    assert outp["ovrl"] == pytest.approx(np.polyval(D.POLY["personalized"]["ovrl"], raw[2]))
    assert outp["ovrl"] != out["ovrl"]


def test_polyfit_before_mean_not_after():
    class Alt(_Sess):
        def run(self, _o, feed):
            self.calls.append(feed["input_1"].shape)
            return [np.array([[(2.0, 3.0, 4.0) if len(self.calls) % 2 else (4.0, 5.0, 2.0)]], np.float32)]

    out = D.DnsmosScorer(Alt()).score(np.zeros(16_000 * 11, np.float32))  # 2 janelas
    want = np.mean([np.polyval(D.POLY["p835"]["ovrl"], v) for v in (4.0, 2.0)])
    assert out["ovrl"] == pytest.approx(want)
    assert out["ovrl"] != pytest.approx(np.polyval(D.POLY["p835"]["ovrl"], 3.0), abs=1e-6)


def test_dnsmos_metric_resamples_and_returns_arrays():
    s = D.DnsmosScorer(_Sess())
    res = D.dnsmos([np.zeros(48_000 * 10, np.float32) + 0.01, np.zeros(48_000 * 11, np.float32) + 0.01], None, 48_000,
                   scorer=s)
    assert set(res) == {"sig", "bak", "ovrl"} and res["ovrl"].shape == (2,)


@pytest.fixture(scope="module")
def real_models():
    d = Paths.from_env().data / "models" / "dnsmos"
    if not (d / "DNSMOS" / "sig_bak_ovr.onnx").is_file():
        pytest.skip(f"modelos DNSMOS ausentes em {d} (rode scripts/m2_t3.py fetch)")
    return d


@pytest.mark.data
def test_real_models_hash_pinned_and_sane_scores(real_models):
    from cctrain.eval.models import verify_model

    verify_model("dnsmos")
    rng = np.random.default_rng(0)
    noise = (0.05 * rng.standard_normal(16_000 * 10)).astype(np.float32)
    for pers in (False, True):
        sc = D.DnsmosScorer.from_dir(real_models, personalized=pers)
        out = sc.score(noise)
        assert all(1.0 <= out[k] <= 5.0 for k in ("sig", "bak", "ovrl"))
