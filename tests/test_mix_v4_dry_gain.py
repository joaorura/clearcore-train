"""Regressão: SIR efetivo = SIR sorteado, com e sem RIR (até 3b36625 o interferente seco recebia o ganho ao quadrado,
porque `wet[k]` e `early[k]` eram o mesmo array e os dois `*= g` são in-place). Dados sintéticos, sem disco do projeto."""

import sys

import numpy as np
import pytest
import soundfile as sf

from cctrain.data import mix_v4 as M


class _Bank:
    has_provenance = False

    def speakers(self):
        return ["a:0", "a:1", "a:2", "a:3", "a:4", "a:5"]

    def sample(self, key, rng):
        return rng.standard_normal(192).astype(np.float32)


def _pools(tmp_path):
    rng = np.random.default_rng(0)
    recs, rec_tier = {}, {}
    for s in range(6):
        lst = []
        for j in range(2):
            x = rng.standard_normal(48_000 * 7) * 0.1 * np.repeat(rng.random(700) < 0.7, 480)
            p = tmp_path / f"s{s}_{j}.wav"
            sf.write(p, x.astype(np.float32), 48_000)
            lst.append((p.name, 48_000, 24_000.0))
        recs[f"a:{s}"] = lst
        rec_tier[f"a:{s}"] = {"full": [0, 1]}
    nz = tmp_path / "noise.wav"
    sf.write(nz, (rng.standard_normal(48_000 * 5) * 0.05).astype(np.float32), 48_000)
    h = np.zeros(9600, np.float32)
    h[200] = 1.0
    h[201:] = (rng.standard_normal(9399) * np.exp(-np.arange(9399) / 1500) * 0.3).astype(np.float32)
    rir = tmp_path / "rir.wav"
    sf.write(rir, h, 48_000)
    keys = sorted(recs)
    return M.V4Pools(target_tiers={"full": keys, "khz24": keys, "khz16": keys}, recs=recs, rec_tier=rec_tier,
                     interf_keys=keys, neighbors={k: [x for x in keys if x != k] for k in keys},
                     noise=[(nz.name, 48_000, 24_000.0)], rirs=[rir.name])


@pytest.mark.parametrize("target_present", [True, False])
def test_effective_sir_matches_drawn_with_and_without_rir(tmp_path, monkeypatch, target_present):
    vp = M.V4Params(segment_s=2.0, batch_size=4, total_steps=100, phase1_frac=0.0, p_rir=0.5,
                    n_probs=(0.0, 0.4, 0.3, 0.2, 0.1), p_target_absent=0.0 if target_present else 1.0)
    ds = M.StageV4Dataset(_pools(tmp_path), _Bank(), tmp_path, vp)
    cap: dict = {}
    orig = M.extract_features

    def spy(x):
        cap.clear()
        cap.update(sys._getframe(1).f_locals)
        return orig(x)

    monkeypatch.setattr(M, "extract_features", spy)
    seen = {True: 0, False: 0}
    for i in range(80):
        d = ds[i]
        fr = cap["frames"]
        wet, early, n_int = cap["wet"], cap["early"], cap["n_int"]
        assert bool(d["has_target"]) == target_present
        for k in range(n_int):
            assert wet[k + 1] is not early[k + 1]
            # wet e early recebem o mesmo ganho: sem RIR continuam iguais entre si
            if not cap["reverb"]:
                np.testing.assert_allclose(early[k + 1], wet[k + 1] * cap["g"] * cap["lim"], rtol=1e-12, atol=0)
        if target_present:
            t_db = M.active_rms_db(wet[0], fr)
            for k, s in enumerate(cap["sirs"]):
                eff = t_db - M.active_rms_db(wet[k + 1], fr)
                assert abs(eff - s) < 0.1, (i, cap["reverb"], s, eff)
        else:
            r_db = M.active_rms_db(wet[1], fr)
            for k in range(1, n_int):  # interferentes entre si: o ±6 dB sorteado vale
                rel = M.active_rms_db(wet[k + 1], fr) - r_db
                assert -6.1 < rel < 6.1, (i, cap["reverb"], rel)
        seen[bool(cap["reverb"])] += 1
    assert seen[True] >= 10 and seen[False] >= 10, seen
