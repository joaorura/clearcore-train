"""P1 e P2 da spec §6.4 com FiLM não-identidade (o M0 só provou a identidade).

P1: ORT dos três grafos do asset aprovado com FiLM × PyTorch (pDFNet3 = DFNet3 upstream + FiLM), `atol 1e-5,
rtol 1e-6`. P2: `tract-check gating` com o mesmo FiLM × réplica cujo modelo de quadro é PyTorch, `|d| ≤ 1e-4 +
1e-4·|ref|`, em 1.000 quadros de mistura real. Três perfis: aleatório (gerador FiLM), extremos γ=e^±Γ e β=±B.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from cctrain.eval.runtime_path import (
    HOP,
    Stage,
    TorchFrameModel,
    _film_t,
    parity_mixture,
    replica_features,
    run_replica,
    split_film,
    stage_fractions,
    tract_gating,
)
from cctrain.export.film import HIDDEN, build_film_identity_asset, build_film_identity_members
from cctrain.hashing import sha256_file
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.models.film import FilmGenerator, load_film_config
from cctrain.paths import clearcore_root
from cctrain.tractcheck import APPROVED_ASSET_SHA256

pytestmark = [pytest.mark.tract, pytest.mark.clearcore, pytest.mark.data]
FRAMES = 1000
WINDOW = 8
P1_ATOL, P1_RTOL = 1e-5, 1e-6
# emb (512 canais, saída da GRU) deriva ~2e-5 em 1.000 quadros entre ORT e PyTorch (excesso +1,08e-5 sobre o atol literal)
EMB_FULL_SEQUENCE_ATOL = 1e-4
P2_ABS, P2_REL = 1e-4, 1e-4
PROFILES = ("random", "extreme_enc_up_df_down", "extreme_enc_down_df_up")


def _approved() -> Path:
    path = clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin"
    assert sha256_file(path) == APPROVED_ASSET_SHA256
    return path


def film_vectors(profile: str) -> np.ndarray:
    """[gamma_enc, beta_enc, gamma_df, beta_df], 4 x 256 f32 (mesma ordem do `tract-check gating ... film.f32`)."""
    cfg = load_film_config()
    gamma_max, beta_max = math.exp(cfg.log_gamma_bound), cfg.beta_bound
    if profile == "random":
        torch.manual_seed(5)
        gen = FilmGenerator(cfg)
        with torch.no_grad():
            # std 0.02: γ ∈ [0,85; 1,16], |β| ≤ 0,25. Com 0,3 a mistura cai 95% no estágio zero (lsnr < −10 dB) e
            # o ERB/DF quase não roda.
            gen.out.weight.normal_(std=0.02)
            gen.out.bias.normal_(std=0.02)
            emb = torch.randn(1, cfg.embedding_dim)
            emb = emb / emb.norm()
            v = gen(emb, torch.ones(1, 1))
        return np.concatenate([t.reshape(-1).numpy() for t in (v.gamma_enc, v.beta_enc, v.gamma_df, v.beta_df)])
    up, down = (gamma_max, beta_max), (1.0 / gamma_max, -beta_max)
    (ge, be), (gd, bd) = (up, down) if profile == "extreme_enc_up_df_down" else (down, up)
    return np.concatenate([np.full(HIDDEN, v, np.float32) for v in (ge, be, gd, bd)]).astype(np.float32)


def _split(vec: np.ndarray) -> dict[str, np.ndarray]:
    return split_film(vec)


def _to_torch(feat_erb: np.ndarray, feat_spec: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    ferb = torch.from_numpy(np.ascontiguousarray(feat_erb, np.float32))[None, None]
    fspec = torch.from_numpy(np.stack([feat_spec.real, feat_spec.imag], 0).astype(np.float32))[None]
    return ferb, fspec


@pytest.fixture(scope="module")
def mixture(paths) -> np.ndarray:
    x = parity_mixture(paths.data)
    assert x.size == FRAMES * HOP
    return x


@pytest.fixture(scope="module")
def base():
    torch.set_num_threads(1)
    model, _ = load_upstream_dfnet3(device="cpu")
    return model


@pytest.fixture(scope="module")
def film_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("film")
    build_film_identity_members(_approved(), out / "members")
    return out


@pytest.fixture(scope="module")
def sessions(film_dir):
    def make(name):
        return ort.InferenceSession(str(film_dir / "members" / name), providers=["CPUExecutionProvider"])

    return {g: make(f"{g}.onnx") for g in ("enc", "erb_dec", "df_dec")}


def _excess(got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> float:
    return float(np.max(np.abs(got - ref) - (atol + rtol * np.abs(ref))))


def _tile(a: np.ndarray, steps: int) -> np.ndarray:
    return np.broadcast_to(a, (1, steps, HIDDEN)).copy()


def _ort_vs_torch(base, sessions, ferb, fspec, vec, steps: int, atol_override: dict[str, float] | None = None,
                  ) -> dict[str, float]:
    """Excesso de P1 por saída dos 3 grafos sobre `steps` quadros, partindo de estado zero."""
    v = _split(vec)
    worst: dict[str, float] = {}
    with torch.no_grad():
        e0, e1, e2, e3, emb, c0, lsnr = base.enc(ferb, fspec, film=_film_t(v["gamma_enc"], v["beta_enc"]))
        m = base.erb_dec(emb, e3, e2, e1, e0)
        coefs = base.df_dec(emb, c0, film=_film_t(v["gamma_df"], v["beta_df"]))
    outs = dict(zip([o.name for o in sessions["enc"].get_outputs()],
                    sessions["enc"].run(None, {"feat_erb": ferb.numpy(), "feat_spec": fspec.numpy(),
                                               "gamma": _tile(v["gamma_enc"], steps),
                                               "beta": _tile(v["beta_enc"], steps)}), strict=True))
    for name, ref in zip(("e0", "e1", "e2", "e3", "emb", "c0", "lsnr"), (e0, e1, e2, e3, emb, c0, lsnr), strict=True):
        # lsnr = sigmoid · (lsnr_max − lsnr_min) + lsnr_min: o atol de 1e-5 vale na saída da sigmoid (escala 50 dB)
        atol = P1_ATOL * float(base.enc.lsnr_scale) if name == "lsnr" else P1_ATOL
        atol = (atol_override or {}).get(f"enc.{name}", atol)
        worst[f"enc.{name}"] = _excess(outs[name], ref.numpy(), atol, P1_RTOL)
    got_m = sessions["erb_dec"].run(None, {"emb": emb.numpy(), "e3": e3.numpy(), "e2": e2.numpy(), "e1": e1.numpy(),
                                           "e0": e0.numpy()})[0]
    worst["erb_dec.m"] = _excess(got_m, m.numpy(), P1_ATOL, P1_RTOL)
    got_c = sessions["df_dec"].run(None, {"emb": emb.numpy(), "c0": c0.numpy(), "gamma": _tile(v["gamma_df"], steps),
                                          "beta": _tile(v["beta_df"], steps)})[0]
    worst["df_dec.coefs"] = _excess(got_c, coefs.numpy(), P1_ATOL, P1_RTOL)
    return worst


@pytest.mark.parametrize("profile", PROFILES)
def test_p1_ort_matches_pytorch_with_non_identity_film(base, sessions, mixture, profile, m0_runs):
    vec = film_vectors(profile)
    v = _split(vec)
    assert not np.allclose(v["gamma_enc"], 1.0) and not np.allclose(v["beta_enc"], 0.0)
    assert not np.allclose(v["gamma_df"], 1.0) and not np.allclose(v["beta_df"], 0.0)
    _, feat_erb, feat_spec = replica_features(mixture)
    ferb, fspec = _to_torch(feat_erb, feat_spec)
    steps = ferb.shape[2]
    windows = {}
    for start in range(0, steps - WINDOW + 1, WINDOW):
        w = _ort_vs_torch(base, sessions, ferb[:, :, start:start + WINDOW], fspec[:, :, start:start + WINDOW], vec,
                          WINDOW)
        for k, x in w.items():
            windows[k] = max(windows.get(k, -1.0), x)
    # sequência inteira de 1.000 quadros: os mesmos atol/rtol literais; só enc.emb tem tolerância explícita maior
    drift = _ort_vs_torch(base, sessions, ferb, fspec, vec, steps, atol_override={"enc.emb": EMB_FULL_SEQUENCE_ATOL})
    (m0_runs / f"p1_film_{profile}.json").write_text(json.dumps(
        {"profile": profile, "frames": steps, "window": WINDOW, "atol": P1_ATOL, "rtol": P1_RTOL,
         "worst_excess_windowed": windows, "worst_excess_full_sequence": drift}, indent=2) + "\n")
    assert all(x <= 0 for x in windows.values()), windows
    assert all(x <= 0 for x in drift.values()), drift


@pytest.mark.parametrize("profile", PROFILES)
def test_p2_tract_matches_pytorch_with_non_identity_film(base, mixture, film_dir, tmp_path, profile, m0_runs):
    asset = build_film_identity_asset(_approved(), film_dir / "asset").path
    vec = film_vectors(profile)
    rt = tract_gating(asset, mixture, tmp_path / "rt", film=vec)
    identity = tract_gating(asset, mixture, tmp_path / "id")
    py = run_replica(mixture, TorchFrameModel(base, vec, mixture))
    excess = np.abs(py.audio - rt.audio) - (P2_ABS + P2_REL * np.abs(rt.audio))
    lsnr_excess = np.abs(py.lsnr - rt.lsnr) - (P2_ABS + P2_REL * np.abs(rt.lsnr))
    counts = {st.name.lower(): int(np.count_nonzero(rt.stages == int(st))) for st in Stage}
    report = {
        "profile": profile, "frames": FRAMES,
        "max_abs_diff": float(np.max(np.abs(py.audio - rt.audio))),
        "worst_excess_audio": float(excess.max()), "worst_excess_lsnr": float(lsnr_excess.max()),
        "stage_mismatches": int((py.stages != rt.stages).sum()),
        "stage_counts_tract": counts, "fractions_tract": stage_fractions(rt.stages),
        "film_vs_identity_max_abs_audio": float(np.max(np.abs(rt.audio - identity.audio))),
        "film_vs_identity_max_abs_lsnr": float(np.max(np.abs(rt.lsnr - identity.lsnr))),
    }
    (m0_runs / f"p2_film_{profile}.json").write_text(json.dumps(report, indent=2) + "\n")
    # o FiLM precisa mover a saída do tract (senão o teste seria o de identidade)
    assert report["film_vs_identity_max_abs_lsnr"] > 1e-2, report
    assert report["stage_mismatches"] == 0, report
    assert excess.max() <= 0, report
    assert lsnr_excess.max() <= 0, report


@pytest.mark.parametrize("profile", ["random", "extreme_enc_up_df_down"])
def test_p2_profiles_exercise_erb_and_df(mixture, film_dir, tmp_path, profile):
    """O P2 só vale se o ERB e o DF rodam: esses dois perfis mantêm o lsnr entre −10 e 20 dB em ≥ 20 quadros."""
    asset = build_film_identity_asset(_approved(), film_dir / "asset").path
    rt = tract_gating(asset, mixture, tmp_path / "rt", film=film_vectors(profile))
    for st in (Stage.ERB_ONLY, Stage.ERB_DF):
        assert int(np.count_nonzero(rt.stages == int(st))) >= 20, stage_fractions(rt.stages)
