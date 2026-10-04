"""Réplica Python de DfTract::process (vendor/crates/deep_filter/src/tract.rs) com gating por lsnr (spec §3.5).

Espelha, quadro a quadro: atalho de silêncio (média quadrática < 1e-7: saída zero, lsnr -15, sem análise nem
modelo), anéis rolling_spec_buf_x (5) e rolling_spec_buf_y (7), apply_stages, máscara ERB por banda, deep
filtering de ordem 5 nos 96 primeiros bins e síntese. O FrameModel tem estado e só avança o decoder que roda
(decisão de planejamento 1). emulate_gating_offline é a aproximação usada no treino (sem esse estado).
"""

from __future__ import annotations

import enum
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from cctrain import tractcheck
from cctrain.hashing import sha256_file

HOP = 480
N_FREQS = 481
NB_ERB = 32
NB_DF = 96
DF_ORDER = 5
CONV_LOOKAHEAD = 2
DF_LOOKAHEAD = 2
LOOKAHEAD = max(CONV_LOOKAHEAD, DF_LOOKAHEAD)
NORM_ALPHA = 0.99
SILENCE_MEAN_SQUARE = np.float32(1e-7)
SILENT_LSNR = np.float32(-15.0)
DEFAULT_THRESHOLDS = (np.float32(-10.0), np.float32(30.0), np.float32(20.0))
STATEFUL_SHA256 = {
    "enc.onnx": "c061d8a021d060f00a01a3191bf5a253247f359987b2a13043ef0ad46931191e",
    "erb_dec.onnx": "0c84ecc2841445c0a4eee5486f43212bf41241a4459a887bf067a79cdeb057b4",
    "df_dec.onnx": "a65a3dd9b763add2cfa7e822e7e910d0d70297426dff1b3863336a970b4ea655",
}


class Stage(enum.IntEnum):
    ZERO = 0
    ERB_ONLY = 1
    ERB_DF = 2
    NONE = 3
    SILENT = 4


_STAGE_OF = {(False, True, False): Stage.ZERO, (True, False, False): Stage.ERB_ONLY,
             (True, False, True): Stage.ERB_DF, (False, False, False): Stage.NONE}


def apply_stages(lsnr, thresholds=DEFAULT_THRESHOLDS) -> tuple[bool, bool, bool]:
    lo, hi_erb, hi_df = (np.float32(t) for t in thresholds)
    value = np.float32(lsnr)
    if value < lo:
        return (False, True, False)
    if value > hi_erb:
        return (False, False, False)
    if value > hi_df:
        return (True, False, False)
    return (True, False, True)


def stage_of(flags: tuple[bool, bool, bool]) -> Stage:
    return _STAGE_OF[flags]


def mean_square_f32(frame: np.ndarray) -> np.float32:
    x = np.asarray(frame, dtype=np.float32)
    total = np.add.accumulate(x * x, dtype=np.float32)[-1]
    return np.float32(total / np.float32(x.size))


def apply_band_gains(spec_frame: np.ndarray, gains: np.ndarray, widths: Sequence[int]) -> None:
    view = spec_frame.view(np.float32).reshape(-1, 2)
    start = 0
    for width, gain in zip(widths, gains):
        view[start : start + int(width)] *= np.float32(gain)
        start += int(width)


def deep_filter(frames: Sequence[np.ndarray], coefs: np.ndarray) -> np.ndarray:
    out = np.zeros(NB_DF, dtype=np.complex64)
    for i, frame in enumerate(frames):
        out += frame[:NB_DF] * coefs[:, i]
    return out


@dataclass
class EncoderStep:
    lsnr: np.float32
    tensors: dict[str, np.ndarray]


class FrameModel(Protocol):
    def reset(self) -> None: ...

    def encode(self, feat_erb: np.ndarray, feat_spec: np.ndarray) -> EncoderStep: ...

    def erb_decode(self, step: EncoderStep) -> np.ndarray: ...

    def df_decode(self, step: EncoderStep) -> np.ndarray: ...


class StatefulOrtFrameModel:
    ENC_STATE = {"h_in": ("h_out", (1, 1, 256)), "feat_erb_buf": ("feat_erb_buf_out", (1, 1, 2, 32)),
                 "feat_spec_buf": ("feat_spec_buf_out", (1, 2, 2, 96))}
    ERB_STATE = {"h_in": ("h_out", (2, 1, 256))}
    DF_STATE = {"h_in": ("h_out", (2, 1, 256)), "c0_buf": ("c0_buf_out", (1, 64, 4, 96))}

    def __init__(self, model_dir: Path, verify_sha256: bool = True):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        if verify_sha256:
            for name, sha in STATEFUL_SHA256.items():
                if sha256_file(model_dir / name) != sha:
                    raise ValueError(f"{model_dir / name}: SHA-256 não confere com models/stateful/README.md")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._sessions = {g: ort.InferenceSession(str(model_dir / f"{g}.onnx"), opts,
                                                  providers=["CPUExecutionProvider"])
                          for g in ("enc", "erb_dec", "df_dec")}
        self._names = {g: [o.name for o in s.get_outputs()] for g, s in self._sessions.items()}
        self._specs = {"enc": self.ENC_STATE, "erb_dec": self.ERB_STATE, "df_dec": self.DF_STATE}
        self.reset()

    def reset(self) -> None:
        self._state = {g: {k: np.zeros(shape, np.float32) for k, (_, shape) in spec.items()}
                       for g, spec in self._specs.items()}

    def _run(self, graph: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(zip(self._names[graph], self._sessions[graph].run(None, {**feeds, **self._state[graph]})))
        for inp, (outp, _) in self._specs[graph].items():
            self._state[graph][inp] = out.pop(outp)
        return out

    def encode(self, feat_erb: np.ndarray, feat_spec: np.ndarray) -> EncoderStep:
        out = self._run("enc", {
            "feat_erb": np.asarray(feat_erb, np.float32).reshape(1, 1, 1, NB_ERB),
            "feat_spec": np.ascontiguousarray(np.asarray(feat_spec, np.float32).T).reshape(1, 2, 1, NB_DF),
        })
        lsnr = np.float32(out.pop("lsnr").reshape(-1)[0])
        return EncoderStep(lsnr, out)

    def erb_decode(self, step: EncoderStep) -> np.ndarray:
        out = self._run("erb_dec", {k: step.tensors[k] for k in ("emb", "e3", "e2", "e1", "e0")})
        return out["m"].reshape(-1)[:NB_ERB].astype(np.float32)

    def df_decode(self, step: EncoderStep) -> np.ndarray:
        out = self._run("df_dec", {"emb": step.tensors["emb"], "c0": step.tensors["c0"]})
        c = out["coefs"].reshape(NB_DF, DF_ORDER, 2)  # mesma reinterpretação de tract.rs (set_shape)
        return (c[..., 0] + 1j * c[..., 1]).astype(np.complex64)


@dataclass
class ReplicaResult:
    audio: np.ndarray
    lsnr: np.ndarray
    stages: np.ndarray


def run_replica(audio: np.ndarray, model: FrameModel, thresholds=DEFAULT_THRESHOLDS) -> ReplicaResult:
    from libdf import DF, erb, erb_norm, unit_norm

    frames = len(audio) // HOP
    framed = np.asarray(audio[: frames * HOP], np.float32).reshape(frames, HOP)
    silent = np.array([mean_square_f32(f) < SILENCE_MEAN_SQUARE for f in framed], dtype=bool)
    active = np.flatnonzero(~silent)
    out = np.zeros((frames, HOP), np.float32)
    lsnr = np.full(frames, SILENT_LSNR, np.float32)
    stages = np.full(frames, int(Stage.SILENT), np.uint8)
    model.reset()
    if active.size:
        state = DF(48_000, 960, HOP, NB_ERB, 2)
        spec = state.analysis(np.ascontiguousarray(framed[active].reshape(1, -1)))
        feat_erb = np.asarray(erb_norm(erb(spec, state.erb_widths()), NORM_ALPHA))[0]
        feat_spec = np.asarray(unit_norm(np.ascontiguousarray(spec[..., :NB_DF]), NORM_ALPHA))[0]
        widths = np.asarray(state.erb_widths())
        spec = spec[0]
        rx = deque(np.zeros(N_FREQS, np.complex64) for _ in range(max(DF_ORDER, LOOKAHEAD)))
        ry = deque(np.zeros(N_FREQS, np.complex64) for _ in range(DF_ORDER + CONV_LOOKAHEAD))
        enhanced = np.zeros_like(spec)
        for t in range(spec.shape[0]):
            rx.popleft()
            ry.popleft()
            rx.append(spec[t].copy())
            ry.append(spec[t].copy())
            step = model.encode(feat_erb[t], np.stack([feat_spec[t].real, feat_spec[t].imag], axis=-1))
            flags = apply_stages(step.lsnr, thresholds)
            gains = model.erb_decode(step) if flags[0] else (np.zeros(NB_ERB, np.float32) if flags[1] else None)
            coefs = model.df_decode(step) if flags[2] else None
            target = ry[DF_ORDER - 1]
            if gains is not None:
                apply_band_gains(target, gains, widths)
            frame_out = target.copy()
            if coefs is not None:
                frame_out[:NB_DF] = deep_filter(list(rx), coefs)
            enhanced[t] = frame_out
            lsnr[active[t]] = step.lsnr
            stages[active[t]] = int(stage_of(flags))
        synth = np.asarray(state.synthesis(np.ascontiguousarray(enhanced[None])), np.float32)
        out[active] = synth.reshape(-1, HOP)
    return ReplicaResult(out.reshape(-1), lsnr, stages)


def tract_gating(asset: Path, audio: np.ndarray, out_prefix: Path, film: np.ndarray | None = None) -> ReplicaResult:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    inp = Path(f"{out_prefix}.in.f32")
    np.asarray(audio, "<f4").tofile(inp)
    args = ["gating", asset, inp, out_prefix]
    if film is not None:
        film_path = Path(f"{out_prefix}.film.f32")
        np.asarray(film, "<f4").tofile(film_path)
        args.append(film_path)
    proc = tractcheck.run(*args)
    if proc.returncode != 0:
        raise RuntimeError(f"tract-check gating falhou: {proc.stdout}{proc.stderr}")
    return ReplicaResult(np.fromfile(f"{out_prefix}.out.f32", "<f4"), np.fromfile(f"{out_prefix}.lsnr.f32", "<f4"),
                         np.fromfile(f"{out_prefix}.stage.u8", np.uint8))


RUNTIME_DELAY = 1440  # latência do runtime (G0: latency_samples)
FILM_HIDDEN = 256


def split_film(vec: np.ndarray) -> dict[str, np.ndarray]:
    """`[gamma_enc, beta_enc, gamma_df, beta_df]`, 4 x 256 f32 (a ordem de `tract-check gating ... film.f32`)."""
    v = np.asarray(vec, np.float32).reshape(-1)
    if v.size != 4 * FILM_HIDDEN:
        raise ValueError(f"vetor FiLM com {v.size} valores (esperado {4 * FILM_HIDDEN})")
    return dict(zip(("gamma_enc", "beta_enc", "gamma_df", "beta_df"), np.split(v, 4), strict=True))


def identity_film() -> np.ndarray:
    return np.concatenate([np.ones(FILM_HIDDEN, np.float32), np.zeros(FILM_HIDDEN, np.float32)] * 2)


def replica_features(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesmas features que `run_replica` entrega ao modelo: só quadros ativos. -> (active, feat_erb, feat_spec)."""
    from libdf import DF, erb, erb_norm, unit_norm

    audio = np.asarray(audio, np.float32)
    framed = audio[: len(audio) // HOP * HOP].reshape(-1, HOP)  # como run_replica: o resto < 1 hop é descartado
    active = np.flatnonzero([mean_square_f32(f) >= SILENCE_MEAN_SQUARE for f in framed])
    if active.size == 0:
        return active, np.zeros((0, NB_ERB), np.float32), np.zeros((0, NB_DF), np.complex64)
    state = DF(48_000, 960, HOP, NB_ERB, 2)
    spec = state.analysis(np.ascontiguousarray(framed[active].reshape(1, -1)))
    feat_erb = np.asarray(erb_norm(erb(spec, state.erb_widths()), NORM_ALPHA))[0]
    feat_spec = np.asarray(unit_norm(np.ascontiguousarray(spec[..., :NB_DF]), NORM_ALPHA))[0]
    return active, feat_erb.astype(np.float32), feat_spec


def _film_t(a: np.ndarray, b: np.ndarray) -> tuple[Tensor, Tensor]:
    return torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None]


class TorchFrameModel:
    """`FrameModel` cujo cálculo é o do PyTorch (DFNet3 + FiLM), pré-calculado e servido por quadro.

    É o caminho que mede os pesos do estágio F sem o export do M3 (o tract só roda grafos exportados). O estágio de cada
    quadro sai do lsnr do encoder (que roda sempre); os decoders só avançam nos quadros em que o runtime os executa,
    então cada decoder roda sobre a subsequência dos quadros que o chamam, como o tract. O encoder stateful do runtime
    equivale ao `enc` do PyTorch sobre os quadros sem `pad_feat` (a defasagem de lookahead é feita pelos anéis de
    espectro da réplica). `vec` = `[gamma_enc, beta_enc, gamma_df, beta_df]`; `None` = FiLM neutro (γ=1, β=0).
    """

    def __init__(self, base, vec: np.ndarray | None, audio: np.ndarray, thresholds=DEFAULT_THRESHOLDS):
        _, self.feat_erb, self.feat_spec = replica_features(audio)
        v = split_film(identity_film() if vec is None else vec)
        self.lsnr = np.zeros(0, np.float32)
        self.gains: dict[int, np.ndarray] = {}
        self.coefs: dict[int, np.ndarray] = {}
        self.t = 0
        if not len(self.feat_erb):
            return
        ferb = torch.from_numpy(np.ascontiguousarray(self.feat_erb))[None, None]
        fspec = torch.from_numpy(np.stack([self.feat_spec.real, self.feat_spec.imag], 0).astype(np.float32))[None]
        with torch.no_grad():
            e0, e1, e2, e3, emb, c0, lsnr = base.enc(ferb, fspec, film=_film_t(v["gamma_enc"], v["beta_enc"]))
            self.lsnr = lsnr.reshape(-1).numpy().astype(np.float32)
            flags = [apply_stages(x, thresholds) for x in self.lsnr]
            erb_idx = [t for t, f in enumerate(flags) if f[0]]
            df_idx = [t for t, f in enumerate(flags) if f[2]]
            if erb_idx:
                i = torch.tensor(erb_idx)
                m = base.erb_dec(emb[:, i], e3[:, :, i], e2[:, :, i], e1[:, :, i], e0[:, :, i])
                self.gains = {t: m[0, 0, k].numpy().astype(np.float32) for k, t in enumerate(erb_idx)}
            if df_idx:
                i = torch.tensor(df_idx)
                c = base.df_dec(emb[:, i], c0[:, :, i], film=_film_t(v["gamma_df"], v["beta_df"]))
                c = c[0].numpy().reshape(len(df_idx), NB_DF, DF_ORDER, 2)
                self.coefs = {t: (c[k, ..., 0] + 1j * c[k, ..., 1]).astype(np.complex64)
                              for k, t in enumerate(df_idx)}

    def reset(self) -> None:
        self.t = 0

    def encode(self, feat_erb: np.ndarray, feat_spec: np.ndarray) -> EncoderStep:
        t = self.t
        assert np.array_equal(np.asarray(feat_erb, np.float32), self.feat_erb[t]), f"features desalinhadas em {t}"
        assert np.array_equal(np.asarray(feat_spec), np.stack([self.feat_spec[t].real, self.feat_spec[t].imag], -1))
        self.t += 1
        return EncoderStep(np.float32(self.lsnr[t]), {"t": t})

    def erb_decode(self, step: EncoderStep) -> np.ndarray:
        return self.gains[step.tensors["t"]]

    def df_decode(self, step: EncoderStep) -> np.ndarray:
        return self.coefs[step.tensors["t"]]


def align_runtime(audio: np.ndarray, n: int, delay: int = RUNTIME_DELAY) -> np.ndarray:
    """Remove a latência do runtime (1.440 amostras) e ajusta ao comprimento `n` da entrada."""
    est = np.asarray(audio, np.float32)[delay:]
    return np.concatenate([est, np.zeros(max(0, n - est.size), np.float32)])[:n]


def runtime_forward(model, film: np.ndarray | None, noisy: np.ndarray, thresholds=DEFAULT_THRESHOLDS
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Caminho do runtime em Python (réplica quadro a quadro de `DfTract::process` com FiLM e gating §3.5).

    `model`: DFNet3/pDFNet3 PyTorch (aceita `PDFNet3`: usa `.base`), `film`: vetor `[γ_enc, β_enc, γ_df, β_df]`
    (4 x 256) ou `None` (sem perfil, FiLM neutro). `noisy`: mono 48 kHz. Devolve `(enh, stage_per_frame)`: `enh` com o
    comprimento da entrada e a latência de 1.440 amostras já removida (alinhada à entrada); `stage_per_frame` (uint8,
    `Stage`) é indexado pelos quadros da entrada. Paridade com o `tract-check gating`: `tests/test_eval_runtime_replica.py`.
    """
    base = getattr(model, "base", model)
    audio = np.asarray(noisy, np.float32).reshape(-1)
    res = run_replica(audio, TorchFrameModel(base, film, audio, thresholds), thresholds)
    return align_runtime(res.audio, audio.size), res.stages


def emulate_gating_offline(spec_noisy: Tensor, spec_masked: Tensor, spec_enhanced: Tensor, lsnr: Tensor,
                           thresholds=DEFAULT_THRESHOLDS) -> tuple[Tensor, Tensor]:
    lo, hi_erb, hi_df = (float(t) for t in thresholds)
    level = lsnr.detach()
    zero = level < lo
    none = (~zero) & (level > hi_erb)
    erb_only = (~zero) & (~none) & (level > hi_df)

    def w(mask: Tensor) -> Tensor:
        return mask[:, None, :, None, None]

    out = torch.where(w(zero), torch.zeros_like(spec_enhanced), spec_enhanced)
    out = torch.where(w(erb_only), spec_masked, out)
    out = torch.where(w(none), spec_noisy, out)
    stages = torch.full_like(level, int(Stage.ERB_DF), dtype=torch.long)
    stages[zero] = int(Stage.ZERO)
    stages[erb_only] = int(Stage.ERB_ONLY)
    stages[none] = int(Stage.NONE)
    return out, stages


def stage_fractions(stages: np.ndarray) -> dict[str, float]:
    s = np.asarray(stages)
    return {st.name.lower(): float(np.mean(s == int(st))) for st in Stage}


PARITY_SNR_SCHEDULE_DB = (40.0, 25.0, 10.0, 5.0, -5.0)
PARITY_SILENT_RANGES = ((0, 20), (500, 520))


def parity_mixture(data_root: Path, frames: int = 1000, snr_schedule_db: Sequence[float] = PARITY_SNR_SCHEDULE_DB,
                   block_frames: int = 200, silent_ranges: Sequence[tuple[int, int]] = PARITY_SILENT_RANGES
                   ) -> np.ndarray:
    """Fala VCTK + ruído em blocos de `block_frames` quadros com SNR variável (um por bloco, cíclico) e lacunas
    silenciosas (quadros [a, b)) no início e no meio, para exercitar todos os estágios e o congelamento/retomada
    do estado nos quadros pulados."""
    from cctrain.data.audio import load_audio, resample

    data_root = Path(data_root)
    parts = []
    for name in ("p225_003", "p225_008", "p225_011", "p225_022"):
        x, sr = load_audio(data_root / f"raw/vctk-0.92/test/{name}_mic1.flac")
        parts += [resample(x, sr, 48_000), np.zeros(9_600, np.float32)]
    n = frames * HOP
    speech = np.concatenate(parts)
    speech = np.tile(speech, int(np.ceil(n / speech.size)))[:n].astype(np.float64)
    noise, sr = load_audio(data_root / "raw/noise-wikimedia-cc0/street_traffic_rain_cc0_10s.wav")
    noise = np.tile(resample(noise, sr, 48_000), int(np.ceil(n / (10 * 48_000)) + 1))[:n].astype(np.float64)
    block = block_frames * HOP
    for k, start in enumerate(range(0, n, block)):
        sl = slice(start, min(start + block, n))
        snr_db = snr_schedule_db[k % len(snr_schedule_db)]
        noise[sl] *= np.sqrt(np.mean(speech[sl] ** 2) / (np.mean(noise[sl] ** 2) * 10 ** (snr_db / 10)))
    x = speech + noise
    x *= min(1.0, 0.9 / np.max(np.abs(x)))
    for a, b in silent_ranges:
        x[a * HOP : b * HOP] = 0.0
    return x.astype(np.float32)
