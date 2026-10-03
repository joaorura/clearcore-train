"""Loader do piloto (IterableDataset): mistura em CPU e features libdf por amostra (spec §5.1).

Só aceita registros na allowlist de treino. Usado no M0 para medir amostras/s e RSS; a receita completa
(enrollment, máscara de perfil, banco de embeddings) é do M2.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from cctrain.data.audio import load_audio, resample
from cctrain.data.features import SR, analysis, extract_features
from cctrain.data.manifest import (
    FILES_DIR,
    TRAIN_SPLITS,
    FileRecord,
    ManifestError,
    SourceManifest,
    load_sources,
    read_file_manifest,
    training_records,
)
from cctrain.data.mix import (
    SIR_RANGE_DB,
    SNRS_DB,
    MixKind,
    crop_or_pad,
    loop_to_length,
    make_mixture,
    mixture_fractions,
)

# O loader de piloto é o único que aceita `pilot` (medição do M0); o trainer do T1 usa TRAIN_SPLITS.
PILOT_SPLITS = (*TRAIN_SPLITS, "pilot")


@dataclass(frozen=True)
class PilotLoaderConfig:
    segment_s: float = 3.0
    silence_fraction: float = 0.05
    p_rir: float = 0.3
    p_two_interferers: float = 0.5
    seed: int = 20261003


class PilotMixDataset(IterableDataset):
    def __init__(self, speech: Sequence[FileRecord], noise: Sequence[FileRecord], rirs: Sequence[FileRecord],
                 data_root: Path, cfg: PilotLoaderConfig | None = None,
                 sources: Mapping[str, SourceManifest] | None = None,
                 allowed_splits: Sequence[str] = PILOT_SPLITS):
        super().__init__()
        sources = sources or load_sources()
        allowed = frozenset(allowed_splits)
        self.speech = training_records([r for r in speech if r.split in allowed], sources, allow_pilot=True)
        self.noise = training_records([r for r in noise if r.split in allowed], sources, allow_pilot=True)
        self.rirs = training_records([r for r in rirs if r.split in allowed], sources, allow_pilot=True)
        self.by_speaker: dict[str, list[FileRecord]] = defaultdict(list)
        for r in self.speech:
            if r.speaker_key is None:
                raise ManifestError(f"{r.path}: registro de fala sem speaker")
            self.by_speaker[r.speaker_key].append(r)
        if len(self.by_speaker) < 2:
            raise ValueError("o piloto precisa de pelo menos 2 falantes")
        if not self.noise:
            raise ValueError("sem ruído")
        self.speakers = sorted(self.by_speaker)
        self.data_root = Path(data_root)
        self.cfg = cfg = cfg or PilotLoaderConfig()
        fr = mixture_fractions(cfg.silence_fraction)
        self.kinds = list(fr)
        self.kind_p = np.array([fr[k] for k in self.kinds])

    def _load48(self, r: FileRecord) -> np.ndarray:
        x, sr = load_audio(self.data_root / r.path)
        return resample(x, sr, SR)

    def _sample(self, rng: np.random.Generator) -> dict:
        n = int(self.cfg.segment_s * SR)
        kind = self.kinds[int(rng.choice(len(self.kinds), p=self.kind_p))]
        spk = self.speakers[int(rng.integers(len(self.speakers)))]
        target = crop_or_pad(self._load48(self.by_speaker[spk][int(rng.integers(len(self.by_speaker[spk])))]), n, rng)
        others = [s for s in self.speakers if s != spk]
        interferers = []
        if kind is not MixKind.TARGET_NOISE:
            count = 2 if (rng.random() < self.cfg.p_two_interferers and len(others) > 1) else 1
            for s in rng.choice(others, size=count, replace=False):
                recs = self.by_speaker[str(s)]
                interferers.append(crop_or_pad(self._load48(recs[int(rng.integers(len(recs)))]), n, rng))
        noise = loop_to_length(self._load48(self.noise[int(rng.integers(len(self.noise)))]), n, rng)
        if kind is MixKind.INTERFERER_ONLY and rng.random() < 0.5:
            noise = None
        def rir():
            if self.rirs and rng.random() < self.cfg.p_rir:
                return self._load48(self.rirs[int(rng.integers(len(self.rirs)))])
            return None
        snr = float(SNRS_DB[int(rng.integers(len(SNRS_DB)))])
        sir = float(rng.uniform(*SIR_RANGE_DB)) if interferers else float("nan")
        mix = make_mixture(target, interferers, noise, kind=kind, snr_db=snr, sir_db=sir, rir_target=rir(),
                           rir_interferer=rir())
        feats = extract_features(mix.noisy)
        return {"noisy": mix.noisy, "clean": mix.clean, "spec": feats.spec, "spec_clean": analysis(mix.clean),
                "feat_erb": feats.feat_erb, "feat_spec": feats.feat_spec, "kind": int(kind), "snr_db": snr,
                "sir_db": sir}

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.cfg.seed, info.id if info else 0])
        while True:
            yield self._sample(rng)


def pilot_records(data_root: Path, librispeech_speakers: int = 20, allowed_splits: Sequence[str] = PILOT_SPLITS
                  ) -> tuple[list[FileRecord], list[FileRecord], list[FileRecord]]:
    """Registros do piloto já filtrados por `allowed_splits` (dev/teste nunca entram; spec §4.5)."""
    allowed = frozenset(allowed_splits)

    def read(path: Path) -> list[FileRecord]:
        return [r for r in read_file_manifest(path) if r.split in allowed]

    vctk = read(FILES_DIR / "vctk-0.92-pilot.jsonl")
    libri = read(FILES_DIR / "librispeech-dev-clean-pilot.jsonl")
    keep = sorted({r.speaker for r in libri})[:librispeech_speakers]
    speech = vctk + [r for r in libri if r.speaker in keep]
    cv_manifest = Path(data_root) / "manifests" / "cv27-pt.jsonl"
    if cv_manifest.is_file():
        speech += read(cv_manifest)
    noise = read(FILES_DIR / "noise-wikimedia-cc0-pilot.jsonl")  # split test: filtrado fora do treino
    synthetic_noise = FILES_DIR / "noise-synthetic-v1-pilot.jsonl"  # pilot-only, cost measurement
    if synthetic_noise.is_file():
        noise += read(synthetic_noise)
    rirs = read(FILES_DIR / "rir-synthetic-v1-pilot.jsonl")
    return speech, noise, rirs
