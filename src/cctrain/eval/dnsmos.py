"""DNSMOS P.835 (SIG, BAK, OVRL) e a variante personalizada (`-p`, penaliza falante interferente), via onnxruntime.

Reimplementa o janelamento e os polinômios de ajuste do `DNSMOS/dnsmos_local.py` do repositório microsoft/DNS-Challenge
(revisão fixada em `manifests/models/dnsmos.yaml`, que traz o SHA-256 do script, dos `.onnx` e a licença): janelas de
9,01 s com passo de 1 s, áudio mais curto repetido até 9,01 s, polinômio por janela, depois a média. Sem o P.808
(`model_v8.onnx`): não é gate. A entrada é 16 kHz; `dnsmos()` reamostra de outro `sr` pelo caminho fixo `to_16k`.
Métrica não intrusiva: a referência é aceita e ignorada, para casar com a assinatura `f(est, ref, sr, **cfg)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np

from cctrain.data.resample import to_16k
from cctrain.eval.metrics import as_utterances

SAMPLING_RATE = 16_000
INPUT_LENGTH_S = 9.01

# Coeficientes de np.poly1d (maior grau primeiro), copiados de `get_polyfit_val` do dnsmos_local.py.
POLY: dict[str, dict[str, tuple[float, ...]]] = {
    "p835": {
        "ovrl": (-0.06766283, 1.11546468, 0.04602535),
        "sig": (-0.08397278, 1.22083953, 0.0052439),
        "bak": (-0.13166888, 1.60915514, -0.39604546),
    },
    "personalized": {
        "ovrl": (-0.00533021, 0.005101, 1.18058466, -0.11236046),
        "sig": (-0.01019296, 0.02751166, 1.19576786, -0.24348726),
        "bak": (-0.04976499, 0.44276479, -0.1644611, 0.96883132),
    },
}


class _Session(Protocol):
    def run(self, output_names, input_feed): ...


class DnsmosScorer:
    def __init__(self, session: _Session, *, personalized: bool = False):
        self.session = session
        self.personalized = personalized

    @classmethod
    def from_dir(cls, directory: Path, *, personalized: bool = False, threads: int | None = 1) -> DnsmosScorer:
        import onnxruntime as ort

        rel = ("pDNSMOS" if personalized else "DNSMOS") + "/sig_bak_ovr.onnx"
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        return cls(ort.InferenceSession(str(Path(directory) / rel), opts, providers=["CPUExecutionProvider"]),
                   personalized=personalized)

    def score(self, audio16k: np.ndarray) -> dict[str, float]:
        audio = np.asarray(audio16k, np.float64)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("DNSMOS: áudio mono 1-D não vazio")
        fs = SAMPLING_RATE
        len_samples = int(INPUT_LENGTH_S * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH_S) + 1
        raws = []
        for idx in range(num_hops):
            seg = audio[int(idx * fs): int((idx + INPUT_LENGTH_S) * fs)]
            if len(seg) < len_samples:
                continue
            out = np.asarray(self.session.run(None, {"input_1": seg.astype("float32")[None]})[0])
            raws.append(out.reshape(-1)[:3])  # (SIG, BAK, OVRL) brutos
        raw = np.stack(raws)  # [janelas, 3]
        poly = POLY["personalized" if self.personalized else "p835"]
        # como o script: o polinômio vale por janela e só depois vem a média (o polinômio não é linear)
        return {k: float(np.mean(np.polyval(poly[k], raw[:, i]))) for i, k in enumerate(("sig", "bak", "ovrl"))}


def dnsmos(est, ref, sr: int, *, scorer: DnsmosScorer) -> Mapping[str, np.ndarray]:
    """SIG/BAK/OVRL por enunciado (arrays). `ref` é ignorado (métrica sem referência)."""
    del ref
    rows = []
    for u in as_utterances(est):
        x = np.asarray(u, np.float32) if sr == SAMPLING_RATE else to_16k(np.asarray(u, np.float32), sr)
        rows.append(scorer.score(x))
    return {k: np.array([r[k] for r in rows], np.float64) for k in ("sig", "bak", "ovrl")}
