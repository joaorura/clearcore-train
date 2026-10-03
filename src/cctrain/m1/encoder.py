"""Encoder SpeechBrain portado como função áudio -> embedding, para EER, banco e dev do M1 (Tarefa 12).

Mesmos módulos que vão no `enrollment.onnx` (`SpeechBrainFbank` + `ECAPATDNN` portado); o gerador FiLM não entra.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from cctrain.data.audio import load_audio
from cctrain.data.resample import to_16k
from cctrain.hashing import sha256_file
from cctrain.models.fbank import SpeechBrainFbank
from cctrain.models.sb_port import CKPT_NAME, load_speechbrain_state, port_ecapa


class Encoder:
    def __init__(self, fbank: torch.nn.Module, ecapa: torch.nn.Module, device: str, sha256: str):
        self.fbank, self.ecapa, self.device, self.sha256 = fbank.to(device).eval(), ecapa.to(device).eval(), device, sha256

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        """[192] L2-normalizado (a mesma normalização do `EnrollmentModel`)."""
        x = torch.from_numpy(np.asarray(audio16k, np.float32))[None].to(self.device)
        with torch.no_grad():
            e = self.ecapa(self.fbank(x))
            e = e / torch.sqrt((e * e).sum(dim=-1, keepdim=True) + 1e-12)
        return e.reshape(-1).float().cpu().numpy()


def load_encoder(ckpt_dir: Path, device: str = "cpu") -> Encoder:
    state = load_speechbrain_state(ckpt_dir)  # confere o SHA-256 fixado antes de desserializar
    return Encoder(SpeechBrainFbank(), port_ecapa(state), device, sha256_file(Path(ckpt_dir) / CKPT_NAME))


def clip_embedder(enc: Encoder, data_root: Path) -> Callable[[str], np.ndarray]:
    def embed(rel: str) -> np.ndarray:
        x, sr = load_audio(Path(data_root) / rel)
        return enc.embed(to_16k(x, sr))

    return embed


def clips_embedder(enc: Encoder, data_root: Path, *, io_threads: int = 6) -> Callable[[Sequence[str]], Mapping[str, np.ndarray]]:
    """Versão em lote: leitura e reamostragem em threads (libsndfile e numpy soltam o GIL), encoder na thread principal."""

    def load(rel: str) -> tuple[str, np.ndarray]:
        x, sr = load_audio(Path(data_root) / rel)
        return rel, to_16k(x, sr)

    def many(rels: Sequence[str]) -> dict[str, np.ndarray]:
        out = {}
        with ThreadPoolExecutor(io_threads) as ex:
            for i in range(0, len(rels), 64):  # em blocos: a RAM da máquina é curta
                for rel, x in ex.map(load, rels[i : i + 64]):
                    out[rel] = enc.embed(x)
        return out

    return many
