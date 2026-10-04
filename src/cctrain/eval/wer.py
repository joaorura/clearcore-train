"""WER com Whisper small em ONNX (onnxruntime, CPU; M2 Tarefa 3). Sem dependência nova: mel em torch/torchaudio,
sessões `encoder_model.onnx` + `decoder_model_merged.onnx` (SHA-256 em `manifests/models/whisper.yaml`) e decodificação
gulosa própria com cache de chave/valor. Falas acima de 30 s são cortadas em janelas de 30 s e os textos concatenados.

Normalização do texto (hipótese e referência): NFKC, minúsculas, colchetes/parênteses removidos, símbolos e pontuação
viram espaço, sem remover acentos (como o `BasicTextNormalizer` do Whisper). Limitação registrada: numerais saem em
dígitos no Whisper e por extenso nas referências; o erro é o mesmo em todos os braços, então o uso é a razão entre
sistemas (G2: WER <= 0,85 x DFNet3), não o valor absoluto.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

WINDOW_S = 30
SR = 16_000
N_SAMPLES = WINDOW_S * SR
N_FFT, HOP, N_MELS = 400, 160, 80
MIN_SAMPLES = SR // 10  # 0,1 s
EOT, SOT, TRANSCRIBE, NO_TIMESTAMPS = 50257, 50258, 50359, 50363
LANG_ID = {"en": 50259, "pt": 50267}
_SOURCE_LANG = {"librispeech": "en", "vctk": "en", "cv27-pt": "pt", "cml-tts-pt": "pt", "tts-portuguese-corpus": "pt"}


def language_for(speaker_key_or_source: str) -> str:
    src = speaker_key_or_source.split(":", 1)[0]
    for prefix, lang in _SOURCE_LANG.items():
        if src.startswith(prefix):
            return lang
    raise ValueError(f"fonte sem idioma de referência: {src!r}")


# ------------------------------------------------------------------------------------------------ texto e WER
_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]|\([^)]*\)")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _BRACKETS.sub(" ", s.lower())
    s = "".join(" " if unicodedata.category(c)[0] in "SP" else c for c in s)  # símbolos e pontuação; acentos ficam
    return " ".join(s.split())


@dataclass(frozen=True)
class WordErrors:
    sub: int
    dele: int
    ins: int
    n_ref: int

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins


def word_errors(hyp: str, ref: str) -> WordErrors:
    """Substituições, remoções e inserções (distância de edição de palavras) depois de normalizar."""
    h, r = normalize_text(hyp).split(), normalize_text(ref).split()
    n, m = len(r), len(h)
    cost = np.zeros((n + 1, m + 1), np.int64)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j - 1] + (r[i - 1] != h[j - 1]), cost[i - 1, j] + 1, cost[i, j - 1] + 1)
    i, j, sub, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i, j] == cost[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            sub += int(r[i - 1] != h[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return WordErrors(sub, dele, ins, n)


def wer_per_utterance(hyps: Sequence[str], refs: Sequence[str]) -> np.ndarray:
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hipóteses para {len(refs)} referências")
    out = []
    for h, r in zip(hyps, refs, strict=True):
        c = word_errors(h, r)
        out.append(c.errors / c.n_ref if c.n_ref else np.nan)
    return np.array(out, np.float64)


def corpus_wer(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """WER do conjunto: soma de erros / soma de palavras de referência (média ponderada pelo tamanho)."""
    cs = [word_errors(h, r) for h, r in zip(hyps, refs, strict=True)]
    n = sum(c.n_ref for c in cs)
    return sum(c.errors for c in cs) / n if n else float("nan")


# ------------------------------------------------------------------------------------------------ áudio -> mel
def chunk_audio(x16k: np.ndarray) -> list[np.ndarray]:
    x = np.asarray(x16k, np.float32)
    chunks = [x[i:i + N_SAMPLES] for i in range(0, len(x), N_SAMPLES)]
    return [c for c in chunks if len(c) >= MIN_SAMPLES]


_FILTERS: torch.Tensor | None = None


def log_mel(x16k: np.ndarray) -> np.ndarray:
    """Log-mel do Whisper: [80, 3000] (janela de 30 s, zeros à direita), STFT 400/160 Hann, filtros Slaney até 8 kHz,
    log10 com piso (max - 8) e (x + 4) / 4."""
    global _FILTERS
    if _FILTERS is None:
        _FILTERS = torchaudio.functional.melscale_fbanks(N_FFT // 2 + 1, 0.0, 8000.0, N_MELS, SR, norm="slaney",
                                                         mel_scale="slaney")
    x = torch.zeros(N_SAMPLES)
    seg = torch.from_numpy(np.asarray(x16k[:N_SAMPLES], np.float32))
    x[: seg.numel()] = seg
    stft = torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True)
    mel = _FILTERS.T @ (stft[..., :-1].abs() ** 2)
    log = torch.clamp(mel, min=1e-10).log10()
    log = (torch.maximum(log, log.max() - 8.0) + 4.0) / 4.0
    return log.numpy().astype(np.float32)


# ------------------------------------------------------------------------------------------------ decodificação
def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class WhisperOnnx:
    def __init__(self, encoder, decoder, vocab: dict[str, int], suppress: Sequence[int],
                 begin_suppress: Sequence[int], *, n_layers: int = 12, n_heads: int = 12, head_dim: int = 64,
                 max_new_tokens: int = 224):
        self.enc, self.dec = encoder, decoder
        self.id2tok = {i: t for t, i in vocab.items()}
        self.suppress, self.begin_suppress = np.array(sorted(set(suppress)), np.int64), list(begin_suppress)
        self.n_layers, self.n_heads, self.head_dim, self.max_new = n_layers, n_heads, head_dim, max_new_tokens
        self._bd = _byte_decoder()

    @classmethod
    def from_dir(cls, directory: Path, *, threads: int | None = None, max_new_tokens: int = 224) -> WhisperOnnx:
        import onnxruntime as ort

        d = Path(directory)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        mk = lambda f: ort.InferenceSession(str(d / "onnx" / f), opts, providers=["CPUExecutionProvider"])  # noqa: E731
        gen = json.loads((d / "generation_config.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        added = json.loads((d / "added_tokens.json").read_text())
        vocab = json.loads((d / "vocab.json").read_text())
        vocab.update(added)
        return cls(mk("encoder_model.onnx"), mk("decoder_model_merged.onnx"), vocab, gen["suppress_tokens"],
                   gen["begin_suppress_tokens"], n_layers=cfg["decoder_layers"], n_heads=cfg["decoder_attention_heads"],
                   head_dim=cfg["d_model"] // cfg["decoder_attention_heads"], max_new_tokens=max_new_tokens)

    def _detokenize(self, ids: Sequence[int]) -> str:
        text = "".join(self.id2tok[i] for i in ids if i < EOT and i in self.id2tok)
        return bytearray(self._bd[c] for c in text if c in self._bd).decode("utf-8", errors="replace").strip()

    def _transcribe_window(self, x16k: np.ndarray, language: str) -> str:
        feats = log_mel(x16k)[None]
        hidden = self.enc.run(None, {"input_features": feats})[0]
        names = [i.name for i in self.dec.get_inputs() if i.name.startswith("past_key_values.")]
        empty = np.zeros((1, self.n_heads, 0, self.head_dim), np.float32)
        past = {n: empty for n in names}
        tokens = [SOT, LANG_ID[language], TRANSCRIBE, NO_TIMESTAMPS]
        out_ids: list[int] = []
        step_ids, use_cache = np.array([tokens], np.int64), False
        for step in range(self.max_new):
            feed = {"input_ids": step_ids, "encoder_hidden_states": hidden, **past,
                    "use_cache_branch": np.array([use_cache])}
            res = self.dec.run(None, feed)
            logits = res[0][0, -1].astype(np.float64)
            logits[self.suppress] = -np.inf
            logits[NO_TIMESTAMPS:] = -np.inf  # nada de marcas de tempo (nem o próprio token) na saída
            if step == 0:
                logits[self.begin_suppress] = -np.inf
            nxt = int(np.argmax(logits))
            if nxt == EOT:
                break
            out_ids.append(nxt)
            present = {o.name: v for o, v in zip(self.dec.get_outputs()[1:], res[1:], strict=True)}
            for n in names:
                p = n.replace("past_key_values", "present")
                if ".decoder." in n or not use_cache:
                    past[n] = present[p]  # decoder: sempre; encoder: só da 1ª passada (depois vem vazio)
            step_ids, use_cache = np.array([[nxt]], np.int64), True
        return self._detokenize(out_ids)

    def transcribe(self, audio16k: np.ndarray, *, language: str) -> str:
        if language not in LANG_ID:
            raise ValueError(f"idioma {language!r} fora de {sorted(LANG_ID)}")
        chunks = chunk_audio(np.asarray(audio16k, np.float32))
        return " ".join(t for t in (self._transcribe_window(c, language) for c in chunks) if t)


def wer_audio(est, ref_texts: Sequence[str], sr: int, *, asr: WhisperOnnx, language: str) -> np.ndarray:
    """WER por enunciado do áudio `est` (reamostrado a 16 kHz pelo caminho fixo) contra as transcrições `ref_texts`."""
    from cctrain.data.resample import to_16k
    from cctrain.eval.metrics import as_utterances

    hyps = [asr.transcribe(np.asarray(u, np.float32) if sr == SR else to_16k(np.asarray(u, np.float32), sr),
                           language=language) for u in as_utterances(est)]
    return wer_per_utterance(hyps, ref_texts)
