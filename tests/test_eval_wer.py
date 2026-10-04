"""WER com Whisper em ONNX (M2 Tarefa 3): normalização, contagem de erros, mel, decodificação e um enunciado real."""

from __future__ import annotations

import numpy as np
import pytest

from cctrain.eval import wer as W
from cctrain.paths import Paths


def test_normalize_text_basic():
    assert W.normalize_text("Olá, Mundo! É  O  fim.") == "olá mundo é o fim"
    assert W.normalize_text("DON'T stop (maybe) [noise]") == "don t stop"  # colchetes e parênteses saem; apóstrofo vira espaço
    assert W.normalize_text("  ") == ""
    assert W.normalize_text("ﬁnal") == "final"  # NFKC


def test_word_errors_counts_sub_del_ins():
    c = W.word_errors("a x c", "a b c d")
    assert (c.sub, c.dele, c.ins, c.n_ref) == (1, 1, 0, 4)
    c = W.word_errors("a b c d e", "a b c d")
    assert (c.sub, c.dele, c.ins) == (0, 0, 1)
    c = W.word_errors("", "um dois")
    assert (c.sub, c.dele, c.ins) == (0, 2, 0)
    c = W.word_errors("um dois", "")
    assert c.n_ref == 0 and c.ins == 2


def test_wer_per_utterance_and_corpus_micro_average():
    hyps = ["a b", "x y z w", "igual"]
    refs = ["a b", "x y z", "igual"]
    per = W.wer_per_utterance(hyps, refs)
    assert per.tolist() == [0.0, pytest.approx(1 / 3), 0.0]
    assert W.corpus_wer(hyps, refs) == pytest.approx(1 / 6)  # soma de erros / soma de palavras de referência
    assert np.isnan(W.wer_per_utterance(["a"], [""])[0])  # referência vazia: indefinido


def test_chunking_splits_long_audio_at_30_seconds():
    x = np.zeros(16_000 * 65, np.float32)
    chunks = W.chunk_audio(x)
    assert [len(c) for c in chunks] == [480_000, 480_000, 16_000 * 5]
    assert W.chunk_audio(np.zeros(100, np.float32)) == []  # < 0,1 s: ignorado


def test_log_mel_shape_range_and_tone_bin():
    t = np.arange(16_000 * 3) / 16_000
    x = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    m = W.log_mel(x)
    assert m.shape == (80, 3000) and m.dtype == np.float32
    assert m.max() <= 1.5 and m.min() >= m.max() - 2.0 - 1e-6  # piso max - 8 em log10, depois (x+4)/4
    assert 20 < int(m[:, 10].argmax()) < 40  # 1 kHz cai por volta do mel 25-35 (escala Slaney, 80 bandas até 8 kHz)


def test_language_by_source():
    assert W.language_for("librispeech-dev-clean:84") == "en" and W.language_for("vctk-0.92:p225") == "en"
    assert W.language_for("cv27-pt:abc") == "pt" and W.language_for("cml-tts-pt:1") == "pt"
    assert W.language_for("tts-portuguese-corpus:x") == "pt"
    with pytest.raises(ValueError):
        W.language_for("desconhecido:1")


@pytest.fixture(scope="module")
def asr():
    from cctrain.eval.models import ModelMismatch, verify_model

    try:
        d = verify_model("whisper")
    except (ModelMismatch, FileNotFoundError) as exc:
        pytest.skip(f"Whisper ausente ou diferente do manifest: {exc}")
    return W.WhisperOnnx.from_dir(d, threads=4)


@pytest.mark.data
def test_real_whisper_transcribes_a_librispeech_utterance(asr):
    import soundfile as sf

    from cctrain.data.resample import to_16k

    root = Paths.from_env().data / "raw" / "librispeech-dev-clean"
    trans = next(root.rglob("*.trans.txt"))
    uid, ref = trans.read_text().splitlines()[0].split(" ", 1)
    x, sr = sf.read(next(trans.parent.glob(f"{uid}.flac")), dtype="float32")
    hyp = asr.transcribe(to_16k(x, sr) if sr != 16_000 else x, language="en")
    assert W.wer_per_utterance([hyp], [ref])[0] < 0.25, (hyp, ref)
    assert asr.transcribe(np.zeros(16_000 * 3, np.float32), language="en") is not None
