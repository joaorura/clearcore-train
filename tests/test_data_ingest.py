import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cctrain.data import ingest
from cctrain.data.manifest import FileRecord, ManifestError
from cctrain.data.sources import IntegrityError
from cctrain.disk import GB


def _rec(path, speaker, seconds, sr=16000):
    return FileRecord(path=path, sha256="0" * 64, source_id="s", license="CC-BY-4.0", role="train", speaker=speaker,
                      split="train", sr=sr, num_samples=int(seconds * sr), band_hz=8000.0, codec="flac",
                      attribution="a")


def test_librispeech_cap_never_exceeds_240s_per_speaker_and_is_deterministic():
    recs = [_rec(f"{spk}/{i:03d}.flac", spk, 12.5) for spk in ("1", "2") for i in range(40)]
    recs += [_rec("3/000.flac", "3", 100.0)]
    kept = ingest.librispeech_cap(recs)
    per = {}
    for r in kept:
        per[r.speaker] = per.get(r.speaker, 0) + r.num_samples / r.sr
    assert max(per.values()) <= 240.0
    assert per["1"] == 237.5 and per["3"] == 100.0
    assert [r.path for r in kept] == [r.path for r in ingest.librispeech_cap(list(reversed(recs)))]


def test_dns_keeps_only_files_with_determinable_freesound_origin():
    names = ["datasets_fullband/noise_fullband/fan_Freesound_validated_238962_2.wav",
             "datasets_fullband/noise_fullband/audioset_siren_xyz.wav",
             "datasets_fullband/noise_fullband/DEMAND_street.wav",
             "datasets_fullband/noise_fullband/door.wav"]
    assert ingest.dns_freesound_cc0(names) == {names[0]}
    assert ingest.dns_freesound_cc0([{"path": names[0]}, {"path": names[3]}]) == {names[0]}


@pytest.mark.parametrize("lic,labels,ok", [
    ("http://creativecommons.org/publicdomain/zero/1.0/", "Wind,Rain", True),
    ("http://creativecommons.org/licenses/by/3.0/", ["Engine"], True),
    ("http://creativecommons.org/licenses/by-nc/3.0/", "Wind", False),
    ("http://creativecommons.org/licenses/sampling+/1.0/", "Wind", False),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "Wind,Male_speech_and_man_speaking", False),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "Laughter", False),
    (None, "Wind", False),
])
def test_fsd50k_allowed(lic, labels, ok):
    assert ingest.fsd50k_allowed({"license": lic, "labels": labels}) is ok


def test_vctk_selection_is_mic1_of_speakers_only():
    names = ["wav48_silence_trimmed/p225/p225_001_mic1.flac", "wav48_silence_trimmed/p225/p225_001_mic2.flac",
             "wav48_silence_trimmed/log/x_mic1.flac", "txt/p225/p225_001.txt"]
    assert ingest.vctk_mic1_members(names) == {names[0]: "raw/vctk-0.92/full/p225/p225_001_mic1.flac"}


def test_speaker_sex_table():
    txt = ";ID  |SEX| SUBSET\n19  | F | train-clean-100 | 25.19 | Kristin LeMoine\n26  | M | train-clean-100 | 25.1 | x\n"
    assert ingest.librispeech_speaker_sex(txt) == {"19": "F", "26": "M"}


def test_cv_selection_excludes_portugal_and_keeps_client_id_private():
    rows = [{"client_id": "c1", "path": "a.mp3", "variant": "Portuguese (Brasil)"},
            {"client_id": "c2", "path": "b.mp3", "variant": ""},
            {"client_id": "c3", "path": "c.mp3", "variant": "Portuguese (Portugal)"}]
    assert ingest.cv_select_rows(rows) == {"a.mp3": "c1", "b.mp3": "c2"}
    with pytest.raises(ManifestError):
        ingest.cv_select_rows(rows[2:])


def test_cml_select_caps_speaker_and_total():
    sel = ingest.cml_select_factory(hours_cap=1.0, speaker_cap_s=100.0)
    pre = "cml_tts_dataset_portuguese_v0.1"
    sz = 48000 * 40 + 44  # 40 s
    got = [sel(f"{pre}/train/audio/7/1/{i}.wav", sz) for i in range(4)]
    assert got[:2] == ["train/audio/7/1/0.wav", "train/audio/7/1/1.wav"] and got[2:] == [None, None]
    assert sel(f"{pre}/train/audio/8/1/0.wav", sz) == "train/audio/8/1/0.wav"
    assert sel(f"{pre}/train/transcript.txt", 10) == "train/transcript.txt"
    assert sel("other/x.wav", sz) is None


def test_phase_deficit_blocks_with_gb_and_passes_when_free(tmp_path):
    ok = ingest.phase_deficit_gb("1-noise-rir-encoder", [], 69 * GB, 10.0, workdir=tmp_path)
    assert ok == 0.0
    deficit = ingest.phase_deficit_gb("1-noise-rir-encoder", [], 20 * GB, 10.0, workdir=tmp_path)
    assert deficit > 0
    # fase concluída não conta: o resto cabe com o livre atual
    assert ingest.phase_deficit_gb("5-cv27-pt", ["1-noise-rir-encoder", "2-vctk", "3-librispeech-100-test-clean",
                                                 "4-cml-tts-pt"], 40 * GB, 10.0, workdir=tmp_path) == 0.0


class _Handler(BaseHTTPRequestHandler):
    data = b""
    cuts = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        a, b = 0, len(self.data) - 1
        if rng:
            lo, _, hi = rng.removeprefix("bytes=").partition("-")
            a = int(lo)
            b = min(int(hi), len(self.data) - 1) if hi else b
        body = self.data[a:b + 1]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", '"v1"')
        if rng:
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.data)}")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server():
    data = np.random.default_rng(0).bytes(5 * (1 << 20) + 123)
    _Handler.data = data
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/f.bin", data
    srv.shutdown()


def test_download_resumable_completes_and_is_idempotent(server, tmp_path, monkeypatch):
    url, data = server
    monkeypatch.setattr(ingest, "BLOCK", 1 << 20)
    dest = tmp_path / "f.bin"
    d = ingest.download_resumable(url, dest, expected_size=len(data), expected_md5=hashlib.md5(data).hexdigest(),
                                  log=lambda m: None, connections=3)
    assert dest.read_bytes() == data and d.sha256 == hashlib.sha256(data).hexdigest()
    assert not list(tmp_path.glob("*.part*"))
    d2 = ingest.download_resumable(url, dest, expected_size=len(data), log=lambda m: None)
    assert d2.sha256 == d.sha256


def test_download_resumes_from_done_blocks_across_processes(server, tmp_path, monkeypatch):
    url, data = server
    monkeypatch.setattr(ingest, "BLOCK", 1 << 20)
    dest = tmp_path / "f.bin"
    part = dest.with_name("f.bin.part")
    with open(part, "wb") as f:
        f.truncate(len(data))
        f.write(b"")
    with open(part, "r+b") as f:
        f.write(data[: 2 << 20])  # blocos 0 e 1 prontos; os demais zerados
    dest.with_name("f.bin.part.meta").write_text(json.dumps(
        {"validator": '"v1"', "size": len(data), "block": 1 << 20, "done": [0, 1]}))
    served = []
    orig = _Handler.do_GET

    def spy(self):
        served.append(self.headers.get("Range"))
        orig(self)

    monkeypatch.setattr(_Handler, "do_GET", spy)
    ingest.download_resumable(url, dest, expected_size=len(data), log=lambda m: None, connections=2)
    assert dest.read_bytes() == data
    assert "bytes=0-1048575" not in served and "bytes=1048576-2097151" not in served


def test_download_wrong_hash_fails_closed(server, tmp_path, monkeypatch):
    url, data = server
    monkeypatch.setattr(ingest, "BLOCK", 1 << 20)
    with pytest.raises(IntegrityError):
        ingest.download_resumable(url, tmp_path / "f.bin", expected_size=len(data), expected_md5="0" * 32,
                                  log=lambda m: None)
    assert not (tmp_path / "f.bin").exists() and not list(tmp_path.glob("*.part*"))


def test_write_flac_native_rate_roundtrip(tmp_path):
    wav = tmp_path / "a.wav"
    x = (np.sin(np.linspace(0, 200, 24000)) * 8000).astype("int16")
    sf.write(str(wav), x, 24000, subtype="PCM_16")
    dur = ingest._write_flac(str(wav), tmp_path / "a.flac")
    y, sr = sf.read(str(tmp_path / "a.flac"), dtype="int16")
    assert sr == 24000 and dur == 1.0 and np.array_equal(x, y)


def test_empty_wav_is_dropped_without_empty_flac(tmp_path):
    wav = tmp_path / "e.wav"
    sf.write(str(wav), np.zeros(0, dtype="int16"), 24000, subtype="PCM_16")
    assert ingest._wav_to_flac_job((str(wav), str(tmp_path / "e.flac"))) == 0.0
    assert not wav.exists() and not (tmp_path / "e.flac").exists()


def test_fetch_script_blocked_phase_exits_2_with_deficit(tmp_path, monkeypatch, capsys):
    import importlib.util
    import shutil
    from collections import namedtuple

    monkeypatch.setenv("CCTRAIN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CCTRAIN_RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "data").mkdir()
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(100 * GB, 80 * GB, 20 * GB))
    spec = importlib.util.spec_from_file_location("fetch_m1", Path(__file__).resolve().parents[1] / "scripts" / "fetch_m1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--phases", "1"]) == 2
    status = json.loads((tmp_path / "runs" / "m1" / "fetch-status.json").read_text())
    assert status["1-noise-rir-encoder"]["status"] == "blocked_disk" and status["1-noise-rir-encoder"]["deficit_gb"] > 0
    assert "déficit" in capsys.readouterr().out
    assert not list((tmp_path / "data").rglob("*"))  # nada foi baixado
