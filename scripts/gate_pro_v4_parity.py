#!/usr/bin/env python3
"""Gate de paridade do pDFNet3 Pro v4-presence (adaptado de gate_pro_v3_parity.py): forward PyTorch x grafos
stateful hop a hop, agora com a saída `p_target` do enc e o áudio pelo gating de presença do manifest.

Runtimes: ONNX Runtime (sempre), OpenVINO CPU (via --ov-python, um Python com `openvino`) e tract 0.19.16 (via
tract-check `stream`, se --tract-bin existir). Perfis: sem perfil (present=0), perfil do locutor-alvo e de outro
locutor (embedding do enrollment.onnx sobre fala real VCTK). Métrica: SNR saída-vs-referência e máx|erro| de m, coefs
e lsnr; meta >= 60 dB em fp32. Modo `audio`: caminho do runtime com gating por lsnr (réplica de DfTract::process)
com o FrameModel ORT stateful contra o FrameModel PyTorch, SNR do áudio.

    python scripts/gate_pro_v3_parity.py --asset-dir runs/.../members --enrollment enrollment.onnx --out report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from cctrain.data.audio import load_audio, resample
from cctrain.eval.runtime_path import (
    DF_ORDER,
    NB_DF,
    NB_ERB,
    EncoderStep,
    parity_mixture,
    replica_features,
    run_replica,
)
from cctrain.export.pdfnet3_pro_v4 import EMB_DIM, load_pro_v4, presence_state_machine, pro_reference
from cctrain.paths import Paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pro_v3_stream_runner import run as stream_run  # noqa: E402

TARGET_DB = 100.0  # meta do v4-presence (fp32); o v3 exigia 60 dB
KEYS = ("m", "coefs", "lsnr", "p_target")


def snr_db(ref: np.ndarray, x: np.ndarray) -> float:
    ref, x = np.asarray(ref, np.float64), np.asarray(x, np.float64)
    err = ((ref - x) ** 2).sum()
    return float("inf") if err == 0 else float(10 * np.log10((ref ** 2).sum() / err))


def compare(ref: dict, got: dict) -> dict:
    out = {}
    for k in KEYS:
        r, g = np.asarray(ref[k]), np.asarray(got[k]).reshape(np.shape(ref[k]))
        out[k] = {"snr_db": round(snr_db(r, g), 2), "max_abs_err": float(np.abs(r - g).max())}
    out["pass"] = all(v["snr_db"] >= TARGET_DB for v in out.values() if isinstance(v, dict))
    return out


def enroll_embedding(enrollment: Path, names: list[str], data: Path) -> np.ndarray:
    import onnxruntime as ort

    x = []
    for n in names:
        a, sr = load_audio(data / f"raw/vctk-0.92/test/{n}_mic1.flac")
        x.append(resample(a, sr, 16_000))
    audio = np.concatenate(x).astype(np.float32)[None]
    s = ort.InferenceSession(str(enrollment), providers=["CPUExecutionProvider"])
    out = dict(zip([o.name for o in s.get_outputs()], s.run(None, {"audio": audio})))
    return out["embedding"].reshape(EMB_DIM).astype(np.float32)


def clips(data: Path) -> dict[str, np.ndarray]:
    mix = parity_mixture(data)  # VCTK p225 + ruído de rua, SNR 40/25/10/5/-5 dB, lacunas de silêncio
    sp, sr = load_audio(data / "raw/vctk-0.92/test/p226_008_mic1.flac")
    sp = resample(sp, sr, 48_000).astype(np.float64)
    nz, sr = load_audio(data / "raw/noise-wikimedia-cc0/street_traffic_rain_cc0_10s.wav")
    nz = np.tile(resample(nz, sr, 48_000), 3)[: sp.size].astype(np.float64)
    nz *= np.sqrt(np.mean(sp ** 2) / (np.mean(nz ** 2) * 10 ** (5 / 10)))
    other = sp + nz
    other *= 0.9 / np.abs(other).max()
    return {"vctk_p225_street_mix_1000f": mix, "vctk_p226_street_5dB": other.astype(np.float32)}


class PrecomputedFrameModel:
    """FrameModel que serve m/coefs/lsnr já calculados com os TRÊS grafos em todo quadro (estados sempre avançam,
    fiel ao forward de treino). `run_replica` aplica só a seleção da saída por estágio de lsnr (ZERO / NONE /
    ERB_ONLY / ERB_DF, limiares -10/30/20), a mesma de `emulate_gating_offline` usada nas perdas do treino."""

    def __init__(self, out: dict, gate: dict | None = None, present: float = 0.0):
        self.m = np.asarray(out["m"], np.float32).reshape(-1, NB_ERB)
        c = np.asarray(out["coefs"], np.float32).reshape(-1, NB_DF, DF_ORDER, 2)
        self.c = (c[..., 0] + 1j * c[..., 1]).astype(np.complex64)
        self.lsnr = np.asarray(out["lsnr"], np.float32).reshape(-1)
        if gate is not None and present > 0:
            # gating de presença (manifest): passa -> lsnr 0 (estágio ERB+DF), ZERO -> lsnr -100 (estágio ZERO)
            on = presence_state_machine(np.asarray(out["p_target"]).reshape(-1), gate["tau_on"], gate["tau_off"],
                                        gate["hangover_frames"])
            self.lsnr = np.where(on, 0.0, -100.0).astype(np.float32)
        self.t = 0

    def reset(self):
        self.t = 0

    def encode(self, feat_erb, feat_spec):
        t = self.t
        self.t += 1
        return EncoderStep(np.float32(self.lsnr[t]), {"t": t})

    def erb_decode(self, step):
        return self.m[step.tensors["t"]]

    def df_decode(self, step):
        return self.c[step.tensors["t"]]


def audio_compare(audio: np.ndarray, refd: dict, got: dict, gate: dict, present: float) -> dict:
    py = run_replica(audio, PrecomputedFrameModel(refd, gate, present))
    rt = run_replica(audio, PrecomputedFrameModel(got, gate, present))
    return {"snr_db": round(snr_db(py.audio, rt.audio), 2), "max_abs_err": float(np.abs(py.audio - rt.audio).max()),
            "stage_mismatches": int((py.stages != rt.stages).sum())}


def main() -> None:
    paths = Paths.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--enrollment", type=Path, default=paths.runs / "m3_deploy_pro_v3_20261013/enrollment.onnx")
    ap.add_argument("--ov-python", type=Path, default=Path.home() / ".local/share/whisper-ov/.venv/bin/python")
    ap.add_argument("--tract-bin", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-audio", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(1)  # GRU do PyTorch degenera com threads sob carga
    pro = load_pro_v4(a.ckpt)
    gate = json.loads((a.asset_dir / "manifest.json").read_text())["gating"]
    profiles = {
        "none": (np.zeros(EMB_DIM, np.float32), 0.0),
        "target_p225": (enroll_embedding(a.enrollment, ["p225_003", "p225_008", "p225_011"], paths.data), 1.0),
        "other_p226": (enroll_embedding(a.enrollment, ["p226_008"], paths.data), 1.0),
    }
    report: dict = {"asset_dir": str(a.asset_dir), "ckpt": str(a.ckpt), "target_snr_db": TARGET_DB, "gating": gate,
                    "cases": {}}
    runner = Path(__file__).resolve().parent / "pro_v3_stream_runner.py"
    tmp = Path(tempfile.mkdtemp(prefix="gate_pro_v4p_"))
    for cname, audio in clips(paths.data).items():
        _, ferb, fspec = replica_features(audio)
        for pname, (emb, present) in profiles.items():
            key = f"{cname}/{pname}"
            ref = pro_reference(pro, ferb, fspec, emb, present)
            refd = {"m": ref.m, "coefs": ref.coefs, "lsnr": ref.lsnr, "p_target": ref.p_target}
            inp = {"feat_erb": ferb.astype(np.float32),
                   "feat_spec": np.stack([fspec.real, fspec.imag], 1).astype(np.float32),
                   "speaker_embedding": emb, "speaker_present": np.full(1, present, np.float32)}
            case = {"frames": int(ferb.shape[0])}
            outs = {"ort": stream_run(a.asset_dir, "ort", inp)}
            npz = tmp / f"{cname}_{pname}.npz"
            np.savez(npz, **inp)
            if a.ov_python and a.ov_python.exists():
                ov_out = tmp / f"{cname}_{pname}.ov.npz"
                p = subprocess.run([str(a.ov_python), "-I", str(runner), "--backend", "openvino", "--asset-dir",
                                    str(a.asset_dir), "--inputs", str(npz), "--out", str(ov_out)],
                                   capture_output=True, text=True)
                if p.returncode == 0:
                    outs["openvino_cpu"] = dict(np.load(ov_out))
                else:
                    case["openvino_cpu"] = {"error": (p.stderr or p.stdout)[-2000:]}
            if a.tract_bin and a.tract_bin.exists():
                got = run_tract(a.tract_bin, a.asset_dir, inp, tmp / f"{cname}_{pname}")
                if "error" in got:
                    case["tract"] = got
                else:
                    outs["tract_0.19.16"] = got
            for rname, got in outs.items():
                case[rname] = compare(refd, got)
                if not a.skip_audio:
                    case[rname]["audio_all_decoders_stage_select"] = audio_compare(audio, refd, got, gate, present)
                    case[rname]["pass"] = case[rname]["pass"] and (
                        case[rname]["audio_all_decoders_stage_select"]["snr_db"] >= TARGET_DB)
            report["cases"][key] = case
            print(key, json.dumps(case), flush=True)
    report["pass"] = all(v.get("pass", False) for c in report["cases"].values() for v in c.values()
                         if isinstance(v, dict))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if report["pass"] else "FAIL")


def run_tract(tract_bin: Path, asset_dir: Path, inp: dict, prefix: Path) -> dict:
    """tract-check stream ASSET_DIR PREFIX: lê PREFIX.{feat_erb,feat_spec,speaker_embedding,speaker_present}.f32,
    grava PREFIX.{m,coefs,lsnr,p_target}.out.f32."""
    for k, v in inp.items():
        np.asarray(v, "<f4").tofile(f"{prefix}.{k}.f32")
    p = subprocess.run([str(tract_bin), "stream", str(asset_dir), str(prefix)], capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": (p.stderr or p.stdout)[-2000:]}
    return {k: np.fromfile(f"{prefix}.{k}.out.f32", "<f4") for k in KEYS}


if __name__ == "__main__":
    main()
