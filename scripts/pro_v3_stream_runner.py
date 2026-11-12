#!/usr/bin/env python3
"""Runner hop a hop (S=1) do asset stateful pdfnet3-pro-v3-full, só com numpy + o runtime (ORT ou OpenVINO).

Não importa cctrain, para rodar também no Python do OpenVINO. Lê o manifest.json do asset, realimenta todos os pares
<nome>_in/<nome>_out por nome e roteia as saídas do enc para os decoders pelo nome (`routing` do manifest).
Roda os três grafos em todo quadro (sem gating): é o modo de paridade numérica.

    python pro_v3_stream_runner.py --backend {ort,openvino} --asset-dir DIR --inputs in.npz --out out.npz

in.npz: feat_erb [T,32], feat_spec [T,2,96] (re/im), speaker_embedding [192], speaker_present [1].
out.npz: m [T,32], coefs [T,96,10], lsnr [T] (+ p_target [T] se o enc tiver essa saída, contrato v4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class OrtGraph:
    def __init__(self, path: Path):
        import onnxruntime as ort

        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        o.log_severity_level = 3
        self.s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
        self.out = [x.name for x in self.s.get_outputs()]

    def __call__(self, feeds: dict) -> dict:
        return dict(zip(self.out, self.s.run(None, feeds)))


class OvGraph:
    def __init__(self, path: Path, device: str = "CPU"):
        import openvino as ov

        core = ov.Core()
        cm = core.compile_model(core.read_model(str(path)), device, {"INFERENCE_PRECISION_HINT": "f32"})
        self.req = cm.create_infer_request()
        self.out = [o.get_any_name() for o in cm.outputs]
        self.cm = cm

    def __call__(self, feeds: dict) -> dict:
        self.req.infer(feeds)
        return {n: self.req.get_tensor(n).data.copy() for n in self.out}


def run(asset_dir: Path, backend: str, inp: dict) -> dict:
    asset_dir = Path(asset_dir)
    man = json.loads((asset_dir / "manifest.json").read_text())
    cls = {"ort": OrtGraph, "openvino": OvGraph}[backend]
    graphs = {g: cls(asset_dir / spec["file"]) for g, spec in man["graphs"].items()}
    state = {g: {p["in"]: np.zeros(p["shape"], np.float32) for p in spec["state_pairs"]}
             for g, spec in man["graphs"].items()}
    pairs = {g: [(p["in"], p["out"]) for p in spec["state_pairs"]] for g, spec in man["graphs"].items()}
    cond = {"speaker_embedding": np.asarray(inp["speaker_embedding"], np.float32).reshape(1, -1),
            "speaker_present": np.asarray(inp["speaker_present"], np.float32).reshape(1)}

    def step(g: str, feeds: dict) -> dict:
        need = man["graphs"][g]["inputs"]
        f = {k: v for k, v in {**feeds, **cond}.items() if k in need}
        out = graphs[g]({**f, **state[g]})
        for i, o in pairs[g]:
            state[g][i] = out.pop(o)
        return out

    ferb, fspec = np.asarray(inp["feat_erb"], np.float32), np.asarray(inp["feat_spec"], np.float32)
    t_total = ferb.shape[0]
    m = np.zeros((t_total, 32), np.float32)
    coefs = np.zeros((t_total, 96, 10), np.float32)
    lsnr = np.zeros(t_total, np.float32)
    has_p = "p_target" in man["graphs"]["enc"]["outputs"]  # contrato pro-v4-presence
    p_target = np.zeros(t_total, np.float32)
    for t in range(t_total):
        e = step("enc", {"feat_erb": ferb[t].reshape(1, 1, 1, 32), "feat_spec": fspec[t].reshape(1, 2, 1, 96)})
        r = man["routing"]
        m[t] = step("erb_dec", {k: e[v.split(".")[1]] for k, v in r["erb_dec"].items()})["m"].reshape(32)
        coefs[t] = step("df_dec", {k: e[v.split(".")[1]] for k, v in r["df_dec"].items() if v != "runtime"}
                        )["coefs"].reshape(96, 10)
        lsnr[t] = e["lsnr"].reshape(-1)[0]
        if has_p:
            p_target[t] = e["p_target"].reshape(-1)[0]
    out = {"m": m, "coefs": coefs, "lsnr": lsnr}
    if has_p:
        out["p_target"] = p_target
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ort", "openvino"], required=True)
    ap.add_argument("--asset-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    inp = dict(np.load(a.inputs))
    np.savez(a.out, **run(a.asset_dir, a.backend, inp))


if __name__ == "__main__":
    main()
