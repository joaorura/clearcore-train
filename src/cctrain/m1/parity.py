"""Paridade do `enrollment.onnx` final (M1 Tarefa 12): tract-check × módulo PyTorch (nível `m1.tract_parity`) e
ORT × módulo (P1: atol 1e-5, rtol 1e-6), em entradas de fala real, níveis extremos e bordas do G11."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cctrain import tractcheck
from cctrain.export.enrollment import OUTPUT_NAMES

P1_ATOL, P1_RTOL = 1e-5, 1e-6


def enrollment_parity(model, onnx_path: Path, cases: Mapping[str, np.ndarray], workdir: Path) -> list[dict]:
    """Uma linha por caso: maior |tract − módulo| nas 5 saídas, P1 (ORT × módulo) e finitude em módulo/ORT/tract."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rows = []
    for name, x in cases.items():
        x = np.asarray(x, np.float32)
        with torch.no_grad():
            ref = [t.numpy() for t in model(torch.from_numpy(x)[None])]
        got_ort = sess.run(None, {"audio": x[None]})
        ort_ok = all(np.allclose(g.reshape(r.shape), r, atol=P1_ATOL, rtol=P1_RTOL) for g, r in zip(got_ort, ref, strict=True))
        inp, out = workdir / f"{name}.f32", workdir / f"{name}.json"
        x.astype("<f4").tofile(inp)
        proc = tractcheck.run("enroll", onnx_path, inp, out)
        rep = json.loads(out.read_text())
        if not rep.get("ok"):
            raise RuntimeError(f"tract-check falhou em {name}: {rep.get('error')} {proc.stderr}")
        worst, finite = 0.0, all(np.isfinite(r).all() for r in ref) and all(np.isfinite(g).all() for g in got_ort)
        for oname, r in zip(OUTPUT_NAMES, ref, strict=True):
            g = np.asarray(rep["outputs"][oname], np.float32)
            finite = finite and bool(np.isfinite(g).all()) and g.shape == r.shape
            worst = max(worst, float(np.abs(g - r).max())) if g.shape == r.shape else float("inf")
        rows.append({"case": name, "seconds": len(x) / 16_000, "tract_max_abs_diff": worst, "ort_ok": bool(ort_ok),
                     "finite": bool(finite)})
    return rows


def summarize_parity(rows: Sequence[Mapping], *, durations_s: Sequence[int], required: Sequence[str]) -> dict:
    """Resumo para o veredito: durações exigidas cobertas (`required` são os casos que as cobrem) e o pior desvio
    entre elas; `finite` vale para todos os casos (incluindo as bordas)."""
    by = {r["case"]: r for r in rows}
    present = [c for c in required if c in by]
    covered = sorted({int(round(by[c]["seconds"])) for c in present} & {int(d) for d in durations_s})
    worst = max((by[c]["tract_max_abs_diff"] for c in present), default=float("nan"))
    return {"durations_s": covered, "max_abs_diff": worst, "finite": bool(rows) and all(r["finite"] for r in rows),
            "missing": [c for c in required if c not in by], "p1_ort_ok": all(r["ort_ok"] for r in rows),
            "worst_case_all": max((r["tract_max_abs_diff"] for r in rows), default=float("nan"))}
