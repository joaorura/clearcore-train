#!/usr/bin/env python3
"""`eval_quality.py` completo com o checkpoint do pro-v42 no lugar do keep: modelos v2 (gating atual), v31 e v42
(gating novo). O checkpoint vem de EVAL_V42_CKPT (env, para valer também nos processos `spawn`). Uso igual ao
eval_quality: `EVAL_V42_CKPT=... eval_quality_v42.py run --out DIR [--set noise]` e `... report --out DIR`."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import eval_quality as eq  # noqa: E402

_ck = os.environ.get("EVAL_V42_CKPT")
if not _ck:
    raise SystemExit("defina EVAL_V42_CKPT")
eq.MODELS.pop("keep", None)
eq.MODELS["v42"] = (Path(_ck), "v4", "new")
eq.PROF = ("v2_current", "v31_new", "v42_new")
eq.NOPROF = ("v2_noprof", "v31_noprof", "v42_noprof", "dfn3_noprof")
eq.MODEL_OF = {"v2_current": "v2", "v31_new": "v31", "v42_new": "v42"}

if __name__ == "__main__":
    eq.main()
