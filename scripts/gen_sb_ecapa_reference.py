"""Gera tests/golden/sb_ecapa_ref.npz (+ .meta.json): saídas do SpeechBrain ECAPA (`EncoderClassifier.encode_batch`).

O SpeechBrain NÃO é dependência do projeto. Este script roda só com um diretório efêmero em build/:

    uv pip install --python "$CCTRAIN_PY" --target build/sb-ref --no-deps \
        speechbrain==1.1.1 hyperpyyaml ruamel.yaml joblib sentencepiece huggingface_hub tqdm
    PYTHONPATH=build/sb-ref python scripts/gen_sb_ecapa_reference.py [CKPT_DIR]

CKPT_DIR (padrão $CCTRAIN_DATA_DIR/models/spkrec-ecapa-voxceleb) precisa ter os arquivos do manifest, com SHA-256 conferido.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from sb_cases import CASES, REAL_FILES, build_inputs  # noqa: E402

from cctrain.hashing import sha256_file  # noqa: E402
from cctrain.paths import Paths  # noqa: E402

GOLDEN = ROOT / "tests" / "golden"
LOGMEL_CASES = ("s2_rms-20", "s6_rms-20", "s6_rms-40", "s6_peak0.99", "real_cat")


def main(ckpt: Path) -> None:
    import speechbrain
    from speechbrain.inference.speaker import EncoderClassifier

    torch.set_num_threads(4)
    clf = EncoderClassifier.from_hparams(
        source=str(ckpt), savedir=str(Path("build") / "sb-ref-savedir"), run_opts={"device": "cpu"},
        overrides={"pretrained_path": str(ckpt)},  # os hyperparams apontam para o HF; força o diretório local pinado
    )
    clf.eval()
    real = {name: sf.read(ckpt / name, dtype="float32")[0] for name in REAL_FILES}
    out: dict[str, np.ndarray] = {}
    for name, x in build_inputs(real).items():
        wav = torch.from_numpy(x)[None]
        with torch.no_grad():
            emb = clf.encode_batch(wav, normalize=False)[0, 0]
            out[f"emb/{name}"] = emb.numpy().astype(np.float32)
            if name in LOGMEL_CASES:
                feats = clf.mods.mean_var_norm(clf.mods.compute_features(wav), torch.ones(1))
                out[f"logmel/{name}"] = feats[0].numpy().astype(np.float32)  # [T, 80]
    GOLDEN.mkdir(exist_ok=True)
    path = GOLDEN / "sb_ecapa_ref.npz"
    np.savez_compressed(path, **out)
    meta = {
        "generator": "scripts/gen_sb_ecapa_reference.py",
        "speechbrain": speechbrain.__version__,
        "torch": torch.__version__,
        "checkpoint_sha256": sha256_file(ckpt / "embedding_model.ckpt"),
        "cases": list(CASES),
        "logmel_cases": list(LOGMEL_CASES),
        "npz_sha256": sha256_file(path),
        "note": "emb = encode_batch(normalize=False) [192]; logmel = Fbank + InputNormalization(sentence) [T, 80]",
    }
    (GOLDEN / "sb_ecapa_ref.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Paths.from_env().data / "models" / "spkrec-ecapa-voxceleb")
