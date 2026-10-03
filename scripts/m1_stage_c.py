"""Tarefa 11 do M1. Uso: m1_stage_c.py {smoke,gate}. Grava em $CCTRAIN_RUNS_DIR/m1/.

smoke: 2 min de estágio C no piloto (VCTK `pilot`, ruído sintético) com o encoder PROTÓTIPO (pesos aleatórios do M0),
       sob o gpu_lock; escreve `stage-c-probe.json` (PSS do trainer + workers) e o checkpoint em runs/m1/stage-c-smoke.
gate:  sobre o checkpoint do smoke, mede o desvio offline x runtime em misturas de dev (LibriSpeech dev-clean, mesmo
       sexo, SIR 0 dB), escolhe o caminho pelo gate congelado `runtime-deviation-v1` e roda o gate funcional.
       Os números do smoke NÃO são evidência do M1 (encoder aleatório, 2 min de treino, ruído sintético).
A rodada real (dados da Tarefa 8, splits da 9, banco e encoder portado da 12) está pendente.
"""

import json
import sys

from cctrain.data.manifest import FILES_DIR, read_file_manifest
from cctrain.eval import functional as fn
from cctrain.export.enrollment import build_prototype
from cctrain.export.film import build_film_identity_asset
from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
from cctrain.paths import Paths, clearcore_root
from cctrain.train.checkpoint import list_checkpoints
from cctrain.train.stage_c import StageCConfig, build_pilot_bank, enrollment_embedder, run_stage_c

PATHS = Paths.from_env()
OUT = PATHS.runs / "m1"
RUN_ID = "stage-c-smoke"


def _save(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")
    print(name, json.dumps(obj, default=str)[:600])


def _embedder():
    return enrollment_embedder(build_prototype(0, "expand"), "cpu")


def smoke(minutes: float = 2.0) -> None:
    from cctrain.data.loader import pilot_records

    speech, noise, rirs = pilot_records(PATHS.data)
    v2 = FILES_DIR / "rir-synthetic-v2.jsonl"
    rirs = read_file_manifest(v2) if v2.is_file() else rirs
    speech = [r for r in speech if r.source_id == "vctk-0.92"]  # 8 falantes `pilot`; o CV27 só alongaria o banco
    bank = build_pilot_bank(_embedder(), speech, PATHS.data, variants=4)
    cfg = StageCConfig(run_id=RUN_ID, batch_size=16, segment_s=3.0, steps_per_epoch=50, same_sex=False,
                       num_workers=4, prefetch_factor=2, probe_every=5)
    from cctrain.train.stage_c import StageCData

    ckpt = run_stage_c(cfg, bank, max_minutes=minutes, data=StageCData(speech, noise, rirs, None))
    print("checkpoint", ckpt)


def gate() -> None:
    from cctrain.data.loader import pilot_records

    ckpt = list_checkpoints(OUT / RUN_ID / "checkpoints")[-1].path
    raw = PATHS.data / "raw" / "librispeech-dev-clean"
    dev = [r for r in read_file_manifest(FILES_DIR / "librispeech-dev-clean-pilot.jsonl") if r.split == "dev"]
    sex = fn.read_librispeech_sex(raw / "SPEAKERS.TXT")
    _, noise, _ = pilot_records(PATHS.data)
    mixes = fn.build_dev_mixtures(dev, noise, PATHS.data, sex, _embedder(), n_mixtures=40, seed=0)
    asset = build_film_identity_asset(clearcore_root() / "vendor/approved/df-compatible-release-asset-v1.bin",
                                      OUT / "t11-film-asset").path
    runtime = fn.tract_runtime(asset, OUT / "t11-tract")
    base, _ = load_upstream_dfnet3(device="cpu")
    from cctrain.models.film import FilmGenerator, load_film_config
    from cctrain.models.pdfnet3 import PDFNet3
    from cctrain.train.checkpoint import load_checkpoint

    gen = FilmGenerator(load_film_config())
    gen.load_state_dict(load_checkpoint(ckpt).model)
    model = PDFNet3(base, gen.eval()).eval()
    per = []
    for m in mixes[:8]:
        y, stages = fn.enhance_offline(base, m.noisy, film_model=model, embedding=m.embedding)
        per.append(fn.deviation_metrics(y, stages, runtime(m.noisy, fn.film_vector(gen, m.embedding))))
    dev_deviation = fn.pool_deviation(per)
    path = fn.deviation_gate(dev_deviation)
    res = fn.functional_gate(ckpt, mixes, path=path, runtime=runtime, base=base)
    _save("t11_functional_smoke.json", {
        "WARNING": "smoke: encoder protótipo, 2 min, ruído sintético; não é evidência do gate do M1",
        "checkpoint": ckpt.name, "deviation_dev_checkpoint": dev_deviation, "deviation_path": path,
        "levels_sha256": fn.load_deviation_levels().sha256, "functional": res.__dict__,
        "dev_speakers": len({m.speaker for m in mixes}), "dev_noise": "noise-synthetic-v1 (piloto)"})


if __name__ == "__main__":
    {"smoke": smoke, "gate": gate}[sys.argv[1]]()
