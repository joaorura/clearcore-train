"""M2 Tarefa 1. Uso: m2_t1.py {sex,ab1}. Grava em $CCTRAIN_RUNS_DIR/m2/.

sex   sexo por falante só de metadado declarado: `manifests/speakers/sex-v1.jsonl` (corpora públicos, no git) e
      `$CCTRAIN_DATA_DIR/manifests/speakers/sex-cv27-pt-v1.jsonl` (client_id do Common Voice: fora do git); cobertura por corpus/split.
ab1   controle da AB1 do M1: o enrollment limpo também passa pelo DFNet3 (CPU, falantes do dev com sexo declarado).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cctrain.hashing import sha256_file  # noqa: E402
from cctrain.paths import Paths, repo_root  # noqa: E402

REPO = repo_root()
PATHS = Paths.from_env()
OUT = PATHS.runs / "m2"
SEX_PUBLIC = REPO / "manifests" / "speakers" / "sex-v1.jsonl"
SEX_CV_REL = Path("manifests") / "speakers" / "sex-cv27-pt-v1.jsonl"
LS_SOURCES = ("librispeech-dev-clean", "librispeech-test-clean", "librispeech-train-clean-100")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(OUT / name)
    log(f"gravado {name}")


def cmd_sex(a) -> None:
    from cctrain.data.splits import speaker_sex
    from cctrain.m1 import pools
    from cctrain.m2.sex import (
        TTS_PT_DECLARED,
        coverage_by_corpus,
        cv_declared_sex,
        librispeech_declared_sex,
        load_speaker_sex,
        vctk_declared_sex,
        write_sex_rows,
    )

    raw = PATHS.data / "raw"
    public: dict[str, str] = {}
    for sid in LS_SOURCES:
        public.update(librispeech_declared_sex((raw / sid / "SPEAKERS.TXT").read_text(errors="replace"), sid))
    public.update(vctk_declared_sex((raw / "vctk-0.92" / "speaker-info.txt").read_text(errors="replace")))
    public.update(TTS_PT_DECLARED)
    digest = write_sex_rows(public, SEX_PUBLIC, source="declared:SPEAKERS.TXT+speaker-info.txt+tts-pt-docs")
    SEX_PUBLIC.with_suffix(".sha256").write_text(f"{digest}  {SEX_PUBLIC.name}\n")
    for k, v in public.items():  # confere contra o leitor já usado no M1 (splits.speaker_sex)
        if k.startswith(("librispeech", "vctk")) and speaker_sex(k, PATHS.data) != v:
            raise SystemExit(f"divergência com splits.speaker_sex em {k}")
    log(f"sex-v1.jsonl: {len(public)} falantes públicos, sha256 {digest[:12]}")

    cv_stats: dict = {"status": "sem --cv-archive: CV fora do mapa"}
    cv_sex: dict[str, str] = {}
    cv_sha = None
    if a.cv_archive:
        import tarfile

        from cctrain.data.pilot import CV_ARCHIVE_SHA256, CV_VALIDATED_SUFFIX

        archive = Path(a.cv_archive)
        got = sha256_file(archive)
        if got != CV_ARCHIVE_SHA256:
            raise SystemExit(f"{archive.name}: SHA-256 {got} != {CV_ARCHIVE_SHA256} (lock do CV)")
        log(f"tar do CV conferido ({got[:12]}); lendo validated.tsv em stream")
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(CV_VALIDATED_SUFFIX):
                    fh = tar.extractfile(member)
                    text = fh.read().decode("utf-8").splitlines()  # type: ignore[union-attr]
                    tsv_member = member.name
                    break
            else:
                raise SystemExit("validated.tsv ausente no tar")
        cv_sex, cv_stats = cv_declared_sex(csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE))
        cv_stats["archive_sha256"] = got
        cv_stats["archive_member"] = tsv_member
        cv_stats["validated_tsv_sha256"] = hashlib.sha256("\n".join(text).encode()).hexdigest()
        cv_path = PATHS.data / SEX_CV_REL
        cv_sha = write_sex_rows(cv_sex, cv_path, source="declared:cv27-pt validated.tsv gender")
        lock = REPO / "manifests" / "speakers" / "sex-cv27-pt-v1.sha256"  # só o hash: o arquivo tem client_id
        lock.write_text(f"{cv_sha}  data/{SEX_CV_REL.as_posix()}\n")
        log(f"sex-cv27-pt-v1.jsonl (fora do git): {len(cv_sex)} client_id, sha256 {cv_sha[:12]}")
    elif (PATHS.data / SEX_CV_REL).is_file():
        cv_sha = sha256_file(PATHS.data / SEX_CV_REL)

    sex = load_speaker_sex(SEX_PUBLIC, *([PATHS.data / SEX_CV_REL] if (PATHS.data / SEX_CV_REL).is_file() else []))
    speech = pools.speech_records(PATHS)
    by_split: dict[str, set[str]] = {}
    for r in speech:
        if r.speaker_key and r.split in ("dev", "train", "test"):
            by_split.setdefault(r.split, set()).add(r.speaker_key)
    cov = coverage_by_corpus({k: sorted(v) for k, v in by_split.items()}, sex)
    inferred = 0  # nenhum caminho de código infere sexo; todo rótulo vem de um arquivo declarado acima
    save("t1_sex.json", {"public_rows": len(public), "public_sha256": digest, "cv_rows": len(cv_sex),
                         "cv_sha256": cv_sha, "cv_stats": cv_stats, "inferred_labels": inferred, "coverage": cov})
    for corpus, per in sorted(cov.items()):
        for split, row in sorted(per.items()):
            log(f"{corpus:28s} {split:5s} {row}")


def cmd_ab1(a) -> None:
    import m1_t12 as m1

    from cctrain.m1 import pools
    from cctrain.m1.ab1 import ab1_indicator
    from cctrain.m1.encoder import load_encoder
    from cctrain.train.embed_bank import dfnet3_denoiser

    dev, _ = m1.dev_pool()
    utt: dict[str, list[str]] = {}
    for r in dev:
        utt.setdefault(r.speaker_key, []).append(r.path)
    _, noise_dev = pools.noise_pools()
    enc = load_encoder(m1.CKPT_DIR, device="cpu")
    res = ab1_indicator(enc.embed, {k: sorted(v) for k, v in sorted(utt.items())}, PATHS.data,
                        [r.path for r in noise_dev], denoise=dfnet3_denoiser("cpu"), seed=0,
                        progress=lambda i, n: log(f"falante {i + 1}/{n}"))
    res["note"] = ("indicador barato + controle (enrollment limpo também pelo DFNet3 upstream sem perfil, CPU); mesmos "
                   "48 falantes do dev e semente do M1; informativo, a AB1 completa é do M4")
    save("t1_ab1_control.json", res)
    log(f"distorção só do denoise (cos(den(limpo), limpo)): {res['clean_through_denoiser_vs_clean']['cos']}")
    for k, v in res["by_snr_db"].items():
        c = v["control_clean_denoised_ref"]
        log(f"SNR {k} dB: original delta {v['delta_denoised_minus_raw']['mean']:+.3f} {v['delta_denoised_minus_raw']['ci95']}"
            f" | controle cru {c['cos_raw_vs_ref']['mean']:.3f} den {c['cos_denoised_vs_ref']['mean']:.3f} "
            f"delta {c['delta_denoised_minus_raw']['mean']:+.3f} {c['delta_denoised_minus_raw']['ci95']}")


def main() -> int:
    import torch

    torch.set_num_threads(4)  # a máquina é compartilhada com outras tarefas: sem sobreassinatura de threads
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sex")
    p.add_argument("--cv-archive", default=None, help="tar.gz original do Common Voice 27 (SHA-256 conferido contra o lock)")
    sub.add_parser("ab1")
    a = ap.parse_args()
    {"sex": cmd_sex, "ab1": cmd_ab1}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
