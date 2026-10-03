import pytest

from cctrain.paths import Paths


@pytest.fixture(scope="session")
def paths() -> Paths:
    return Paths.from_env()


@pytest.fixture(scope="session")
def m0_runs(paths: Paths):
    out = paths.runs / "m0"
    out.mkdir(parents=True, exist_ok=True)
    return out


# --- Tarefa 10: encoder portado do SpeechBrain ECAPA (pesos reais do checkpoint pinado) ---


@pytest.fixture(scope="session")
def sb_ckpt_dir(paths: Paths):
    d = paths.data / "models" / "spkrec-ecapa-voxceleb"
    if not (d / "embedding_model.ckpt").is_file():
        pytest.skip(f"checkpoint SpeechBrain ausente em {d} (rode fetch_checkpoint)")
    return d


@pytest.fixture(scope="session")
def sb_enrollment_model(sb_ckpt_dir):
    from cctrain.export.enrollment import random_generator
    from cctrain.models.sb_port import build_enrollment, load_speechbrain_state, port_ecapa

    encoder = port_ecapa(load_speechbrain_state(sb_ckpt_dir))
    return build_enrollment(encoder, random_generator(seed=0, out_std=0.1)).eval()


@pytest.fixture(scope="session")
def sb_enrollment_onnx(sb_enrollment_model, tmp_path_factory):
    from cctrain.export.enrollment import export_enrollment_onnx

    return export_enrollment_onnx(sb_enrollment_model, tmp_path_factory.mktemp("sb-onnx") / "enrollment-sb.onnx")


@pytest.fixture(scope="session")
def m1_runs(paths: Paths):
    out = paths.runs / "m1"
    out.mkdir(parents=True, exist_ok=True)
    return out
