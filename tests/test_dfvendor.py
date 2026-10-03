import importlib
import json

from cctrain.dfvendor import VENDORED_ROOT, ensure_df_importable, ensure_dfnet3_checkpoint, vendored_info
from cctrain.hashing import sha256_file


def test_vendored_metadata_matches_files():
    info = vendored_info()
    assert info["tag"] == "v0.5.6"
    assert info["commit"].startswith("978576aa")
    meta = info["files"]["models/DeepFilterNet3.zip"]
    assert meta["size"] == 7986207
    assert sha256_file(VENDORED_ROOT / "models/DeepFilterNet3.zip") == meta["sha256"]


def test_df_package_imports():
    df = ensure_df_importable()
    importlib.import_module("df.deepfilternet3")
    importlib.import_module("df.enhance")

    assert df.__file__.startswith(str(VENDORED_ROOT))


def test_patches_are_marked():
    io_py = (VENDORED_ROOT / "DeepFilterNet/df/io.py").read_text()
    assert "CCTRAIN-PATCH(io-audiometadata)" in io_py


def test_checkpoint_extraction_is_idempotent(paths):
    d1 = ensure_dfnet3_checkpoint(paths)
    d2 = ensure_dfnet3_checkpoint(paths)
    assert d1 == d2
    assert (d1 / "config.ini").is_file()
    assert any((d1 / "checkpoints").iterdir())
    # o marcador fica na raiz da extração; o zip traz um diretório DeepFilterNet3/ no topo
    marker = json.loads((paths.data / "models" / "DeepFilterNet3" / ".cctrain-extracted.json").read_text())
    assert marker["zip_sha256"] == vendored_info()["files"]["models/DeepFilterNet3.zip"]["sha256"]
