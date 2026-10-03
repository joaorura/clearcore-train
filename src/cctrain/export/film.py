"""FiLM nos grafos do DFNet3 por edição de grafo (spec §6.1, estágio C).

`add_film` copiado de tools/accelerators/gen_film_onnx.py do Clearcore (MIT OR Apache-2.0), trocando
SystemExit por FilmGraphError.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from cctrain.export.pack import DFNET_MEMBERS, PackResult, pack_asset, read_members
from cctrain.hashing import sha256_bytes, sha256_file

SITES = {
    "enc": "/emb_gru/linear_in/1/Relu_output_0",
    "df_dec": "/df_gru/linear_in/linear_in.1/Relu_output_0",
}
HIDDEN = 256
CONFIG_INI_SHA256 = "415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290"
APPROVED_ASSET_SHA256 = "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616"
_FILM_NODES = ("film_mul", "film_add")


class FilmGraphError(RuntimeError):
    """Grafo fora do esperado para a inserção do FiLM."""


def add_film(model: onnx.ModelProto, site: str) -> onnx.ModelProto:
    g = model.graph
    if any(i.name in ("gamma", "beta") for i in g.input):
        raise FilmGraphError("modelo já tem gamma/beta")
    producers = [i for i, nd in enumerate(g.node) if site in nd.output]
    if len(producers) != 1:
        raise FilmGraphError(f"site {site}: {len(producers)} produtores (esperado 1)")
    for name in ("gamma", "beta"):
        g.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, "S", HIDDEN]))
    scaled, filmed = site + "_gamma", site + "_film"
    for nd in g.node:
        for k, x in enumerate(nd.input):
            if x == site:
                nd.input[k] = filmed
    at = producers[0] + 1
    g.node.insert(at, helper.make_node("Mul", [site, "gamma"], [scaled], name="film_mul"))
    g.node.insert(at + 1, helper.make_node("Add", [scaled, "beta"], [filmed], name="film_add"))
    onnx.checker.check_model(model)
    return model


def _initializers(model: onnx.ModelProto) -> dict[str, tuple]:
    out = {}
    for t in model.graph.initializer:
        a = numpy_helper.to_array(t)
        out[t.name] = (a.dtype.str, a.shape, np.ascontiguousarray(a).tobytes())
    return out


def verify_film_graph(original: bytes, filmed: bytes, site: str) -> None:
    a, b = onnx.load_from_string(original), onnx.load_from_string(filmed)
    if _initializers(a) != _initializers(b):
        raise FilmGraphError("initializers diferem do asset aprovado")
    new = [n for n in b.graph.node if n.name in _FILM_NODES]
    if [(n.name, n.op_type) for n in new] != [("film_mul", "Mul"), ("film_add", "Add")]:
        raise FilmGraphError(f"nós novos inesperados: {[(n.name, n.op_type) for n in new]}")
    rest = [n for n in b.graph.node if n.name not in _FILM_NODES]
    if len(rest) != len(a.graph.node):
        raise FilmGraphError("número de nós originais mudou")
    for x, y in zip(a.graph.node, rest):
        inputs_y = [site if i == site + "_film" else i for i in y.input]
        if (x.op_type, list(x.output), list(x.input)) != (y.op_type, list(y.output), inputs_y):
            raise FilmGraphError(f"nó {x.name} alterado")


def build_film_identity_members(approved_asset: Path, out_dir: Path) -> dict[str, Path]:
    if sha256_file(approved_asset) != APPROVED_ASSET_SHA256:
        raise FilmGraphError(f"{approved_asset}: não é o asset aprovado")
    members = read_members(approved_asset)
    if sorted(members) != sorted(DFNET_MEMBERS):
        raise FilmGraphError(f"membros inesperados: {sorted(members)}")
    if sha256_bytes(members["config.ini"]) != CONFIG_INI_SHA256:
        raise FilmGraphError("config.ini difere do v0.5.6")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in DFNET_MEMBERS:
        data = members[name]
        stem = name.removesuffix(".onnx")
        if stem in SITES:
            filmed = add_film(onnx.load_from_string(data), SITES[stem]).SerializeToString()
            verify_film_graph(data, filmed, SITES[stem])
            data = filmed
        paths[name] = out_dir / name
        paths[name].write_bytes(data)
    return paths


def build_film_identity_asset(approved_asset: Path, out_dir: Path,
                              name: str = "film-identity-m0.tar.gz") -> PackResult:
    members = build_film_identity_members(approved_asset, Path(out_dir) / "members")
    return pack_asset(Path(out_dir) / name, [(m, members[m]) for m in DFNET_MEMBERS])
