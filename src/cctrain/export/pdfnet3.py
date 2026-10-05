"""Exportação dos membros do pDFNet3 com FiLM e injeção de pesos retreinados do Estágio F.

O pDFNet3 preserva a topologia ONNX testada e validada no Tract 0.19.16 do asset aprovado
(vendor/approved/df-compatible-release-asset-v1.bin), inserindo os nós FiLM (/emb_gru/linear_in/1/Relu_output_0
e /df_gru/linear_in/linear_in.1/Relu_output_0) e substituindo os initializers/constantes pelos pesos retreinados
do checkpoint do Estágio F (ex.: runs/m2/t3/checkpoints/ckpt-00022500.pt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from cctrain.export.film import (
    CONFIG_INI_SHA256,
    FilmGraphError,
    SITES,
    add_film,
)
from cctrain.export.pack import DFNET_MEMBERS, PackResult, pack_asset, read_members
from cctrain.hashing import sha256_bytes


def fold_conv_bn(
    conv_w: np.ndarray,
    bn_w: np.ndarray,
    bn_b: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Dobra BatchNorm em Conv2d/ConvTranspose2d correspondente."""
    scale = bn_w / np.sqrt(bn_var + eps)
    dim_diff = conv_w.ndim - 1
    scale_expanded = scale[(...,) + (None,) * dim_diff]
    w_folded = conv_w * scale_expanded
    b_folded = bn_b - bn_mean * scale
    return w_folded.astype(np.float32), b_folded.astype(np.float32)


def gru_to_onnx(
    w_ih: np.ndarray,
    w_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte pesos de GRU PyTorch [r, z, n] para ONNX [z, r, h] no formato [num_dir, 3*H, input_size]."""
    r_ih, z_ih, n_ih = np.split(w_ih, 3, axis=0)
    w_onnx = np.concatenate([z_ih, r_ih, n_ih], axis=0)[None].astype(np.float32)

    r_hh, z_hh, n_hh = np.split(w_hh, 3, axis=0)
    r_onnx = np.concatenate([z_hh, r_hh, n_hh], axis=0)[None].astype(np.float32)

    r_bih, z_bih, n_bih = np.split(b_ih, 3)
    r_bhh, z_bhh, n_bhh = np.split(b_hh, 3)
    b_onnx = np.concatenate([z_bih, r_bih, n_bih, z_bhh, r_bhh, n_bhh])[None].astype(np.float32)

    return w_onnx, r_onnx, b_onnx


def set_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    """Atualiza o tensor de um initializer existente no grafo ONNX."""
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            new_init = numpy_helper.from_array(np.ascontiguousarray(array, dtype=np.float32), name=name)
            model.graph.initializer[i].CopyFrom(new_init)
            return
    raise FilmGraphError(f"Initializer {name!r} não encontrado no grafo")


def set_constant(model: onnx.ModelProto, output_name: str, array: np.ndarray) -> None:
    """Atualiza o atributo 'value' de um nó Constant existente no grafo ONNX identificado pelo output."""
    for n in model.graph.node:
        if n.op_type == "Constant" and output_name in n.output:
            for a in n.attribute:
                if a.name == "value":
                    new_t = numpy_helper.from_array(np.ascontiguousarray(array, dtype=np.float32))
                    a.t.CopyFrom(new_t)
                    return
    raise FilmGraphError(f"Nó Constant com saída {output_name!r} não encontrado no grafo")


def inject_stage_f_weights_into_enc(
    enc_proto: onnx.ModelProto,
    base_state: dict[str, np.ndarray],
) -> onnx.ModelProto:
    """Injeta pesos retreinados no enc.onnx já com FiLM inserido."""
    for k in (
        "erb_conv1.0.weight",
        "erb_conv2.0.weight",
        "erb_conv3.0.weight",
        "df_conv0.1.weight",
        "df_conv1.0.weight",
        "df_fc_emb.0.weight",
        "emb_gru.linear_in.0.weight",
        "emb_gru.linear_out.0.weight",
        "lsnr_fc.0.bias",
    ):
        set_initializer(enc_proto, k, base_state["enc." + k])

    for prefix, c_idx, b_idx, wn, bn in [
        ("enc.erb_conv0", 1, 2, "onnx::Conv_282", "onnx::Conv_283"),
        ("enc.erb_conv1", 1, 2, "onnx::Conv_285", "onnx::Conv_286"),
        ("enc.erb_conv2", 1, 2, "onnx::Conv_288", "onnx::Conv_289"),
        ("enc.erb_conv3", 1, 2, "onnx::Conv_291", "onnx::Conv_292"),
        ("enc.df_conv0", 2, 3, "onnx::Conv_294", "onnx::Conv_295"),
        ("enc.df_conv1", 1, 2, "onnx::Conv_297", "onnx::Conv_298"),
    ]:
        wf, bf = fold_conv_bn(
            base_state[f"{prefix}.{c_idx}.weight"],
            base_state[f"{prefix}.{b_idx}.weight"],
            base_state[f"{prefix}.{b_idx}.bias"],
            base_state[f"{prefix}.{b_idx}.running_mean"],
            base_state[f"{prefix}.{b_idx}.running_var"],
        )
        set_initializer(enc_proto, wn, wf)
        set_initializer(enc_proto, bn, bf)

    set_initializer(enc_proto, "onnx::MatMul_337", base_state["enc.lsnr_fc.0.weight"].T)

    w_gru, r_gru, b_gru = gru_to_onnx(
        base_state["enc.emb_gru.gru.weight_ih_l0"],
        base_state["enc.emb_gru.gru.weight_hh_l0"],
        base_state["enc.emb_gru.gru.bias_ih_l0"],
        base_state["enc.emb_gru.gru.bias_hh_l0"],
    )
    set_constant(enc_proto, "/emb_gru/Constant_3_output_0", w_gru)
    set_constant(enc_proto, "/emb_gru/Constant_4_output_0", r_gru)
    set_constant(enc_proto, "/emb_gru/Constant_5_output_0", b_gru)

    onnx.checker.check_model(enc_proto)
    return enc_proto


def inject_stage_f_weights_into_erb_dec(
    erb_proto: onnx.ModelProto,
    base_state: dict[str, np.ndarray],
) -> onnx.ModelProto:
    """Injeta pesos retreinados no erb_dec.onnx."""
    for k in (
        "emb_gru.linear_in.0.weight",
        "emb_gru.linear_out.0.weight",
        "convt3.0.weight",
        "convt2.0.weight",
        "convt1.0.weight",
    ):
        set_initializer(erb_proto, k, base_state["erb_dec." + k])

    for prefix, c_idx, b_idx, wn, bn in [
        ("erb_dec.conv3p", 0, 1, "onnx::Conv_288", "onnx::Conv_289"),
        ("erb_dec.convt3", 1, 2, "onnx::Conv_291", "onnx::Conv_292"),
        ("erb_dec.conv2p", 0, 1, "onnx::Conv_294", "onnx::Conv_295"),
        ("erb_dec.convt2", 1, 2, "onnx::Conv_297", "onnx::Conv_298"),
        ("erb_dec.conv1p", 0, 1, "onnx::Conv_300", "onnx::Conv_301"),
        ("erb_dec.convt1", 1, 2, "onnx::Conv_303", "onnx::Conv_304"),
        ("erb_dec.conv0p", 0, 1, "onnx::Conv_306", "onnx::Conv_307"),
        ("erb_dec.conv0_out", 0, 1, "onnx::Conv_309", "onnx::Conv_310"),
    ]:
        wf, bf = fold_conv_bn(
            base_state[f"{prefix}.{c_idx}.weight"],
            base_state[f"{prefix}.{b_idx}.weight"],
            base_state[f"{prefix}.{b_idx}.bias"],
            base_state[f"{prefix}.{b_idx}.running_mean"],
            base_state[f"{prefix}.{b_idx}.running_var"],
        )
        set_initializer(erb_proto, wn, wf)
        set_initializer(erb_proto, bn, bf)

    w0, r0, b0 = gru_to_onnx(
        base_state["erb_dec.emb_gru.gru.weight_ih_l0"],
        base_state["erb_dec.emb_gru.gru.weight_hh_l0"],
        base_state["erb_dec.emb_gru.gru.bias_ih_l0"],
        base_state["erb_dec.emb_gru.gru.bias_hh_l0"],
    )
    set_constant(erb_proto, "/emb_gru/Constant_3_output_0", w0)
    set_constant(erb_proto, "/emb_gru/Constant_4_output_0", r0)
    set_constant(erb_proto, "/emb_gru/Constant_5_output_0", b0)

    w1, r1, b1 = gru_to_onnx(
        base_state["erb_dec.emb_gru.gru.weight_ih_l1"],
        base_state["erb_dec.emb_gru.gru.weight_hh_l1"],
        base_state["erb_dec.emb_gru.gru.bias_ih_l1"],
        base_state["erb_dec.emb_gru.gru.bias_hh_l1"],
    )
    set_constant(erb_proto, "/emb_gru/Constant_9_output_0", w1)
    set_constant(erb_proto, "/emb_gru/Constant_10_output_0", r1)
    set_constant(erb_proto, "/emb_gru/Constant_11_output_0", b1)

    onnx.checker.check_model(erb_proto)
    return erb_proto


def inject_stage_f_weights_into_df_dec(
    df_proto: onnx.ModelProto,
    base_state: dict[str, np.ndarray],
) -> onnx.ModelProto:
    """Injeta pesos retreinados no df_dec.onnx já com FiLM inserido."""
    for k in (
        "df_convp.1.weight",
        "df_gru.linear_in.0.weight",
        "df_skip.weight",
        "df_out.0.weight",
        "df_fc_a.0.bias",
    ):
        set_initializer(df_proto, k, base_state["df_dec." + k])

    wf, bf = fold_conv_bn(
        base_state["df_dec.df_convp.2.weight"],
        base_state["df_dec.df_convp.3.weight"],
        base_state["df_dec.df_convp.3.bias"],
        base_state["df_dec.df_convp.3.running_mean"],
        base_state["df_dec.df_convp.3.running_var"],
    )
    set_initializer(df_proto, "onnx::Conv_268", wf)
    set_initializer(df_proto, "onnx::Conv_269", bf)

    set_initializer(df_proto, "onnx::MatMul_321", base_state["df_dec.df_fc_a.0.weight"].T)

    w0, r0, b0 = gru_to_onnx(
        base_state["df_dec.df_gru.gru.weight_ih_l0"],
        base_state["df_dec.df_gru.gru.weight_hh_l0"],
        base_state["df_dec.df_gru.gru.bias_ih_l0"],
        base_state["df_dec.df_gru.gru.bias_hh_l0"],
    )
    set_initializer(df_proto, "onnx::GRU_291", w0)
    set_initializer(df_proto, "onnx::GRU_292", r0)
    set_initializer(df_proto, "onnx::GRU_293", b0)

    w1, r1, b1 = gru_to_onnx(
        base_state["df_dec.df_gru.gru.weight_ih_l1"],
        base_state["df_dec.df_gru.gru.weight_hh_l1"],
        base_state["df_dec.df_gru.gru.bias_ih_l1"],
        base_state["df_dec.df_gru.gru.bias_hh_l1"],
    )
    set_initializer(df_proto, "onnx::GRU_311", w1)
    set_initializer(df_proto, "onnx::GRU_312", r1)
    set_initializer(df_proto, "onnx::GRU_313", b1)

    onnx.checker.check_model(df_proto)
    return df_proto


def load_checkpoint_base_state(ckpt_path: Path) -> dict[str, np.ndarray]:
    """Carrega os pesos do base model do checkpoint especificado em formato numpy."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(f"{ckpt_path}: chave 'model' não encontrada no checkpoint")
    base_state = {
        k[len("base."):]: v.detach().cpu().numpy()
        for k, v in ckpt["model"].items()
        if k.startswith("base.")
    }
    if not base_state:
        raise ValueError(f"{ckpt_path}: nenhum parâmetro 'base.*' encontrado")
    return base_state


def build_pdfnet3_stage_f_members(
    approved_asset: Path,
    ckpt_path: Path,
    out_dir: Path,
) -> dict[str, Path]:
    """Constrói os 4 membros canônicos do pDFNet3 com os pesos retreinados do Estágio F."""
    members = read_members(approved_asset)
    if sorted(members) != sorted(DFNET_MEMBERS):
        raise FilmGraphError(f"membros inesperados: {sorted(members)}")
    if sha256_bytes(members["config.ini"]) != CONFIG_INI_SHA256:
        raise FilmGraphError("config.ini difere do v0.5.6")

    base_state = load_checkpoint_base_state(ckpt_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. enc.onnx: add_film + injeção de pesos
    enc_m = add_film(onnx.load_from_string(members["enc.onnx"]), SITES["enc"])
    enc_m = inject_stage_f_weights_into_enc(enc_m, base_state)
    enc_bytes = enc_m.SerializeToString()

    # 2. erb_dec.onnx: injeção de pesos
    erb_m = onnx.load_from_string(members["erb_dec.onnx"])
    erb_m = inject_stage_f_weights_into_erb_dec(erb_m, base_state)
    erb_bytes = erb_m.SerializeToString()

    # 3. df_dec.onnx: add_film + injeção de pesos
    df_m = add_film(onnx.load_from_string(members["df_dec.onnx"]), SITES["df_dec"])
    df_m = inject_stage_f_weights_into_df_dec(df_m, base_state)
    df_bytes = df_m.SerializeToString()

    # 4. config.ini: cópia exata do v0.5.6 aprovado
    cfg_bytes = members["config.ini"]

    paths: dict[str, Path] = {}
    for name, data in (
        ("enc.onnx", enc_bytes),
        ("erb_dec.onnx", erb_bytes),
        ("df_dec.onnx", df_bytes),
        ("config.ini", cfg_bytes),
    ):
        p = out_dir / name
        p.write_bytes(data)
        paths[name] = p

    return paths


def build_pdfnet3_stage_f_asset(
    approved_asset: Path,
    ckpt_path: Path,
    out_path: Path,
    out_members_dir: Path | None = None,
) -> PackResult:
    """Exporta e empacota o asset de release do pDFNet3 com pesos do Estágio F."""
    out_path = Path(out_path)
    members_dir = out_members_dir or out_path.parent / "members"
    members = build_pdfnet3_stage_f_members(approved_asset, ckpt_path, members_dir)
    return pack_asset(out_path, [(m, members[m]) for m in DFNET_MEMBERS])
