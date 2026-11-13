"""pro-v5: adaptadores LoRA/FiLM sobre a base congelada (DFNet3 publicado).

(a) perfil desligado -> saída bit-idêntica ao DFNet3 publicado (mesmo com adaptadores já treinados, B ≠ 0);
(b) passo 0 com perfil ligado -> bit-idêntica ao Pro sem adaptadores com perfil (e ao DFNet3: condicionamento zerado);
(c) só os parâmetros esperados (condicionamento + adaptadores) recebem gradiente;
(d) checkpoint salva/carrega os adaptadores (e o eval `tse_eval.build_model` os reconstrói).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cctrain.dfvendor import ensure_df_importable  # noqa: E402

ensure_df_importable()


@pytest.fixture(scope="module")
def env():
    from cctrain.m2.config import load_stage_f_config
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3
    from cctrain.train.stage_f import StageFHParams
    from train_pro_v4_presence import build_model

    hp = StageFHParams.from_config(load_stage_f_config())
    ref, _ = load_upstream_dfnet3(device="cpu")
    _freeze(ref)
    return hp, ref, build_model


def _inputs(b=3, t=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    spec = torch.randn(b, 1, t, 481, 2, generator=g) * 0.1
    ferb = torch.randn(b, 1, t, 32, generator=g) * 0.3 - 0.5
    fspec = torch.randn(b, 1, t, 96, 2, generator=g) * 0.1
    emb = torch.nn.functional.normalize(torch.randn(b, 192, generator=g), dim=-1)
    return spec, ferb, fspec, emb


def _p3(env, film=False, seed=1):
    from cctrain.models.adapters import AdapterConfig, attach_adapters

    hp, _, build_model = env
    torch.manual_seed(seed)
    m = build_model("cpu", hp, presence=False)
    cfg = AdapterConfig(gru_rank=8, enc_rank=16, gru_film=film)
    sites = attach_adapters(m, cfg)
    return m.eval(), cfg, sites


def _randomize(m, std=0.05):
    """Simula adaptadores e condicionamento já treinados (ramos de saída ≠ 0)."""
    from cctrain.models.adapters import param_groups

    g = param_groups(m)
    named = dict(m.named_parameters())
    with torch.no_grad():
        for n in g["adapter"] + g["cond"]:
            if "freq_mixer.norm" in n:
                continue
            named[n].add_(torch.randn_like(named[n]) * std)


def _freeze(m):
    """Como no uso (base congelada): no CPU o kernel da GRU muda conforme os pesos exigem gradiente (oneDNN só sem
    grad), o que dá ~1 ulp de diferença; as comparações bit a bit são feitas no mesmo regime."""
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def _eq(x, y):
    return all(torch.equal(a, b) for a, b in zip(x, y))


@pytest.mark.parametrize("film", [False, True])
def test_a_profile_off_bit_identical_to_dfn3(env, film):
    _, ref, _ = env
    m, _, _ = _p3(env, film)
    _randomize(m)
    _freeze(m)
    spec, ferb, fspec, emb = _inputs()
    with torch.no_grad():
        want = ref(spec.clone(), ferb, fspec)
        got = m(spec.clone(), ferb, fspec, emb, torch.zeros(3, 1))
    assert _eq(got, want)


def test_a_mixed_batch_off_rows_match(env):
    """Lote misto: as linhas sem perfil continuam iguais ao DFNet3 (as com perfil mudam)."""
    _, ref, _ = env
    m, _, _ = _p3(env, True)
    _randomize(m)
    _freeze(m)
    spec, ferb, fspec, emb = _inputs()
    mask = torch.tensor([[0.0], [1.0], [0.0]])
    with torch.no_grad():
        want = ref(spec.clone(), ferb, fspec)
        got = m(spec.clone(), ferb, fspec, emb, mask)
    for w, g_ in zip(want, got):
        torch.testing.assert_close(g_[[0, 2]], w[[0, 2]], rtol=1e-6, atol=1e-6)  # sub-lote da GRU: ~1 ulp
    assert not torch.allclose(got[0][1], want[0][1], atol=1e-4)


def test_b_step0_profile_on_identical(env):
    hp, ref, build_model = env
    m, _, _ = _p3(env, True)
    _freeze(m)
    torch.manual_seed(1)
    plain = _freeze(build_model("cpu", hp, presence=False))  # mesmo Pro, mesma semente, sem adaptadores
    spec, ferb, fspec, emb = _inputs()
    on = torch.ones(3, 1)
    with torch.no_grad():
        got = m(spec.clone(), ferb, fspec, emb, on)
        want_pro = plain(spec.clone(), ferb, fspec, emb, on)
        want_dfn3 = ref(spec.clone(), ferb, fspec)
    assert _eq(got, want_pro)
    assert _eq(got, want_dfn3)  # condicionamento do zero: no passo 0 o Pro com perfil é o DFNet3


def test_c_only_expected_params_get_grad(env):
    from cctrain.models.adapters import apply_trainable, param_groups
    from cctrain.train.stage_f import prepare_student

    m, _, sites = _p3(env, True)
    prepare_student(m)
    tr = apply_trainable(m, train_cond=True, train_adapters=True)
    g = param_groups(m)
    spec, ferb, fspec, emb = _inputs(t=30)
    out = m(spec.clone(), ferb, fspec, emb, torch.tensor([[1.0], [0.0], [1.0]]))
    (out[0].pow(2).mean() + out[2].mean()).backward()
    named = dict(m.named_parameters())
    for n in g["base"]:
        assert named[n].grad is None, n
    ids = {id(p) for p in tr}
    for n in g["adapter"]:
        assert id(named[n]) in ids
    # ramos de saída zerados recebem gradiente no passo 0 (B do LoRA, FiLM das GRUs, out_proj da atenção)
    for n in g["adapter"]:
        if n.endswith("_B") or ".film." in n:
            assert named[n].grad is not None and named[n].grad.abs().sum() > 0, n
    assert named["cross_attn.out_proj.weight"].grad.abs().sum() > 0
    assert named["film.out.weight"].grad.abs().sum() > 0
    assert len(sites["gru"]) == 3 and len(sites["enc_conv"]) == 5 and len(sites["enc_linear"]) == 1


def test_d_checkpoint_roundtrip(env, tmp_path):
    from cctrain.eval.tse_eval import build_model as eval_build
    from cctrain.models.adapters import set_gate
    from cctrain.train.checkpoint import load_checkpoint, save_checkpoint

    m, cfg, _ = _p3(env, True)
    _randomize(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    p = save_checkpoint(tmp_path, step=7, model=m.state_dict(), optimizer=opt, sampler_state={"step": 7},
                        cfg_hash="x", commit=None, extra={"adapter": cfg.to_dict()})
    ck = load_checkpoint(p)
    assert any(".lora." in k for k in ck.model) and any("lora_B" in k for k in ck.model)
    m2 = eval_build(p, "v4", "cpu")
    assert m2.adapter_cfg == cfg
    spec, ferb, fspec, emb = _inputs()
    with torch.no_grad():
        a = m(spec.clone(), ferb, fspec, emb, torch.ones(3, 1))
        b = m2(spec.clone(), ferb, fspec, emb, torch.ones(3, 1))
    assert _eq(a, b)
    # o caminho do eval que chama os submódulos direto: set_gate liga os adaptadores
    z = torch.randn(3, 5, 512)
    _freeze(m2)
    with torch.no_grad():
        set_gate(m2, torch.ones(3), emb)
        x = m2.base.enc.emb_gru(z)[0]
        set_gate(m2, None)
        y = m2.base.enc.emb_gru(z)[0]
        y_ref = env[1].enc.emb_gru(z)[0]
    assert torch.equal(y, y_ref) and not torch.allclose(x, y)
