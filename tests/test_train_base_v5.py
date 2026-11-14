"""pro-v5-fullft: --train-base {frozen,all,gru} (grupos do AdamW, gradiente na base, caminho sem perfil)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cctrain.dfvendor import ensure_df_importable  # noqa: E402

ensure_df_importable()

from cctrain.models.adapters import COND_PREFIXES  # noqa: E402
from test_v5_fast_step import _batch  # noqa: E402

LR, LR_BASE = 3e-4, 1e-5


@pytest.fixture(scope="module")
def tp():
    return pytest.importorskip("train_pro_v5")


def _build(tp, train_base, variant="P1", extra=()):
    a = tp.parse_args(["--variant", variant, "--train-base", train_base, "--p-drop", "0.15",
                       "--profile-drop-mode", "batch", *extra])
    hp = tp.StageFHParams.from_config(tp.load_stage_f_config(), run_id="t", num_workers=0, batch_size=4)
    model, _, _, _, train_cond, _ = tp.build_v5(a, "cpu", hp, tp.Paths.from_env())
    tr = tp.prepare(model, train_cond, a.train_base)
    with torch.no_grad():  # condicionamento ≠ 0 (os ramos de saída começam zerados)
        for p in tr[: model._n_new_trainable]:
            p.add_(torch.randn_like(p) * 1e-3)
    return a, hp, model, tr


def _names(model, params):
    ids = {id(p): n for n, p in model.named_parameters()}
    return [ids[id(p)] for p in params]


def _bn_names(model):
    out = set()
    for mn, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            out |= {f"{mn}.{pn}" for pn, _ in m.named_parameters(recurse=False)}
    return out


@pytest.mark.parametrize("mode", ["frozen", "all", "gru"])
def test_lr_groups(tp, mode):
    a, _, model, tr = _build(tp, mode)
    groups = tp.opt_groups(model, tr, LR, LR_BASE, 0.01)
    opt = torch.optim.AdamW(groups, lr=LR)
    new = _names(model, opt.param_groups[0]["params"])
    assert all(n.startswith(COND_PREFIXES) for n in new)
    assert opt.param_groups[0]["lr"] == LR and opt.param_groups[0]["weight_decay"] == 0.01
    if mode == "frozen":
        assert len(opt.param_groups) == 1
        return
    assert len(opt.param_groups) == 2
    g = opt.param_groups[1]
    assert g["lr"] == LR_BASE and g["weight_decay"] == 0.0
    base = _names(model, g["params"])
    assert base and all(n.startswith("base.") for n in base)
    assert not set(base) & _bn_names(model)
    trainable_base = {n for n, p in model.named_parameters() if p.requires_grad and n.startswith("base.")}
    assert trainable_base == set(base)
    if mode == "gru":
        assert all(".emb_gru.gru." in n or ".df_gru.gru." in n for n in base)
        assert len({n.split(".gru.")[0] for n in base}) == 3
    else:
        assert any("erb_conv" in n for n in base) and any("df_gru" in n for n in base)
    # o cosseno multiplica as duas LRs pelo mesmo fator
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)
    assert sched.get_last_lr() == [LR * 0.5, LR_BASE * 0.5]


@pytest.mark.parametrize("mode", ["all", "gru"])
@pytest.mark.parametrize("profile_on", [True, False])
def test_grad_reaches_base(tp, mode, profile_on):
    a, hp, model, tr = _build(tp, mode)
    from df.loss import Istft

    istft = Istft(960, tp.HOP, torch.hann_window(960).sqrt())
    b = _batch()
    if not profile_on:  # lote todo sem perfil (alvo = toda a fala, como o mixer faz com mask0)
        b["mask"] = torch.zeros_like(b["mask"])
        b["spec_clean"] = b["spec_clean"] + b["spec_interf"]
    parts = tp.step_v5(model, istft, tp.ModelParams(), hp, b, a, False)
    assert torch.isfinite(parts["total"])
    parts["total"].backward()
    k = model._n_new_trainable
    gb = [p.grad for p in tr[k:]]
    assert all(torch.isfinite(g).all() for g in gb if g is not None)
    # quase todos recebem gradiente (exceções: df_fc_a, que o DFNet3 não usa no forward, e df_convp atrás de ReLU
    # morta neste lote aleatório)
    frac_nz = sum(int(g is not None and g.abs().sum() > 0) for g in gb) / len(gb)
    assert frac_nz > 0.9, frac_nz
    gnew = sum(float(p.grad.abs().sum()) for p in tr[:k] if p.grad is not None)
    if profile_on:
        assert gnew > 0
    else:  # sem perfil o condicionamento é × 0: nada chega às camadas novas
        assert gnew == 0


def test_profile_off_frozen_no_grad(tp):
    """Com a base congelada o lote sem perfil não move nenhum treinável (o v5 de antes, gradiente exatamente 0)."""
    a, hp, model, tr = _build(tp, "frozen")
    b = _batch()
    b["mask"] = torch.zeros_like(b["mask"])
    y = model(b["spec"].clone(), b["feat_erb"], b["feat_spec"], b["embedding"], b["mask"])
    sum(t.float().square().mean() for t in y if torch.is_tensor(t) and t.requires_grad).backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in tr)


def test_runner_profile_off_eager(tp):
    from cctrain.train.v5_fast import StepRunner
    from df.loss import Istft

    a, hp, model, tr = _build(tp, "all")
    istft = Istft(960, tp.HOP, torch.hann_window(960).sqrt())
    runner = StepRunner(model, istft, tp.ModelParams(), hp, a, False, tr, mode="eager", loss_fn=tp.step_v5)
    opt = torch.optim.AdamW(tp.opt_groups(model, tr, LR, LR_BASE, 0.0), lr=LR)
    w0 = [p.detach().clone() for p in tr[model._n_new_trainable:]]
    for on in (True, False, True):
        b = _batch()
        if not on:
            b["mask"] = torch.zeros_like(b["mask"])
        parts = runner(b, on)
        assert torch.isfinite(parts["total"])
        opt.step()
    moved = sum(int(not torch.equal(p, q)) for p, q in zip(tr[model._n_new_trainable:], w0))
    assert moved > 0.9 * len(w0)
