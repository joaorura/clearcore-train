"""perf/v5-graph: passo sem sync do pro-v5 (`cctrain.train.v5_fast`).

(a) `IstftSF` (sem o teste NOLA do `torch.istft`) = `df.loss.Istft` bit a bit;
(b) `AdapterGate.static` (lote homogêneo) = caminho dinâmico (unique/nonzero), saída e gradientes;
(c) `StepRunner` (eager, perdas sf e fast) = `step_v5_ref`: perda e gradientes por 20 passos travados (fp32, CPU);
(d) em CUDA: modo `graph` (CUDA Graph de forward+perdas+backward) = eager, 20 passos travados.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cctrain.dfvendor import ensure_df_importable  # noqa: E402

ensure_df_importable()

T_FR = 48


def _batch(b=4, t=T_FR, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    u = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    clean = r(b, 1, t, 481, 2) * 0.1
    interf = r(b, 1, t, 481, 2) * 0.1
    spec = clean + interf + r(b, 1, t, 481, 2) * 0.02
    bt = {"spec": spec, "spec_clean": clean, "spec_interf": interf, "mask": torch.ones(b, 1),
          "feat_erb": r(b, 1, t, 32) * 0.3 - 0.5, "feat_spec": r(b, 1, t, 96, 2) * 0.1,
          "embedding": torch.nn.functional.normalize(r(b, 192), dim=-1),
          "target_active": u(b, t) > 0.4, "interf_active": u(b, t) > 0.5, "digital_silence": u(b, t) > 0.9,
          "interf_active_wet45": u(b, t) > 0.4, "n_interf": torch.tensor([0, 1, 2, 1][:b]),
          "has_target": torch.tensor([True, True, False, True][:b]), "max_freq": torch.tensor([24000., 8000., 16000., 12000.][:b])}
    return {k: v.to(device) for k, v in bt.items()}


@pytest.fixture(scope="module")
def tp():
    return pytest.importorskip("train_pro_v5")


def _setup(tp, device="cpu"):
    a = tp.parse_args(["--variant", "P3"])
    hp = tp.StageFHParams.from_config(tp.load_stage_f_config(), run_id="t", num_workers=0, batch_size=4)
    model, _, _, _, train_cond, _ = tp.build_v5(a, device, hp, tp.Paths.from_env())
    tr = tp.prepare(model, train_cond)
    with torch.no_grad():  # LoRA B ≠ 0 e condicionamento ≠ 0: os adaptadores participam
        for p in tr:
            p.add_(torch.randn_like(p) * 1e-3)
    from df.loss import Istft

    win = torch.hann_window(960).sqrt()
    istft = Istft(960, tp.HOP, win).to(device)
    return a, hp, model, tr, istft, tp.ModelParams()


def _flat(tr):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in tr])


def test_istft_sf_bitwise():
    from df.loss import Istft

    from cctrain.train.v5_fast import IstftSF

    for win in (torch.hann_window(960), torch.hann_window(960).sqrt()):
        i = Istft(960, 480, win)
        x = torch.randn(3, 1, 50, 481, 2)
        assert torch.equal(i(x), IstftSF(i)(x))


def test_static_gate_matches_dynamic(tp):
    a, hp, model, tr, istft, params = _setup(tp)
    b = _batch()
    outs = []
    for static in (None, True):
        model.adapter_gate.static = static
        for p in tr:
            p.grad = None
        y = model(b["spec"].clone(), b["feat_erb"], b["feat_spec"], b["embedding"], b["mask"])
        sum(t.float().square().mean() for t in y).backward()
        outs.append(([t.detach() for t in y], _flat(tr)))
        model.adapter_gate.static = None
    for x, y in zip(outs[0][0], outs[1][0]):
        assert torch.equal(x, y)
    assert torch.equal(outs[0][1], outs[1][1])


def _lockstep(tp, runner_kw, device="cpu", steps=20, rtol_g=1e-4):
    from cctrain.train.v5_fast import StepRunner

    a, hp, model, tr, istft, params = _setup(tp, device)
    runner = StepRunner(model, istft, params, hp, a, False, tr, **runner_kw)
    opt = torch.optim.AdamW(tr, lr=1e-3)
    worst = (0.0, 0.0)
    for i in range(steps):
        b = _batch(seed=i % 4, device=device)
        for p in tr:
            p.grad = None
        pr = tp.step_v5_ref(model, istft, params, hp, b, a, False)
        pr["total"].backward()
        lr_, gr = float(pr["total"]), _flat(tr).clone()
        ref_grads = [None if p.grad is None else p.grad.clone() for p in tr]
        del pr  # nenhum grafo do autograd vivo na captura (AccumulateGrad guardaria a stream padrão)
        pv = runner(b, True)
        lv, gv = float(pv["total"]), _flat(tr).clone()
        worst = (max(worst[0], abs(lv - lr_) / abs(lr_)), max(worst[1], float((gv - gr).norm() / gr.norm())))
        for p, g in zip(tr, ref_grads):
            p.grad = g
        opt.step()
    assert worst[0] <= 1e-5, worst
    assert worst[1] <= rtol_g, worst


@pytest.mark.parametrize("fast_loss", [False, True])
def test_runner_eager_matches_reference_cpu(tp, fast_loss):
    _lockstep(tp, {"mode": "eager", "loss_fn": tp.step_v5 if fast_loss else None})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph exige GPU")
def test_runner_graph_matches_reference_cuda(tp):
    torch.backends.cudnn.deterministic = True
    _lockstep(tp, {"mode": "graph", "loss_fn": tp.step_v5}, device="cuda", rtol_g=1e-3)
