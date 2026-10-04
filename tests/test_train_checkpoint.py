import random

import numpy as np
import pytest
import torch

from cctrain.train.checkpoint import (
    CheckpointError,
    config_hash,
    list_checkpoints,
    load_latest,
    restore_rng,
    save_checkpoint,
)


def _model_opt():
    m = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5))
    m(torch.ones(1, 3)).sum().backward()
    opt.step()
    sched.step()
    return m, opt, sched


def _save(d, step, m, opt, sched, metric=None, cfg="abc", keep=3, sampler=None):
    return save_checkpoint(d, step=step, model=m.state_dict(), optimizer=opt, scheduler=sched,
                           sampler_state=sampler or {"epoch": 0, "index": step}, cfg_hash=cfg, commit="deadbeef",
                           metric=metric, keep=keep)


def test_roundtrip_restores_state_and_rng(tmp_path):
    m, opt, sched = _model_opt()
    random.seed(1), np.random.seed(2), torch.manual_seed(3)
    _save(tmp_path, 7, m, opt, sched)
    expect = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.seed(99), np.random.seed(99), torch.manual_seed(99)
    ck = load_latest(tmp_path, expected_cfg_hash="abc")
    assert ck.step == 7 and ck.sampler_state == {"epoch": 0, "index": 7} and ck.commit == "deadbeef"
    restore_rng(ck)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expect
    m2, opt2, sched2 = _model_opt()
    m2.load_state_dict(ck.model)
    opt2.load_state_dict(ck.optimizer)
    sched2.load_state_dict(ck.scheduler)
    assert torch.equal(m.weight, m2.weight)
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]


def test_keeps_last_three_plus_best(tmp_path):
    m, opt, sched = _model_opt()
    metrics = {1: 5.0, 2: 1.0, 3: 4.0, 4: 3.0, 5: 2.0, 6: 6.0}
    for s, v in metrics.items():
        _save(tmp_path, s, m, opt, sched, metric=v)
    steps = [c.step for c in list_checkpoints(tmp_path)]
    assert steps == [2, 4, 5, 6]  # os 3 últimos (4, 5, 6) e o melhor (2: metric 1.0)
    assert load_latest(tmp_path).step == 6


def test_atomic_write_leaves_no_partial_file_and_ignores_temporaries(tmp_path):
    m, opt, sched = _model_opt()
    _save(tmp_path, 1, m, opt, sched)
    (tmp_path / "ckpt-00000002.pt.tmp").write_bytes(b"meio escrito")  # sobra de uma queda no meio da gravação
    assert load_latest(tmp_path).step == 1
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp.pt")]


def test_corrupt_latest_falls_back_to_previous_valid(tmp_path):
    m, opt, sched = _model_opt()
    _save(tmp_path, 1, m, opt, sched)
    _save(tmp_path, 2, m, opt, sched)
    latest = tmp_path / "ckpt-00000002.pt"
    latest.write_bytes(latest.read_bytes()[:100])  # truncado
    assert load_latest(tmp_path).step == 1


def test_empty_dir_returns_none_and_cfg_mismatch_refuses(tmp_path):
    assert load_latest(tmp_path) is None
    m, opt, sched = _model_opt()
    _save(tmp_path, 1, m, opt, sched, cfg="aaa")
    with pytest.raises(CheckpointError):
        load_latest(tmp_path, expected_cfg_hash="bbb")


def test_config_hash_is_stable_and_sensitive():
    a = {"lr": 1e-3, "nested": {"b": 2, "a": 1}}
    assert config_hash(a) == config_hash({"nested": {"a": 1, "b": 2}, "lr": 1e-3})
    assert config_hash(a) != config_hash({**a, "lr": 2e-3})


def test_pinned_checkpoint_is_never_pruned(tmp_path):
    m, opt, sched = _model_opt()
    save_checkpoint(tmp_path, step=1, model=m.state_dict(), optimizer=opt, scheduler=sched,
                    sampler_state={"epoch": 0, "index": 1}, cfg_hash="abc", commit=None, keep=1, pin=True)
    for s in (2, 3, 4):
        _save(tmp_path, s, m, opt, sched, keep=1)
    assert sorted(p.name for p in tmp_path.glob("ckpt-*.pt")) == ["ckpt-00000001.pt", "ckpt-00000004.pt"]
    from cctrain.train.checkpoint import pinned_steps

    assert pinned_steps(tmp_path) == {1}
