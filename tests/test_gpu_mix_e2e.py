"""Equivalência fim a fim CPU (`StageV4Dataset`) vs `--gpu-mix` (receita + `GpuMixer`) com dados reais; pula sem o
banco de embeddings. Na GPU quando houver, senão em CPU (mesmo código torch)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.parametrize("dtype,wire", [("f64", "dense"), ("f32", "dense"), ("f32", "sparse")])
def test_gpu_mix_matches_cpu(paths, capsys, dtype, wire):
    if not (paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz").is_file():
        pytest.skip("banco de embeddings ausente")
    from gpu_mix_equiv import make, mixer_kwargs, run

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds, mixer, _ = make(dev, mixer_kwargs(dtype), batch_size=6)
    r = run(ds, mixer, list(range(24)), 6, wire)
    with capsys.disabled():
        print(f"\n{dev} {dtype} {wire}: {r}")
    assert r["reverb"] > 0 and r["dry_with_interf"] > 0
    assert not r["meta_mismatch"]
    assert max(r["label_frames_diff_pct"].values()) == 0.0
    assert max(r["audio_rel_peak"].values()) < 1e-5
    assert max(r[k] for k in ("level_db_maxdiff", "snr_eff_db_maxdiff")) < 1e-3
    fr = r["feat_rel_peak"]
    assert max(fr["spec"], fr["spec_clean"], fr["spec_interf"], fr["feat_spec"]) < 1e-5 and fr["feat_erb"] < 1e-3


def _same(a, b):
    """Igual, ou até 1e-6 do pico (o `index_add_` da GPU soma com atomics: a ordem das trilhas varia ~1 ulp entre execuções)."""
    if torch.is_tensor(a):
        if a.shape != b.shape:
            return False
        if a.dtype.is_floating_point:
            a, b = torch.nan_to_num(a.double(), nan=-1e30), torch.nan_to_num(b.double(), nan=-1e30)
        return torch.equal(a, b) or float((a.double() - b.double()).abs().max()) <= 1e-6 * float(b.double().abs().max()) + 1e-12
    return a == b


def test_sparse_wire_equals_dense(paths):
    """`collate_recipes(sparse=True)` (só quadros ligados no H2D) dá exatamente o mesmo lote que o denso."""
    if not (paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz").is_file():
        pytest.skip("banco de embeddings ausente")
    from gpu_mix_equiv import make

    from cctrain.data.gpu.recipe import collate_recipes

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds, mixer, _ = make(dev, batch_size=6)
    items = [ds[i] for i in range(12)]
    ref = mixer(collate_recipes(items))
    sp = mixer(collate_recipes(items, sparse=True))
    assert ref.keys() == sp.keys()
    for k in ref:
        assert _same(ref[k], sp[k]), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_overlap_batches_identical(paths):
    """`MixPrefetcher` (mistura numa stream lateral) entrega os mesmos lotes que a mistura síncrona, na mesma ordem."""
    if not (paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz").is_file():
        pytest.skip("banco de embeddings ausente")
    from gpu_mix_equiv import make, mixer_kwargs

    from cctrain.data.gpu.prefetch import MixPrefetcher
    from cctrain.data.gpu.recipe import collate_recipes

    ds, mixer, _ = make("cuda", mixer_kwargs("f32"), batch_size=6)
    rbs = [collate_recipes([ds[i] for i in range(s, s + 6)]) for s in range(0, 24, 6)]
    ref = []
    for rb in rbs:
        o = mixer(rb)
        ref.append((o, mixer.fill_invalid(o, ds)))
        o.pop("valid")
        o.pop("index")
    pf = MixPrefetcher(iter(rbs), mixer, ds, "cuda")
    got = list(pf)
    pf.close()
    assert len(got) == len(ref)
    for (o, nr), (b, nr2) in zip(ref, got):
        assert nr == nr2 and o.keys() == b.keys()
        for k in o:
            assert _same(o[k], b[k]), k
