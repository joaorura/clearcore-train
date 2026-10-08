"""Testes unitários para o módulo de Cross-Attention Causal em Sub-bandas e PDFNet3Pro.

Valida:
1. Causalidade estrita: A saída no tempo t não muda se o futuro t+1 for perturbado.
2. Identidade exata no init: Com pesos zerados (padrão), o módulo é uma identidade perfeita.
3. Isolamento espectral vocal: A modulação fina atua estritamente nas sub-bandas ERB 3..19 (300 Hz - 4 kHz).
4. Fallback gracioso do PDFNet3Pro: Quando cross_attention=False, resultado é bit-exato ao pDFNet3.
5. Exportabilidade ONNX: Exportação com eixos dinâmicos de tempo sem erros de opset dinâmico,
   e paridade numérica com ONNX Runtime.
"""

from __future__ import annotations

import io
import math
import time

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from cctrain.models.cross_attention import (
    DEFAULT_VOICE_ERB_END,
    DEFAULT_VOICE_ERB_START,
    CausalSubbandCrossAttention,
    SubbandCrossAttentionConfig,
)
from cctrain.models.dfnet3_upstream import df_features_batch, load_upstream_dfnet3
from cctrain.models.film import FilmConfig, FilmGenerator
from cctrain.models.pdfnet3 import PDFNet3
from cctrain.models.pdfnet3_pro import PDFNet3Pro


@pytest.fixture(scope="module")
def upstream():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model, df_state = load_upstream_dfnet3(device="cpu")
    yield model, df_state
    torch.use_deterministic_algorithms(False)


def _generator(scale: float = 0.5) -> FilmGenerator:
    gen = FilmGenerator(FilmConfig(math.log(10.0), 4.0))
    if scale:
        g = torch.Generator().manual_seed(42)
        with torch.no_grad():
            gen.out.weight.copy_(torch.randn(gen.out.weight.shape, generator=g) * scale)
            gen.out.bias.copy_(torch.randn(gen.out.bias.shape, generator=g) * scale)
    return gen


def _audio(n: int = 2, seconds: float = 0.5, seed: int = 10) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(int(48_000 * seconds)) / 48_000.0
    return [
        (0.1 * np.sin(2 * np.pi * (200 + 100 * i) * t) + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
        for i in range(n)
    ]


# ==============================================================================
# 1. Testes de Causalidade Estrita
# ==============================================================================


@pytest.mark.parametrize("t_cut", [0, 5, 12, 19])
def test_cross_attention_strict_causality(t_cut: int):
    """Garante que a saída em tau <= t_cut não sofre qualquer influência de dados em tau > t_cut."""
    torch.manual_seed(42)
    cfg = SubbandCrossAttentionConfig(zero_init=False)
    module = CausalSubbandCrossAttention(cfg).eval()

    total_t = 20
    q = torch.randn(2, total_t, cfg.query_dim)
    emb = torch.randn(2, cfg.embedding_dim)

    # 1. Forward original
    out_orig = module(q, emb)

    # 2. Perturba estritamente o futuro a partir de t_cut + 1
    q_perturbed = q.clone()
    if t_cut + 1 < total_t:
        q_perturbed[:, t_cut + 1 :] += 10.0 * torch.randn_like(q_perturbed[:, t_cut + 1 :])

    out_perturbed = module(q_perturbed, emb)

    # 3. O passado e o presente (0 até t_cut inclusive) DEVEM ser bit-exatamente idênticos
    diff_out = (out_orig.output[:, : t_cut + 1] - out_perturbed.output[:, : t_cut + 1]).abs().max().item()
    diff_gains = (
        out_orig.subband_gains[:, :, : t_cut + 1] - out_perturbed.subband_gains[:, :, : t_cut + 1]
    ).abs().max().item()
    diff_attn = (
        out_orig.attn_weights[:, :, : t_cut + 1] - out_perturbed.attn_weights[:, :, : t_cut + 1]
    ).abs().max().item()

    assert diff_out == 0.0, f"Vazamento causal nos embeddings em t<={t_cut}: max diff = {diff_out}"
    assert diff_gains == 0.0, f"Vazamento causal nos ganhos de sub-bandas em t<={t_cut}: max diff = {diff_gains}"
    assert diff_attn == 0.0, f"Vazamento causal nos pesos de atenção em t<={t_cut}: max diff = {diff_attn}"

    # 4. O futuro (se existente) deve refletir a perturbação
    if t_cut + 1 < total_t:
        diff_future = (out_orig.output[:, t_cut + 1 :] - out_perturbed.output[:, t_cut + 1 :]).abs().max().item()
        assert diff_future > 0.0, "Futuro perturbado não alterou a saída futura"


def test_cross_attention_streaming_frame_parity():
    """Valida que processar quadro a quadro (T=1) é numericamente idêntico a processar em lote (T=N)."""
    torch.manual_seed(101)
    cfg = SubbandCrossAttentionConfig(zero_init=False)
    module = CausalSubbandCrossAttention(cfg).eval()

    t_steps = 15
    q = torch.randn(1, t_steps, cfg.query_dim)
    emb = torch.randn(1, cfg.embedding_dim)

    # Execução em lote
    with torch.no_grad():
        batch_out = module(q, emb)

    # Execução streaming quadro a quadro
    stream_outputs = []
    stream_gains = []
    with torch.no_grad():
        for t in range(t_steps):
            frame_q = q[:, t : t + 1, :]
            frame_out = module(frame_q, emb)
            stream_outputs.append(frame_out.output)
            stream_gains.append(frame_out.subband_gains)

    stream_out_cat = torch.cat(stream_outputs, dim=1)
    stream_gains_cat = torch.cat(stream_gains, dim=2)

    diff_out = (batch_out.output - stream_out_cat).abs().max().item()
    diff_gains = (batch_out.subband_gains - stream_gains_cat).abs().max().item()

    assert diff_out < 1e-6, f"Divergência entre modo lote e streaming: {diff_out}"
    assert diff_gains < 1e-6, f"Divergência nos ganhos streaming: {diff_gains}"


# ==============================================================================
# 2. Testes de Identidade e Isolamento Espectral Vocal (300 Hz - 4 kHz)
# ==============================================================================


def test_cross_attention_zero_init_is_exact_identity():
    """Com inicialização padrão (zero_init=True), o módulo deve ser a identidade matemática exata."""
    cfg = SubbandCrossAttentionConfig(zero_init=True)
    module = CausalSubbandCrossAttention(cfg).eval()

    q = torch.randn(3, 8, cfg.query_dim)
    emb = torch.randn(3, cfg.embedding_dim)
    mask = torch.ones(3, 1)

    out = module(q, emb, mask=mask)

    assert torch.equal(out.output, q), "Embedding de saída não é idêntico à entrada em zero_init"
    assert torch.equal(out.subband_gains, torch.ones_like(out.subband_gains)), "Ganhos em zero_init devem ser 1.0"


def test_cross_attention_zero_mask_bypasses():
    """Com mask=0, mesmo com pesos não nulos, a saída deve ser a identidade exata."""
    cfg = SubbandCrossAttentionConfig(zero_init=False)
    module = CausalSubbandCrossAttention(cfg).eval()

    q = torch.randn(2, 6, cfg.query_dim)
    emb = torch.randn(2, cfg.embedding_dim)
    mask = torch.zeros(2, 1)

    out = module(q, emb, mask=mask)

    assert torch.equal(out.output, q), "Com máscara zero, saída deve ser idêntica à entrada"
    assert torch.equal(out.subband_gains, torch.ones_like(out.subband_gains)), "Com máscara zero, ganhos devem ser 1.0"


def test_cross_attention_subband_isolation():
    """Valida que modulações espectrais atuam estritamente na faixa de voz humana (300 Hz - 4 kHz)."""
    cfg = SubbandCrossAttentionConfig(
        zero_init=False,
        voice_band_start=DEFAULT_VOICE_ERB_START,  # Banda 3 (300 Hz)
        voice_band_end=DEFAULT_VOICE_ERB_END,  # Banda 20 (4000 Hz)
    )
    module = CausalSubbandCrossAttention(cfg).eval()

    q = torch.randn(2, 5, cfg.query_dim)
    emb = torch.randn(2, cfg.embedding_dim)
    out = module(q, emb)

    gains = out.subband_gains  # [B, 1, T, 32]

    # Sub-bandas < 300 Hz (0, 1, 2) DEVEM ser neutras (1.0)
    assert torch.equal(gains[..., :DEFAULT_VOICE_ERB_START], torch.ones_like(gains[..., :DEFAULT_VOICE_ERB_START]))

    # Sub-bandas > 4000 Hz (20..31) DEVEM ser neutras (1.0)
    assert torch.equal(gains[..., DEFAULT_VOICE_ERB_END:], torch.ones_like(gains[..., DEFAULT_VOICE_ERB_END:]))

    # Sub-bandas vocais (3..19) DEVEM receber modulação não neutra
    voice_gains = gains[..., DEFAULT_VOICE_ERB_START:DEFAULT_VOICE_ERB_END]
    assert not torch.equal(voice_gains, torch.ones_like(voice_gains))

    # Atenção deve somar 1.0 sobre as sub-bandas vocais (softmax)
    attn_sums = out.attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)


# ==============================================================================
# 3. Testes do PDFNet3Pro (Dual-Conditioning e Fallback)
# ==============================================================================


def test_pdfnet3_pro_fallback_is_bit_exact(upstream):
    """Com cross_attention=False, PDFNet3Pro deve ser bit-exato ao pDFNet3 aprovado."""
    base, df_state = upstream
    spec, erb, specf = df_features_batch(_audio(seed=1), df_state)
    emb = torch.randn(2, 192, generator=torch.Generator().manual_seed(10))
    mask = torch.ones(2, 1)

    gen = _generator(0.5)
    model_std = PDFNet3(base, gen).eval()
    model_pro = PDFNet3Pro(base, gen, enable_cross_attention=True).eval()

    with torch.no_grad():
        ref = model_std(spec.clone(), erb.clone(), specf.clone(), emb.clone(), mask.clone())
        # Fallback explícito
        out_fb = model_pro(spec.clone(), erb.clone(), specf.clone(), emb.clone(), mask.clone(), cross_attention=False)

    for i, (r, f) in enumerate(zip(ref, out_fb)):
        diff = (r - f).abs().max().item() if r.numel() else 0.0
        assert diff == 0.0, f"Divergência no canal {i} do fallback: {diff}"


def test_pdfnet3_pro_zero_init_matches_std(upstream):
    """Com inicialização zero padrão, PDFNet3Pro deve ser bit-exato ao pDFNet3."""
    base, df_state = upstream
    spec, erb, specf = df_features_batch(_audio(seed=2), df_state)
    emb = torch.randn(2, 192, generator=torch.Generator().manual_seed(20))
    mask = torch.ones(2, 1)

    gen = _generator(0.5)
    model_std = PDFNet3(base, gen).eval()
    # Inicialização zero padrão
    model_pro = PDFNet3Pro(base, gen).eval()

    with torch.no_grad():
        ref = model_std(spec.clone(), erb.clone(), specf.clone(), emb.clone(), mask.clone())
        out = model_pro(spec.clone(), erb.clone(), specf.clone(), emb.clone(), mask.clone(), cross_attention=True)

    for i, (r, o) in enumerate(zip(ref, out)):
        diff = (r - o).abs().max().item() if r.numel() else 0.0
        assert diff == 0.0, f"Divergência no canal {i} em zero-init: {diff}"


def test_pdfnet3_pro_mask_zero_matches_base(upstream):
    """Com mask=0, PDFNet3Pro deve ser bit-exato ao modelo base DeepFilterNet3."""
    base, df_state = upstream
    spec, erb, specf = df_features_batch(_audio(seed=3), df_state)
    emb = torch.randn(2, 192, generator=torch.Generator().manual_seed(30))
    mask_zero = torch.zeros(2, 1)

    gen = _generator(0.5)
    model_pro = PDFNet3Pro(base, gen).eval()

    with torch.no_grad():
        ref_base = base(spec.clone(), erb.clone(), specf.clone())
        out_pro = model_pro(spec.clone(), erb.clone(), specf.clone(), emb.clone(), mask_zero, cross_attention=True)

    for i, (r, o) in enumerate(zip(ref_base, out_pro)):
        diff = (r - o).abs().max().item() if r.numel() else 0.0
        assert diff == 0.0, f"Divergência no canal {i} com máscara zero: {diff}"


def test_pdfnet3_pro_active_dual_conditioning(upstream):
    """Com pesos treinados/ativos, Dual-Conditioning deve enriquecer a inferência sem inf/nan."""
    base, df_state = upstream
    spec, erb, specf = df_features_batch(_audio(seed=4), df_state)
    emb = torch.randn(2, 192, generator=torch.Generator().manual_seed(40))
    mask = torch.ones(2, 1)

    gen = _generator(0.5)
    ca_active = CausalSubbandCrossAttention(SubbandCrossAttentionConfig(zero_init=False))
    model_pro = PDFNet3Pro(base, gen, cross_attention=ca_active).eval()
    model_std = PDFNet3(base, gen).eval()

    with torch.no_grad():
        ref_std = model_std(spec.clone(), erb.clone(), specf.clone(), emb.clone(), mask.clone())
        out_pro, details = model_pro(
            spec.clone(), erb.clone(), specf.clone(), emb.clone(), mask.clone(),
            cross_attention=True, return_attention_details=True
        )

    # Saída espectral deve diferir da versão apenas com FiLM
    diff = (ref_std[0] - out_pro[0]).abs().max().item()
    assert diff > 0.0, "Dual conditioning ativo não alterou a predição espectral"

    # Todas as saídas devem ser finitas
    for o in out_pro:
        assert torch.isfinite(o).all()

    assert torch.isfinite(details.attn_weights).all()
    assert torch.isfinite(details.subband_gains).all()


# ==============================================================================
# 4. Testes de Exportabilidade ONNX e Compatibilidade de Operadores
# ==============================================================================


def test_cross_attention_onnx_export_and_runtime_parity():
    """Valida que o módulo de Cross-Attention exporta para ONNX sem erros dinâmicos e roda no ONNX Runtime."""
    torch.manual_seed(55)
    cfg = SubbandCrossAttentionConfig(zero_init=False)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, m: CausalSubbandCrossAttention):
            super().__init__()
            self.m = m

        def forward(self, q: torch.Tensor, emb: torch.Tensor, mask: torch.Tensor):
            res = self.m(q, emb, mask)
            return res.output, res.subband_gains

    wrapper = ExportWrapper(CausalSubbandCrossAttention(cfg)).eval()

    q_dummy = torch.randn(1, 7, cfg.query_dim)
    emb_dummy = torch.randn(1, cfg.embedding_dim)
    mask_dummy = torch.ones(1, 1)

    buf = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (q_dummy, emb_dummy, mask_dummy),
        buf,
        input_names=["query", "embedding", "mask"],
        output_names=["output", "subband_gains"],
        dynamic_axes={"query": {1: "time"}, "output": {1: "time"}, "subband_gains": {2: "time"}},
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_bytes = buf.getvalue()
    proto = onnx.load_from_string(onnx_bytes)
    onnx.checker.check_model(proto)

    # Executa no ONNX Runtime com múltiplos comprimentos temporais
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])

    for test_t in (1, 5, 24):
        q_test = torch.randn(1, test_t, cfg.query_dim)
        emb_test = torch.randn(1, cfg.embedding_dim)
        mask_test = torch.ones(1, 1)

        with torch.no_grad():
            ref_out, ref_gains = wrapper(q_test, emb_test, mask_test)

        ort_inputs = {
            "query": q_test.numpy(),
            "embedding": emb_test.numpy(),
            "mask": mask_test.numpy(),
        }
        ort_out, ort_gains = sess.run(None, ort_inputs)

        err_out = np.abs(ref_out.numpy() - ort_out).max()
        err_gains = np.abs(ref_gains.numpy() - ort_gains).max()

        assert err_out < 1e-4, f"Erro ONNX vs PyTorch no tempo={test_t}: {err_out}"
        assert err_gains < 1e-4, f"Erro nos ganhos ONNX vs PyTorch no tempo={test_t}: {err_gains}"


def test_cross_attention_latency_realtime_budget():
    """Valida que a latência de inferência por frame é bem inferior a 0.2 ms."""
    torch.manual_seed(99)
    cfg = SubbandCrossAttentionConfig()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, m: CausalSubbandCrossAttention):
            super().__init__()
            self.m = m

        def forward(self, q: torch.Tensor, emb: torch.Tensor, mask: torch.Tensor):
            res = self.m(q, emb, mask)
            return res.output, res.subband_gains

    wrapper = ExportWrapper(CausalSubbandCrossAttention(cfg)).eval()

    q_1frame = torch.randn(1, 1, cfg.query_dim)
    emb_dummy = torch.randn(1, cfg.embedding_dim)
    mask_dummy = torch.ones(1, 1)

    buf = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (q_1frame, emb_dummy, mask_dummy),
        buf,
        input_names=["query", "embedding", "mask"],
        output_names=["output", "subband_gains"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])

    inputs = {
        "query": q_1frame.numpy(),
        "embedding": emb_dummy.numpy(),
        "mask": mask_dummy.numpy(),
    }

    # Warmup
    for _ in range(50):
        sess.run(None, inputs)

    # Benchmark 1000 iterações de frame único
    n_iters = 1000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        sess.run(None, inputs)
    t1 = time.perf_counter()

    lat_ms = (t1 - t0) / n_iters * 1000.0
    # Latência deve ser < 0.2 ms
    assert lat_ms < 0.2, f"Latência por frame excedeu o teto de 0.2 ms: {lat_ms:.4f} ms"
