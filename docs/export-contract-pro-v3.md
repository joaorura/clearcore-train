# Contrato de export: `pdfnet3-pro-v3-full` (stateful, S=1)

Asset gerado por `scripts/m3_export_pro_v3.py` (código em `src/cctrain/export/pdfnet3_pro_v3.py`) a partir de
`runs/m2/pro-v2-mhca-k16-tse/checkpoints/ckpt-00002000.pt` (Pro v3). É o modelo **completo**: DFNet3 base do
checkpoint + FiLM (enc e df_dec) + `FrequencyDWConvMixer` + `CausalSubbandCrossAttention` (16 tokens x 128), com a
mesma conta de `PDFNet3Pro.forward`.

## Quadro e DSP

- 48 kHz, hop 480 amostras (10 ms), FFT 960, 32 bandas ERB, 96 bins de DF, ordem DF 5; `config.ini` = cópia do
  DFNet3 v0.5.6 aprovado. Features (`feat_erb`, `feat_spec`), anéis de espectro e síntese iguais ao stateful atual.
- Cada grafo processa **um quadro por chamada** (S=1). Formas fixas e explícitas no grafo.

## Condicionamento

| Entrada | Forma | Significado |
|---|---|---|
| `speaker_embedding` | `[1,192]` f32 | ECAPA-TDNN L2-normalizado (saída `embedding` do enrollment); zeros sem perfil |
| `speaker_present` | `[1]` f32 | 1.0 com perfil, 0.0 sem perfil |

Dentro dos grafos: FiLM (`film.*`), tokens = `cross_attn.fallback_proj(e).view(16,128)` (é o que o treino usou: o
checkpoint não tem pesos de `LearnedSpeakerAttentionPooler`), K/V, atenção e ganhos de sub-banda. Com
`speaker_present`=0 o FiLM é identidade e os deltas da atenção e do mixer são zero: o grafo vira o DFNet3 base do
checkpoint. O runtime **não** alimenta gamma/beta.

## Grafos, E/S e estados

Toda porta de estado é par `<nome>_in` / `<nome>_out`, float32, mesma forma, estado inicial = zeros; o runtime
realimenta os pares pelo nome.

| Grafo | Entradas | Saídas | Estados |
|---|---|---|---|
| `enc.onnx` | `feat_erb [1,1,1,32]`, `feat_spec [1,2,1,96]`, `speaker_embedding`, `speaker_present` | `e0 [1,64,1,32]`, `e1 [1,64,1,16]`, `e2 [1,64,1,8]`, `e3 [1,64,1,8]`, `emb [1,1,512]`, `c0 [1,64,1,96]`, `lsnr [1,1,1]`, `erb_gain [1,1,1,32]` | `enc_h [1,1,256]`, `feat_erb_buf [1,1,2,32]`, `feat_spec_buf [1,2,2,96]` |
| `erb_dec.onnx` | `emb`, `e3`, `e2`, `e1`, `e0`, `erb_gain` (do enc) | `m [1,1,1,32]` (já x `erb_gain`) | `erb_h [2,1,256]` |
| `df_dec.onnx` | `emb`, `c0` (do enc), `speaker_embedding`, `speaker_present` | `coefs [1,1,96,10]` (= `[96, 5, re/im]`) | `df_h [2,1,256]`, `c0_buf [1,64,4,96]` |

Roteamento: cada entrada de decoder com o mesmo nome de uma saída do enc recebe essa saída (`routing` no
manifest). `emb` é o embedding já condicionado pela cross-attention; `lsnr` sai do embedding antes dela (como no
treino). A cross-attention é por quadro (não tem cache temporal).

## Execução (decisão de 2026-11-09)

Os três grafos rodam em **todo** quadro ativo (estados sempre avançam, fiel ao forward de treino). O treino calcula as
perdas sobre a saída **selecionada por estágio de lsnr** (`_gated` → `emulate_gating_offline`: lsnr < -10 → zero;
> 30 → ruidoso; (20, 30] → só máscara ERB; senão ERB+DF), e `gate_loss` treina o estágio zero para o interferente.
Manter essa seleção de saída no runtime é o comportamento treinado. Atalho de silêncio digital (média quadrática
< 1e-7) como hoje.

## Asset

`pdfnet3-pro-v3-full-stateful-asset-v1.tar.gz` (tar determinístico): `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`,
`config.ini`, `manifest.json`. O `manifest.json` traz `model`, `hop`, `sample_rate`, `graphs.<g>.{inputs,outputs,
state_pairs}` com formas, `routing`, e `files.<f>.sha256`.

Opset 12 (o mesmo dos grafos aprovados). Ops: Conv, ConvTranspose, GRU, Einsum, Gemm, MatMul, Softmax, Erf (GELU),
Tanh, Exp, Sigmoid, Relu, e ops de forma. Validado em ONNX Runtime 1.30, OpenVINO 2026.4 (CPU) e tract 0.19.16
(`tools/tract-check stream`), com `into_optimized`.

## Gate de paridade

`scripts/gate_pro_v3_parity.py`: `PDFNet3Pro.forward` (sem `pad_feat`, decoders em todo quadro) x grafos hop a hop,
em ORT, OpenVINO CPU e tract; perfis sem perfil / locutor-alvo / outro locutor; SNR e máx|erro| de `m`, `coefs`,
`lsnr` e do áudio final (DSP da réplica de `DfTract::process` com seleção de estágio). Meta >= 60 dB em fp32.
