# Export dos adaptadores do pro-v5 (proposta, não implementado)

Modelo: DFNet3 publicado congelado + camadas do Pro (FiLM, `freq_mixer`, cross-attention) + adaptadores
(`src/cctrain/models/adapters.py`). Os adaptadores ficam como **nós separados** no grafo, nunca fundidos nos pesos da
base, e são multiplicados pela flag de perfil `speaker_present` (g ∈ {0, 1}), que já é entrada do contrato v3.

## Como cada adaptador vira nós ONNX (S = 1, um quadro por chamada)

- **LoRA em conv 1×1 do encoder** (`enc.erb_conv{1,2,3}.1`, `enc.df_conv{0,1}.1/.2`) e no `enc.df_fc_emb`:
  `y = Conv(x; W) + g·s·Conv(Conv(x; A); B)` (ou `MatMul` para o linear). Dois Conv/MatMul de posto r e um `Mul`
  por `speaker_present` antes do `Add`, inseridos entre a conv pontual e a BatchNorm já fundida da base.
- **LoRA na GRU** (`enc.emb_gru.gru`, `erb_dec.emb_gru.gru`, `df_dec.df_gru.gru`): o op `GRU` do ONNX não aceita
  delta de peso. Com S = 1 a GRU vira uma célula explícita (a mesma conta do `GRU` com `linear_before_reset=1`):
  `gx = W_ih x + b_ih + g·s·B_ih(A_ih x)`, `gh = W_hh h + b_hh + g·s·B_hh(A_hh h)`, r/z/n e `h' = (1−z)⊙n + z⊙h`.
  São ~12 nós por camada (MatMul, Add, Sigmoid, Tanh, Mul, Split); a base continua nos mesmos pesos `W_*`.
  Alternativa sem reescrever a GRU: dois ramos `GRU` (pesos base e pesos somados W + sBA) e `Where(speaker_present)`
  na saída e no estado. Dobra o custo da GRU com perfil e funde os pesos dentro do ramo ligado, então não é a opção
  preferida.
- **FiLM na saída da GRU** (`--gru-film`): `y' = y ⊙ (1 + g·γ) + g·β`, com `[γ, β] = Gemm(speaker_embedding)`.
  `γ, β` dependem só do perfil: podem ser calculados no `enrollment.onnx` e entrar como constante por sessão, ou
  ficar dentro do grafo (1 Gemm por quadro, barato).

Com `speaker_present = 0` todos os deltas são zero e o grafo calcula exatamente o DFNet3 publicado (o treino garante
isso: teste `tests/test_adapters_v5.py::test_a_*`).

## O que muda no contrato v3 do runtime (Rust)

| Grafo | Entradas | Saídas / estados | Mudança |
|---|---|---|---|
| `enc.onnx` | `feat_erb`, `feat_spec`, `speaker_embedding`, `speaker_present` | `e0..e3`, `emb`, `c0`, `lsnr`, `erb_gain`; `enc_h [1,1,256]` | nenhuma de interface; nós novos internos (LoRA nas convs, `df_fc_emb` e GRU). Pesos da base = DFNet3 publicado |
| `erb_dec.onnx` | `emb`, `e3`, `e2`, `e1`, `e0`, `erb_gain` | `m`; `erb_h [2,1,256]` | **passa a receber `speaker_present`** (e `speaker_embedding` se `--gru-film`), porque a GRU do decoder ERB agora tem LoRA |
| `df_dec.onnx` | `emb`, `c0`, `speaker_embedding`, `speaker_present` | `coefs`; `df_h [2,1,256]`, `c0_buf` | nenhuma de interface (já recebe as duas entradas) |

Os pares de estado não mudam de nome nem de forma (a célula explícita usa o mesmo `h`). O manifesto ganha a versão do
contrato (`v3.1-adapter`), a lista de entradas do `erb_dec` com `speaker_present` e o `routing` dela. O runtime precisa:
(1) alimentar `speaker_present`/`speaker_embedding` também no `erb_dec`; (2) aceitar os pesos base = DFNet3 publicado
(o mesmo `config.ini`). Gate de paridade: o `gate_pro_v3_parity.py` com os adaptadores, incluindo o caso sem perfil
contra o DFNet3 publicado (meta ≥ 60 dB fp32).
