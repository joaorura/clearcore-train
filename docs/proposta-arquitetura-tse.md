# Proposta de arquitetura para o isolamento da voz do usuário (TSE)

Documento de proposta (não é plano aprovado nem resultado). Data: 14/11/2026. Base lida: `docs/pro-v5-processo.md`,
notas do vault de 13 e 14/11/2026, `docs/plan-pro-v4.md`, `docs/adapter-export.md`, `docs/export-contract-pro-v3.md`,
o código em `train/pro-v5-keep` (`models/pdfnet3_pro.py`, `cross_attention.py`, `adapters.py`, `export/pdfnet3_pro_v3.py`,
`third_party/DeepFilterNet/.../deepfilternet3.py`), o runtime (`vendor/crates/deep_filter/src/tract.rs`) e os artefatos
do experimento de teto (`clearcore-train/runs/m4/ceiling-dev/report.txt`, `ecapa-cost.json`, `ceiling-heads/train.log`).

Convenção: **[medido]** = número nosso, com a fonte; **[código]** = lido no código, não medido; **[hipótese]** =
inferência minha, não testada. Nada aqui foi treinado nem executado; os custos são contas à mão.

## 1. Diagnóstico arquitetural

### 1.1 O que os dados dizem

1. **Todas as receitas ficam sobre a mesma curva de troca.** [medido, `pro-v5-processo.md` §3] Corte do interferente
   (C8) contra preservação do alvo (C1/C6): 61% com C6 −3,5 dB (job 141 @300k); 73% com −6,75 (141 @900k); 71% com
   −5,6 (144); 74% com −5,4 (145); 96% com −18 (132); e o v3.1, com todos os pesos livres, 91–95% com −14,5. Pesos de
   perda, gate corrigido, continuação com corte forte e fine-tuning completo mudam o ponto, não a curva.
2. **A informação de "quem está falando" não está na representação do trunk.** [medido, `ceiling-dev/report.txt`] Com o
   trunk do v3.1 congelado, a cabeça de presença sobre `emb_conditioned` dá AUC 0,807 com GRU de 32, 0,807 com 2×128 e
   0,806 com 2×128 mais contexto causal de ~1,3 s. Cabeça maior e mais contexto **não mudaram nada**. O `lsnr` do v3.1
   dá 0,799; os micro fine-tunes com gradiente no trunk (v4-presence A/B/C), 0,806–0,812.
3. **ECAPA em janela deslizante acrescenta informação, mas pouca e cara.** [medido] Cosseno ECAPA de 1 s sozinho: AUC
   0,785–0,797; com GRU de 16, 0,817; combinado com o trunk, **0,862** (0,886 com rótulo `rel20`). No ponto de 1,9% de
   alvo zerado, isso zera só 27,8% do só-interferente. Custo: 19,1 GMAC/s (janela de 1 s a cada 100 ms) contra ~0,5
   GMAC/s do modelo, ou seja ~38×.
4. **A recuperação depois do interferente leva 1–2 s.** [medido, v3.1] Ganho do alvo sozinho: −9,6 / −9,7 / −6,8 dB em
   0–0,5 / 0,5–1 / 1–2 s após o interferente (VAD pico−45), contra −3,6 dB antes de qualquer interferente. Vem da
   máscara (97% dos quadros em ERB+DF), não do gate.
5. **Quando o gate depende de cabeça congelada, o erro não se corrige.** [medido] ~14–15,6% do alvo zerado no P3 até o
   `lsnr_corr` (`d40e517`), depois ~1%.
6. **O trunk pequeno consegue cortar.** [medido] v3.1 e job 132 chegam a 91–96% de C8, e o só-interferente cai 38–48
   dB no v3.1 (`plan-pro-v4.md` §13). O que falta não é força de supressão.

Nota sobre o item 3: o resumo da tarefa fala em "teto 0,862 mesmo com o trunk ajustado". No relatório que encontrei,
0,862 é trunk **congelado** + ECAPA de 1 s; com o trunk ajustado (500 passos) o AUC ficou em 0,81.

### 1.2 O que o código diz

1. **O condicionamento entra tarde e em três pontos.** [código] FiLM na entrada do `enc.emb_gru` e do `df_dec.df_gru`,
   e a cross-attention **depois** da GRU do encoder. As convoluções do encoder e os skips `e0..e3` (para o decoder ERB)
   e `c0` (para o decoder DF) não veem o falante. A LoRA r16 do encoder é um delta fixo ligado pelo perfil, igual para
   qualquer falante.
2. **"16 tokens" é um vetor só.** [código, `plan-pro-v4.md` §7] `tokens = fallback_proj(e).view(16,128)`: função afim
   do ECAPA de 192 dimensões, posto ≤ 193. O `LearnedSpeakerAttentionPooler` existe em `speaker_memory.py`, mas nenhum
   checkpoint o treinou (`export-contract-pro-v3.md`). A atenção sobre chaves constantes não mede semelhança entre o
   quadro e a voz cadastrada: é uma não linearidade estática. Ganho medido da atenção sobre o FiLM: +0,2 a +0,5 dB.
3. **A consulta da atenção já passou pela GRU.** [código] O quadro é comparado com o perfil depois de misturado ao
   estado recorrente, que é justamente o que o item 1.1.4 mostra estar contaminado.
4. **Os ganhos finos de sub-banda quase não chegam à saída.** [código, treino e runtime] De 0 a 4,8 kHz a saída é o
   filtro DF aplicado ao espectro **ruidoso**; a máscara ERB só vale acima disso (`spec_e[..., nb_df:] = spec_m[...]`,
   e em `tract.rs`: "This spectrum will only be used for the upper frequencies"). O `subband_gain_head` atua em
   300–4000 Hz, com limite de ±6 dB (`gain_bound` 0,5). No estágio ERB+DF (97% dos quadros) ele não altera nada.
   Toda a rejeição do interferente na faixa da voz passa pelos coeficientes do `df_dec`.
5. **Cálculo constante refeito a cada quadro.** [código, não perfilado] O export calcula dentro dos grafos, por quadro,
   coisas que só dependem do perfil: gerador FiLM no `enc` e no `df_dec` (2 × ~311 k MACs), `fallback_proj` (~393 k) e
   `bio_to_kv` (~524 k). São ~1,5 M MACs/hop; o ONNX Runtime não dobra constante que vem de entrada. O `precompute_kv`
   já existe e não é usado no export.

### 1.3 Leitura

- **[hipótese principal] O teto é de discriminabilidade, não de capacidade.** Uma curva de troca fixa que só desliza
  com a receita é o que se espera de um detector com AUC ~0,8: mudar a perda muda o limiar. Os itens 1.1.2 e 1.1.6
  apoiam; a nota do vault atribui o teto à "capacidade das camadas novas", o que os itens 1.1.1 (o v3.1, sem
  restrição de capacidade, está na mesma curva) e 1.1.2 contrariam. Não está provado: o teste do oráculo (seção 3.1)
  separa as duas.
- **[hipótese] O ECAPA é a representação errada para a decisão por quadro.** Ele é treinado para ser invariante ao
  conteúdo fonético e resume a voz num ponto. O modelo precisa aprender, com 972 falantes de banco, a traduzir esse
  ponto em "como este falante soa neste fonema" no espaço ERB/DF. "Pegar mais espaço da voz" = guardar a voz
  cadastrada como **conjunto de quadros no espaço de features do próprio modelo** e comparar quadro com quadro.
- **[hipótese] Contexto temporal maior, sozinho, não resolve.** O item 1.1.2 testou 1,3 s de contexto sobre o trunk e
  não ganhou nada. Contexto só ajuda sobre features que carregam identidade.
- **[hipótese] A demora de 1–2 s é o tempo de integração de evidência fraca.** Com evidência por quadro mais forte e
  sem memória (casamento com o cadastro antes da GRU), a reabertura deixa de depender do estado.
- **[hipótese] Resolução espectral não é o gargalo de C6/C8.** Os dois critérios medem trechos sem sobreposição (só
  interferente, só alvo). A janela de 20 ms mal resolve harmônicos de vozes graves e isso limita a separação em
  sobreposição, mas não explica o que falha hoje.

## 2. Propostas

Custos por hop de 10 ms. Referência: ~5 M MACs/hop (0,5 GMAC/s) para o Pro v3 [`plan-pro-v4.md` §3.3; não recontei].
O artigo do DeepFilterNet3 cita ~0,34 GMAC/s (de memória); a diferença pode ser o item 1.2.5 ou o método de contagem.
A latência algorítmica (janela de 20 ms + 2 quadros de lookahead do `config.ini`) não muda em nenhuma das três.

### 2.1 Proposta A (1º): cadastro nativo multi-token com casamento por quadro antes da GRU

```
CADASTRO (uma vez, fora do tempo real)
 áudio do cadastro -> STFT/ERB/DF -> convs do encoder do DFNet3 publicado (congeladas)
   -> x^e_t [512] por quadro ativo -> projeção + pooling -> K=48 tokens [48,128]  -> K/V pré-calculados
                                    -> média/desvio (+ ECAPA 192) -> g [256] -> gammas/betas de todos os FiLM
 perfil = { K/V [4,48,32] x2, FiLM pronto, ECAPA }            (~60 kB)

TEMPO REAL (por quadro)
 feat_erb, feat_spec
   -> convs do encoder, com FiLM(g) por canal em cada conv          (skips e0..e3 e c0 passam a ver o falante)
   -> x_t [512]  (antes da GRU)
   -> q = Wq x_t ; atenção sobre os 48 tokens -> contexto c_t [128] + estatísticas do casamento s_t [8]
   -> x'_t = x_t + Wo c_t                                           (zero-init: passo 0 = modelo base)
   -> cabeça de turno: GRU(48) sobre [s_t, P x'_t] -> p_alvo, p_interf
   -> enc.emb_gru( x'_t ) -> emb
   -> erb_dec(emb, skips | p)  e  df_dec(emb, c0 | p)               (p entra por FiLM na saída das GRUs)
   -> gate: p_alvo substitui o lsnr no estágio ZERO quando há perfil
```

- **Onde o falante condiciona**: (i) cedo, em todas as convs do encoder (FiLM por canal, constante por sessão);
  (ii) no casamento quadro × cadastro **antes** da GRU, que dá evidência de identidade sem memória; (iii) nos
  decoders, por `p_alvo`/`p_interf`. O ECAPA fica só como entrada auxiliar de `g` (aquecimento) e pode sair depois.
- **Custo**: casamento ~145 k MACs (Wq 65 k, atenção 12 k, Wo 65 k), praticamente o que a cross-attention atual já
  gasta (~137 k); FiLM cedo ~13 k; cabeça de turno ~20 k. Total **+~45 k MACs/hop (~+1%)**. Movendo o cálculo
  constante para o cadastro (item 1.2.5) saem ~1,5 M MACs/hop: o grafo final fica **mais barato que o v3**
  [a confirmar com perfil do ORT]. Parâmetros: +~0,25 M. Memória de perfil: ~60 kB.
- **Export e runtime (contrato v3)**: o `enrollment.onnx` passa a devolver `speaker_kv` e `speaker_film`, e precisa
  rodar as convs do encoder sobre o cadastro (grafo novo `enroll_enc.onnx`, mesmos pesos do DFNet3 publicado). O
  `enc.onnx` troca `speaker_embedding` por `speaker_kv` + `speaker_film`, ganha a saída `p_target` e o estado `p_h`
  (ambos já previstos no contrato v4-presence, `gating.mode = "presence"`). `erb_dec` e `df_dec` recebem `p_target`.
  Ops: MatMul, Softmax, Gemm, GRU, já usados e validados em ORT, OpenVINO e tract. O perfil salvo muda de formato:
  recalcular a partir do áudio do cadastro, se o app o guarda [não verifiquei], ou pedir novo cadastro.
- **Pesos aproveitados**: todo o DFNet3 publicado; `q_proj`/`out_proj` da atenção do v3.1 como partida opcional;
  módulos novos em zero-init, com os mesmos testes de identidade no passo 0 (`tests/test_adapters_v5.py`).
- **Treino em duas etapas**: A1, trunk congelado + LoRA (receita do job 141), para comparar direto com o teto atual;
  A2, fine-tuning completo a partir do melhor A1, com LR do trunk ~10× menor que a dos módulos novos. Para o A2
  proponho **dois assets**: sem perfil o app usa os grafos do DFNet3 publicado; com perfil, o modelo TSE. A garantia
  "sem perfil = DFNet3" passa a valer por construção e o treino TSE deixa de carregar essa restrição.
- **Treino, custo extra**: o cadastro vira sequência de quadros. Com o encoder do cadastro congelado, dá para
  pré-calcular um banco (ex.: 64 quadros ativos × 512 por variante, 16.602 variantes, ~1,1 GB em fp16) e mantê-lo na
  VRAM, sem custo por passo.
- **Riscos**: (1) as features das convs do DFNet3 foram treinadas para ruído e podem não separar falantes; a sonda T0
  mede isso em horas. (2) Em sobreposição o quadro da mistura casa mal com o cadastro limpo. (3) 972 falantes de
  alvo podem ser poucos para um casamento que generalize; o plano v4 já previa +625. (4) A deriva do fine-tuning
  completo (2–4 k passos) não teve a causa identificada; o A2 pode repeti-la.
- **Literatura**: SEF-Net e USEF-TSE (cross-attention entre mistura e cadastro, sem embedding) [busca]; Pärnamaa e
  Saabas, PSE sem modelo de embedding separado, usando as features internas do próprio modelo [busca]; SpEx+
  (encoder compartilhado entre mistura e cadastro), SpeakerBeam (rede auxiliar treinada em conjunto), CIENet
  (atenção TF com o cadastro), pDeepFilterNet2 (treina tudo; citado no vault) [de memória].

### 2.2 Proposta B (2º): ramo "quem fala" separado, em taxa reduzida, com o DFNet3 congelado

```
 feat_erb + log|DF 96 bins|  --a cada 2 quadros (20 ms)-->  convs separáveis -> casamento com os tokens do cadastro
   -> GRU(64), memória de ~2 s -> p_alvo, p_interf, vetor de turno u_t [32]        (grafo novo: spk.onnx)

 DFNet3 congelado + adaptadores P3  <- u_t por FiLM na saída das GRUs (`--gru-film` já existe em adapters.py)
 ganho suave por p_alvo sobre os coeficientes DF e a máscara ERB, ataque rápido (alvo) e soltura lenta
 gate ZERO pelo p_alvo
```

- **Onde condiciona**: fora do trunk. O ramo próprio decide o turno; o trunk recebe a decisão pronta.
- **Custo**: ~0,3 M MACs por execução, a cada 2 hops = **~0,15 M MACs/hop (~+3%)**; +~0,15 M parâmetros.
- **Export e runtime**: aditivo. Quarto grafo `spk.onnx` com estado próprio; `enc`/`erb_dec`/`df_dec` ganham a entrada
  `turn`. Mantém "sem perfil = DFNet3 bit a bit" e o desenho do `adapter-export.md`. O runtime passa a rodar um grafo
  em cadência diferente (um contador de hop).
- **Pesos**: DFNet3 publicado e os adaptadores do job 141 (`--init-from`).
- **Riscos**: herda o que o trunk congelado não conseguir fazer em sobreposição; o ganho por `p` é por quadro, não por
  bin. Como classificador de presença, parte do mesmo problema do item 1.1.3: só funciona se o casamento nativo for
  bem melhor que o ECAPA deslizante.
- **Quando escolher**: se o oráculo (3.1) passar com o trunk congelado. Aí B é a forma mais barata de entregar.
- **Literatura**: Personal VAD (detector de atividade do falante-alvo, pequeno e em fluxo); Delcroix et al., "Listen
  only to me!" (falso alarme com alvo inativo, verificação à parte) [busca]; VoiceFilter-Lite (perda assimétrica e
  força de supressão adaptativa contra a supersupressão) [busca; detalhes de memória]; E3Net (multitarefa reduz a
  supersupressão do alvo) [busca]; TSOS de Eskimez et al. [de memória].

### 2.3 Proposta C (3º): professor grande + destilação, com banco de análise de resolução dupla

```
 PROFESSOR (só na rig): modelo TSE causal grande (dezenas de M de parâmetros, mesma latência), treinado no mix_v4
 ALUNO = Proposta A, mais um segundo banco só de análise: janela de 40 ms voltada para o passado, hop de 10 ms
         (resolução de ~25 Hz para o pitch; não soma latência porque não entra na síntese)
 perda = perdas atuais + destilação da saída e da presença do professor
```

- **Onde condiciona**: como em A; o segundo banco alimenta só o casamento e a cabeça de turno.
- **Custo**: inferência +10–20% (FFT de 1920 por hop + convs de entrada); treino: semanas de rig para o professor.
- **Export e runtime**: como A, mais uma FFT de análise e um anel de entrada no DSP em Rust (`dsp_pipeline.rs`).
- **Risco**: o mais alto. Nada nos dados indica que resolução ou capacidade seja o gargalo de C6/C8 (seção 1.3).
- **Quando escolher**: se A passar C8 mas a separação **em sobreposição** (C7, ESTOI) estacionar.
- **Literatura**: E3Net (destilação professor–aluno em PSE, aluno até ~3× mais rápido) [busca]; TEA-PSE 1–3 (dois
  estágios, ~22 M de parâmetros, vencedor do DNS) [busca]; X-TF-GridNet, pDCCRN [de memória].

### 2.4 Independente da proposta

- Tirar dos grafos o cálculo que só depende do perfil (item 1.2.5): ~1,5 M MACs/hop a menos, sem mudar a saída.
- Decidir o destino do `subband_gain_head` (item 1.2.4): removê-lo ou levá-lo para os coeficientes DF.

## 3. Plano de experimentos

Régua: avaliador rápido (9 critérios) a cada 100 k amostras, seleção `c8c1gate`; avaliador completo só no vencedor.
Fronteira a bater [medido]: C8 61% @ C6 −3,5 dB; 73% @ −6,75; 96% @ −18. Vazão: ~80–105 amostras/s por GPU, 2 GPUs =
1,55× (500 k amostras em ~1 h com 2 GPUs). Erro-padrão do C1 de 1,3–1,6 dB: diferença menor que isso não decide.

### 3.1 Dia 1: teste de teto (decide em menos de um dia)

Dois testes em paralelo, 6 GPUs.

**Oráculo de presença (3 braços × 2 GPUs, 500 k amostras, ~1 h cada).** Receita do job 141, mais duas entradas por
quadro com a atividade real do alvo e do interferente (dos stems, VAD pico−45), somadas antes da GRU do encoder e por
FiLM nos decoders, zero-init. No avaliador, a mesma atividade sai dos stems que o método da sombra já usa.

| Braço | Entrada | Trunk |
|---|---|---|
| O1 | oráculo limpo | congelado (P3) |
| O2 | oráculo atrasado 150 ms, com 10% de erro em rajadas | congelado (P3) |
| O3 | oráculo limpo | GRUs livres, LR 10× menor |

Passa = algum snapshot com C8 ≥ 85%, C6 ≥ −3,5 dB, C1 ≥ −1 dB e R0 PASS.

| Resultado | Conclusão | Próximo passo |
|---|---|---|
| O1 passa | o gargalo é saber quem fala; o trunk congelado basta | A1 ou B |
| O1 falha, O3 passa | faltam as duas coisas | A direto na etapa A2 |
| O1 e O3 falham | não é arquitetura: perdas, dados ou régua | parar e rever a receita de perdas |
| O2 perto de O1 | tolera detector imperfeito | cabeça de turno simples |
| O2 bem abaixo de O1 | exige evidência quase instantânea | prioriza o casamento antes da GRU |

**Sonda T0 (1 GPU, ~30 min por rodada).** Reaproveita `scripts/ceiling_*`: mesmo cache de dev, mesmas cabeças, mais
variantes com casamento entre as features pré-GRU (convs congeladas) e o banco de quadros do cadastro (K=16 e 48, com
e sem ECAPA). Bases: 0,807 (trunk) e 0,862 (trunk + ECAPA de 1 s, 38×). Passa = AUC ≥ 0,90 no rótulo padrão **e**
≥ 50% do só-interferente zerado com ≤ 2% do alvo (hoje 27,8%).

> Atualização (14/11): medido, o melhor casamento implantável (só cadastro fixo) dá AUC ~0,88 (erro ~21%); os 0,897/0,916
> vieram de ECAPA deslizante na mistura (≈ 0,77 s de CPU por s de áudio), não implantável. O Personal VAD 2.0 aberto
> deu AUC 0,65–0,68 (erro 36–40%). Detalhes em `pro-v5-processo.md` §9.4. Medir também o AUC nos 200 ms após a troca de falante.

Esforço de código: entradas do oráculo no treino e no avaliador, banco de quadros do cadastro e as cabeças novas.
É apertado para um dia; se faltar tempo, o oráculo tem prioridade.

### 3.2 Dias 2–3: A1 de ponta a ponta

Três braços × 2 GPUs, 1 M amostras (~2 h): casamento K=16; K=48; K=48 + FiLM cedo. Trunk congelado + LoRA, receita do
141. Critério: fronteira deslocada além do ruído (C8 ≥ 85% com C6 ≥ −3,5 e C1 ≥ −1), confirmada no avaliador completo
e, se o C1 ficar a menos de 1,5 dB do limite, numa segunda semente. Olhar o C6 estratificado por tempo desde o
interferente: a meta é sumir a diferença entre "antes" e "0–0,5 s depois".

### 3.3 Dias 4–6: A2 e fechamento

Fine-tuning completo a partir do melhor A1, 2 M amostras em 4–6 GPUs, LR efetiva 3e-4 nos módulos novos e 3e-5 no
trunk, snapshots a cada 100 k, parada se dois snapshots seguidos piorarem C1 e C6. Meta: 9/9 no rápido e 11/11 no
completo. Depois: export (contrato novo), paridade ≥ 60 dB em ORT/OpenVINO/tract e medição de ms/hop na CPU de
referência. Pré-requisito já pendente: integrar `perf/v5-gpupool`.

## 4. Recomendação

1. Não lançar mais receitas de perda sobre a arquitetura atual: os dados mostram uma curva de troca fixa.
2. Rodar primeiro o oráculo de presença e a sonda T0 (um dia): eles dizem se o teto é de discriminabilidade ou de
   capacidade, e se o casamento nativo supera o ECAPA deslizante.
3. Se confirmarem, implementar a Proposta A: cadastro como 48 tokens no espaço do próprio modelo, casamento antes da
   GRU, FiLM cedo e cabeça de turno; custo de ~+1% em MACs, e menor que o v3 após tirar o cálculo constante dos grafos.
4. Fine-tuning completo só na etapa A2, com dois assets (DFNet3 publicado sem perfil, TSE com perfil); B é a saída
   barata se o trunk congelado bastar, C fica para um eventual teto em sobreposição.
5. Fazer já, em qualquer cenário: pré-calcular FiLM e K/V no cadastro e resolver o `subband_gain_head` inerte.

## 5. Fontes

Consultadas por busca na web em 14/11/2026 (só resumo e metadados; não conferi tabelas dos artigos):

- Pärnamaa e Saabas, "Personalized Speech Enhancement Without a Separate Speaker Embedding Model", Interspeech 2024:
  https://arxiv.org/abs/2406.09928
- Zeng e Li, "USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction" (estende o SEF-Net):
  https://arxiv.org/abs/2409.02615
- Wang et al., "VoiceFilter-Lite", Interspeech 2020: https://google.github.io/speaker-id/publications/VoiceFilter-Lite
- Delcroix et al., "Listen only to me! How well can target speech extraction handle false alarms?":
  https://arxiv.org/abs/2204.04811
- Zhao et al., "Target Confusion in End-to-end Speaker Extraction": https://arxiv.org/abs/2204.01355
- Thakker et al., "Fast Real-time Personalized Speech Enhancement: E3Net and Knowledge Distillation", Interspeech
  2022: https://arxiv.org/abs/2204.00771
- Ju et al., "TEA-PSE 3.0": https://arxiv.org/abs/2303.07704

De memória, sem conferência: SpEx+ (Ge et al., 2020), SpeakerBeam (Žmolíková et al., 2019), CIENet (2024),
X-TF-GridNet (Hao et al.), pDCCRN e a métrica TSOS (Eskimez et al., 2022), Personal VAD (Ding et al., 2020),
pDeepFilterNet2, e os ~0,34 GMAC/s do DeepFilterNet3.
