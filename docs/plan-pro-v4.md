# Plano do pDFNet3 Pro v4: extração de locutor-alvo com N interferentes

Data: 2026-11-10. Status: **desenho**. Nada foi treinado; os números "medidos" abaixo saíram de logs, manifests e
inspeções feitas nesta data. Os números "estimados" estão marcados como estimativa.

## 0. Resumo das decisões

| Tema | Decisão | Motivo |
|---|---|---|
| Arquitetura | Mantém a do v3 (FiLM + `FrequencyDWConvMixer` + `CausalSubbandCrossAttention` K=16, D=128). Há um braço A/B opcional: `lsnr` lido **depois** da cross-attention (§7) | As falhas observadas vêm da função objetivo, dos dados e do gating, não de falta de capacidade (§7). Assim o contrato do runtime continua o mesmo |
| Ponto de partida | `runs/m2/pro-v2-mhca-k16-tse/checkpoints/ckpt-00002000.pt` (v3) | É o v2 75000 com os tokens K=16 já adaptados, a mesma forma de tensores do contrato v3. O v2 75000 tem `fallback_proj` 192→1024 (K=8), incompatível com o contrato v3 |
| Gating (item 2) | **As duas medidas juntas**: (a) no runtime, com `speaker_present=1`, os estágios passam a ser só ZERO ou ERB+DF, sem passthrough nem ERB-only; (b) o `lsnr` passa a ser supervisionado como "SNR do alvo contra todo o resto", também nas amostras só-interferente, com hinge nos dois sentidos | (a) elimina de forma determinística e sem custo o vazamento por passthrough. (b) faz o estágio ZERO disparar no só-interferente (hoje dispara em 50–58% dos quadros) |
| ArcFace (item 3) | **Remover.** Ablação opcional com um ECAPA congelado aplicado à saída | Bug confirmado: a perda não depende de nenhum parâmetro (§4.1) |
| `target_absence_loss` | **Corrigir.** A máscara passa a ser "interferente ativo E alvo inativo", a perda fica invariante a nível e recebe peso real | Hoje usa `interferer_frames`, que inclui a sobreposição, e vale ~3e-6, ou seja, não age (§4.1) |
| Dados | Gerador novo `mix_v4`: 1 alvo + N∈{0..4} interferentes, SIR por interferente de −15 a +30 dB, nível do alvo de −50 a −12 dBFS, ruído de "nenhum" a −5 dB, RIR por falante, turnos e sobreposição parcial, negativos difíceis | Pedido do usuário. O gerador atual tem N≤2, SIR de −5 a 25 dB global, nível fixo de −25 dBFS no só-interferente e nenhum caso silencioso |
| Orçamento | 120k passos (~12 h de GPU, estimativa), mais 20k do braço B (~2 h) e ~1,5 h de avaliação | Medido: 0,251 s/passo de parede no v2. O v4 custa ~0,35 s/passo (estimativa, §6) |
| Runtime do app | Muda só a **seleção de estágio** quando `speaker_present=1`. Grafos, E/S, opset e MACs não mudam | §3.3 |

## 1. Diagnóstico (o que o v4 ataca)

### 1.1 Validação v3 com perfil (`eval-tse`, 70 falantes de dev, IC95 bootstrap)

| Condição | v2_full | v3_full | v3 sem perfil |
|---|---|---|---|
| D: só alvo + ruído SNR 5, ΔSI-SDR / ΔSTOI | +3,72 / −0,035 | +3,58 / −0,041 | **+7,38 / +0,059** |
| A: SIR 0, ΔSI-SDR / supressão do interferente / ganho do alvo | −0,55 / 19,6 / **−11,6 dB** | −1,11 / 20,1 / **−12,1 dB** | −0,23 / 0,8 / −0,8 |
| C: só interferente, sem ruído: atenuação / % ZERO | 27,6 dB / 54,6% | 28,3 dB / 58,1% | 0,2 / 5,3% |
| C: só interferente, com ruído | 18,5 dB / 49,9% | 17,1 dB / 49,5% | 2,5 / 15,8% |

Leitura:

- O perfil não **extrai**: ele **atenua tudo**. O alvo perde 10–12 dB e o interferente 18–20 dB, o que dá um
  diferencial de só ~8 dB. Com o perfil trocado (o do interferente), o alvo cai 19 dB e o interferente 10 dB.
  O modelo usa o perfil, mas aprendeu que "perfil ligado = suprimir forte".
- No caso sem interferente, o perfil custa −3,8 dB de ΔSI-SDR e −0,10 de ΔSTOI. Nenhum termo do treino protege
  esse caso, e em ~63% das amostras com perfil há interferente.
- Só metade dos quadros só-interferente cai no estágio ZERO.

### 1.2 Causas encontradas no código (`main` @ 2a83c5d)

1. **O ArcFace é estruturalmente nulo.** Detalhes em §4.1.
2. **`target_absence_loss` é inócua e está mal mascarada.** `absence_mask = b["interferer_frames"] & mask`
   (`scripts/train_pro.py:143`), e `interferer_frames` é a atividade do interferente, venha ou não com o alvo
   (`stage_f.py:328-332`). Na sobreposição, a perda pede saída zero e briga com a reconstrução. Ela é a média de
   `est²` em amostras de forma de onda, e vale **3,4e-6 de média** no log do v2 (peso 0,5), contra `mr` ≈ 5–7.
   Não age.
3. **`gate_loss` é inócua.** Média de 9e-5 no v2, porque só pune `lsnr > 30`. Não empurra o `lsnr` para baixo
   de −10.
4. **O `lsnr` não é supervisionado no só-interferente.** `lsnr_mask = mask0 or has_target`
   (`stage_f.py:338`): a amostra `INTERFERER_ONLY` com perfil não treina o `lsnr`. O alvo do `lsnr`
   (`_lsnr_target`, `stage_c.py:255`) **já** trata o interferente como ruído quando há perfil, porque `spec_clean`
   é só o alvo. Falta a supervisão onde ela mais importa.
5. **A distribuição de treino é estreita.** N≤2 interferentes. SIR U[−5, 25] dB contra a soma dos interferentes,
   num só ganho. O só-interferente entra em −25 dBFS fixo (`mix.py: INTERFERER_ONLY_RMS_DBFS`). A mistura é
   normalizada ao pico, então o nível absoluto do alvo só varia pelo nível natural do arquivo. Não existe caso
   silencioso: a grade de SNR tem 40 dB como teto e o "sem ruído" só existe no só-interferente. Os interferentes
   compartilham uma RIR. A sobreposição é quase total: crop de 3 s de cada fala, sem turnos. Os interferentes são
   sorteados uniformemente, sem negativos difíceis.
6. **Passthrough.** Com `lsnr > 30`, a saída é a entrada ruidosa (`emulate_gating_offline`,
   `runtime_path.py:342-360`). Num cômodo silencioso com um interferente fraco (SIR de +20 a +30 dB), o `lsnr`
   do quadro é alto e o interferente passa inteiro. É o relato do usuário ("isolamento muito pequeno").
7. **O v2 está em platô.** O `mr` médio por janela de 7,5k passos ficou entre 6,5 e 6,9 do início ao fim dos 75k
   passos do v2. Mais passos com o mesmo objetivo e os mesmos dados não resolvem. O v3 tem só 2000 passos (LR de
   pico 3,15e-5).

## 2. Dados

### 2.1 Inventário medido (manifests + `pools`, 2026-11-10)

| Split | Falantes | Horas | Composição |
|---|---|---|---|
| treino atual | 973 | 121,8 | CV27-pt filtrado 603 / 39,4 h; VCTK 87 / 33,3 h; CML-TTS 30 / 21,7 h; LibriSpeech-100 251 / 16,6 h; TTS-PT 1 / 10,5 h; voz do dono 1 / 0,2 h |
| dev | 98 | 14,6 | CV 40, Libri dev 40, CML 10, VCTK 8 |
| teste | 124 | 18,1 | CV 60, Libri test 40, CML 10, VCTK 14 |

Outros recursos medidos:

- Banco de enrollment `bank-m2-sbecapa-v2`: 972 falantes, 16.602 variantes. Pipeline: concatenação de 6–12 s →
  RIR → ruído → denoise opcional → ECAPA. Gerá-lo levou 13.856 s de CPU (~14 s por falante). Tem proveniência, e
  `exclude_utterances` impede que o enrollment use a fala que está na mistura (sem vazamento).
- Ruído de treino: 12.580 arquivos, 29,9 h (mics locais, DNS, FSD50K). Babble sintético: 3.000 wavs.
- RIR: 256 sintéticas (v2), **uma fonte por sala**, RT60 de ~0,07 a ~0,95 s.
- `cv27-pt-full`: 3.269 `client_id`, 182 h. Fora de dev/teste e fora dos 603 já usados, sobram **1.511 falantes
  com ≥5 clipes (15,5 h)**, 625 deles com ≥8 clipes. O grosso das horas restantes (~120 h) são clipes adicionais
  dos 603 falantes que já usamos, porque o filtro limita a 100 clipes por cliente.
- Sexo declarado: 7.563 falantes públicos (M 3.897 / F 3.666) e 1.054 clientes do CV.

### 2.2 Ampliação para o v4

1. **Interferentes**: entram os 1.511 falantes novos do CV (≥5 clipes) mais os clipes extras dos 603. Interferente
   não precisa de enrollment, então não precisa de banco. Isso eleva o pool de interferentes de 973 para ~2.480
   falantes. Antes, a lista é filtrada contra os `client_id` de dev e teste (100 deles aparecem no `full`).
   Risco: o CV sem filtro tem `client_id` compartilhado. Para interferente o efeito é pequeno.
2. **Alvos**: os 972 do banco mais os 625 falantes novos com ≥8 clipes, filtrados por `cv_filter` com
   `min_clips=8`, porque o enrollment de 6 s e a fala da mistura precisam vir de clipes disjuntos. A extensão do
   banco custa ~625 × 14 s ≈ **2,5 h de CPU** e roda em paralelo à Fase 0.
3. **RIR v3 multi-fonte**: ~400 salas × 6 posições de fonte (≈2.400 RIRs, `gen_rir_v2.py` adaptado,
   pyroomacoustics). Alvo e interferentes da mesma mistura recebem posições diferentes **da mesma sala**. Custo de
   CPU de minutos.
4. **Centróides por falante** para os negativos difíceis: ECAPA de 1–3 clipes por falante novo (minutos de CPU) e
   o centróide do banco para os demais.

### 2.3 Gerador `mix_v4` (substitui `make_mixture` no v4; o M1–M3 continua reprodutível)

Segmento de **4 s** (hoje são 3 s), para caber turnos de 1–2 s. Cada amostra é uma função pura de
`(seed, época, índice)`, como hoje.

| Fator | Distribuição | Justificativa |
|---|---|---|
| Nº de interferentes N | P(0)=0,25, P(1)=0,35, P(2)=0,20, P(3)=0,12, P(4)=0,08 | Hoje ~37% das amostras são só alvo + ruído, e mesmo assim o perfil piora esse caso: o problema não é a fração, é não haver termo que proteja. O N=0 cai para 25% para abrir espaço a N=2–4, e toda amostra N=0 com perfil passa pelo `L_guard` (§4.2). N=1 é o caso real mais comum. N=2–4 atende ao pedido. Acima de 4 o som vira babble, e isso entra pelo ruído de babble sintético (classe já existente) |
| SIR por interferente (dB, contra o nível ativo do alvo) | mistura de faixas: 65% U[−10, +15], 15% U[−15, −10] (interferente mais alto), 20% U[+15, +30] (interferente fraco ou distante) | A faixa de +15 a +30 é a do passthrough relatado. A de −15 a −10 é a do vizinho falando alto. Cada interferente sorteia o seu SIR. A SIR total (alvo contra a soma) é registrada para a análise |
| Nível do alvo (RMS de fala ativa, dBFS) | U[−50, −12]; 10% em [−50, −40] ("sussurro/longe") | Varia o nível absoluto **depois** da mistura e só aplica limitador de pico se passar de −1 dBFS (não normaliza para cima). Limitação: não há corpus de sussurro de verdade. Fica aproximado por nível baixo mais 50% de inclinação espectral (−6 dB/oit acima de 1 kHz); é anotado como risco |
| Ruído de fundo | 20% **silencioso**: piso de microfone de −80 a −60 dBFS (rosa ou branco, ou mic local com SNR ≥ 50 dB). 30% SNR U[20, 40]. 35% U[5, 20]. 15% U[−5, 5]. O SNR é medido contra o alvo, ou contra os interferentes quando N≥1 e o alvo está ausente | O caso silencioso é o que mais aparece no relato do usuário e não existe no gerador atual. O piso de −80 a −60 evita o atalho de silêncio digital do runtime (média quadrática < 1e-7, −70 dBFS) e reproduz o mic real |
| Reverberação | p=0,5 por mistura. Todos os falantes na mesma sala (RIR v3), em posições distintas. Alvo da perda = **caminho direto + reflexões iniciais (50 ms)** alinhado | O alvo totalmente seco força a derreverberação, que é uma fonte de supressão agressiva do próprio alvo. Reflexões iniciais são a prática do DNS Challenge |
| Padrão temporal | 35% sobreposição contínua; 30% conversa (turnos de 0,5–2 s, sobreposição de 0–30% em cada troca); 15% blocos (só-alvo → ambos → só-interferente, em ordem aleatória); 20% esparso (cada trilha ligada ou desligada em janelas aleatórias, com silêncio comum) | Garante as quatro classes de quadro (só-alvo, só-interferente, ambos, silêncio). Meta por lote: ~35/20/30/15% dos quadros, com a proporção registrada no log a cada 20 passos |
| Alvo ausente ("perfil de quem não fala") | 8% das amostras: N≥1, alvo mudo, saída esperada zero, nível do interferente pela regra de nível do alvo (e não −25 dBFS fixo) | Treina o ZERO e a perda de vazamento em todos os níveis |
| Negativos difíceis | 35% dos interferentes saem dos 50 vizinhos mais próximos do alvo por cosseno de centróide; 50% desses do mesmo sexo declarado quando há essa informação. O resto é uniforme | Falantes parecidos são o caso em que o FiLM global confunde. O cosseno do ECAPA mede melhor que o sexo |
| Enrollment | Variante do banco nunca construída com a fala da mistura (`exclude_utterances`). `p_enroll_denoise` 0,5. RIR e ruído do enrollment independentes da mistura | Mantém a regra §4.4 (sem vazamento) e o descasamento de sessão do uso real |
| Sem perfil (`p_drop`) | 0,15, com alvo `clean_all` e destilação do DFNet3 (como hoje) | Preserva o caminho sem perfil, que vira o DFNet3 base no runtime |

Os rótulos por quadro de cada amostra são `target_active`, `interf_active` (OR dos interferentes) e `digital_silence`.
Eles vêm de VAD **relativo à trilha** já escalonada: nível a até 30 dB do máximo da própria trilha, com piso de
−75 dBFS. O `frame_activity` atual usa piso absoluto de −60 dBFS na trilha antes do ganho, o que erraria com alvo a
−50 dBFS.

## 3. Gating (item 2): análise e recomendação

### 3.1 Opção (a): mudar o gating no runtime quando `speaker_present=1`

- Proposta: com perfil, o estágio é ZERO se `lsnr < −10` e ERB+DF caso contrário. O passthrough (`> 30`) e o
  ERB-only (`(20, 30]`) somem. Sem perfil, tudo fica como hoje (comportamento DFNet3).
- A favor: elimina de forma determinística o vazamento no cômodo silencioso. Com perfil, nenhum quadro passa cru,
  e o modelo sempre aplica máscara e DF condicionados. Custo computacional zero, porque desde a decisão de
  2026-11-09 os três grafos já rodam em todo quadro. O ERB-only existia para poupar o DF em SNR alto, e essa
  economia já não existe.
- Contra: muda o runtime; isso é declarado no contrato v4 (§3.3). Em SNR muito alto, o DF passa a agir sempre
  sobre o alvo limpo, com risco de artefato. Isso é coberto pelo treino, que vai emular exatamente essa política,
  e pelo critério "dano no só-alvo" (§8).

### 3.2 Opção (b): o `lsnr` aprende a tratar outras vozes como ruído quando há perfil

- O alvo já tem essa semântica (`LocalSnrTarget(spec_clean=alvo, noise=spec−alvo)`). O que falta é supervisão no
  só-interferente e força nos hinges. Proposta:
  - `lsnr_mask` = todos os quadros, menos os de silêncio digital, também nas amostras com alvo ausente.
  - `L_zero = mean_{perfil ∧ ¬alvo ∧ interf} relu(lsnr + 12)²`: empurra para abaixo do limiar −10, com 2 dB de
    margem.
  - `L_keep = mean_{perfil ∧ alvo} relu(−8 − lsnr)²`: impede que o ZERO coma o alvo, sobretudo em nível baixo e
    nas sílabas fracas.
- A favor: o ZERO gera silêncio exato (vazamento −∞), sem depender só da máscara.
- Contra: sozinho, depende de o `lsnr` (lido antes da cross-attention, só com o FiLM) acertar quadro a quadro. Hoje
  acerta 50–58%. E não fecha o passthrough, porque qualquer erro para cima de 30 vaza o interferente inteiro.

### 3.3 Recomendação: (a) + (b)

(a) é a garantia barata contra o pior modo de falha (passthrough). (b) faz o ZERO cobrir o só-interferente, que é
onde o usuário percebe o isolamento. O treino passa a emular o gating **por amostra**: `emulate_gating_offline`
recebe limiares por amostra, `(−10, +∞, +∞)` com `mask=1` e `(−10, 30, 20)` com `mask=0`. As perdas são calculadas
sobre a saída selecionada, como hoje.

**O que muda no runtime do app (contrato v4 = contrato v3 + isto):**

- `manifest.json` ganha `gating: {"profile": {"min_db": -10, "max_db_erb": null, "max_db_df": null},
  "no_profile": {"min_db": -10, "max_db_erb": 30, "max_db_df": 20}}`. `null` significa sem limite. Asset sem esse
  campo mantém o comportamento atual.
- No Clearcore, `RuntimeParams::with_thresholds` (`vendor/crates/deep_filter/src/tract.rs:170`) já existe. Falta
  escolher o conjunto de limiares conforme `speaker_present` no caminho tract e também nos backends OpenVINO e
  TensorRT (`crates/runtime-openvino/src/dfn3.rs`, `crates/runtime-tensorrt/src/tensorrt/dfn3.rs`,
  `crates/model/src/dsp_pipeline.rs`). A seleção acontece por quadro, em CPU, sem custo mensurável.
- A política (a) **pode ser ligada já com o v3** como experimento: a Fase 0 mede o efeito dela isolado (§5).
- Grafos, nomes, formas, opset 12, estados, hop e MACs (~5 M/hop) ficam idênticos. O gate de paridade
  (`gate_pro_v3_parity.py`) é reaproveitado, com a seleção de estágio nova na réplica.

## 4. Perdas

### 4.1 Bugs confirmados

**ArcFace** (`scripts/train_pro.py:167-179`):
`arcface_angular_margin_loss(b["embedding"], b["embedding"], b["interferer_embedding"])`. Os três argumentos são
entradas do lote (o embedding de enrollment e o embedding do banco do interferente). Nenhum passa pelo modelo, então
o gradiente em relação aos parâmetros é **identicamente zero**. O valor também é ~0: com `p = t`,
`cos_target = 1` e `cos_target_m = cos(0,25) = 0,969`. O logit do alvo fica 32·(0,969 − cos_i) acima do logit do
interferente, de ~−21 para `cos_i` ≈ 0,3, e a CE fica em ~e⁻²¹. Medido no log do v3: média de 7,45e-11, com um
único valor não nulo em 100 registros. **Correção: remover.** Uma versão com sentido exigiria o embedding da
**saída**, com um ECAPA congelado diferenciável sobre `est`, a 16 kHz, e InfoNCE entre `cos(ECAPA(est), e_alvo)` e
`cos(ECAPA(est), e_interf_k)`. Custo estimado de +30–50% por passo. Esse sinal é fraco e fácil de "enganar" perto
das perdas de energia de §4.2, então fica como **ablação opcional** (braço C, peso 0,05, a partir da Fase 2) e só
roda se o critério de negativos difíceis falhar.

**`target_absence_loss`**: ver §1.2-2. É substituída por `L_leak`.

### 4.2 Perda total do v4

```
L = L_mr + L_os + λ_leak·L_leak + λ_ovl·L_ovl + λ_guard·L_guard + L_lsnr + λ_z·L_zero + λ_k·L_keep + L_t + L_dfreg
```

| Termo | Definição | Onde |
|---|---|---|
| `L_mr` | `dfnet_loss` multirresolução (mag + complexa), igual a hoje. Alvo: alvo com reflexões iniciais (perfil) ou `clean_all` (sem perfil) | todas |
| `L_os` | `asymmetric_formant_loss` (proteção de 300–3.400 Hz contra supressão excessiva), igual a hoje. Peso por amostra ×2 quando N≥1, para combater o "atenua tudo" | todas |
| `L_leak` | Invariante a nível: `mean_q relu(10·log10((E_out(q)+ε)/(E_mix(q)+ε)) + 40) / 40` sobre os quadros `perfil ∧ ¬alvo ∧ interf`, com `E` a energia espectral do quadro (STFT do hop) | substitui `target_absence_loss` |
| `L_ovl` | Supressão do interferente na sobreposição, sem punir o alvo: `mean_q relu(⟨|Y|,|I|⟩_q/‖I‖²_q − ⟨|S|,|I|⟩_q/‖I‖²_q)` sobre os quadros `perfil ∧ alvo ∧ interf`. Mede a projeção de magnitude da saída sobre a do interferente (seco e escalonado), descontada a projeção do alvo limpo. Ou seja, pune só o que vai além da correlação natural entre as duas falas | amostras com N≥1 |
| `L_guard` | Hinge "perfil não piora": `relu(SI-SDR(ŷ_semperfil).detach() − SI-SDR(ŷ_perfil) + 0) / 10`. `ŷ_semperfil` vem de um forward extra do próprio aluno com `mask=0` sob `no_grad`, os dois com gating emulado | amostras N=0 com perfil (~21% do lote) |
| `L_lsnr` | `local_snr_loss` com o alvo atual (alvo contra tudo o resto); máscara = todos os quadros menos o silêncio digital (§3.2) | todas |
| `L_zero`, `L_keep` | hinges de §3.2 | perfil |
| `L_t` | destilação do DFNet3 upstream nas amostras sem perfil (`lambda_t` = 500), igual a hoje | sem perfil |
| `L_dfreg` | regularização temporal do DF, igual a hoje | todas |

**Calibração dos pesos (Fase 0)**: rodar 200 passos sem atualizar os pesos, a partir do v3, e fixar `λ_leak`,
`λ_ovl`, `λ_guard`, `λ_z` e `λ_k` de modo que cada termo pese 5–15% de `L_mr` na mediana. O peso inicial esperado é
da ordem de 1 para `L_leak` e `L_guard` (o `L_mr` vale ~5–7). A lição do v2/v3 foi que termos com magnitude de 1e-4
a 1e-6 (`abs`, `gate`, `arcface`) ficam decorativos. O log a cada 20 passos registra a contribuição **relativa** de
cada termo. Um termo abaixo de 1% por 2k passos dispara alarme no log.

### 4.3 Por que isto ataca cada sintoma

- "Perfil piora sem interferente": `L_guard` (em toda amostra N=0 com perfil) + `L_keep` + reflexões iniciais no alvo
  (menos derreverberação forçada). A Fase 0 mede antes quanto da perda vem do estágio ZERO.
- "Suprime o interferente, mas não melhora o alvo": `L_os` ×2 com N≥1. O `L_ovl` pune só a parte do interferente,
  e a calibração impede que `L_leak` domine.
- "Só 50–58% de ZERO no só-interferente": supervisão do `lsnr` no só-interferente + `L_zero` + `L_leak` (o que não
  for zerado é mascarado).
- "Isolamento pequeno no uso real": gating (a) + SIR de +15 a +30 + caso silencioso + níveis variados.

## 5. Currículo e orçamento

### 5.1 GPU e tempo por passo (medido)

- GPU: **NVIDIA RTX PRO 1000 Blackwell Laptop, 8.151 MiB** (ociosa na medição: 42 MiB em uso). CPU com 16 threads,
  30 GiB de RAM (23 GiB em uso no momento), 67 GB livres em `/home`.
- v2 (`pro-v2-dualpath-tse-20261008`): 75.000 passos, lote 16 × 3 s, de 22:16 a 03:30 (5 h 14 min), ou seja
  **0,251 s/passo de parede**. O `dt` de computação (sem a carga de dados) tem mediana de 0,239 s. GPU a 90% de uso
  e 4,5 GB de memória.
- v3 (`pro-v2-mhca-k16-tse`): 2.000 passos em ~8 min, **0,240 s/passo**.

### 5.2 Estimativa para o v4

0,25 s × 4/3 (segmento de 4 s; os GRUs são lineares em T) ≈ 0,33 s, mais ~0,02 s do forward `no_grad` do
`L_guard` (21% do lote, 1/3 do custo) ≈ **0,35 s/passo**, com lote 16 × 4 s e memória estimada em ~6 GB. Se não
couber, o lote cai para 12. O carregador fica mais caro (até 5 falas, 5 RIRs e ruído por amostra), então a Fase 0
mede `measure_loader_throughput` com 8 workers. A meta é ≥ 3 lotes/s. Abaixo disso, as RIRs são pré-convolvidas
por época ou se usa `fftconvolve` em lote.

### 5.3 Fases

| Fase | Passos | Dados | LR | Saída |
|---|---|---|---|---|
| 0: preparação (sem treino) | n/a | n/a | n/a | Correções de §4.1; `mix_v4` + testes; RIR v3; banco +625 alvos (2,5 h de CPU); pool de interferentes do CV; conjunto de avaliação v4 congelado (§8); **linha de base v2 e v3 nas duas políticas de gating**, com o ganho da política (a) isolado, sem treino; ablação "gating desligado" no caso D, para separar quanto da perda de −3,8 dB é do estágio ZERO e quanto é da máscara; calibração dos pesos; vazão do carregador; **retomada (resume) no `train_pro`**, porque hoje `load_latest` é importado mas nunca usado, e 12 h em notebook exige retomada |
| 1: reorientação | 0–20k | N∈{0,1,2}, SIR −10..+20, níveis e ruído completos, 20% de negativos difíceis | aquecimento de 2k até o pico de 6e-5 | ckpt 20k + avaliação dev curta |
| 1B: braço B (opcional, §7) | 0–20k | igual à Fase 1 | igual | A/B contra a Fase 1, para decidir o `lsnr` pós-CA |
| 2: principal | 20k–100k | distribuição completa de §2.3 (N 0..4, SIR −15..+30, 35% de negativos difíceis) | cosseno | avaliação dev a cada 20k |
| 3: recozimento | 100k–120k | igual, com negativos difíceis em 50% e o caso silencioso em 30% | cosseno até 0 | candidato final |

Um único cosseno de 120k passos (aquecimento de 2k, pico de 6e-5, `weight_decay` 1e-2, clip 1,0, AdamW). As fases
mudam só a distribuição de dados, por época. Pico de 6e-5: é o LR do v2 vindo do estágio F. O v3 usou 3e-5 só para
adaptação, e aqui o objetivo muda bastante. A BatchNorm continua congelada (`prepare_student`).

**Tempo total (estimativa)**: 120k × 0,35 s ≈ **11,7 h de GPU**, mais o braço B de 20k ≈ 2 h, mais as avaliações
≈ 1,5 h, o que dá **~15 h de GPU**. Mais ~2,5 h de CPU do banco, em paralelo. Na prática são duas noites, com
retomada. Variante mínima: 60k passos (~6 h), aceitável se a avaliação de 40k já cumprir os critérios. Variante
estendida: 200k (~20 h), só se a curva de dev ainda subir aos 100k.

### 5.4 Por que não partir do zero nem do v2 75000

- Do zero: perderia o DFNet3 ajustado (ruído e caminho sem perfil) e exigiria mais de 300k passos (~30 h) só para
  voltar ao ponto atual.
- v2 75000: tem K=8 (`fallback_proj` 192→1024) e não carrega na forma do contrato v3. O v3 2000 **é** o v2 75000
  com os tokens K=16 inicializados e 2k passos de adaptação.

## 6. Riscos

1. **Troca de vazamento por dano ao alvo.** Se `L_leak` ou `L_zero` dominarem, volta o "atenua tudo". Mitigação:
   calibração por contribuição relativa, `L_keep`, `L_guard` e o critério de quadros zerados indevidamente (§8).
2. **Negativos difíceis demais para o modelo** (diferencial de ~8 dB hoje). Pode faltar capacidade de
   discriminação fina num modelo de ~5 M MACs. Isso é medido em separado (§8, eixo "parecido"). O braço C (ECAPA
   na saída) e o braço B (`lsnr` pós-CA) são as alavancas.
3. **Dados**: CV sem filtro (`client_id` compartilhado), sussurro só simulado e RIR sintética. Mitigação: filtrar
   os alvos com `cv_filter`; usar o novo CV só como interferente quando o falante tiver menos de 8 clipes.
4. **Vazão do carregador** com até 5 falas, RIRs e ruído por amostra, 8 workers e 7 GiB livres de RAM. Medido na
   Fase 0.
5. **Mudança no runtime**: sem o campo `gating` no manifest, o app mantém o comportamento atual. Um app antigo com
   asset v4 roda, só sem o ganho da política (a).
6. **O caminho sem perfil** pode regredir com `p_drop` 0,15 (era 0,2). Isso é coberto pela destilação e pelo
   critério 7 de §8.

## 7. Arquitetura: K=16/D=128 se mantém

- **Os tokens não carregam mais informação que o embedding.** `tokens = fallback_proj(e).view(16,128)` é uma função
  afim de um vetor de 192 dimensões, com posto ≤ 193. Aumentar K ou D não adiciona informação de falante. O v3
  (K=16) ficou 0,3–0,8 dB **abaixo** do v2 (K=8) por estar subtreinado, e não há evidência de ganho de capacidade.
- **A cross-attention contribui pouco hoje**: `v3_full − v3_film` = +0,2 a +0,5 dB. As falhas grandes (−11 dB no
  alvo, −3,8 dB sem interferente, 50% de ZERO, passthrough) são de objetivo, dados e gating.
- **K=16/D=128 se mantém** para preservar o contrato v3 e o warm start.
- **Braço B (ablação, sem mudar o contrato)**: hoje o `lsnr` sai do embedding **antes** da cross-attention
  (`pdfnet3_pro.py`; o contrato v3 confirma que "lsnr sai do embedding antes dela") e só "vê" o perfil pelo FiLM.
  O braço B lê o `lsnr` de `emb_conditioned`, depois da cross-attention, com a cabeça `lsnr_fc` reinicializada a
  partir da atual. A cross-attention já está no `enc.onnx`, então as E/S do grafo não mudam: só a conta interna.
  Decisão ao fim da Fase 1: o braço B entra se melhorar o %ZERO no só-interferente em ≥5 pontos sem piorar os
  quadros do alvo zerados indevidamente.
- **Critério para declarar a capacidade como gargalo** (o que justificaria um v5 com arquitetura nova): aos 60k,
  perda de treino ≈ perda de dev no subconjunto difícil (N≥2, SIR ≤ 0, parecido), sem gap de generalização, e
  critérios 2–4 de §8 ainda não cumpridos.

## 8. Protocolo de avaliação (v4 x v2 x v3)

**Conjunto `tse-v4-dev`**, congelado na Fase 0 e gerado por `mix_v4` com semente fixa, só com falantes de **dev**
(98). Enrollment = dev-select (6–10 s, disjunto das falas das misturas). O teste (124 falantes) é usado **uma vez**,
na decisão final.

**Matriz**: N ∈ {0, 1, 2, 3, 4} × SIR por interferente ∈ {−10, −5, 0, +5, +10, +20, +30} × nível do alvo ∈ {−45,
−30, −18} dBFS × ruído ∈ {silencioso (piso de −70 dBFS), SNR 20, SNR 5}. Com N=0 não há eixo de SIR, então são
3 × 3 células. Total de 4 × 7 × 3 × 3 + 9 = **261 células**, com 12 itens de 5 s cada (~3.100 itens), mais três
eixos extras com 40 itens:

- **Parecido**: interferente entre os 10 vizinhos mais próximos, mesmo sexo, N=1, SIR 0.
- **Temporal**: cenas de conversa ou blocos (só-alvo / só-interferente / ambos / silêncio rotulados), N∈{1, 3}.
- **Perfil trocado** (especificidade): o mesmo item com o perfil do interferente.

**Execução**: todos os modelos (v2, v3, v4; com e sem perfil) passam pelo **gating do runtime**. A matriz inteira
usa `emulate_gating_offline` com os limiares de cada política. Um núcleo de 30 células passa pela réplica stateful
(`run_replica`, paridade com `tract-check`) para confirmar. v2 e v3 são avaliados nas políticas atual e (a), e o v4
na (a).

**Métricas** (por item, médias por célula, IC95 por bootstrap sobre falantes):

- ΔSI-SDR e ΔSTOI/ΔESTOI do alvo (referência = alvo com reflexões iniciais), mais SI-SDR segmental nos quadros
  com alvo.
- Supressão do interferente e ganho do alvo (dB), pela projeção de mínimos quadrados da saída sobre os sinais
  conhecidos (o mesmo método do `eval-tse`).
- Só-interferente: vazamento em dB (energia de saída / energia de entrada) e **% de quadros em ZERO**.
- Só-alvo: dano = ΔSI-SDR(perfil) − ΔSI-SDR(sem perfil); **% de quadros com alvo ativo em ZERO** (zerados
  indevidamente).
- DNSMOS P.835 (SIG/BAK/OVRL) nas células N∈{0, 1}.

**Critérios de aceite para embutir o v4** (todos em dev, depois confirmados em teste, sempre pelo gating do runtime):

| # | Critério | Meta | Referência hoje |
|---|---|---|---|
| 1 | N=0 (SNR 5): ΔSI-SDR com perfil ≥ ΔSI-SDR sem perfil − 0,5 dB, e ΔSTOI com perfil ≥ 0 | ≥ +6,9 dB | v3: +3,6 dB contra +7,4 |
| 2 | N=1, SIR 0, silencioso: ΔSI-SDR ≥ +4 dB, ΔSTOI ≥ +0,03, ganho do alvo ≥ −3 dB, supressão do interferente ≥ 15 dB | | v3: −1,1 dB, −0,08, −12 dB, 20 dB |
| 3 | N=2 e N=3, SIR 0 por interferente: ΔSI-SDR ≥ +3 dB e ΔSTOI ≥ 0 | | não medido |
| 4 | Só-interferente (N=1..3, perfil): % ZERO ≥ 90% (silencioso) e ≥ 80% (SNR 5); vazamento mediano ≤ −30 dB | | v3: 58% / 50%; −28 / −17 dB |
| 5 | Interferente fraco (SIR +20 e +30, silencioso): supressão do interferente ≥ 15 dB | | passthrough hoje |
| 6 | Quadros do alvo zerados indevidamente ≤ 2% em todo nível ≥ −45 dBFS | | a medir |
| 7 | Sem perfil: ΔSI-SDR e ΔSTOI no C0 a até 0,2 dB / 0,005 do v2 sem perfil | | v2 sem perfil |
| 8 | Robustez de nível: ΔSI-SDR a −45 dBFS a até 1,5 dB do valor a −18 dBFS (N=1, SIR 0) | | a medir |
| 9 | Pareado com o v2: v4 ≥ v2 em ≥ 80% das células com N≥1, e nunca pior que 0,5 dB em células N=0 | | n/a |
| 10 | Especificidade: com perfil trocado, ganho do alvo ≤ −15 dB | | v3: −20 dB (já cumpre) |
| 11 | Runtime: paridade ≥ 60 dB (ORT, OpenVINO CPU, tract), < 10 ms/hop na CPU de referência; MACs inalterados | | v3 cumpre |

Se 1, 4, 5 e 6 forem cumpridos mas 2 ou 3 falharem, o v4 entra assim mesmo: corrige os dois relatos do usuário e não
regride. O que falhar vira o escopo do v5.

## 9. Tarefas de implementação (ordem)

1. `scripts/train_pro.py`: remover o ArcFace, trocar `target_absence_loss` por `L_leak`, adicionar `L_ovl`,
   `L_guard`, `L_zero`, `L_keep`, gating por amostra, retomada, log de contribuição relativa e de classes de quadro.
2. `src/cctrain/data/mix_v4.py` + `StageV4Dataset` (rótulos de atividade relativos, níveis, turnos, N interferentes,
   RIR na mesma sala, negativos difíceis), com testes de propriedade: SIR e nível medidos dentro de ±0,1 dB do
   sorteio; frações de classe de quadro; enrollment sem a fala da mistura.
3. `src/cctrain/eval/runtime_path.py`: `emulate_gating_offline` com limiares por amostra; `run_replica` com a
   política por `speaker_present`.
4. Dados: pool de interferentes do CV-full (excluindo dev e teste), `cv_filter` com `min_clips=8`, banco v3 (+625
   alvos), RIR v3 multi-fonte, centróides.
5. `scripts/eval_pro_v4.py`: matriz de §8 + relatório pareado.
6. Export: `pdfnet3_pro_v3.py` sem mudança de grafo (braço A); com o braço B, mudar a origem do `lsnr` dentro do
   enc. Manifest com `gating`.
7. Clearcore: seleção de limiares por `speaker_present` (tract, OpenVINO, TensorRT) + teste de paridade da réplica.

## 10. Depende do usuário

- **Orçamento de GPU**: ~15 h recomendado (duas noites no notebook, GPU ocupada); mínimo de ~6 h; estendido de
  ~20 h.
- **Aprovar a mudança de gating no runtime** com perfil (§3.3), que é uma mudança de comportamento do app. Ela pode
  ser testada antes com o v3.
- (Opcional) Gravar um conjunto real curto (a voz dele + outra pessoa, num cômodo silencioso e num ruidoso, com
  enrollment separado) como verificação qualitativa final. A voz do dono está no treino (0,2 h), então ela não
  serve como métrica de generalização.

## 11. Execução do run longo (2026-11-10, `train/pro-v4`)

Run `runs/m4/pro-v4` (`scripts/train_pro_v4.py`, `src/cctrain/data/mix_v4.py`), partida do v3 2000 (155/155 tensores),
parada dura por relógio às 07:15 (janela de GPU até 08:30, 1 h 15 de avaliação), checkpoint a cada 15 min e no SIGTERM.

- Orçamento real: ~9 h de GPU numa noite → **85.500 passos** num único cosseno (aquecimento de 1.000, pico 6e-5),
  medido 0,373 s/passo (lote 16 × 4 s, 6 workers, espera do loader ~9 ms, VRAM 5,7 GB, PSS ~4,6 GiB). Fases por
  fração do cosseno: fase 1 nos primeiros 15% (N≤2, SIR −10..+20, 20% negativos difíceis), fase 3 nos últimos 15%
  (50% negativos difíceis, silencioso 30%). Sem braço B nem C.
- Calibração (80 passos sem atualizar, a partir do v3; alvo ~10% de `L_mr` na mediana): λ_leak 2,0, λ_ovl 14,
  λ_guard 1,6, λ_zero 3,3e-3 → 1,3e-2, λ_k 0,11; `L_lsnr` 1e-3 como no plano. No primeiro lançamento (λ_zero 3,3e-3)
  o `L_keep` derrubou os quadros do alvo zerados (23% → 3%) mas também o ZERO no só-interferente (58% → 30%), com
  `L_zero` a 3,3%: o run foi reiniciado com λ_zero dobrado (`runs/m4/pro-v4-aborted-lzero`).
- Held-out: `joao_messias_take3_*` e `joao_messias_take4_*` (48 arquivos; mesma fala gravada ao mesmo tempo em dois
  canais, 08-32-48 e 08-32-51) fora do treino em qualquer papel; ficam 148 arquivos (take1 + take2). O falante
  `joao_messias` não está no banco de enrollment, então só aparece como interferente. A gravação
  `2026-10-08 14-30-02.mp4` é ruído de escritório (não é fala do usuário) e continua no pool de ruído. A correção do
  take3 entrou com o run já em curso: parado no passo 898 por SIGTERM e retomado do checkpoint (mesmo cosseno).
- Desvios: RIR v3 multi-fonte não gerada (cada falante recebe uma RIR independente do pool v2); negativos difíceis
  só entre os 972 falantes do banco (centróide), sem o critério de mesmo sexo; banco de alvos não estendido (+625);
  interferentes extras = 1.511 falantes novos do CV-full (≤20 clipes cada), sem os clipes extras dos 603; o gating
  por amostra está em `train_pro_v4.gate_v4` (o `runtime_path.py` da avaliação não foi alterado).
