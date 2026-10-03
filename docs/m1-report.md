# Relatório do M1: speaker encoder (SpeechBrain ECAPA portado) e gate funcional

Branch `m1/t12` (base `main` b656c8b). Níveis: `configs/eval/gates-v2.yaml` (bloco `m1`; SHA-256 `gates-v2.sha256`),
`configs/m0/budget-v1.yaml` e `configs/eval/runtime-deviation-v1.yaml`. Nenhum nível mudou. Os números abaixo vêm de
`$CCTRAIN_RUNS_DIR/m1/t12_*.json` (fora do git); o veredito é calculado por `cctrain.m1.report.m1_verdict`, que lê os
níveis dos arquivos congelados e falha fechado em medida ausente ou NaN.

## Veredito: NO-GO (estrito pelos níveis congelados)

**Decidiu o gate funcional.** ΔSI-SDR do estágio C contra o DFNet3, mesmo sexo, SIR 0 dB, C3 (falante concorrente sem
ruído), dev: **+0,28 dB, IC 95% [−0,96; +1,42]**. O nível exige limite inferior > 0; o medido é −0,96. Todos os demais
níveis bloqueantes passam (EER nas três listas, paridade tract, base congelada, orçamento). O M1 não fecha: o encoder
está aprovado, o gate funcional não.

| Verificação | Medido | Nível congelado | Resultado | Bloqueia |
|---|---:|---|---|---|
| `eer_librispeech_test_clean` | 0,0142 | <= 0,03 | passa | sim |
| `eer_vctk_heldout` | 0,0182 | <= 0,06 | passa | sim |
| `eer_cv_pt_heldout` | 0,0696 | <= 0,08 | passa (ver ressalva do CV) | sim |
| `tract_parity` (2, 6, 8, 10, 12 s) | 8,6e-6 | <= 1e-4 | passa | sim |
| `functional_c3_ci_lower` | −0,962 | > 0 | **falha** | sim |
| `base_frozen` (SHA-256 `c4a1c6a5…`) | igual antes e depois | igual | passa | sim |
| `budget_ram_pss_gb` | 3,00 | <= 6,0 | passa | sim |
| `budget_vram_reserved_gb` | 2,46 | <= 7,0 | passa | sim |
| `budget_disk_free_gb` | 45,9 | >= 10 | passa | sim |
| `deviation_gate` | −14,997 dB; 96,5% | <= −15 dB e >= 90% | falha por 0,003 dB (escolhe `replica`) | não bloqueia |

## EER do encoder portado (fora do domínio do VoxCeleb)

Trials congeladas da Tarefa 9 (`configs/eval/trials-v1.sha256`, hash conferido antes de embutir; 5.000 alvo + 5.000
não-alvo cada). Encoder: `SpeechBrainFbank` + `ECAPATDNN` portado (SHA-256 do checkpoint `0575cb64…`), áudio inteiro
a 16 kHz, cosseno. IC 95% por bootstrap dos trials (informativo). WeSpeaker ResNet34 só como referência (não é nível).

| Lista | Falantes | Clipes | EER portado | IC 95% | Nível | EER WeSpeaker (ref.) |
|---|---:|---:|---:|---|---:|---:|
| `librispeech_test_clean` | 40 | 2.615 | **1,42%** | [1,17; 1,66] | <= 3% | 0,90% |
| `vctk_heldout` | 14 | 5.428 | **1,82%** | [1,57; 2,12] | <= 6% | 1,02% |
| `cv_pt_heldout` | 60 | 3.405 | **6,96%** | [6,45; 7,48] | <= 8% | 4,44% |

**Ressalva do CV:** `cv_pt_heldout` foi filtrado pelo próprio WeSpeaker (§4.6, Tarefa 9): o EER do WeSpeaker nessa lista é
otimista (viés de seleção) e não serve como medida independente; o do encoder portado não tem esse viés, e é o que vale
para o nível. A margem no CV é de 1,0 ponto percentual. O G5 usa só o WeSpeaker como avaliador (gates-v2): no CV ele
herda a mesma ressalva e deve ser reportado com ela no M4.

## Paridade tract do `enrollment.onnx` final

Asset exportado com o encoder portado e o gerador FiLM do checkpoint final do estágio C (`ckpt-00020160`; 85.271.697 B,
SHA-256 `5ab6bc6b…`). Entradas: fala real do dev (LibriSpeech dev-clean) e sintética em 2, 6, 8, 10 e 12 s a −20 dBFS RMS,
mais fala real a −40 dBFS e pico 0,99 em 2 e 12 s, mais 6 sinais de borda do G11.

- tract × módulo PyTorch, 14 casos de duração/nível: **max|d| = 8,6e-6** (nível 1e-4), sem NaN/Inf em módulo, ORT e tract; P1 (ORT × módulo, atol 1e-5, rtol 1e-6) passa em todos eles.
- Bordas G11 (sem nível próprio no bloco `m1`; informativo): silêncio, DC, onda quadrada, ruído branco e ruído −90 dBFS
  passam com folga (<= 3,9e-5). O **impulso isolado dá 1,4e-4** (acima de 1e-4) e reprova o P1 do ORT. Não conta no nível
  (que são as durações 2-12 s), mas é um aviso: com o gerador treinado o erro nas saídas FiLM é maior que com o gerador
  aleatório da Tarefa 10 (5,2e-5). O M3 (G0/G11, tolerância P2 relativa) é quem decide se importa.

## Banco de embeddings (spec §4.4)

`data/embeddings/bank-m1-sbecapa-v1.npz` (+ `.meta.json`, SHA-256 `3b0db5e7…`; fora do git): **972 falantes de treino,
18.880 embeddings de 192-d**. Por variante: 6-12 s de falas do falante concatenadas com pausas de 150-400 ms, RIR
(p=0,3, `rir-synthetic-v2`), ruído (p=0,7, SNR U[0,30] dB, pool de ruído de treino deduplicado), denoise pelo DFNet3
upstream sem perfil (p=0,5), `resampler-48k-16k-v1`, `check_enrollment_audio` (variante recusada é sorteada de novo).
Só embeddings são guardados (nenhum áudio). Determinístico por `(seed, falante, variante)`, retomável.

- 776 falantes têm as 20 variantes; **196 têm menos** (a maioria 15-19; mínimo 3): CV 139, VCTK 51, LibriSpeech 5, CML 1. Nenhum falante foi descartado.
- **Aparo de silêncio nas bordas** de cada fala antes de concatenar (decisão minha, ver abaixo): sem ele o VCTK mic1 (~1 s de
  silêncio por fala) reprovava na regra de 60% de fala ativa e a 1ª geração, abandonada, rendia poucas variantes.
- Só `train`: `training_records` recusa dev/teste.

## Rodada real do estágio C

60 min de GPU sob `gpu_lock`, em background (`nohup`), checkpoints atômicos a cada 10 min, retomável (o orçamento de
tempo agora conta o já gasto). 20.160 passos (batch 16, segmentos de 3 s, 0,163 s/passo mediano), AdamW LR 1e-3, perdas
§5.4 sem destilação, só o gerador FiLM treina. Dados: falantes de treino com sexo conhecido (338: VCTK 87 + LibriSpeech
251), mesmo sexo, SIR 0 dB; ruído deduplicado de treino (9.077) e RIR sintéticas v2.

- Base: SHA-256 do `state_dict` igual antes e depois. Sonda de RAM (PSS trainer + workers) 3,00 GB; VRAM reservada de pico 2,46 GB.
- Térmica (nvidia-smi por minuto em `stage-c-m1/nvidia-smi.csv`): picos de 69 C e 1.777 MHz, com `clocks_throttle_reasons` de limite de potência (0x4), sem alerta térmico.
- Loss total por 4.000 passos: 9,47 → 9,37 → 9,29 → 9,18 → 9,13 (últimos 160: 9,11). Queda lenta e contínua, sem platô claro.

## Gate funcional no dev

Dev: 48 falantes com sexo conhecido (LibriSpeech dev-clean 40 + VCTK 8; CML e CV não têm sexo), 240 misturas por condição,
enrollment limpo de 6-12 s de outras falas do falante, embedding do encoder portado, checkpoint final (sem seleção de
checkpoint pelo dev), 10.000 reamostragens de bootstrap por falante.

Caminho de medição: o gate de desvio (congelado, não bloqueante) mediu **−14,997 dB / 96,5%** nas 24 primeiras misturas
C3 (7.151 quadros) e escolheu a **réplica do runtime** (`tract-check gating`, asset de FiLM identidade + vetor do gerador),
por 0,003 dB. As duas condições foram medidas por ela.

| Condição | Caminho | ΔSI-SDR (C − DFNet3) | IC 95% | DFNet3 | Estágio C | Passa |
|---|---|---:|---|---:|---:|---|
| **C3, sem ruído (gate)** | réplica | **+0,28 dB** | [−0,96; +1,42] | −2,71 | −2,38 | **não** |
| C3 (informativo) | offline | +0,42 dB | [−0,78; +1,53] | −2,74 | −2,23 | não |
| C4, ruído a 10 dB (adicional) | réplica | +0,33 dB | [−0,68; +1,30] | −2,55 | −2,19 | não |
| C4 (informativo) | offline | +0,32 dB | [−0,65; +1,24] | −2,57 | −2,15 | não |

O ganho médio é positivo e pequeno nas quatro medidas; nenhuma tem limite inferior acima de 0. O resultado não depende do
caminho. Estágios do gating em C3 (réplica): 79% `erb_df`, 13% `zero`, 5% `erb_only`, 2% `none`.

## Indicador da AB1 (spec §7.3, informativo)

Cosseno do embedding de um enrollment com ruído (cru) e denoisado (DFNet3 upstream sem perfil) com o do enrollment limpo,
48 falantes de dev, ruído de dev, delta emparelhado por falante (IC 95%):

| SNR | cos cru | cos denoisado | delta (den. − cru) |
|---:|---:|---:|---|
| 20 dB | 0,974 | 0,901 | −0,073 [−0,083; −0,062] |
| 10 dB | 0,941 | 0,879 | −0,063 [−0,079; −0,048] |
| 5 dB | 0,917 | 0,839 | −0,078 [−0,096; −0,060] |
| 0 dB | 0,880 | 0,787 | −0,093 [−0,114; −0,073] |
| −5 dB | 0,827 | 0,718 | −0,109 [−0,134; −0,085] |

O denoise **afasta** o embedding do limpo em todos os SNR (o encoder portado é robusto a ruído e o DFNet3 distorce a
voz). Limite do indicador: a referência "limpa" não passou pelo DFNet3; sem esse controle não se separa distorção do
denoise de perda de informação. A AB1 completa (efeito no ΔSI-SDR) é do M4.

## Licença e NOTICE do enrollment

`enrollment.onnx` = encoder SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`, revisão `0f99f2d0…`, Apache-2.0, treinado em
VoxCeleb1/2, "research purposes") + gerador FiLM nosso: `weight_license = CC BY-NC 4.0` com NOTICE declarando os pesos do
SpeechBrain e a proveniência VoxCeleb (já em `NOTICE` e `LICENSE-MODELS`; Tarefa 10). Uso não comercial. O `pdfnet3` segue
MIT OR Apache-2.0. O M5 deve citar a revisão e o SHA-256 do checkpoint no PROVENANCE e no manifest do asset.

## Decisões do orquestrador aplicadas

1. `L_OS` é média T-F (não soma), coerente com a escala da perda base; reavaliar se a ablação mostrar peso mal calibrado.
2. O gate de desvio é não bloqueante: só escolhe o caminho de medição do dev. Aqui escolheu a réplica.
3. WeSpeaker é o único avaliador independente (`gates-v2`: G5.evaluators); o SpeechBrain é o modelo embarcado e não se avalia.
4. Níveis não mudam. `gates-v2.yaml` usado explicitamente (o `functional_gate` passou a carregá-lo; antes usava o v1 por omissão; os níveis do bloco `m1` são idênticos).
5. Pools de ruído deduplicados por SHA-256 antes de montar misturas (DNS tinha 24 cópias de 3 hashes, FSD50K 1): 9.589 conteúdos únicos, dos quais 5% (512) reservados para o C4 do dev; 9.077 no treino e no banco.
6. C3 (sem ruído) é o gate; C4 é medida adicional.

## Decisões minhas nesta tarefa (a rever)

- Aparo de silêncio nas bordas das falas do banco (acima); `max_tries` 8 por variante.
- Dev do gate: 240 misturas por condição sobre os 48 falantes com sexo conhecido; 24 misturas para o gate de desvio (a tarefa 11 usava 8).
- Checkpoint avaliado = o último; nenhuma seleção pelo dev, nenhuma segunda rodada, nenhum ajuste depois de ver o resultado.
- EER com áudio inteiro do clipe (sem corte), cosseno; IC por bootstrap de trials (informativo, não é o IC do nível).

## Riscos e limitações

- **Poder estatístico**: 48 falantes dão IC de ±1,2 dB; mesmo um ganho verdadeiro de +0,3 dB não passaria o nível. Isso explica a largura, não autoriza relaxá-lo.
- **Sobreposição banco × mistura**: o banco é pré-computado, então o enrollment de um falante pode conter a fala que cai na mistura de treino (a spec pede "nunca o da mistura"). Não há como evitar sem banco por mistura.
- **Mesmo sexo só com LibriSpeech e VCTK** (338 de 972 falantes do banco): CV, CML-TTS e TTS-Portuguese não têm metadado de sexo no repositório; o estágio C ignora esses falantes.
- Máquina sem a janela do plano (§11 item 7): com swap esgotado e outros apps abertos, o EER com WeSpeaker e o AB1 levaram minutos a mais; a rodada de GPU não foi afetada (0,163 s/passo, o mesmo do smoke).
- Dev do gate sem português (LibriSpeech + VCTK, inglês); o CV/CML dev não entram por falta de sexo.

## Pendências para o M2

1. **Decisão do dono sobre o NO-GO funcional.** A spec (§5.3) já previa que o estágio C "provavelmente não passa G2/G4" e o plano trata o T2 como sonda; o diagnóstico a fazer, sem mexer em nível: (a) encoder está descartado como causa (EER bom, paridade ok); (b) a loss ainda descia aos 60 min (9,13 → 9,11), então a rodada foi curta para o gerador de ~310 mil parâmetros; (c) o ganho médio é pequeno (+0,3 a +0,4 dB), longe do +1 dB que o `probe_rule` do `gates-v2` pede para seguir ao T3. Inferência minha (não registrada): o estágio F, com a base treinável, é o caminho com chance real, como a spec já supunha.
2. Sexo por falante para CML-TTS, CV e TTS-Portuguese (ou decisão de não usá-los no mesmo sexo), para ampliar os 338 falantes úteis.
3. Reavaliar `L_OS` (média T-F) e a escala de `L_gate` na ablação; `p_drop` e `s` são da AB2.
4. Controle da AB1: passar o enrollment limpo pelo DFNet3 e comparar; o indicador atual não separa distorção de perda.
5. Impulso isolado no `enrollment.onnx` (1,4e-4): conferir no M3 com a tolerância P2 relativa do G0/G11.
6. Dev do gate com mais falantes (e português) se o gate for repetido no estágio F; o `cv_pt_heldout` e o G5 pelo WeSpeaker precisam da ressalva do filtro.
7. Banco: 196 falantes com menos de 20 variantes (CV e VCTK); o M2 pode regenerar com `max_tries` maior ou aceitar.

## Reprodução

`python scripts/m1_t12.py {eer,bank,stage-c,evaluate,parity,ab1,report}` (ver o cabeçalho do script). Código: `src/cctrain/eval/eer.py`,
`src/cctrain/train/embed_bank.py`, `src/cctrain/m1/{pools,encoder,parity,ab1,report}.py`. Logs e JSON em `$CCTRAIN_RUNS_DIR/m1/`
(`t12-*.log`, `t12_*.json`, `stage-c-m1/`).
