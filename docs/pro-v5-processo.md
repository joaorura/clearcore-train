# pDFNet3 Pro v4 → v5: processo de treino, diagnósticos e estado

Documento de processo (não é plano). Estado de referência: **14/11/2026, ~13:20** (rig). Os jobs 141, 144 e 145 ainda
estavam rodando; números marcados "parcial" mudam. Base do documento: branch `train/pro-v5-keep` @ `aa1480d`.
Quando uma afirmação é inferência e não medição, está rotulada como tal.

## 1. Resumo

- **Problema**: com perfil de voz ligado, o v3.1 (e os fine-tunings seguintes) picotam/abafam a voz do próprio usuário
  quando ela vem logo depois de um interferente (C6), danificam o alvo em N=0 (C1) e perdem inteligibilidade (C7).
  O objetivo é manter isso sob controle sem perder a remoção de ruído do DFNet3 (R0/R5/R10) nem o isolamento (C8).
- **Referência**: v3.1 = `runs/m4/pro-v4/checkpoints/ckpt-00079998.pt`. Passa 8/9 critérios no avaliador rápido
  (10/11 no completo); só o C6 falha (−14,5 dB). Foi treinado com o bug g² do `mix_v4` (seção 4.2).
- **Beco do fine-tuning completo**: v4-keep e v4.2 (main/corte/voz) melhoram em 1–2k passos e colapsam entre 2k e 4k.
- **Rota atual (P3)**: DFNet3 publicado congelado + camadas Pro do zero + LoRA. Sem perfil a saída é bit-idêntica ao
  DFNet3. A correção do gate ZERO (`d40e517`) levou a 7/9 PASS com C6 ≈ −3,5 dB, mas o C8 (isolamento) estaciona em
  ~60–73% e, ao passar de ~70%, alvo e ruído pioram em todas as receitas.
- **Estado**: nenhum candidato P3 passa todos os critérios; o 141 tem checkpoint intermediário bom (≈300k amostras).
  Decisão pendente entre aceitar intermediário, iniciar do v3.1 ou liberar GRUs (seção 9).

## 2. Métricas e critérios

### 2.1 Método da sombra

O avaliador (`eval/quality-metrics`, `src/cctrain/eval/tse_eval.py`) roda o runtime replicado em Python
(`runtime_path.run_replica(shadow=...)`). O **alvo limpo** passa pelas mesmas decisões que o modelo tomou sobre a
mistura (estágio ZERO/ERB/DF, máscara ERB, coeficientes DF). Com as decisões fixas o processamento é linear:
alvo-sombra + ruído-sombra + interferente-sombra = saída (erro medido −136 dB). Logo o ganho do alvo por quadro
(10 ms, limitado a [−60, +20] dB) é do alvo, sem viés de projeção.

Definições: quadro ativo = nível do alvo ≥ máximo local (±500 ms) − 30 dB e > −75 dBFS; quadro caído = ativo e ganho ≤
max(mediana − 10, −10) dB; dropout = trecho caído de 20–300 ms; > 300 ms = supressão longa; ESTOI só nos trechos
ativos; atenuação de ruído medida na sombra do ruído (silêncio e durante a fala).

### 2.2 Critérios (PASS/FAIL, `scripts/eval_quality.py`)

| Critério | Regra |
|---|---|
| C1 | dano N=0 (usuário, silencioso): ΔSI-SDR(perfil) − ΔSI-SDR(sem perfil) ≥ −1 dB |
| C2 | dropouts/min N=0 ≤ 1,5 × sem perfil |
| C3 | p10 do ganho N=0 ≥ sem perfil − 3 dB |
| C4 | % de fala em dropout + supressão longa N=0 ≤ 5% |
| C5 | dropouts/min N=1 ≤ 1,1 × v3.1 |
| C6 | mediana do ganho do alvo sozinho na conversa (N=1) ≥ −3 dB |
| C7 | ΔESTOI ativo N=1 ≥ max(v3.1 − 0,01; 0) |
| C8 | ≥ 90% dos segmentos só-interferente atenuados ≥ 15 dB (usuário N=1..4); a seleção usa piso de 85% |
| R0/R5/R10 | ruído SNR 0/5/10: ΔSI-SDR e atenuação no silêncio ≥ DFNet3 publicado − 1 dB; ΔESTOI ativo ≥ −0,01 |

Limiares relativos (×1,5, ×1,1) existem porque cada recorte do usuário tem 12 itens (~0,7 min de fala) e a contagem de
dropouts tem ruído de Poisson de ~±8/min; "+1 dropout" ficaria abaixo do ruído de medida.

### 2.3 Dois avaliadores, dois conjuntos

- **Completo**: `scripts/eval_quality.py` (conjuntos `main` 280 itens e `noise` 72 itens), orquestrado por
  `scripts/eval_quality_v42.py` e `scripts/after_train_eval_v42.sh`. 11 critérios (C1–C8, R0, R5, R10). Usado no fim.
- **Rápido**: `scripts/eval_quick.py` (`watch`/`one`), subconjunto do usuário N=0/N=1, 9 critérios (C1–C8 e R0), em CPU,
  a cada snapshot (`--eval-every-samples`, 100k amostras nos runs P3). Reporta também o ganho do alvo sozinho por tempo
  desde o interferente (VAD pico−30 e pico−45 com 100 ms iniciais excluídos) e `c1_se`.
- **Os dois não são intercambiáveis**: o C8 do rápido é menor que o do completo (v3.1: 91% rápido, 95% completo; P3
  132: 96% rápido final, 95% completo no melhor). Todas as tabelas abaixo usam o **rápido**, salvo indicação.

### 2.4 Seleção do melhor checkpoint

`eval_quick.py watch --select`:

- `default`: mais PASS, desempate configurável (`--tiebreak`; no v5 `c6,c1`).
- `c8gate`: porta C8 ≥ 85%; sem candidato, fallback = maior C8 com C1 ≥ −3 dB. `--keep-all` mantém os snapshots.
- `c8c1gate`: igual, mas C1 ≥ piso (−3 dB) também vira porta (`d40e517`).

O job usa `SELECT=${SELECT:-c8gate}`; os runs com o gate corrigido usam `SELECT=c8c1gate`. O melhor fica em
`<run>/best/best.pt` + `best.json`. **Nunca usar o último checkpoint**: no v4.2 o "melhor" virou o último porque o
avaliador morreu por OOM (seção 4.4).

### 2.5 Ressalva estatística

C1 tem erro-padrão de ~1,3 a 1,6 dB nos quick-evals P3 (`c1_se`; 2,4 no v4.2 main) com 12 itens por recorte; a nota do
vault fala em IC de ±1 dB. Diferenças menores que isso entre checkpoints não são sinal, e um PASS em C1 perto de 0 dB
é compatível com −1 dB real. O mesmo vale para C6 e C7, com menos folga.

## 3. Linha do tempo dos experimentos

Colunas: PASS no avaliador rápido (9 critérios) salvo nota; C1/C6 em dB; C8 em %; R0 P/F.

| Modelo / run | Config | PASS | C1 | C6 | C8 | R0 | Conclusão |
|---|---|---|---|---|---|---|---|
| v3.1 `m4/pro-v4` @79998 | v4, N interferentes, gating `lsnr` | 8/9 (10/11 completo) | −0,56 | −14,5 | 91 (95 compl.) | P | referência; só C6 falha |
| v4-keep `m4/pro-v4-keep` @153093 | fine-tune completo do v3.1, `L_pres`/`L_wrec`, guard congelado, 156,5k passos | 2/11 (completo) | −5,57 | −16,1 | 95 | F | piorou o alvo; colapso |
| v4.2 smoke A | 500 passos, `L_turn` + `L_n0` (inferido do `summary.json`) | 5/9 | −1,16 | −7,5 | n/d | P | micro-run promissor |
| v4.2 smoke B | 500 passos, mais peso de vazamento (inferido do `summary.json`) | 5/9 | −1,57 | −9,2 | n/d | P | idem |
| v4.2 smoke C | 500/1000 passos, `L_n0g` (hinge −1 dB), cauda do interferente, VAD pico−45 | 5/9 | −1,39 / −1,35 | −7,9 / −7,1 | n/d | P | escolhido para o treino longo |
| v4.2 main `m4/pro-v42` | smoke C + 20% alvo ausente, 25k passos, guard 2k/4k | 4/9 @2k; 1/9 @4k; 2/9 final | −2,6 → −5,9 | −10,7 → −19,6 | 88 → 90 (96 compl.) | F | colapso entre 2k e 4k, sem recuperação; completo 2/11 |
| v4.2 corte | main com `λ_leak 4`, `λ_leak_absent 5` | 4/9 @2k; 2/9 @4k | −2,75 → −4,83 | −11,2 → −17,3 | 89 → 92 | F | mesmo padrão; parado em 4155 pelo guard |
| v4.2 voz | `λ_turn 1,0`, `λ_n0g 1,5` | 4/9 @2k | −2,39 | −9,3 | 88 | F | morreu ~passo 2250 (`/dev/shm`, 4.5) |
| P3 inicial (jobs 97 e 99) | DFNet3 congelado do zero, lr 3e-4 (97) e 1e-4 (99) | 0–1/9 | −8 a −14 | −17 a −28 | 16–80 | F | cancelados em ≤1,4k passos (97) e 250 (99) |
| **P3 longo (job 132)** `pro-v5-p3-fast` | 4 GPUs, lote 192, 2M amostras (10417 passos), mix_v4 sem g² | 3/9 final (best @7294: 3/9, 3/11 completo) | −8,4 (best −6,1) | −18,0 (best −15,1) | 96 (best 92; 95 compl.) | F | isola bem, abafa o alvo |
| job 136 `pro-v5-p3-keep` | pesos maiores: `turn 2, n0g 4, leak 2, tail 4, leak_absent 2`; 3 GPUs, 1M amostras | 4/9 final (best @4858: 3/9, C8 78) | +0,08 | −8,7 | 75 | F | só pesos de preservação oscila |
| job 141 `pro-v5-p3-zpen` (parcial) | 136 + `d40e517`: `--lsnr-corr 10 --gate-st-tau 2 --lambda-zero-tgt 2 --lambda-p10 1`, `c8c1gate`; 2 GPUs | **7/9 @~300k** (6/9 @200k) | +0,94 | −3,5 | 61 | P | melhor ponto; piora depois |
| job 141 @900k (parcial) | idem | 4/9 | −0,21 | −6,75 | 73 | F (+4,1/43) | deriva no fim; melhor selecionado: @7294 (6/9, C1 +0,63, C6 −4,35, C8 73) |
| job 144 `pro-v5-p3-zpen-cut` @200k (parcial) | 141 + corte forte desde o início: `leak 4, tail 7,5, leak_absent 4`, lote 48 (1 GPU, retomado em 2) | 7/9 | +0,49 | −3,2 | 62 | P | liderava no ponto de 200k |
| job 144 @300k (parcial) | idem | 3/9 | −1,33 (F) | −5,6 | 71 | F (+4,1/44) | C7 +0,009 (F); C8 sobe e o resto cai |
| job 145 `pro-v5-p3-zpen-ft` (parcial) | 141 continuado (`--init-from` ckpt 6827 = 655k amostras) com corte forte, `aa1480d`, 500k amostras | 4/9 @100k | +0,14 | −5,4 | 74 | F (+4,1/45), pior | não destravou o teto do C8 |

Notas sobre a tabela:

- As duas linhas do job 144 usam dois pontos do mesmo run; o mesmo vale para o 141. Os números de 144/145 foram
  conferidos em `quick-eval.jsonl` (passos 4166, 6249 e 1042).
- "n/d": os micro-runs v4.2 não calculavam C8 no avaliador rápido da época.
- O diário do vault lista o v4.2 main com "C8 96%" e o v3.1 com "95%": são os valores do avaliador completo.

Leitura (**inferência, não testada**): em todas as receitas P3, quando o C8 passa de ~70% a voz do alvo e a remoção de
ruído pioram. O provável limite é a capacidade das camadas novas sobre o DFNet3 congelado.

## 4. Diagnósticos e bugs

### 4.1 Picote vem do estado recorrente após o interferente

C6 do v3.1 (alvo sozinho na conversa), por tempo desde o fim do interferente (VAD pico−30): antes de qualquer
interferente −3,6 dB; 0–0,5 s depois −15,9; 0,5–1 s −11,3; 1–2 s −6,9. Com VAD pico−45 e 100 ms iniciais excluídos:
−9,6 / −9,7 / −6,8 dB, ou seja, ~6 dB do efeito eram caudas do interferente contadas como alvo. A atenuação vem da
máscara ERB/DF (97% dos quadros passam por ERB+DF), não do gating; o que sobra indica um estado recorrente
"contaminado" que leva ~1–2 s para se recuperar. Mexer em gating não resolve o C6. Medir sempre estratificado por tempo
e com VAD pico−45. Nota: no P3 o estágio ZERO passou a ser parte do problema (4.5).

### 4.2 Bug g² do `mix_v4` (commit `0145934`, também `f6cec22` no histórico de `pro-v5-keep`)

Desde `c78b1fb` (10/11), com `reverb=False`, `wet[k]` e `early[k]` eram o mesmo array numpy; o ganho do interferente era
aplicado duas vezes (SIR efetivo 2t−2s−i, erros de mais de 40 dB em casos extremos). Afetou ~37,5% das amostras do v3.1
e do v4 e ~28,5% do v4.2. Correção: `early` recebe cópia; teste `tests/test_mix_v4_dry_gain.py`. O **hash de config não
muda** com a correção: registrar no run (`launch-cmd.txt` tem a linha "dados: mix_v4 fix dry-gain 0145934"). Qualquer
comparação com o v3.1 deve lembrar que ele foi treinado com SIR errado.

### 4.3 Deriva do fine-tuning completo

v4-keep (153k passos) e v4.2 (25k) melhoram em ~1–2k passos e colapsam entre 2k e 4k (C6 −7 → −17 dB, dano N=0 −1 → −6
dB, ruído piora), sem recuperação. Mitigações que não bastaram: guard congelado com λ fixo (caiu a 0,57% de `L_mr`),
weight decay (metade da deriva), L2-SP, guard adaptativo, variar pesos de voz/corte. Saída estrutural: P3 (seção 5).

### 4.4 Avaliador periódico morrendo por OOM (`93a28e5`)

`init_df` do DeepFilterNet faz `model.to(get_device())` mesmo pedindo CPU; o avaliador, rodando junto do treino, ocupava
a GPU e dava OOM, deixando `quick-eval.jsonl` vazio e o guard de 2k/4k inerte. O "melhor" virou o último checkpoint. Fix:
forçar CPU no `eval_quick` (`CUDA_VISIBLE_DEVICES=""` antes de importar torch) e repetir snapshots falhos até 3×.

### 4.5 Gate ZERO pela `lsnr_fc` congelada (`d40e517`)

No P3 o ZERO (lsnr < −10 dB com perfil) decide sobre `lsnr.detach()` da `lsnr_fc` congelada do DFNet3, e o único
gradiente dela era `L_lsnr` com alvo = SNR da mistura (o interferente conta como ruído). ~14–15,6% dos quadros ativos do
alvo eram zerados desde o passo 0 e nada corrigia. Não era o embedding (mesmo encoder; embedding no estilo do treino:
−7,3 vs −6,1 dB) nem só o usuário. Correção: `--lsnr-corr` (Linear zero-init sobre a cross-attention × máscara de
perfil, entra no export), `--gate-st-tau`, `--lambda-zero-tgt`, `--lambda-p10`, seleção `c8c1gate`. Efeito: quadros do
alvo zerados 15,6% → ~1%; C6 de −18 para ≈ −3,5 dB. Regra geral: em estágio de gate com `detach()` de cabeça
congelada, garantir que exista gradiente que dependa do perfil.

### 4.6 `/dev/shm` e `RemoveIPC`

O `v42-voz` caiu com "could not unlink the shared memory file" (~passo 2250). Causa provável: `RemoveIPC=yes` do
systemd-logind apaga o `/dev/shm` do usuário quando a última sessão ssh termina. Mitigação: `loginctl enable-linger`
(feito em 13/11/2026) e o laço de relançamento do `job_v5.sbatch` (`MAX_RESTARTS`, padrão 6, retoma do último
checkpoint). A causa é provável, não provada.

### 4.7 TERM duplo no torchrun (`e116124`)

`scancel` já manda um TERM ao torchrun; um segundo o fazia abortar e matar os ranks antes de salvarem (job 97). O sbatch
agora faz `pkill -TERM -P <torchrun>` nos ranks (filhos) e o torchrun termina sozinho.

## 5. Arquitetura P3

- **Base**: DFNet3 publicado, **congelado** (`build_model` carrega `load_upstream_dfnet3`; `--base v31` só na variante
  `ctrl`). Variantes em `scripts/train_pro_v5.py`: P1 (só condicionamento), P2 (+LoRA GRU r8), **P3 (+LoRA encoder r16)**,
  `ctrl` (base v3.1).
- **Treináveis (~1,02 M de 3,16 M, 32%)**: camadas Pro do zero (cross-attention de K=16 tokens/D=128, FiLM,
  `freq_mixer`; ~875k), LoRA r8 nas GRUs (`enc.emb_gru`, `erb_dec.emb_gru`, `df_dec.df_gru`) e LoRA r16 nas convs 1×1
  do encoder e em `enc.df_fc_emb` (adaptadores ~150k). Código: `src/cctrain/models/adapters.py`.
- **Garantias** (`tests/test_adapters_v5.py`): (a) perfil desligado → saída bit-idêntica ao DFNet3 publicado, mesmo com
  adaptadores treinados; (b) passo 0 com perfil ligado → idêntico ao DFNet3 (ramos de saída em zero-init). Com a flag
  `lsnr-corr` ligada, perfil off continua bit-idêntico (Linear zerado, mascarado pelo perfil).
- **Consequência de treino**: lotes sem perfil têm gradiente exatamente 0 nos treináveis. Usar `--p-drop 0` (só
  elimina custo).
- **Flags do `d40e517` (todas desligadas por padrão = caminho anterior, bit a bit)**:

| Flag | Efeito |
|---|---|
| `--variant P3` | P1/P2/P3/ctrl (acima) |
| `--lsnr-corr 10` | correção aditiva do lsnr pelo perfil (escala em dB); torna o gate ZERO dependente de ramo treinável |
| `--gate-st-tau 2` | gate straight-through (forward igual; gradiente via σ((lsnr+10)/τ)) nas amostras com perfil |
| `--lambda-zero-tgt 2` (`--zero-tau 2`, `--zero-tgt-hinge-db −1`) | P(ZERO) + hinge do ganho efetivo do alvo-sombra em quadros de alvo ativo |
| `--lambda-p10 1` (`--p10-hinge-db −3`, `--p10-q 0,1`) | hinge no p10 do ganho efetivo do alvo por item |
| `--init-from CKPT` (`aa1480d`) | inicializa só os pesos (strict) de um checkpoint; optimizer/scheduler/RNG novos, passo 0; entra no hash só quando usada |

- **Export**: `docs/adapter-export.md` (proposta, não implementada): adaptadores viram nós ONNX separados, nunca
  fundidos na base, multiplicados por `speaker_present`; a GRU vira célula explícita com S=1; o `erb_dec.onnx` passa a
  receber `speaker_present`; manifesto `v3.1-adapter`; gate de paridade com `gate_pro_v3_parity.py` (meta ≥ 60 dB em
  fp32, incluindo o caso sem perfil contra o DFNet3). O `lsnr_corr` entra no `EncStep`. O manifesto de gating do
  v4-keep (`mode: "lsnr-profile"`, `zero_below_db: −10`) está em `docs/plan-pro-v4.md` §13; o runtime Rust deste
  checkout ainda não lê o campo (conforme esse plano).

## 6. Desempenho do treino

Ponto de partida: gerador `mix_v4` na CPU (~9 amostras/s com 4 CPUs, GPU a 19%; perfil: `_ramp_env` com `np.convolve`
42%, convolução de RIR em float64 25%). A rig tem PCIe x1 por GPU (~0,82 GB/s). Resultado final: de ~42 para ~80–105
amostras/s por GPU; 4 GPUs ≈ 330 amostras/s no job 132 (`summary.json`: 328,8 amostras/s, 0,388 s/passo, 11,4 GB de VRAM).

| Otimização | Flag / branch | Ganho medido |
|---|---|---|
| Gerador CPU rápido | `perf/mix-fast` (cumsum bit-idêntico, oaconvolve f32, RIRs em RAM, 1 thread por worker) | tira o gargalo mais pesado; OpenBLAS nascia com 16 threads e causava contenção |
| Mistura/rótulos/features na GPU | `--gpu-mix` (`perf/gpu-mix`) | 80 vs 23 amostras/s no gerador; equivalência CPU×GPU 8e-7 no áudio e 0% nos rótulos |
| Fio esparso (H2D) | `--gpu-mix-wire sparse` | lote 170 → 128 MB por H2D (exato) |
| Mistura em f32 | `--gpu-mix-dtype f32` | ganho pequeno (rótulos idênticos, ~1e-6 do pico); usado nos jobs 132–145 |
| Pool de clipes residente na VRAM | `--gpu-pool-gb 5` (`perf/v5-gpupool`) | elimina a cópia H2D; graph + pool: 0,458 s/passo, 104,9 amostras/s/GPU |
| `--p-drop 0` | flag | −14,5% por passo (a LoRA GRU deixa de chamar o cuDNN 2×) |
| Perdas rápidas | `--loss-impl fast` (padrão) | −20% (STFT compartilhada, sem syncs) |
| `torch.compile` nas perdas | `--loss-compile` | −30% |
| CUDA Graph + AdamW fused | `--step-mode graph --opt-impl fused --log-every 25` | com loss-compile: passo 0,672 → 0,300 s (o ganho individual do graph e do fused não foi separado) |
| 2 GPUs (DDP + graph) | `torchrun` | 1,55× |

Descartadas, com o motivo: pré-gerar a base em disco (~900 GB, perde variedade); cache de áudio reamostrado
(~11% de ganho); overlap de H2D em stream (`--gpu-mix-overlap`; limite é banda, não latência); fp16 (sem ganho: com
autocast bf16 a GRU já roda em fp16); `torch.compile` do passo inteiro (`--step-mode compile`), `reduce-overhead`,
`channels_last` (sem ganho e quebram paridade); int16/fp16 no H2D após os ganhos (erro até 1,7 do pico; o codec int16
do pool só é usado onde sem perda, `exact` é o padrão); lote maior com 4 CPUs (CPU-bound); `max-autotune` (compila > 20
min); `--lora-gru-fused` (aproximado, ~1 ulp; não adotado, **não confirmei o motivo exato**).

**Flags recomendadas hoje** (receita do job 141, adicionar `--gpu-pool-gb 5` só depois de integrar `perf/v5-gpupool`):

```
--variant P3 --gpu-mix --gpu-mix-dtype f32 --gpu-mix-wire sparse --p-drop 0 \
--step-mode graph --opt-impl fused --log-every 25 --loss-compile --amp bf16 \
--global-batch 96 --total-samples 1000000 --lr 1.0607e-4 --warmup-samples 50000 --eval-every-samples 100000 \
--lambda-turn 2.0 --lambda-n0g 4.0 --lambda-leak 2 --lambda-tail 4 --lambda-leak-absent 2 \
--lsnr-corr 10 --gate-st-tau 2 --zero-tau 2 --lambda-zero-tgt 2.0 --lambda-p10 1.0
```

O `--lr` é no lote de referência 12 e escalado por √(lote/12): `--lr 1,0607e-4` com lote 96 dá 3,0e-4 efetivo. Manter
a LR efetiva em 3e-4 ao mudar o lote global (jobs 132–145).

## 7. Branches e commits

Tabela completa sobre `train/pro-v5-keep` @ `aa1480d` (conferida com `git merge-base --is-ancestor`).

| Branch | Conteúdo | Em `pro-v5-keep`? |
|---|---|---|
| `eval/quality-metrics` (`1900457`) | `eval_quality.py`, métricas de picote/dano, C1–C8/R*, `diag_c6` | sim |
| `train/pro-v42` (`93a28e5`) | v4.2 (smoke C, config longa, `eval_quick`, fix do OOM do avaliador) | sim |
| `perf/mix-fast` (`3b36625`) | gerador `mix_v4` rápido | sim |
| `perf/gpu-mix` (`56bc2f5`) | `--gpu-mix` (primitivas DSP, features e rótulos em torch) | sim |
| `perf/gpu-mix-dsp` / `-feat` / `-labels` (`e950021`, `5479939`, `774c837`) | os 3 commits-base do gpu-mix, antes de integrados | **não** (1 commit cada à frente; os módulos equivalentes `4c4c911`, `d51db41`, `6ea6afe` estão via `perf/gpu-mix`; **não confirmei** se são cherry-picks idênticos) |
| `perf/v5-speed` (`602ca02`) | flags de velocidade do v5 (`--p-drop`, perfil por lote, fp16, bench) | sim |
| `perf/v5-loss` (`5b1f5de`) | perdas rápidas e `--loss-compile`, paridade fp32/bf16 | sim |
| `perf/v5-graph` (`5848cfa`) | StepRunner (CUDA Graph/compile), sync-free, `IstftSF` por `unfold_backward` | sim |
| `perf/v5-mixspeed` (`ce92418`) | mistura f32/esparsa, lowpass48 como FFT-FIR | sim |
| `perf/v5-gpupool` (`f4f4c35`) | `--gpu-pool-gb`: pool de clipes na VRAM, `PoolMixer`, fix cuFFT | **não** (5 commits à frente) |
| `train/pro-v5-adapter` (`e116124`) | adapters, `train_pro_v5.py`, `job_v5.sbatch`, fix do TERM duplo, `c8gate` | sim |
| `train/pro-v5-fast` (`9a2958f`) | merge de `v5-graph` + `v5-mixspeed`, `job_v5` com `DATA_NOTE`/`LAUNCH_NOTE` | sim |
| `train/pro-v5-keep` (`aa1480d`) | gate ZERO (`d40e517`), `--init-from` (`aa1480d`) | — (este) |

Pontos de atenção:

- **Nenhum branch único tem tudo**: `pro-v5-keep` não contém `perf/v5-gpupool`. Os jobs 132–145 rodaram sem pool
  (`--gpu-mix` + wire sparse). O pool ainda não foi usado em treino real, só em smoke/bench (`smoke-gpupool` na rig).
- Na rig, os worktrees são `clearcore-train-keep` (`aa1480d`, jobs 136–145), `clearcore-train-gpupool` (`f4f4c35`),
  `clearcore-train-fast` (`9a2958f`, jobs 132/136) e `clearcore-train-v5` (`e116124`, padrão do `CODE` no sbatch).
- Não alterar nenhum desses worktrees enquanto houver treino usando o commit.

## 8. Como rodar na rig

### 8.1 Lançar

Acesso, limites e dados: ver `padroes/uso-da-rig-de-gpu-slurm.md` (vault, seção 10 deste documento) e `COMO-USAR.md` na rig. GPU só dentro de job;
dados e runs em `/mnt/4tb-nvme/work/$USER`, nunca em `~`.

```bash
W=/mnt/4tb-nvme/work/$USER
# código = worktree do branch (CODE); dados = clone principal (clearcore-train)
CODE=$W/clearcore-train-keep RUN_ID=pro-v5-p3-xxx SELECT=c8c1gate \
  LAUNCH_NOTE="resumo da receita" \
  sbatch -c 12 --gres=gpu:2 scripts/job_v5.sbatch <flags da seção 6>
```

O `job_v5.sbatch` roda `torchrun` (1 processo por GPU), o avaliador rápido em CPU (`EVAL_THREADS`, padrão 2) e, se o
treino termina com `total_steps`, o `after_train_eval_v42.sh` (eval completo). Escreve `launch-cmd.txt` (comando,
commit, GPUs, nota de dados). O lote global é fixo: relançar com **o mesmo `RUN_ID`** e outro número de GPUs/CPUs
retoma do último checkpoint com a mesma sequência de dados e o mesmo cosseno. Com `--gpu-mix` bastam ~4 CPUs por GPU.
Limite de CPU por nó: 32 (admin autorizou em 13/11/2026).

### 8.2 Onde ficam

`$W/runs/<RUN_ID>/`: `train.log`, `loss-log.jsonl`, `quick-eval.jsonl`/`.log`, `snapshots/` (a cada 100k amostras,
`--keep-all`), `checkpoints/` (a cada 10 min, `--keep 3`), `best/best.{pt,json}`, `summary.json`, `TRAIN_DONE`,
`eval-full/report.md` (após o eval completo), `EVAL_DONE`. Logs do Slurm: `$W/logs/<job>-<id>.out`.

### 8.3 Parar com segurança

`scancel <job>` basta com o `job_v5.sbatch` atual (o `trap` encaminha TERM aos ranks, que salvam e saem; o novo job
retoma). Para jobs lançados com versões antigas do script (ex.: job 99), usar `scancel -b -s TERM <job>` (só o shell do
lote, para o torchrun receber **um** TERM). Nunca enviar dois TERM. `--stop-at HH:MM` do treino também salva e sai.

### 8.4 Avisos de monitoramento

- Todo treino longo precisa de aviso de fim: poll de `squeue` e marcadores `TRAIN_DONE`/`EVAL_DONE`. Incidente: o P3
  longo terminou às 01:43 de 14/11 e só foi visto às 11:18.
- Conferir que `quick-eval.jsonl` tem linhas novas a cada snapshot (avaliador vazio = avaliador morto, seção 4.4).
- O eval completo só roda se `TRAIN_DONE` começa com `total_steps`; parada por sinal/tempo não dispara.
- O eval completo do job 136 estava em andamento (sem `report.md`) ao fechar este documento.

## 9. Pendências e próximos passos

1. **Resultado final dos jobs 144 e 145** (parciais aqui): 144 em 6775/20834 passos, 145 em 2175/5209, 141 a ~10100/10417.
   Conferir o `best.json` de cada um com o eval completo. O melhor candidato hoje é o checkpoint de ~300k amostras do 141
   (7/9; C1 +0,94, C6 −3,5, C8 61%, R0 P).
2. **Decidir a saída para o teto do C8** (candidatos, nenhum testado ainda):
   1. iniciar o condicionamento a partir do v3.1 (`--variant ctrl`/`--base v31` + `--v31-ckpt` existem no script);
   2. liberar as GRUs da base com LR baixa;
   3. aceitar um checkpoint intermediário (~7/9, C8 ~60%).
3. **Integrar `perf/v5-gpupool` em `train/pro-v5-keep`** (e `perf/gpu-mix-*` se for desejado limpar o histórico) e
   validar o pool num treino real (hoje só bench/smoke); só depois recomendar `--gpu-pool-gb`.
4. **Treino longo do vencedor em 6 GPUs**: lembrar que multi-GPU escala mal (PCIe x1; 2 GPUs = 1,55×) e manter a LR
   efetiva e o lote global fixos.
5. **Export para o runtime Rust do app**: implementar o export dos adaptadores (`docs/adapter-export.md`: nós
   separados, `speaker_present` também no `erb_dec`, manifesto `v3.1-adapter`), incluir `lsnr_corr` no `EncStep`,
   passar o gate de paridade (≥ 60 dB fp32, inclusive sem perfil), e fazer o runtime ler o gating `lsnr-profile` do
   manifesto (hoje só `min_db_thresh` fixo em `dsp_pipeline.rs`, segundo `docs/plan-pro-v4.md` §13).
6. Anotar o resultado final (checkpoint escolhido, números do eval completo) neste documento e no vault.

## 10. Ver também (vault Obsidian)

Notas em `/home/joaorura/Documentos/Obsidian Vault/`:

- `projetos/clearcore-train.md` (hub)
- `decisoes/2026-11-13-metricas-de-qualidade-picote-e-criterios.md`
- `decisoes/2026-11-13-picote-vem-do-estado-recorrente-apos-interferente.md`
- `decisoes/2026-11-13-bug-ganho-ao-quadrado-no-mix-v4.md`
- `decisoes/2026-11-13-fine-tuning-completo-deriva-apos-2k-passos.md`
- `decisoes/2026-11-14-p3-dfnet3-congelado-com-adaptadores-do-zero.md`
- `decisoes/2026-11-14-gate-zero-precisa-depender-do-perfil.md`
- `decisoes/2026-11-14-otimizacoes-de-velocidade-do-treino-p3.md`
- `padroes/treino-longo-precisa-de-aviso-e-escolha-do-melhor.md`
- `padroes/uso-da-rig-de-gpu-slurm.md`
- `diario/2026-11-14.md`

No repo: `docs/plan-pro-v4.md`, `docs/adapter-export.md`, `docs/eval-pro-v4-baselines.md`,
`docs/export-contract-pro-v4.md`.
