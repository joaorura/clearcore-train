# Relatório Final do M2: pDFNet3 (DeepFilterNet3 Personalizado por FiLM) e Gates no dev-gate

> **Status:** **NO-GO** (Parada e Diagnóstico)  
> **Data de Execução:** 2026-10-05 00:07:09  
> **Repositório:** `clearcore-train`  
> **Documento de Referência:** `docs/plans/2026-10-03-m2-plan.md` (Tarefa 8)  
> **Níveis e Regras Congeladas:** `gates-v2.yaml` (`0f30917bad2b…`), `m2-selection-v1.yaml` (`4d843ff2d9eb…`), `runtime-deviation-v1.yaml` (`2803f83135f9…`), `tsos-config-v1.yaml` (`b8d7fe70cc1c…`), `m2-devsets-v1.freeze.json` (`87567c89c603…`)

---

## 1. Identificação e Proveniência

Este relatório documenta a avaliação formal e o veredito go/no-go do **Marco M2 (estágio F)** do Clearcore. O modelo sob avaliação é o **pDFNet3**, composto pelo modelo acústico DeepFilterNet3 refinado no estágio F e modulado por FiLM com embeddings de 192 dimensões extraídos pelo speaker encoder SpeechBrain ECAPA portado e congelado no M1.

### 1.1 Modelo Candidato e Professor Upstream
- **Candidato:** Selecionado da matriz de ablações AB2 (`runs/m2/ab2_candidate.json`)
  - *Ramo Vencedor:* `t3`
  - *Época / Passo:* 3 / 22500
  - *Checkpoint:* `/home/joaorura/orca/projects/clearcore-train/runs/m2/t3/checkpoints/ckpt-00022500.pt`
  - *SHA-256 do Checkpoint:* `d5c0fbd7983534e909a991fe68eed17347ae6eb82becd4b2f2904ed86a6d4f7e`
  - *Escore C3 no dev-select:* +1.7927 dB
- **Professor Upstream:** DeepFilterNet3 v0.5.6 pré-treinado e congelado
  - *SHA-256 da Base:* `c4a1c6a5...`
- **Avaliadores Locais (congelados em manifests/models/):**
  - *DNSMOS P.835 e Personalizado (-P):* `microsoft/DNS-Challenge` (ONNX, CC BY 4.0)
  - *Whisper small:* `openai/whisper` (ONNX, MIT)

### 1.2 Integridade do dev-gate e Configurações Congeladas
O `dev-gate` é a metade lacrada do conjunto de desenvolvimento (`data/dev-m2/dev-gate/`), completamente disjunta em falas do `dev-select` e do conjunto de treino.
- **Congelamento do devets:** `configs/eval/m2-devsets-v1.freeze.json` (SHA-256 `87567c89c603e7a5f05f604ef1d0f2c1f5df3cacf62283bce68c1ca216a675eb`)
- **Registro de Abertura:** `data/dev-m2/dev-gate/opened.log` (uso único registrado na Tarefa 8)
- **Hashes das Regras Auditadas:**
  - `gates-v2.yaml`: `0f30917bad2bc45dbe5c969ff54c32eb2f5b75ce0ee0f1440cdf1caab4d63272`
  - `m2-selection-v1.yaml`: `4d843ff2d9eb8e33644078673f4c93c42e991cad62dd7ad9db3af69a3372f2d3`
  - `runtime-deviation-v1.yaml`: `2803f83135f94f7967dc7ca01ae182abbaabf0ad8b20e59508ff97b0a0be554f`
  - `tsos-config-v1.yaml`: `b8d7fe70cc1c20186ef88c08890acb1949789bb1c59beabf8b9f6bf99a693d3a`

---

## 2. Veredito Oficial (Estrito pelos Níveis Congelados)

O veredito foi calculado pela função `cctrain.m2.report.m2_verdict`, falhando fechado diante de qualquer medida ausente, não-finita (NaN) ou fora das tolerâncias congeladas.

> **VEREDITO M2:** **NO-GO** (Parada e Diagnóstico)  
> **Falhas Bloqueantes:** `G2_delta_si_sdr_point`, `G2_wer_ratio_to_dfnet3`, `G2_wer_not_above_noisy`, `G3_tsos_c0_pct`, `G4_trr_different_sex`, `G4_trr_same_sex`

### Tabela de Checagens dos Gates

| Gate / Métrica | Valor Medido | Nível Exigido | Status | Bloqueante? | Notas |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `runtime_deviation` | -15.8138 | <= -15.0 dB | **PASS** | Não | concordância=0.9536711670057333, caminho escolhido=offline |
| `G1_delta_stoi_ci_low` | 0.0604 | >= -0.003 | **PASS** | Sim | IC 95% inferior (decisão do dono) |
| `G1_delta_si_sdr_ci_low` | 5.6950 | >= -0.5 | **PASS** | Sim | IC 95% inferior (decisão do dono) |
| `G1_delta_dnsmos_ovrl_ci_low` | 0.3491 | >= -0.05 | **PASS** | Sim | IC 95% inferior (decisão do dono) |
| `G1_delta_dnsmos_sig_ci_low` | 0.3060 | >= -0.05 | **PASS** | Sim | IC 95% inferior (decisão do dono) |
| `G1_delta_dnsmos_bak_ci_low` | 0.2244 | >= -0.07 | **PASS** | Sim | IC 95% inferior (decisão do dono) |
| `G1_delta_lsd_above_8k_ci_high` | 0.4489 | <= 0.5 | **PASS** | Sim | IC 95% superior (decisão do dono) |
| `G2_delta_si_sdr_point` | 2.2920 | >= 3.0 | **FAIL** | Sim |  |
| `G2_delta_si_sdr_ci_lower_gt_0` | 1.4648 | > 0.0 | **PASS** | Sim |  |
| `G2_delta_dnsmos_p_ovrl` | 0.5210 | >= 0.15 | **PASS** | Sim |  |
| `G2_wer_ratio_to_dfnet3` | 0.9658 | <= 0.85 | **FAIL** | Sim |  |
| `G2_wer_not_above_noisy` | False | True | **FAIL** | Sim |  |
| `G3_tsos_max_over_dfnet3_pp` | -2.8175 | <= 0.5 | **PASS** | Sim | Convenção fixa Opção A (critério estritamente relativo) |
| `G3_tsos_c0_pct` | 17.7386 | <= 1.0 | **FAIL** | Sim |  |
| `G4_trr_different_sex` | 0.8458 | >= 0.9 | **FAIL** | Sim |  |
| `G4_trr_same_sex` | 0.4228 | >= 0.7 | **FAIL** | Sim |  |
| `G4_mode_separation_db` | 45.2227 | >= 15.0 | **PASS** | Sim |  |
| `G5_spk_sim_delta_ci_lower` | 0.0000 | >= -0.02 | **PASS** | Não | informativo M2 |
| `G8_dnsmos_p` | 3.6500 | informativo | **PASS** | Não |  |
| `C8_tanh_saturation` | 0.0050 | <= 0.05 | **PASS** | Não | informativo C8 |

---

## 3. Desvio de Runtime e Caminho Escolhido

Em conformidade com `configs/eval/runtime-deviation-v1.yaml` (spec §3.5), o desvio entre a emulação offline do gating (forward PyTorch + máscara por estágio) e o caminho do runtime (réplica Python quadro a quadro / tract-check) foi medido no candidato:

- **Erro/Sinal:** -15.81 dB (Nível máximo: -15.0 dB)
- **Concordância de Estágios:** 95.37% (Nível mínimo: 90.0%)
- **Caminho Escolhido:** `offline` (`offline` se ambas as medidas passam, senão `replica`)
- **Impacto:** O gate não é bloqueante; sua função estrita é selecionar se a avaliação do dev-gate processa os áudios via emulação offline ou réplica de runtime.

---

## 4. Gate G1 — Modo Sem Perfil (C0, Estágio F)

O Gate G1 avalia o comportamento do modelo quando nenhuma voz de referência é fornecida (ou máscara de perfil desativada), garantindo que o fine-tuning da base não degradou o desempenho em relação ao DFNet3 upstream congelado.

- **Condição:** C0 (alvo + ruído, sem perfil), 48 kHz.
- **Decisão do Dono (2026-10-03):** As margens são estritamente lidas pelo **limite do IC 95% do lado da reprovação** (limite inferior para métricas de ganho/mínimo; limite superior para distorção espectral LSD > 8 kHz), calculado por bootstrap pareado por falante (10.000 reamostragens).

### Medições Detalhadas do G1
| Métrica | Estimativa Pontual | IC 95% [Inferior, Superior] | Limite Exigido | Status |
| :--- | :---: | :---: | :---: | :---: |
| ΔSTOI | +0.0686 | [+0.0604, +0.0774] | >= -0.003 (no inf) | **PASS** |
| ΔSI-SDR (dB) | +6.36 | [+5.70, +7.03] | >= -0.5 dB (no inf) | **PASS** |
| ΔDNSMOS OVRL | +0.4085 | [+0.3491, +0.4685] | >= -0.05 (no inf) | **PASS** |
| ΔDNSMOS SIG | +0.3595 | [+0.3060, +0.4152] | >= -0.05 (no inf) | **PASS** |
| ΔDNSMOS BAK | +0.2698 | [+0.2244, +0.3184] | >= -0.07 (no inf) | **PASS** |
| ΔLSD > 8 kHz (dB)| +0.3388 | [+0.2330, +0.4489] | <= 0.5 dB (no sup) | **PASS** |

---

## 5. Gate G2 — Falante Concorrente (C3 / C4, SIR <= 5 dB)

O Gate G2 é o critério primário de eficácia da personalização por voz, medindo a capacidade do modelo de isolar a voz cadastrada na presença de um falante concorrente do mesmo sexo declarado a SIR <= 5 dB.

- **Condições:** C3 (concorrente sem ruído) e C4 (concorrente com ruído de fundo a 10 dB SNR).
- **Métricas:**
  - **ΔSI-SDR (Candidato com Perfil − DFNet3 Upstream):**
    - Média pontual: +2.29 dB (Exigido: >= +3.0 dB)
    - Limite inferior do IC 95%: +1.46 dB (Exigido: > 0.0 dB)
  - **ΔDNSMOS-P (OVRL Personalizado):**
    - Ganho médio: +0.5210 (Exigido: >= +0.15)
  - **Taxa de Erro de Palavras (WER com Whisper small ONNX):**
    - WER Candidato: 48.99%
    - WER DFNet3: 50.73%
    - WER Ruidoso: 35.06%
    - Razão (Candidato / DFNet3): 0.9658 (Exigido: <= 0.85)
    - Condição Não-Degradação: Candidato <= Ruidoso (False)

---

## 6. Gate G3 — Sobre-Supressão de Fala-Alvo (TSOS)

Mede o risco de atenuação indevida da fala do interlocutor alvo (Target Speech Over-Suppression) segundo Eskimez et al. 2021.

- **Decisão do Dono (Opção A):** Convenção fixa mantida (áudio float [-1, 1], escala 1.0, p=0.3, gamma=0.1, STFT 20/10 ms).
- **Regra Relativa Pré-Registrada:** O acréscimo de TSOS do pDFNet3 sobre o DFNet3 upstream não pode exceder **+0,5 ponto percentual** em condições de voz limpa (C0c).
- **Resultados:**
  - TSOS Candidato (C0c): 7.10%
  - TSOS DFNet3 (C0c): 9.92%
  - ΔTSOS (Candidato − DFNet3): -2.82 p.p. (Exigido: <= +0.5 p.p.)
  - TSOS Candidato no modo sem perfil (C0): 17.74% (Exigido: <= 1.0%)

---

## 7. Gate G4 — Locutor Trocado (C7, TRR e Separação de Modos)

Verifica a rejeição de falantes não-cadastrados quando apenas o falante B fala, mas o perfil carregado pertence ao falante A (Cenário C7, sem `atten_lim_db`).

- **Target Rejection Rate (TRR a 15 dB):**
  - Pares de Sexo Diferente: 0.8458 (Exigido: >= 0.90)
  - Pares de Mesmo Sexo: 0.4228 (Exigido: >= 0.70)
- **Separação de Modos Mediana:**
  - Diferença de atenuação mediana (com perfil de A − sem perfil): +45.22 dB (Exigido: >= 15.0 dB)
- **Curva de TRR por Limiar de Atenuação:**
  - 6 dB: Dif 0.899, Mesmo 0.502
  - 10 dB: Dif 0.873, Mesmo 0.460
  - 15 dB: Dif 0.846, Mesmo 0.423
  - 20 dB: Dif 0.823, Mesmo 0.396

---

## 8. Resumo das Ablações (AB2 e AB1)

### 8.1 Matriz AB2 (Ramos de Regularização e Escala FiLM)
A matriz AB2 explorou a taxa de descarte de embedding (`p_drop`) e a escala da perda FiLM (`s`):
- **Stage 1 (p_drop):** Comparação entre `ab2-pdrop01` (p=0.1), `t3` (p=0.2) e `ab2-pdrop03` (p=0.3).
- **Stage 2 (escala s):** Fixado o melhor `p_drop`, comparação entre `ab2-s00` (s=0.0) e `ab2-s10` (s=0.10).
- **Decisão Final:** Checkpoint eleito `t3` por maximizar o ΔSI-SDR em C3 com elegibilidade plena no G1.

### 8.2 Controle AB1 (Denoise no Enrollment)
- Comparação indicativa entre braço `p_enroll_denoise = 0.0` e `p_enroll_denoise = 0.5` (1 seed por braço, ~13 h cada).
- Status indicativo reportado no relatório da Tarefa 7.

---

## 9. Observabilidade e Diagnóstico Técnico

- **Frações de Estágio do Gating:**
  - `ERB_DF`: 85.0%
  - `ERB_ONLY`: 4.0%
  - `ZERO`: 10.0%
  - `NONE`: 1.0%
- **Saturação do Tanh (C8):** 0.50% dos valores FiLM em |tanh| > 0.95 (Exigido: <= 5.0%).
- **Preservação de Voz (G5 Informativo):** Δ similaridade de falante WeSpeaker IC inf: +0.0000.
- **DNSMOS-P Médio (G8 Informativo):** 3.65.

---

## 10. Conclusão e Decisão de Marco

Com base nas verificações formais executadas no `dev-gate`:

> **DECISÃO DO MARCO M2:** NO-GO M2 — Procedimento de parada e diagnóstico acionado.
