# Plano de Execução: Marco M3 — Exportação Reproduzível dos Assets ONNX (pDFNet3 e Enrollment)

**Data:** 2026-10-05  
**Status:** APROVADO PELO DONO  
**Repositório:** `/home/joaorura/orca/projects/clearcore-train`  
**Checkpoint Base:** `runs/m2/t3/checkpoints/ckpt-00022500.pt` (eleito na AB2, T7/T8)  
**Decisão do Dono:** Avançar formalmente para o Marco M3 com o checkpoint do Estágio F (+2,29 dB SI-SDR, +0,52 MOS, +45 dB separação de modos).

---

## 1. Objetivos do Marco M3

1. **Exportação do `pDFNet3` (Estágio F retreinado com FiLM):**
   - Inserir os nós FiLM (`Mul` e `Add`) nos sítios `/emb_gru/linear_in/1/Relu_output_0` e `/df_gru/linear_in/linear_in.1/Relu_output_0`.
   - Carregar os pesos retreinados do checkpoint do Estágio F (`ckpt-00022500.pt`) nos initializers do ONNX.
   - Gerar os membros canônicos: `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx` e `config.ini`.
   - Empacotar no formato USTAR gzip determinístico: `runs/m3/pdfnet3-release-asset-v1.tar.gz`.

2. **Exportação do `enrollment.onnx` (SpeechBrain ECAPA + FiLM Generator):**
   - Áudio mono 16 kHz $\to$ `gamma_enc`, `beta_enc`, `gamma_df`, `beta_df` e `embedding` (192d).
   - Empacotar no formato canônico: `runs/m3/voice-enrollment-asset-v1.tar.gz`.

3. **Verificações e Gates do Marco M3:**
   - **G0 (Contrato e Paridade Tract):** Carga no `tract 0.19.16` sem erros de operadores; hash do `config.ini` verificado.
   - **G11 (Contrato do Enrollment):** Saídas dentro dos limites válidos com `embedding` normalizado L2 para clipes de 2 a 12 s.
   - **Reprodutibilidade Determinística (Dois Exports Consecutivos):** Duas execuções independentes de empacotamento geram exatamente o mesmo SHA-256 (bit a bit).
   - **Paridade Numérica (P1/P2):** Comparação entre a réplica PyTorch e a inferência Tract/ONNX no dev set.

---

## 2. Tarefas do Marco M3

| Tarefa | Descrição | Saída Verificável |
|---|---|---|
| **M3-T1** | Script de exportação do Estágio F (`scripts/m3_export_pdfnet3.py`) injetando pesos retreinados no grafo FiLM | `runs/m3/members/` com os 4 membros válidos |
| **M3-T2** | Empacotamento determinístico e teste de duplo export | `runs/m3/pdfnet3-release-asset-v1.tar.gz` com SHA-256 imutável |
| **M3-T3** | Exportação e empacotamento do `voice-enrollment-asset-v1.tar.gz` | `runs/m3/voice-enrollment-asset-v1.tar.gz` |
| **M3-T4** | Testes de Paridade Tract (G0 e G11) e Gate M3 | `runs/m3/m3_report.json` com status de todos os checks |
