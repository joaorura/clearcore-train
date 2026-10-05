# Marco M3: Exportação Reproduzível dos Assets ONNX (pDFNet3 e Enrollment) e Paridade com Tract

Data de consolidação: 2026-10-05 04:01:00 UTC  
Status: **MARCO M3 CONCLUÍDO COM SUCESSO (TODOS OS GATES APROVADOS)**  
Repositório: `/home/joaorura/orca/projects/clearcore-train`  
Checkpoint eleito do Estágio F: `runs/m2/t3/checkpoints/ckpt-00022500.pt` (SHA-256 `d5c0fbd7983534e909a991fe68eed17347ae6eb82becd4b2f2904ed86a6d4f7e`)  
Asset upstream de referência: `vendor/approved/df-compatible-release-asset-v1.bin` (SHA-256 `c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616`)  

---

## 1. Resumo Executivo

O Marco M3 foi executado em sua totalidade, gerando os pacotes finais determinísticos para distribuição do **Clearcore Realtime Voice Isolation**:

1. **`pdfnet3-release-asset-v1.tar.gz` (7.979.862 bytes):**
   - SHA-256: `42dfc577fdf8a881ecbafce7777bf6f0a4cf914ffc1aaff2580aec0cbac79505`
   - Incorpora os 54 parâmetros retreinados do Estágio F (eleito na AB2 com $+1,79\text{ dB}$ em C3) sobre a topologia canônica compatível com o Tract 0.19.16.
   - Nós FiLM inseridos nos sítios `/emb_gru/linear_in/1/Relu_output_0` e `/df_gru/linear_in/linear_in.1/Relu_output_0`.
   - **Reprodutibilidade Determinística:** 100% idêntica bit-a-bit em execuções consecutivas independentes.
   - **Gate G0:** APROVADO com 0 incompatibilidades de operadores no Tract e 0 *stage mismatches* contra a réplica PyTorch em 1.000 quadros.

2. **`voice-enrollment-asset-v1.tar.gz` (79.406.455 bytes):**
   - SHA-256: `bd4d6dd941f8527b5011a2bae78169148f33155e25d30c5707e88963e7ea824d`
   - Contém o membro canônico `enrollment.onnx` (85.271.697 bytes, SHA-256 `1e43b16bc037e548fc7afbfa80c495886a75f494c1504c7965715f5fa6fdaece`).
   - Integra o encoder SpeechBrain ECAPA-TDNN portado com o gerador FiLM retreinado do Estágio F.
   - **Reprodutibilidade Determinística:** 100% idêntica bit-a-bit.
   - **Gate G11:** APROVADO para todas as durações de 2 a 12 s (erro máx. $\le 4,53 \times 10^{-6}$, norma L2 do embedding $= 1,000000$) e para todos os 6 sinais de borda (sem NaN/Inf, excesso relativo $\le 0$).

---

## 2. Inventário dos Assets e Membros Exportados

### 2.1 Pacote `pdfnet3-release-asset-v1.tar.gz`
- **Caminho:** `runs/m3/pdfnet3-release-asset-v1.tar.gz`
- **Tamanho:** 7.979.862 bytes
- **SHA-256:** `42dfc577fdf8a881ecbafce7777bf6f0a4cf914ffc1aaff2580aec0cbac79505`
- **Formato:** Tar USTAR compactado com Gzip (`mtime=0`, `uid=0`, `gid=0`, modo 0644).

| Membro | Tamanho (bytes) | SHA-256 | Descrição |
| :--- | :---: | :---: | :--- |
| `enc.onnx` | 1.954.321 | `d823ae2ba955f10873e9a7fcf48ed7b54bb2da088ef7878231d85b2c3a4b7e29` | Encoder pDFNet3 com FiLM + pesos retreinados Estágio F |
| `erb_dec.onnx` | 3.292.397 | `efd60370f69f481c365962120c64a791d511bad8de70fb500bb40f1c62f13869` | Decodificador ERB com pesos retreinados Estágio F |
| `df_dec.onnx` | 3.341.118 | `7faa6a0bbded9adc73d4bc0342f246a5a1934a686ffc087bf7cbf44ac5845622` | Decodificador DF com FiLM + pesos retreinados Estágio F |
| `config.ini` | 2.067 | `415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290` | Configuração canônica v0.5.6 (idêntica ao asset aprovado) |

### 2.2 Pacote `voice-enrollment-asset-v1.tar.gz`
- **Caminho:** `runs/m3/voice-enrollment-asset-v1.tar.gz`
- **Tamanho:** 79.406.455 bytes
- **SHA-256:** `bd4d6dd941f8527b5011a2bae78169148f33155e25d30c5707e88963e7ea824d`

| Membro | Tamanho (bytes) | SHA-256 | Descrição |
| :--- | :---: | :---: | :--- |
| `enrollment.onnx` | 85.271.697 | `1e43b16bc037e548fc7afbfa80c495886a75f494c1504c7965715f5fa6fdaece` | Áudio mono 16 kHz $\to$ `gamma_enc, beta_enc, gamma_df, beta_df` e `embedding` (192d) |

---

## 3. Detalhes de Implementação e Engenharia (M3-T1 a M3-T4)

### 3.1 M3-T1: Injeção de Pesos do Estágio F no Grafo pDFNet3
- **Preservação de Compatibilidade Tract:** Como o `torch.onnx.export` do PyTorch 2.x altera operadores e quebra a compatibilidade estrita do Tract 0.19.16, a exportação utilizou o grafo aprovado e testado do Clearcore como base, inserindo os nós FiLM (`Mul` e `Add`) nos sítios `/emb_gru/linear_in/1/Relu_output_0` e `/df_gru/linear_in/linear_in.1/Relu_output_0`.
- **Mapeamento Matemático Exato:**
  - **Convs e Convs Transpostas:** As camadas convolucionais seguidas por BatchNorm congelado foram dobradas de acordo com:
    $$W_{folded} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}, \quad b_{folded} = \beta - \mu \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$$
  - **GRUs:** Os pesos e biases das camadas GRU foram convertidos da ordem PyTorch `[r, z, n]` para a ordem ONNX `[z, r, h]` com transposição e concatenação para `[1, 3 \times H, input\_size]`.
  - **Camadas Lineares:** Transposição correta para os nós `MatMul` e injeção direta dos pesos `Einsum`.
  - **Paridade Numérica com PyTorch:** A diferença máxima entre o ONNX gerado e o modelo PyTorch em inferência foi de $4,77 \times 10^{-7}$ no `erb_dec` e $2,38 \times 10^{-7}$ no `df_dec`.

### 3.2 M3-T2: Reprodutibilidade Determinística dos Pacotes
- Duas rodadas consecutivas e independentes de exportação foram realizadas em diretórios limpos e comparadas:
  - `pdfnet3-release-asset-v1.tar.gz` export 1: `42dfc577fdf8a881ecbafce7777bf6f0a4cf914ffc1aaff2580aec0cbac79505`
  - `pdfnet3-release-asset-v1.tar.gz` export 2: `42dfc577fdf8a881ecbafce7777bf6f0a4cf914ffc1aaff2580aec0cbac79505`
  - **Resultado:** Idêntico bit-a-bit (0 bytes de desvio).
  - `voice-enrollment-asset-v1.tar.gz` export 1: `bd4d6dd941f8527b5011a2bae78169148f33155e25d30c5707e88963e7ea824d`
  - `voice-enrollment-asset-v1.tar.gz` export 2: `bd4d6dd941f8527b5011a2bae78169148f33155e25d30c5707e88963e7ea824d`
  - **Resultado:** Idêntico bit-a-bit.

---

## 4. Resultados dos Gates de Qualidade

### 4.1 Gate G0: Contrato e Paridade Numérica P2 no Tract (pDFNet3)
Avaliação sobre 1.000 quadros (480.000 amostras) de áudio de mistura real utilizando o binário `tract-check 0.19.16`:

| Perfil FiLM | Erro Máximo de Áudio | Pior Excesso ($|d| - (10^{-4} + 10^{-4}|ref|)$) | Descompassos de Estágio | Status |
| :--- | :---: | :---: | :---: | :---: |
| `neutral` ($\gamma=1, \beta=0$) | $1,79 \times 10^{-7}$ | $-9,99 \times 10^{-5}$ | 0 | **PASS** |
| `random` (gerador com ruído) | $2,38 \times 10^{-7}$ | $-9,99 \times 10^{-5}$ | 0 | **PASS** |
| `extreme_enc_up_df_down` | $6,98 \times 10^{-7}$ | $-9,95 \times 10^{-5}$ | 0 | **PASS** |
| `extreme_enc_down_df_up` | $0,00 \times 10^{0}$ | $-1,00 \times 10^{-4}$ | 0 | **PASS** |

- **Integridade do `config.ini`:** Conferido contra SHA-256 `415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290` (**PASS**).
- **Carga no Tract:** Sem falhas de operadores ou nós desconhecidos (**PASS**).

### 4.2 Gate G11: Contrato e Estabilidade Numérica do Enrollment

#### A. Teste de Duração Dinâmica no Tract 0.19.16
Entrada de áudio sintético modelado como fala (16 kHz) em várias durações:

| Duração | Erro Máx. Tract $\times$ PyTorch | Norma L2 Embedding | Intervalo $\gamma_{enc}$ | Intervalo $\beta_{enc}$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2 s** | $1,91 \times 10^{-6}$ | 1,000000 | $[0,16; 2,26]$ | $[-0,75; +0,87]$ | **PASS** |
| **6 s** | $2,15 \times 10^{-6}$ | 1,000000 | $[0,16; 2,36]$ | $[-0,74; +0,87]$ | **PASS** |
| **8 s** | $4,53 \times 10^{-6}$ | 1,000000 | $[0,16; 2,37]$ | $[-0,72; +0,85]$ | **PASS** |
| **10 s** | $3,81 \times 10^{-6}$ | 1,000000 | $[0,16; 2,36]$ | $[-0,75; +0,82]$ | **PASS** |
| **12 s** | $4,41 \times 10^{-6}$ | 1,000000 | $[0,16; 2,33]$ | $[-0,74; +0,84]$ | **PASS** |

#### B. Robustez a Sinais de Borda no Tract
Avaliação contra entradas degeneradas:

| Sinal de Borda | Finito? (Sem NaN/Inf) | Erro Máximo Absoluto | Excesso Relativo P2 | Status |
| :--- | :---: | :---: | :---: | :---: |
| Silêncio puro | Sim | $3,58 \times 10^{-6}$ | $-9,90 \times 10^{-5}$ | **PASS** |
| Componente DC (+0,1) | Sim | $8,82 \times 10^{-6}$ | $-9,90 \times 10^{-5}$ | **PASS** |
| Onda quadrada em escala plena | Sim | $4,43 \times 10^{-5}$ | $-9,32 \times 10^{-5}$ | **PASS** |
| Ruído branco (pico 0,99) | Sim | $4,77 \times 10^{-6}$ | $-9,98 \times 10^{-5}$ | **PASS** |
| Ruído a -90 dBFS | Sim | $1,14 \times 10^{-5}$ | $-9,89 \times 10^{-5}$ | **PASS** |
| Impulso único isolado | Sim | $1,11 \times 10^{-4}$ | $-7,39 \times 10^{-5}$ | **PASS** |

---

## 5. Bateria de Testes Automatizados

A suíte completa de testes unitários foi implementada em `tests/test_m3_export.py` e executada via `pytest`:

```bash
.venv/bin/python -m pytest tests/test_m3_export.py
======================== 5 passed, 7 warnings in 24.40s ========================
```

Testes aprovados:
1. `test_pdfnet3_members_contract_and_structure`: Validação dos 4 membros, hash do `config.ini`, e conformidade do ONNX checker.
2. `test_pdfnet3_reproducibility`: Validação da reprodutibilidade bit-a-bit do pacote `pdfnet3-release-asset-v1.tar.gz`.
3. `test_voice_enrollment_reproducibility`: Validação da reprodutibilidade bit-a-bit do pacote `voice-enrollment-asset-v1.tar.gz`.
4. `test_g0_tract_gating_parity`: Teste de 1.000 quadros Tract vs Réplica nos 4 perfis FiLM.
5. `test_g11_enrollment_contract_and_tract_parity`: Teste de todas as durações e casos de borda no Tract.

---

## 6. Conclusão e Recomendação

Os assets finais de inferência do Marco M3 estão **validados, reproduzíveis e prontos para produção**. Todos os critérios contratuais com o motor de áudio Tract e a biblioteca de runtime do Clearcore foram estritamente cumpridos. Recomenda-se avançar imediatamente para o empacotamento desktop/turnkey.
