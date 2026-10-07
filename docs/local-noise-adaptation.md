# Pipeline de Adaptação Acústica e Supressão de Ruído Local

Este documento descreve o processo de treinamento e adaptação contínua do modelo **pDFNet3** para novos perfis de ruído acústico, microfones locais e vazamento de vozes de fundo.

---

## 1. Visão Geral da Arquitetura

O processo de adaptação utiliza o modelo pré-treinado **pDFNet3 (DeepFilterNet3 com condicionamento FiLM)**.
Durante a adaptação a novos perfis acústicos:

1. **Encoder Congelado:** As camadas convolucionais de extração de características acústicas (`base.enc`) e as normalizações BatchNorm permanecem rigorosamente congeladas, preservando as representações semânticas e espectrais fundamentais.
2. **Ajuste Focado nos Decoders:** Apenas os decoders de ganho espectral (`erb_dec`) e filtragem complexa profunda (`df_dec`) são atualizados via AdamW com regularização $L_2$.
3. **Multi-Resolução STFT Loss:** A função de perda combina:
   - Perda multi-resolução de magnitude e fase complexa ($\mathcal{L}_{\text{MR}}$).
   - Perda contra sobre-supressão de voz ($\mathcal{L}_{\text{OS}}$), garantindo que formantes de fala limpa não sejam atenuados.
4. **Early Stopping com Split Holding-Out:** A convergência é avaliada a cada 100 passos em fatias acústicas de teste completamente segregadas, interrompendo o treinamento automaticamente quando a perda atinge o menor valor histórico.

---

## 2. Etapas do Processo de Treinamento

### 2.1 Preparação dos Ruídos Locais (`scripts/process_local_noise.py`)
Fatia arquivos de gravação contínua em janelas temporais de 6.0 segundos e aplica augmentações acústicas determinísticas:
- Variações de ganho de entrada (-6 dB, 0 dB, +3 dB).
- Filtros passa-baixas (simulação de distâncias acústicas maiores) e passa-altas.
- Pequenas variações de velocidade / time-stretch (0.95x a 1.05x).

```bash
python scripts/process_local_noise.py \
  --input-videos /caminho/para/gravacoes/*.mp4 \
  --output-dir data/raw/noise-local-mics/ \
  --segment-seconds 6.0
```

> **Aviso de Sigilo:** Arquivos de gravação pessoal e amostras brutas de ruídos reais ficam estritamente restritos a diretórios ignorados pelo Git (`data/raw/` e `.gitignore`). Apenas o pipeline de código é compartilhado.

---

### 2.2 Expansão Acústica Combinatória (`scripts/expand_noise_dataset.py` e `scripts/synthesize_babble_noise.py`)
Para impedir sobreajuste (overfitting) a um ruído estático específico, o pipeline gera misturas compostas:
1. **Ruídos Compostos com RIR:** Convolui os ruídos com respostas ao impulso de salas acústicas reais e soma diferentes fontes (ruído mecânico + ruído de ambiente).
2. **Babble Noise Multi-falante:** Sobrepõe segmentos de múltiplos falantes aleatórios a níveis variados de SNR para criar cenários de cafeteria e conversas cruzadas.

---

### 2.3 Treinamento com Early Stopping (`scripts/train_local_adapt.py`)
Executa o fine-tuning supervisionado dos decoders:

```bash
python scripts/train_local_adapt.py \
  --noise-dir data/raw/noise-local-mics data/raw/noise-composite data/raw/noise-babble-synthetic \
  --speech-path data/raw/cml-tts-pt/dev/audio data/raw/librispeech-dev-clean/dev-clean \
  --batch-size 24 \
  --lr 1.2e-4 \
  --steps 4000 \
  --val-every 100 \
  --patience 6 \
  --max-checkpoints 2 \
  --device cuda
```

- `--max-checkpoints 2`: Limita a retenção de checkpoints intermediários para preservar espaço em disco.
- O melhor checkpoint histórico é salvo de forma definitiva em `runs/local_adapt/checkpoints/ckpt-adapt-best.pt`.

---

### 2.4 Avaliação Objetiva de Métricas (`scripts/poc_mix_and_eval.py`)
Avalia o modelo treinado em um conjunto de teste holding-out sob diferentes faixas de SNR (-5 dB a +10 dB):

```bash
python scripts/poc_mix_and_eval.py \
  --checkpoint runs/local_adapt/checkpoints/ckpt-adapt-best.pt \
  --noise-dir data/raw/noise-local-mics \
  --num-samples 25 \
  --snrs -5 0 5 10 \
  --out-dir runs/poc_eval_report
```

Gera métricas objetivas de:
- **$\Delta$SI-SDR:** Ganho de relação sinal-distorção independente de escala.
- **$\Delta$STOI:** Ganho de inteligibilidade objetiva da fala.
- **Atenuação Residual:** Redução em decibéis de ruídos mecânicos e vozes de fundo.

---

## 3. Exportação e Publicação no GitHub Actions

A exportação de modelos em formato binário ONNX e compilação de runtimes (TensorRT, DirectML, OpenVINO, CoreML) é realizada automaticamente via **GitHub Actions** no repositório oficial [`pdfnet3`](https://github.com/joaorura/pdfnet3):

- **Disparo:** Pushes de tags `v*` (ex: `v1.1.0`) ou acionamento manual via `workflow_dispatch`.
- **Workflow:** `.github/workflows/release-models.yml`.
- **Artefatos Produzidos:**
  - `pdfnet3-onnx-models.tar.gz`: Modelos ONNX causais e stateful (`enc.onnx`, `df_dec.onnx`, `erb_dec.onnx`, `config.ini`).
  - `pdfnet3-all-runtimes.tar.gz`: Pacote universal para inferência com múltiplos Execution Providers.
  - Checksums SHA-256 e notas de versão publicados automaticamente no GitHub Releases.
