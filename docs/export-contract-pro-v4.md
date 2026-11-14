# Contrato de export: `pdfnet3-pro-v4-presence` (stateful, S=1)

Asset gerado por `scripts/m3_export_pro_v4.py` (código em `src/cctrain/export/pdfnet3_pro_v4.py`). É o contrato v3
(`docs/export-contract-pro-v3.md`) **mais** a presença do falante-alvo no grafo `enc` e o gating por presença. Tudo o
que não está escrito aqui é idêntico ao v3: DSP (48 kHz, hop 480, FFT 960, 32 ERB, 96 bins de DF, ordem 5),
`config.ini`, condicionamento (`speaker_embedding` [1,192] ECAPA L2-normalizado, `speaker_present` [1]), decoders,
roteamento, opset 12, estados `<nome>_in/<nome>_out` em zeros no início.

## O que muda

| Grafo | Novo | Forma | Significado |
|---|---|---|---|
| `enc.onnx` | saída `p_target` | `[1,1,1]` f32 | probabilidade (sigmoide) de o falante-alvo estar ativo no quadro |
| `enc.onnx` | estado `p_h_in` / `p_h_out` | `[1,1,32]` f32 | estado da GRU da cabeça de presença (zeros no início) |

Por que no `enc`: a cabeça lê a saída da cross-attention (`emb`, a mesma que vai aos decoders) e o
`speaker_embedding`, e a cross-attention já está no `enc` (v3). Não há grafo novo nem entrada nova. O runtime que
realimenta os pares de estado pelo nome (como manda o v3) não precisa de código novo para o `p_h`; só precisa ler
`p_target`.

Cabeça (`PresenceHead`): `z = relu(W_in·emb)` (512→64), `s = W_s·e` (192→64), `GRU([z, z⊙s])` (128→32), `Linear`
(32→1), sigmoide. Custo extra: **60.448 MACs/hop** (~1,2% dos ~5 M MACs/hop do v3), 60.769 parâmetros.

Sem perfil (`speaker_present`=0) a saída existe, mas não é usada.

## Gating (`manifest.json` → `gating`)

```json
"gating": {"mode": "presence", "output": "p_target", "graph": "enc",
           "tau_on": 0.3, "tau_off": 0.1, "hangover_frames": 12,
           "applies_when": "speaker_present == 1",
           "no_profile": {"mode": "lsnr", "min_db": -10.0, "max_db_erb": 30.0, "max_db_df": 20.0}}
```

Com `speaker_present` = 1, faça por quadro ativo (fora do atalho de silêncio digital):

```
ativo = (p_target >= tau_off) se ativo no quadro anterior, senão (p_target >= tau_on)   # histerese
hang  = hangover_frames se ativo, senão max(hang - 1, 0)                              # hangover
estágio = ERB+DF se (ativo ou hang > 0), senão ZERO
```

Estado inicial `ativo=false, hang=0`; zere ao trocar de perfil ou no reset do stream. Com perfil não há passthrough
nem ERB-only. Sem perfil vale o gating por `lsnr` de sempre (ZERO < −10, ruidoso > 30, só ERB em (20, 30]). Os três
grafos continuam rodando em todo quadro ativo (decisão de 2026-11-09): o gating só seleciona a saída. Asset sem
`gating.mode == "presence"` mantém o comportamento do v3.

Os valores de τ/hangover do asset de teste saíram da varredura em dev do micro fine-tune (500 passos), com a meta
"% ZERO no alvo ≤ 2%" (não atingida na grade: o melhor ponto deu 3,1% no alvo / 26% no só-interferente): são provisórios e serão recalculados no checkpoint final do treino longo (o runtime deve
ler do manifest, nunca fixar no código).

## Rótulo e treino da presença (para referência)

Rótulo = `target_active` do `mix_v4`: VAD por energia relativa da trilha do alvo limpo já escalonada (quadro de 10 ms
a até 30 dB do máximo da própria trilha, piso −75 dBFS), o mesmo critério em qualquer nível do alvo. BCE com pesos
por classe de quadro (só-alvo 0,25, sobreposição 0,25, só-interferente 0,30, silêncio/ruído 0,20). O treino emula a
mesma máquina de estados (`scripts/train_pro_v4_presence.py`).

## Asset de teste

`pdfnet3-pro-v4-presence-stateful-asset-v1.tar.gz` (tar determinístico, reprodutível), a partir de
`runs/m4/v4-presence-smoke-c/checkpoints/ckpt-00000500.pt` (micro fine-tune de 500 passos do v3.1: **não é um modelo
de produção**, serve para integrar o runtime). Membros: `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`, `config.ini`,
`manifest.json` (com `gating` e `presence`).

## Gate de paridade

`scripts/gate_pro_v4_parity.py` (adaptado do v3): forward PyTorch x grafos hop a hop em ONNX Runtime, OpenVINO CPU e
tract 0.19.16 (`tools/tract-check stream`, que agora grava `p_target`), perfis sem perfil / alvo / outro locutor;
SNR de `m`, `coefs`, `lsnr`, `p_target` e do áudio com a seleção de estágio pelo gating de presença do manifest.
Meta ≥ 100 dB em fp32. Asset de teste (sha256 `367165f8…2c92`): **PASS**, mínimo 116,3 dB entre ORT, OpenVINO CPU e tract, 0 divergências de
estágio no áudio.
