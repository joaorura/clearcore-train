# Clearcore Train — relatório M0 (piloto)

## Resumo

- disco: **go (condicional — margem 6.29 GB; itens fora da projeção listados abaixo)**; ver "Go/no-go do orçamento"
- vram_b16: **go**
- ram_loader: **go**
- tempo_t3: **go**
- I1: **aprovado**; I2 (asset do repo): **aprovado**
- B fixado: 4.0
- Réplica do gating: paridade ok
- ECAPA no tract (pooling): expand: aprovado; broadcast: aprovado

## Decisões do orquestrador

- **Common Voice:** o Common Voice 27.0 pt (`cv27-pt`, filtro pt-BR "Portuguese (Brasil)" por metadado) substitui o
  26.0: as versões são cumulativas e o 26.0 exigia termos separados (decisão do orquestrador, não medido aqui).
- **Ruído do piloto:** os 3 ruídos CC0 do Clearcore são split `test` (não treináveis). O ruído treinável do piloto é
  sintético (`noise-synthetic-v1`, `LicenseRef-cctrain-generated`) e serve só para medir custo (decisão do
  orquestrador, não medido aqui).
- **Desvio emulação offline × runtime:** informativo no M0, sem gate (o `budget-v1` está congelado por SHA-256). Valores
  medidos em `offline_vs_runtime.json`: erro/sinal de -17,5 dB e
  92,6% de concordância de estágio (silêncio fora da conta:
  3,8% dos quadros). Alimenta a definição do gate antes do M2.
- **RIR `synthetic-v1`:** RT60 medido de 0,10 a 0,61 s (mediana
  0,38 s) em 64 RIR do piloto, com `max_order` limitado a
  20 (lido de `rir-synthetic-v1-pilot.params.jsonl`). A cauda longa não existe: aceito no M0
  porque o piloto mede custo, não qualidade (decisão do orquestrador).
- **Mistura:** convenção de SNR (ruído escalado contra alvo + interferente) e alvo seco não alinhado à RIR aceitos no
  M0, a conferir contra a spec §4.4 (decisão do orquestrador, não medido aqui).
- **Splits e loader:** `training_records` recusa `test`/`dev`; o loader tem `allowed_splits`.
- **Ruling C5 (16 kHz):** o ruling C5 do orquestrador tirou o caminho de 16 kHz (LibriSpeech dev-clean, split `dev`)
  do loader e da medição de B: o treino e a medição de B não consomem nada a 16 kHz. O fator FLAC a 16 kHz do
  LibriSpeech continua só na projeção de disco (medição de espaço, não de treino).
- **Unidade de disco:** a reserva do budget (`disk.reserve_gb: 10`) e a guarda `cctrain.disk` usam GB decimais
  (1 GB = 10^9 B), a mesma unidade do relatório; a guarda usava GiB (reserva de 10,74 GB), o que deixava a margem
  reportada inconsistente com a guarda.
- **Go/no-go:** segue estritamente os níveis congelados de `configs/m0/budget-v1.yaml` (verificados por SHA-256);
  nenhum nível foi ajustado depois das medições.

## Ambiente

```json
{
  "python": "3.11.16",
  "torch": "2.11.0+cu128",
  "cuda": "12.8",
  "gpu": "NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU",
  "ram_total_gb": 30.790889739990234,
  "ram_available_gb": 3.9514808654785156,
  "swap_used_gb": 44.731407165527344
}
```

## Identidade (I1, I2)

```json
{
  "i1": {
    "test": "I1",
    "device": "cpu",
    "deterministic": true,
    "max_abs_diff_per_output": [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "pass": true
  },
  "i2_fixture_clearcore": {
    "approved_bit_exact_vs_golden": true,
    "approved_out_sha256": "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa",
    "film_default_out_sha256": "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa",
    "film_explicit_out_sha256": "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa",
    "film_hidden": 256,
    "frames": 1000,
    "golden_sha256": "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa",
    "max_abs_diff_default_vs_golden": 0.0,
    "max_abs_diff_explicit_vs_approved": 0.0,
    "max_abs_diff_explicit_vs_golden": 0.0,
    "pass": true
  },
  "i2_asset_do_repo": {
    "approved_bit_exact_vs_golden": true,
    "approved_out_sha256": "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa",
    "film_default_out_sha256": "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa",
    "film_explicit_out_sha256": "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa",
    "film_hidden": 256,
    "frames": 1000,
    "golden_sha256": "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa",
    "max_abs_diff_default_vs_golden": 0.0,
    "max_abs_diff_explicit_vs_approved": 0.0,
    "max_abs_diff_explicit_vs_golden": 0.0,
    "pass": true,
    "asset_sha256": "17cfdd1e83c52ffb8a53d07bc69fa99946b268a0269a2b667434032075c94ebb",
    "asset_size": 7982867,
    "returncode": 0
  },
  "film_asset": {
    "sha256": "17cfdd1e83c52ffb8a53d07bc69fa99946b268a0269a2b667434032075c94ebb",
    "size": 7982867,
    "members_equal_to_clearcore_fixture": {
      "enc.onnx": true,
      "erb_dec.onnx": true,
      "df_dec.onnx": true,
      "config.ini": true
    }
  }
}
```

## Congelamentos

```json
{
  "gates_v1_sha256": "4919f879dc0d34d96270c2ee882b99683affe6304f6d9a03381d2bfa57cd1c13",
  "budget_v1_sha256": "ff007e54641f7ef17f22081cdbb4bd96c93c7fdee0221412288187666fdd0545",
  "resampler_v1": {
    "dtype": "float32-le",
    "file": "coeffs.f32",
    "sha256": "d230da4dc7dd9ebbdee979dc33a11d3ee61cf2fe29477c768745442865fd7043"
  }
}
```

## Réplica do gating × tract-check

```json
{
  "frames": 1000,
  "max_abs_diff": 1.6391277313232422e-07,
  "worst_excess_over_tol": -9.99209878500551e-05,
  "worst_frame": 587,
  "first_frames_excess": [
    -9.999999747378752e-05,
    -9.999999747378752e-05,
    -9.999999747378752e-05,
    -9.999999747378752e-05,
    -9.999999747378752e-05,
    -9.999999747378752e-05,
    -9.999999747378752e-05,
    -9.999999747378752e-05,
    -9.999999747378752e-05,
    -9.999999747378752e-05
  ],
  "frames_over_tol": 0,
  "lsnr_worst_excess": -9.153203427558765e-05,
  "stage_mismatches": 0,
  "per_stage": {
    "zero": {
      "frames": 218,
      "worst_excess_over_tol": -9.9993783805985e-05
    },
    "erb_only": {
      "frames": 87,
      "worst_excess_over_tol": -9.99209878500551e-05
    },
    "erb_df": {
      "frames": 516,
      "worst_excess_over_tol": -9.994481661124155e-05
    },
    "none": {
      "frames": 139,
      "worst_excess_over_tol": -9.999844041885808e-05
    },
    "silent": {
      "frames": 40,
      "worst_excess_over_tol": -9.999999747378752e-05
    }
  },
  "fractions_tract": {
    "zero": 0.218,
    "erb_only": 0.087,
    "erb_df": 0.516,
    "none": 0.139,
    "silent": 0.04
  },
  "fractions_replica": {
    "zero": 0.218,
    "erb_only": 0.087,
    "erb_df": 0.516,
    "none": 0.139,
    "silent": 0.04
  },
  "pass": true
}
```

## Emulação offline × runtime (dado para o M2)

```json
{
  "max_abs_diff": 0.3279372453689575,
  "error_to_signal_db": -17.539694196709547,
  "stage_agreement": 0.9260416666666667,
  "stage_agreement_frames": 960,
  "silent_fraction_excluded": 0.038076152304609256,
  "offline_stage_fractions": {
    "zero": 0.254,
    "erb_only": 0.057,
    "erb_df": 0.55,
    "none": 0.139,
    "silent": 0.0
  },
  "note": "emulação offline sem o estado congelado dos decoders; usada no treino (spec §3.5)"
}
```

## Protótipo fbank + ECAPA no tract

```json
{
  "expand": {
    "all_pass": true,
    "falhas": []
  },
  "broadcast": {
    "all_pass": true,
    "falhas": []
  }
}
```

## Disco (fatores medidos e projeção do conjunto mínimo)

```json
{
  "medido": {
    "by_sr": {
      "16000": {
        "files": 2703,
        "hours": 5.387811319444447,
        "pcm16_bytes": 620675864,
        "flac_bytes": 359034309,
        "stored_bytes": 359034309,
        "codecs": [
          "flac"
        ],
        "flac_over_pcm16": 0.5784570173007404,
        "flac_gb_per_hour": 0.06663824839304526,
        "stored_gb_per_hour": 0.06663824839304526
      },
      "32000": {
        "files": 1547,
        "hours": 2.0011799999999993,
        "pcm16_bytes": 461071872,
        "flac_bytes": 157837556,
        "stored_bytes": 43303906,
        "codecs": [
          "mp3"
        ],
        "flac_over_pcm16": 0.34232744521010383,
        "flac_gb_per_hour": 0.07887224337640794,
        "stored_gb_per_hour": 0.021639185880330616
      },
      "48000": {
        "files": 440,
        "hours": 0.5633330439814819,
        "pcm16_bytes": 194687900,
        "flac_bytes": 94184238,
        "stored_bytes": 94455030,
        "codecs": [
          "flac",
          "mp3",
          "wav"
        ],
        "flac_over_pcm16": 0.48377037299185005,
        "flac_gb_per_hour": 0.16719104090598327,
        "stored_gb_per_hour": 0.1676717370108773
      }
    },
    "by_source": {
      "cv27-pt": {
        "32000": {
          "files": 1547,
          "hours": 2.0011799999999993,
          "pcm16_bytes": 461071872,
          "flac_bytes": 157837556,
          "stored_bytes": 43303906,
          "codecs": [
            "mp3"
          ],
          "flac_over_pcm16": 0.34232744521010383,
          "flac_gb_per_hour": 0.07887224337640794,
          "stored_gb_per_hour": 0.021639185880330616
        },
        "48000": {
          "files": 15,
          "hours": 0.018313333333333334,
          "pcm16_bytes": 6329088,
          "flac_bytes": 1677104,
          "stored_bytes": 528099,
          "codecs": [
            "mp3"
          ],
          "flac_over_pcm16": 0.26498351737248715,
          "flac_gb_per_hour": 0.09157830360393156,
          "stored_gb_per_hour": 0.028836858390971967
        }
      },
      "librispeech-dev-clean": {
        "16000": {
          "files": 2703,
          "hours": 5.387811319444447,
          "pcm16_bytes": 620675864,
          "flac_bytes": 359034309,
          "stored_bytes": 359034309,
          "codecs": [
            "flac"
          ],
          "flac_over_pcm16": 0.5784570173007404,
          "flac_gb_per_hour": 0.06663824839304526,
          "stored_gb_per_hour": 0.06663824839304526
        }
      },
      "noise-synthetic-v1": {
        "48000": {
          "files": 16,
          "hours": 0.044444444444444446,
          "pcm16_bytes": 15360000,
          "flac_bytes": 10709720,
          "stored_bytes": 10709720,
          "codecs": [
            "flac"
          ],
          "flac_over_pcm16": 0.6972473958333333,
          "flac_gb_per_hour": 0.2409687,
          "stored_gb_per_hour": 0.2409687
        }
      },
      "noise-wikimedia-cc0": {
        "48000": {
          "files": 3,
          "hours": 0.008333449074074075,
          "pcm16_bytes": 2880040,
          "flac_bytes": 1460477,
          "stored_bytes": 2880274,
          "codecs": [
            "wav"
          ],
          "flac_over_pcm16": 0.5071030263468563,
          "flac_gb_per_hour": 0.17525480590547352,
          "stored_gb_per_hour": 0.34562807961000536
        }
      },
      "vctk-0.92": {
        "48000": {
          "files": 406,
          "hours": 0.49224181712962994,
          "pcm16_bytes": 170118772,
          "flac_bytes": 80336937,
          "stored_bytes": 80336937,
          "codecs": [
            "flac"
          ],
          "flac_over_pcm16": 0.4722402828066499,
          "flac_gb_per_hour": 0.1632062417379781,
          "stored_gb_per_hour": 0.1632062417379781
        }
      }
    },
    "rir": {
      "count": 64,
      "stored_bytes": 2467385
    },
    "dirs_bytes": {
      "venv": 7301766861,
      "tract_check_target": 577054505,
      "rustup": 623763680,
      "data": 509926303,
      "third_party": 8667869
    },
    "free_bytes": 57973571584
  },
  "projecao": {
    "rows": [
      {
        "id": "vctk-0.92-mic1",
        "gb": 7.1810746364710365,
        "factor_source": "medido"
      },
      {
        "id": "tts-portuguese",
        "gb": 1.71366553824877,
        "factor_source": "medido"
      },
      {
        "id": "cml-tts-pt-cap30h",
        "gb": 2.723407401878354,
        "factor_source": "interpolado"
      },
      {
        "id": "librispeech-cap4min",
        "gb": 10.388902924475756,
        "factor_source": "medido"
      },
      {
        "id": "cv27-pt-mp3",
        "gb": 4.85,
        "factor_source": "fixo (pesquisa)"
      },
      {
        "id": "dns-noise-fullband-cc0",
        "gb": 1.6719104090598327,
        "factor_source": "medido"
      },
      {
        "id": "fsd50k-eval-cc0-ccby",
        "gb": 2.184950010122239,
        "factor_source": "interpolado"
      },
      {
        "id": "rir",
        "gb": 0.38552890625,
        "factor_source": "medido"
      }
    ],
    "measured_gb": {
      "venv": 7.301766861,
      "tract_check_target": 0.577054505,
      "rustup": 0.62376368
    },
    "estimated_gb": {
      "evaluators_whisper_dnsmos": 3.0,
      "test_sets": 3.0,
      "prerendered_dev_test": 3.0,
      "runs_checkpoints": 2.0
    },
    "total_gb": 50.60202487250599
  },
  "livre_mais_ja_usado_gb": 66.887541365,
  "ja_usado_gb": 9.012511349
}
```

## Loader

```json
{
  "batch_size": 16,
  "num_workers": 4,
  "prefetch_factor": 2,
  "batches": 30,
  "seconds": 2.2280031529953703,
  "samples_per_s": 215.43955149016676,
  "wall_s_per_hop": 1.5472244118023404e-05,
  "rss_max_gb": 2.391071319580078,
  "segment_s": 3.0
}
```

## Passo de treino na GPU

```json
{
  "step_b16": {
    "batch": 16,
    "teacher": false,
    "steps": 20,
    "s_per_step_median": 0.1575678325170884,
    "s_per_step_max": 0.21259016898693517,
    "peak_allocated_gb": 2.9276037216186523,
    "peak_reserved_gb": 3.421875,
    "rss_gb": 2.432933807373047,
    "loss_finite": true,
    "gpu": {
      "samples": 23,
      "max_temperature_c": 55.0,
      "min_sm_clock_mhz": 2077.0,
      "mean_power_w": 14.985652173913044,
      "max_memory_used_mib": 4068.0,
      "nonzero_throttle_reasons": [
        "0x0000000000000004",
        "0x0000000000000400"
      ]
    }
  },
  "step_b32": {
    "batch": 32,
    "teacher": false,
    "steps": 20,
    "s_per_step_median": 0.32406267200713046,
    "s_per_step_max": 0.3726509510015603,
    "peak_allocated_gb": 5.768388271331787,
    "peak_reserved_gb": 6.763671875,
    "rss_gb": 2.6551971435546875,
    "loss_finite": true,
    "gpu": {
      "samples": 10,
      "max_temperature_c": 57.0,
      "min_sm_clock_mhz": 1170.0,
      "mean_power_w": 41.479,
      "max_memory_used_mib": 7490.0,
      "nonzero_throttle_reasons": [
        "0x0000000000000004",
        "0x0000000000000400"
      ]
    }
  },
  "step_b16_teacher": {
    "batch": 16,
    "teacher": true,
    "steps": 20,
    "s_per_step_median": 0.255182409498957,
    "s_per_step_max": 0.28365988898440264,
    "peak_allocated_gb": 4.068089008331299,
    "peak_reserved_gb": 4.35546875,
    "rss_gb": 2.6009521484375,
    "loss_finite": true,
    "gpu": {
      "samples": 7,
      "max_temperature_c": 56.0,
      "min_sm_clock_mhz": 1912.0,
      "mean_power_w": 44.629999999999995,
      "max_memory_used_mib": 5024.0,
      "nonzero_throttle_reasons": [
        "0x0000000000000004",
        "0x0000000000000400"
      ]
    }
  }
}
```

## Regra de B

```json
{
  "p99_by_site": {
    "enc": 1.0232189440727217,
    "df_dec": 0.5673782849311828
  },
  "max_by_site": {
    "enc": 7.426816940307617,
    "df_dec": 5.117401123046875
  },
  "values_by_site": {
    "enc": 15360000,
    "df_dec": 15360000
  },
  "p99_pooled": 0.8317146450281112,
  "rule": "B = min(50, max(4, 2*p99_pooled))",
  "beta_bound": 4.0,
  "samples": 200,
  "limitation": "ativações medidas com o ruído sintético do piloto (pilot-only, cost measurement) e fala VCTK/CV27; máximo observado 7.4 > B=4"
}
```

## Limitações das medições

- **Perda proxy e professor stand-in:** o passo de GPU mede o custo com uma perda proxy, não a perda real do `df/loss.py`, e o professor é um stand-in; s/passo e VRAM com professor são ordem de grandeza (a perda real e o professor real entram no M2).
- **Sem estabilização térmica:** 20 passos por configuração, curtos demais para o estado estacionário; temperatura máxima observada 57 C em B32.
- **Throttle:** razões não nulas de throttle em todas as medições (0x0000000000000004, 0x0000000000000400); `min_sm_clock` de 1170 MHz em B32 contra 2077 MHz em B16 sem professor (1912 MHz em B16 com professor): s/passo em B32 pode estar inflado.
- **Memória do host:** durante as medições havia 3.95 GB de RAM disponível (de 30.8 GB) e 44.7 GB de swap em uso; isso pode afetar a vazão do loader.
- **Loader:** medido com o ruído sintético do piloto, pequeno o bastante para caber no page cache; a vazão de 215 amostras/s e o RSS de 2.39 GB não valem para dados de treino reais maiores que a RAM.

## Pendências

- CV 27.0 pt: manifest `cv27-pt.jsonl` presente
- Disco: fatores FLAC de fala por fonte (VCTK a 48 kHz, LibriSpeech dev-clean a 16 kHz, demais SR interpolados); ruído pelo agregado por SR; `tract_check_target` medido no checkout principal.
- Decisões de planejamento 1-15 do plano bootstrap+M0 valem como registradas lá.

## Próximos passos

Pendências **obrigatórias do M1** (o plano do M1 deve começar por elas):

1. Baixar ruído real de treino: Freesound CC0 do DNS e FSD50K (CC0/CC BY).
2. Gerar RIR `synthetic-v2` com cauda longa (o v1 cobre só RT60 de 0,1 a 0,6 s).
3. Conferir a convenção de SNR (ruído escalado contra alvo + interferente) e o alinhamento do alvo seco × reverberado
   contra a spec §4.4.
4. Definir o gate para o desvio emulação offline × runtime antes do M2.
5. Publicar o commit `70dc85c` do Clearcore: a dependência git do `tract-check` está pinada nele.
6. Triagem dos *deferred minors* do ledger (lista abaixo).
7. Re-projetar o disco incluindo o que o M0 deixou fora da projeção (RIR `synthetic-v2` com cauda longa, ambiente
   `export` do M3, DNS5 dev do G8, pico transitório de download/extração do CV27 e do FSD50K) antes do M1.

O que cada marco seguinte consome deste relatório está nos "Próximos planos" do plano bootstrap + M0 (M1: pooling que
passou no tract, parâmetros do fbank, fatores FLAC por SR e vazão do loader; M2: B fixado, s/passo e VRAM, desvio
offline × runtime; M3: asset reproduzível e I2; M4: réplica do gating e projeção de disco; M5: pacote e proveniência).

*Deferred minors* do ledger, para triagem:

| Tarefa | Minor |
|---|---|
| 2 | ruído "invalid character #" de fixtures do workspace-policy do Clearcore no build; fallback de `binary()` aponta para `repo_root` em worktree; toolchain só fixa via cwd |
| 3 | JSONL com U+2028/U+2029 quebra `splitlines`; `_read_lock` não converte `JSONDecodeError`; sr/num_samples sem validação de tipo; ruídos Wikimedia com role train split test entrariam em `training_records` |
| 4 | sem `.gitattributes` marcando `resampler/**` como binário (risco autocrlf no hash); coeficientes custom não validam shape 241; testes bit-exatos de `golden_inputs` dependem da plataforma |
| 5 | `verify_frozen` com lock vazio dá `IndexError`; `load_budget` sem validação de chaves; dataclasses frozen com dicts mutáveis; hash não fixado literalmente em teste; `test_budget_is_frozen` não prova quebra por edição |
| 6 | `verify_film_graph` não exige que todo consumidor do site leia `_film` nem valida forma/tipo de gamma/beta; `add_film` muta o `ModelProto` recebido; `_initializers` ignora sparse/Constant; `pack.py` deixa `.tmp` órfão em falha; nível zlib não fixado; testes de recusa de `read_members`/`pack_asset` só cobrem bit executável |
| 7 | `validate()` não checa embedding_dim/mid_dim; teste do Γ=ln10 não asserta o valor; limite Γ=ln100 não testado; regra de B duplicada entre `film.py` e YAML |
| 8 | I1 não cobre o ramo `lsnr_dropout=True`; `film_sites` não prova que a modulação ocorre após o ReLU; `state_hash` falha com bfloat16 |
| 9 | prototype com BN em estatísticas de inicialização (P1 não exercita BN não trivial); `scale_to_*` dividem por zero com sinal nulo |
| 10 | `requests.HTTPError` de `raise_for_status` e erros do RemoteZip podem expor URL pré-assinada completa |
| 11 | NaN em `rt60_measured` geraria JSON inválido; teste de determinismo não compara sha256 do FLAC |
| 12 | `pilot_records(librispeech_speakers=...)` ficou morto; `_chunks` não faz retry em ConnectionError/timeout |
| 13 | testes do loader só pegam o 1º item; `pilot_records` sem guarda para speaker None nem prefixo de source_id; SIR espúrio em INTERFERER_ONLY; ruído carregado sem uso em TARGET_INTERFERER; seed repetido por época |
| 14 | teste de gating não confere estágio contra limiares; lsnr NaN vira `erb_df` silenciosamente; cauda parcial descartada; tool mais estrito que o runtime com saídas extras não-f32 |
| infra | worktree novo não tem `build/` (basetemp do pytest) e `env.sh` sobrescreve PYTHONPATH: implementadores precisam `mkdir -p build` e reexportar `CCTRAIN_WT`/`PYTHONPATH` |

## Go/no-go do orçamento (níveis de configs/m0/budget-v1.yaml)

| item | valor | limite | margem (limite - valor) | veredito |
|---|---:|---:|---:|---|
| disco | 50.602 | 56.888 | 6.286 | go |
| vram_b16 | 4.355 | 7.000 | 2.645 | go |
| ram_loader | 2.391 | 6.000 | 3.609 | go |
| tempo_t3 | 10.633 | 35.000 | 24.367 | go |

O veredito de disco (**go**) tem margem de 6.29 GB (limite 56.89 GB menos projeção 50.60 GB), já descontada a reserva de 10 GB (GB decimais, como no budget). **Não estão na projeção:**

- RIR `synthetic-v2` com cauda longa (a projeção conta as 10 000 RIR do `synthetic-v1`, curtas)
- ambiente `export` do M3 (Python 3.10, torch 1.13.1)
- DNS5 dev do G8
- pico transitório de download/extração do CV27 e do FSD50K (arquivos comprimidos e extraídos coexistem)

Por isso o go de disco é condicional: re-projetar o disco incluindo esses itens antes do M1.
