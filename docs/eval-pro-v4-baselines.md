# Comparação de execuções

| execução | checkpoint | gating | linhas | tempo (s) |
|---|---|---|---|---|
| v2_current | `pro-v2-dualpath-tse-20261008/ckpt-00075000.pt` | current | 7348 | 488 |
| v2_new | `pro-v2-dualpath-tse-20261008/ckpt-00075000.pt` | new | 7348 | 525 |
| v3_current | `pro-v2-mhca-k16-tse/ckpt-00002000.pt` | current | 7348 | 480 |
| v3_new | `pro-v2-mhca-k16-tse/ckpt-00002000.pt` | new | 7348 | 487 |

## Critérios (nominal -30 dBFS)

| # | Critério | v2_current | v2_new | v3_current | v3_new |
|---|---|---|---|---|---|
| 1 | N=0 (SNR 5): perfil não piora | **FAIL**: FAIL (0/1); pior -8.37 (margem; dSTOI em centésimos) | **FAIL**: FAIL (0/1); pior -7.58 (margem; dSTOI em centésimos) | **FAIL**: FAIL (0/1); pior -7.01 (margem; dSTOI em centésimos) | **FAIL**: FAIL (0/1); pior -6.51 (margem; dSTOI em centésimos) |
| 2 | N=1, SIR 0, silencioso | **FAIL**: FAIL (0/1); pior -18.48 (pior margem) | **FAIL**: FAIL (0/1); pior -15.96 (pior margem) | **FAIL**: FAIL (0/1); pior -17.80 (pior margem) | **FAIL**: FAIL (0/1); pior -15.81 (pior margem) |
| 3 | N=2 e N=3, SIR 0 por interferente | **FAIL**: FAIL (0/6); pior -17.52 (pior margem) | **FAIL**: FAIL (0/6); pior -17.53 (pior margem) | **FAIL**: FAIL (0/6); pior -19.20 (pior margem) | **FAIL**: FAIL (0/6); pior -19.30 (pior margem) |
| 4 | Só-interferente (N=1..3, perfil) | **FAIL**: FAIL (5/42); pior %ZERO 29.6%, pior vazamento -10.2 dB | **FAIL**: FAIL (5/42); pior %ZERO 29.6%, pior vazamento -10.6 dB | **FAIL**: FAIL (5/42); pior %ZERO 29.8%, pior vazamento -9.5 dB | **FAIL**: FAIL (5/42); pior %ZERO 29.8%, pior vazamento -9.7 dB |
| 5 | Interferente fraco (SIR +20/+30, silencioso) | **FAIL**: FAIL (2/8); pior -9.47 dB (pior margem) | **FAIL**: FAIL (2/8); pior -8.39 dB (pior margem) | **FAIL**: FAIL (2/8); pior -9.61 dB (pior margem) | **FAIL**: FAIL (2/8); pior -8.70 dB (pior margem) |
| 6 | Alvo zerado indevidamente (N=0) | **FAIL**: FAIL (0/3); pior 24.05% (pior) | **FAIL**: FAIL (0/3); pior 24.05% (pior) | **FAIL**: FAIL (0/3); pior 23.32% (pior) | **FAIL**: FAIL (0/3); pior 23.32% (pior) |
| 7 | Sem perfil ~ v2 sem perfil | **PASS**: PASS; pior 0.000 dB / 0.0000 | **PASS**: PASS; pior 0.000 dB / 0.0000 | **PASS**: PASS; pior 0.072 dB / 0.0023 | **PASS**: PASS; pior 0.072 dB / 0.0023 |
| 8 | Robustez de nível (N=1, SIR 0) | **FAIL**: FAIL (1/3); pior queda +5.79 dB | **FAIL**: FAIL (1/3); pior queda +6.06 dB | **FAIL**: FAIL (2/3); pior queda +5.12 dB | **FAIL**: FAIL (2/3); pior queda +5.52 dB |
| 9 | Pareado com v2 | **PASS**: PASS; 252/252 células N>=1 (100%), pior N=0 +0.00 dB | **FAIL**: FAIL; 197/252 células N>=1 (78%), pior N=0 -0.63 dB | **FAIL**: FAIL; 95/252 células N>=1 (38%), pior N=0 -2.16 dB | **FAIL**: FAIL; 135/252 células N>=1 (54%), pior N=0 -1.30 dB |
| 10 | Especificidade (perfil trocado) | **PASS**: PASS (3/3); pior -19.0 dB (pior) | **PASS**: PASS (3/3); pior -18.7 dB (pior) | **PASS**: PASS (3/3); pior -18.7 dB (pior) | **PASS**: PASS (3/3); pior -18.4 dB (pior) |
| 11 | Runtime | **N/A**: fora do harness (gate de export/paridade) | **N/A**: fora do harness (gate de export/paridade) | **N/A**: fora do harness (gate de export/paridade) | **N/A**: fora do harness (gate de export/paridade) |

## Números principais (nível -30 dBFS)

| métrica | v2_current | v2_new | v3_current | v3_new |
|---|---|---|---|---|
| N=0 silent: dSI-SDR perfil | -27.74 | -27.32 | -29.90 | -29.04 |
| N=0 silent: dSI-SDR sem perfil | -9.28 | -9.28 | -9.68 | -9.68 |
| N=0 silent: dano do perfil (dB) | -18.46 | -18.04 | -20.22 | -19.36 |
| N=0 silent: dSTOI perfil | -0.170 | -0.140 | -0.202 | -0.170 |
| N=0 silent: alvo zerado (ZERO %) | 9.23 | 9.23 | 15.19 | 15.19 |
| N=0 snr20: dSI-SDR perfil | -6.62 | -5.24 | -7.15 | -6.55 |
| N=0 snr20: dSI-SDR sem perfil | -2.70 | -2.70 | -2.77 | -2.77 |
| N=0 snr20: dano do perfil (dB) | -3.92 | -2.55 | -4.38 | -3.78 |
| N=0 snr20: dSTOI perfil | -0.123 | -0.105 | -0.134 | -0.117 |
| N=0 snr20: alvo zerado (ZERO %) | 11.38 | 11.38 | 11.94 | 11.94 |
| N=0 snr5: dSI-SDR perfil | +4.77 | +5.13 | +4.90 | +5.21 |
| N=0 snr5: dSI-SDR sem perfil | +9.16 | +9.16 | +9.12 | +9.12 |
| N=0 snr5: dano do perfil (dB) | -4.39 | -4.02 | -4.21 | -3.91 |
| N=0 snr5: dSTOI perfil | -0.084 | -0.076 | -0.070 | -0.065 |
| N=0 snr5: alvo zerado (ZERO %) | 24.05 | 24.05 | 23.32 | 23.32 |
| N=1 SIR 0 silent: dSI-SDR | -6.20 | -3.87 | -4.09 | -3.82 |
| N=2 SIR 0 silent: dSI-SDR | -1.65 | -1.60 | -2.23 | -2.31 |
| N=3 SIR 0 silent: dSI-SDR | +0.54 | +0.61 | +0.35 | +0.40 |
| N=4 SIR 0 silent: dSI-SDR | +0.31 | +0.34 | +0.08 | +0.09 |
| N=1 SIR 0 silent: supressão / ganho do alvo | 25.9 / -18.1 | 25.9 / -16.4 | 28.3 / -16.8 | 27.5 / -17.2 |
| só-interferente silent (N=1..3 média, SIR 0): %ZERO / vazamento mediano | 62% / -36 dB | 62% / -36 dB | 62% / -35 dB | 62% / -35 dB |
| N=1 SIR 0 snr5: dSI-SDR | +1.35 | +1.19 | +1.48 | +1.29 |
| N=2 SIR 0 snr5: dSI-SDR | -1.12 | -1.00 | -1.54 | -1.43 |
| N=3 SIR 0 snr5: dSI-SDR | +1.34 | +1.34 | +1.38 | +1.38 |
| N=4 SIR 0 snr5: dSI-SDR | +0.98 | +0.95 | +0.81 | +0.79 |
| N=1 SIR 0 snr5: supressão / ganho do alvo | 18.3 / -9.2 | 18.4 / -9.5 | 17.5 / -8.8 | 17.6 / -9.1 |
| só-interferente snr5 (N=1..3 média, SIR 0): %ZERO / vazamento mediano | 58% / -23 dB | 58% / -23 dB | 57% / -18 dB | 57% / -18 dB |
| N=1 SIR +20 silencioso: supressão (dB) | 11.2 | 12.5 | 11.9 | 13.0 |
| só-interferente N=1 SIR +20 silencioso: %ZERO | 52 | 52 | 51 | 51 |
| N=1 SIR +30 silencioso: supressão (dB) | 10.8 | 11.4 | 12.2 | 12.9 |
| só-interferente N=1 SIR +30 silencioso: %ZERO | 50 | 50 | 49 | 49 |
| perfil trocado N=1 SIR 0 silencioso: ganho do alvo (dB) | -23.0 | -22.8 | -23.9 | -23.6 |
| usuário real N=0 snr5: dSI-SDR perfil / sem perfil | +2.98 / +6.33 | +3.72 / +6.33 | +3.71 / +6.43 | +4.57 / +6.43 |
| usuário real N=1 SIR 0 silencioso: dSI-SDR / supressão / ganho | +1.23 / 25.3 / -10.7 | +2.03 / 24.4 / -10.1 | +1.36 / 23.1 / -8.7 | +2.15 / 21.9 / -8.1 |
| usuário real N=2 SIR 0 silencioso: dSI-SDR / supressão / ganho | -1.54 / 23.4 / -15.9 | -1.11 / 22.9 / -15.4 | -1.19 / 20.6 / -13.5 | -0.68 / 19.8 / -12.8 |
| usuário real N=3 SIR 0 silencioso: dSI-SDR / supressão / ganho | +0.28 / 16.7 / -11.9 | +0.28 / 16.7 / -11.9 | +0.67 / 16.0 / -10.9 | +0.71 / 15.9 / -10.8 |
| usuário real N=4 SIR 0 silencioso: dSI-SDR / supressão / ganho | +0.53 / 20.3 / -14.8 | +0.54 / 20.3 / -14.8 | +0.46 / 19.6 / -14.2 | +0.45 / 19.5 / -14.2 |
| usuário real só-interferente N=1 SIR 0 silencioso: %ZERO | 62 | 62 | 61 | 61 |
