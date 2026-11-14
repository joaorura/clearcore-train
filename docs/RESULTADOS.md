# Resultados do pDFNet3 Pro v4 → v5: o que tentamos, o que falhou e o que aprendemos

Resumo de uma página (estado de 14/11/2026). Detalhes, números por checkpoint e comandos em
[`pro-v5-processo.md`](pro-v5-processo.md); proposta de próxima arquitetura em
[`proposta-arquitetura-tse.md`](proposta-arquitetura-tse.md).

**Objetivo**: com perfil de voz ligado, cortar os outros falantes (C8 ≥ 90%) sem picotar/abafar a voz do usuário
(C1, C6, C7) e sem perder a remoção de ruído do DFNet3 (R0). Critérios na seção 2 do documento de processo.

## Números-chave (avaliador rápido, 9 critérios)

| Modelo | PASS | C1 (dB) | C6 (dB) | C8 (%) | R0 |
|---|---|---|---|---|---|
| **v3.1** (referência) | 8/9 | −0,56 | −14,5 | 91 | P |
| Melhor P3 (job 141 @~300k) | 7/9 | +0,94 | −3,5 | 61 | P |
| Fine-tuning completo do DFNet3 (job 148 @200k) | 7/9 | −5,2 | −2,9 | 43 | P |
| Maior corte no P3 (job 145 final) | 3/9 | −1,44 | — | 80 | F |
| **Oráculo de presença limpo** (O1 @200k) | **9/9** | +7,2 (@100k) | −2,7 | **100** | P |
| Oráculo atrasado 150 ms + 10% erro (O2 @200k) | 4/9 | +3,7 | −5,5 | 86 | F |

Nenhum modelo treinável passou todos os critérios. Ganhar C6 (picote após o interferente) sempre custou C8 ou C1.

## O que falhou e por quê

| Experimento | Hipótese | Resultado | Lição |
|---|---|---|---|
| v4-keep / v4.2 (fine-tuning completo do v3.1 com perdas de preservação) | perdas de turno/dano N=0 seguram o alvo | melhora em 1–2k passos e colapsa entre 2k e 4k; 2/11 no completo | deriva estrutural; guard, weight decay e L2-SP não seguraram |
| Ajustar só pesos de perda (job 136) | o problema é de balanço da perda | oscila na mesma curva (C8 75, C6 −8,7) | pesos movem o ponto, não a curva |
| P3: DFNet3 congelado + camadas Pro + LoRA (jobs 132–145) | base congelada evita a deriva | 7/9 no melhor, mas C8 para em ~60–80% e, acima de ~70%, alvo e ruído pioram | não é deriva; é outro teto |
| Gate ZERO pela `lsnr_fc` congelada | — (bug) | 14–16% dos quadros do alvo zerados desde o passo 0; corrigido em `d40e517` (C6 −18 → −3,5) | gate com `detach()` de cabeça congelada precisa de gradiente que dependa do perfil |
| Bug g² no `mix_v4` | — (bug) | ganho do interferente aplicado 2× em ~37% das amostras do v3.1 | a referência foi treinada com SIR errado; o hash de config não mudou com a correção |
| Fine-tuning completo do DFNet3 (jobs 148 `all`, 149 `gru`) | o teto do P3 é capacidade da base congelada | mesma curva: C8 sobe a 75% e C1 cai a −8,8 | liberar pesos não resolve |
| Oráculo de presença (O1/O3/O2) | o teto é saber **quem** fala | com presença perfeita: 9/9 e 100% de corte, até com base congelada; com 150 ms de atraso: 4/9 | o gargalo é discriminação de falante com baixa latência |

## Erros de processo (francos)

- **9,5 h sem monitoramento**: o P3 longo (job 132) terminou às 01:43 de 14/11 e só foi visto às 11:18. Todo treino
  longo agora tem aviso de fim (`squeue` + marcadores `TRAIN_DONE`/`EVAL_DONE`).
- **Avaliador morrendo por OOM** (`93a28e5`): o avaliador periódico ocupava a GPU via `init_df`, morria em silêncio,
  deixava `quick-eval.jsonl` vazio e desligava o guard; o "melhor" checkpoint do v4.2 virou o último. Corrigido forçando
  CPU e repetindo snapshots falhos; regra: conferir que o avaliador produz linhas novas.
- **Kill duplo no torchrun** (`e116124`): dois TERM abortavam os ranks antes de salvarem.
- **`/dev/shm` apagado no logout** (provável `RemoveIPC`): um run morreu no meio; mitigado com linger e relançamento.
- Comparações com o v3.1 carregam o viés do bug g² e o ruído estatístico de ~±1,5 dB no C1 (12 itens por recorte).

## Próximo passo

Proposta A de [`proposta-arquitetura-tse.md`](proposta-arquitetura-tse.md): cadastro multi-token com casamento por
quadro antes da GRU, com requisito de decisão ≤ ~30 ms e precisão nas trocas de turno. Antes, o O2' (oráculo com 30 ms
de atraso e 5% de erro) confirma o requisito. Sonda de AUC da presença: o melhor resultado **implantável** (só cadastro fixo, atraso 0 ms) é AUC ~0,88 (erro ~21%); os 0,897/0,916 do casamento + ECAPA usam ECAPA calculado na mistura em janela deslizante e **não são implantáveis em tempo real** (ver `pro-v5-processo.md` §9.4). A meta de 0,90 **não** foi atingida de forma implantável.
