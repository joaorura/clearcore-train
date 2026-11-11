#!/usr/bin/env bash
# Avalia o checkpoint final do pDFNet3 Pro v4 na GPU: matriz completa de 261 células + extras + usuário real, gating NOVO
# (plano §3.3), 11 critérios, relatório pareado contra o v2 (gating atual) como referência dos critérios 7 e 9.
# Uso: scripts/eval_pro_v4_final.sh <ckpt.pt> [saida] [arch=v4] [gating=new]
# Variáveis: EVAL_DEVICE (cuda|cpu), EVAL_SUBSET (full|cpu|smoke), EVAL_WORKERS, EVAL_THREADS, EVAL_REF.
# Tempo: matriz completa (3.620 itens, 7.348 linhas) ~8 min por configuração na CPU com 5 processos; na GPU deve ser menor. Retoma se interrompido (rows.jsonl é incremental).
set -euo pipefail
CKPT="${1:?uso: $0 <ckpt.pt> [saida] [arch] [gating]}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh 2>/dev/null || true
OUT="${2:-$CCTRAIN_RUNS_DIR/eval-v4/v4-$(basename "${CKPT%.pt}")}"
ARCH="${3:-v4}"
GATING="${4:-new}"
REF="${EVAL_REF:-$CCTRAIN_RUNS_DIR/eval-v4/v2_current}"   # baseline v2 com gating atual (critérios 7 e 9)
WORKERS="${EVAL_WORKERS:-12}"
DEVICE="${EVAL_DEVICE:-cuda}"
SUBSET="${EVAL_SUBSET:-full}"
THREADS="${EVAL_THREADS:-1}"
if [ ! -f "$CCTRAIN_DATA_DIR/eval-v4/recipes.jsonl" ]; then
  "$CCTRAIN_PY" scripts/eval_pro_v4.py prep
fi
REFARG=()
[ -f "$REF/rows.jsonl" ] && REFARG=(--ref "$REF")
"$CCTRAIN_PY" scripts/eval_pro_v4.py eval --ckpt "$CKPT" --arch "$ARCH" --gating "$GATING" --device "$DEVICE" \
  --workers "$WORKERS" --threads "$THREADS" --subset "$SUBSET" --out "$OUT" --label "v4_${GATING}" "${REFARG[@]}"
echo "relatório: $OUT/report.md"
