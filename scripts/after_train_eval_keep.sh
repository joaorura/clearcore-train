#!/usr/bin/env bash
# Dispara a avaliação do pDFNet3 Pro v4-keep assim que o treino (TRAIN_PID) terminar.
# Escolhe o checkpoint de maior passo e roda scripts/eval_soft_gate.py (caminho do runtime, CPU) com
#   v2_current (v2 75000, gating atual), v31_new (v3.1, gating novo) e keep_new (este run, gating novo)
# no mesmo conjunto (data/eval-v4: matriz -30 dBFS SIR 0, silencioso/SNR 5, N=0..4 + usuário real take4 + temporal).
# Escreve EVAL_DONE ou EVAL_FAILED em RUN_DIR.
# DRY_RUN=1: não espera o PID nem roda nada; imprime o ckpt escolhido e os comandos.
set -uo pipefail

TRAIN_PID="${TRAIN_PID:?defina TRAIN_PID}"
ROOT=/home/joaorura/orca/projects/clearcore-train
WT=/home/joaorura/orca/projects/clearcore-train-wt/pro-v4-keep
RUN_DIR="${RUN_DIR:-$ROOT/runs/m4/pro-v4-keep}"
CKPT_DIR="$RUN_DIR/checkpoints"
PY="$ROOT/.venv/bin/python"
V2="$ROOT/runs/m2/pro-v2-dualpath-tse-20261008/checkpoints/ckpt-00075000.pt"
V31="$ROOT/runs/m4/pro-v4/checkpoints/ckpt-00079998.pt"
WORKERS="${EVAL_WORKERS:-3}"
LOG="$RUN_DIR/after-train-eval.log"
DONE="$RUN_DIR/EVAL_DONE"
FAILED="$RUN_DIR/EVAL_FAILED"
DRY_RUN="${DRY_RUN:-0}"

log() { local m; m="[$(date '+%F %T')] $*"; if [ "$DRY_RUN" = 1 ]; then echo "$m"; else echo "$m" >>"$LOG"; fi; }

pick_ckpt() {  # maior passo (ckpt-NNNNNNNN.pt, zero-padded); ignora .tmp
  ls -1 "$CKPT_DIR"/ckpt-*.pt 2>/dev/null | grep -E '/ckpt-[0-9]+\.pt$' | sort | tail -n1
}

fail() {
  log "FALHA: $*"
  if [ "$DRY_RUN" != 1 ]; then
    { echo "motivo: $*"; echo "--- ultimas linhas do log ---"; tail -n 80 "$LOG"; } >"$FAILED"
  fi
  exit 1
}

run() {  # run <descrição> <cmd...>
  local what="$1"; shift
  log "CMD ($what): $*"
  [ "$DRY_RUN" = 1 ] && return 0
  ( cd "$WT" && source scripts/env.sh 2>/dev/null; "$@" ) >>"$LOG" 2>&1
}

[ "$DRY_RUN" = 1 ] || { : >"$LOG"; rm -f "$DONE" "$FAILED"; }
log "disparador iniciado (pid $$), aguardando PID $TRAIN_PID (DRY_RUN=$DRY_RUN)"
if [ "$DRY_RUN" != 1 ]; then
  while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
fi
log "treino (PID $TRAIN_PID) encerrado ou inexistente"

CKPT="$(pick_ckpt)"
[ -n "$CKPT" ] || fail "nenhum checkpoint em $CKPT_DIR"
for f in "$V2" "$V31" "$PY"; do [ -e "$f" ] || fail "ausente: $f"; done
NAME="$(basename "${CKPT%.pt}")"
OUT="$RUN_DIR/eval-$NAME"
log "checkpoint escolhido: $CKPT -> $OUT"

run "avaliação" "$PY" scripts/eval_soft_gate.py run --out "$OUT" --workers "$WORKERS" \
  --model "v2=$V2:v2:current" --model "v31=$V31:v4:new" --model "keep=$CKPT:v4:new" \
  || fail "eval_soft_gate run falhou"
run "relatório" "$PY" scripts/eval_soft_gate.py report --out "$OUT" || fail "eval_soft_gate report falhou"

if [ "$DRY_RUN" = 1 ]; then log "DRY_RUN: sentinela $DONE nao escrita"; exit 0; fi
[ -s "$OUT/report.md" ] || fail "report.md ausente em $OUT"
{ echo "ckpt: $CKPT"; echo "report: $OUT/report.md"; echo "rows: $OUT/rows.jsonl"; date '+%F %T'; } >"$DONE"
log "concluido: $DONE"
