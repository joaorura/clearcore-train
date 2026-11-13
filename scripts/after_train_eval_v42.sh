#!/usr/bin/env bash
# Depois do treino pro-v42: espera o treino (TRAIN_PID) e o avaliador rápido (WATCH_PID) terminarem, escolhe o melhor
# checkpoint do eval rápido (<run>/best/best.pt; sem ele, o último checkpoint) e roda o eval_quality COMPLETO
# (main + noise, v2/v31/v42) com 2 processos de CPU. Escreve EVAL_DONE ou EVAL_FAILED em RUN_DIR.
set -uo pipefail

TRAIN_PID="${TRAIN_PID:?defina TRAIN_PID}"
WATCH_PID="${WATCH_PID:-}"
ROOT=/home/joaorura/orca/projects/clearcore-train
WT=/home/joaorura/orca/projects/clearcore-train-wt/pro-v42
RUN_DIR="${RUN_DIR:-$ROOT/runs/m4/pro-v42}"
PY="$ROOT/.venv/bin/python"
WORKERS="${EVAL_WORKERS:-2}"
LOG="$RUN_DIR/after-train-eval.log"
DONE="$RUN_DIR/EVAL_DONE"
FAILED="$RUN_DIR/EVAL_FAILED"

log() { echo "[$(date '+%F %T')] $*" >>"$LOG"; }
fail() { log "FALHA: $*"; { echo "motivo: $*"; echo "--- log ---"; tail -n 60 "$LOG"; } >"$FAILED"; exit 1; }
run() { local what="$1"; shift; log "CMD ($what): $*"; ( cd "$WT" && source scripts/env.sh 2>/dev/null; "$@" ) >>"$LOG" 2>&1; }

: >"$LOG"; rm -f "$DONE" "$FAILED"
log "disparador (pid $$): aguardando treino $TRAIN_PID e avaliador ${WATCH_PID:--}"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
log "treino encerrado"
if [ -n "$WATCH_PID" ]; then
  n=0; while kill -0 "$WATCH_PID" 2>/dev/null && [ $n -lt 120 ]; do sleep 30; n=$((n+1)); done
  kill "$WATCH_PID" 2>/dev/null
fi
log "avaliador rápido encerrado"

CKPT="$RUN_DIR/best/best.pt"
if [ ! -s "$CKPT" ]; then
  CKPT="$(ls -1 "$RUN_DIR"/checkpoints/ckpt-*.pt 2>/dev/null | grep -E '/ckpt-[0-9]+\.pt$' | sort | tail -n1)"
  log "sem best.pt; usando o último checkpoint"
fi
[ -n "$CKPT" ] && [ -s "$CKPT" ] || fail "nenhum checkpoint"
OUT="$RUN_DIR/eval-full"
log "checkpoint: $CKPT ($(cat "$RUN_DIR/best/best.json" 2>/dev/null | grep -m1 '"step"')) -> $OUT"
export EVAL_V42_CKPT="$CKPT"
run "main" "$PY" scripts/eval_quality_v42.py run --out "$OUT" --set main --workers "$WORKERS" || fail "run main"
run "noise" "$PY" scripts/eval_quality_v42.py run --out "$OUT" --set noise --workers "$WORKERS" || fail "run noise"
run "report" "$PY" scripts/eval_quality_v42.py report --out "$OUT" || fail "report"
[ -s "$OUT/report.md" ] || fail "report.md ausente"
{ echo "ckpt: $CKPT"; echo "report: $OUT/report.md"; date '+%F %T'; } >"$DONE"
log "concluído: $DONE"
