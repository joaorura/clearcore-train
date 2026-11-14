#!/usr/bin/env bash
# Lança o treino longo do pro-v42 (nohup), o avaliador rápido (eval_quick watch, CPU, 1 processo) e o after-train.
# Uso: launch_v42.sh [args extras do train_pro_v42.py]. Para parar: kill -TERM $(cat <run>/train.pid) (salva e sai).
set -euo pipefail
ROOT=/home/joaorura/orca/projects/clearcore-train
WT=/home/joaorura/orca/projects/clearcore-train-wt/pro-v42
RUN_ID="${RUN_ID:-pro-v42}"
RUN_DIR="$ROOT/runs/m4/$RUN_ID"
PY="$ROOT/.venv/bin/python"
mkdir -p "$RUN_DIR"; rm -f "$RUN_DIR/TRAIN_DONE"
cd "$WT"
source scripts/env.sh 2>/dev/null || true
CMD=("$PY" scripts/train_pro_v42.py --run-id "$RUN_ID" "$@")
echo "${CMD[*]}" >"$RUN_DIR/launch-cmd.txt"
nohup "${CMD[@]}" >"$RUN_DIR/train.log" 2>&1 &
TP=$!
echo $TP >"$RUN_DIR/train.pid"
nohup "$PY" scripts/eval_quick.py watch --snap-dir "$RUN_DIR/snapshots" --cache "$ROOT/runs/m4/pro-v42/quick-cache" \
  --jsonl "$RUN_DIR/quick-eval.jsonl" --best-dir "$RUN_DIR/best" --done-file "$RUN_DIR/TRAIN_DONE" --threads 2 \
  >"$RUN_DIR/quick-eval.log" 2>&1 &
WP=$!
echo $WP >"$RUN_DIR/watch.pid"
TRAIN_PID=$TP WATCH_PID=$WP RUN_DIR="$RUN_DIR" nohup bash scripts/after_train_eval_v42.sh >"$RUN_DIR/after-train-eval.nohup" 2>&1 &
echo $! >"$RUN_DIR/after.pid"
TRAIN_PID=$TP RUN_DIR="$RUN_DIR" nohup bash scripts/guard_v42.sh >"$RUN_DIR/guard.nohup" 2>&1 &
echo $! >"$RUN_DIR/guard.pid"
echo "train $TP | watch $WP | after $(cat "$RUN_DIR/after.pid") | log $RUN_DIR/train.log"
