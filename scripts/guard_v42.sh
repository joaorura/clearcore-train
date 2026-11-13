#!/usr/bin/env bash
# Parada antecipada do pro-v42: depois das avaliações rápidas dos passos 2000 e 4000, se nas DUAS o C8 < 88% e o C6
# não estiver ≥ 4 dB melhor que o v3.1 (−14,5 → exige ≥ −10,5 dB), manda SIGTERM ao treino (salva e sai).
set -uo pipefail
TRAIN_PID="${TRAIN_PID:?}"; RUN_DIR="${RUN_DIR:?}"
JL="$RUN_DIR/quick-eval.jsonl"; LOG="$RUN_DIR/guard.log"; PY=/home/joaorura/orca/projects/clearcore-train/.venv/bin/python
echo "[$(date '+%F %T')] guarda iniciada (treino $TRAIN_PID)" >>"$LOG"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  v=$("$PY" - "$JL" <<'PY' 2>/dev/null
import json, sys
try:
    rows = {r["step"]: r for r in map(json.loads, open(sys.argv[1]).read().splitlines()) if r.get("step") in (2000, 4000)}
except FileNotFoundError:
    rows = {}
if len(rows) < 2:
    print("wait"); sys.exit()
bad = all(r.get("c8_pct", 0) < 88 and r["c6_db"] < -10.5 for r in rows.values())
print("stop" if bad else "ok", " ".join(f"{s}:C8={r.get('c8_pct')}/C6={r['c6_db']:.1f}" for s, r in sorted(rows.items())))
PY
)
  case "$v" in
    stop*) echo "[$(date '+%F %T')] PARADA: $v" >>"$LOG"; kill -TERM "$TRAIN_PID"; exit 0;;
    ok*) echo "[$(date '+%F %T')] segue: $v" >>"$LOG"; exit 0;;
  esac
  sleep 60
done
echo "[$(date '+%F %T')] treino terminou antes da decisão" >>"$LOG"
