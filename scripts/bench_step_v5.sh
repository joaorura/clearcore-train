#!/usr/bin/env bash
# Bench + paridade do passo do pro-v5 (scripts/bench_step_v5.py) numa GPU da rig:
#   srun --gres=gpu:1 -c 4 --mem=32G -t 45 bash scripts/bench_step_v5.sh
# Cada grupo de VARIANTS roda num processo (compilações não se misturam); EXTRA vai para o bench (ex. --deterministic).
set -uo pipefail
W=/mnt/4tb-nvme/work/joaorura
ROOT=$W/clearcore-train
CODE="${CODE:-$W/clearcore-train-graph}"
export CCTRAIN_ROOT=$ROOT CCTRAIN_WT=$CODE CCTRAIN_DATA_DIR=$ROOT/data CCTRAIN_RUNS_DIR=$W/runs
export XDG_CACHE_HOME=$W/cache TORCH_HOME=$W/cache/torch MPLCONFIGDIR=$W/cache/mpl TMPDIR=$W/cache/tmp
export TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor-graph
PY=$W/env/bin/python
cd "$CODE"
OUT="${OUT:-$W/speed-out/graph-$(date +%m%d-%H%M%S)}"
mkdir -p "$OUT"
GROUPS_="${GROUPS_:-ref-nolog,sf,sf+bench,sf+fused,sf+cl,sf+zg graph,graph+fused,graph+fused+bench c-default c-ro c-mat}"
i=0
for g in $GROUPS_; do
  i=$((i + 1))
  echo "=== grupo $i: $g ==="
  OMP_NUM_THREADS=1 timeout "${GROUP_TIMEOUT:-1500}" "$PY" scripts/bench_step_v5.py --variants "$g" --workers 3 \
    --out "$OUT/g$i.json" ${EXTRA:-} 2>&1 | grep -vE "Warning|warnings.warn|^\s*$" | tail -n 60
done
echo "resultados em $OUT"
