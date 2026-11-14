#!/usr/bin/env bash
# Smoke/bench do pro-v5 P3 numa GPU (rig: srun --gres=gpu:1 -c 8 bash scripts/bench_v5.sh). Cada configuração roda
# poucos passos num run descartável (runs/smoke-v5/...) e imprime a linha do último passo e o summary (s/passo = GPU,
# espera = loader; amostras/s = batch/(s/passo+espera)). Configs: "amp:batch[:extra]" (extra: flags unidas por "_").
set -uo pipefail
W=/mnt/4tb-nvme/work/joaorura
ROOT=$W/clearcore-train
CODE="${CODE:-$W/clearcore-train-v5}"
export CCTRAIN_ROOT=$ROOT CCTRAIN_WT=$CODE CCTRAIN_DATA_DIR=$ROOT/data CCTRAIN_RUNS_DIR=$W/runs
export XDG_CACHE_HOME=$W/cache TORCH_HOME=$W/cache/torch MPLCONFIGDIR=$W/cache/mpl TMPDIR=$W/cache/tmp
PY=$W/env/bin/python
cd "$CODE"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
WK="${WK:-$((CPUS - 1))}"
STEPS="${STEPS:-30}"
CONFIGS="${CONFIGS:-bf16:48:--gpu-mix none:48:--gpu-mix bf16:96:--gpu-mix bf16:144:--gpu-mix bf16:48}"
for c in $CONFIGS; do
  IFS=: read -r amp gb extra <<<"$c"
  extra="${extra//_/ }"  # "_" separa flags no extra (CONFIGS é separado por espaço): --gpu-mix_--p-drop_0
  id="bench-$amp-$gb$(echo "${extra:-}" | tr -d " -")-$(date +%H%M%S)"
  echo "=== $c (workers $WK, $STEPS passos) ==="
  OMP_NUM_THREADS=1 timeout 1200 "$PY" scripts/train_pro_v5.py --run-id "$id" --runs-subdir smoke-v5 --amp "$amp" \
    --global-batch "$gb" --max-steps "$STEPS" --workers "$WK" --eval-every-samples 0 ${extra:-} 2>&1 \
    | grep -E "passo [0-9]+/|Error|error|out of memory|Traceback" | tail -n 3 | cut -c1-400
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ('step','s_per_step_median','loader_wait_median','samples_per_s','vram_peak_gb','stop_reason','gpu_mix_cpu_refills')})" \
    "$W/runs/smoke-v5/$id/summary.json" 2>/dev/null || echo "sem summary (falhou)"
done
