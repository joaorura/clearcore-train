#!/usr/bin/env bash
#SBATCH --job-name=cc-pro-v5
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/4tb-nvme/work/joaorura/logs/%x-%j.out
#SBATCH --signal=B:TERM@900
# pro-v5 na rig (base DFNet3 congelada + condicionamento do zero + LoRA): treino em N GPUs (torchrun, 1 processo por
# GPU, all-reduce só dos treináveis) + avaliador rápido periódico em CPU (best/best.pt: mais PASS, desempate C6, C1)
# + after-train (eval_quality_v42 completo). Guarda só técnica (NaN), dentro do treino.
#
# Batch GLOBAL fixo (--global-batch, padrão do script 48): relançar com outro nº de GPUs/CPUs e o MESMO RUN_ID retoma do
# último checkpoint com a mesma sequência de dados e o mesmo cosseno (só muda o batch por GPU).
#   1 GPU agora:          RUN_ID=pro-v5-p3 sbatch -c 8  --gres=gpu:1 scripts/job_v5.sbatch
#   mais GPUs depois:     scancel <job>; RUN_ID=pro-v5-p3 sbatch -c 24 --gres=gpu:3 scripts/job_v5.sbatch
#   (o scancel manda TERM: o treino salva e sai; o novo job retoma.)
# CPUs: EVAL_THREADS (2) para o avaliador; o resto dividido entre os ranks (1 processo de treino + workers de dados).
set -euo pipefail
W=/mnt/4tb-nvme/work/joaorura
ROOT=$W/clearcore-train                   # clone principal: dados (data/) e manifests ignorados
CODE="${CODE:-$W/clearcore-train-v5}"      # worktree do branch train/pro-v5-adapter (código)
RUNS=$W/runs
export CCTRAIN_ROOT=$ROOT CCTRAIN_WT=$CODE
export CCTRAIN_DATA_DIR=$ROOT/data CCTRAIN_RUNS_DIR=$RUNS
export UV_PROJECT_ENVIRONMENT=$W/env
export XDG_CACHE_HOME=$W/cache TORCH_HOME=$W/cache/torch MPLCONFIGDIR=$W/cache/mpl TMPDIR=$W/cache/tmp
umask 077
mkdir -p "$TMPDIR"
PY=$W/env/bin/python
RUN_ID="${RUN_ID:-pro-v5-p3}"
RUN_DIR="$RUNS/$RUN_ID"
EVAL_THREADS="${EVAL_THREADS:-2}"
NG=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NG" -ge 1 ] || NG=1
CPUS="${SLURM_CPUS_PER_TASK:-8}"
WPR=$(( (CPUS - EVAL_THREADS) / NG - 1 )); [ "$WPR" -ge 1 ] || WPR=1
if [ -z "${QUICK_EVAL:-}" ]; then
  [ -f "$ROOT/data/eval-v4/user_take4_stream_48k.wav" ] && QUICK_EVAL=1 || QUICK_EVAL=0
fi
CACHE="${QUICK_CACHE:-$RUNS/m4/pro-v42/quick-cache}"   # linhas de referência v3.1/DFNet3 (metas), compartilhado
mkdir -p "$RUN_DIR"; rm -f "$RUN_DIR/TRAIN_DONE"
cd "$CODE"
source scripts/env.sh 2>/dev/null || true
echo "[$(date '+%F %T')] job ${SLURM_JOB_ID:-?} | GPUs $NG (${CUDA_VISIBLE_DEVICES:-?}) | CPUs $CPUS | workers/rank $WPR | commit $(git rev-parse --short HEAD) | run $RUN_DIR"

CMD=("$PY" -m torch.distributed.run --standalone --nproc_per_node "$NG" scripts/train_pro_v5.py
     --run-id "$RUN_ID" --runs-subdir . --workers "$WPR" "$@")
{ echo "${CMD[*]}"; echo "commit $(git rev-parse HEAD) | dados: mix_v4 com fix dry-gain 0145934"; } >"$RUN_DIR/launch-cmd.txt"
OMP_NUM_THREADS=1 "${CMD[@]}" >>"$RUN_DIR/train.log" 2>&1 &
TP=$!
echo $TP >"$RUN_DIR/train.pid"

WP=""
if [ "$QUICK_EVAL" = 1 ]; then
  "$PY" scripts/eval_quick.py watch --snap-dir "$RUN_DIR/snapshots" --cache "$CACHE" --tiebreak c6,c1 \
    --jsonl "$RUN_DIR/quick-eval.jsonl" --best-dir "$RUN_DIR/best" --done-file "$RUN_DIR/TRAIN_DONE" \
    --threads "$EVAL_THREADS" >>"$RUN_DIR/quick-eval.log" 2>&1 &
  WP=$!; echo $WP >"$RUN_DIR/watch.pid"
fi

forward() {
  echo "[$(date '+%F %T')] sinal recebido: TERM -> treino $TP (salva e sai)"
  kill -TERM "$TP" 2>/dev/null || true
}
trap forward TERM INT
rc=0
while kill -0 "$TP" 2>/dev/null; do
  wait "$TP" || rc=$?
done
echo "[$(date '+%F %T')] treino saiu (rc=$rc); $(cat "$RUN_DIR/TRAIN_DONE" 2>/dev/null || echo 'sem TRAIN_DONE')"

# só roda o eval completo se o treino chegou ao fim (não em parada por sinal/tempo, que será retomada)
if [ "$QUICK_EVAL" = 1 ] && grep -q '^total_steps' "$RUN_DIR/TRAIN_DONE" 2>/dev/null; then
  sed -e "s#^ROOT=.*#ROOT=$ROOT#" -e "s#^WT=.*#WT=$CODE#" -e "s#^PY=.*#PY=$PY#" scripts/after_train_eval_v42.sh \
    >"$RUN_DIR/after_train_eval_rig.sh"
  TRAIN_PID=999999999 WATCH_PID="$WP" RUN_DIR="$RUN_DIR" EVAL_WORKERS=4 \
    bash "$RUN_DIR/after_train_eval_rig.sh" >"$RUN_DIR/after-train-eval.nohup" 2>&1 || true
else
  [ -n "$WP" ] && { sleep 120; kill "$WP" 2>/dev/null || true; }
fi
exit $rc
