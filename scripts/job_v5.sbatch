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
#   (o scancel manda TERM: o treino salva e sai; o novo job retoma. Jobs com a versão anterior deste script, como o
#    99: parar com `scancel -b -s TERM <job>`, só o shell do lote, para o torchrun receber um único TERM.)
# CPUs: EVAL_THREADS (2) para o avaliador; o resto dividido entre os ranks (1 processo de treino + workers de dados).
# Com --gpu-mix (mistura/rótulos/features na GPU) ~4 CPUs por GPU bastam: ex. -c 14 --gres=gpu:3.
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
     --run-id "$RUN_ID" --runs-subdir . --workers "$WPR" --ckpt-every-min 10 "$@")
{ echo "${CMD[*]}"; echo "commit $(git rev-parse HEAD) | job ${SLURM_JOB_ID:-?} | GPUs $NG | CPUs $CPUS"
  echo "dados: ${DATA_NOTE:-mix_v4 fix dry-gain 0145934$(printf '%s ' "$@" | grep -q -- --gpu-mix && echo ' + gpu-mix 56bc2f5')}"
  [ -n "${LAUNCH_NOTE:-}" ] && echo "$LAUNCH_NOTE"
} >>"$RUN_DIR/launch-cmd.txt"

WP=""
if [ "$QUICK_EVAL" = 1 ]; then
  "$PY" scripts/eval_quick.py watch --snap-dir "$RUN_DIR/snapshots" --cache "$CACHE" --tiebreak c6,c1 --select "${SELECT:-c8gate}" --keep-all \
    --jsonl "$RUN_DIR/quick-eval.jsonl" --best-dir "$RUN_DIR/best" --done-file "$RUN_DIR/TRAIN_DONE" \
    --threads "$EVAL_THREADS" >>"$RUN_DIR/quick-eval.log" 2>&1 &
  WP=$!; echo $WP >"$RUN_DIR/watch.pid"
fi

# Relançamento automático: se o treino cair sem terminar (ex.: worker do DataLoader com "could not unlink the shared
# memory file", que matou o v42-voz no passo 2250), relança e retoma do último checkpoint, até MAX_RESTARTS vezes.
# Causa provável daquele erro: systemd-logind RemoveIPC=yes apaga o /dev/shm do usuário quando a última sessão de login
# (ssh) dele termina; mitigado com `loginctl enable-linger joaorura` (feito em 13/11/2026), e este laço cobre o resto.
MAX_RESTARTS="${MAX_RESTARTS:-6}"
got_term=0
TP=""
forward() {
  got_term=1
  echo "[$(date '+%F %T')] sinal recebido: TERM -> treino $TP (salva e sai)"
  # TERM direto nos ranks (filhos do torchrun), não no torchrun: um 2º TERM no torchrun (scancel já manda um) o faz
  # abortar o close e matar os ranks antes de salvarem (aconteceu no job 97). Os ranks salvam e saem; o torchrun termina.
  [ -n "$TP" ] && { pkill -TERM -P "$TP" 2>/dev/null || kill -TERM "$TP" 2>/dev/null || true; }
}
trap forward TERM INT
rc=0
for attempt in $(seq 0 "$MAX_RESTARTS"); do
  rm -f "$RUN_DIR/TRAIN_DONE"
  echo "[$(date '+%F %T')] tentativa $attempt" >>"$RUN_DIR/train.log"
  OMP_NUM_THREADS=1 "${CMD[@]}" >>"$RUN_DIR/train.log" 2>&1 &
  TP=$!
  echo $TP >"$RUN_DIR/train.pid"
  rc=0
  while kill -0 "$TP" 2>/dev/null; do
    wait "$TP" || rc=$?
  done
  echo "[$(date '+%F %T')] treino saiu (rc=$rc, tentativa $attempt); $(cat "$RUN_DIR/TRAIN_DONE" 2>/dev/null || echo 'sem TRAIN_DONE')"
  [ "$got_term" = 1 ] && break
  grep -qE '^(total_steps|max_steps|nan|nan_params|data_end)' "$RUN_DIR/TRAIN_DONE" 2>/dev/null && break
  sleep 30
done

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
