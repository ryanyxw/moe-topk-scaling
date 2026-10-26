#!/bin/bash
# Usage: train/launch.sh <log name> <train.py args...>   (normally started by scheduler.py)
# Runs train.py in the image on $GPUS (default 6 GPUs, skipping GPU 1).
# Log -> $SCALING_ROOT/logs/<name>.log, status -> $SCALING_ROOT/logs/<name>.status
#
# Watchdog: kills the run if the first iteration doesn't log within FIRST_TIMEOUT seconds,
# or if no new iteration is logged for STEP_TIMEOUT seconds afterwards (NCCL hang / deadlock).
# Env knobs: GPUS, CDMC (CUDA_DEVICE_MAX_CONNECTIONS), OMP (threads/rank), FIRST_TIMEOUT, STEP_TIMEOUT.
set -uo pipefail
NAME=$1; shift
source "$(dirname "$0")/../env/config.sh"
HERE=$SCALING_ROOT

GPUS=${GPUS:-0,2,3,4,5,6}
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
FIRST_TIMEOUT=${FIRST_TIMEOUT:-1500}
STEP_TIMEOUT=${STEP_TIMEOUT:-600}
LOG="$HERE/logs/$NAME.log"
STATUS="$HERE/logs/$NAME.status"
CNAME="bench_$NAME"
mkdir -p "$HERE/logs" "$HOME/.cache/huggingface"
docker rm -f "$CNAME" >/dev/null 2>&1
: > "$LOG"; echo RUNNING > "$STATUS"
echo "[launch.sh] $NAME args: $*" >> "$LOG"

docker run --rm --name "$CNAME" ${CPUSET:+--cpuset-cpus=$CPUSET} --gpus "\"device=$GPUS\"" --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  $DOCKER_MOUNTS \
  -e CUDA_DEVICE_MAX_CONNECTIONS=${CDMC:-1} -e NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -e NCCL_NVLS_ENABLE=0 -e OMP_NUM_THREADS=${OMP:-32} -e HF_TOKEN="${HF_TOKEN:-}" \
  "$IMAGE" \
  python -m torch.distributed.run --nproc-per-node="$NGPU" --master-port=${PORT:-29500} "$REPO/train/train.py" "$@" \
  >> "$LOG" 2>&1 &
PID=$!
# GPU telemetry (power/util/clocks/mem + throttle) to spot contention or throttling.
nvidia-smi dmon -s pucm -d 15 -o T > "$HERE/logs/$NAME.dmon" 2>&1 &
DMON=$!

start=$(date +%s); last_n=0; last_t=$start; verdict=""; fail_t=""
while kill -0 $PID 2>/dev/null; do
  sleep 15
  n=$(grep -c " iteration  *[0-9]*/" "$LOG")
  now=$(date +%s)
  if [ "$n" -gt "$last_n" ]; then last_n=$n; last_t=$now; fi
  if [ "$last_n" -eq 0 ] && [ $((now - start)) -gt "$FIRST_TIMEOUT" ]; then verdict="HANG_FIRST_ITER"; fi
  if [ "$last_n" -gt 0 ] && [ $((now - last_t)) -gt "$STEP_TIMEOUT" ]; then verdict="HANG_AFTER_ITER_$last_n"; fi
  # One rank dying (e.g. OOM) leaves the others blocked in NCCL forever: fail fast.
  if [ -z "$fail_t" ] && grep -qE "Pretrain failed on rank|OutOfMemoryError" "$LOG"; then fail_t=$now; fi
  if [ -n "$fail_t" ] && [ $((now - fail_t)) -gt 60 ]; then
    if grep -q "out of memory" "$LOG"; then verdict="OOM_RANK_FAILED"; else verdict="RANK_FAILED"; fi
  fi
  if [ -n "$verdict" ]; then
    echo "[watchdog] $verdict after $((now - start))s; killing $CNAME" >> "$LOG"
    docker kill "$CNAME" >/dev/null 2>&1
    break
  fi
done
wait $PID; rc=$?
kill $DMON 2>/dev/null
if [ -z "$verdict" ]; then
  if grep -q "out of memory" "$LOG"; then verdict="OOM"
  elif [ $rc -ne 0 ]; then verdict="FAILED_rc$rc"
  else verdict="OK"; fi
fi
echo "$verdict $(( $(date +%s) - start ))s" > "$STATUS"
echo "[launch.sh] $NAME: $verdict"
