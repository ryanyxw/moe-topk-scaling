#!/bin/bash
# Start one eval worker per GPU (default 0,2-7: never GPU 1). Workers share eval_queue.txt and claim
# models atomically; rerunning only evaluates tasks that have no result yet.
#   eval/run_evals.sh            # all GPUs in $EVAL_GPUS
#   tail -f $EVAL_ROOT/logs/workers.log
source "$(dirname "$0")/../env/config.sh"
mkdir -p $EVAL_ROOT/logs
for g in ${EVAL_GPUS:-0 2 3 4 5 6 7}; do
  nohup "$REPO/eval/worker.sh" $g > $EVAL_ROOT/logs/worker_g$g.out 2>&1 &
  echo "worker gpu$g pid $!"; sleep 3
done
