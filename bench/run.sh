#!/bin/bash
# Usage: bench/run.sh <name> --preset <preset> [hydra overrides...]
# Runs on 6 GPUs (skips GPU 1). Log -> bench/logs/<name>.log
set -euo pipefail
NAME=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env/config.sh"
GPUS=${GPUS:-0,2,3,4,5,6}
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$HERE/logs" "$HOME/.cache/huggingface"

docker run --rm --name "bench_$NAME" --gpus "\"device=$GPUS\"" --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MBRIDGE:/opt/Megatron-Bridge" -v "$HERE:/bench" \
  -v "$HF_HOME_HOST:/root/.cache/huggingface" \
  -e CUDA_DEVICE_MAX_CONNECTIONS=${CDMC:-1} -e NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -e NCCL_NVLS_ENABLE=0 -e OMP_NUM_THREADS=${OMP:-32} -e HF_TOKEN="${HF_TOKEN:-}" \
  "$IMAGE" \
  python -m torch.distributed.run --nproc-per-node="$NGPU" /bench/bench_pretrain.py "$@" \
  2>&1 | tee "$HERE/logs/$NAME.log"
