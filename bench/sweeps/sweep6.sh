#!/bin/bash
# Reproducibility re-run of the PP6 baseline (full offload) + GBS768 + per-stage recompute, eval disabled, 14 iters.
cd "$(dirname "$0")/.."
B="--preset pp3_ep2_offload --split balanced --pp 6 --ep 1 train.global_batch_size=384 train.micro_batch_size=2 train.train_iters=14 logger.timing_log_level=1"
./launch.sh s6_base $B
./launch.sh s6_rcstage3 $B --moe-recompute-stages 3
./launch.sh s6_gbs768 $B train.global_batch_size=768 train.train_iters=9
python3 summarize.py "s6_*" s3_pp6_gbs384_mbs2
