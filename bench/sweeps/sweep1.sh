#!/bin/bash
# GBS=192 (1.57M tokens/step) for PP3 x EP2 + CPU offload, MBS 1 vs 2
cd "$(dirname "$0")/../.."
COMMON="train.train_iters=10 train.global_batch_size=192 validation.eval_iters=0 scheduler.lr_warmup_iters=2 logger.timing_log_level=1"
bench/run.sh pp3_gbs192_mbs1 --preset pp3_ep2_offload train.micro_batch_size=1 $COMMON > /dev/null 2>&1
bench/run.sh pp3_gbs192_mbs2 --preset pp3_ep2_offload train.micro_batch_size=2 $COMMON > /dev/null 2>&1
