#!/bin/bash
# Layout sweep at GBS=192 (1.57M tok/step), MBS=2. Baseline: pp3_gbs192_mbs2 (even split) = 211 TFLOP/s/GPU.
cd "$(dirname "$0")/.."
C="train.train_iters=10 train.global_batch_size=192 train.micro_batch_size=2 validation.eval_iters=0 scheduler.lr_warmup_iters=2 logger.timing_log_level=1 logger.logging_level=20"
./launch.sh s2_pp3_bal        --preset pp3_ep2_offload --split balanced $C
./launch.sh s2_pp3_bal_vpp2   --preset pp3_ep2_offload --split balanced --vpp 2 $C
./launch.sh s2_pp6_ep1_bal    --preset pp3_ep2_offload --split balanced --pp 6 --ep 1 $C
./launch.sh s2_pp3_bal_norc   --preset pp3_ep2_offload --split balanced $C model.recompute_granularity=null model.recompute_modules=null
