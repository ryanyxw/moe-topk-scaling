#!/bin/bash
# PP6 x EP1 (balanced split) at GBS=384 (3.1M tok/step): amortize bubble/optimizer, spend spare memory.
cd "$(dirname "$0")/.."
B="--preset pp3_ep2_offload --split balanced --pp 6 --ep 1"
C="train.train_iters=10 train.global_batch_size=384 validation.eval_iters=0 scheduler.lr_warmup_iters=2 logger.timing_log_level=1"
./launch.sh s3_pp6_gbs384_mbs2       $B $C train.micro_batch_size=2
./launch.sh s3_pp6_gbs384_mbs4       $B $C train.micro_batch_size=4
./launch.sh s3_pp6_gbs384_mbs2_rcln  $B $C train.micro_batch_size=2 model.recompute_modules=[layernorm]
./launch.sh s3_pp6_gbs384_mbs2_off80 $B $C train.micro_batch_size=2 optimizer.optimizer_offload_fraction=0.8
