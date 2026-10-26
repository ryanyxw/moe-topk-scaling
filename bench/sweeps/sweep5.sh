#!/bin/bash
# PP6 x EP1, GBS384, MBS2, full optimizer offload: per-stage MoE recompute (later stages skip it) + GBS768 control.
cd "$(dirname "$0")/.."
B="--preset pp3_ep2_offload --split balanced --pp 6 --ep 1 train.global_batch_size=384 train.micro_batch_size=2 train.train_iters=10 logger.timing_log_level=1"
./launch.sh s5_rcstage3 $B --moe-recompute-stages 3
./launch.sh s5_rcstage4 $B --moe-recompute-stages 4
./launch.sh s5_gbs768 $B train.global_batch_size=768 train.train_iters=8
python3 summarize.py "s5_*" s3_pp6_gbs384_mbs2 s3_pp6_gbs384_mbs2_off80
