#!/bin/bash
# Forced uniform MoE routing (benchmark mode) so results don't depend on how routing collapses on mock data.
cd "$(dirname "$0")/.."
B="--preset pp3_ep2_offload --split balanced --pp 6 --ep 1 train.global_batch_size=384 train.micro_batch_size=2 train.train_iters=12 logger.timing_log_level=1 model.moe_router_force_load_balancing=true"
./launch.sh s7_base $B
./launch.sh s7_mbs1_rcln $B train.micro_batch_size=1 "model.recompute_modules=[layernorm]"
./launch.sh s7_rcstage3 $B --moe-recompute-stages 3
./launch.sh s7_off90 $B optimizer.optimizer_offload_fraction=0.9
./launch.sh s7_gbs768 $B train.global_batch_size=768 train.train_iters=8
python3 summarize.py "s7_*"
