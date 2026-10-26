#!/bin/bash
# PP6 x EP1 balanced, GBS384, MBS2: recompute/offload/GBS/CUDA-graph variants around the off80 winner.
cd "$(dirname "$0")/.."
B="--preset pp3_ep2_offload --split balanced --pp 6 --ep 1 train.global_batch_size=384 train.micro_batch_size=2 train.train_iters=10 logger.timing_log_level=1"
./launch.sh s4_off70 $B optimizer.optimizer_offload_fraction=0.7
./launch.sh s4_moeact_ln $B "model.recompute_modules=[moe_act,layernorm]"
./launch.sh s4_off80_cg_router $B optimizer.optimizer_offload_fraction=0.8 "model.cuda_graph_modules=[attn,mamba,moe_router,moe_preprocess]"
./launch.sh s4_off80_gbs768 $B optimizer.optimizer_offload_fraction=0.8 train.global_batch_size=768 train.train_iters=8
python3 summarize.py "s4_*" s3_pp6_gbs384_mbs2_off80
