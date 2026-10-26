#!/usr/bin/env python3
"""Throughput benchmark for Nemotron 3 Nano (30B-A3B) pretraining on a 6-GPU node.

Wraps the stock recipe (examples/models/nemotron/nemotron_3_nano/pretrain_nemotron_3_nano.py)
with named presets for layouts that fit on 6 GPUs, then applies Hydra-style CLI overrides.

    torchrun --nproc-per-node=6 bench_pretrain.py --preset fsdp_ep2 train.train_iters=20
"""

import argparse
import logging

import torch
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


logger = logging.getLogger(__name__)

# Megatron logs per-iteration timers (min/max gathered across all ranks) only from the last rank,
# whose logger is filtered here; emit them from rank 0 instead so the breakdown reaches the log.
import megatron.core.timers as _mtimers  # noqa: E402

_orig_timers_log = _mtimers.Timers.log


def _timers_log_rank0(self, names, rank=None, normalizer=1.0, reset=True, barrier=False):
    return _orig_timers_log(self, names, rank=0, normalizer=normalizer, reset=reset, barrier=barrier)


_mtimers.Timers.log = _timers_log_rank0

PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"


def split_pattern(pattern: str, stages: int) -> str:
    """Insert '|' stage separators so each PP stage gets ~equal layers (MoE layers dominate cost)."""
    n = len(pattern)
    cuts = [round(n * k / stages) for k in range(1, stages)]
    out, prev = [], 0
    for c in cuts + [n]:
        out.append(pattern[prev:c])
        prev = c
    return "|".join(out)


# Rough relative per-token training cost of each layer type (active matmul params + attention at 8K):
# MoE 'E' ~80M active (6x10M routed + 20M shared), Mamba 'M' ~39M, attention '*' ~23M + 8K-seq attention.
LAYER_COST = {"E": 1.0, "M": 0.5, "*": 1.1}
LM_HEAD_COST = 4.4  # 2688 x 131072 output projection + cross-entropy, on the last stage


# MoE recompute replays the MoE forward (~1/3 of its fwd+bwd); stages that skip it make 'E' cheaper.
E_COST_NO_RECOMPUTE = 0.78


def balanced_split(pattern: str, stages: int, moe_recompute_stages: int | None = None) -> str:
    """Choose '|' cut points minimizing the max per-stage cost (LM head counted on the last stage)."""
    n = len(pattern)

    def prefix_for(rc):
        out = [0.0]
        for c in pattern:
            out.append(out[-1] + (LAYER_COST[c] if (rc or c != "E") else E_COST_NO_RECOMPUTE))
        return out

    prefix_rc, prefix_norc = prefix_for(True), prefix_for(False)
    cur_stage = [0]

    def seg(i, j, last):
        rc = moe_recompute_stages is None or cur_stage[0] < moe_recompute_stages
        pre = prefix_rc if rc else prefix_norc
        return pre[j] - pre[i] + (LM_HEAD_COST if last else 0.0)

    # DP over (stage, end index) -> (max cost, cuts)
    best = {(0, 0): (0.0, [])}
    for k in range(stages):
        nxt = {}
        for (kk, i), (mx, cuts) in best.items():
            if kk != k:
                continue
            last = k == stages - 1
            cur_stage[0] = k
            for j in ([n] if last else range(i + 1, n - (stages - k - 1) + 1)):
                cand = (max(mx, seg(i, j, last)), cuts + [j])
                if (k + 1, j) not in nxt or cand[0] < nxt[(k + 1, j)][0]:
                    nxt[(k + 1, j)] = cand
        best.update(nxt)
    _, cuts = best[(stages, n)]
    out, prev = [], 0
    for c in cuts:
        out.append(pattern[prev:c])
        prev = c
    return "|".join(out)


def common(cfg):
    m = cfg.model
    # HybridEP/DeepEP are not built in this image; single-node NVLink all-to-all is fine for EP=2.
    m.moe_token_dispatcher_type = "alltoall"
    m.moe_flex_dispatcher_backend = None
    # Benchmark only: no checkpoints, no eval.
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = None
    cfg.validation.eval_iters = 0  # end-of-run eval (32 iters x GBS) is slow and trips the watchdog
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.log_memory_to_tensorboard = False
    cfg.logger.log_timers_to_tensorboard = False  # its write() resets timers before the console log
    cfg.train.manual_gc_interval = 10


def bf16_states(cfg):
    """bf16 grads + bf16 Adam moments, fp32 master weights (precision-aware optimizer)."""
    o = cfg.optimizer
    o.use_precision_aware_optimizer = True
    o.main_params_dtype = torch.float32
    o.main_grads_dtype = torch.bfloat16
    o.exp_avg_dtype = torch.bfloat16
    o.exp_avg_sq_dtype = torch.bfloat16


def preset_fsdp_ep2(cfg):
    """Megatron-FSDP (shard params+grads+optim over DP=6), EP=2 -> expert-DP=3."""
    cfg.model.expert_model_parallel_size = 2
    cfg.dist.use_megatron_fsdp = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.average_in_collective = False
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    # TE CUDA graphs are not compatible with FSDP param gather on this path.
    cfg.model.cuda_graph_impl = "none"
    bf16_states(cfg)


def preset_pp3_ep2_offload(cfg):
    """PP=3 x DP=2 (EP=2 inside DP); fp32 Adam states + master weights offloaded to CPU."""
    m = cfg.model
    m.expert_model_parallel_size = 2
    m.pipeline_model_parallel_size = 3
    m.hybrid_layer_pattern = split_pattern(PATTERN, 3)
    o = cfg.optimizer
    o.optimizer_cpu_offload = True
    o.optimizer_offload_fraction = 1.0
    o.overlap_cpu_optimizer_d2h_h2d = True
    o.use_precision_aware_optimizer = True
    o.main_params_dtype = torch.float32
    o.main_grads_dtype = torch.float32
    o.exp_avg_dtype = torch.float32
    o.exp_avg_sq_dtype = torch.float32


def limit_moe_recompute_to_first_stages(n_stages: int) -> None:
    """Only the first n PP stages recompute MoE layers (1F1B: stage k holds PP-k microbatches in flight)."""
    from megatron.core import parallel_state
    from megatron.core.transformer.moe.moe_layer import MoELayer

    orig_init = MoELayer.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        if parallel_state.get_pipeline_model_parallel_rank() >= n_stages:
            self.moe_layer_recompute = False

    MoELayer.__init__ = init


PRESETS = {
    "fsdp_ep2": preset_fsdp_ep2,
    "pp3_ep2_offload": preset_pp3_ep2_offload,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS), required=True)
    p.add_argument("--pp", type=int, help="override pipeline size (re-splits the hybrid pattern)")
    p.add_argument("--vpp", type=int, help="virtual pipeline stages per PP rank")
    p.add_argument("--ep", type=int, help="override expert-parallel size")
    p.add_argument("--split", choices=["even", "balanced"], default="even", help="PP stage split of the hybrid pattern")
    p.add_argument("--data", help="Megatron indexed dataset prefix (default: mock data)")
    p.add_argument("--moe-recompute-stages", type=int, help="only the first N PP stages recompute MoE layers")
    args, overrides = p.parse_known_args()
    if args.moe_recompute_stages is not None:
        limit_moe_recompute_to_first_stages(args.moe_recompute_stages)

    cfg = nemotron_3_nano_pretrain_config()
    common(cfg)
    PRESETS[args.preset](cfg)
    if args.data:
        cfg.dataset.blend = ([args.data], None)
        cfg.dataset.split = "100,0,0"
    m = cfg.model
    if args.ep:
        m.expert_model_parallel_size = args.ep
    if args.pp or args.vpp or args.split != "even":
        pp = args.pp or m.pipeline_model_parallel_size
        vpp = args.vpp or 1
        m.pipeline_model_parallel_size = pp
        m.virtual_pipeline_model_parallel_size = vpp if vpp > 1 else None
        if args.split == "balanced":
            m.hybrid_layer_pattern = balanced_split(PATTERN, pp * vpp, args.moe_recompute_stages)
        else:
            m.hybrid_layer_pattern = split_pattern(PATTERN, pp * vpp) if pp * vpp > 1 else PATTERN
    logger.warning(
        "bench layout: PP=%s VPP=%s EP=%s pattern=%s",
        m.pipeline_model_parallel_size, m.virtual_pipeline_model_parallel_size,
        m.expert_model_parallel_size, m.hybrid_layer_pattern,
    )

    omega, excluded = create_omegaconf_dict_config(cfg)
    if overrides:
        omega = parse_hydra_overrides(omega, overrides)
    apply_overrides(cfg, OmegaConf.to_container(omega, resolve=True), excluded)

    pretrain(config=cfg, forward_step_func=forward_step)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
