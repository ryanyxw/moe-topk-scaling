#!/usr/bin/env python3
"""Scaling-law runs for a small Nemotron-3-Nano-style hybrid MoE (fixed total params, swept top-k).

Model: hidden 1024, 25 layers (11 Mamba / 11 MoE / 3 attention, Nano-style interleave), 128 routed
experts (ffn 1024) + shared expert (ffn 512); everything else as in the Nemotron 3 Nano recipe.

Schedule: WSD (warmup -> constant -> 1-sqrt cooldown over the last --cooldown fraction to 0).
A checkpoint is saved at the start of the cooldown so a longer run for the same top-k can resume
from it (--load ... --extend) instead of re-training the constant-LR phase.

    torchrun --nproc-per-node=1 train.py --name k4_lr3e-3 --topk 4 --lr 3e-3 --flops 8e18
"""

import argparse
import glob
import logging
import os
import re
import sys


# TE grouped-GEMM backend for the 128 experts must be chosen before TE is imported.
#   loop: one cuBLAS GEMM per expert (TE default) | cutlass: CUTLASS grouped kernel |
#   fused: native GroupedTensor grouped GEMM | both: fused + cutlass
_GG = {"loop": {}, "cutlass": {"NVTE_USE_CUTLASS_GROUPED_GEMM": "1"},
       "fused": {"NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM": "1"},
       "both": {"NVTE_USE_CUTLASS_GROUPED_GEMM": "1", "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM": "1"}}
for _i, _a in enumerate(sys.argv):
    if _a == "--grouped-gemm" and _i + 1 < len(sys.argv):
        os.environ.update(_GG[sys.argv[_i + 1]])

import torch

import megatron.core.timers as _mtimers
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.flop_utils import num_floating_point_operations


logger = logging.getLogger(__name__)

# Megatron logs timers from the last rank only; log them from rank 0 so they reach the console.
_orig_timers_log = _mtimers.Timers.log


def _timers_log_rank0(self, names, rank=None, normalizer=1.0, reset=True, barrier=False):
    return _orig_timers_log(self, names, rank=0, normalizer=normalizer, reset=reset, barrier=barrier)


_mtimers.Timers.log = _timers_log_rank0

PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEME"  # 25 layers: 11 M, 11 E, 3 *
SEQ = 8192
GBS = 128  # 128 x 8192 = 1.05M tokens / step
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/nvme/data")
SCALING_ROOT = os.environ.get("SCALING_ROOT", "/mnt/nvme/scaling")
DATA = f"{DATA_ROOT}/tok"
VAL_SHARD = "016_00001"
N_TRAIN_SHARDS = 40  # ~28B tokens: fixed blend for the whole isoFLOP sweep
RUNS = f"{SCALING_ROOT}/runs"
LOGS = f"{SCALING_ROOT}/logs"


def _has_ckpt(d):
    """True if d holds a persistent or non-persistent checkpoint (an empty dir would mean 'train from scratch')."""
    return bool(glob.glob(f"{d}/iter_*") or glob.glob(f"{d}/non_persistent/iter_*"))


def train_val_blend(n_train=N_TRAIN_SHARDS, start=0):
    shards = sorted(p[:-4] for p in glob.glob(f"{DATA}/*.idx") if not p.endswith(f"{VAL_SHARD}.idx"))
    train = shards[start:start + n_train]
    if len(train) < n_train:
        raise SystemExit(f"only {len(train)}/{n_train} training shards tokenized yet")
    if not os.path.exists(f"{DATA}/{VAL_SHARD}.idx"):
        raise SystemExit("validation shard not tokenized yet")
    # weights=None -> each shard built for exactly one epoch, blended by size: the sample order does not
    # depend on train_iters, so a resumed/extended run continues the identical token stream.
    return [(train, None), ([f"{DATA}/{VAL_SHARD}"], None), None]


def build(args):
    cfg = nemotron_3_nano_pretrain_config()
    m = cfg.model
    m.hybrid_layer_pattern = PATTERN
    m.num_layers = len(PATTERN)
    m.hidden_size = 1024
    m.ffn_hidden_size = 1024
    m.mamba_num_heads = 32
    m.mamba_head_dim = 64
    m.mamba_state_dim = 128
    m.mamba_num_groups = 8
    m.num_attention_heads = 8
    m.num_query_groups = 2
    m.kv_channels = 128
    m.num_moe_experts = 128
    m.moe_ffn_hidden_size = 1024
    m.moe_shared_expert_intermediate_size = 512
    m.moe_router_topk = args.topk
    m.init_method_std = 0.02
    m.seq_length = SEQ
    m.vocab_size = 131072  # Nemotron tokenizer; needed by the FLOP counter before the tokenizer is built
    # Parallelism: data-parallel only (+ optional EP), no PP/TP for a 3.3B-total model.
    m.tensor_model_parallel_size = 1
    m.pipeline_model_parallel_size = 1
    m.expert_model_parallel_size = args.ep
    m.pipeline_model_parallel_layout = None
    m.moe_token_dispatcher_type = "alltoall"  # HybridEP/DeepEP not built in this image
    m.moe_flex_dispatcher_backend = None
    m.recompute_granularity = "selective" if args.recompute else None
    m.recompute_modules = args.recompute or None
    m.cross_entropy_fusion_impl = args.ce
    if args.no_grad_accum_fusion:
        m.gradient_accumulation_fusion = False
    if args.cuda_graph == "none":
        m.cuda_graph_impl = "none"
    cfg.comm_overlap.tp_comm_overlap = False

    cfg.dataset.seq_length = SEQ
    cfg.dataset.num_workers = 4
    cfg.dataset.path_to_cache = f"{DATA_ROOT}/cache"
    if args.mock:
        cfg.dataset.blend = None
    else:
        cfg.dataset.blend = None
        cfg.dataset.split = None
        cfg.dataset.blend_per_split = train_val_blend(args.train_shards, args.shard_start)

    t = cfg.train
    t.global_batch_size = args.gbs or GBS
    t.micro_batch_size = args.mbs
    t.manual_gc = True
    t.manual_gc_interval = 50

    o = cfg.optimizer
    o.lr = args.lr
    o.min_lr = 0.0
    o.weight_decay = 0.1
    o.adam_beta1, o.adam_beta2 = 0.9, 0.95
    o.clip_grad = 1.0
    o.optimizer_cpu_offload = False

    # Steps from the FLOP budget, using Bridge's own (hybrid-aware) FLOP counter.
    flops_per_step = num_floating_point_operations(cfg, batch_size=t.global_batch_size)
    if args.iters:
        iters = args.iters
    else:
        iters = round(args.flops / flops_per_step)
    t.train_iters = iters
    decay = max(1, round(args.cooldown * iters))
    if args.save_splits > 1:  # cooldown start must be a multiple of the persistent save interval
        decay = iters - (iters - decay) // args.save_splits * args.save_splits
    s = cfg.scheduler
    # Warmup: 200 steps (0.2B tokens), capped at 10% for very short runs. Extensions must pass the
    # original value explicitly (--warmup) so the resumed schedule matches the constant-LR phase.
    s.lr_warmup_iters = args.warmup if args.warmup is not None else min(200, round(0.1 * iters))
    s.lr_decay_style = "WSD"
    s.lr_wsd_decay_style = "minus_sqrt"
    s.lr_decay_iters = iters
    s.lr_wsd_decay_iters = decay
    s.override_opt_param_scheduler = bool(args.load)
    s.use_checkpoint_opt_param_scheduler = False

    run_dir = f"{RUNS}/{args.name}"
    ck = cfg.checkpoint
    ck.save = None if args.no_save else f"{run_dir}/ckpt"
    # Save exactly at the cooldown start (the constant-LR state other budgets resume from).
    # --save-splits N also saves N-1 earlier constant-LR checkpoints (branch points for extra cooldowns).
    ck.save_interval = (iters - decay) // args.save_splits if not args.no_save else None
    ck.load = args.load
    ck.ckpt_step = args.load_step  # load the constant-LR (pre-cooldown) checkpoint, not the final one
    ck.finetune = False
    ck.async_save = False
    ck.ckpt_format = "torch_dist"
    # Crash safety: rolling non-persistent checkpoint (only the latest is kept) every N steps.
    # The scheduler resumes a failed run from it and deletes it once the run finishes OK.
    if not args.no_save and args.np_interval:
        ck.non_persistent_save_interval = args.np_interval
        ck.non_persistent_ckpt_type = "global"

    cfg.validation.eval_interval = args.eval_interval
    cfg.validation.eval_iters = args.eval_iters
    lg = cfg.logger
    lg.log_interval = args.log_interval
    lg.log_throughput = True
    lg.log_timers_to_tensorboard = False
    lg.log_memory_to_tensorboard = False
    lg.tensorboard_dir = f"{run_dir}/tb"
    lg.wandb_project = None
    cfg.rng.seed = args.seed
    if args.profile:
        cfg.profiling.use_pytorch_profiler = True
        cfg.profiling.profile_step_start = 6
        cfg.profiling.profile_step_end = 8
        cfg.profiling.profile_ranks = [0]

    tokens = iters * t.global_batch_size * SEQ
    logger.warning(
        "scaling run %s: topk=%d lr=%g iters=%d (warmup %d, cooldown %d from iter %d) tokens=%.3fB "
        "flops/step=%.4e total=%.4e flops/token=%.4e",
        args.name, args.topk, args.lr, iters, s.lr_warmup_iters, decay, iters - decay, tokens / 1e9,
        flops_per_step, flops_per_step * iters, flops_per_step / (t.global_batch_size * SEQ),
    )
    return cfg


def _offset_train_sampler(offset):
    """Index the training blend from (consumed_samples - offset). The train loader is built first."""
    import megatron.bridge.data.loaders as L
    orig, done = L.build_pretraining_data_loader, []

    def wrapped(dataset, consumed_samples, *a, **kw):
        if not done:
            done.append(1)
            if consumed_samples < offset:
                raise SystemExit(f"--data-offset {offset} > consumed samples {consumed_samples} at resume")
            print(f"[data offset] consumed {consumed_samples} -> new-blend index {consumed_samples - offset}",
                  file=sys.stderr, flush=True)
            consumed_samples -= offset
        return orig(dataset, consumed_samples, *a, **kw)
    L.build_pretraining_data_loader = wrapped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--flops", type=float, help="training FLOP budget (Bridge FLOP counter)")
    p.add_argument("--iters", type=int, help="override: number of steps")
    p.add_argument("--cooldown", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--load", help="checkpoint dir to resume from (constant-LR checkpoint)")
    p.add_argument("--load-step", type=int, help="iteration to load from --load (the cooldown-start ckpt)")
    p.add_argument("--parent", help="extend this finished run: load its cooldown-start ckpt, warmup 200")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--save-splits", type=int, default=1,
                   help="persistent constant-LR ckpts at k/N of the cooldown start, k=1..N (final run: cooldown branches)")
    p.add_argument("--np-interval", type=int, default=500, help="rolling crash-recovery ckpt interval (0=off)")
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--mbs", type=int, default=1)
    p.add_argument("--recompute", nargs="*", default=None)
    p.add_argument("--cuda-graph", choices=["te", "none"], default="te")
    p.add_argument("--grouped-gemm", choices=sorted(_GG), default="loop", help="TE grouped GEMM backend")
    p.add_argument("--no-grad-accum-fusion", action="store_true")
    p.add_argument("--ce", choices=["native", "te"], default="te", help="fused cross-entropy implementation")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--train-shards", type=int, default=N_TRAIN_SHARDS,
                   help="number of sorted training shards in the blend (sweep: 40; final run: all 471)")
    p.add_argument("--shard-start", type=int, default=0,
                   help="first training shard; with --data-offset, continue a run on fresh shards once its blend runs out")
    p.add_argument("--data-offset", type=int, default=0,
                   help="consumed samples at the switch to the --shard-start blend; the sampler indexes the new "
                        "blend from (consumed - offset), so no sample of the old blend is repeated")
    p.add_argument("--gbs", type=int, help="override global batch (benchmark/profiling only)")
    p.add_argument("--profile", action="store_true", help="torch profiler trace of steps 6-8 into <run>/tb")
    args = p.parse_args()
    if not (args.flops or args.iters):
        p.error("need --flops or --iters")
    if args.parent:
        # Extend a finished run: resume its constant-LR phase from its first persistent ckpt (the cooldown
        # start) with the parent's warmup, read from its log header, so the LR schedule is identical.
        # A scheduler retry passes --load <own ckpt> as well; then only the warmup comes from the parent.
        pdir = f"{RUNS}/{args.parent}/ckpt"
        its = sorted(int(d[5:]) for d in os.listdir(pdir) if d.startswith("iter_")) if os.path.isdir(pdir) else []
        if len(its) < 2:
            p.error(f"parent {args.parent} has no [cooldown-start, final] ckpts: {its}")
        hdr = [re.search(r"scaling run \S+: .*?\(warmup (\d+),", l) for f in sorted(glob.glob(f"{LOGS}/{args.parent}*.log"))
               for l in open(f, errors="replace")]
        hdr = [m for m in hdr if m]
        if not hdr:
            p.error(f"no scaling-run header in logs of parent {args.parent}")
        args.warmup = int(hdr[0][1]) if args.warmup is None else args.warmup
        if args.load is None or not _has_ckpt(args.load):
            args.load, args.load_step = pdir, its[0]
    if args.load:
        # Bridge silently trains from scratch if the load dir is missing; never let that happen.
        if not os.path.isabs(args.load) or not _has_ckpt(args.load):
            p.error(f"--load must be an existing absolute path: {args.load}")
        if args.load_step is not None and not os.path.isdir(f"{args.load}/iter_{args.load_step:07d}"):
            p.error(f"no checkpoint iter_{args.load_step:07d} under {args.load}")
    cfg = build(args)
    if args.data_offset:
        _offset_train_sampler(args.data_offset)
    pretrain(config=cfg, forward_step_func=forward_step)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
