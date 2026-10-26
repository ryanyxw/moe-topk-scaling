# Checkpoints, data and eval outputs

None of these are in git; they are on the machine the study ran on (local NVMe RAID at `/mnt/nvme`).
All paths below use the defaults in [`env/config.sh`](env/config.sh); override `SCALING_ROOT`, `DATA_ROOT` or `EVAL_ROOT` to relocate.

| What | Path | Size |
|---|---|---|
| FineWeb-Edu sample/350BT parquet | `/mnt/nvme/data/raw/<shard>.parquet` | 930 GB |
| Tokenized shards (Megatron `.bin/.idx`, int32) | `/mnt/nvme/data/tok/<shard>.{bin,idx}` | 1.3 TB, 472 shards |
| Held-out validation shard | `/mnt/nvme/data/tok/016_00001` (the last shard) | |
| Dataset index cache | `/mnt/nvme/data/cache` | |
| Training checkpoints (Megatron `torch_dist`) | `/mnt/nvme/scaling/runs/<run>/ckpt/iter_<step>` | 88 GB per run (two ckpts), 262 GB for the final run |
| TensorBoard | `/mnt/nvme/scaling/runs/<run>/tb` | |
| Training logs | `/mnt/nvme/scaling/logs/<run>.log` (copies: [`results/train_logs/`](results/train_logs)) | |
| HF exports (vLLM-ready) | `/mnt/nvme/scaling/eval/hf/<model>` | 131 GB, 21 models |
| NeMo Evaluator outputs (per-sample logs, caches) | `/mnt/nvme/scaling/eval/results/<model>/` (scores: [`results/evals/raw/`](results/evals/raw)) | 364 GB |

Each checkpoint includes optimizer state, so any run can be resumed or extended.
A run keeps two persistent checkpoints: the **cooldown start** (end of the constant-LR phase, which the
next budget resumes from) and the **final** (after the 1-sqrt cooldown to LR 0; this is the one evaluated).
The rolling crash-recovery checkpoint (`ckpt/non_persistent`) is deleted when a run finishes.

## IsoFLOP runs (C1–C4)

"Resumed from" means the run loaded that parent's cooldown-start checkpoint and continued at constant LR
(same warmup, same data order) before its own, longer cooldown. Val loss is the full validation set at the final step.

| Run | Budget (FLOPs) | top-k | Tokens | Val loss | Cooldown start | Final | Resumed from |
|---|---|---|---|---|---|---|---|
| `c1_k1` | 8.0e+18 | 1 | 4.38B | 2.7574 | `iter_0003340` | `iter_0004175` | scratch |
| `c1_k2` | 8.0e+18 | 2 | 4.07B | 2.8271 | `iter_0003105` | `iter_0003881` | scratch |
| `c1_k4` | 8.0e+18 | 4 | 3.57B | 2.7428 | `iter_0002722` | `iter_0003402` | scratch |
| `c1_k8` | 8.0e+18 | 8 | 2.86B | 2.7421 | `iter_0002182` | `iter_0002728` | scratch |
| `c1_k16` | 8.0e+18 | 16 | 2.05B | 2.8100 | `iter_0001563` | `iter_0001954` | scratch |
| `c2_k4` | 2.0e+19 | 4 | 8.92B | 2.5726 | `iter_0006804` | `iter_0008505` | `c1_k4` |
| `c2_k8` | 2.0e+19 | 8 | 7.15B | 2.5707 | `iter_0005457` | `iter_0006821` | `c1_k8` |
| `c2_k16` | 2.0e+19 | 16 | 5.12B | 2.6016 | `iter_0003909` | `iter_0004886` | `c1_k16` |
| `c2_k32` | 2.0e+19 | 32 | 3.27B | 2.6903 | `iter_0002494` | `iter_0003117` | scratch |
| `c3_k4` | 5.0e+19 | 4 | 22.30B | 2.4602 | `iter_0017010` | `iter_0021262` | `c2_k4` |
| `c3_k8` | 5.0e+19 | 8 | 17.88B | 2.4385 | `iter_0013642` | `iter_0017052` | `c2_k8` |
| `c3_k16` | 5.0e+19 | 16 | 12.81B | 2.4565 | `iter_0009772` | `iter_0012215` | `c2_k16` |
| `c3_k32` | 5.0e+19 | 32 | 8.17B | 2.5197 | `iter_0006235` | `iter_0007794` | `c2_k32` |
| `c4_k4` | 1.0e+20 | 4 | 44.59B | 2.3898 | `iter_0034018` | `iter_0042523` | `c3_k4` |
| `c4_k8` | 1.0e+20 | 8 | 35.76B | 2.3619 | `iter_0027283` | `iter_0034104` | `c3_k8` |
| `c4_k16` | 1.0e+20 | 16 | 25.62B | 2.3750 | `iter_0019544` | `iter_0024430` | `c3_k16` |

Notes:
- `c3_k8` trained on 4 GPUs (DP2 x EP2) instead of 2; only data parallelism changed, the expert split is the same.
- `c4_k4` and `c4_k8` ran out of the 40-shard blend at step 28770 and were resumed from their step-28500
  rolling checkpoint on shards 40–79 with `--shard-start 40 --train-shards 40 --data-offset 3648000`
  (see [`train/queue.txt`](train/queue.txt) and `results/train_logs/c4_k*.retry*.log.gz`).
- C0 (2e18, LR sweep) runs used `--no-save`: no checkpoints, only logs.

## Final run

`final_k12`: top-k 12, 4.5e20 FLOPs, 134.3B tokens, 130139 steps of 126 x 8192 tokens, all 471 training shards,
6 GPUs (data parallel 6, experts split over 2 GPUs), val loss **2.2405** (predicted 2.253).

| Checkpoint | Step | LR phase | Val loss (20-batch in-run eval) |
|---|---|---|---|
| `runs/final_k12/ckpt/iter_0026027` | 26027 | constant | 2.458 |
| `runs/final_k12/ckpt/iter_0052054` | 52054 | constant | 2.414 |
| `runs/final_k12/ckpt/iter_0078081` | 78081 | constant | 2.394 |
| `runs/final_k12/ckpt/iter_0104108` | 104108 | cooldown start | 2.375 |
| `runs/final_k12/ckpt/iter_0130135` | 130135 | regular save interval, 4 steps before the end | |
| `runs/final_k12/ckpt/iter_0130139` | 130139 | **final** | 2.2405 (full validation set) |

The four constant-LR checkpoints are branch points: a different cooldown can be started from any of them with
`--load .../ckpt --load-step <step>`.

## Exported HF models and evals

[`eval/models.txt`](eval/models.txt) maps each evaluated model to its run and checkpoint (`<model> <run> <top-k> [iter]`,
default = the run's final checkpoint). The 21 models are the 16 isoFLOP finals, the final run, and its four
constant-LR checkpoints (`final_k12_26k` ... `final_k12_104k`).

To evaluate one checkpoint from scratch (see the README for the full pipeline):

```bash
echo "c1_k8 c1_k8 8" > /tmp/one.txt && MODELS=/tmp/one.txt eval/export_all.sh   # Megatron -> HF, ~1 min on one GPU
echo c1_k8 > /tmp/q.txt && QUEUE=/tmp/q.txt eval/worker.sh 0                   # serve with vLLM on GPU 0, run 10 tasks
python3 eval/collect.py                                                         # scores -> results/evals/results.json
```
