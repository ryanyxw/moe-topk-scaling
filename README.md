# MoE top-k scaling

How does loss scale with the number of active experts (top-k) and training tokens, at a fixed total parameter
count? This repo has everything used to answer that for a small Nemotron-3-Nano-style hybrid Mamba/MoE model:
environment, data prep, training, isoFLOP fits, downstream evals, raw results and the report.

**Report:** [`results/report.html`](results/report.html) (download and open it, or view the
[rendered copy](https://claude.ai/artifact/4fXVnFyvEtxWSHDSpayaqw)).
**Checkpoints and data** are not in git; [`CHECKPOINTS.md`](CHECKPOINTS.md) lists where each one is and how it was made.

## Results

- **Model:** 25 layers (11 Mamba, 11 MoE, 3 attention), hidden 1024, 128 routed experts (FFN 1024) plus 1 shared
  expert, 3.3B total parameters for every top-k. Data: FineWeb-Edu sample/350BT, Nemotron tokenizer, 8192-token sequences,
  1.05M tokens per step, WSD schedule with a 1-sqrt cooldown over the last 20%.
- **IsoFLOP sweep** at 8e18, 2e19, 5e19 and 1e20 FLOPs (top-k 1–32). The optimal top-k rises slowly with compute:
  k* ≈ 5.9, 6.3, 8.2, 9.1 (k* ∝ C^0.18–0.24), while optimal tokens grow ∝ C^0.91–0.93.
- **Joint fit** L(k, D) = E + A·k^−α + B·D^−β on the C1–C4 points (RMS residual 0.0027) predicted **2.253** for top-k 12
  at 4.5e20 FLOPs (134B tokens). The **final run measured 2.2405**, 0.013 better than predicted, at a 4.5× extrapolation.
- **Downstream** (NeMo Evaluator, 10 tasks × 21 checkpoints): the 8-task average tracks validation loss at r = −0.99,
  top-k 8 is best at every sweep budget on both loss and accuracy, and the final run averages 60.0% (HellaSwag 61.1).
  MMLU stays at chance for every model.

Tables: [`results/train_runs.txt`](results/train_runs.txt) (every run's validation loss),
[`results/evals/results.json`](results/evals/results.json) (every benchmark score with standard error).

## Layout

```
env/          Dockerfile (CUDA 12.9 Megatron-Bridge image), build script, paths config (env/config.sh)
third_party/  Megatron-Bridge submodule, pinned to the commit used (e2f2496; Megatron-LM 3e95097)
bench/        throughput benchmark of the full Nemotron 3 Nano 30B-A3B recipe on 6 GPUs (the first task)
data/         FineWeb-Edu download + tokenization into Megatron indexed shards
train/        train.py (model + schedule), launch.sh (docker + watchdog), scheduler.py (GPU job queue),
              queue.txt (every job as run), results.py (summarize logs)
analysis/     isoFLOP parabola fits, joint L(k, D) fit, plots
eval/         Megatron -> HF export, vLLM serving, NeMo Evaluator configs and workers, result collection
  validate/   checks that the exported model matches the Megatron checkpoint (used once, see below)
results/      report.html, figures, train_runs.txt, train_logs/ and bench_logs/ (gzipped), evals/ (scores, raw lm-eval JSON)
```

All paths come from [`env/config.sh`](env/config.sh). The defaults are the paths on the original machine
(`/mnt/nvme/data`, `/mnt/nvme/scaling`). Set `DATA_ROOT`, `SCALING_ROOT`, `EVAL_ROOT` in the environment to use other paths.
Containers mount these trees at the same path, so host and container paths agree.

## Reproducing

Hardware used: 8× H100 80GB (GPU 1 was never used), driver 550 (hence the CUDA 12.9 image), ~3 TB of NVMe.

### 0. Environment

```bash
git clone --recursive https://github.com/ryanyxw/moe-topk-scaling.git && cd moe-topk-scaling
env/build.sh            # builds mbridge:cu129: torch 2.13 cu129, TE 2.20.2, mamba-ssm 2.3.1 (patched), Apex, Megatron-Bridge
```

The Megatron-Bridge checkout is mounted into the container at runtime, so the code used is the submodule's.
Nothing in `third_party/` is modified.

### 1. Data (~930 GB download, 1.3 TB tokenized)

```bash
data/run_prep.sh        # docker container "dataprep"; resumable; progress in $DATA_ROOT/prep.log
```

Downloads the 472 parquet files of `HuggingFaceFW/fineweb-edu` `sample/350BT` and tokenizes each into
`$DATA_ROOT/tok/<shard>.{bin,idx}` with the Nemotron 3 Nano tokenizer (EOS after every document, no BOS; identical
to Megatron's `preprocess_data.py --append-eod`). The last shard (`016_00001`) is the validation set.
The sweep trains on the first 40 shards (C4 top-k 4/8 continue on shards 40–79); the final run uses all 471.

### 2. Training

```bash
CPUSET=0-23,104-127 nohup python3 train/scheduler.py > /dev/null 2>&1 &    # log: $SCALING_ROOT/logs/scheduler.log
```

The scheduler runs the jobs in [`train/queue.txt`](train/queue.txt) on free GPU slots, in order, re-reading the file
every 30 s, and retries a crashed run from its rolling checkpoint. Each job calls `train/launch.sh`, which runs
`train/train.py` in the image and writes `$SCALING_ROOT/logs/<run>.{log,status}` and `$SCALING_ROOT/runs/<run>/ckpt`.
Create `$SCALING_ROOT/STOP_SCHEDULER` to stop it once nothing is running. (`CPUSET` kept training off the cores used by
tokenization; leave it unset if tokenization is finished.)

A single run without the scheduler, e.g. C1 top-k 8 on GPUs 0 and 2:

```bash
GPUS=0,2 train/launch.sh c1_k8 --name c1_k8 --topk 8 --flops 8e18 --lr 1.5e-3 --ep 2 --mbs 4 --recompute moe_act \
  --grouped-gemm cutlass --eval-iters 20 --eval-interval 1000 --log-interval 10
```

`--flops` sets the number of steps from Megatron-Bridge's FLOP counter. `--parent <run>` resumes from that run's
cooldown-start checkpoint, so larger budgets reuse the constant-LR phase (C1 → C2 → C3 → C4). Time on 2 GPUs: C1 5–6 h,
C2 8–12 h, C3 21–23 h, C4 up to 40 h (top-k 16); the final run took 91 h on 6 GPUs.
The order of jobs and the one manual step (C4 top-k 4/8 switching data shards) are described in `train/queue.txt`.

```bash
python3 train/results.py "c*" "final*"     # status, tokens, throughput, final validation loss for every run
```

### 3. Fits and figures

```bash
python3 analysis/fit_isoflop.py     # per-budget parabola minima (k*, D*, L*) and power laws in C
python3 analysis/joint_fit_k.py     # joint L(k, D) fit (the one used for the prediction); joint_fit.py uses FLOPs/token instead of k
python3 analysis/plot_isoflop.py    # results/figures/isoflop.svg, extrap.svg
```

The validation losses are written into these scripts (copied from `train/results.py` output).
`MIN_C=8e18 python3 analysis/joint_fit_k.py` drops C0, as in the report.

### 4. Downstream evals

```bash
eval/setup.sh            # NEL venv (nemo-evaluator-launcher 0.2.6), local tokenizer, pulls vLLM 0.12 + lm-eval 26.03 images
eval/export_all.sh       # every model in eval/models.txt -> $EVAL_ROOT/hf/<model> (Megatron-Bridge export, ~1 min each)
eval/run_evals.sh        # one worker per GPU (0,2-7); ~2.3 h for all 21 models; log: $EVAL_ROOT/logs/workers.log
eval/report.sh           # results/evals/results.json, results/figures/evals.svg, and the eval section of results/report.html
```

Each worker takes the next model from [`eval/eval_queue.txt`](eval/eval_queue.txt), serves it with vLLM on one GPU,
checks its validation loss through the server, writes a NeMo Evaluator config with [`eval/make_cfg.py`](eval/make_cfg.py),
runs it and waits for every task, retrying tasks with no result up to 3 times. Tasks (lm-eval-harness, scored by answer
log-likelihood): HellaSwag 10-shot, ARC-Challenge 0-shot and 25-shot, ARC-Easy, PIQA, Winogrande 5-shot, OpenBookQA,
SciQ, LAMBADA, MMLU 5-shot.

Notes on the export:
- `export_hf.py` builds an HF `NemotronHForCausalLM` config from the 30B Nano config with this model's sizes,
  and exports with `AutoBridge.export_ckpt(..., strict=True)`. Top-k 1 sets `norm_topk_prob=False` (Megatron does not
  normalize a single expert's weight).
- `fix_config.py` rewrites the config in the original Nano layout with its remote code, because vLLM 0.12 does not read
  the transformers-5 layout. Both come from the base model pinned at revision `bf77c31`.
- Check: vLLM's per-token log-probs match a Megatron forward pass of the checkpoint to 0.003 at every position
  (`eval/validate/mcore_loss.py` vs the server), and the served validation loss is 0.01–0.02 below the Megatron
  in-run number for every model ([`results/evals/vllm_loss.log`](results/evals/vllm_loss.log); different windows).
  Hugging Face transformers 5's native NemotronH is ≈0.14 per token off (`eval/validate/cmp_hf_vllm.py`), so it is not used.

### Benchmark (first task)

`bench/` is the throughput benchmark of the stock Nemotron 3 Nano 30B-A3B pretraining recipe on 6 GPUs
(`bench/sweeps/sweep*.sh`, summarized by `bench/summarize.py`; logs in `results/bench_logs/`). The best setup reached
74K tokens/s (27.8% MFU, 275 TFLOP/s per GPU): PP6, GBS 384, MBS 2, optimizer CPU offload, MoE activation
recompute on the first 3 pipeline stages (`s7_rcstage3`, measured with forced-balanced routing).
