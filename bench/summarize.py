#!/usr/bin/env python3
"""Summarize benchmark logs: steady-state step time, tokens/s, model TFLOP/s/GPU, MFU.

Usage: python bench/summarize.py [--skip N] [log names or globs...]   (default: all logs)
Steady state = iterations after the first --skip (default 3: warmup + CUDA graph capture).
"""

import argparse
import glob
import os
import re
import statistics

H100_BF16_DENSE_PEAK = 989.4  # TFLOP/s
SEQ = 8192
NGPU = 6

ITER = re.compile(
    r"iteration +(\d+)/ +\d+ .*?per iteration \(ms\): ([\d.]+) .*?\(TFLOP/s/GPU\): ([\d.]+)"
    r".*?global batch size: +(\d+) .*?lm loss: ([\d.E+-]+)"
)
MEM = re.compile(r"\[Rank (\d+)\].*?mem-max-allocated-gigabytes: ([\d.]+)")
TIMER = re.compile(r"^\s+([\w-]+) \.+: \(([\d.]+), ([\d.]+)\)")


def summarize(path, skip):
    rows, mem, timers = [], {}, {}
    with open(path, errors="replace") as f:
        for line in f:
            if m := ITER.search(line):
                rows.append((int(m[1]), float(m[2]), float(m[3]), int(m[4]), float(m[5])))
            for m in MEM.finditer(line):
                mem[int(m[1])] = max(mem.get(int(m[1]), 0.0), float(m[2]))
            if m := TIMER.match(line):
                timers.setdefault(m[1], []).append(float(m[3]))  # max across ranks
    text = open(path, errors="replace").read()
    oom = "out of memory" in text
    status_path = path[:-4] + ".status"
    status = open(status_path).read().split()[0] if os.path.exists(status_path) else "?"
    issues = []
    if status not in ("OK", "?", "RUNNING"):
        issues.append(status)
    nan_iters = [int(x) for x in re.findall(r"number of nan iterations: +(\d+)", text)]
    skipped = [int(x) for x in re.findall(r"number of skipped iterations: +(\d+)", text)]
    if any(nan_iters):
        issues.append(f"nan_iters={sum(nan_iters)}")
    if any(skipped):
        issues.append(f"skipped={sum(skipped)}")
    losses = [r[4] for r in rows]
    if any(not (l == l and abs(l) < 1e4) for l in losses):
        issues.append("nonfinite_loss")
    elif len(losses) >= 4 and losses[-1] >= losses[0]:
        issues.append("loss_not_decreasing")
    if "Traceback" in text:
        issues.append("traceback")
    if "[watchdog]" in text:
        issues.append("watchdog_kill")
    steady = [r for r in rows if r[0] > skip]
    if not steady:
        return dict(name=os.path.basename(path)[:-4], iters=len(rows), oom=oom, issues=issues)
    ms = statistics.median(r[1] for r in steady)
    tf = statistics.median(r[2] for r in steady)
    gbs = steady[-1][3]
    tok_s = gbs * SEQ / (ms / 1000)
    opt = timers.get("optimizer", [])
    fb = timers.get("forward-backward", [])
    return dict(
        name=os.path.basename(path)[:-4],
        iters=len(rows),
        n=len(steady),
        step_s=ms / 1000,
        tok_s=tok_s,
        tok_s_gpu=tok_s / NGPU,
        tflops=tf,
        mfu=tf / H100_BF16_DENSE_PEAK,
        loss=steady[-1][4],
        mem=max(mem.values()) if mem else float("nan"),
        fb_s=statistics.median(fb[skip:] or fb) / 1000 if fb else float("nan"),
        opt_s=statistics.median(opt[skip:] or opt) / 1000 if opt else float("nan"),
        oom=oom,
        issues=issues,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip", type=int, default=3)
    p.add_argument("logs", nargs="*")
    a = p.parse_args()
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    pats = a.logs or ["*"]
    paths = sorted({q for pat in pats for q in glob.glob(os.path.join(here, pat if pat.endswith(".log") else pat + ".log"))})
    hdr = f"{'config':34} {'it':>3} {'step_s':>7} {'tok/s':>8} {'tok/s/gpu':>9} {'TFLOP/s':>7} {'MFU':>6} {'fwdbwd_s':>8} {'optim_s':>7} {'peakGB':>6} {'loss':>6}  health"
    print(hdr)
    print("-" * len(hdr))
    for path in paths:
        s = summarize(path, a.skip)
        if "step_s" not in s:
            print(f"{s['name']:34} {s['iters']:>3}  {'OOM' if s['oom'] else 'no steady-state iterations'}  {','.join(s['issues'])}")
            continue
        print(
            f"{s['name']:34} {s['iters']:>3} {s['step_s']:7.2f} {s['tok_s']:8.0f} {s['tok_s_gpu']:9.0f} "
            f"{s['tflops']:7.1f} {s['mfu']:6.1%} {s['fb_s']:8.2f} {s['opt_s']:7.2f} {s['mem']:6.1f} {s['loss']:6.2f}  "
            f"{','.join(s['issues']) or 'ok'}"
        )


if __name__ == "__main__":
    main()
