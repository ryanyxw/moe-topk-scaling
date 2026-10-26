#!/usr/bin/env python3
"""Summarize scaling runs: status, top-k, LR, budget, tokens, final validation loss, throughput, health.

    python3 results.py [glob ...]      (default: all runs in logs/ that have a 'scaling run' header)
"""

import glob
import os
import re
import statistics
import sys


LOGS = os.environ.get("SCALING_ROOT", "/mnt/nvme/scaling") + "/logs"
HDR = re.compile(
    r"scaling run (\S+): topk=(\d+) lr=([\d.e-]+) iters=(\d+) .*?tokens=([\d.]+)B flops/step=([\d.e+]+) total=([\d.e+]+)"
)
ITER = re.compile(r"iteration +(\d+)/ +(\d+) .*?\(TFLOP/s/GPU\): ([\d.]+).*?lm loss: ([\d.E+-]+)")
VAL = re.compile(r"validation loss at iteration (\d+) on validation set \| lm loss value: ([\d.E+-]+)")
VAL2 = re.compile(r"validation loss at iteration (\d+) \| lm loss value: ([\d.E+-]+)")


def parse(path):
    text = open(path, errors="replace").read()
    h = HDR.search(text)
    if not h:
        return None
    name = os.path.basename(path)[:-4]
    st_path = path[:-4] + ".status"
    status = open(st_path).read().split()[0] if os.path.exists(st_path) else "?"
    its = [(int(a), int(b), float(c), float(d)) for a, b, c, d in ITER.findall(text)]
    vals = [(int(i), float(v)) for i, v in VAL.findall(text)] or [(int(i), float(v)) for i, v in VAL2.findall(text)]
    iters = int(h[4])
    final_val = next((v for i, v in reversed(vals) if i == iters), None)
    tf = statistics.median([x[2] for x in its[len(its) // 4:]]) if len(its) > 4 else float("nan")
    issues = []
    if re.search(r"number of nan iterations: +[1-9]", text):
        issues.append("nan")
    if re.search(r"number of skipped iterations: +[1-9]", text):
        issues.append("skipped")
    if "Traceback" in text:
        issues.append("traceback")
    losses = [x[3] for x in its]
    if any(not (l == l and l < 20) for l in losses):
        issues.append("bad_loss")
    # Loss spike: a logged train loss > 0.5 above the running minimum after the first 10%.
    if len(losses) > 20:
        m = min(losses[: len(losses) // 10])
        for l in losses[len(losses) // 10:]:
            if l > m + 0.5:
                issues.append("spike")
                break
            m = min(m, l)
    return dict(
        name=name, status=status, k=int(h[2]), lr=float(h[3]), iters=iters, tokens=float(h[5]),
        flops=float(h[7]), it=its[-1][0] if its else 0, tf=tf, loss=losses[-1] if losses else float("nan"),
        val=final_val, issues=issues,
    )


def main():
    pats = sys.argv[1:] or ["*"]
    paths = sorted({p for pat in pats for p in glob.glob(f"{LOGS}/{pat}.log")})
    rows = [r for r in map(parse, paths) if r]
    print(f"{'run':28} {'status':16} {'k':>3} {'lr':>7} {'budget':>8} {'tokens':>7} {'progress':>11} {'TF/s/GPU':>8} {'train':>6} {'VAL':>7}  issues")
    for r in sorted(rows, key=lambda r: (r["flops"], r["k"], r["lr"])):
        val = f"{r['val']:.4f}" if r["val"] is not None else "-"
        print(
            f"{r['name']:28} {r['status']:16} {r['k']:>3} {r['lr']:>7.1e} {r['flops']:>8.1e} {r['tokens']:>6.2f}B "
            f"{r['it']:>5}/{r['iters']:<5} {r['tf']:>8.1f} {r['loss']:>6.3f} {val:>7}  {','.join(r['issues']) or 'ok'}"
        )


if __name__ == "__main__":
    main()
