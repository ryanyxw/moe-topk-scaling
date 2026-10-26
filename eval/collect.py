"""Collect per-model benchmark scores from NEL results into results/evals/results.json and print a table."""
import glob, json, os
E = os.environ.get("EVAL_ROOT", "/mnt/nvme/scaling/eval")  # exports, configs, NEL outputs
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "evals", "results.json")
# lm-eval result key -> (column, metric). acc_norm where the standard leaderboards use it.
KEYS = {"hellaswag": ("HellaSwag", "acc_norm"), "arc_challenge": ("ARC-C", "acc_norm"), "adlr_arc_challenge_llama": ("ARC-C 25s", "acc_norm"),
        "arc_easy": ("ARC-E", "acc_norm"), "piqa": ("PIQA", "acc_norm"), "winogrande": ("Winogrande 5s", "acc"),
        "openbookqa": ("OBQA", "acc_norm"), "sciq": ("SciQ", "acc_norm"), "lambada_openai": ("LAMBADA", "acc"),
        "mmlu": ("MMLU 5s", "acc")}
COLS = [v[0] for v in KEYS.values()]
out = {}
for mdir in sorted(glob.glob(f"{E}/results/*/")):
    name = os.path.basename(mdir.rstrip("/")); row = {}
    for f in glob.glob(f"{mdir}*/*/artifacts/*/results_*.json"):
        for k, v in json.load(open(f))["results"].items():
            if k in KEYS:
                col, met = KEYS[k]
                row[col] = (v[f"{met},none"], v[f"{met}_stderr,none"])
    if row: out[name] = row
json.dump(out, open(OUT, "w"), indent=1)
print(f"{'model':16s}" + "".join(f"{c:>14s}" for c in COLS))
for n, r in out.items():
    print(f"{n:16s}" + "".join(f"{100*r[c][0]:14.1f}" if c in r else f"{'-':>14s}" for c in COLS))
