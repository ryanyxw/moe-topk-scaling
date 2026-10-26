#!/bin/bash
# Rebuild results/evals/results.json, the eval figure, and the "Downstream evaluations" section of the report.
set -euo pipefail
source "$(dirname "$0")/../env/config.sh"
python3 $REPO/eval/collect.py
python3 $REPO/analysis/plot_evals.py
python3 $REPO/eval/report_evals.py $REPO/results/report.html
