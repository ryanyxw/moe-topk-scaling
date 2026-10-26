#!/bin/bash
# One-time eval setup: NEL venv, local tokenizer for the lm-eval client, shared HF dataset cache, images.
set -euo pipefail
source "$(dirname "$0")/../env/config.sh"
E=$EVAL_ROOT; mkdir -p $E/{hf,results,cfg,logs,status,tokenizer,hf_cache}
[ -x $NEL_VENV/bin/nemo-evaluator-launcher ] || { python3 -m venv $NEL_VENV && $NEL_VENV/bin/pip install -q -r $REPO/eval/requirements.txt; }
# The lm-eval client inside the eval container reads the tokenizer from a mount, not the Hub
# (concurrent Hub downloads of tokenizer.json failed partway and fell back to a broken tiktoken path).
python3 - <<PY
from huggingface_hub import hf_hub_download
import shutil
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    shutil.copy(hf_hub_download("$BASE_MODEL", f, revision="$BASE_REV"), "$E/tokenizer/" + f)
PY
docker pull "$VLLM_IMAGE"
docker pull nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03
# Opt out of launcher telemetry.
mkdir -p ~/.config/nemo-evaluator; [ -f ~/.config/nemo-evaluator/config.yaml ] || printf 'telemetry:\n  level: 0\n' > ~/.config/nemo-evaluator/config.yaml
echo "eval setup done: $E"
