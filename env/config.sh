# Paths and machine settings, sourced by every shell script in this repo.
# Defaults are the machine the study ran on; override any of them in the environment.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO
export DATA_ROOT=${DATA_ROOT:-/mnt/nvme/data}             # raw/ parquet, tok/ Megatron shards, cache/ dataset indices
export SCALING_ROOT=${SCALING_ROOT:-/mnt/nvme/scaling}    # runs/<name>/ckpt, logs/, eval/ (exports, eval outputs)
export MBRIDGE=${MBRIDGE:-$REPO/third_party/Megatron-Bridge}  # mounted over /opt/Megatron-Bridge in the image
export IMAGE=${IMAGE:-mbridge:cu129}                       # built by env/build.sh
export VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.12.0}
export EVAL_ROOT=${EVAL_ROOT:-$SCALING_ROOT/eval}          # hf/, results/, cfg/, logs/, status/
export NEL_VENV=${NEL_VENV:-$EVAL_ROOT/nel-venv}           # nemo-evaluator-launcher 0.2.6 (eval/requirements.txt)
export HF_HOME_HOST=${HF_HOME_HOST:-$HOME/.cache/huggingface}
# Base model the architecture, tokenizer and HF remote code come from, pinned to the revision used.
export BASE_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
export BASE_REV=bf77c3174f68ad409e1c2aa60daeb46e32d1c606

# Mount the data and output trees at the same path inside containers, so host and container paths agree.
DOCKER_MOUNTS="-v $MBRIDGE:/opt/Megatron-Bridge -v $REPO:$REPO -v $DATA_ROOT:$DATA_ROOT -v $SCALING_ROOT:$SCALING_ROOT \
  -v $HF_HOME_HOST:/root/.cache/huggingface -e DATA_ROOT=$DATA_ROOT -e SCALING_ROOT=$SCALING_ROOT"
