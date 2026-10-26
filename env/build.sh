#!/bin/bash
# Build the training image (mbridge:cu129): CUDA 12.9, torch 2.13, TE 2.20.2, mamba-ssm 2.3.1, Apex, Megatron-Bridge.
# Needs the submodule: git submodule update --init --recursive
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO"
[ -f third_party/Megatron-Bridge/3rdparty/Megatron-LM/setup.py ] || { echo "run: git submodule update --init --recursive"; exit 1; }
DOCKER_BUILDKIT=1 docker build -f env/Dockerfile.cu129 -t "$IMAGE" . 2>&1 | tee env/build.log
