#!/bin/bash
# Download FineWeb-Edu sample/350BT (~930 GB parquet) and tokenize it (~1.3 TB) into $DATA_ROOT, in the image.
# Resumable: rerun after an interruption. Progress -> $DATA_ROOT/prep.log
set -euo pipefail
source "$(dirname "$0")/../env/config.sh"
mkdir -p "$DATA_ROOT"
docker run -d --name dataprep $DOCKER_MOUNTS -e HF_TOKEN="${HF_TOKEN:-}" "$IMAGE" \
  bash -c "python $REPO/data/prep_data.py > $DATA_ROOT/prep.log 2>&1"
echo "started container dataprep; tail -f $DATA_ROOT/prep.log"
