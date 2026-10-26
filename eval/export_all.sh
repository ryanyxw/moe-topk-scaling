#!/bin/bash
# Export every model in models.txt (<name> <run> <topk> [iter]) to $EVAL_ROOT/hf/<name>: HF format with a
# vLLM-ready config. Default iter = the run's last checkpoint. One GPU ($EXPORT_GPU), ~1 min per model.
source "$(dirname "$0")/../env/config.sh"
E=$EVAL_ROOT; GPU=${EXPORT_GPU:-7}
mkdir -p $E/hf $E/logs
while read name run k it; do
  [ -f $E/hf/$name/config_tf5.json ] && continue
  [ -z "$it" ] && it=$(ls $SCALING_ROOT/runs/$run/ckpt | grep iter_ | sort | tail -1 | sed 's/iter_//')
  if [ -f $E/hf/$name/config.json ]; then step="python $REPO/eval/fix_config.py $E/hf/$name"
  else step="python $REPO/eval/export_hf.py $SCALING_ROOT/runs/$run/ckpt/iter_$it $E/hf/$name $k && python $REPO/eval/fix_config.py $E/hf/$name"; fi
  docker run --rm --gpus "\"device=$GPU\"" --ipc=host --network=host $DOCKER_MOUNTS -v $E:$E "$IMAGE" \
    bash -c "$step" > $E/logs/export_$name.log 2>&1
  echo "$(date +%T) $name iter_$it $(grep -E '^(EXPORTED|FIXED)' $E/logs/export_$name.log | tr '\n' ' ')"
done < "${MODELS:-$REPO/eval/models.txt}"
