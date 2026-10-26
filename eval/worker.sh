#!/bin/bash
# One worker per GPU: claim the next model in eval_queue.txt, serve it with vLLM, check its loss, and run the
# NEL tasks that have no result yet (up to 3 attempts). Usage: eval/worker.sh <gpu>  (eval/run_evals.sh starts one per GPU).
# Status per model in $EVAL_ROOT/status/<name>; event log in $EVAL_ROOT/logs/workers.log.
source "$(dirname "$0")/../env/config.sh"
G=$1; E=$EVAL_ROOT; PORT=$((8100 + G)); C=vllm_g$G; X=$REPO/eval
cd $E; mkdir -p status logs cfg results
S=$HF_HOME_HOST/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/$BASE_REV
PY=$NEL_VENV/bin/python
while read name; do
  mkdir status/$name.lock 2>/dev/null || continue  # atomic claim
  st() { echo "$(date +%T) gpu$G $*" > status/$name; echo "$(date +%T) gpu$G $name $*" >> logs/workers.log; }
  [ -z "$($PY $X/missing.py $name)" ] && { st DONE; continue; }
  st CLAIMED
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do sudo cp -L $S/$f hf/$name/$f; done
  docker rm -f $C >/dev/null 2>&1
  docker run -d --name $C --gpus "\"device=$G\"" --ipc=host --network=host -v $E/hf:/models "$VLLM_IMAGE" \
    /models/$name --served-model-name $name --trust-remote-code --max-model-len 8192 --mamba_ssm_cache_dtype float32 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.5 --port $PORT > /dev/null
  ok=""; for i in $(seq 90); do sleep 10; curl -s localhost:$PORT/v1/models > /dev/null && { ok=1; break; }; docker ps -q -f name=$C | grep -q . || break; done
  if [ -z "$ok" ]; then docker logs $C > logs/vllm_$name.log 2>&1; docker rm -f $C >/dev/null; st FAILED_SERVE; continue; fi
  st SERVING
  grep -q "VLLM_LOSS $name " logs/vllm_loss.log || docker run --rm --network=host $DOCKER_MOUNTS "$IMAGE" \
    python $X/vllm_loss.py http://localhost:$PORT $name 16 2>&1 | grep VLLM_LOSS >> logs/vllm_loss.log
  for attempt in 1 2 3; do
    miss=$($PY $X/missing.py $name); [ -z "$miss" ] && break
    n=$($PY $X/make_cfg.py $name $PORT $miss)
    st "EVAL attempt $attempt: $miss"
    $NEL_VENV/bin/nemo-evaluator-launcher run --config cfg/$name.yaml > logs/nel_${name}_$attempt.log 2>&1
    # The launcher submits and returns; jobs run in the background. Wait for this invocation's exit files.
    inv=$(grep -oE "Invocation ID: [0-9a-f]+" logs/nel_${name}_$attempt.log | awk '{print $3}')
    [ -z "$inv" ] && { st "NEL submit failed (attempt $attempt)"; sleep 30; continue; }
    t0=$(date +%s)
    while [ "$(ls results/$name/*-$inv/*/logs/stage.exit 2>/dev/null | wc -l)" -lt "$n" ] && [ $(( $(date +%s) - t0 )) -lt 18000 ]; do
      sleep 30
      curl -s localhost:$PORT/v1/models > /dev/null || { st "SERVER_DIED"; break 2; }
    done
  done
  docker logs $C > logs/vllm_$name.log 2>&1; docker rm -f $C > /dev/null
  miss=$($PY $X/missing.py $name)
  if [ -z "$miss" ]; then st DONE; else st "INCOMPLETE: missing $miss"; fi
done < "${QUEUE:-$X/eval_queue.txt}"
echo "$(date +%T) gpu$G worker exit" >> logs/workers.log
