"""Mean next-token loss through a vLLM server (prompt_logprobs) on the same windows as hf_loss.py."""
import os, sys, json, urllib.request
from megatron.core.datasets.indexed_dataset import IndexedDataset
url, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = IndexedDataset(os.environ.get("DATA_ROOT", "/mnt/nvme/data") + "/tok/016_00001")  # held-out validation shard
toks, i = [], 0
while len(toks) < n * 8193:
    toks.extend(ds[i].tolist()); i += 1
tot = 0.0
for w in range(n):
    p = toks[w * 8193:(w + 1) * 8193][:8191]  # fits max_model_len 8192 with 1 generated token
    body = json.dumps({"model": model, "prompt": p, "max_tokens": 1, "prompt_logprobs": 0, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"})))
    lp = r["choices"][0]["prompt_logprobs"][1:]
    vals = [d[str(t)]["logprob"] for d, t in zip(lp, p[1:])]
    tot += -sum(vals) / len(vals)
print(f"VLLM_LOSS {model} {tot / n:.4f} over {n} windows (8190 targets each)")
