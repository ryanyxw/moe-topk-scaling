import os, sys, json, urllib.request, torch
from megatron.core.datasets.indexed_dataset import IndexedDataset
from transformers import AutoModelForCausalLM
path, url, model, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ds = IndexedDataset(os.environ.get("DATA_ROOT", "/mnt/nvme/data") + "/tok/016_00001")  # held-out validation shard
toks, i = [], 0
while len(toks) < n * 8193:
    toks.extend(ds[i].tolist()); i += 1
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
for w in range(n):
    p = toks[w * 8193:(w + 1) * 8193][:8191]
    with torch.no_grad():
        lp = torch.log_softmax(m(torch.tensor([p]).cuda()).logits.float()[0], -1)
    hf = lp[:-1].gather(1, torch.tensor(p[1:]).cuda()[:, None])[:, 0].cpu()
    body = json.dumps({"model": model, "prompt": p, "max_tokens": 1, "prompt_logprobs": 0, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"})))
    vl = torch.tensor([d[str(t)]["logprob"] for d, t in zip(r["choices"][0]["prompt_logprobs"][1:], p[1:])])
    d = (vl - hf)
    for a, b in ((0, 64), (64, 1024), (1024, 4096), (4096, 8190)):
        print(f"win {w} pos {a:5d}-{b:5d}: hf {-hf[a:b].mean():.4f} vllm {-vl[a:b].mean():.4f} mean|diff| {d[a:b].abs().mean():.4f}")
