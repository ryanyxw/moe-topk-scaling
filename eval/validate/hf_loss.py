"""Mean next-token loss of an HF export on the first N 8192-token windows of the validation shard."""
import os, sys, torch
from megatron.core.datasets.indexed_dataset import IndexedDataset
from transformers import AutoModelForCausalLM
path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16
ds = IndexedDataset(os.environ.get("DATA_ROOT", "/mnt/nvme/data") + "/tok/016_00001")  # held-out validation shard
toks, i = [], 0
while len(toks) < n * 8193:
    toks.extend(ds[i].tolist()); i += 1
x = torch.tensor(toks[: n * 8193]).view(n, 8193).cuda()
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
tot = 0.0
with torch.no_grad():
    for r in x:
        logits = m(r[None, :-1]).logits.float()
        tot += torch.nn.functional.cross_entropy(logits[0], r[1:]).item()
print(f"HF_LOSS {path} {tot / n:.4f} over {n} windows")
