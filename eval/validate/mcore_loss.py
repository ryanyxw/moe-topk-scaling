"""Ground-truth per-position loss from the Megatron checkpoint on the same windows as cmp_hf_vllm.py."""
import os, sys, json, torch
from pathlib import Path
from megatron.core.datasets.indexed_dataset import IndexedDataset
from transformers import AutoConfig
from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import temporary_distributed_context
ckpt, hfdir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = IndexedDataset(os.environ.get("DATA_ROOT", "/mnt/nvme/data") + "/tok/016_00001")  # held-out validation shard
toks, i = [], 0
while len(toks) < n * 8193:
    toks.extend(ds[i].tolist()); i += 1
cfg = AutoConfig.from_pretrained(hfdir)
bridge = AutoBridge.from_hf_config(cfg)
with temporary_distributed_context(backend="nccl"):
    model = bridge.load_megatron_model(ckpt, wrap_with_ddp=False)
    m = model[0].cuda().eval()
    for w in range(n):
        p = torch.tensor([toks[w * 8193:(w + 1) * 8193][:8191]]).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(input_ids=p, position_ids=torch.arange(p.shape[1], device="cuda")[None], attention_mask=None)
        lp = torch.log_softmax(logits.float()[0], -1)
        t = lp[:-1].gather(1, p[0, 1:, None])[:, 0].cpu()
        for a, b in ((0, 64), (64, 1024), (1024, 4096), (4096, 8190)):
            print(f"win {w} pos {a:5d}-{b:5d}: mcore {-t[a:b].mean():.4f}")
