"""Export a scaling-run Megatron checkpoint to a HF NemotronHForCausalLM directory.
Usage: python export_hf.py <megatron iter dir> <out dir> <topk>"""
import json, shutil, sys
from pathlib import Path
from transformers import AutoConfig
from huggingface_hub import hf_hub_download
from megatron.bridge import AutoBridge

BASE_MODEL, BASE_REV = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
base_file = lambda f: hf_hub_download(BASE_MODEL, f, revision=BASE_REV)  # pinned snapshot the study used

src, out, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
base = json.load(open(base_file("config.json")))
base.pop("auto_map")  # transformers >= 4.57 / vLLM implement nemotron_h natively
base.update(
    hidden_size=1024, hybrid_override_pattern="MEMEM*EMEMEM*EMEMEM*EMEME", num_hidden_layers=25,
    intermediate_size=1024, moe_intermediate_size=1024, moe_shared_expert_intermediate_size=512,
    mamba_num_heads=32, mamba_head_dim=64, n_groups=8, ssm_state_size=128,
    num_attention_heads=8, num_key_value_heads=2, head_dim=128,
    n_routed_experts=128, num_experts_per_tok=k,
    norm_topk_prob=k > 1,  # Megatron skips top-k weight normalization when k == 1
    max_position_embeddings=8192,
)
tmp = out.with_name(out.name + ".cfg"); tmp.mkdir(parents=True, exist_ok=True)
json.dump(base, open(tmp / "config.json", "w"), indent=2)
cfg = AutoConfig.from_pretrained(tmp)
bridge = AutoBridge.from_hf_config(cfg)
bridge.export_ckpt(str(src), str(out), show_progress=False, strict=True)
for f in (src / "tokenizer").iterdir():
    shutil.copy(f, out / f.name)
shutil.rmtree(tmp)
print("EXPORTED", out, json.load(open(out / "config.json"))["num_experts_per_tok"])
