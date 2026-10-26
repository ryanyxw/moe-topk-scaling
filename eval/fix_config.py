"""Rewrite an export's config.json in the original Nemotron-3-Nano HF layout (hybrid_override_pattern +
remote code) so vLLM 0.12 / transformers 4.x can read it. Weights are untouched."""
import json, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
BASE_MODEL, BASE_REV = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
base_file = lambda f: hf_hub_download(BASE_MODEL, f, revision=BASE_REV)  # pinned snapshot the study used
d = Path(sys.argv[1]); new = json.load(open(d / ("config_tf5.json" if (d / "config_tf5.json").exists() else "config.json")))
assert new["layers_block_type"] and new["hidden_size"] == 1024
base = json.load(open(base_file("config.json")))
k = new["num_experts_per_tok"]
base.update(
    hidden_size=1024, hybrid_override_pattern="MEMEM*EMEMEM*EMEMEM*EMEME", num_hidden_layers=25,
    intermediate_size=1024, moe_intermediate_size=1024, moe_shared_expert_intermediate_size=512,
    mamba_num_heads=32, mamba_head_dim=64, n_groups=8, ssm_state_size=128,
    num_attention_heads=8, num_key_value_heads=2, head_dim=128,
    n_routed_experts=128, num_experts_per_tok=k, norm_topk_prob=new["norm_topk_prob"],
    max_position_embeddings=8192,
)
assert base["norm_topk_prob"] == (k > 1)
shutil.copy(d / "config.json", d / "config_tf5.json")  # transformers-5 layout, kept for reference
json.dump(base, open(d / "config.json", "w"), indent=2)  # keeps auto_map -> remote code
for f in ("configuration_nemotron_h.py", "modeling_nemotron_h.py",  # remote code (auto_map)
          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):  # original (transformers-4) tokenizer files
    shutil.copy(base_file(f), d / f)
for f in d.glob("*.safetensors"):
    f.chmod(0o644)
print("FIXED", d, "k =", k)
