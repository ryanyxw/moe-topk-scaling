"""Write a nemo-evaluator-launcher config for one served model.
Usage: make_cfg.py <name> <port> <task key>[,<task key>...] [limit]
Task keys are the lm-eval result keys in TASKS (also what collect.py reads)."""
import os, sys, yaml

E = os.environ.get("EVAL_ROOT", "/mnt/nvme/scaling/eval")  # exports, configs, NEL outputs
# result key -> (NEL task name, lm-eval task override)
TASKS = {
    "hellaswag": ("hellaswag", None),                                  # 10-shot (container default)
    "adlr_arc_challenge_llama": ("adlr_arc_challenge_llama_25_shot", None),  # ARC-C 25-shot, letter answers
    "arc_challenge": ("arc_challenge", None),                          # ARC-C 0-shot, cloze
    "arc_easy": ("arc_challenge", "arc_easy"),                         # ARC-E 0-shot
    "piqa": ("piqa", None),
    "winogrande": ("adlr_winogrande_5_shot", None),
    "openbookqa": ("openbookqa", None),
    "sciq": ("arc_challenge", "sciq"),
    "lambada_openai": ("arc_challenge", "lambada_openai"),
    "mmlu": ("mmlu_logits", None),                                     # MMLU 5-shot, log-likelihood
}

if __name__ == "__main__":
    name, port, keys = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    tasks = []
    for k in keys:
        nm, override = TASKS[k]
        c = {"name": nm}
        if override:
            c["nemo_evaluator_config"] = {"config": {"params": {"task": override}}}
        tasks.append(c)
    cfg = {
        "defaults": [{"execution": "local"}, {"deployment": "none"}, "_self_"],
        "execution": {
            "output_dir": f"{E}/results/{name}", "mode": "sequential",
            # Own network namespace per container (the adapter proxy picks ports that collide under
            # --network=host); vLLM is reached through the docker bridge. Tokenizer and datasets are local.
            "extra_docker_args": f"-v {E}/tokenizer:/tokenizer:ro -v {E}/hf_cache:/hf_cache -e HF_HOME=/hf_cache",
        },
        "target": {"api_endpoint": {"url": f"http://172.17.0.1:{port}/v1/completions", "model_id": name, "type": "completions"}},
        "evaluation": {
            "nemo_evaluator_config": {"config": {"params": {
                "parallelism": 32, "request_timeout": 600, "max_retries": 5,
                **({"limit_samples": limit} if limit else {}),
                "extra": {"tokenizer": "/tokenizer", "tokenizer_backend": "huggingface", "args": "--batch_size 32"}}}},
            "tasks": tasks,
        },
    }
    yaml.safe_dump(cfg, open(f"{E}/cfg/{name}.yaml", "w"), sort_keys=False)
    print(len(tasks))
