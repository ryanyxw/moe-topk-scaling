#!/usr/bin/env python3
"""Emit queue lines for the scaling sweep. Per-top-k memory/throughput settings measured on 2xH100."""
import sys

def kargs(k):
    if k <= 4:
        return "--ep 2 --mbs 4"
    if k == 8:
        return "--ep 2 --mbs 4 --recompute moe_act"
    return "--ep 2 --mbs 2 --recompute moe_act"

COMMON = "--grouped-gemm cutlass --eval-iters 20 --eval-interval 1000 --log-interval 10"

def job(name, k, lr, flops, slot="pair", extra=""):
    return f"{name} {slot} --topk {k} --lr {lr} --flops {flops} {kargs(k) if slot == 'pair' else '--ep 1 --mbs 1'} {COMMON} {extra}".strip()

if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "lr":  # LR sweep at C0 = 2e18 (quarter of C1); no checkpoints needed
        for k in (4, 1, 16):
            for lr in ("1.5e-3", "3e-3", "6e-3"):
                print(job(f"c0_k{k}_lr{lr}", k, lr, "2e18", extra="--no-save"))
        # Noise floor: seed repeat of k4/lr3e-3 on the otherwise idle single GPU 7.
        print(job("c0_k4_lr3e-3_seed2", 4, "3e-3", "2e18", slot="single", extra="--no-save --seed 2"))
