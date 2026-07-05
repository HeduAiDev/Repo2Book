"""Driver: compute_hash worked example for ch03 (retrofit "数值4").

Builds a baseline VllmConfig and three one-field variants, prints each
sub-config factor hash + the final 10-char graph fingerprint, so the chapter
can show *which* config changes trigger a torch.compile recompile and which
do not.

Run:  python3 run_compute_hash.py
Emits machine-readable JSON to stdout (captured into trace_compute_hash.json).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.normpath(os.path.join(HERE, "..", "..", "implementation"))
sys.path.insert(0, IMPL)

import config_wiring as cw  # noqa: E402

# Simulate a 2-GPU box so the TP=2 variant assembles (backend -> "mp") instead
# of raising "world size > available GPUs". Baseline TP=1 is unaffected.
cw.current_platform._device_count = 2


def factor_breakdown(vc):
    """Recompute the per-sub-config factor hashes exactly as VllmConfig.
    compute_hash collects them (vllm/config/vllm.py L367-L473), so the table
    can attribute a fingerprint change to a specific factor."""
    mc = vc.model_config
    return {
        "version": "0.15.1",
        "model": mc.compute_hash() if mc else "None",
        "cache": vc.cache_config.compute_hash() if vc.cache_config else "None",
        "parallel": vc.parallel_config.compute_hash() if vc.parallel_config else "None",
        "scheduler": vc.scheduler_config.compute_hash() if vc.scheduler_config else "None",
        "compilation": vc.compilation_config.compute_hash() if vc.compilation_config else "None",
        "kernel": vc.kernel_config.compute_hash() if vc.kernel_config else None,
    }


def build(**overrides):
    args = cw.EngineArgs(**overrides)
    return args.create_engine_config()


scenarios = [
    ("baseline", "TP=1, max_num_seqs=256, O2", {}),
    ("variantA_tp", "TP 1 -> 2 (changes parallel factor)", {"tensor_parallel_size": 2}),
    ("variantB_seqs", "max_num_seqs 256 -> 512 (scheduler factor unchanged)", {"max_num_seqs": 512}),
    ("variantC_optlevel", "optimization_level O2 -> O0 (changes compilation/kernel factor)",
     {"optimization_level": cw.OptimizationLevel.O0}),
]

results = []
baseline_fp = None
for key, desc, ov in scenarios:
    vc = build(**ov)
    fp = vc.compute_hash()
    if key == "baseline":
        baseline_fp = fp
    fb = factor_breakdown(vc)
    results.append({
        "scenario": key,
        "description": desc,
        "override": {k: (v.name if isinstance(v, cw.OptimizationLevel) else v)
                     for k, v in ov.items()},
        "factors": fb,
        "fingerprint": fp,
        "same_as_baseline": (fp == baseline_fp),
    })

print(json.dumps({"results": results, "baseline_fingerprint": baseline_fp},
                 indent=2, ensure_ascii=False))
