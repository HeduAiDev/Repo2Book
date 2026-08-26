# Driver: ch03-optimization-levels worked example.
# Applies O0/O2 presets (plus a user-explicit override and an enforce_eager
# override) through the faithful-subset companion and records the FINAL field
# values after VllmConfig.__post_init__ (vllm/config/vllm.py:L972-L1300).
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
import config_wiring as cw  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


def preset_leaf_count(preset):
    n = 0
    for v in preset.values():
        if isinstance(v, dict):
            n += preset_leaf_count(v)
        else:
            n += 1
    return n


def non_none_preset_keys(args, preset):
    """Count preset leaf keys already non-None on the freshly built EngineArgs
    (i.e. user-set BEFORE VllmConfig.__post_init__ applies the preset)."""
    def walk(obj, defaults, prefix):
        hits = []
        for key, value in defaults.items():
            if isinstance(value, dict):
                child = getattr(obj, key, None)
                if child is not None:
                    hits.extend(walk(child, value, f"{prefix}.{key}"))
            else:
                cur = getattr(obj, key, None)
                if cur is not None:
                    hits.append(f"{prefix}.{key}")
        return hits

    roots = {"compilation_config": args.compilation_config, "kernel_config": args.kernel_config}
    hits = []
    for root_name, preset_dict in preset.items():
        hits.extend(walk(roots[root_name], preset_dict, root_name))
    return hits


def run_case(idx, title, kwargs):
    args = cw.EngineArgs(model=MODEL, max_model_len=4096, **kwargs)
    preset = cw.OPTIMIZATION_LEVEL_TO_CONFIG[args.optimization_level]
    preset_keys_total = preset_leaf_count(preset)
    already_set = non_none_preset_keys(args, preset)
    rec = {
        "scenario": idx,
        "title": title,
        "optimization_level": int(args.optimization_level),
        "optimization_level_name": args.optimization_level.name,
        "preset_leaf_keys_total": preset_keys_total,
        "user_set_preset_keys_before_apply": len(already_set),
        "user_set_preset_keys": already_set,
        "preset_keys_filled_by_preset": preset_keys_total - len(already_set),
    }
    cfg = args.create_engine_config(cw.UsageContext.LLM_CLASS)
    cc = cfg.compilation_config
    rec.update(
        {
            "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
            "world_size": cfg.parallel_config.world_size,
            "executor_backend": cfg.parallel_config.distributed_executor_backend,
            "mode": int(cc.mode),
            "mode_name": cc.mode.name,
            "cudagraph_mode": int(cc.cudagraph_mode),
            "cudagraph_mode_name": cc.cudagraph_mode.name,
            "use_inductor_graph_partition": cc.use_inductor_graph_partition,
            "ir_enable_torch_wrap": cc.ir_enable_torch_wrap,
            "custom_ops": list(cc.custom_ops),
            "kernel_enable_flashinfer_autotune": cfg.kernel_config.enable_flashinfer_autotune,
            "fuse_norm_quant": cc.pass_config.fuse_norm_quant,
            "fuse_act_quant": cc.pass_config.fuse_act_quant,
            "fuse_allreduce_rms": cc.pass_config.fuse_allreduce_rms,
            "fuse_attn_quant": cc.pass_config.fuse_attn_quant,
            "enable_sp": cc.pass_config.enable_sp,
            "max_cudagraph_capture_size": cc.max_cudagraph_capture_size,
            "cudagraph_capture_sizes": list(cc.cudagraph_capture_sizes or []),
            "cudagraph_num_of_warmups": cc.cudagraph_num_of_warmups,
            "enforce_eager": cfg.model_config.enforce_eager,
            "async_scheduling": cfg.scheduler_config.async_scheduling,
        }
    )
    return rec


cases = [
    (1, "O0 默认(纯 eager 立即启动)", {"optimization_level": cw.OptimizationLevel.O0}),
    (2, "O2 默认 TP=1(出厂默认档)", {"optimization_level": cw.OptimizationLevel.O2}),
    (3, "O2 TP=2(谓词按配置求值)", {"optimization_level": cw.OptimizationLevel.O2, "tensor_parallel_size": 2, "distributed_executor_backend": "mp"}),
    (4, "O2 + 用户显式 cudagraph_mode=PIECEWISE", {"optimization_level": cw.OptimizationLevel.O2, "compilation_config": {"cudagraph_mode": cw.CUDAGraphMode.PIECEWISE}}),
    (5, "O2 + enforce_eager=True(用户旗标压过预设)", {"optimization_level": cw.OptimizationLevel.O2, "enforce_eager": True}),
]

records = [run_case(i, t, kw) for i, t, kw in cases]

doc = {
    "pin": "vLLM v0.27.1 (6e448d0ea); trace source = faithful-subset companion implementation/config_wiring.py on host",
    "mechanism": "O0-O3 优化级: 预设字典 + 谓词默认值 + 递归只填 None (vllm/config/vllm.py:L104-L327, L811-L853, L1272-L1300)",
    "environment_note": "host 平台 seam: 谓词函数体降为通用 CUDA 路径默认值(与真实通用路径 custom_ops=none 时的求值一致); fuse_allreduce_rms 真实谓词还门控 Hopper/Blackwell+flashinfer 探测(vllm.py:L155-L175), host seam 只保留 TP>1 前置——绝对值 True/False 在 TP>1 且无 flashinfer 的真机上可能为 False, writer 需就近挑明",
    "anchors": [
        "vllm/config/vllm.py:L104-L116",
        "vllm/config/vllm.py:L229-L251",
        "vllm/config/vllm.py:L275-L297",
        "vllm/config/vllm.py:L322-L327",
        "vllm/config/vllm.py:L811-L853",
        "vllm/config/vllm.py:L1193-L1234",
        "vllm/config/vllm.py:L1272-L1300",
        "vllm/config/compilation.py:L447",
        "vllm/config/compilation.py:L607",
    ],
    "scenarios": records,
}

out = Path(__file__).resolve().parent / "ch03-optimization-levels.json"
out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
print(json.dumps(doc, indent=1, ensure_ascii=False))
