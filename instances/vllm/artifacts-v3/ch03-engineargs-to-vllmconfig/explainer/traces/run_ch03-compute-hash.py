# Driver: ch03-compute-hash worked example.
# Builds five configs that differ in exactly one hash-relevant/irrelevant knob
# each (vllm/config/vllm.py:L431-L537; scheduler.py:L193-L219; parallel.py:
# L774-L829) and records total + per-subconfig hashes through the
# faithful-subset companion. Also checks determinism (fresh rebuild).
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
import config_wiring as cw  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


def build(kwargs):
    args = cw.EngineArgs(model=MODEL, max_model_len=4096, **kwargs)
    return args.create_engine_config(cw.UsageContext.LLM_CLASS)


def summarize(idx, title, change_note, kwargs):
    cfg = build(kwargs)
    return {
        "scenario": idx,
        "title": title,
        "change_note": change_note,
        "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
        "world_size": cfg.parallel_config.world_size,
        "distributed_executor_backend": cfg.parallel_config.distributed_executor_backend,
        "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
        "max_num_seqs": cfg.scheduler_config.max_num_seqs,
        "scheduler_hash": cfg.scheduler_config.compute_hash(),
        "parallel_hash": cfg.parallel_config.compute_hash(),
        "cache_hash": cfg.cache_config.compute_hash(),
        "model_hash": cfg.model_config.compute_hash(),
        "compilation_hash": cfg.compilation_config.compute_hash(),
        "kernel_hash": cfg.kernel_config.compute_hash(),
        "total_hash": cfg.compute_hash(),
        "total_hash_len": len(cfg.compute_hash()),
    }


BASE = {"max_num_batched_tokens": 16384, "max_num_seqs": 1024}
records = [
    summarize(1, "基线: TP=1, backend=None(->uni), tokens=16384, seqs=1024",
              "参照点", dict(BASE)),
    summarize(2, "改 max_num_seqs 1024->512",
              "SchedulerConfig.compute_hash 只收 max_num_batched_tokens (scheduler.py:L193-L219, #29585)",
              dict(BASE, max_num_seqs=512)),
    summarize(3, "改 backend: None(->uni) -> 显式 mp (world_size 仍 1)",
              "ParallelConfig.compute_hash 的 ignored_factors 含 distributed_executor_backend (parallel.py:L774-L829)",
              dict(BASE, distributed_executor_backend="mp")),
    summarize(4, "改 TP 1->2 (在场景3 的 mp 配置上, 单变量隔离)",
              "tensor_parallel_size 进 ParallelConfig 因子: 集体通信进计算图; 且 TP=2 使 O2 预设谓词 fuse_allreduce_rms (vllm.py:L155-L175) 翻 True -> pass_config 入 CompilationConfig.compute_hash (compilation.py:L780, default-include 声明字段) -> compilation 子 hash 连带变(单变量输入的派生涟漪)",
              dict(BASE, distributed_executor_backend="mp", tensor_parallel_size=2)),
    summarize(5, "改 max_num_batched_tokens 16384->8192",
              "LoRA 静态缓冲尺寸 + Inductor 32/64 位索引选择 (#29585)",
              dict(BASE, max_num_batched_tokens=8192)),
]

# Determinism: rebuild the base config from scratch (fresh VllmConfig with a
# different instance_id) and recompute.
cfg_again = build(dict(BASE))
records.append({
    "scenario": 6,
    "title": "基线重建(全新 VllmConfig 实例)再算一遍",
    "change_note": "确定性检查: 同一配置 -> 同一指纹",
    "total_hash": cfg_again.compute_hash(),
    "total_hash_len": len(cfg_again.compute_hash()),
    "deterministic_same_as_scenario_1": cfg_again.compute_hash() == records[0]["total_hash"],
})

doc = {
    "pin": "vLLM v0.27.1 (6e448d0ea); trace source = faithful-subset companion implementation/config_wiring.py on host",
    "mechanism": "compute_hash: 计算图 10 位指纹 (vllm/config/vllm.py:L431-L537)",
    "environment_note": "已知偏差: 精简版 hash_factors 用 default=str 序列化、因子集为子集、__version__ 用 '0.27.1' 常量替身——绝对哈希值与真实 vLLM 不逐位对齐; 教学只消费『哪些因子入哈希/哪些被 ignore』的作用域语义(变/不变结论与真实源码一致, 由 ignored_factors 集合逐字保留保证)",
    "anchors": [
        "vllm/config/vllm.py:L431-L537",
        "vllm/config/scheduler.py:L193-L219",
        "vllm/config/parallel.py:L774-L829",
        "vllm/compilation/backends.py:L1034",
    ],
    "scenarios": records,
}

out = Path(__file__).resolve().parent / "ch03-compute-hash.json"
out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
print(json.dumps(doc, indent=1, ensure_ascii=False))
