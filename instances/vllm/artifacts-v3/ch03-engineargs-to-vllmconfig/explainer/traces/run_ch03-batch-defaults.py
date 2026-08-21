# Driver: ch03-batch-defaults worked example.
# Runs get_batch_defaults + _set_default_max_num_seqs_and_batched_tokens_args
# (vllm/engine/arg_utils.py:L2515-L2596, L2712-L2802) on injected platform
# probes (H100 / A100 / RTX 4090) across usage contexts and performance modes.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
import config_wiring as cw  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
H100 = cw.Platform(_device_name="NVIDIA H100 80GB HBM3", _total_memory=80 * cw.GiB_bytes)
A100 = cw.Platform(_device_name="NVIDIA A100-SXM4-80GB", _total_memory=80 * cw.GiB_bytes)
RTX4090 = cw.Platform(_device_name="NVIDIA GeForce RTX 4090", _total_memory=24 * cw.GiB_bytes)
DEFAULT_PLATFORM = cw.current_platform


def run_case(idx, title, platform, usage_context, kwargs):
    cw.current_platform = platform
    rec = {
        "scenario": idx,
        "title": title,
        "device_name": platform.get_device_name(),
        "device_memory_bytes": platform.get_device_total_memory(),
        "device_memory_gib": platform.get_device_total_memory() / cw.GiB_bytes,
        "usage_context": usage_context.value,
        "input_kwargs": {k: str(v) for k, v in kwargs.items()},
    }
    world_size = 1
    tok_dict, seqs_dict = cw.EngineArgs.get_batch_defaults(world_size)
    rec["branch_taken"] = (
        ">=70GiB 且非 A100 -> H100/MI300x 档"
        if rec["device_memory_bytes"] >= 70 * cw.GiB_bytes and "a100" not in platform.get_device_name().lower()
        else "else 分支(A100 特判 #17885 或 <70GiB 小卡)"
    )
    rec["table_default_max_num_batched_tokens"] = tok_dict[usage_context]
    rec["table_default_max_num_seqs"] = seqs_dict[usage_context]
    args = cw.EngineArgs(model=MODEL, **kwargs)
    assert args.max_num_batched_tokens is None and args.max_num_seqs is None, "must start as None"
    cfg = args.create_engine_config(usage_context)
    sched = cfg.scheduler_config
    mml = cfg.model_config.max_model_len
    rec["max_model_len"] = mml
    rec["enable_chunked_prefill"] = sched.enable_chunked_prefill
    rec["cap_arithmetic_seq_x_len"] = rec["table_default_max_num_seqs"] * mml
    rec["final_max_num_batched_tokens"] = sched.max_num_batched_tokens
    rec["final_max_num_seqs"] = sched.max_num_seqs
    return rec


cases = [
    (1, "H100 80GB · LLM() 离线", H100, cw.UsageContext.LLM_CLASS, {"max_model_len": 4096}),
    (2, "H100 80GB · vllm serve 在线", H100, cw.UsageContext.OPENAI_API_SERVER, {"max_model_len": 4096}),
    (3, "A100 80GB · LLM() 离线(#17885 反例)", A100, cw.UsageContext.LLM_CLASS, {"max_model_len": 4096}),
    (4, "A100 80GB · vllm serve 在线", A100, cw.UsageContext.OPENAI_API_SERVER, {"max_model_len": 4096}),
    (5, "RTX 4090 24GB · LLM() 离线(小卡走同一 else)", RTX4090, cw.UsageContext.LLM_CLASS, {"max_model_len": 4096}),
    (6, "H100 · LLM() · performance_mode=throughput(默认翻倍)", H100, cw.UsageContext.LLM_CLASS, {"max_model_len": 4096, "performance_mode": "throughput"}),
    (7, "H100 · LLM() · 关 chunked prefill(抬到 max_model_len)", H100, cw.UsageContext.LLM_CLASS, {"max_model_len": 32768, "enable_chunked_prefill": False}),
]

records = []
for idx, title, plat, uc, kw in cases:
    records.append(run_case(idx, title, plat, uc, kw))
cw.current_platform = DEFAULT_PLATFORM

doc = {
    "pin": "vLLM v0.27.1 (6e448d0ea); trace source = faithful-subset companion implementation/config_wiring.py on host",
    "mechanism": "批大小默认按显存×设备名×使用场景推导 (vllm/engine/arg_utils.py:L2515-L2596 + L2712-L2802)",
    "environment_note": "平台探针为 host seam 注入(设备名/显存字符串), 分支逻辑与常量逐字来自 arg_utils.py GPU 主线; get_batch_defaults 的 TPU/CPU 平台分支是 dossier delete 项、不在本 trace",
    "anchors": [
        "vllm/engine/arg_utils.py:L2515-L2596",
        "vllm/engine/arg_utils.py:L2712-L2802",
        "vllm/config/scheduler.py:L42-L44",
    ],
    "scenarios": records,
}

out = Path(__file__).resolve().parent / "ch03-batch-defaults.json"
out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(doc, indent=1, ensure_ascii=False))
