# Driver: ch03-async-scheduling-tri-state worked example.
# Walks VllmConfig.__post_init__'s async_scheduling tri-state decision
# (vllm/config/vllm.py:L1052-L1143) through the faithful-subset companion
# (implementation/config_wiring.py) on five scenarios. Raw facts only.
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
import config_wiring as cw  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


def run_scenario(idx, title, kwargs):
    rec = {
        "scenario": idx,
        "title": title,
        "input": {"model": MODEL, "max_model_len": 4096, **kwargs},
    }
    args = cw.EngineArgs(model=MODEL, max_model_len=4096, **kwargs)
    rec["input_async_scheduling"] = str(kwargs.get("async_scheduling", "None(default)"))
    try:
        cfg = args.create_engine_config(cw.UsageContext.LLM_CLASS)
        executor_class = cw.Executor.get_class(cfg)
        rec["runner_type"] = cfg.model_config.runner_type
        rec["speculative_method"] = (
            cfg.speculative_config.method if cfg.speculative_config else None
        )
        rec["executor_backend"] = cfg.parallel_config.distributed_executor_backend
        rec["executor_class"] = executor_class.__name__
        rec["executor_supports_async_scheduling"] = (
            executor_class.supports_async_scheduling()
        )
        rec["final_async_scheduling"] = cfg.scheduler_config.async_scheduling
        rec["scheduler_cls_selected"] = cfg.scheduler_config.get_scheduler_cls().__name__
        rec["max_concurrent_batches"] = cfg.max_concurrent_batches
        rec["disable_nccl_for_dp_synchronization"] = (
            cfg.parallel_config.disable_nccl_for_dp_synchronization
        )
        rec["outcome"] = "ok"
    except ValueError:
        rec["outcome"] = "raised"
        rec["raised"] = traceback.format_exc(limit=1).strip().splitlines()[-1]
    return rec


scenarios = [
    (1, "None(默认) + generate + uni 执行器", {}),
    (2, "None(默认) + pooling 模型", {"runner": "pooling"}),
    (3, "None(默认) + medusa 投机解码", {"speculative_config": {"method": "medusa"}}),
    (4, "显式 True + ray 执行器(不支持 async)", {"async_scheduling": True, "distributed_executor_backend": "ray"}),
    (5, "显式 False(用户手动关)", {"async_scheduling": False}),
]

records = [run_scenario(i, t, kw) for i, t, kw in scenarios]

doc = {
    "pin": "vLLM v0.27.1 (6e448d0ea); trace source = faithful-subset companion implementation/config_wiring.py on host (platform seam: generic CUDA/H100)",
    "mechanism": "async_scheduling 三态决策 (vllm/config/vllm.py:L1052-L1143)",
    "environment_note": "host 平台 seam = current_platform 注入(generic CUDA)；执行器 supports_async_scheduling 取值与真实源码一致(abstract.py:L364 base False / uniproc_executor.py:L146 True / multiproc_executor.py:L526 True / ray 继承 base False)",
    "anchors": [
        "vllm/config/vllm.py:L1052-L1143",
        "vllm/config/vllm.py:L1056-L1057",
        "vllm/config/vllm.py:L1064-L1094",
        "vllm/config/vllm.py:L1095-L1143",
        "vllm/v1/executor/abstract.py:L364",
        "vllm/v1/executor/uniproc_executor.py:L146",
        "vllm/v1/executor/multiproc_executor.py:L526",
        "vllm/config/vllm.py:L539-L550",
        "vllm/config/scheduler.py:L170-L190",
        "vllm/config/scheduler.py:L148-L151",
    ],
    "scenarios": records,
}

out = Path(__file__).resolve().parent / "ch03-async-tri-state.json"
out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(doc, indent=1, ensure_ascii=False))
