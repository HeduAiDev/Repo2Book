# ch12 m2/m3 驱动：批队列深度仲裁矩阵 + 默认仲裁链 + EngineCore 装配落点。
# 纯 host 控制流（ScriptedExecutor 替身，不触 worker/CUDA）。
# 真源锚点：vllm/config/vllm.py:L539-L550（max_concurrent_batches）、
# L1095-L1143（None→True 默认仲裁）、config/scheduler.py:L170-L178（换型）、
# core.py:L206-L212/L231-L234（装配）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.core import EngineCore  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402
from implementation.async_scheduler import AsyncScheduler  # noqa: E402


class ScriptedExecutor:
    """Executor 替身（同 tests）：装配测试只需要它存在。"""

    def execute_model(self, scheduler_output, non_block=False):
        return None

    def sample_tokens(self, grammar_output, non_block=False):
        return None

    def take_draft_token_ids(self):
        return None


def make_cfg(**kw):
    args = dict(
        async_scheduling=None, pp_size=1, runner_type="generate", spec_method=None,
        disable_padded_drafter_batch=False, executor_backend="uniproc",
        use_v2_model_runner=False, max_model_len=64, arbitrate=True,
    )
    args.update(kw)
    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(async_scheduling=args.pop("async_scheduling")),
        pp_size=args.pop("pp_size"), runner_type=args.pop("runner_type"),
        spec_method=args.pop("spec_method"),
        disable_padded_drafter_batch=args.pop("disable_padded_drafter_batch"),
        executor_backend=args.pop("executor_backend"),
        use_v2_model_runner=args.pop("use_v2_model_runner"),
        max_model_len=args.pop("max_model_len"),
    )
    if args.pop("arbitrate"):
        cfg.check_and_set_default_async_scheduling()
    return cfg


trace = {
    "mechanism": "m2 max_concurrent_batches 深度仲裁 + m3 step_fn 三级间接装配",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor_max_concurrent_batches": "vllm/config/vllm.py:L539-L550",
    "anchor_default_arbitration": "vllm/config/vllm.py:L1095-L1143",
    "anchor_swap": "vllm/config/scheduler.py:L170-L178",
    "anchor_assembly": "vllm/v1/engine/core.py:L206-L212, L231-L234",
}

# ---------------------------------------------------------------- 段一：深度全矩阵
matrix = []
cases = [
    # (说明, kwargs, 期望注释)
    ("默认配置：async=None→仲裁 True + V1 + 单 PP", dict()),
    ("显式 async=True + V2 runner + PP=4", dict(async_scheduling=True, use_v2_model_runner=True, pp_size=4, arbitrate=False)),
    ("显式 async=True + V1 + PP=4（V1 不完全支持 async+PP → 落 pp_size）", dict(async_scheduling=True, pp_size=4, arbitrate=False)),
    ("async=False + PP=4（纯 PP 填流水线）", dict(async_scheduling=False, pp_size=4, arbitrate=False)),
    ("async=False + 单 PP（无 async 无 PP → 1，不建队列）", dict(async_scheduling=False, pp_size=1, arbitrate=False)),
]
for label, kw in cases:
    cfg = make_cfg(**kw)
    matrix.append({
        "case": label,
        "async_scheduling": cfg.scheduler_config.async_scheduling,
        "pp_size": cfg.pp_size,
        "use_v2_model_runner": cfg.use_v2_model_runner,
        "max_concurrent_batches": cfg.max_concurrent_batches,
    })
trace["depth_matrix"] = matrix
trace["depth_formula"] = (
    "async 且 V2 runner → pp_size+1；async 且 V1 且 pp<=1 → 2；其余 → pp_size"
    "（vllm.py:L539-L550 注释：PP 要 pp_size 个并发批填流水线，async 要 2 个做重叠）"
)
trace["migration_note"] = (
    "v0.21：max_concurrent_batches 是 executor 的 @cached_property（uniproc=async?2:1 /"
    " multiproc=pp<=1 and async?2:pp_size），EngineCore 经 executor 间接读；"
    "v0.27.1 唯一出处上移 VllmConfig（vllm.py:L540），EngineCore 直读 core.py:L206"
    "——v2 读者按旧代码找 executor 属性会扑空（dossier WC4 迁移陷阱）"
)

# ---------------------------------------------------------------- 段二：默认仲裁五类降级
arbitration = []


def arb_case(label, **kw):
    cfg = make_cfg(**kw)
    cls = cfg.scheduler_config.get_scheduler_cls()
    arbitration.append({
        "case": label,
        "async_scheduling_after": cfg.scheduler_config.async_scheduling,
        "scheduler_cls": cls.__name__,
    })


arb_case("生成模型 + uniproc（默认链）")
arb_case("pooling 模型（异步反而拖慢）", runner_type="pooling")
arb_case("spec 方法 medusa（非 EAGLE 系/NgramGPU/dspark）", spec_method="medusa")
arb_case("spec 方法 eagle（EAGLE 系 → 仍默认开）", spec_method="eagle")
arb_case("disable_padded_drafter_batch=True", spec_method="eagle", disable_padded_drafter_batch=True)
arb_case("executor 后端 ray（supports_async_scheduling 未覆写 → False）", executor_backend="ray")
trace["default_arbitration"] = arbitration

# 显式 True 撞不兼容 → 硬失败（不做静默降级）
try:
    cfg = make_cfg(async_scheduling=True, executor_backend="ray", arbitrate=False)
    cfg.check_and_set_default_async_scheduling()
    trace["explicit_true_hard_fail"] = "未触发（异常）"
except ValueError as e:
    trace["explicit_true_hard_fail"] = {
        "raised": "ValueError",
        "message": str(e),
        "anchor": "vllm/config/vllm.py:L1091-L1094",
    }

# ---------------------------------------------------------------- 段三：装配落点（三级间接的末端）
assembly = []
for label, kw in (
    ("默认仲裁（async=True → 深度 2 → 建队列 → 绑重叠版）", dict()),
    ("pooling 降级（async=False → 深度 1 → 不建队列 → 绑同步版）", dict(runner_type="pooling")),
):
    cfg = make_cfg(**kw)
    engine = EngineCore(cfg, model_executor=ScriptedExecutor())
    assembly.append({
        "case": label,
        "batch_queue_size": engine.batch_queue_size,
        "batch_queue_built": engine.batch_queue is not None,
        "deque_maxlen": engine.batch_queue.maxlen if engine.batch_queue is not None else None,
        "step_fn_name": engine.step_fn.__name__,
        "async_scheduling_field": engine.async_scheduling,
        "scheduler_cls": type(engine.scheduler).__name__,
    })
trace["assembly"] = assembly
trace["binding_note"] = (
    "core.py:L231-L233：step_fn 绑定只看 batch_queue 是否存在，不看 async_scheduling"
    " 字样——标志(True)→深度(2)→队列(deque maxlen=2)→step(step_with_batch_queue)"
    " 三级间接；读绑定代码看不到 async 是刻意的（dossier m3）"
)

out = os.path.join(os.path.dirname(__file__), "m2_depth.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
