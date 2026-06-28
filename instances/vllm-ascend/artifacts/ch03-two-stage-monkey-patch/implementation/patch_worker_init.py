# worker 段清单 —— vllm_ascend/patch/worker/__init__.py（subtract-only 精简版）
#
# adapt_patch()（默认 is_global_patch=False）import 本包，本 __init__ 再按多种条件级联 import。
#
# SOURCE: vllm_ascend/patch/worker/__init__.py:L18-L70
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.utils import is_310p, vllm_version_is

# 条件加载骨架①（版本门控）：v2 model runner patch 依赖 v0.21.0 之后的上游 main API。
# 本书 pin 的基座正是 v0.21.0 → _V2_MODEL_RUNNER_SUPPORTED 为 False → patch_v2 整组与
# patch_routed_experts_capture 全部不加载。
# v2 model runner patches depend on upstream main APIs beyond v0.21.0.
_V2_MODEL_RUNNER_SUPPORTED = not vllm_version_is("0.21.0")

# 条件加载骨架②（能力门控）：仅当存在 Triton 后端才加载 triton patch。
if HAS_TRITON:
    import vllm_ascend.patch.worker.patch_triton

    if _V2_MODEL_RUNNER_SUPPORTED:
        import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa


import vllm_ascend.patch.worker.patch_weight_utils  # noqa
import vllm_ascend.patch.worker.patch_distributed  # noqa
# SUBTRACTED: 一系列无条件 worker patch（minimax_m2 / mamba_utils / qwen3_next_mtp /
#   rejection_sampler / kimi_k25 / draft_quarot / cudagraph / deepseek_mtp / gqa_c8 等）
#   已折叠——同构「import 触发副作用」(worker/__init__.py:L33-L59)。

# 条件加载骨架③（SoC 门控）：按 310P 与否加载不同的模型算子 patch。
if not is_310p():
    import vllm_ascend.patch.worker.patch_qwen3_5  # noqa
    import vllm_ascend.patch.worker.patch_gdn_attn  # noqa
    import vllm_ascend.patch.worker.patch_qwen3_dflash  # noqa
    import vllm_ascend.patch.worker.patch_qwen3vl  # noqa
else:
    import vllm_ascend.patch.worker.patch_idex_310  # noqa

# 条件加载骨架④（可选依赖门控）：torchair/npugraph_ex 仅 NPU 可用，CPU-only 环境静默跳过。
try:  # noqa: SIM105
    import vllm_ascend.patch.worker.patch_npugraph_ex_triton  # noqa
except ImportError:
    pass

# 版本门控的回响：patch_v2 整组在 v0.21.0 上全部跳过。
if _V2_MODEL_RUNNER_SUPPORTED:
    import vllm_ascend.patch.worker.patch_v2.patch_uva  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_input_batch  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_model_state  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_block_table  # noqa
    import vllm_ascend.patch.worker.patch_v2.patch_attn_utils  # noqa

# only patch routed experts capture in main2main.
if _V2_MODEL_RUNNER_SUPPORTED:
    import vllm_ascend.patch.worker.patch_routed_experts_capture  # noqa
