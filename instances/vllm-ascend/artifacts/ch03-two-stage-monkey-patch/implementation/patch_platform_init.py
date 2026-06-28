# platform 段清单 —— vllm_ascend/patch/platform/__init__.py（subtract-only 精简版）
#
# 这是一个「裸 import 列表」：每条 import 一执行，就触发对应模块的模块级副作用 = 一次 patch。
# adapt_patch(is_global_patch=True) 只 import 本包，本 __init__ 再级联 import 各 patch_*。
#
# SOURCE: vllm_ascend/patch/platform/__init__.py:L17-L50
import os

import vllm_ascend.patch.platform.patch_camem_allocator  # noqa
import vllm_ascend.patch.platform.patch_distributed  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa
import vllm_ascend.patch.platform.patch_kv_cache_utils  # noqa
import vllm_ascend.patch.platform.patch_mla_prefill_backend  # noqa
from vllm_ascend.utils import is_310p

# 条件加载骨架①：按 SoC 二选一加载 mamba_config（310P 走专用实现）。
if not is_310p():
    import vllm_ascend.patch.platform.patch_mamba_config  # noqa
else:
    import vllm_ascend.patch.platform.patch_mamba_config_310  # noqa

# SUBTRACTED: 中段一长串模型/协议特化 patch（minimax_* / glm* / deepseek_v4_* /
#   anthropic / tool_call_parser / torch_accelerator 等约 13 条裸 import）已折叠——
#   它们都是同构「import 触发副作用」，逐条保留不增信息 (platform/__init__.py:L33-L48)。
import vllm_ascend.patch.platform.patch_mamba_manager  # noqa

# 条件加载骨架②：环境变量门控 —— 整类替换样本 patch_multiproc_executor 默认不加载，
# 仅在 EPLB / EXPERT_MAP_RECORD 场景才被 import。
if os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1") or os.getenv("EXPERT_MAP_RECORD", "false") == "true":
    import vllm_ascend.patch.platform.patch_multiproc_executor  # noqa

import vllm_ascend.patch.platform.patch_scheduler  # noqa
