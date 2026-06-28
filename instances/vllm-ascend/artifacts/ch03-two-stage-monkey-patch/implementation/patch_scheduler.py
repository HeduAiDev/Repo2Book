# 技法③：方法替换 —— vllm_ascend/patch/platform/patch_scheduler.py（subtract-only）
#
# 招式：定义一个带显式 self 形参的模块级普通函数，再赋给 Scheduler._mamba_block_aligned_split。
# 与整类替换的区别：不建子类、不动其它方法，只换这一个绑定方法。
# 台账原因：原 vLLM 该方法含 assert，外部 KV connector 命中时会失败，此版去掉了 assert。
#
# SOURCE: vllm_ascend/patch/platform/patch_scheduler.py:L1-L2
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


def _mamba_block_aligned_split(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    # SOURCE: vllm_ascend/patch/platform/patch_scheduler.py:L5-L42
    num_computed_tokens = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    # SUBTRACTED: prefill 阶段块对齐 / eagle 剪枝的多行说明注释 (patch_scheduler.py:L12-L25)。
    if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
        block_size = self.cache_config.block_size
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        # eagle prune
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
        num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
        if num_computed_tokens_after_sched < last_cache_position:
            # align to block_size
            num_new_tokens = num_new_tokens // block_size * block_size
        elif num_computed_tokens < last_cache_position < num_computed_tokens_after_sched:
            # force to cache the last chunk
            num_new_tokens = last_cache_position - num_computed_tokens
        else:
            # prefill the last few tokens
            pass
    return num_new_tokens


# 招式核心：把模块级函数挂到目标类上，替换其同名方法（绑定时 self 自然传入）。
Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
