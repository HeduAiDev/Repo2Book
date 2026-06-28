# 案例3 配套 — resolve_kv_cache_block_sizes：把 PR#40860 的「多组+CP 直接 raise」换成真实 lcm/gcd 计算。
# 技法③方法替换 + 技法⑤from-import 缓存陷阱（kv_cache_utils + engine.core 两处重绑）。
# subtract-only：与 vllm_ascend/patch/platform/patch_kv_cache_utils.py 同名/同结构/同控制流。
#
# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_utils.py:L1-L20
import math
from collections import defaultdict

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_utils.py:L23-L58
def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support context parallelism (dcp/pcp > 1).
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp * pcp to maintain alignment, consistent
    with the pre-PR-#40860 behavior of block_size * dcp * pcp.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        # 对照基座此处是 `raise ValueError(...)` 禁止多组+CP；Ascend 改为真实计算。
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_utils.py:L61-L92
def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    # SUBTRACTED: DeepseekV4 专用 KV cache tensor 分桶/对齐布局规划的完整函数体
    # (patch_kv_cache_utils.py:L68-L92)。动机：昇腾要求所有 cache tensor 连续，故按
    # compress_ratio / SlidingWindowMLASpec 分桶后统一成 UniformTypeKVCacheSpecs。
    # 与本章 5 个主线案例正交，逐行展开会喧宾夺主——保留函数名 + 动机即可。
    raise NotImplementedError  # 占位：真实实现见源码，本章不逐行展开


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_utils.py:L95-L184
def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    # SUBTRACTED: DeepseekV4 把 grouped specs 切成对齐 layer-tuple 的 KVCacheGroupSpec 的
    # 完整实现 (patch_kv_cache_utils.py:L101-L184)——按 page_size pad 对齐、_approximate_gcd /
    # round_up 算 layer tuple 数。同案例正交，仅点名存在与动机（连续布局）。
    raise NotImplementedError  # 占位：真实实现见源码，本章不逐行展开


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_utils.py:L187-L247
def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """DeepseekV4 KV cache tensor layout planning."""
    # SUBTRACTED: 按 page_size 分桶、为每个 (tuple_idx, bucket) 发一条 KVCacheTensor 的完整布局
    # 实现 (patch_kv_cache_utils.py:L204-L247)，含 may_override_num_blocks。单一模型布局细节，
    # 与本章主线正交，保留签名 + 动机即可。
    raise NotImplementedError  # 占位：真实实现见源码，本章不逐行展开


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_utils.py:L250-L258
# 技法③替换 resolve_kv_cache_block_sizes（+ 三个 DeepseekV4 布局函数），并因 engine.core
# `from ... import resolve_kv_cache_block_sizes` 直接引用，额外补绑（技法⑤from-import 缓存陷阱）。
vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
