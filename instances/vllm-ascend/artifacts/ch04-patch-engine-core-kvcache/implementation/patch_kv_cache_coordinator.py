# 案例3 — CP + hybrid 前缀缓存：去掉 dcp==1/pcp==1 断言，引入 _get_effective_block_size 解锁。
# 技法①整类替换（AscendHybridKVCacheCoordinator）+ 技法②/③工厂替换（get_kv_cache_coordinator）
# + 技法⑤from-import 缓存陷阱修复（kv_cache_manager 的早绑引用补绑）。
# subtract-only：与 vllm_ascend/patch/platform/patch_kv_cache_coordinator.py 同名/同结构/同控制流。
#
# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L1-L32
import sys
from collections.abc import Mapping
from math import lcm

import vllm
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
)

from vllm_ascend.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec

USE_MULTI_GROUPS_KV_CACHE = True

_orig_get_kv_cache_coordinator = vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L35-L48
def _is_deepseek_v4_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> bool:
    if getattr(kv_cache_spec, "model_version", None) == "deepseek_v4":
        return True

    nested_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
    if nested_specs is None:
        return False

    if isinstance(nested_specs, Mapping):
        nested_specs = nested_specs.values()
    elif not isinstance(nested_specs, (list, tuple, set)):
        return False

    return any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in nested_specs)


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L51-L52
def _is_deepseek_v4_kv_cache_config(kv_cache_config: KVCacheConfig) -> bool:
    return any(_is_deepseek_v4_kv_cache_spec(group.kv_cache_spec) for group in kv_cache_config.kv_cache_groups)


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L55-L297
class AscendHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        eagle_attn_layer_names: list[str] | None = None,
        metrics_collector: KVCacheMetricsCollector | None = None,
        max_num_batched_tokens: int | None = None,
    ):
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L64-L130
        # 对照基座 HybridKVCacheCoordinator.__init__ 在此处直接
        # `assert dcp_world_size == 1` / `assert pcp_world_size == 1`（hybrid 不许开 CP）。
        # Ascend 子类去掉这两条断言，改为保存 dcp/pcp 并在 _get_effective_block_size 里按 CP 缩放。
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # Fall back to `max_model_len` when unset so the recycling-aware
        # admission cap (vLLM PR #40946) collapses to the prior uncapped
        # behavior. The scheduler always supplies the real value at runtime.
        if max_num_batched_tokens is None:
            max_num_batched_tokens = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens

        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group}
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        if enable_caching:
            assert all(
                self._get_effective_block_size(g.kv_cache_spec) % hash_block_size == 0
                for g in kv_cache_config.kv_cache_groups
            ), "block_size must be divisible by hash_block_size"
        self.verify_and_split_kv_cache_groups()

        self.use_eagle = use_eagle

    def _get_effective_block_size(self, kv_cache_spec: KVCacheSpec) -> int:
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L132-L142
        block_size = kv_cache_spec.block_size
        if isinstance(kv_cache_spec, MambaSpec) and self.enable_caching:
            return block_size
        if self.dcp_world_size * self.pcp_world_size > 1:
            block_size *= self.dcp_world_size * self.pcp_world_size
        if hasattr(kv_cache_spec, "compress_ratio"):
            compress_ratio = kv_cache_spec.compress_ratio or 1
            compress_ratio = compress_ratio if compress_ratio >= 1 else 1
            block_size *= compress_ratio
        return block_size

    def verify_and_split_kv_cache_groups(self) -> None:
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L144-L188
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        attention_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, "Expected same manager class for identical KV cache specs."
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, "HybridKVCacheCoordinator requires at least two attention groups."

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # Attention-group indices (into ``self.attention_groups``) that
        # contain at least one EAGLE/MTP KV cache group.
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(gid in self.eagle_group_ids for gid in group_ids)
        }

        # The LCM of the block sizes of all attention types.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. Requiring this because we don't support partial
        # block cache hit yet.
        # NOTE: 这里 block_sizes 用 _get_effective_block_size（带 CP×compress 缩放），
        # 而非裸 spec.block_size——这是解锁 CP+hybrid 的关键替换点。
        block_sizes = [self._get_effective_block_size(spec) for spec, _, _ in self.attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L190-L297
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            # SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L213-L219
            target_block_size = kv_cache_spec.block_size
            if not isinstance(kv_cache_spec, MambaSpec) and self.dcp_world_size * self.pcp_world_size > 1:
                target_block_size *= self.dcp_world_size * self.pcp_world_size
            if target_block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(block_hashes, self.hash_block_size, target_block_size)

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # SUBTRACTED: 不动点迭代的完整扫描实现（patch_kv_cache_coordinator.py:L225-L281）——
        # eagle 单次验证 / is_simple_hybrid 提前退出 / 逐组调 manager_cls.find_longest_cache_hit /
        # 长度单调收敛。与对照基座 HybridKVCacheCoordinator.find_longest_cache_hit 同构，非 patch 重点。
        # 本章差异仅在每组用 effective_block_size = self._get_effective_block_size(spec) 替换裸
        # spec.block_size 来折算命中长度（下方截断同此替换）。
        for spec, group_ids, manager_cls in self.attention_groups:
            effective_block_size = self._get_effective_block_size(spec)  # ← 有效块替换点
            num_blocks = hit_length // effective_block_size
            hit_length = num_blocks * effective_block_size

        # Truncate full attention blocks to final hit_length (if present).
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // self._get_effective_block_size(spec)
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L300-L357
def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    eagle_attn_layer_names: list[str] | None = None,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if _is_deepseek_v4_kv_cache_config(kv_cache_config):
        return AscendHybridKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            eagle_attn_layer_names=eagle_attn_layer_names,
            metrics_collector=metrics_collector,
            max_num_batched_tokens=max_num_batched_tokens,
        )

    cp_enabled = dcp_world_size > 1 or pcp_world_size > 1

    # Only CP hybrid prefix caching needs AscendHybridKVCacheCoordinator.
    # Otherwise keep upstream coordinators (non-CP / unitary / no-prefix-cache).
    if not cp_enabled or len(kv_cache_config.kv_cache_groups) == 1 or not enable_caching:
        return _orig_get_kv_cache_coordinator(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            hash_block_size,
            metrics_collector,
        )
    return AscendHybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
        max_num_batched_tokens=max_num_batched_tokens,
    )


# SOURCE: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:L360-L368
vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]

# `kv_cache_manager` imports `get_kv_cache_coordinator` with
# `from ... import ...`, so if it was loaded before this patch runs
# (for example through the recompute scheduler path), it keeps the
# old function object. Update that cached binding as well.  ← 技法⑤from-import 缓存陷阱修复
_kv_cache_manager = sys.modules.get("vllm.v1.core.kv_cache_manager")
if _kv_cache_manager is not None:
    _kv_cache_manager.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]
