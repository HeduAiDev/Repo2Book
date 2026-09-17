# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derivation of canonical page mappings for KV offloading.

The only place in the offloading stack that reasons about parallelism
(TP/DCP/PCP); everything downstream consumes byte mappings. The canonical
page of a layer is the full offloaded block without parallelism: all KV
heads, all block_size * dcp * pcp tokens, in the worker's page encoding.
Uncertifiable layers get an opaque fallback mapping (fail closed).
"""

from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import CanonicalPageMapping

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# SUBTRACTED: 并行复制页去重机器——_RankContext / ByteRegion / _coalesce_runs /
#   _local_to_canonical_token / _interleave_cp_tokens / _packed_kv_regions /
#   _split_kv_regions / _attention_byte_regions / _layer_mapping /
#   _opaque_fallback_mapping / _run_intervals / _is_exact_partition /
#   _verify_tiling / _unpadded_page_size（L31-L444 全部推导体）——
#   减法计划删除项 1：TP 复制页（MLA latent 每 rank 同字节）的写放大优化，
#   默认 replicated_layout=False、mapping=None 直通；教学主线=每 rank 各存自己
#   分片。derive_canonical_mappings 的**调用点**保留（worker.register_kv_caches），
#   入口恒返 {}：CanonicalKVCacheRef.mapping=None = uncertified（docstring 原语义）。


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/canonical_mapping.py:L387-L444
def derive_canonical_mappings(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    kv_caches: dict,
) -> dict[str, CanonicalPageMapping]:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:L387-L444
    """Per-layer canonical page mappings for this worker.

    Empty when the worker group is not exactly the TP x PCP grid; layers
    absent from the result have no canonical representation.
    """
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    group_size = tp_size * pcp_size
    if parallel_config.world_size != group_size:
        return {}

    # SUBTRACTED: 逐 rank 的字节区间推导与 tiling 校验循环（L417-L443）——
    #   减法计划删除项 1：本精简版恒 uncertified（空映射直通）。
    return {}
