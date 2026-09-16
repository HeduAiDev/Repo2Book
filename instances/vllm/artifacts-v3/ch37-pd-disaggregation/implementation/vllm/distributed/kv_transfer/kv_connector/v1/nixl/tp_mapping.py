# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP 映射：本地 rank 决定「向远端哪些 rank 收发、各取哪一段」。

对称 TP（P/D 同 tp_size）下退化成「每个 rank 只跟对应的一个远端 rank 打交道」，
块号一一对应——这是本章的正典。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L1-L142
# SUBTRACTED: 异构 TP（tp_size != remote_tp_size）的分支——减法计划删除项 1：
#   * P_TP > D_TP 的多 rank 读（`abs_tp = remote_tp_size // tp_size` 与 np.unique
#     的 GQA 去重）；
#   * rank_offset_factor 的按头切分偏移；
#   * _build_local_splits_from_plan（base_worker）与 tp_ratio<0 的多读循环。
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    TransferTopology,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

# ======================================================================
# Data structures
# ======================================================================


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L22-L28
@dataclass(frozen=True)
class ReadSpec:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L22-L28
    """Specification for a single remote block read operation."""

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L31-L36
def _is_attention_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, AttentionSpec)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L35-L36
def _is_ssm_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, MambaSpec)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L39-L57
@dataclass(frozen=True)
class TPMapping:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L39-L57
    """Complete local-to-remote TP mapping for one remote engine.

    Generated once per remote engine during handshake.
    """

    # Remote TP ranks that this local rank reads from, per group.
    # Position = local piece index.
    source_ranks_per_group: tuple[tuple[int, ...], ...]

    # Superset of all source ranks (union of all groups).
    all_source_ranks: tuple[int, ...]

    # Maps each source rank to its FA head slot index.
    rank_to_attention_slot: dict[int, int]

    # FA head offset factor for hetero-TP (D_TP > P_TP).
    rank_offset_factor: int


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L65-L142
# SUBTRACTED: 异构 TP 的两条分支（见模块头）——对称 TP 下 attn_ranks 恒为单元素。
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L65-L142
def compute_tp_mapping(
    transfer_topology: TransferTopology,
    remote_tp_size: int,
    group_spec_types: tuple[type[KVCacheSpec], ...],
) -> TPMapping:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:L65-L142
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.
    """
    tp_rank = transfer_topology.tp_rank
    total_num_kv_heads = transfer_topology.total_num_kv_heads
    # --- Attention source ranks ---
    # D (local TP) >= P (remote TP): local rank reads from the matching remote
    # rank. With MLA the cache is duplicated, so the same expression spreads
    # mla ranks onto remote k*tp_rank.
    attn_ranks = [tp_rank * remote_tp_size // transfer_topology.tp_size]

    # --- SSM source ranks ---
    # SUBTRACTED: SSM 状态槽的多 rank 分片（has_ssm 分支）——混合模型归 ch14/ch16。
    ssm_ranks: list[int] = []

    all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))

    # --- Per-group ordered source ranks ---
    source_ranks_per_group = tuple(
        tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
        for t in group_spec_types
    )

    # --- Attention head slots ---
    head_to_slot: dict[int, int] = {}
    for i, r in enumerate(attn_ranks):
        head_to_slot[r * total_num_kv_heads // remote_tp_size] = i
    rank_to_attention_slot = {
        r: head_to_slot.get(r * total_num_kv_heads // remote_tp_size, 0)
        for r in all_ranks
    }

    # --- Rank offset factor ---
    # SUBTRACTED: D_TP > P_TP 的按头偏移（tp_size > total_num_kv_heads 与
    #   tp_rank % (tp_size // remote_tp_size) 两条）——对称 TP 恒 0。
    rank_offset_factor = 0

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
    )


__all__ = ["ReadSpec", "TPMapping", "compute_tp_mapping", "_is_attention_spec", "_is_ssm_spec"]
