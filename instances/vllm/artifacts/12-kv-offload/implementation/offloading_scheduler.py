# SPDX-License-Identifier: Apache-2.0
"""
OffloadingConnectorScheduler — scheduler-side reactive prefix lookup.

REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py (881 LOC)

This module mirrors the load-bearing slices of the production scheduler:
  * SchedulerOffloadConfig / GroupOffloadConfig — derived block-size facts
  * RequestOffloadState / RequestGroupState — per-req tracking
  * `_maximal_prefix_lookup` / `_sliding_window_lookup` — REACTIVE block-hash
    matching (this is the key Trap-D reframe)
  * `get_num_new_matched_tokens` — public API consumed by KVConnectorBase_V1

Outline reframe (Trap E variant):
  Outline §12.3 says "Prefetch — predict which KV blocks will be used,
  pre-fetch back to GPU". This is misleading. vLLM at 98661fe does NOT
  do PREDICTIVE / ML-based prefetch. It does REACTIVE block-hash matching:
  when a request arrives, the scheduler hashes its prompt prefix against
  the offload manager's keyspace and asks "which blocks of this prefix
  are already offloaded?". That is a DETERMINISTIC PREFIX LOOKUP, not a
  prediction.

  Genuine predictive prefetch would require a model of which prefixes
  WILL hit (e.g. a Markov chain of past requests, or an ML predictor).
  No such code exists in `vllm/v1/kv_offload/` or any kv_connector at
  this commit. The "prefetch" framing in the outline is a PLATFORM
  misnomer that the chapter must correct. See impl-notes O07.

Sliding-window lookup is not "prediction" either: it scans block hashes
from the END of the request and counts CONSECUTIVE hits. The "window"
refers to the ATTENTION sliding window, not a predictive lookback.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Optional

from .offload_manager import OffloadingManager
from .offload_spec import (
    OffloadKey,
    ReqContext,
    make_offload_key,
)


def cdiv(a: int, b: int) -> int:
    """Ceiling division. REFERENCE: vllm.utils.math_utils.cdiv used in scheduler.py."""
    return -(-a // b)


# REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L61-L82
@dataclass(frozen=True)
class GroupOffloadConfig:
    """Per KV-cache-group block-size facts.

    `gpu_block_size` < `offloaded_block_size` is the typical case
    (CPU blocks are larger to amortize PCIe overhead). The
    `hash_block_size_factor` accounts for prefix-cache hashing
    happening at a finer granularity than offload.
    """

    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int
    sliding_window_size_in_blocks: Optional[int] = None  # None = full attention


@dataclass(frozen=True)
class SchedulerOffloadConfig:
    """Per-spec config materialized once at scheduler init.

    REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L85-L111
    """

    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int
    num_workers: int

    @classmethod
    def from_groups(
        cls,
        gpu_block_sizes: tuple[int, ...],
        block_size_factor: int,
        hash_block_size: int,
        sliding_windows: tuple[Optional[int], ...] | None = None,
        num_workers: int = 1,
    ) -> "SchedulerOffloadConfig":
        # SIMPLIFIED: original derives from `OffloadingSpec`; we accept the
        # primitives for educational clarity.
        if sliding_windows is None:
            sliding_windows = (None,) * len(gpu_block_sizes)
        assert len(sliding_windows) == len(gpu_block_sizes)
        configs: list[GroupOffloadConfig] = []
        for idx, gpu_block_size in enumerate(gpu_block_sizes):
            offloaded_block_size = gpu_block_size * block_size_factor
            assert offloaded_block_size % hash_block_size == 0
            sliding = sliding_windows[idx]
            sliding_blocks = (
                cdiv(sliding, offloaded_block_size) if sliding is not None else None
            )
            configs.append(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=offloaded_block_size,
                    hash_block_size_factor=offloaded_block_size // hash_block_size,
                    sliding_window_size_in_blocks=sliding_blocks,
                )
            )
        return cls(
            kv_group_configs=tuple(configs),
            block_size_factor=block_size_factor,
            num_workers=num_workers,
        )


# REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L114-L122
@dataclass
class RequestGroupState:
    """Per-(req, group) tracking. The `next_stored_block_idx` cursor
    advances as blocks are offloaded so re-visited requests don't store
    twice."""

    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    next_stored_block_idx: int = 0
    num_hit_blocks: int = 0


@dataclass
class _SimpleRequest:
    """Educational stand-in for vllm.v1.request.Request.

    REFERENCE: real Request is a 200+ field dataclass; we keep only the
    fields the scheduler reads here.
    """

    request_id: str
    block_hashes: list[bytes]
    num_tokens: int
    num_computed_tokens: int = 0
    kv_transfer_params: Optional[dict[str, Any]] = None


# REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L125-L182
@dataclass
class RequestOffloadState:
    """Per-request scheduler-side state.

    A request can have at most ONE in-flight LOAD job, but multiple
    in-flight STORE jobs (per group). The `transfer_jobs` set tracks
    both — invariant enforced by the connector.
    """

    config: SchedulerOffloadConfig
    req: _SimpleRequest
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    req_context: ReqContext = field(init=False)
    num_locally_computed_tokens: int = 0
    transfer_jobs: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        self.req_context = ReqContext(
            kv_transfer_params=self.req.kv_transfer_params
        )

    def update_offload_keys(self) -> None:
        """Compute offload keys from block hashes for each group.

        REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L143-L158
        """
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            # Slice the block_hashes at the right factor + offset.
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_num_hit_blocks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_blocks = (
                num_cached_tokens // group_config.offloaded_block_size
            )


class OffloadingConnectorScheduler:
    """Scheduler-side reactive prefix-lookup.

    REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L185-L500

    Public API consumed by KVConnectorBase_V1:
      * `get_num_new_matched_tokens(req, num_computed)` — returns
        (num_offloaded_tokens_beyond_computed, is_async)
    """

    def __init__(
        self,
        manager: OffloadingManager,
        config: SchedulerOffloadConfig,
        enable_prefix_caching: bool = True,
    ):
        self.manager = manager
        self.config = config

        # REFERENCE: scheduler.py:L191-L210
        full_attn_groups: list[int] = []
        sliding_groups: list[int] = []
        for gc in config.kv_group_configs:
            if gc.sliding_window_size_in_blocks is None:
                full_attn_groups.append(gc.group_idx)
            else:
                sliding_groups.append(gc.group_idx)
        # sort sliding-window groups by window size DESC
        sliding_groups.sort(
            key=lambda i: config.kv_group_configs[i].sliding_window_size_in_blocks or 0,
            reverse=True,
        )
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_groups)
        self._lookup_groups: tuple[int, ...] = tuple(
            full_attn_groups
        ) + self._sliding_window_groups

        self._req_status: dict[str, RequestOffloadState] = {}
        # If GPU prefix caching is enabled, dedupe in-flight loads.
        # REFERENCE: scheduler.py:L217-L219
        self._blocks_being_loaded: Optional[set[OffloadKey]] = (
            set() if enable_prefix_caching else None
        )

    def _maximal_prefix_lookup(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
    ) -> Optional[int]:
        """Count consecutive offloaded keys from the start.

        REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L244-L261

        This is the load-bearing reactive-prefix routine. It is NOT
        predictive: it walks the prefix in order and stops at the first miss.
        Backends MAY return None to defer the lookup (LMCache async warming);
        we still continue so the backend can pipeline work, but we record
        the deferral for the caller.
        """
        hit_count = 0
        defer_lookup = False
        for key in keys:
            result = self.manager.lookup(key, req_context)
            if result is None:
                defer_lookup = True
                # Pretend hit so the backend gets a chance to pipeline.
                result = True
            if not result:
                break
            hit_count += 1
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> Optional[int]:
        """Find the suffix end-index of the last run of `sliding_window_size`
        consecutive hits, scanning from the end.

        REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L263-L287

        Return value:
          0     → no sufficient run found
          int N → end index (exclusive) up to which hits are confirmed
          None  → backend deferred lookup; retry next step
        """
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            result = self.manager.lookup(keys[idx], req_context)
            if result is None:
                defer_lookup = True
                result = False  # for sliding window, defer = no hit
            if not result:
                consecutive_hits = 0
            else:
                consecutive_hits += 1
                if consecutive_hits == sliding_window_size:
                    return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState) -> None:
        """Mark all hit blocks as recently used (drives LRU/ARC).

        REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L289-L303
        """
        for gc, gs in zip(self.config.kv_group_configs, req_status.group_states):
            if gc.sliding_window_size_in_blocks is None:
                self.manager.touch(gs.offload_keys)
            else:
                blocks_to_skip = max(
                    0, gs.num_hit_blocks - gc.sliding_window_size_in_blocks
                )
                self.manager.touch(gs.offload_keys[blocks_to_skip:])

    def get_num_new_matched_tokens(
        self,
        request: _SimpleRequest,
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """Public API. Returns (num_tokens_to_load_from_offload, is_async).

        REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L443-L486
        """
        # First: bind / fetch RequestOffloadState.
        is_new = False
        rs = self._req_status.get(request.request_id)
        if rs is None:
            is_new = True
            rs = RequestOffloadState(config=self.config, req=request)
            self._req_status[request.request_id] = rs
        else:
            for gs in rs.group_states:
                gs.block_ids.clear()

        rs.update_offload_keys()
        rs.num_locally_computed_tokens = num_computed_tokens

        # Compute the cross-group num_hit_tokens.
        num_hit_tokens = self._lookup(rs)
        if is_new:
            rs.update_num_hit_blocks(
                num_computed_tokens + (num_hit_tokens or 0)
            )

        self._touch(rs)
        return num_hit_tokens, bool(num_hit_tokens)

    def _lookup(
        self,
        rs: RequestOffloadState,
    ) -> Optional[int]:
        """Cross-group prefix lookup. Returns total hit tokens beyond
        num_locally_computed_tokens.

        SIMPLIFIED: the production version handles full-attention +
        sliding-window groups in the same pass with a re-iteration loop
        when a tighter group-result invalidates an earlier one. Here we
        do a single pass for clarity; the demo only exercises one
        full-attention group at a time.
        REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L305-L441
        """
        num_computed_tokens = rs.num_locally_computed_tokens
        max_hit_size_tokens = rs.req.num_tokens
        if self._sliding_window_groups:
            max_hit_size_tokens -= 1
        num_hit_tokens = 0
        defer_lookup = False

        for group_idx in self._lookup_groups:
            gc = self.config.kv_group_configs[group_idx]
            gs = rs.group_states[group_idx]
            offloaded_block_size = gc.offloaded_block_size
            offload_keys = gs.offload_keys

            assert (
                len(offload_keys)
                >= rs.req.num_tokens // offloaded_block_size
            )

            max_hit_size_tokens = min(
                max_hit_size_tokens, len(offload_keys) * offloaded_block_size
            )
            if max_hit_size_tokens - num_computed_tokens < offloaded_block_size:
                return 0

            num_blocks = min(
                cdiv(max_hit_size_tokens, offloaded_block_size), len(offload_keys)
            )
            start_block_idx = num_computed_tokens // offloaded_block_size
            sliced_keys = offload_keys[start_block_idx:num_blocks]
            sliding = gc.sliding_window_size_in_blocks

            if sliding is None:
                num_hit_blocks = self._maximal_prefix_lookup(
                    sliced_keys, rs.req_context
                )
            else:
                num_hit_blocks = self._sliding_window_lookup(
                    sliced_keys, sliding, rs.req_context
                )

            if num_hit_blocks == 0:
                return 0

            if num_hit_blocks is None:
                defer_lookup = True
            else:
                max_hit_size_tokens = min(
                    max_hit_size_tokens,
                    offloaded_block_size * (start_block_idx + num_hit_blocks),
                )

            new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
            if new_num_hit_tokens < offloaded_block_size:
                return 0
            num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            return None

        # Dedupe against blocks-being-loaded.
        if self._blocks_being_loaded:
            for gc, gs in zip(self.config.kv_group_configs, rs.group_states):
                offloaded_block_size = gc.offloaded_block_size
                num_blocks = cdiv(
                    num_computed_tokens + num_hit_tokens, offloaded_block_size
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                slice_keys = gs.offload_keys[start_block_idx:num_blocks]
                if any(k in self._blocks_being_loaded for k in slice_keys):
                    return None
        return num_hit_tokens


# Compute helpers used by tests / demos
def overlap_blocks_per_step(
    step_compute_ms: float,
    transfer_latency_us_per_block: float,
) -> int:
    """How many block transfers fit in one step's compute budget?

    REFERENCE: §1 movement 4 of brief — `N_overlap = step_compute / transfer_latency`.
    For decode (50 ms) at 250 µs/block: 50000/250 = 200 blocks/step.
    """
    if transfer_latency_us_per_block <= 0:
        return 0
    return int(math.floor((step_compute_ms * 1000.0) / transfer_latency_us_per_block))


def headroom_freed_gb(
    cpu_offload_size_gb: float,
    hit_rate: float,
    hbm_to_dram_ratio: float = 1.0,
) -> float:
    """HBM headroom freed by offload.

    headroom = cpu_offload_size_GB * hit_rate * (HBM/DRAM ratio).
    Default ratio 1.0 since vLLM's CPU blocks mirror GPU block byte size
    (aside from block_size_factor); set to 0.5 if running with 2x CPU blocks.
    """
    return cpu_offload_size_gb * hit_rate * hbm_to_dram_ratio
