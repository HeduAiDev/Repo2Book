# vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py
#   —— subtract-only companion（★ key 与地址的生成器 + scheduler→worker 传递契约）
#
# 三件事：①PoolKey —— 池中一个 KV chunk 的全局名字（内容寻址，复用成立的根基）；
#   ②ChunkedTokenDatabase —— process_tokens 把 token 切 chunk 出 key、prepare_value 由 block_id
#   算 (addr,size)；store/lookup/load 三路共用。③LoadSpec/ReqMeta/AscendConnectorMetadata ——
#   调度器→worker 的请求清单与「搬多少」载体。
#
# host 无 vllm：BlockHash/cdiv/KVConnectorMetadata 经 runtime_stub 接住；切 chunk / 命名 / 地址
#   算术本身是纯 Python，可跑。
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

# SUBTRACTED: import torch（仅 ReqMeta.current_event 的 torch.npu.Event 类型标注用，已降级为无标注）；
#   NewRequestData（仅 RequestTracker.from_new_request 用，已删）。原 config_data.py:L8-L13
from runtime_stub import BlockHash, BlockHashList, KVConnectorMetadata, cdiv, logger

_GROUPED_BLOCK_HASH_DOMAIN = b"vllm-ascend-grouped-block-hash-v1\0"
_GROUPED_BLOCK_HASH_LENGTH_PREFIX_BYTES = 4


@dataclass
class KeyMetadata:  # SOURCE: config_data.py:L21
    """Parameters related to the key."""

    model_name: str
    head_or_tp_rank: int
    pcp_rank: int
    dcp_rank: int
    pp_rank: int
    kv_cache_group_id: int = 0
    cache_role: str = "kv"
    cache_family: str = "default"


@dataclass(order=True)
class PoolKey:  # SOURCE: config_data.py:L42
    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self):  # SOURCE: config_data.py:L46
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.key_metadata.kv_cache_group_id,
                self.key_metadata.cache_role,
                self.key_metadata.cache_family,
                self.chunk_hash,
            )
        )

    def to_string(self):  # SOURCE: config_data.py:L61
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}"
            f"@group:{self.key_metadata.kv_cache_group_id}"
            f"@cache_role:{self.key_metadata.cache_role}"
            f"@cache_family:{self.key_metadata.cache_family}"
            f"@{self.chunk_hash}"
        )

    # SUBTRACTED: split_layers（L73-L84）—— layerwise 逐层流水模式才用，把一个 chunk key 拆成每层一个
    #   LayerPoolKey；本主线走非 layerwise 整请求搬运。原 config_data.py:L73-L84


# SUBTRACTED: class LayerPoolKey(PoolKey)（L88-L118）—— layerwise 模式的逐层 key。
# SUBTRACTED: infer_cache_family_from_ratio / infer_cache_family_ratio / get_cache_family_granularity /
#   _get_layer_compress_ratio / _get_group_spec_ratios / infer_group_cache_families（L121-L199）——
#   DSV4 compress(c1/c4/c128) 家族推断；default 家族 ratio 恒为 1。原 config_data.py:L121-L199


class ChunkedTokenDatabase:  # SOURCE: config_data.py:L202
    def __init__(  # SOURCE: config_data.py:L203
        self,
        metadata: list[KeyMetadata],
        block_size: list[int],
        partitions: list[int] | None = None,
        use_hybrid: bool = False,
        hash_block_size: int | None = None,
    ):
        self.metadata = metadata
        self.block_size = block_size
        self.group_kv_caches_base_addr: dict[int, list[int]] = {}
        self.group_block_len: dict[int, list[int]] = {}
        self.group_block_stride: dict[int, list[int]] = {}
        self.group_cache_families: dict[str, dict[int, str]] = {"kv": {}, "state": {}}
        # SUBTRACTED: group_num_layers / partitions / use_hybrid 的 kv-event 与 PP-adaptor 用途。
        self.partitions = partitions
        self.use_hybrid = use_hybrid
        self.hash_block_size = self.block_size[0] if hash_block_size is None else hash_block_size

    def _make_key_by_hash(  # SOURCE: config_data.py:L228
        self,
        chunk_hash: str,
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        cache_family: str | None = None,
        layer_id: int | None = None,
    ):
        assert self.metadata is not None
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
        group_metadata = self.metadata[kv_cache_group_id]
        return PoolKey(
            KeyMetadata(
                model_name=group_metadata.model_name,
                head_or_tp_rank=group_metadata.head_or_tp_rank,
                pcp_rank=group_metadata.pcp_rank,
                dcp_rank=group_metadata.dcp_rank,
                pp_rank=group_metadata.pp_rank,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                cache_family=cache_family,
            ),
            chunk_hash,
        )

    def get_block_size(self, kv_cache_group_id: int) -> int:  # SOURCE: config_data.py:L254
        if kv_cache_group_id >= len(self.block_size):
            return self.block_size[0]
        return self.block_size[kv_cache_group_id]

    def set_group_buffers(  # SOURCE: config_data.py:L259
        self,
        group_kv_caches_base_addr: dict[int, list[int]],
        group_block_len: dict[int, list[int]],
        group_block_stride: dict[int, list[int]] | None = None,
        cache_role: str = "kv",
        group_cache_families: dict[int, str] | None = None,
        group_num_layers: dict[int, int] | None = None,
    ) -> None:
        # SUBTRACTED: cache_role == "state" 分支（DSV4 compressor/indexer 状态，L268-L271）。
        self.group_kv_caches_base_addr = group_kv_caches_base_addr
        self.group_block_len = group_block_len
        self.group_block_stride = group_block_stride or {}
        if group_cache_families is not None:
            self.group_cache_families[cache_role] = group_cache_families.copy()

    def _get_group_buffers(  # SOURCE: config_data.py:L281
        self, kv_cache_group_id: int, cache_role: str = "kv"
    ) -> tuple[list[int], list[int], list[int] | None]:
        return (
            self.group_kv_caches_base_addr[kv_cache_group_id],
            self.group_block_len[kv_cache_group_id],
            self.group_block_stride.get(kv_cache_group_id),
        )

    def prepare_value(  # SOURCE: config_data.py:L292
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        addr_list: list[int] = []
        size_list: list[int] = []
        group_block_size = self.get_block_size(kv_cache_group_id)
        block_id = block_ids[start // group_block_size]
        group_addrs, group_block_len, group_block_stride = self._get_group_buffers(kv_cache_group_id, cache_role)
        length = len(group_block_len)
        if length == 0:
            return addr_list, size_list, block_id
        for index, base_addr in enumerate(group_addrs):
            block_len = group_block_len[index % length]
            block_stride = group_block_stride[index % length] if group_block_stride else block_len
            addr = base_addr + block_id * block_stride
            size = int(block_len / group_block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list, block_id

    # SUBTRACTED: prepare_value_layer（L317-L330）—— layerwise 模式按 layer_id 算每层地址。

    def process_tokens(  # SOURCE: config_data.py:L332
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        cache_family: str | None = None,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        """Process the tokens and return the corresponding cache engine keys."""
        if not block_hashes:
            return
        group_block_size = self.get_block_size(kv_cache_group_id)
        # SUBTRACTED: cache_family_ratio 缩放（group_block_size *= ratio，再把 start/end //= ratio，
        #   L345-L348, L370-L371）—— DSV4 compress 才有；default 家族 ratio 恒为 1。
        block_hashes = get_block_hashes(block_hashes, group_block_size, self.hash_block_size)
        if not block_hashes:
            return
        if not isinstance(block_hashes[0], str):
            block_hashes = [h.hex() for h in block_hashes]  # type: ignore[union-attr]
        for chunk_id, hash_val in enumerate(block_hashes):
            start_idx = chunk_id * group_block_size
            if start_idx >= token_len:
                break
            end_idx = min(start_idx + group_block_size, token_len)
            if start_idx < mask_num:
                continue
            if end_idx <= start_idx:
                continue
            yield (
                start_idx,
                end_idx,
                self._make_key_by_hash(
                    hash_val,
                    kv_cache_group_id=kv_cache_group_id,
                    cache_role=cache_role,
                    cache_family=cache_family,
                ),
            )

    def process_tokens_with_block_ids(  # SOURCE: config_data.py:L385
        self,
        token_len: int,
        block_hashes: BlockHashList | list[str],
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
        cache_family: str | None = None,
    ) -> Iterable[tuple[int, int, PoolKey, int]]:
        for start_idx, end_idx, key in self.process_tokens(
            token_len,
            block_hashes,
            mask_num,
            kv_cache_group_id=kv_cache_group_id,
            cache_role=cache_role,
            cache_family=cache_family,
        ):
            block_idx = start_idx // self.get_block_size(kv_cache_group_id)
            if block_idx >= len(block_ids):
                continue
            block_id = block_ids[block_idx]
            if skip_null_blocks and block_id <= 0:
                continue
            yield start_idx, end_idx, key, block_id

    # SUBTRACTED: decode_adaptor_prefill_pp（L412-L434）—— prefill PP 切分场景重写 key 的 @pp_rank；
    #   单 PP(prefill_pp_size=1) 直接返回原值。原 config_data.py:L412-L434


def normalize_block_ids_by_group(  # SOURCE: config_data.py:L437
    block_ids: tuple[list[int], ...] | list[int] | list[list[int]],
) -> list[list[int]]:
    if isinstance(block_ids, tuple):
        return [group.copy() for group in block_ids]
    if isinstance(block_ids, list):
        if not block_ids:
            return [[]]
        if isinstance(block_ids[0], list):
            grouped_block_ids = cast("list[list[int]]", block_ids)
            return [group.copy() for group in grouped_block_ids]
        flat_block_ids = cast("list[int]", block_ids)
        return [flat_block_ids.copy()]
    raise ValueError(f"Unsupported block_ids type {type(block_ids)}")


def get_block_hashes(  # SOURCE: config_data.py:L451
    block_hashes: BlockHashList | list[str],
    group_block_size: int,
    hash_block_size: int,
) -> BlockHashList | list[str]:
    if group_block_size == hash_block_size:
        return block_hashes
    assert group_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
    scale_factor = group_block_size // hash_block_size
    return [
        _rehash_block_hash_group(block_hashes[idx : idx + scale_factor])
        for idx in range(0, len(block_hashes) // scale_factor * scale_factor, scale_factor)
    ]


def _rehash_block_hash_group(block_hashes: Sequence[BlockHash | str]) -> BlockHash:  # SOURCE: config_data.py:L466
    hasher = hashlib.sha256()
    hasher.update(_GROUPED_BLOCK_HASH_DOMAIN)
    hasher.update(len(block_hashes).to_bytes(_GROUPED_BLOCK_HASH_LENGTH_PREFIX_BYTES, "big"))
    for block_hash in block_hashes:
        hash_bytes = _block_hash_to_bytes(block_hash)
        hasher.update(len(hash_bytes).to_bytes(_GROUPED_BLOCK_HASH_LENGTH_PREFIX_BYTES, "big"))
        hasher.update(hash_bytes)
    return BlockHash(hasher.digest())


def _block_hash_to_bytes(block_hash: BlockHash | str) -> bytes:  # SOURCE: config_data.py:L477
    if isinstance(block_hash, str):
        if len(block_hash) == 64:
            try:
                return bytes.fromhex(block_hash)
            except ValueError:
                return block_hash.encode("utf-8")
        return block_hash.encode("utf-8")
    return bytes(block_hash)


@dataclass
class LoadSpec:  # SOURCE: config_data.py:L490
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in kvpool
    kvpool_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool

    token_len: int = 0


@dataclass(init=False)
class RequestTracker:  # SOURCE: config_data.py:L502
    req_id: str
    token_len: int
    allocated_block_ids_by_group: list[list[int]]
    num_saved_tokens: int = 0
    token_ids: list[int] | None = None

    def __init__(  # SOURCE: config_data.py:L519
        self,
        req_id: str,
        token_len: int,
        allocated_block_ids_by_group: list[list[int]] | None = None,
        allocated_block_ids: list[int] | list[list[int]] | None = None,
        num_saved_tokens: int = 0,
        token_ids: list[int] | None = None,
    ) -> None:
        self.req_id = req_id
        self.token_len = token_len
        block_ids = allocated_block_ids_by_group
        if block_ids is None:
            block_ids = normalize_block_ids_by_group(allocated_block_ids or [])
        self.allocated_block_ids_by_group = block_ids
        self.num_saved_tokens = num_saved_tokens
        self.token_ids = token_ids

    @property
    def allocated_block_ids(self) -> list[int]:  # SOURCE: config_data.py:L538
        return self.allocated_block_ids_by_group[0] if self.allocated_block_ids_by_group else []

    # SUBTRACTED: from_new_request（L545-L557）—— 仅依赖 vllm NewRequestData 的便捷构造，build_connector_meta
    #   走直接构造 RequestTracker(...)。原 config_data.py:L545-L557

    def update(  # SOURCE: config_data.py:L559
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
    ) -> None:
        """Update the request tracker when a running request is scheduled again."""
        normalized = normalize_block_ids_by_group(new_block_ids)
        if len(normalized) > len(self.allocated_block_ids_by_group):
            self.allocated_block_ids_by_group.extend(
                [[] for _ in range(len(normalized) - len(self.allocated_block_ids_by_group))]
            )
        for group_id, ids in enumerate(normalized):
            self.allocated_block_ids_by_group[group_id].extend(ids)


@dataclass(init=False)
class ReqMeta:  # SOURCE: config_data.py:L574
    req_id: str
    token_len_chunk: int
    block_ids_by_group: list[list[int]]
    block_hashes: list[BlockHash]
    can_save: bool | None = None
    load_spec: LoadSpec | None = None
    is_last_chunk: bool | None = None
    # SUBTRACTED: current_event: torch.npu.Event 类型标注（host 无 NPU；运行期由 wait_for_save 注入）。
    current_event = None
    kv_cache_group_ids: list[int] | None = None
    kv_cache_families_by_group: list[str] | None = None
    skip_null_blocks_by_group: list[bool] | None = None
    disable_tp_key_sharding: bool = False
    token_ids: list[int] | None = None
    original_block_size: list[int] | int | None = None
    event_id: int | None = None

    def __init__(  # SOURCE: config_data.py:L603
        self,
        req_id: str,
        token_len_chunk: int,
        block_ids_by_group: list[list[int]] | None = None,
        block_hashes: list[BlockHash] | None = None,
        can_save: bool | None = None,
        load_spec: LoadSpec | None = None,
        is_last_chunk: bool | None = None,
        current_event=None,
        kv_cache_group_ids: list[int] | None = None,
        kv_cache_families_by_group: list[str] | None = None,
        skip_null_blocks_by_group: list[bool] | None = None,
        disable_tp_key_sharding: bool = False,
        token_ids: list[int] | None = None,
        original_block_size: list[int] | int | None = None,
        block_ids: list[int] | list[list[int]] | None = None,
        event_id: int | None = None,
    ) -> None:
        self.req_id = req_id
        self.token_len_chunk = token_len_chunk
        if block_ids_by_group is None:
            block_ids_by_group = normalize_block_ids_by_group(block_ids or [])
        self.block_ids_by_group = block_ids_by_group
        self.block_hashes = [] if block_hashes is None else block_hashes
        self.can_save = can_save
        self.load_spec = load_spec
        self.is_last_chunk = is_last_chunk
        self.current_event = current_event
        self.kv_cache_group_ids = kv_cache_group_ids
        self.kv_cache_families_by_group = kv_cache_families_by_group
        self.skip_null_blocks_by_group = skip_null_blocks_by_group
        self.disable_tp_key_sharding = disable_tp_key_sharding
        self.token_ids = token_ids
        self.original_block_size = original_block_size
        self.event_id = event_id

    @property
    def block_ids(self) -> list[int]:  # SOURCE: config_data.py:L641
        return self.block_ids_by_group[0] if self.block_ids_by_group else []

    @staticmethod
    def from_request_tracker(  # SOURCE: config_data.py:L649
        tracker: RequestTracker,
        cache_transfer_granularity: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
        original_block_size: list[int] | int | None = None,
        kv_cache_group_families: list[str] | None = None,
    ) -> ReqMeta | None:
        """Create the request metadata from a request tracker."""
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, cache_transfer_granularity) * cache_transfer_granularity
            if discard_partial_chunks
            else 0
        )
        num_tokens_to_save = (
            (input_token_len // cache_transfer_granularity * cache_transfer_granularity)
            if discard_partial_chunks
            else input_token_len
        )

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        token_ids = None
        if tracker.token_ids:
            token_ids = tracker.token_ids

        if load_spec is not None and load_spec.can_load:
            logger.debug("Scheduled to load %d tokens for request %s", load_spec.kvpool_cached_tokens, tracker.req_id)
        else:
            load_spec = None
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids_by_group=tracker.allocated_block_ids_by_group,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
            original_block_size=original_block_size,
            kv_cache_group_ids=list(range(len(tracker.allocated_block_ids_by_group))),
            kv_cache_families_by_group=kv_cache_group_families,
        )


class AscendConnectorMetadata(KVConnectorMetadata):  # SOURCE: config_data.py:L714
    def __init__(self, unfinished_request_ids, preempted_req_ids):  # SOURCE: config_data.py:L715
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:  # SOURCE: config_data.py:L720
        """Add a request to the metadata."""
        self.requests.append(req_meta)


# SUBTRACTED: class LayerMultiBlockReqMeta（L726-L777）—— layerwise 模式逐层搬运的请求元数据。
# SUBTRACTED: class AscendStoreKVConnectorWorkerMetadata（L781-L800）—— mamba 块异步释放的 worker→
#   scheduler 回执（hybrid/mamba 路径）。原 config_data.py:L781-L800
