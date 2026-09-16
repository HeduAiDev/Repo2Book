# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""connector 公共层：块号类型、TP 聚合器、传输拓扑（TP 比例与水位的唯一真相源）。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py（逐段行号见各符号上方）
# SUBTRACTED:
#   * copy_kv_blocks / kv_postprocess_*（host buffer 搬运与异构块尺寸/布局换算，
#     L176-L303）——减法计划删除项 3/1：CUDA 主线 use_host_buffer=False，
#     对称 TP 下 block_size_ratio=1、无布局转换；
#   * cross-layer 块族（TransferTopology.__post_init__ 的 _cross_layers_blocks
#     分支）——减法计划删除项 4。
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)

# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L31-L34
EngineId = str
# block ids as returned by the hybrid KV cache manager. list[list[int]] are allow
# mutability and are for connector internal use only.
BlockIds = tuple[list[int], ...] | list[list[int]]


# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L37-L50
def get_kv_connector_cache_layout():
    # NOTE (NickLucche) When running disaggregated PD with NIXL, HND layout is
    # used for faster transfer.
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    vllm_config = get_current_vllm_config()
    kv_config = vllm_config.kv_transfer_config
    if kv_config is not None:
        connector_cls = KVConnectorFactory.get_connector_class(kv_config)
        required_kvcache_layout = connector_cls.get_required_kvcache_layout(vllm_config)
        if required_kvcache_layout is not None:
            return required_kvcache_layout
        logger.info_once(
            "Connectors do not specify a kv cache layout, defaulting to NHD."
        )
    return "NHD"


# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L53-L173
class KVOutputAggregator:
    """Utility class to aggregate the output of all workers into a single
    output corresponding to Rank 0 for scheduler."""

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L57-L62
    def __init__(self, expected_finished_count: int):
        # Complete transfer tracker. Used to track finished requests
        # [req_id -> n_remaining_workers]
        self._recv_remaining_count = dict[str, int]()
        self._send_remaining_count = dict[str, int]()
        self._expected_finished_count = expected_finished_count

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L64-L66
    @classmethod
    def from_connector(cls, connector: Any, world_size: int):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L64-L66
        return cls(connector.get_finished_count() or world_size)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L68-L173
    # SUBTRACTED: kv_connector_stats / worker_meta / kv_cache_events 的聚合
    #   （L121-L155）——观测旁路，按减法计划删除项 6 删。
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L68-L173
    def aggregate(
        self, outputs: list["ModelRunnerOutput | None"], output_rank: int = 0
    ) -> "ModelRunnerOutput | None":
        if not outputs[output_rank]:
            return None

        # Aggregate kv_connector_output from all workers

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L68-L173
        def update_finished_set(
            req_ids: set[str] | None,
            remaining_count_dict: dict[str, int],
            finished_set: set[str],
        ) -> None:
            for req_id in req_ids or ():
                remaining_count = remaining_count_dict.get(
                    req_id, self._expected_finished_count
                )
                remaining_count_dict[req_id] = remaining_count - 1
                if remaining_count_dict[req_id] == 0:
                    finished_set.add(req_id)
                    del remaining_count_dict[req_id]

        finished_sending = set[str]()
        finished_recving = set[str]()
        invalid_block_ids = set[int]()
        for model_runner_output in outputs:
            assert model_runner_output is not None
            kv_output = model_runner_output.kv_connector_output
            if not kv_output:
                continue
            # Allow the worker to dynamically update the expected number of
            # finished sending/recving for new requests.
            if (
                kv_output.expected_finished_count > 0
                and kv_output.expected_finished_count != self._expected_finished_count
            ):
                logger.debug(
                    "Expected finished requests updated from %d to %d",
                    self._expected_finished_count,
                    kv_output.expected_finished_count,
                )
                self._expected_finished_count = kv_output.expected_finished_count

            update_finished_set(
                kv_output.finished_sending, self._send_remaining_count, finished_sending
            )
            update_finished_set(
                kv_output.finished_recving, self._recv_remaining_count, finished_recving
            )

            invalid_block_ids |= kv_output.invalid_block_ids

        # select output of the worker specified by output_rank
        output = outputs[output_rank]

        from vllm.v1.outputs import KVConnectorOutput

        assert output is not None
        output.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            invalid_block_ids=invalid_block_ids,
            expected_finished_count=self._expected_finished_count,
        )

        return output


# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L306-L323
def yield_req_data(
    scheduler_output,
) -> Iterator[tuple[str, tuple[list[int], ...] | None, bool]]:
    """
    Yields:
        (req_id, new_block_id_groups, preempted)
    """
    # new requests
    for req_data in scheduler_output.scheduled_new_reqs:
        yield req_data.req_id, req_data.block_ids, False

    # cached requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    if cached_reqs is None:
        return
    yield from zip(
        cached_reqs.req_ids,
        cached_reqs.new_block_ids,
        (req_id in cached_reqs.resumed_req_ids for req_id in cached_reqs.req_ids),
    )


# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L326-L362
def get_current_attn_backends(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> list[type[AttentionBackend]]:
    """Get all distinct attention backends for the given layers."""
    layer_type = cast(type[Any], object)
    layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)
    if layers:
        seen: dict[str, type[AttentionBackend]] = {}
        for layer in layers.values():
            backend = layer.get_attn_backend()
            seen[backend.full_cls_name()] = backend
        return list(seen.values())

    # Fallback for tests, when static_forward_context is empty.
    logger.debug(
        "No layers found in the vLLM config. Falling back to default attention backend."
    )
    from vllm.v1.attention.backend import get_attn_backend

    with set_current_vllm_config(vllm_config):
        return [
            get_attn_backend(
                head_size=vllm_config.model_config.get_head_size(),
                dtype=vllm_config.model_config.dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                use_mla=vllm_config.model_config.use_mla,
            )
        ]


# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L365-L369
def get_current_attn_backend(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> type[AttentionBackend]:
    """Get the first attention backend for the given layers."""
    return get_current_attn_backends(vllm_config, layer_names)[0]


# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L372-L400
@dataclass(frozen=True)
class EngineTransferInfo:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L372-L400
    """Common per-remote-engine transfer state, computed at handshake.

    Stored per ``(engine_id, pp_rank)`` inside ``TransferTopology._engines``.
    """

    remote_tp_size: int

    remote_block_len: int
    """Block length (bytes)"""

    remote_block_size: int
    """Tokens per block."""

    remote_physical_blocks_per_logical: int
    """Physical blocks per logical block."""

    remote_pp_rank: int = 0
    """Remote producer PP rank for this engine."""


# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L406-L612
# SUBTRACTED: cross-layer 块探测（tensor_shape 入参与 __post_init__ 里的
#   _cross_layers_blocks 推断，L448-L465）——减法计划删除项 4。
# SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L406-L612
@dataclass
class TransferTopology:
    """Single source of truth for local TP identity and per-engine remote info."""

    tp_rank: int
    tp_size: int
    block_size: int
    engine_id: EngineId
    is_mla: bool
    is_mamba: bool
    total_num_kv_heads: int
    attn_backends: list[type[AttentionBackend]]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L406-L612
    def __post_init__(self):
        self.local_physical_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self._engines: dict[tuple[EngineId, int], EngineTransferInfo] = {}

        # Figure out whether the first dimension of the cache is K/V
        # or num_blocks.
        attn_backend = self.attn_backends[0]
        if not self.is_mamba:
            _MOCK_BLOCK_SIZE = 16
            kv_cache_shape: tuple[int, ...] = attn_backend.get_kv_cache_shape(
                num_blocks=1,
                block_size=_MOCK_BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
            )
            logger.debug("Test kv_cache_shape: %s", kv_cache_shape)
            assert kv_cache_shape[0] == 1, (
                "KV cache layout must be blocks-first; expected mocked "
                f"num_blocks=1 in leading dim, got shape {kv_cache_shape}."
            )
            if not self.is_mla:
                assert len(kv_cache_shape) == 4, (
                    "Attention KV cache layout must be standardized as "
                    "[num_blocks, num_kv_heads, block_size, content_size], "
                    f"got shape {kv_cache_shape}."
                )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L471-L489
    def register_remote_engine(
        self,
        remote_engine_id: EngineId,
        info: EngineTransferInfo,
    ) -> EngineTransferInfo:
        """Register a remote engine, unifying worker dicts state."""
        assert remote_engine_id != self.engine_id, (
            f"Cannot register local engine {self.engine_id} as remote. "
            f"Local identity is set via __init__ params."
        )
        engine_key = (remote_engine_id, info.remote_pp_rank)
        if engine_key in self._engines:
            return self._engines[engine_key]
        self._engines[engine_key] = info
        return info

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L491-L494
    def get_engine_info(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> EngineTransferInfo:
        return self._engines[(remote_engine_id, remote_pp_rank)]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L496-L499
    def unregister_remote_engine(self, remote_engine_id: EngineId) -> None:
        # Remove all pp_rank entries for the remote engine.
        for key in [k for k in self._engines if k[0] == remote_engine_id]:
            del self._engines[key]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L505-L507
    @property
    def cross_layers_blocks(self) -> bool:
        # SUBTRACTED: cross-layer 布局探测——本章恒 False（逐层注册路径）。
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L505-L507
        return False

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L523-L540
    def tp_ratio(self, remote_tp_size: int) -> int:
        """Calculate the tensor parallel ratio between local and remote TP."""
        if self.tp_size >= remote_tp_size:
            assert self.tp_size % remote_tp_size == 0, (
                f"Local tensor parallel size {self.tp_size} is not divisible "
                f"by remote tensor parallel size {remote_tp_size}."
            )
            return self.tp_size // remote_tp_size
        assert remote_tp_size % self.tp_size == 0, (
            f"Remote tensor parallel size {remote_tp_size} is not divisible "
            f"by local tensor parallel size {self.tp_size}."
        )
        return -(remote_tp_size // self.tp_size)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L542-L548
    def block_size_ratio(self, remote_block_size: int) -> int:
        """Calculate the block size ratio between local and remote."""
        assert self.block_size % remote_block_size == 0, (
            f"Local block size {self.block_size} is not divisible "
            f"by remote block size {remote_block_size} or vice versa."
        )
        return self.block_size // remote_block_size

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L550-L565
    def is_kv_replicated(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> bool:
        """Whether the KV cache is replicated across TP workers due to the
        number of TP workers being greater than the number of KV heads.
        """
        return (
            self._engines[(remote_engine_id, remote_pp_rank)].remote_tp_size
            > self.total_num_kv_heads
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L561-L565
    def replicates_kv_cache(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> bool:
        # MLA is always replicated as the hidden dim can't be split.
        return self.is_mla or self.is_kv_replicated(remote_engine_id, remote_pp_rank)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L572-L582
    def handshake_target_ranks(self, remote_tp_size: int) -> list[int]:
        """Pre-registration: compute which remote TP ranks to handshake with."""
        tp_ratio = self.tp_ratio(remote_tp_size)
        if tp_ratio > 0:
            return [self.tp_rank // tp_ratio]
        abs_ratio = -tp_ratio
        return [self.tp_rank * abs_ratio + i for i in range(abs_ratio)]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/utils.py:L599-L612
    def describe(self, remote_engine_id: EngineId, remote_pp_rank: int = 0) -> str:
        """One-line summary of transfer config for logging."""
        info = self._engines[(remote_engine_id, remote_pp_rank)]
        return (
            f"TransferTopology("
            f"tp_ratio={self.tp_ratio(info.remote_tp_size)}, "
            f"num_kv_heads={self.total_num_kv_heads if not self.is_mla else 1}, "
            f"local_tp={self.tp_size}, "
            f"remote_tp={info.remote_tp_size}, "
            f"remote_pp={remote_pp_rank}, "
            f"local_rank={self.tp_rank}, "
            f"remote_block_len={info.remote_block_len})"
        )


__all__ = [
    "BlockIds",
    "EngineId",
    "KVOutputAggregator",
    "EngineTransferInfo",
    "TransferTopology",
    "get_current_attn_backend",
    "get_current_attn_backends",
    "get_kv_connector_cache_layout",
    "yield_req_data",
]
