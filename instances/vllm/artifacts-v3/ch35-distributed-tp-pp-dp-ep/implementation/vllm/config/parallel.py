# SOURCE: vllm/config/parallel.py
# ch34 切面（站 3/7/16 的配置面）：ParallelConfig 的 DP/EP 字段族 + 波共识原语 +
# 无状态 DP 组构造 + use_all2all/use_sequence_parallel_moe 判定 + needs_dp_coordinator
# 的判定输入。SUBTRACTED：EPLBConfig 整类（L65-L121）、多卡加载/NUMA/DS store/
# elastic EP 校验、compute_hash（L774-L829）、__post_init__ 的非本章分支——详见行内
# 标记。全部保留行对 pin v0.27.1 现核。

from __future__ import annotations

import socket
from typing import Literal, overload

import torch
from pydantic import Field
from torch.distributed import ProcessGroup, ReduceOp, Store

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

# SUBTRACTED: _NUMACTL_CPUSET_PATTERN（L35）/ ExpertPlacementStrategy（L36）/
#   EPLBPolicyOption（L40）/ DCPCommBackend（L41）/ EPLBCommunicatorBackend（L42）
#   ——本章消费面之外（EPLB 归 ch39、DCP 归进阶维度、EPLB 组本身按删除项 1 裁除）。
# SUBTRACTED: DistributedExecutorBackend（L37）——ch17 域的执行器后端表。
# SOURCE: vllm/config/parallel.py:L38 DataParallelBackend —— 逐字
DataParallelBackend = Literal["ray", "mp"]
# SOURCE: vllm/config/parallel.py:L43-L56 All2AllBackend —— 家族表逐字（m22 的
#   后端旋钮清单）
All2AllBackend = Literal[
    "naive",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "deepep_v2",
    "mori_high_throughput",
    "mori_low_latency",
    "nixl_ep",
    "allgather_reducescatter",
    "flashinfer_all2allv",  # temporary alias for flashinfer_nvlink_two_sided
    "flashinfer_nvlink_two_sided",
    "flashinfer_nvlink_one_sided",
]


# SOURCE: vllm/config/parallel.py:L119-L1036 ParallelConfig（@config 装饰位 L118）
@config
class ParallelConfig:
    # SOURCE: vllm/config/parallel.py:L119 ParallelConfig 类头（锚点双置）
    """Configuration for the distributed execution."""

    # SOURCE: vllm/config/parallel.py:L122-L123 pipeline_parallel_size
    pipeline_parallel_size: int = Field(default=1, ge=1)
    """Number of pipeline parallel groups."""
    # SOURCE: vllm/config/parallel.py:L124-L125 tensor_parallel_size
    tensor_parallel_size: int = Field(default=1, ge=1)
    """Number of tensor parallel groups."""
    # SOURCE: vllm/config/parallel.py:L126-L128 prefill_context_parallel_size
    prefill_context_parallel_size: int = Field(default=1, ge=1)
    """Number of ranks that split prefill sequence computation. PCP expands
    the process world size but does not increase the KV-cache shard count."""
    # SOURCE: vllm/config/parallel.py:L129-L131 data_parallel_size
    data_parallel_size: int = Field(default=1, ge=1)
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor, prefill-context, and data parallel sizes."""
    # SOURCE: vllm/config/parallel.py:L132-L135 data_parallel_size_local
    data_parallel_size_local: int = Field(default=1, ge=0)
    """Number of local data parallel groups. A value of 0 is a sentinel used by
    the engine-args layer to signal that data parallelism was specified
    externally (see `ParallelConfig.__post_init__`)."""
    # SOURCE: vllm/config/parallel.py:L136-L138 data_parallel_rank
    data_parallel_rank: int = Field(default=0, ge=0)
    """Rank of the data parallel group. The runtime check at
    ``__post_init__`` further bounds this by ``data_parallel_size``."""
    # SOURCE: vllm/config/parallel.py:L139-L140 data_parallel_rank_local
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group, set only in SPMD mode."""
    # SOURCE: vllm/config/parallel.py:L141-L142 data_parallel_master_ip
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    # SUBTRACTED: data_parallel_rpc_port（L143-L144）——ray DP 面的控制端口。
    # SOURCE: vllm/config/parallel.py:L145-L146 data_parallel_master_port
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    # SOURCE: vllm/config/parallel.py:L147-L148 data_parallel_backend
    data_parallel_backend: DataParallelBackend = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    # SOURCE: vllm/config/parallel.py:L149-L155 data_parallel_external_lb（docstring
    #   原文长段逐字）
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    wide-EP setup in Kubernetes. Supported only for MoE deployments; non-MoE
    models should use independent vLLM instances without --data-parallel-*
    arguments. Set implicitly when --data-parallel-rank is provided explicitly
    to vllm serve."""
    # SOURCE: vllm/config/parallel.py:L156-L162 data_parallel_hybrid_lb
    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode."""
    # SOURCE: vllm/config/parallel.py:L163-L164 is_moe_model
    is_moe_model: bool | None = None
    """Whether the deployed model is MoE (if known)."""
    # SOURCE: vllm/config/parallel.py:L165-L166 enable_expert_parallel
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    # SUBTRACTED: enable_ep_weight_filter / enable_eplb / eplb_config /
    #   expert_placement_strategy（L167-L187）——EPLB 归 ch39。
    # SOURCE: vllm/config/parallel.py:L188-L198 all2all_backend（docstring 家族表）
    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """All2All backend for MoE expert parallel communication. Available options:

    - "allgather_reducescatter": All2all based on allgather and reducescatter
    - "deepep_high_throughput": Use deepep high-throughput kernels
    - "deepep_low_latency": Use deepep low-latitude kernels
    - "mori_high_throughput": MoRI EP with InterNodeV1 for multi-node
    - "mori_low_latency": MoRI EP with InterNodeV1LL for multi-node
    - "nixl_ep": Use nixl-ep kernels
    - "flashinfer_nvlink_two_sided": Use flashinfer two-sided kernels for mnnvl
    - "flashinfer_nvlink_one_sided": Use flashinfer high-throughput a2a kernels"""
    # SUBTRACTED: max_parallel_loading_workers（L200-L204）——多卡加载面（ch17）。
    # SOURCE: vllm/config/parallel.py:L205-L206 disable_custom_all_reduce —— 逐字
    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""
    # SUBTRACTED: enable_elastic_ep（L208-L209）——弹性 EP 全家按删除项 1 裁除。
    # SUBTRACTED: enable_dbo / ubatch_size / dbo_*_threshold 的阈值面（L211-L223）
    #   ——DBO 归 ch12/ch19（删除项 4）；dp_utils 消费的 num_ubatches 属性位保留
    #   （L211-L214 两字段逐字）：
    enable_dbo: bool = False
    """Enable dual batch overlap for the model executor."""
    ubatch_size: int = Field(default=0, ge=0)
    """Number of ubatch size."""
    # SOURCE: vllm/config/parallel.py:L227-L230 disable_nccl_for_dp_synchronization
    disable_nccl_for_dp_synchronization: bool | None = None
    """Forces the dp synchronization logic in vllm/v1/worker/dp_utils.py
    to use Gloo instead of NCCL for its all reduce."""
    # SUBTRACTED: assigned_physical_gpu_ids / numa 族（L230-L268）——ch17 域。
    # SOURCE: vllm/config/parallel.py:L270 master_addr
    master_addr: str = "127.0.0.1"
    # SOURCE: vllm/config/parallel.py:L273 master_port
    master_port: int = 29501
    # SOURCE: vllm/config/parallel.py:L276 node_rank
    node_rank: int = Field(default=0, ge=0)
    # SOURCE: vllm/config/parallel.py:L279 nnodes
    nnodes: int = Field(default=1, ge=1)
    # SOURCE: vllm/config/parallel.py:L243-L246 distributed_executor_backend ——
    #   类型面收窄：真实为 str | DistributedExecutorBackend | type[Executor] | None
    #   （Executor 类臂归 ch17 域、DistributedExecutorBackend 表 L37 已裁）；None
    #   的默认回填（L909-L941 的 ray/平台判定）归 ch17 域，运行面恒 "mp"。
    distributed_executor_backend: str | DataParallelBackend | None = None
    """Backend to use for distributed model running."""
    # SUBTRACTED: worker_cls / sd_worker_cls / worker_extension_cls / ray 族
    #   （L298-L309）——ch17 域。
    # SOURCE: vllm/config/parallel.py:L317-L322 distributed_timeout_seconds
    distributed_timeout_seconds: int | None = None
    """Timeout in seconds for distributed operations (e.g., init_process_group).
    If set, this value is passed to torch.distributed.init_process_group as the
    timeout parameter. If None, PyTorch's default timeout is used (600s for NCCL).
    Increase this for multi-node setups where model downloads may be slow."""
    # SOURCE: vllm/config/parallel.py:L323-L324 cpu_distributed_timeout_seconds
    cpu_distributed_timeout_seconds: int | None = None
    """Timeout (in seconds) for cpu communication groups."""
    # SOURCE: vllm/config/parallel.py:L327-L328 world_size
    world_size: int = Field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""
    # SOURCE: vllm/config/parallel.py:L330-L331 rank
    rank: int = 0
    """Global rank in distributed setup."""
    # SOURCE: vllm/config/parallel.py:L333-L337 _data_parallel_master_port_list
    _data_parallel_master_port_list: list[int] = Field(default_factory=list)
    """List of open port auto-queried for data parallel messaging.
    Set to be private as it's not intended to be configured by users."""
    # SOURCE: vllm/config/parallel.py:L338-L341 _coord_store_port
    _coord_store_port: int = 0
    """Port of the coordination TCPStore. Can be set by the API server; workers
    connect as clients to exchange self-picked group ports at runtime."""
    # SUBTRACTED: decode_context_parallel_size / dcp_* / cp_*（L342-L364）——DCP
    #   维度按删除项 1 裁除（建组段与 _DCP 访问器一并裁）。
    # SOURCE: vllm/config/parallel.py:L375-L377 data_parallel_index
    data_parallel_index: int = Field(init=False)
    """Equal to the data parallel rank but not used for torch process groups
    and not overridden for dense models."""
    # SUBTRACTED: _api_process_count/_api_process_rank（L379-L397，API 扩展内部
    #   面）、enable_fault_tolerance/fault_tolerance_config（L399-L406，FT 域）。

    # SOURCE: vllm/config/parallel.py:L562-L567 local_engines_only — 逐字
    @property
    def local_engines_only(self) -> bool:
        """
        Client manages local+remote EngineCores in pure internal LB case.
        Client manages local EngineCores in hybrid and external LB case.
        """
        # SOURCE: vllm/config/parallel.py:L562-L567（锚点双置）
        return self.data_parallel_external_lb or self.data_parallel_hybrid_lb

    # SOURCE: vllm/config/parallel.py:L569-L584 get_next_dp_init_port — 逐字（锚点双置）
    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        pop a new port from the prepared port list each time we need to
        initialize a new process group related to data parallelism.
        """
        if self._data_parallel_master_port_list:
            answer = self._data_parallel_master_port_list.pop()
        else:
            answer = self.data_parallel_master_port
            self.data_parallel_master_port += 1

        return answer

    # SOURCE: vllm/config/parallel.py:L586-L611 _pick_stateless_dp_port — 逐字（锚点双置）
    #   （coord store 分支保留；宿主测试走无 store 路径）
    def _pick_stateless_dp_port(self) -> tuple[int, socket.socket | None]:
        """Return ``(port, listen_socket)`` for DP group init.

        With a coord store, rank 0 binds a socket and publishes the port;
        others read it.  Without one, pops a pre-allocated port and
        returns ``listen_socket=None``.
        """
        # SOURCE: vllm/config/parallel.py:L586-L611（锚点双置）
        if not self._coord_store_port:
            return self.get_next_dp_init_port(), None

        from vllm.distributed.utils import get_cached_tcp_store_client

        store = get_cached_tcp_store_client(
            self.data_parallel_master_ip, self._coord_store_port
        )

        key = "dp_master_port"
        if self.data_parallel_rank == 0:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((self.data_parallel_master_ip, 0))
            s.listen()
            port = s.getsockname()[1]
            store.set(key, str(port).encode())
            return port, s
        else:
            return int(store.get(key).decode()), None

    @overload
    # SOURCE: vllm/config/parallel.py:L614-L616 stateless_init_dp_group 重载 1
    def stateless_init_dp_group(
        self, return_store: Literal[False] = ...
    ) -> ProcessGroup: ...
    @overload
    # SOURCE: vllm/config/parallel.py:L618-L620 stateless_init_dp_group 重载 2
    def stateless_init_dp_group(
        self, return_store: Literal[True] = ...
    ) -> tuple[ProcessGroup, Store]: ...
    # SOURCE: vllm/config/parallel.py:L621-L662 stateless_init_dp_group 本体 — 逐字（锚点双置）
    def stateless_init_dp_group(
        self, return_store: bool = False
    ) -> ProcessGroup | tuple[ProcessGroup, Store]:
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        from torch.distributed import DistNetworkError

        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        max_retries = 5
        last_exc: Exception | None = None
        for _ in range(max_retries):
            try:
                port, listen_socket = self._pick_stateless_dp_port()
                # use gloo since the engine process might not have cuda device
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    port,
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo",
                    return_store=return_store,
                    listen_socket=listen_socket,
                )
            except DistNetworkError as e:
                # We only want to retry when the root cause is EADDRINUSE.
                if "EADDRINUSE" in str(e):
                    logger.warning(
                        "Address already in use. Retrying with a new port."
                    )
                    last_exc = e
                    continue  # try again with a new port
                raise e

        # If we get here all retries have failed.
        assert last_exc is not None
        raise last_exc

    # The all_reduce at the end of attention (during o_proj) means that
    # inputs are replicated across each rank of the tensor parallel group.
    # If using expert-parallelism, replicated tokens results in useless
    # duplicate computation and communication.
    #
    # In this case, ensure the input to the experts is sequence parallel
    # to avoid the excess work.
    #
    # SOURCE: vllm/config/parallel.py:L673-L687 use_sequence_parallel_moe — 逐字
    @property
    def use_sequence_parallel_moe(self) -> bool:
        # SOURCE: vllm/config/parallel.py:L673-L687（锚点双置）
        return (
            self.all2all_backend
            in (
                "allgather_reducescatter",
                "deepep_high_throughput",
                "deepep_low_latency",
                "mori_high_throughput",
                "mori_low_latency",
                "nixl_ep",
            )
            and self.enable_expert_parallel
            and self.tensor_parallel_size > 1
            and self.data_parallel_size > 1
        )

    # SOURCE: vllm/config/parallel.py:L690-L695 use_all2all — 逐字
    @property
    def use_all2all(self) -> bool:
        # SOURCE: vllm/config/parallel.py:L690-L695（锚点双置）
        return (
            self.data_parallel_size > 1
            or self.use_sequence_parallel_moe
            or (self.enable_expert_parallel and self.prefill_context_parallel_size > 1)
        )

    # SUBTRACTED: use_batched_dp_moe（L698-L708）——DeepEP LL/nixl 专用批式面。

    # SOURCE: vllm/config/parallel.py:L710-L711 node_rank_within_dp
    @property
    def node_rank_within_dp(self) -> int:
        # SOURCE: vllm/config/parallel.py:L710-L711（锚点双置）
        return self.node_rank % self.nnodes_within_dp

    # SOURCE: vllm/config/parallel.py:L714-L720 nnodes_within_dp — 逐字
    @property
    def nnodes_within_dp(self) -> int:
        # SOURCE: vllm/config/parallel.py:L714-L720（锚点双置）
        if self.nnodes == 1:
            return 1
        data_parallel_node_size = (
            self.data_parallel_size // self.data_parallel_size_local
        )
        return self.nnodes // data_parallel_node_size

    # SOURCE: vllm/config/parallel.py:L723-L724 local_world_size
    @property
    def local_world_size(self) -> int:
        # SOURCE: vllm/config/parallel.py:L723-L724（锚点双置）
        return self.world_size // self.nnodes_within_dp

    # SOURCE: vllm/config/parallel.py:L727-L735 has_unfinished_dp — 逐字
    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        # SOURCE: vllm/config/parallel.py:L727-L735（锚点双置）
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    # SOURCE: vllm/config/parallel.py:L738-L762 sync_dp_state — 逐字
    @staticmethod
    def sync_dp_state(
        dp_group: ProcessGroup, has_unfinished: bool, pending_pause: bool
    ) -> tuple[bool, bool]:
        """Combined all-reduce for DP state synchronization.

        Uses a single SUM all-reduce on a 2-element tensor:
          [0] = 1 if this rank has unfinished work, else 0.
                SUM > 0 ≡ logical OR across ranks → any rank has work.
          [1] = 1 if this rank has a pending pause request, else 0.
                SUM == dp_size ≡ all ranks reached pause consensus.

        has_unfinished_global is true if any rank has unfinished work,
        or if some ranks are waiting for a pause consensus.

        Returns:
            (has_unfinished_global, pause_consensus)
        """
        # SOURCE: vllm/config/parallel.py:L738-L762（锚点双置）
        tensor = torch.tensor(
            [int(has_unfinished), int(pending_pause)], dtype=torch.int32, device="cpu"
        )
        torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=dp_group)
        dp_size = dp_group.size()
        pause_count = tensor[1].item()
        has_unfinished_global = tensor[0].item() > 0 or pause_count % dp_size != 0
        return has_unfinished_global, pause_count == dp_size

    # SUBTRACTED: sync_kv_cache_memory_size（L764-L772）——内存对齐面（ch14 域）。
    # SUBTRACTED: compute_hash（L774-L829）——DP 配置一致性 hash（跨章观测面）。

    # SOURCE: vllm/config/parallel.py:L549-L551 world_size_across_dp — 逐字
    @property
    def world_size_across_dp(self) -> int:
        """Process world size across TP, PCP, PP, and DP."""
        # SOURCE: vllm/config/parallel.py:L549-L551（锚点双置）
        return self.world_size * self.data_parallel_size

    # SUBTRACTED: use_ubatching（L554-L556）——DBO 域（删除项 4）。
    # SOURCE: vllm/config/parallel.py:L558-L559 num_ubatches — 逐字（dp_utils 消费位）
    @property
    def num_ubatches(self) -> int:
        # SOURCE: vllm/config/parallel.py:L558-L559（锚点双置）
        return 2 if self.enable_dbo else self.ubatch_size

    # SOURCE: vllm/config/parallel.py:L831-L837 __post_init__ 世界尺寸段 + L905 的
    #   data_parallel_index 回填 — 逐字（中间校验族按 SUBTRACTED 裁除）
    def __post_init__(self) -> None:
        # Continue with the rest of the initialization
        # SOURCE: vllm/config/parallel.py:L831-L988（锚点双置）
        self.world_size = (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.prefill_context_parallel_size
        )

        # SUBTRACTED: external_launcher 的 world_size×dp（L839-L841）、elastic EP
        #   校验族（L843-L859）、offline DP 校验（L896-L901）与其余 post_init
        #   校验（L475-L497 等）——各归其域。
        self.data_parallel_index = self.data_parallel_rank
        # SUBTRACTED: distributed_executor_backend 的 env/默认回填（L907-L912+）。

