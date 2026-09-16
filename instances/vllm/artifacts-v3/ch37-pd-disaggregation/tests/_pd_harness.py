# ch37 测试脚手架：把「两台引擎」装配到同一个进程里。
#
# 保留的边界（与真实部署同构）：
#   * 调度器侧 connector ↔ worker 侧 connector 只经 KVConnectorMetadata 通讯
#     （build_connector_meta → worker.start_load_kv → get_finished → KVConnectorOutput）；
#   * 带外握手走**真 ZMQ 套接字 + 真 msgspec 编解码**（P 侧 ROUTER 监听、D 侧 REQ）；
#   * 端口/主机经 vllm.envs 覆盖，两台引擎各有各的 side channel。
#
# 唯一被替换的是 NIXL 数据面本身（host 无 RDMA/NIXL，见 implementation/
# vllm/distributed/nixl_utils.py 的 seam 说明）。
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_IMPL = Path(__file__).resolve().parents[1] / "implementation"
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))

from vllm import envs  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.kv_transfer import KVTransferConfig  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

BLOCK_SIZE = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 8
NUM_BLOCKS = 8
LAYER_NAMES = ["layer.0.attn"]


def make_kv_config(
    num_blocks: int = NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    num_kv_heads: int = NUM_KV_HEADS,
    head_size: int = HEAD_SIZE,
    layer_names: list[str] | None = None,
) -> KVCacheConfig:
    """HND 布局：cache 形状 [num_blocks, num_kv_heads, block_size, 2*head_size]。"""
    layer_names = list(layer_names or LAYER_NAMES)
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float32,
    )
    tensor_bytes = spec.page_size_bytes * num_blocks
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_bytes, shared_by=list(layer_names))
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)
        ],
    )


def make_kv_caches(kv_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """按 kv_cache_config 的页几何分配真 KV 张量（HND：K/V 打包进内容维）。"""
    caches: dict[str, torch.Tensor] = {}
    for group in kv_config.kv_cache_groups:
        spec = group.kv_cache_spec
        shape = (
            kv_config.num_blocks,
            spec.num_kv_heads,
            spec.block_size,
            2 * spec.head_size,
        )
        for name in group.layer_names:
            caches[name] = torch.zeros(shape, dtype=spec.dtype)
    return caches


def make_vllm_config(
    *,
    engine_id: str,
    kv_role: str | None,
    connector: str | None = "NixlConnector",
    tp_size: int = 1,
    block_size: int = BLOCK_SIZE,
    extra_config: dict | None = None,
    kv_buffer_device: str = "cpu",
) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(
            model="dummy",
            dtype="float32",
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            use_mla=False,
        ),
        cache_config=CacheConfig(block_size=block_size, cache_dtype="auto"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size, pipeline_parallel_size=1
        ),
        scheduler_config=SchedulerConfig(),
        kv_transfer_config=KVTransferConfig(
            kv_connector=connector,
            kv_role=kv_role,
            engine_id=engine_id,
            kv_buffer_device=kv_buffer_device,
            kv_connector_extra_config=dict(extra_config or {}),
        ),
    )


class StubModelExecutor:
    """真实里是 MultiprocExecutor：本章只需要它交出各 worker 的握手元数据。"""

    def __init__(self, worker_connectors):
        self.worker_connectors = list(worker_connectors)
        self.kv_output_aggregator = None

    def get_kv_connector_handshake_metadata(self):
        return [
            {(0, 0): wc.get_handshake_metadata()} for wc in self.worker_connectors
        ]

    def init_kv_output_aggregator(self, connector) -> None:
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator

        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, len(self.worker_connectors)
        )


class Engine:
    """一台引擎的本章切面：EngineCore（含调度器侧 connector）+ 各 worker 侧 connector。

    真实部署里这两半在两个进程；这里同进程装配，但**只经 metadata 交换**，
    因此调用序列与真实引擎一致（EngineCore 启动装配亦走真源码路径）。
    """

    def __init__(
        self,
        name: str,
        *,
        kv_role: str,
        connector: str = "NixlConnector",
        port: int,
        host: str = "127.0.0.1",
        num_blocks: int = NUM_BLOCKS,
        block_size: int = BLOCK_SIZE,
        tp_size: int = 1,
        extra_config: dict | None = None,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        from vllm.v1.engine.core import EngineCore

        self.name = name
        self.engine_id = f"engine-{name}"
        self.kv_config = make_kv_config(num_blocks=num_blocks, block_size=block_size)
        self.kv_caches = make_kv_caches(self.kv_config)
        self.vllm_config = make_vllm_config(
            engine_id=self.engine_id,
            kv_role=kv_role,
            connector=connector,
            tp_size=tp_size,
            block_size=block_size,
            extra_config=extra_config,
        )
        ConfigCls = KVConnectorFactory.get_connector_class(
            self.vllm_config.kv_transfer_config
        )
        self.worker_connector = ConfigCls(
            self.vllm_config, KVConnectorRole.WORKER, self.kv_config
        )
        self.worker_connector.register_kv_caches(self.kv_caches)
        # 端口在 Scheduler 侧 connector 的 __init__ 里读 envs，故构造前先覆写。
        with side_channel(host, port):
            self.core = EngineCore(
                self.vllm_config,
                self.kv_config,
                model_executor=StubModelExecutor([self.worker_connector]),
            )
        self.scheduler_connector = self.core.scheduler.connector
        sc = self.scheduler_connector.connector_scheduler
        assert sc is not None and sc.side_channel_port == port and sc.side_channel_host == host

    @property
    def scheduler(self):
        """调度器侧实体（NixlPull/PushConnectorScheduler）。"""
        return self.scheduler_connector.connector_scheduler

    @property
    def worker(self):
        """worker 侧实体（NixlPull/PushConnectorWorker）。"""
        return self.worker_connector.connector_worker

    # ---- 引擎步进（真实引擎里由 model runner 逐 step 驱动） ----

    def build_meta(self, scheduler_output=None):
        """调度器步：build_connector_meta 会清空四账本，测试需显式调用一次。"""
        from vllm.v1.core.sched.output import SchedulerOutput

        return self.scheduler_connector.build_connector_meta(
            scheduler_output or SchedulerOutput()
        )

    def worker_step(self, metadata=None):
        """worker 步：吃一份 metadata，跑收发，回报完成集合。"""
        if metadata is None:
            metadata = self.build_meta()
        self.worker_connector._connector_metadata = metadata
        self.worker_connector.start_load_kv(metadata)
        self.worker_connector.wait_for_save()
        return self.worker_connector.get_finished(set())

    def close(self) -> None:
        self.worker_connector.shutdown()
        self.scheduler_connector.shutdown()


def port_for(index: int) -> int:
    """给两台（或多台）引擎分配互不冲突的 side channel 端口。"""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class side_channel:
    """上下文管理器：把 envs 的 side channel 指向本引擎的 host/port。"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._saved = None

    def __enter__(self):
        self._saved = (
            envs.VLLM_NIXL_SIDE_CHANNEL_HOST,
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT,
        )
        envs.VLLM_NIXL_SIDE_CHANNEL_HOST = self.host
        envs.VLLM_NIXL_SIDE_CHANNEL_PORT = self.port
        return self

    def __exit__(self, *exc):
        envs.VLLM_NIXL_SIDE_CHANNEL_HOST, envs.VLLM_NIXL_SIDE_CHANNEL_PORT = self._saved


def make_engine(name: str, *, kv_role: str, connector: str = "NixlConnector", **kw) -> Engine:
    port = kw.pop("port", None) or port_for(0)
    host = kw.pop("host", "127.0.0.1")
    with side_channel(host, port):
        return Engine(name, kv_role=kv_role, connector=connector, port=port, host=host, **kw)


def pump_until(fn, timeout: float = 5.0, interval: float = 0.01):
    """轮询等待（握手走后台线程 + 真 ZMQ，必然有一小段异步窗口）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(interval)
    return fn()


def fill_blocks(caches: dict[str, torch.Tensor], block_ids, value: float) -> None:
    for cache in caches.values():
        for b in block_ids:
            cache[b].fill_(value)


def block_checksum(caches: dict[str, torch.Tensor], block_id: int) -> float:
    total = 0.0
    for cache in caches.values():
        total += float(cache[block_id].sum())
    return total
