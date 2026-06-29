# vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py
#   —— subtract-only companion（★ 后端代表实现：把抽象契约落到 Mooncake 分布式 store）
#
# 一个 Backend 子类讲透契约怎么落地：exists→batch_is_exist、put→batch_put_from_multi_buffers、
#   get→batch_get_into_multi_buffers（多 buffer 批量 RDMA 读写外部池）。setup() 经 transfer
#   engine 连 metadata_server/master_server。
#
# 注意：本文件依赖 mooncake / torch_npu / mooncake_transfer_engine —— host 无 NPU/CANN 不可发车，
#   测试不导入它（用纯内存 Backend 替身验契约调用顺序）。此处只读其控制流（put/get/exists 落点）。
import json
import os
from dataclasses import dataclass
from typing import Any

import torch

# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824  # 1.0 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB

# SUBTRACTED: _mooncake_setup_supports_ssd_offload / _ssd_setup_kwargs（L26-L58）——
#   SSD offload 旁路（v0.3.11+ 才有），正交于 put/get/exists 契约。原 mooncake_backend.py:L26-L58


class MooncakeBackend(Backend):  # SOURCE: backend/mooncake_backend.py:L61
    def __init__(self, parallel_config: ParallelConfig, lazy_init: bool = False):  # SOURCE: mooncake_backend.py:L62
        self.config = MooncakeStoreConfig.load_from_env()
        if self.config.protocol != "ascend":
            raise NotImplementedError(f"MooncakeBackend does not support protocol {self.config.protocol!r}.")
        self.store: Any | None = None
        self.local_seg: str | None = None
        # SUBTRACTED: _use_fabric_mem / _lazy_init(DSV4 compress 惰性建 store) 三态与锁（L69-L72）——
        #   DSV4 专属优化；本主线直接同步建 store。原 mooncake_backend.py:L69-L89
        self.store = self._setup_store()

    def _setup_store(self):  # SOURCE: backend/mooncake_backend.py:L90
        from mooncake.store import MooncakeDistributedStore  # type: ignore

        store = MooncakeDistributedStore()
        local_hostname = get_ip()
        # SUBTRACTED: SSD per-rank 目录建立 + ASCEND_ENABLE_USE_FABRIC_MEM 统一内存直传分支
        #   （L102-L142 的 ssd_kwargs / fabric_mem 二选一）——保留默认 transfer-engine 连接路径。
        transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
        self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
        ret = store.setup(
            local_hostname=self.local_seg,
            metadata_server=self.config.metadata_server,
            global_segment_size=self.config.global_segment_size,
            local_buffer_size=self.config.local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
            engine=transfer_engine.get_engine(),
        )
        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error("Initialize mooncake failed. ret=%d", ret)
            raise RuntimeError(msg)
        return store

    def set_device(self):  # SOURCE: backend/mooncake_backend.py:L159
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):  # SOURCE: backend/mooncake_backend.py:L164
        # SUBTRACTED: if not self._use_fabric_mem 守卫（L165）——统一内存模式下跳过显式注册；
        #   主线走 transfer engine 注册显存区段。原 mooncake_backend.py:L164-L168
        local_hostname = get_ip()
        global_te.get_transfer_engine(local_hostname, device_name=None)
        global_te.register_buffer(ptrs, lengths)

    def exists(self, keys: list[str]) -> list[int]:  # SOURCE: backend/mooncake_backend.py:L170
        # SUBTRACTED: lazy_init 未建 store 时返回全 0（L171-L176）——DSV4 惰性路径容错。
        assert self.store is not None
        return self.store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):  # SOURCE: mooncake_backend.py:L180
        try:
            # SUBTRACTED: self._ensure_initialized()(lazy_init 首次建 store)（L182）——主线已同步建好。
            assert self.store is not None
            config = ReplicateConfig()
            if self.config.preferred_segment:
                config.preferred_segment = self.local_seg
            config.prefer_alloc_in_same_node = self.config.prefer_alloc_in_same_node
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
            failed_codes = [int(value) for value in res if value < 0]
            if failed_codes:
                logger.error("Failed to put %d keys out of %d.", len(failed_codes), len(keys))
        except Exception as e:
            logger.error("Failed to put %d keys. error=%s", len(keys), type(e).__name__)

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):  # SOURCE: mooncake_backend.py:L217
        # SUBTRACTED: lazy_init 未建 store 时报错返回（L218-L226）——DSV4 惰性路径。
        assert self.store is not None
        try:
            res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
            res_list = list(res)
            failed_codes = [int(value) for value in res_list if value < 0]
            if failed_codes:
                logger.error("Failed to get %d keys out of %d.", len(failed_codes), len(keys))
            # 返回值约定：get 把正数(读到的字节数)归一成 0=成功，负数=失败块。
            for i, value in enumerate(res_list):
                if value > 0:
                    res_list[i] = 0
            return res_list
        except Exception as e:
            logger.error("Failed to get %d keys. error=%s", len(keys), type(e).__name__)
            return None


@dataclass
class MooncakeStoreConfig:  # SOURCE: backend/mooncake_backend.py:L266
    metadata_server: str
    global_segment_size: int | str
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    preferred_segment: bool
    prefer_alloc_in_same_node: bool
    # SUBTRACTED: enable_ssd_offload / ssd_offload_path 字段 + __post_init__ 校验（L276-L287）——SSD 旁路。

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":  # SOURCE: backend/mooncake_backend.py:L289
        with open(file_path) as file:
            config = json.load(file)
        master_server_address = os.getenv("MOONCAKE_MASTER", None)
        # SUBTRACTED: _parse_global_segment_size 的 "1GB"/"512MB" 单位解析（L322-L392）——
        #   尺寸字符串解析；主线直接读 int 字节数。原 mooncake_backend.py:L297-L302
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server"),
            global_segment_size=int(config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)),
            local_buffer_size=int(config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            protocol=config.get("protocol", "ascend"),
            device_name=config.get("device_name", ""),
            master_server_address=master_server_address
            if master_server_address is not None
            else config.get("master_server_address"),
            preferred_segment=config.get("preferred_segment", False),
            prefer_alloc_in_same_node=config.get("prefer_alloc_in_same_node", True),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":  # SOURCE: backend/mooncake_backend.py:L314
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)
