# vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py
#   —— subtract-only companion（★ 可插拔后端契约：整套池化后端的全部 6 个方法）
#
# 这是整个 worker 侧搬运代码唯一依赖的抽象：注册显存区段(register_buffer)、按 key 查在不在
#   (exists)、把 (addr,size) 区段写入池(put)、把池里数据读回 (addr,size)(get)。换后端 = 换一个
#   实现类（mooncake/memcache/yuanrong…），调度/搬运代码一行不改。本文件与真实源码逐字一致。
from abc import ABC, abstractmethod

# SUBTRACTED: from vllm.config import ParallelConfig —— host 无 vllm，类型由 runtime_stub 接住。
#   原 vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py:L3
from runtime_stub import ParallelConfig


class Backend(ABC):  # SOURCE: backend/backend.py:L6
    @abstractmethod
    def __init__(self, parallel_config: ParallelConfig):  # SOURCE: backend/backend.py:L8
        pass

    @abstractmethod
    def set_device(self):  # SOURCE: backend/backend.py:L12
        pass

    @abstractmethod
    def register_buffer(self, ptrs: list[int], lengths: list[int]):  # SOURCE: backend/backend.py:L16
        pass

    @abstractmethod
    def exists(self, keys: list[str]) -> list[int]:  # SOURCE: backend/backend.py:L20
        pass

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):  # SOURCE: backend/backend.py:L24
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):  # SOURCE: backend/backend.py:L28
        pass
