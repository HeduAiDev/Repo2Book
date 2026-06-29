# vllm/v1/attention/backend.py —— subtract-only 精简版（vLLM 侧后端契约）
#
# 本章对账对象：AttentionBackend 抽象基类。任何后端（含昇腾 OOT 后端）要接进 vLLM
# 注意力框架，必须实现这里的 4 个 @abstractmethod：
#   get_name / get_impl_cls / get_builder_cls / get_kv_cache_shape
# 另有 get_supported_kernel_block_sizes —— 基类自带默认实现（[MultipleOf(1)]），
# 后端可覆写（昇腾覆写返回 [128]）但并非 @abstractmethod。
#
# 纯 Python ABC，可在 host 跑：实例化一个缺方法的子类会抛 TypeError，正是契约的体现。
from abc import ABC, abstractmethod
from typing import ClassVar

import torch

# SUBTRACTED: AttentionType 枚举与大量 quant/config 相关 import（backend.py:L4-L45）——
#   与本章「契约的 4 个 @abstractmethod」正交，折叠。


# SOURCE: vllm/v1/attention/backend.py:L48-L52
class MultipleOf:
    base: int

    def __init__(self, base: int):
        # SOURCE: vllm/v1/attention/backend.py:L51-L52
        self.base = base


# SOURCE: vllm/v1/attention/backend.py:L55-L96
class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # Does attention's forward() include kv cache update?
    forward_includes_kv_cache_update: bool = True

    # 基类自带默认实现、后端可覆写（昇腾覆写返回 [128]）；非 @abstractmethod。
    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        # SOURCE: vllm/v1/attention/backend.py:L68-L70
        return [MultipleOf(1)]

    # ↓↓↓ 以下 4 个是 @abstractmethod —— OOT 后端「必须实现」的硬性契约点 ↓↓↓
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backend.py:L72-L75
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type:
        # SOURCE: vllm/v1/attention/backend.py:L77-L80
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:
        # SOURCE: vllm/v1/attention/backend.py:L82-L85
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backend.py:L87-L96
        raise NotImplementedError

    # SUBTRACTED: get_kv_cache_block_dim / get_kv_cache_stride_order /
    #   get_required_kv_cache_layout / supports_dtype 等一众带默认实现的 @classmethod
    #   （backend.py:L98+）—— 本章只锁 4 个 @abstractmethod + get_supported_kernel_block_sizes，
    #   其余默认方法与「后端选择/契约对账」立意正交，折叠。
    #
    # 注意：基座 vllm/v1/attention/ 全目录里**没有** swap_blocks/copy_blocks 的 def，也无任何
    #   调用——它们只在昇腾 attention_v1.py 出现，是昇腾自带的 v0 遗留接口，**非** v1 基类契约。
