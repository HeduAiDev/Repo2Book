# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""注意力后端契约（本章只用到身份查询与 KV 形状探测）。

# SOURCE: vllm/v1/attention/backend.py:L1-L200（AttentionBackend / AttentionMetadata 协议）
# SUBTRACTED: 全部真实后端（FlashAttention/FlashInfer/Triton/MLA/CPU_ATTN...）与
#   builder 协议——宿主没有加速器；本章只需要「后端名进兼容 hash」「KV 形状进
#   区域几何推断」这两件事（metadata.py:L122 / kv_connector/utils.py:L427-L446）。
"""

from dataclasses import dataclass
from typing import Any


# SOURCE: vllm/v1/attention/backend.py:L49-L60
@dataclass
class MultipleOf:
    # SOURCE: vllm/v1/attention/backend.py:L49-L60
    base: int


# SOURCE: vllm/v1/attention/backend.py:L56-L200
class AttentionBackend:
    """微缩契约：本章消费 get_name / full_cls_name / get_kv_cache_shape。"""

    # SOURCE: vllm/v1/attention/backend.py:L75-L88
    @classmethod
    def get_name(cls) -> str:
        # SOURCE: vllm/v1/attention/backend.py:L75-L88
        """后端短名（进 NixlAgentMetadata.attn_backend_name 与兼容 hash）。"""
        raise NotImplementedError

    # SOURCE: vllm/v1/attention/backend.py:L151-L165
    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        # SOURCE: vllm/v1/attention/backend.py:L151-L165
        """(模块, 类名)——用于去重与日志。"""
        return (cls.__module__, cls.__name__)

    # SOURCE: vllm/v1/attention/backend.py:L90-L120
    @classmethod
    def get_kv_cache_shape(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backend.py:L90-L120
        """KV cache 张量形状（本章的 HND 契约：blocks-first，K/V 打包进内容维）。"""
        raise NotImplementedError


# SOURCE: vllm/v1/attention/backend.py:L404-L430
class AttentionMetadata:
    """本章不跑前向，这个类型只用于签名（save_kv_layer 的 no-op 参数）。"""


# SOURCE: vllm/v1/attention/backends/utils.py:L83-L110（get_kv_cache_layout）
_KV_CACHE_LAYOUT: str = "HND"


# SOURCE: vllm/v1/attention/backends/utils.py:L83-L110（get_kv_cache_layout）
def get_kv_cache_layout() -> str:
    # SOURCE: vllm/v1/attention/backends/utils.py:L83-L110（get_kv_cache_layout）
    """当前 KV 缓存布局：NixlConnector 强制 HND 以换取传输性能。

    源码里它由 connector 的 `get_required_kvcache_layout` 在引擎启动时设定
    （kv_connector/utils.py:L37-L50）；本章直接取该结果。
    """
    return _KV_CACHE_LAYOUT


# SOURCE: vllm/v1/attention/selector.py:L101-L160（真后端选择）
class HostAttentionBackend(AttentionBackend):
    """宿主替身后端：无真实 kernel，只提供名字与形状。

    # SEAM: 真实源码经 `get_attn_backend()` 从平台插件里挑 FlashAttention /
    #   FlashInfer / Triton 等真后端；宿主无加速器，本章取一个名字固定的替身。
    #   偏离面：不产生任何 attention 计算——但 connector 只读它的**名字**与
    #   **KV 形状**（HND: [num_blocks, num_kv_heads, block_size, 2*head_size]），
    #   二者与真后端在本章配置下一致。
    """

    # SOURCE: vllm/v1/attention/selector.py:L101-L160（真后端选择）
    @classmethod
    def get_name(cls) -> str:
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L95-L110
        return "FLASH_ATTN"

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L134-L160
    @classmethod
    def get_kv_cache_shape(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L134-L160
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L83-L93
    # SEAM: 真 FlashAttention 只接受 16 的倍数（[MultipleOf(16)]），故逻辑块尺寸
    #   与 kernel 块尺寸可能不等（`_physical_blocks_per_logical_kv_block` > 1）；
    #   宿主替身后端接受任意块尺寸，该比值恒 1（对称 TP 正典路径）。
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L83-L93
    @classmethod
    def get_supported_kernel_block_sizes(cls) -> list:
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L83-L93
        return [MultipleOf(1)]


# SOURCE: vllm/v1/attention/selector.py:L101-L160
def get_attn_backend(
    head_size: int,
    dtype: Any,
    kv_cache_dtype: str,
    use_mla: bool = False,
) -> type[AttentionBackend]:
    """本章恒返回宿主替身后端（源码按平台/模型挑真实后端）。"""
    return HostAttentionBackend


__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "HostAttentionBackend",
    "MultipleOf",
    "get_attn_backend",
    "get_kv_cache_layout",
]
