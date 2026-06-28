"""Subtract-only companion — vLLM 侧平台抽象基类与平台枚举。

规范源码：vllm/platforms/interface.py（PlatformEnum / Platform）
         vllm/v1/attention/backends/registry.py（AttentionBackendEnum）

NPUPlatform 继承本文件的 Platform、把 _enum 设为 OOT；基类给出 get_*_cls 钩子的
通用默认实现（返回 vLLM 自带后端的 qualname），子类逐一覆盖成昇腾实现。
"""
import enum
from typing import Any


# SOURCE: vllm/platforms/interface.py:L38-L47
class PlatformEnum(enum.Enum):
    """Enumeration of supported hardware platforms."""

    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


# SOURCE: vllm/v1/attention/backends/registry.py:L34-L44
class AttentionBackendEnum(enum.Enum):
    # SUBTRACTED: 原枚举有 FLASH_ATTN_DIFFKV / FLASH_ATTN_MLA … 数十个成员
    #   （原 vllm/v1/attention/backends/registry.py:L44-L120），本章只需 FLASH_ATTN
    #   这一个用于 get_attn_backend_cls 的特例判定。
    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"


# SOURCE: vllm/platforms/interface.py:L105-L774
class Platform:
    _enum: PlatformEnum
    device_name: str
    device_type: str
    # SUBTRACTED: 基类还声明了 dispatch_key / simple_compile_backend / ray_device_key …
    #   以及大量需真硬件的 classmethod（内存查询/device 上下文/通信等）。本章只保留
    #   ‘平台身份判定’与 ‘返回 qualname 的工厂钩子’这两族，其余原 interface.py:L105-L1010 省略。
    simple_compile_backend: str = "inductor"

    # SOURCE: vllm/platforms/interface.py:L178-L179
    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    @classmethod
    def get_pass_manager_cls(cls) -> str:
        # SOURCE: vllm/platforms/interface.py:L199-L204
        """
        Get the pass manager class for this platform.
        """
        return "vllm.compilation.passes.pass_manager.PostGradPassManager"

    @classmethod
    def get_compile_backend(cls) -> str:
        # SOURCE: vllm/platforms/interface.py:L206-L211
        """
        Get the custom compile backend for current platform.
        """
        return cls.simple_compile_backend

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # SOURCE: vllm/platforms/interface.py:L741-L745
        """
        Return the punica wrapper for current platform.
        """
        raise NotImplementedError

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # SOURCE: vllm/platforms/interface.py:L769-L774
        """
        Get device specific communicator class for distributed communication.
        """
        return "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"  # noqa

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        # SOURCE: vllm/platforms/interface.py:L885-L890
        """
        Get static graph wrapper class for static graph.
        """
        return "vllm.compilation.base_static_graph.AbstractStaticGraphWrapper"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config,
                             num_heads: int | None = None) -> str:
        # SOURCE: vllm/platforms/interface.py:L248-L256
        # SUBTRACTED: 基类默认实现（依平台返回 vLLM 自带 attention backend）。本章焦点是
        #   NPUPlatform 的覆盖版（vllm_ascend_platform.py），基类只留签名作对照。
        raise NotImplementedError
