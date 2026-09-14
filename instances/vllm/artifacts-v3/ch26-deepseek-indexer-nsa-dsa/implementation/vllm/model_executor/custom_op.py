# SOURCE: vllm/model_executor/custom_op.py
# ch26 消费面：CustomOp 基类（SparseAttnIndexer 的派发骨架——
# @CustomOp.register("sparse_attn_indexer") + dispatch_forward 的平台派发 +
# forward → _forward_method 的调用面）+ PluggableLayer（mla.py:L35 的
# @PluggableLayer.register("multi_head_latent_attention") 注册位——ch25 同款）。
# HOST SEAM：host（is_cpu）派发到 forward_cpu→forward_native——真实 CPU 平台
# 同型；enforce/compile_native 面按真实默认。
from __future__ import annotations

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# SOURCE: vllm/model_executor/custom_op.py:L60 op_registry —— 逐字位
op_registry: dict = {}
# SOURCE: vllm/model_executor/custom_op.py op_registry_oot —— 逐字位
op_registry_oot: dict = {}


# SOURCE: vllm/model_executor/custom_op.py PluggableLayer —— 减法子集
class PluggableLayer(torch.nn.Module):
    """A layer that can be replaced by a custom operation from an OOT plugin.

    Framework code instantiates these layers as placeholders. At import time,
    an OOT plugin can call `MyLayer.register("name")` to register a
    replacement class. The model loader checks the registry and uses the
    replacement class instead of the placeholder.
    """

    @staticmethod
    def register(name: str):
        # SOURCE: vllm/model_executor/custom_op.py PluggableLayer.register
        def decorator(cls):
            op_registry[name] = cls
            return cls

        return decorator


# SOURCE: vllm/model_executor/custom_op.py:L103-L201 CustomOp —— 减法子集
#   （__new__/__init__/forward/forward_* 家族/dispatch_forward 的平台派发；
#   OOT 派发位与环境开关面按消费承载）
class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    name: str = ""

    def __new__(cls, *args, **kwargs):
        # SOURCE: vllm/model_executor/custom_op.py:L115-L133 __new__ —— 逐字
        #   （OOT 覆盖类的实例化位）
        try:
            op_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@CustomOp.register, or it's the CustomOp base class itself."
            ) from None

        if op_name not in op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            op_cls_to_instantiate = op_registry_oot[op_name]
            logger.debug(
                "Instantiating custom op: %s using %s",
                op_name,
                str(op_cls_to_instantiate),
            )
        return super().__new__(op_cls_to_instantiate)

    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
        # SOURCE: vllm/model_executor/custom_op.py:L135-L139 __init__ —— 逐字
        super().__init__()
        self._enforce_enable = enforce_enable
        self._forward_method = self.dispatch_forward(compile_native=compile_native)

    def forward(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/custom_op.py:L141-L142 forward —— 逐字
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        # SOURCE: vllm/model_executor/custom_op.py:L138-L149 —— 逐字
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/custom_op.py:L146-L147 —— 逐字
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        # SOURCE: vllm/model_executor/custom_op.py:L151-L153 —— 逐字
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        # By default, we assume that XPU ops are compatible with the
        # PyTorch-native implementation.
        # SOURCE: vllm/model_executor/custom_op.py:L155-L158 —— 逐字
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with the
        # PyTorch-native implementation.
        # SOURCE: vllm/model_executor/custom_op.py:L160-L163 —— 逐字
        return self.forward_native(*args, **kwargs)

    def forward_oot(self, *args, **kwargs):
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        # SOURCE: vllm/model_executor/custom_op.py:L165-L170 —— 逐字
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self, compile_native: bool):
        # SOURCE: vllm/model_executor/custom_op.py:L174-L200 dispatch_forward
        #   —— 减法子集（enabled_custom_ops/disabled_custom_ops 记账 + 平台
        #   派发；maybe_compile 直通位）
        from vllm.config import get_cached_compilation_config
        from vllm.platforms import current_platform

        compilation_config = get_cached_compilation_config()

        enabled = self._enforce_enable or self.enabled()
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])
        else:
            compilation_config.disabled_custom_ops.update([self.__class__.name])

        if not enabled:
            # Compile forward_native to avoid eager torch ops if inside
            # opaque torch custom op (e.g., fused_moe, unified_attention, etc.)
            return self.maybe_compile(self.forward_native, enable=compile_native)

        if current_platform.is_rocm():
            return self.forward_hip
        elif current_platform.is_cpu():
            return self.forward_cpu
        elif current_platform.is_xpu():
            return self.forward_xpu
        elif current_platform.is_out_of_tree():
            return self.forward_oot
        else:
            return self.forward_cuda

    def maybe_compile(self, fn, *, enable: bool = True):
        """Compile fn if compilation enabled.
        Useful for CustomOp instances called from within a torch custom op,
        meaning the forward call is hidden from the model-level torch.compile.

        NOTE: this does not enable fusion across ops, so opaque torch ops
        (e.g., torch ops with a "no compile" tag) are not affected.
        """
        # SOURCE: vllm/model_executor/custom_op.py maybe_compile —— HOST SEAM
        #   直通（torch.compile 面 → ch19 域）
        return fn

    @classmethod
    def enabled(cls) -> bool:
        # SOURCE: vllm/model_executor/custom_op.py:L272-L280 enabled —— 减法
        #   子集：host 无环境显式开启 → False（真实默认_disabled 形态）
        from vllm.platforms import current_platform

        if current_platform.is_out_of_tree():
            return False
        return False

    @classmethod
    def register(cls, name: str, default_on: bool = False):
        # SOURCE: vllm/model_executor/custom_op.py:L315-L337 register —— 逐字
        #   主干（名字登记 + default_on 位）
        # SOURCE: vllm/model_executor/custom_op.py:L325-L332（锚点双置）
        def decorator(op_cls):
            if not issubclass(op_cls, CustomOp):
                raise ValueError(f"{op_cls.__name__} must inherit from CustomOp")
            if name in op_registry:
                raise ValueError(f"Custom op with name {name} already exists")
            op_cls.name = name
            op_registry[name] = op_cls
            op_cls._default_on = default_on
            return op_cls

        return decorator
