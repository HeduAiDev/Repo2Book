# SOURCE: vllm/model_executor/custom_op.py
# ch23 消费面：CustomOp（RMSNorm/SiluAndMul/RotaryEmbedding 的派发基类）+
# PluggableLayer（@PluggableLayer.register("logits_processor") 的注册点——
# m13 的 v0.27 新面：OOT 平台可换层）。派发主干逐字；maybe_compile 的
# torch.compile 包装 → ch19 编译域。
from __future__ import annotations

import torch
import torch.nn as nn

from vllm.config import get_cached_compilation_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# SOURCE: vllm/model_executor/custom_op.py:L15-L22 注册表声明（逐字）
# Dictionary of all custom ops (classes, indexed by registered name).
# To check if an op with a name is enabled, call .enabled() on the class.
# Examples:
# - MyOp.enabled()
# - op_registry["my_op"].enabled()
op_registry: dict[str, type["CustomOp"] | type["PluggableLayer"]] = {}
op_registry_oot: dict[str, type["CustomOp"] | type["PluggableLayer"]] = {}


# SOURCE: vllm/model_executor/custom_op.py:L25-L28 maybe_get_oot_by_class（逐字）
def maybe_get_oot_by_class(class_type: type) -> type:
    class_name = class_type.__name__
    if class_name in op_registry_oot:
        return op_registry_oot[class_name]
    return class_type


# SOURCE: vllm/model_executor/custom_op.py:L32 PluggableLayer
class PluggableLayer(nn.Module):
    """Base class for pluggable layers.

    A PluggableLayer is a *module-composing* abstraction: it may instantiate other
    ``torch.nn.Module`` objects as sub-layers, and its functionality depends on
    these sub-layers following a generalized invocation sequence. Also, it is stateful
    and may hold parameters or buffers.

    Unlike :class:`CustomOp`, PluggableLayer does NOT provide per-platform
    ``forward_*`` dispatch. Instead, it supports out-of-tree (OOT) replacement
    of the entire layer class at instantiation time, allowing customized
    initialization and submodule composition.
    """

    # SOURCE: vllm/model_executor/custom_op.py:L47-L71 PluggableLayer.__new__（逐字）
    def __new__(cls, *args, **kwargs):
        try:
            layer_class_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@PluggableLayer.register, or it's the PluggableLayer itself."
            ) from None

        if layer_class_name not in op_registry_oot:
            layer_cls_to_instantiate = cls
        else:
            layer_cls_to_instantiate = op_registry_oot[layer_class_name]
            logger.debug(
                "Instantiating pluggable layer: %s using %s",
                layer_class_name,
                str(layer_cls_to_instantiate),
            )
        return super().__new__(layer_cls_to_instantiate)

    # SOURCE: vllm/model_executor/custom_op.py:L69-L77 PluggableLayer.register
    #   （逐字——in-tree 注册面；logits_processor 经此进 op_registry）
    @classmethod
    def register(cls, name: str):
        # SOURCE: vllm/model_executor/custom_op.py:L69-L77 PluggableLayer.register
        def decorator(op_cls):
            # SOURCE: vllm/model_executor/custom_op.py:L69-L77 PluggableLayer.register
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_registry[name] = op_cls
            return op_cls

        return decorator

    # SOURCE: vllm/model_executor/custom_op.py:L83-L98 PluggableLayer.register_oot
    #   （逐字——OOT 平台后门；v0.27 新面 m13）
    @classmethod
    def register_oot(cls, _decorated_layer_cls=None, name: str | None = None):
        # SOURCE: vllm/model_executor/custom_op.py:L83-L98 PluggableLayer.register_oot
        def decorator(layer_cls):
            # SOURCE: vllm/model_executor/custom_op.py:L83-L98 PluggableLayer.register_oot
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in op_registry_oot, f"Duplicate layer name: {reg_name}"
            layer_cls.name = reg_name
            op_registry_oot[reg_name] = layer_cls
            return layer_cls

        if _decorated_layer_cls is None:
            # Called with parentheses: @PluggableLayer.register_oot()
            # or @PluggableLayer.register_oot(name="...")
            return decorator
        elif isinstance(_decorated_layer_cls, type):  # Check if it's a class
            # Called without parentheses: @PluggableLayer.register_oot
            return decorator(_decorated_layer_cls)
        else:
            raise TypeError("Decorator can only be applied to classes.")


# SOURCE: vllm/model_executor/custom_op.py:L103 CustomOp
class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    # SOURCE: vllm/model_executor/custom_op.py:L109-L128 CustomOp.__new__（逐字）
    def __new__(cls, *args, **kwargs):
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

    # SOURCE: vllm/model_executor/custom_op.py:L130-L133 CustomOp.__init__（逐字）
    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
        super().__init__()
        self._enforce_enable = enforce_enable
        self._forward_method = self.dispatch_forward(compile_native=compile_native)

    # SOURCE: vllm/model_executor/custom_op.py:L135-L136 CustomOp.forward（逐字）
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L138-L144 forward_native（逐字）
    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.

        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    # SOURCE: vllm/model_executor/custom_op.py:L146-L147 forward_cuda（逐字）
    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    # SOURCE: vllm/model_executor/custom_op.py:L149-L151 forward_hip（逐字）
    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L153-L156 forward_xpu（逐字）
    def forward_xpu(self, *args, **kwargs):
        # By default, we assume that XPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L158-L161 forward_cpu（逐字）
    def forward_cpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L163-L167 forward_tpu（逐字）
    def forward_tpu(self, *args, **kwargs):
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L169-L172 forward_oot（逐字）
    def forward_oot(self, *args, **kwargs):
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L174-L205 dispatch_forward
    #   减法子集（主干逐字：enabled 判定 → 平台分支；host 平台面全 False →
    #   else forward_cuda 与真实非 CUDA 平台同型）
    def dispatch_forward(self, compile_native: bool):
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.
        # SOURCE: vllm/model_executor/custom_op.py:L174-L205 dispatch_forward
        compilation_config = get_cached_compilation_config()

        # NOTE(shen-shanshan): CustomOp object can be enforce enabled, e.g.,
        # enable device-specific kernels in ViT models when enabling graph
        # mode. By default, it will follow the compilation_config to determine
        # whether enable itself.
        # This enforce_enable mechanism will be removed after we adding a
        # separate compilation_config for multi-modal part.
        enabled = self._enforce_enable or self.enabled()
        if enabled:
            compilation_config.enabled_custom_ops.append(self.__class__.name)
        else:
            compilation_config.disabled_custom_ops.append(self.__class__.name)

        if not enabled:
            # Compile forward_native to avoid eager torch ops if inside
            # opaque torch custom op (e.g. fused_moe, unified_attention, etc.)
            return self.maybe_compile(self.forward_native, enable=compile_native)

        if current_platform.is_rocm():
            return self.forward_hip
        elif current_platform.is_cpu():
            return self.forward_cpu
        elif current_platform.is_tpu():
            return self.forward_tpu
        elif current_platform.is_xpu():
            return self.forward_xpu
        elif current_platform.is_out_of_tree():
            return self.forward_oot
        else:
            return self.forward_cuda

    # SOURCE: vllm/model_executor/custom_op.py:L209-L236 maybe_compile
    #   子集（未启编译时原函数直返的分支逐字；torch.compile 包装 → ch19）
    def maybe_compile(self, fn, *, enable: bool = True):
        """
        Compile fn if compilation enabled.
        Useful for CustomOp instances called from within a torch custom op,
        meaning the forward call is hidden from the model-level torch.compile.

        NOTE: this does not enable fusion across ops, so opaque custom ops
        should still be unwrapped wherever possible.
        """
        # Do not compile if compilation disabled
        # SOURCE: vllm/model_executor/custom_op.py:L209-L236 maybe_compile
        if not enable:
            return fn
        # SUBTRACTED: torch.compile 包装（custom_op.py:L222-L280）——ch19 编译域
        return fn

    # SOURCE: vllm/model_executor/custom_op.py:L271-L293 enabled（逐字）
    @classmethod
    def enabled(cls) -> bool:
        # if no name, then it was not registered
        # SOURCE: vllm/model_executor/custom_op.py:L271-L293 enabled（逐字）
        compilation_config = get_cached_compilation_config()
        custom_ops = compilation_config.custom_ops
        if not hasattr(cls, "name"):
            logger.warning_once(
                "Custom op %s was not registered, which means it won't appear "
                "in the op registry. It will be enabled/disabled based on the "
                "global settings.",
                cls.__name__,
            )
            return CustomOp.default_on()

        enabled = f"+{cls.name}" in custom_ops
        disabled = f"-{cls.name}" in custom_ops
        if enabled and disabled:
            raise ValueError(
                "custom_ops cannot both enable and disable the same operation: "
                f"{cls.name}. Remove either the '+' or '-' directive"
            )

        return (CustomOp.default_on() or enabled) and not disabled

    # SOURCE: vllm/model_executor/custom_op.py:L295-L312 default_on（逐字）
    @staticmethod
    def default_on() -> bool:
        """
        Behavior controlled by `CompilationConfig.custom_ops`: On by default if
        'all', off by default if 'none'.
        When PyTorch Inductor is used, 'none' is the default value,
        otherwise 'all'.
        """
        # SOURCE: vllm/model_executor/custom_op.py:L295-L312 default_on（逐字）
        compilation_config = get_cached_compilation_config()
        count_none = compilation_config.custom_ops.count("none")
        count_all = compilation_config.custom_ops.count("all")
        if count_none + count_all != 1:
            raise ValueError(
                "custom_ops must contain exactly one base mode: 'all' or 'none'"
            )

        return not count_none > 0 or count_all > 0

    # SOURCE: vllm/model_executor/custom_op.py:L314-L325 CustomOp.register
    #   （逐字——@CustomOp.register("rms_norm")/("silu_and_mul") 的注册面）
    @classmethod
    def register(
        cls,
        name: str,
        dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    ):
        # SOURCE: vllm/model_executor/custom_op.py:L314-L325 CustomOp.register
        def decorator(op_cls):
            # SOURCE: vllm/model_executor/custom_op.py:L314-L325 CustomOp.register
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_cls._dynamic_arg_dims = dynamic_arg_dims
            op_registry[name] = op_cls
            return op_cls

        return decorator

    # SOURCE: vllm/model_executor/custom_op.py:L338-L~352 CustomOp.register_oot
    #   —— SUBTRACTED（OOT 自定义算子注册面；本章无消费位）
