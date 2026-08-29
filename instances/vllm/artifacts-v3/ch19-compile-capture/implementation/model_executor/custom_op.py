# Subtract-only companion for v3 ch19 — vllm/model_executor/custom_op.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`).
#
# Deletions here (dossier subtraction_plan.delete #1):
#   - PluggableLayer 整类（L32-L100）+ maybe_get_oot_by_class（L25-L29）+
#     register_oot（L329-L360）+ op_registry_oot 全局量与 __new__ 的 OOT
#     替换分支（L109-L128——OOT 整层替换机制，正文只作边界注）。
from __future__ import annotations

import functools
import inspect

import torch
import torch.nn as nn

from .._host_seams import (
    current_platform,
    init_logger,
    maybe_disable_graph_partition,
)
from ..config import get_cached_compilation_config

logger = init_logger(__name__)

# Dictionary of all custom ops (classes, indexed by registered name).
# To check if an op with a name is enabled, call .enabled() on the class.
# Examples:
# - MyOp.enabled()
# - op_registry["my_op"].enabled()
# SOURCE: vllm/model_executor/custom_op.py:L16-L21 op_registry
op_registry: dict[str, type["CustomOp"]] = {}
# SUBTRACTED: op_registry_oot（L22——delete[1]：OOT 注册表随 OOT 机制整删）。


# SUBTRACTED: PluggableLayer 整类与 maybe_get_oot_by_class（L25-L100
#   ——delete[1]：OOT 整层替换/模块级组装抽象，v0.27 姊妹机制，正文边界注）。


# SOURCE: vllm/model_executor/custom_op.py:L103-L327 CustomOp —— 多平台算子
#   基类：forward_* 多实现 + op_registry 全局注册 + 构造期 dispatch_forward
#   一次性绑定 _forward_method（运行期零分支）
class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    # SUBTRACTED: __new__（L109-L128——delete[1]：其唯一职能是查 op_registry_
    #   oot 做整类替换；无 OOT 表时是对 super().__new__ 的透传，删净后走
    #   nn.Module 默认构造）。

    # SOURCE: vllm/model_executor/custom_op.py:L130-L133 __init__ —— 构造期
    #   一次性 dispatch
    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):  # SOURCE: vllm/model_executor/custom_op.py:L130-L133
        super().__init__()
        self._enforce_enable = enforce_enable
        self._forward_method = self.dispatch_forward(compile_native=compile_native)

    # SOURCE: vllm/model_executor/custom_op.py:L135-L136 forward —— 运行期
    #   只一次属性转发零分支
    def forward(self, *args, **kwargs):  # SOURCE: vllm/model_executor/custom_op.py:L135-L136
        return self._forward_method(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L138-L144 forward_native
    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    # SOURCE: vllm/model_executor/custom_op.py:L146-L147 forward_cuda
    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    # SOURCE: vllm/model_executor/custom_op.py:L149-L151 forward_hip
    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L153-L156 forward_xpu
    def forward_xpu(self, *args, **kwargs):
        # By default, we assume that XPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L158-L161 forward_cpu
    def forward_cpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L163-L167 forward_tpu
    def forward_tpu(self, *args, **kwargs):
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L169-L172 forward_oot
    def forward_oot(self, *args, **kwargs):
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L174-L207 dispatch_forward ——
    #   构造期冻结：enabled() 查 ±name 开关，disabled 走 maybe_compile(
    #   forward_native)；enabled 按平台分支返回具名方法
    def dispatch_forward(self, compile_native: bool):  # SOURCE: vllm/model_executor/custom_op.py:L174-L207
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.
        compilation_config = get_cached_compilation_config()

        # NOTE(shen-shanshan): CustomOp object can be enforce enabled, e.g.,
        # enable device-specific kernels in ViT models when enabling graph
        # mode. By default, it will follow the compilation_config to determine
        # whether enable itself.
        # This enforce_enable mechanism will be removed after we adding a
        # separate compilation_config for multi-modal part.
        enabled = self._enforce_enable or self.enabled()
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])
        else:
            compilation_config.disabled_custom_ops.update([self.__class__.name])

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

    # SOURCE: vllm/model_executor/custom_op.py:L209-L269 maybe_compile ——
    #   不透明算子内部的 CustomOp 调用对模型级 torch.compile 不可见，其
    #   forward_native 可被单独编译（不跨算子融合）
    def maybe_compile(self, fn, *, enable: bool = True):
        """
        Compile fn if compilation enabled.
        Useful for CustomOp instances called from within a torch custom op,
        meaning the forward call is hidden from the model-level torch.compile.

        NOTE: this does not enable fusion across ops, so opaque custom ops
        should still be unwrapped wherever possible.
        """
        from ..config.compilation import CompilationMode

        # Do not compile if compilation disabled
        if not enable:
            return fn

        # Do not compile if global compilation disabled
        compilation_config = get_cached_compilation_config()
        if compilation_config.mode == CompilationMode.NONE:
            return fn

        # If eager backend is used, do not compile either
        if compilation_config.backend == "eager":
            return fn

        # SOURCE: vllm/model_executor/custom_op.py:L233-L261 编译尾段
        #   （dynamic_arg_dims 包装分支 + dynamic=True 的 torch.compile——
        #   dossier elide 注授权裁至正文点名 mark_dynamic；伴读版经
        #   HOST SEAM simple_compile_backend/maybe_disable_graph_partition
        #   走 eager 面，控制流不变）
        compile_options = maybe_disable_graph_partition(
            current_platform.simple_compile_backend
        )
        backend = current_platform.simple_compile_backend

        dynamic_arg_dims = getattr(self.__class__, "_dynamic_arg_dims", None)
        if dynamic_arg_dims is not None:
            compiled_fn = torch.compile(
                fn,
                dynamic=False,
                backend=backend,
                options=compile_options,
            )
            sig = inspect.signature(fn)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):  # SOURCE: vllm/model_executor/custom_op.py:L248-L261
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                for name, dims in dynamic_arg_dims.items():
                    arg = bound.arguments.get(name)
                    if arg is not None and isinstance(arg, torch.Tensor):
                        dims_list = [dims] if isinstance(dims, int) else dims
                        for d in dims_list:
                            real_d = arg.ndim + d if d < 0 else d
                            torch._dynamo.mark_dynamic(arg, real_d)
                return compiled_fn(*args, **kwargs)

            return wrapper

        # dynamic=True to avoid recompilations
        return torch.compile(
            fn,
            dynamic=True,
            backend=backend,
            options=compile_options,
        )

    @classmethod
    def enabled(cls) -> bool:  # SOURCE: vllm/model_executor/custom_op.py:L271-L293
        # if no name, then it was not registered
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

    @staticmethod
    def default_on() -> bool:  # SOURCE: vllm/model_executor/custom_op.py:L295-L311
        """
        Behavior controlled by `CompilationConfig.custom_ops`: On by default if
        'all', off by default if 'none'.
        When PyTorch Inductor is used, 'none' is the default value,
        otherwise 'all'.
        """
        compilation_config = get_cached_compilation_config()
        count_none = compilation_config.custom_ops.count("none")
        count_all = compilation_config.custom_ops.count("all")
        if count_none + count_all != 1:
            raise ValueError(
                "custom_ops must contain exactly one base mode: 'all' or 'none'"
            )

        return not count_none > 0 or count_all > 0

    # Decorator to register custom ops.
    # SOURCE: vllm/model_executor/custom_op.py:L313-L327 register
    @classmethod
    def register(
        cls,
        name: str,
        dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    ):
        def decorator(op_cls):  # SOURCE: vllm/model_executor/custom_op.py:L320-L325
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_cls._dynamic_arg_dims = dynamic_arg_dims
            op_registry[name] = op_cls
            return op_cls

        return decorator

    # SUBTRACTED: register_oot（L329-L360——delete[1]：OOT 注册装饰器）。
