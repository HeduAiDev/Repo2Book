"""ch23 第 1 级 dispatch：CustomOp 基类 + RMSNorm 范例（只做减法）。

对应 vllm/model_executor/custom_op.py 与 vllm/model_executor/layers/layernorm.py。

第 1 级 dispatch 的本质：CustomOp.__init__ 在「构造期」一次性调 dispatch_forward()，
据 enabled()（读 CompilationConfig.custom_ops）+ 平台，选定 self._forward_method 指向
forward_cuda（预编译融合 kernel）还是 forward_native（纯 torch，留给 Inductor 融合）。
之后运行期 forward 只做一次零开销转发。注释明言：假设 vLLM 只为单一后端构建，故可一次性定死。
"""

from __future__ import annotations

import functools
import inspect

import torch
import torch.nn as nn

from ._runtime import current_platform, get_cached_compilation_config

# SOURCE: vllm/model_executor/custom_op.py:L21 (op_registry)
op_registry: dict[str, type["CustomOp"]] = {}

# SUBTRACTED: op_registry_oot / maybe_get_oot_by_class / register_oot / PluggableLayer
# (custom_op.py:L22-L100, L322-L353)：out-of-tree 硬件插件「整类替换」旁路，给厂商插件用。
# 本章主线是 in-tree 两级 dispatch，删去不影响 RMSNorm/CUDA 主路径与 dispatch 逻辑。


# SOURCE: vllm/model_executor/custom_op.py:L103 (CustomOp)
class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    # SOURCE: vllm/model_executor/custom_op.py:L109 (__new__)
    def __new__(cls, *args, **kwargs):
        # SUBTRACTED: __new__ 原本检查 op_name 是否在 op_registry_oot 中、若是则实例化 OOT
        # 替换类（custom_op.py:L119-L127）。本章无 OOT，恒走 in-tree 分支，直接实例化 cls。
        # TypeError 异常消息全文亦省略。
        return super().__new__(cls)

    # SOURCE: vllm/model_executor/custom_op.py:L130 (__init__ — 第 1 级 dispatch 入口)
    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
        super().__init__()
        self._enforce_enable = enforce_enable
        # 关键：构造期就把 forward 方法定死，之后 forward 只转发到它。
        self._forward_method = self.dispatch_forward(compile_native=compile_native)

    # SOURCE: vllm/model_executor/custom_op.py:L135 (forward — 零开销转发)
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L138 (forward_native)
    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method."""
        raise NotImplementedError

    # SOURCE: vllm/model_executor/custom_op.py:L146 (forward_cuda)
    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    # SOURCE: vllm/model_executor/custom_op.py:L149 (forward_hip)
    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    # 默认非 CUDA 平台回退到 forward_native（XPU/CPU/TPU/OOT）。
    # SOURCE: vllm/model_executor/custom_op.py:L153 (forward_xpu)
    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L158 (forward_cpu)
    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L163 (forward_tpu)
    def forward_tpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L169 (forward_oot)
    def forward_oot(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    # SOURCE: vllm/model_executor/custom_op.py:L174 (dispatch_forward — 第 1 级 dispatch 主体)
    def dispatch_forward(self, compile_native: bool):
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.
        compilation_config = get_cached_compilation_config()

        enabled = self._enforce_enable or self.enabled()
        if enabled:
            # 记账到 config（供日志/调试），不影响 dispatch 结果。
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

    # SOURCE: vllm/model_executor/custom_op.py:L209 (maybe_compile)
    def maybe_compile(self, fn, *, enable: bool = True):
        """
        Compile fn if compilation enabled.
        Useful for CustomOp instances called from within a torch custom op,
        meaning the forward call is hidden from the model-level torch.compile.

        NOTE: this does not enable fusion across ops, so opaque custom ops
        should still be unwrapped wherever possible.
        """
        from ._runtime import CompilationMode

        if not enable:
            return fn
        compilation_config = get_cached_compilation_config()
        if compilation_config.mode == CompilationMode.NONE:
            return fn
        if compilation_config.backend == "eager":
            return fn

        backend = current_platform.simple_compile_backend
        # SUBTRACTED: maybe_disable_graph_partition 的 compile_options 与 dynamic_arg_dims
        # 逐参 mark_dynamic 包裹（custom_op.py:L233-L261）是动态形状细节；本章只需点明
        # 「enabled=False 时对 forward_native 单独 torch.compile，避免它退化成零散 eager 算子」。
        dynamic_arg_dims = getattr(self.__class__, "_dynamic_arg_dims", None)
        if dynamic_arg_dims is not None:
            compiled_fn = torch.compile(fn, dynamic=False, backend=backend)
            sig = inspect.signature(fn)

            @functools.wraps(fn)
            # SOURCE: vllm/model_executor/custom_op.py:L249 (maybe_compile 内部 wrapper)
            def wrapper(*args, **kwargs):
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
        return torch.compile(fn, dynamic=True, backend=backend)

    @classmethod
    # SOURCE: vllm/model_executor/custom_op.py:L271 (enabled — 开关逻辑)
    def enabled(cls) -> bool:
        # if no name, then it was not registered
        compilation_config = get_cached_compilation_config()
        custom_ops = compilation_config.custom_ops
        if not hasattr(cls, "name"):
            # SUBTRACTED: warning_once 文本省略（custom_op.py:L277-L282）。
            return CustomOp.default_on()

        enabled = f"+{cls.name}" in custom_ops
        disabled = f"-{cls.name}" in custom_ops
        assert not (enabled and disabled), f"Cannot enable and disable {cls.name}"

        return (CustomOp.default_on() or enabled) and not disabled

    @staticmethod
    # SOURCE: vllm/model_executor/custom_op.py:L291 (default_on — Inductor='none'/否则='all')
    def default_on() -> bool:
        """
        Behavior controlled by `CompilationConfig.custom_ops`: On by default if
        'all', off by default if 'none'.
        When PyTorch Inductor is used, 'none' is the default value,
        otherwise 'all'.
        """
        compilation_config = get_cached_compilation_config()
        count_none = compilation_config.custom_ops.count("none")
        count_all = compilation_config.custom_ops.count("all")
        assert count_none + count_all == 1

        return not count_none > 0 or count_all > 0

    @classmethod
    # SOURCE: vllm/model_executor/custom_op.py:L307 (register — 把类登记进 op_registry 并赋 name)
    def register(
        cls,
        name: str,
        dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    ):
        # SOURCE: vllm/model_executor/custom_op.py:L313 (register 内部 decorator)
        def decorator(op_cls):
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_cls._dynamic_arg_dims = dynamic_arg_dims
            op_registry[name] = op_cls
            return op_cls

        return decorator

    # SUBTRACTED: register_oot (custom_op.py:L331-L353) — OOT 注册旁路，本章无 OOT，删去。


# SOURCE: vllm/model_executor/layers/layernorm.py:L56 (fused_add_rms_norm — 预编译融合 kernel 入口)
def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SUBTRACTED: 真实调用 from vllm import _custom_ops as ops; ops.fused_add_rms_norm(...)
    # 这个 C++/CUDA 预编译融合 kernel，原位写 x、residual（layernorm.py:L62-L70）。host 无该
    # kernel，故用纯 torch 复现其「先 residual-add 再 RMSNorm」的数值语义（in-place 写回 x/residual）。
    # 关键对照点不变：forward_cuda 调用的是一个对 Inductor 不透明的预编译融合 kernel。
    orig_dtype = x.dtype
    xf = x.to(torch.float32) + residual.to(torch.float32)
    residual.copy_(xf.to(orig_dtype))
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + variance_epsilon)
    x.copy_((xf.to(orig_dtype)) * weight)
    return x, residual


# SOURCE: vllm/model_executor/layers/layernorm.py:L102 (RMSNorm — 本章 CustomOp 范例)
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # SOURCE: vllm/model_executor/layers/layernorm.py:L112 (__init__)
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        # super().__init__() 即触发 CustomOp.__init__ → dispatch_forward，定死 _forward_method。
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)
        # SUBTRACTED: ROCm aiter 分派(L133-137)与 Oink SM100 fast-path 检测(L139-185)：
        # 硬件特例的可选加速路径，删之不损 dispatch 主线。

    @staticmethod
    # SOURCE: vllm/model_executor/layers/layernorm.py:L187 (forward_static — 纯 torch RMSNorm)
    def forward_static(
        x: torch.Tensor,
        variance_epsilon: float,
        hidden_size: int,
        orig_dtype: torch.dtype,
        weight: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        variance_size_override: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        x = x.to(torch.float32)
        if residual is not None:
            # residual promoted f16->f32 automatically,
            # otherwise Inductor eliminates the casts to and from f16,
            # increasing memory usage (and complicating pattern matching)
            x = x + residual
            residual = x.to(orig_dtype)

        if x.shape[-1] != hidden_size:
            raise ValueError(
                f"Expected hidden_size to be {hidden_size}, but found: {x.shape[-1]}"
            )

        if variance_size_override is None:
            x_var = x
        else:
            if hidden_size < variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{variance_size_override}, but found: {hidden_size}"
                )
            x_var = x[:, :, :variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x.to(orig_dtype)
        if weight is not None:
            x = x * weight
        if residual is None:
            return x
        else:
            return x, residual

    # SOURCE: vllm/model_executor/layers/layernorm.py:L233 (forward_native — 可被 Inductor 融合的一路)
    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            # SUBTRACTED: 真实走 ir.ops.rms_norm(...)（一个走 IR 层、内部仍是纯 torch 的
            # 无 residual RMSNorm，layernorm.py:L240-L246）。host 无 ir 层，等价改调
            # forward_static（同样纯 torch、可被 Inductor 看见/融合），数值语义一致。
            return self.forward_static(
                x,
                self.variance_epsilon,
                self.hidden_size,
                x.dtype,
                self.weight.data if self.has_weight else None,
                None,
                self.variance_size_override,
            )

        return self.forward_static(
            x,
            self.variance_epsilon,
            self.hidden_size,
            x.dtype,
            self.weight.data if self.has_weight else None,
            residual,
            self.variance_size_override,
        )

    # SOURCE: vllm/model_executor/layers/layernorm.py:L258 (forward_cuda — 预编译融合 kernel 一路)
    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # SUBTRACTED: 真实 `if residual is None and not envs.VLLM_BATCH_INVARIANT`，调
            # ir.ops.rms_norm 预编译 kernel（layernorm.py:L263-L266）。host 无该 kernel，故用
            # forward_static 复现数值；关键对照点：这一路对 Inductor 是黑盒（不可融合）。
            return self.forward_static(
                x,
                self.variance_epsilon,
                self.hidden_size,
                x.dtype,
                self.weight.data if self.has_weight else None,
                None,
                self.variance_size_override,
            )
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        # SUBTRACTED: Oink SM100 in-place fast-path(L271-L310)与 VLLM_BATCH_INVARIANT 分支
        # (L317-L318)：可选/特例。主线即调用融合 CUDA kernel fused_add_rms_norm。
        return fused_add_rms_norm(
            x, residual, self.weight.data, self.variance_epsilon
        )
