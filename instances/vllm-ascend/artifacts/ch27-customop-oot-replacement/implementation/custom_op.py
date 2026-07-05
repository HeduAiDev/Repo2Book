# 只做减法的精简版 —— 对照基座 vLLM v0.21.0
# 真实文件：vllm/model_executor/custom_op.py
#
# 本章只需基座 CustomOp 的两条主线：
#   「换身」__new__ —— 按 op_registry_oot 把被实例化的类替成 OOT 子类；
#   「换头」dispatch_forward —— 按 enabled + 平台把 forward 绑到 forward_oot / forward_native。
# register / register_oot 给出「谁写进哪张表」。其余编译/平台分支按 subtraction_plan 删。
import torch
import torch.nn as nn

from vllm.config import get_cached_compilation_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# SOURCE: vllm/model_executor/custom_op.py:L21-22
# 两张全局注册表：op_registry 收 in-tree 算子；op_registry_oot 收 OOT（昇腾这类插件）子类。
op_registry: dict[str, type["CustomOp"]] = {}
op_registry_oot: dict[str, type["CustomOp"]] = {}

# SUBTRACTED: maybe_get_oot_by_class(L25-29) / class PluggableLayer(L32+) —— 与本章顶替主线无关，
#             本章只讲 CustomOp 子类经 op_registry_oot 的换身换头。原 vllm/model_executor/custom_op.py:L25-100


class CustomOp(nn.Module):
    # SOURCE: vllm/model_executor/custom_op.py:L103
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    def __new__(cls, *args, **kwargs):
        # SOURCE: vllm/model_executor/custom_op.py:L109
        # 「换身」：实例化前先按类名查 op_registry_oot —— 命中则把真正被实例化的类替成 OOT 子类。
        # 模型里写 RMSNorm(...) 时 cls.__name__=='RMSNorm' 命中 op_registry_oot['RMSNorm']==AscendRMSNorm，
        # 于是 super().__new__ 造出来的其实是昇腾子类实例；调用方语法不变，对象身份已换。
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
        # SOURCE: vllm/model_executor/custom_op.py:L130
        super().__init__()
        self._enforce_enable = enforce_enable
        # 构造期一次性选定 forward 后端，绑进 _forward_method（前向热路径零额外分发开销）。
        self._forward_method = self.dispatch_forward(compile_native=compile_native)

    def forward(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/custom_op.py:L135
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/custom_op.py:L138
        """PyTorch-native implementation of the forward method."""
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/custom_op.py:L146
        raise NotImplementedError

    # SUBTRACTED: forward_hip/forward_xpu/forward_cpu/forward_tpu 的默认回退（L149-167）——
    #             本章只关心 oot 与 native 两条；其余平台默认回退与昇腾顶替无关。

    def forward_oot(self, *args, **kwargs):
        # SOURCE: vllm/model_executor/custom_op.py:L169
        # 基座默认：OOT 实现等同 native。所以「不覆写 forward_oot == 无顶替」——
        # 昇腾子类正是靠覆写它来「换头」。
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self, compile_native: bool):
        # SOURCE: vllm/model_executor/custom_op.py:L174
        # 「换头」：按 enabled + 平台选定 forward 实现。两条主线：
        #   未 enabled → 编译 forward_native（走基座原生实现）；
        #   enabled 且 is_out_of_tree() → forward_oot（昇腾顶替实现）。
        compilation_config = get_cached_compilation_config()

        enabled = self._enforce_enable or self.enabled()
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])
        else:
            compilation_config.disabled_custom_ops.update([self.__class__.name])

        if not enabled:
            # Compile forward_native to avoid eager torch ops if inside
            # opaque torch custom op (e.g. fused_moe, unified_attention, etc.)
            return self.maybe_compile(self.forward_native, enable=compile_native)

        # SUBTRACTED: is_rocm()/is_cpu()/is_tpu()/is_xpu() 平台分支（L196-203）——
        #             本章只保留昇腾平台关心的 is_out_of_tree()→forward_oot 与兜底 forward_cuda。
        if current_platform.is_out_of_tree():
            return self.forward_oot
        else:
            return self.forward_cuda

    def maybe_compile(self, fn, *, enable: bool = True):
        # SOURCE: vllm/model_executor/custom_op.py:L209
        # Do not compile if compilation disabled
        if not enable:
            return fn
        # SUBTRACTED: torch.compile 编译机理（mode/backend/dynamic_arg_dims/wrapper，L218-269）——
        #             torch.compile 插件管线与本章「顶替」主线无关，host 无编译后端；精简版直接返回 fn，
        #             足以验「未 enabled → 走 forward_native」这条控制流。
        return fn

    @classmethod
    def enabled(cls) -> bool:
        # SOURCE: vllm/model_executor/custom_op.py:L271
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
        assert not (enabled and disabled), f"Cannot enable and disable {cls.name}"

        return (CustomOp.default_on() or enabled) and not disabled

    @staticmethod
    def default_on() -> bool:
        # SOURCE: vllm/model_executor/custom_op.py:L291
        compilation_config = get_cached_compilation_config()
        count_none = compilation_config.custom_ops.count("none")
        count_all = compilation_config.custom_ops.count("all")
        assert count_none + count_all == 1

        return not count_none > 0 or count_all > 0

    # Decorator to register custom ops.
    @classmethod
    def register(
        cls,
        name: str,
        dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    ):
        # SOURCE: vllm/model_executor/custom_op.py:L307
        # in-tree 注册：name 是 lowercase 名（如 'rms_norm'），写进 op_registry。
        def decorator(op_cls):
            # SOURCE: vllm/model_executor/custom_op.py:L313
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_cls._dynamic_arg_dims = dynamic_arg_dims
            op_registry[name] = op_cls
            return op_cls

        return decorator

    # Decorator to register out-of-tree(oot) custom ops.
    @classmethod
    def register_oot(cls, _decorated_op_cls=None, name: str | None = None):
        # SOURCE: vllm/model_executor/custom_op.py:L332
        # OOT 注册：把昇腾子类写进 op_registry_oot，并把 op.name 覆盖成注册键（如 'RMSNorm'，
        # 而非基座 register 设的 lowercase 'rms_norm'）——dispatch 的 enabled() 以类名为启停键。
        def decorator(op_cls):
            # SOURCE: vllm/model_executor/custom_op.py:L333
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in op_registry_oot, f"Duplicate op name: {reg_name}"
            op_cls.name = reg_name
            op_registry_oot[reg_name] = op_cls
            return op_cls

        if _decorated_op_cls is None:
            # Called with parentheses: @CustomOP.register_oot(name="...")
            return decorator
        elif isinstance(_decorated_op_cls, type):  # Check if it's a class
            # 本章以 register_oot(_decorated_op_cls=op_cls, name=name) 的「带类」形态调用，
            # 等价于立刻对该类执行 decorator。
            return decorator(_decorated_op_cls)
        else:
            raise TypeError("Decorator can only be applied to classes.")
