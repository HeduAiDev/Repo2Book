# ch30 vLLM 侧扩展点(1)：模型注册表 —— subtract-only 精简版
#
# 真实源码 vllm/model_executor/models/registry.py：_ModelRegistry.register_model 是 vLLM 留给
# 外部（含 OOT 插件）登记整模型的扩展点。昇腾的 register_model() 就是来调它，把架构名映射到
# 「<module>:<class>」懒加载字符串。
#
# 按 subtraction_plan 思路（embed_excerpt 已 elide）删去：register_model 的 TypeError 校验分支、
# docstring 的 CUDA fork 解释；以及 _ModelRegistry 其余方法、_BaseRegisteredModel/
# _RegisteredModel/_LazyRegisteredModel 三个 dataclass 的实现 —— 本章只读「字符串懒加载 +
# 已注册同名覆盖」这条注册主干。

from dataclasses import dataclass, field
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# SUBTRACTED: registry.py:L709-L930 —— _BaseRegisteredModel / _RegisteredModel /
#   _LazyRegisteredModel 三个 frozen dataclass（懒加载/直引模型的内部表示），及 _ModelRegistry
#   的其余方法（get_supported_archs / resolve_model_cls / inspect_model_cls / ...）。本章只看
#   register_model 这一个扩展点入口。


@dataclass
class _ModelRegistry:
    # SOURCE: vllm/model_executor/models/registry.py:L931-L934
    # Keyed by model_arch
    models: dict = field(default_factory=dict)

    # SOURCE: vllm/model_executor/models/registry.py:L939-L982
    def register_model(
        self,
        model_arch: str,
        model_cls: type[nn.Module] | str,
    ) -> None:
        # SUBTRACTED: docstring —— 解释 model_cls 可为类或 `<module>:<class>` 字符串，后者
        #   延迟 import 以避免 fork 子进程里提前初始化 CUDA（`RuntimeError: Cannot re-initialize
        #   CUDA in forked subprocess`）。设计动机由正文 design_decisions 讲清。
        # SUBTRACTED: registry.py:L955-L957 —— `model_arch` 非 str 的 TypeError 校验分支。

        if model_arch in self.models:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.",
                model_arch,
                model_cls,
            )

        if isinstance(model_cls, str):
            split_str = model_cls.split(":")
            if len(split_str) != 2:
                msg = "Expected a string in the format `<module>:<class>`"
                raise ValueError(msg)

            model = _LazyRegisteredModel(*split_str)  # noqa: F821  —— SUBTRACTED 的懒加载表示
        elif isinstance(model_cls, type) and issubclass(model_cls, nn.Module):
            model = _RegisteredModel.from_model_cls(model_cls)  # noqa: F821
        else:
            msg = (
                "`model_cls` should be a string or PyTorch model class, "
                f"not a {type(model_arch)}"
            )
            raise TypeError(msg)

        self.models[model_arch] = model


# SOURCE: vllm/model_executor/models/registry.py:L1319
# SUBTRACTED: registry.py:L1319-L1330 内置模型表的初始化字典（把 vLLM 自带的全部架构名预填成
#   _LazyRegisteredModel）。精简版建空表即可——昇腾 register_model() 往里追加 DeepseekV4。
ModelRegistry = _ModelRegistry()
