# ch30 vLLM 侧扩展点(3)：model-loader 注册表 —— subtract-only 精简版
#
# 真实源码 vllm/model_executor/model_loader/__init__.py：register_model_loader 装饰器把
# 「load_format 字符串 → loader 类」写进全局 dict _LOAD_FORMAT_TO_MODEL_LOADER，校验须
# issubclass(BaseModelLoader)；get_model_loader 按 --load-format 取类实例化。昇腾的
# netloader/rfork 就是往这个 dict 各注一个键。
#
# 按 subtraction_plan 思路（embed_excerpt 已 elide）删去：register_model_loader 的 docstring
# >>> 示例、_LOAD_FORMAT_TO_MODEL_LOADER 的内置条目（auto/hf/gguf/tensorizer/... → 各内置
# loader 类，逐个 import）。保留装饰器 + 注册表 + get_model_loader 分发主干。

from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

logger = init_logger(__name__)

# SOURCE: vllm/model_executor/model_loader/__init__.py:L48-L64
# SUBTRACTED: __init__.py:L48-L63 内置条目 —— bitsandbytes/dummy/gguf/runai_streamer/
#   sharded_state/tensorizer/... 各 load_format → 各内置 loader 类（须逐个 import 对应类）。
#   精简版只留 auto/hf → DefaultModelLoader 作代表，昇腾 netloader/rfork 注册后追加进本 dict。
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {
    "auto": DefaultModelLoader,
    "hf": DefaultModelLoader,
}


# SOURCE: vllm/model_executor/model_loader/__init__.py:L67-L116
def register_model_loader(load_format: str):
    # SUBTRACTED: docstring 的 >>> 用法示例（@register_model_loader("my_loader") class ...）。

    def _wrapper(model_loader_cls):
        # SOURCE: vllm/model_executor/model_loader/__init__.py:L96-L114
        if load_format in _LOAD_FORMAT_TO_MODEL_LOADER:
            logger.warning(
                "Load format `%s` is already registered, and will be "
                "overwritten by the new loader class `%s`.",
                load_format,
                model_loader_cls,
            )
        if not issubclass(model_loader_cls, BaseModelLoader):
            raise ValueError(
                "The model loader must be a subclass of `BaseModelLoader`."
            )
        _LOAD_FORMAT_TO_MODEL_LOADER[load_format] = model_loader_cls
        logger.info(
            "Registered model loader `%s` with load format `%s`",
            model_loader_cls,
            load_format,
        )
        return model_loader_cls

    return _wrapper


# SOURCE: vllm/model_executor/model_loader/__init__.py:L119-L125
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    load_format = load_config.load_format
    if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
        raise ValueError(f"Load format `{load_format}` is not supported")
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)
