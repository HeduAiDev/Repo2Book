# SOURCE: vllm/model_executor/model_loader/__init__.py
# ch23 主角文件之八（m8/m10）：get_model_loader（load_format→loader 查表）与
# get_model（统一入口）。
# SUBTRACTED：delete[9]——bitsandbytes/tensorizer/runai_streamer/sharded_state/
#   modelexpress 各 loader 文件不进；_LOAD_FORMAT_TO_MODEL_LOADER 表只留
#   default 与 dummy 两项（auto/hf/fastsafetensors 等别名随 Default 项合并）；
#   register_model_loader OOT 注册面不进。
from typing import Literal

from torch import nn

from vllm.config import ModelConfig, VllmConfig
# import 面归一：真实为 from vllm.config.load import LoadConfig（本章 config
# 承载为单门面 vllm/config/__init__.py——ch03 域的包分层不重建）
from vllm.config import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name,
    get_model_architecture,
    get_model_cls,
)

# Reminder: Please update docstring in `LoadConfig`
# if a new load format is added here
# SOURCE: vllm/model_executor/model_loader/__init__.py:L33-L49 LoadFormats
#   减法子集（delete[9]：只留 default/dummy 两格）
LoadFormats = Literal[
    "auto",
    "dummy",
]
# SOURCE: vllm/model_executor/model_loader/__init__.py:L50-L66 查表
#   子集（delete[9]：default 系别名合并进两项）
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {
    "auto": DefaultModelLoader,
    "dummy": DummyModelLoader,
}


# SUBTRACTED: register_model_loader OOT 注册装饰器
#   （model_loader/__init__.py:L69-L119）——OOT 平台后门，本章无消费


# SOURCE: vllm/model_executor/model_loader/__init__.py:L122-L127 get_model_loader
#   （逐字——load_format 查表，未登记即 ValueError）
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    # SOURCE: vllm/model_executor/model_loader/__init__.py:L122-L127 get_model_loader
    load_format = load_config.load_format
    if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
        raise ValueError(f"Load format `{load_format}` is not supported")
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)


# SOURCE: vllm/model_executor/model_loader/__init__.py:L130-L142 get_model
#   统一入口：选 loader → loader.load_model(vllm_config, model_config, prefix)
def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    # SOURCE: vllm/model_executor/model_loader/__init__.py:L130-L142 get_model
    loader = get_model_loader(load_config or vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )
