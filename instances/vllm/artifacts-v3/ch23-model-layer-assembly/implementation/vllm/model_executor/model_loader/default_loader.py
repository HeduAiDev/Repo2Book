# SOURCE: vllm/model_executor/model_loader/default_loader.py
# ch23 主角文件之十（m6/m10）：DefaultModelLoader.load_weights——checkpoint
# 的 (name, tensor) 流式喂给模型自己声明的 load_weights（权重怎么落由模型层
# 决定，loader 只管产出流）。
# SUBTRACTED：Source/dataclass 与 checkpoint IO 族（_prepare_weights/
#   get_all_weights 的 safetensors 流式迭代/glob 收集）——仓库 IO 面（dossier
#   站5锚点止于 L415-L427；host 测试以 in-memory (name,tensor) 流注入——
#   SeededLoader 覆写 get_all_weights）；torchao safetensors 策略与
#   _init_ep_weight_filter（量化/EP 特例）；@instrument 观测面。
from __future__ import annotations

import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)


# SOURCE: vllm/model_executor/model_loader/default_loader.py:L43
#   DefaultModelLoader —— 减法子集（load_weights 主线保留）
class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # SOURCE: vllm/model_executor/model_loader/default_loader.py:L46-L47
    #   DEFAULT_NUM_THREADS（逐字）
    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    # SUBTRACTED: Source dataclass（default_loader.py:L49-L69）与
    #   _prepare_weights/_filter_safetensors/get_all_weights 的 checkpoint
    #   IO 族（L71-L414）——safetensors/glob 流式产出面；host 测试以
    #   in-memory 流覆写 get_all_weights
    counter_before_loading_weights: float = 0.0
    # SOURCE: vllm/model_executor/model_loader/default_loader.py:L72
    #   counter_after_loading_weights（逐字位）
    counter_after_loading_weights: float = 0.0

    # SOURCE: vllm/model_executor/model_loader/default_loader.py:L74-L__init__
    #   （逐字——直通基类）
    def __init__(self, load_config: LoadConfig):
        # SOURCE: vllm/model_executor/model_loader/default_loader.py:L74-L__init__
        super().__init__(load_config)

    # SOURCE: vllm/model_executor/model_loader/default_loader.py:L342
    #   download_model —— SUBTRACTED 方法体（HuggingFace 下载面；host 无网络
    #   消费位）：
    #   def download_model(self, model_config: ModelConfig) -> None:
    #       self._prepare_weights(self.model_config, self.source)
    def download_model(self, model_config: ModelConfig) -> None:
        # SUBTRACTED: _prepare_weights 下载面（default_loader.py:L343-L344）
        # SOURCE: vllm/model_executor/model_loader/default_loader.py:L342
        raise NotImplementedError("SUBTRACTED: checkpoint 下载面——仓库 IO 域")

    # SUBTRACTED: @instrument(span_name="Load weights") 观测装饰
    #   （default_loader.py:L414）

    # SOURCE: vllm/model_executor/model_loader/default_loader.py:L415-L427
    #   load_weights —— 减法子集（delete 面：torchao safetensors 策略与
    #   _init_ep_weight_filter 两个量化/EP 特例前置删除；核心两行逐字：
    #   model.load_weights(self.get_all_weights(model_config, model)))
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # SUBTRACTED: torchao safetensors_load_strategy 策略面
        #   （default_loader.py:L416-L425）——torchao 量化特例
        # SUBTRACTED: self._init_ep_weight_filter(model_config)
        #   （default_loader.py:L426）——EP 专家过滤特例（ch28）

        # SOURCE: vllm/model_executor/model_loader/default_loader.py:L415-L427
        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))

        # SUBTRACTED: counter_after_loading_weights 计时与 logger.info_once
        #   汇报尾（default_loader.py:L428-L431）——观测面（返回的
        #   loaded_weights 集合语义不受影响）

    # SOURCE: vllm/model_executor/model_loader/default_loader.py get_all_weights
    #   —— SUBTRACTED 全方法（safetensors 流式产出族）：
    #   真实签名 def get_all_weights(model_config, model) -> Iterable[tuple[str,
    #   Tensor]]——逐文件 yield (name, tensor)。host 测试以 SeededLoader 覆写
    #   此位注入 in-memory 权重流（同签名同语义）。
