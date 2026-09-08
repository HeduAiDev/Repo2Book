# SOURCE: vllm/model_executor/model_loader/base_loader.py
# ch23 主角文件之九（m10）：BaseModelLoader.load_model 编排——四段主干：
# set_default_torch_dtype + target_device 上下文 → initialize_model 建空壳 →
# self.load_weights → process_weights_after_loading → model.eval()。
# SUBTRACTED：delete[12]——log_model_inspection/peak memory 日志段/
# _has_online_quant+finalize_layerwise_processing（在线量化收尾）——观测与
# 量化域；@instrument 观测装饰面。
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config import LoadConfig
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype


# SOURCE: vllm/model_executor/model_loader/base_loader.py:L25 BaseModelLoader
class BaseModelLoader(ABC):
    """Base class for model loaders."""

    # SOURCE: vllm/model_executor/model_loader/base_loader.py:L28-L29 __init__
    #   （逐字）
    def __init__(self, load_config: LoadConfig):
        # SOURCE: vllm/model_executor/model_loader/base_loader.py:L28-L29 __init__
        self.load_config = load_config

    # SOURCE: vllm/model_executor/model_loader/base_loader.py:L31-L34
    #   download_model 抽象（逐字）
    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        # SOURCE: vllm/model_executor/model_loader/base_loader.py:L31-L34
        raise NotImplementedError

    # SOURCE: vllm/model_executor/model_loader/base_loader.py:L36-L40
    #   load_weights 抽象（逐字——inplace 权重装载的独立 API 面）
    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        # SOURCE: vllm/model_executor/model_loader/base_loader.py:L36-L40
        raise NotImplementedError

    # SUBTRACTED: @instrument(span_name="Load model") 观测装饰
    #   （base_loader.py:L42）——tracing 域

    # SOURCE: vllm/model_executor/model_loader/base_loader.py:L43-L82 load_model
    #   —— 减法子集（delete[12]：dtype/device 上下文 → 建空壳 → load_weights →
    #   process_weights_after_loading → eval() 四段主干逐字；观测/在线量化/
    #   peak memory 段删除）
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        # SOURCE: vllm/model_executor/model_loader/base_loader.py:L43-L82 load_model
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            # SUBTRACTED: log_model_inspection（base_loader.py:L61）——
            #   delete[12] 观测面
            self.load_weights(model, model_config)

            # SUBTRACTED: peak GPU memory 日志段（base_loader.py:L66-L73）
            #   ——delete[12] 观测面
            # SUBTRACTED: _has_online_quant + finalize_layerwise_processing
            #   （base_loader.py:L75-L78）——delete[12] 在线量化收尾（ch27）

            process_weights_after_loading(model, model_config, target_device)

        return model.eval()


# SUBTRACTED: log_model_inspection / _has_online_quant 两函数
#   （base_loader.py:L85-L101）——delete[12]
