# SOURCE: vllm/model_executor/model_loader/dummy_loader.py
# ch23 消费面：DummyModelLoader（load_format="dummy" 的 loader——profiling/
# 性能评估用；get_model_loader 表的第二项）。
# SUBTRACTED：随机权重初始化与在线量化层处理（get_layerwise_info/
#   initialize_dummy_weights/materialize_layer 族）——性能评估/量化域；
#   host 测试只消费其「免 checkpoint 建模型」的装载语义（权重保持未初始化）。
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


# SOURCE: vllm/model_executor/model_loader/dummy_loader.py:L22 DummyModelLoader
class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    # SOURCE: vllm/model_executor/model_loader/dummy_loader.py:L25-L31 __init__
    #   （逐字——extra_config 拒收）
    def __init__(self, load_config: LoadConfig):
        # SOURCE: vllm/model_executor/model_loader/dummy_loader.py:L25-L31 __init__
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    # SOURCE: vllm/model_executor/model_loader/dummy_loader.py:L33-L34
    #   download_model（逐字）
    def download_model(self, model_config: ModelConfig) -> None:
        # SOURCE: vllm/model_executor/model_loader/dummy_loader.py:L33-L34
        pass  # Nothing to download

    # SOURCE: vllm/model_executor/model_loader/dummy_loader.py:L36-L42
    #   load_weights —— 减法子集（随机权重赋值与在线量化层处理删除——
    #   性能评估/量化域；空壳权重保持 torch.empty 未初始化态）
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # SUBTRACTED: get_layerwise_info/_process_online_quant_layer/
        #   initialize_dummy_weights 循环（dummy_loader.py:L37-L41）——
        #   NOTE(woosuk) 随机权重是性能评估用；本章只消费「免 checkpoint
        #   建模型」语义
        # SOURCE: vllm/model_executor/model_loader/dummy_loader.py:L36-L42
        return
