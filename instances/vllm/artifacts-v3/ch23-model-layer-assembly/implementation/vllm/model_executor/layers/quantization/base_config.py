# SOURCE: vllm/model_executor/layers/quantization/base_config.py
# ch23 消费面：量化基类注解面（UnquantizedLinearMethod / UnquantizedEmbedding
# Method 的基类 + embedding 探测）。量化方法族本体归 ch27。
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

import torch
from torch import nn


# SOURCE: vllm/model_executor/layers/quantization/base_config.py:L20
#   QuantizeMethodBase
class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    uses_meta_device: bool = False
    """
    Whether this method creates weights on meta device for online quantization.
    When True, weights are created on meta device and quantized layer-wise
    in process_weights_after_loading, reducing peak memory during model loading.
    """

    # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L30-L36
    #   create_weights 抽象签名（逐字）
    @abstractmethod
    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L30-L36
        raise NotImplementedError

    # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L38-L43
    #   apply 抽象签名（逐字）
    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L38-L43
        raise NotImplementedError

    # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L45-L51
    #   embedding 非必需方法（逐字——method_has_implemented_embedding 的对照基线）
    # Not required functions
    def embedding(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Gather embeddings in the layer based on indices in the input tensor.

        Expects create_weights to have been called before on the layer."""
        # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L45-L51
        raise NotImplementedError

    # Not required functions
    # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L53-L67
    #   tie_weights 默认实现（逐字——ParallelLMHead.tie_weights 的委托目标）
    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        """Tie ``layer``'s weight to ``embed_tokens``' weight.

        The default shares the weight tensor, which is the standard behavior for
        tied word embeddings and matches what ``ParallelLMHead.tie_weights`` did
        directly before quantization methods became responsible for it.
        Quantization methods that need special weight handling (e.g. repacked
        weights) override this.

        Expects create_weights to have been called before on the layer."""
        # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L53-L67
        layer.weight = embed_tokens.weight
        return layer

    # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L69-L75
    #   process_weights_after_loading 默认（逐字）
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process the weight after loading.

        This can be used for example to transpose weights for computation.
        """
        # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L69-L75
        return


# SOURCE: vllm/model_executor/layers/quantization/base_config.py:L75-L85
#   method_has_implemented_embedding（逐字）
def method_has_implemented_embedding(method_class: type[QuantizeMethodBase]) -> bool:
    """
    Not all quant methods have embedding implemented, so we need to check that
    it exists for our given method. We check by making sure the function
    has been changed from the base implementation.
    """
    # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L75-L85
    base_embedding = inspect.getattr_static(QuantizeMethodBase, "embedding", None)
    class_embedding = inspect.getattr_static(method_class, "embedding", None)
    return base_embedding is not None and class_embedding is not base_embedding


# SOURCE: vllm/model_executor/layers/quantization/base_config.py:L87 QuantizationConfig
#   —— HOST SEAM 注解位（真实为数百行抽象配置基类；本章无量化配置消费，
#   llama.py/linear.py 只把它用作类型注解——from __future__ annotations 下
#   字符串化不求值）
class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    # SOURCE: vllm/model_executor/layers/quantization/base_config.py:L87 QuantizationConfig
    pass
