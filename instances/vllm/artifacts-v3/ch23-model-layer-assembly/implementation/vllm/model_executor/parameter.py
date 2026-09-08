# SOURCE: vllm/model_executor/parameter.py
# ch23 消费面：vLLM 参数类族的最小真实子集——UnquantizedLinearMethod /
# UnquantizedEmbeddingMethod 的 create_weights 产 ModelWeightParameter；层构造
# 的 update_param_tp_status 以 BasevLLMParameter 为 isinstance 键。
# SUBTRACTED：PackedColumnParameter/PackedvLLMParameter/PerTensorScaleParameter/
#   BlockQuantScaleParameter 等量化参数族（delete[7]——量化装载归 ch27）。
from __future__ import annotations

from typing import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


# SOURCE: vllm/model_executor/parameter.py:L32 BasevLLMParameter
class BasevLLMParameter(Parameter):
    """
    Base parameter for vLLM linear layers. Extends the torch.nn.parameter
    by taking in a linear weight loader. Will copy the loaded weight
    into the parameter when the provided weight loader is called.
    """

    # SOURCE: vllm/model_executor/parameter.py:L39-L75 BasevLLMParameter 的
    #   __new__/__init__ —— 减法子集（weight_loader 属性位 + tp_rank/tp_size
    #   戳记逐字；TPU sync weight_loader 包装删除——同 set_weight_attrs 的
    #   SUBTRACTED 位）
    def __new__(cls, data: torch.Tensor | None, **kwargs):
        # SOURCE: vllm/model_executor/parameter.py:L39-L75 BasevLLMParameter 的
        return super().__new__(cls, data=data, requires_grad=False)

    # SOURCE: vllm/model_executor/parameter.py:L42 __init__ 签名
    def __init__(self, data: torch.Tensor, weight_loader: Callable):
        """
        Initialize the BasevLLMParameter

        Args:
            data: torch tensor with the parameter data
            weight_loader: weight loader callable
        """
        # SUBTRACTED: TPU use_sync_weight_loader 包装（parameter.py:L51-L63）
        self._weight_loader = weight_loader
        # SOURCE: vllm/model_executor/parameter.py:L66-L67 tp 戳记（逐字）
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

    # SOURCE: vllm/model_executor/parameter.py:L69-L87 weight_loader 属性（逐字）
    @property
    def weight_loader(self) -> Callable:
        # NOTE(@ksayers) some models such as mamba_mixer2 override the
        # weight loader to support custom loading. In the future, model-specific
        # weight loading should be implemented via Model.load_weights. In the
        # meantime, support deleting and overriding `weight_loader` attribute
        # SOURCE: vllm/model_executor/parameter.py:L69-L87 weight_loader 属性（逐字）
        if self._weight_loader is None:
            raise AttributeError(
                f"{self.__class__.__name__} weight_loader attribute has been deleted"
            )
        return self._weight_loader

    @weight_loader.setter
    def weight_loader(self, value: Callable):
        # SOURCE: vllm/model_executor/parameter.py:L81-L82 weight_loader.setter
        self._weight_loader = value

    @weight_loader.deleter
    def weight_loader(self):
        # SOURCE: vllm/model_executor/parameter.py:L85-L87 weight_loader.deleter
        self._weight_loader = None  # type: ignore[assignment]

    # SOURCE: vllm/model_executor/parameter.py:L88-L91 _is_1d_and_scalar（逐字）
    def _is_1d_and_scalar(self, loaded_weight: torch.Tensor):
        cond1 = self.data.ndim == 1 and self.data.numel() == 1
        cond2 = loaded_weight.ndim == 0 and loaded_weight.numel() == 1
        return cond1 and cond2

    # SOURCE: vllm/model_executor/parameter.py:L93-L96 _assert_and_load（逐字）
    def _assert_and_load(self, loaded_weight: torch.Tensor):
        assert self.data.shape == loaded_weight.shape or self._is_1d_and_scalar(
            loaded_weight
        )
        self.data.copy_(loaded_weight)

    # SUBTRACTED: load_column_parallel_weight/load_row_parallel_weight/
    #   load_merged_column_weight/load_qkv_weight 的 Base 基类版
    #   （parameter.py:L99-L109）——被 _ColumnvLLMParameter/RowvLLMParameter
    #   的真实现覆盖（下方）；_shard_id_as_int（L111-L121）与 __torch_function__
    #   （L123-L126）随 weight_loader_v2 路径删除（delete[7]）


# SOURCE: vllm/model_executor/parameter.py:L129 _ColumnvLLMParameter
class _ColumnvLLMParameter(BasevLLMParameter):
    """
    Private class defining weight loading functionality
    (load_merged_column_weight, load_qkv_weight)
    for parameters being loaded into linear layers with column
    parallelism. This includes QKV and MLP layers which are
    not already fused on disk. Requires an output dimension
    to be defined. Called within the weight loader of
    each of the column parallel linear layers.
    """

    # SOURCE: vllm/model_executor/parameter.py:L140-L142 __init__（逐字）
    def __init__(self, output_dim: int, **kwargs):
        self._output_dim = output_dim
        super().__init__(**kwargs)

    # SOURCE: vllm/model_executor/parameter.py:L144-L146 output_dim 属性（逐字）
    @property
    def output_dim(self):
        # SOURCE: vllm/model_executor/parameter.py:L144-L146 output_dim 属性（逐字）
        return self._output_dim

    # SOURCE: vllm/model_executor/parameter.py:L148-L155 load_column_parallel_weight
    #   （逐字）
    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        # SOURCE: vllm/model_executor/parameter.py:L148-L155 load_column_parallel_weight
        shard_size = self.data.shape[self.output_dim]
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    # SOURCE: vllm/model_executor/parameter.py:L156-L173 load_merged_column_weight
    #   —— 减法子集（主干逐字；Packed 判整分支删除——delete[7]）
    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        # SOURCE: vllm/model_executor/parameter.py:L156-L173 load_merged_column_weight
        shard_offset: int = kwargs["shard_offset"]
        shard_size: int = kwargs["shard_size"]

        # SUBTRACTED: PackedColumnParameter/PackedvLLMParameter 的 packed_dim
        #   判整（parameter.py:L174-L182）——delete[7]，量化装载归 ch27

        param_data = self.data

        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


# SOURCE: vllm/model_executor/parameter.py:L204 RowvLLMParameter
class RowvLLMParameter(BasevLLMParameter):
    """
    Parameter class defining weight_loading functionality
    (load_row_parallel_weight) for parameters being loaded
    into linear layers with row parallel functionality.
    Requires an input_dim to be defined.
    """

    # SOURCE: vllm/model_executor/parameter.py:L212-L214 __init__（逐字）
    def __init__(self, input_dim: int, **kwargs):
        self._input_dim = input_dim
        super().__init__(**kwargs)

    # SOURCE: vllm/model_executor/parameter.py:L217-L219 input_dim 属性（逐字）
    @property
    def input_dim(self):
        # SOURCE: vllm/model_executor/parameter.py:L217-L219 input_dim 属性（逐字）
        return self._input_dim

    # SOURCE: vllm/model_executor/parameter.py:L220-L230 load_row_parallel_weight
    #   （逐字）
    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        # SOURCE: vllm/model_executor/parameter.py:L220-L230 load_row_parallel_weight
        shard_size = self.data.shape[self.input_dim]
        loaded_weight = loaded_weight.narrow(
            self.input_dim, self.tp_rank * shard_size, shard_size
        )

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


# SOURCE: vllm/model_executor/parameter.py:L233 ModelWeightParameter（逐字）
class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for linear layer weights. Uses both column and
    row parallelism.
    """

    pass
