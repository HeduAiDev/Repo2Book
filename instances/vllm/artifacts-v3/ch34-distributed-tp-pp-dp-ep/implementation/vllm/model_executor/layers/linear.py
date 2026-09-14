# SOURCE: vllm/model_executor/layers/linear.py
# ch34 切面（m7/站 11）：RowParallelLinear.forward（TP 消费现场：部分和→
# all_reduce、bias 只在 rank0 融进 GEMM）与 ColumnParallelLinear.forward（切片→
# all_gather）。
# SUBTRACTED：LinearMethodBase ABC（ch23 域的量化方法契约，本类以无基类承载）；
# ModelWeightParameter/weight_loader/set_weight_attrs 族（ch23 装配域——裸
# Parameter + 形状断言承载 nn.Linear 面）；process_weights_after_loading 的 CPU
# gemm 派发（ch23 域）；LinearBase.update_param_tp_status（L296-L309，loader 维度
# 申报面）；MergedColumn/QKV/DCPGroup 等子类族（ch23 域）。

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.utils import divide


# SOURCE: vllm/model_executor/layers/linear.py:L184-L230 UnquantizedLinearMethod
class UnquantizedLinearMethod:
    """Linear method without quantization."""

    # SOURCE: vllm/model_executor/layers/linear.py:L187-L214 create_weights ——
    #   语义子集（ModelWeightParameter/weight_loader 族归 ch23 装配域；裸
    #   Parameter + 形状断言即 implementer 契约的 nn.Linear 面）
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py:L187-L214（锚点双置）
        # This method creates unquantized linear weights.
        # The weights are not quantized, and they are not sharded.
        # The amount of memory allocated for the weights is
        # sum(output_partition_sizes) * input_size_per_partition.
        # SUBTRACTED: weight_loader = extra_weight_attrs.pop(...) 与
        #   ModelWeightParameter(...)/set_weight_attrs（L201-L214）——ch23 装配域。
        #   形状契约：weight == [sum(output_partition_sizes), in_per_partition]
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        assert list(weight.shape) == [sum(output_partition_sizes), input_size_per_partition]
        layer.register_parameter("weight", weight)

    # SUBTRACTED: process_weights_after_loading（L216-L220）——CPU gemm 派发面
    #   （ch23 域）。

    # SOURCE: vllm/model_executor/layers/linear.py:L222-L230 apply —— 语义子集
    #   （dispatch_unquantized_gemm 的平台派发归 ch23；F.linear 即其无量化语义；
    #   VLLM_BATCH_INVARIANT 分支 L228-L229 归 ch19）
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/linear.py:L230 apply 兜底行（锚点双置）
        return F.linear(x, layer.weight, bias)


# SOURCE: vllm/model_executor/layers/linear.py:L233-L309 LinearBase —— 装配子集
#   （quant_config 的方法工厂 elif 分支归 ch27；update_param_tp_status 归 ch23）
class LinearBase(torch.nn.Module):
    # SOURCE: vllm/model_executor/layers/linear.py:L233-L309（锚点双置）
    # SOURCE: vllm/model_executor/layers/linear.py:L247-L294 __init__ —— 逐字
    #   minus 量化工厂分支（L283-L289）与 tp_rank/tp_size 覆写参数（DCP 域）
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py:L262-L294（锚点双置）
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        self.quant_method = UnquantizedLinearMethod()
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        # SUBTRACTED: PluggableLayer 注册面（L290-L294）——ch23 域。

    # SUBTRACTED: update_param_tp_status（L296-L309）——loader 的维度申报面
    #   （ch23 装配域）。


# SOURCE: vllm/model_executor/layers/linear.py:L419-L617 ColumnParallelLinear ——
#   前向切面（m7 的另一半：列切输出→all_gather）；__init__ 装配子集
#   （fp8 块形状豁免/MergedColumn output_sizes/loader kwargs 归 ch23/ch27）
class ColumnParallelLinear(LinearBase):
    # SOURCE: vllm/model_executor/layers/linear.py:L419-L617（锚点双置）
    # SOURCE: vllm/model_executor/layers/linear.py:L450-L531 __init__ —— 逐字
    #   minus _maybe_allow_fp8_block_shape_mismatch（L527）/output_sizes 检查
    #   （L461-L465）/weight_loader kwarg（L522-L529）/set_weight_attrs（L524-L531）
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py:L467-L531（锚点双置）
        # Divide the weight matrix along the last dimension.
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.gather_output = gather_output

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
        else:
            self.register_parameter("bias", None)
        # SUBTRACTED: set_weight_attrs 与 update_param_tp_status 的挂接
        #   （L524-L531）——ch23 装配域。

    # SOURCE: vllm/model_executor/layers/linear.py:L591-L609 forward — 逐字
    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        output_parallel = self.quant_method.apply(self, input_, bias)

        if self.gather_output and self.tp_size > 1:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


# SOURCE: vllm/model_executor/layers/linear.py:L1613-L1782 RowParallelLinear ——
#   站 11 切面：__init__ 装配子集 + forward 逐字（TP 消费现场）
class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.down_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """
    # SOURCE: vllm/model_executor/layers/linear.py:L1613-L1782（锚点双置）

    # SOURCE: vllm/model_executor/layers/linear.py:L1648-L1715 __init__ —— 逐字
    #   minus weight_loader kwarg（L1694-L1701）与 set_weight_attrs（L1709-L1714）/
    #   update_param_tp_status（L1715）——ch23 装配域。行切契约（implementer 的
    #   nn.Linear 面）：A 按列切（X 已被上游 ColumnParallel 沿特征维切过），
    #   本 rank 权重 weight.shape == [output_size, input_size/tp]，各 rank 对
    #   [tokens, in/tp] 的输入产出 [tokens, out] 部分和，all_reduce 合拢
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config=None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py:L1661-L1705（锚点双置）
        # Divide the weight matrix along the first dimension.
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
        else:
            self.register_parameter("bias", None)

    # SOURCE: vllm/model_executor/layers/linear.py:L1748-L1774 forward — 逐字
    #   （本章站 11 的本体：输入切片防御→bias 只在 rank0→all_reduce 合拢）
    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # SOURCE: vllm/model_executor/layers/linear.py:L1748-L1774（锚点双置）
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        # Matrix multiply.
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
