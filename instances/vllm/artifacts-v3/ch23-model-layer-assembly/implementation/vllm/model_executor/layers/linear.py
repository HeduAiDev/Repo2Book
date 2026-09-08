# SOURCE: vllm/model_executor/layers/linear.py
# ch23 主角文件之二（m2/m7 承重墙）：LinearBase + ColumnParallelLinear +
# MergedColumnParallelLinear + QKVParallelLinear + RowParallelLinear——
# 三个并行线性积木与其 weight_loader 分片算术（TP 切分语义的装载侧）。
# SUBTRACTED：dossier.subtraction_plan.delete[7]（三个 weight_loader 的量化/
#   低比特特例：BlockQuantScaleParameter/Marlin/bitsandbytes_4bit/
#   is_sharded_weight 分支与 weight_loader_v2 全族）——量化装载归 ch27；
#   豁免保留（delete[7] 明示「checkpoint 格式面非量化面，不删」）：QKV 的
#   L1236-L1265/L1309-L1313 与 Merged 的 L788-L793/L823-L826（fused-on-disk
#   骨架）+ adjust_scalar_to_fused_array + validate_shard_id 的 tuple 支 +
#   needs/警告两支尾部。ReplicatedLinear/DCPGroupColumnParallelLinear/
#   MinimaxM3 索引器等特例族——章界外（Replicated 非本章面、DCP 归 ch34、
#   索引器归 ch26）。
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from torch.nn.parameter import Parameter
from typing_extensions import TypeIs

import vllm.envs as envs
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.utils import (
    dispatch_unquantized_gemm,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

# SUBTRACTED: WEIGHT_LOADER_V2_SUPPORTED 表与 register_weight_loader_v2_
#   supported_method 装饰器（linear.py:L50-L73）——量化方法名注册表，
#   weight_loader_v2 路径的选路键（delete[7]）；未量化路径恒走
#   weight_loader（v1），表与选路三元一并删除


# SUBTRACTED: adjust_marlin_shard/adjust_block_scale_shard/
#   adjust_bitsandbytes_4bit_shard（linear.py:L75-L113）——delete[7]，量化
#   装载算术归 ch27（三者的调用位都在 delete[7] 删除的量化块内）


# SOURCE: vllm/model_executor/layers/linear.py:L116-L140 adjust_scalar_to_
#   fused_array（逐字——per-tensor scale 进融合数组：delete[7] 豁免段
#   L1240-L1243/L763-L766/L874-L877 的调用位仍在，函数随之保留）
def adjust_scalar_to_fused_array(
    param_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: int | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    """
    # SOURCE: vllm/model_executor/layers/linear.py:L116-L140 adjust_scalar_to_fused_array
    qkv_idxs = {"q": 0, "k": 1, "v": 2}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    return param_data[shard_id], loaded_weight


# SOURCE: vllm/model_executor/layers/linear.py:L143 LinearMethodBase（逐字）
class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    # SOURCE: vllm/model_executor/layers/linear.py:L147 create_weights 抽象签名
    @abstractmethod
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
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        # SOURCE: vllm/model_executor/layers/linear.py:L147 create_weights 抽象签名
        raise NotImplementedError

    # SOURCE: vllm/model_executor/layers/linear.py:L172 apply 抽象签名
    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        # SOURCE: vllm/model_executor/layers/linear.py:L172 apply 抽象签名
        raise NotImplementedError


# SOURCE: vllm/model_executor/layers/linear.py:L184 UnquantizedLinearMethod
class UnquantizedLinearMethod(QuantizeMethodBase):
    """Linear method without quantization."""

    # SOURCE: vllm/model_executor/layers/linear.py:L186-L207 create_weights
    #   （逐字——ModelWeightParameter 产出 + 属性戳）
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
        # This method creates unquantized linear weights.
        # The weights are not quantized, and they are not sharded.
        # The amount of memory allocated for the weights is
        # sum(output_partition_sizes) * input_size_per_partition.
        # SOURCE: vllm/model_executor/layers/linear.py:L186-L207 create_weights
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    # SOURCE: vllm/model_executor/layers/linear.py:L216-L220
    #   process_weights_after_loading —— 减法子集（CPU 平台的
    #   dispatch_cpu_unquantized_gemm 分支删除——onednn/zentorch 平台 GEMM
    #   派发族（layers/utils.py 域）；非 CPU 路径本就无操作）
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # SUBTRACTED: current_platform.is_cpu() 的
        #   dispatch_cpu_unquantized_gemm(layer, remove_weight=True)
        #   （linear.py:L217-L220）——CPU 平台 GEMM 域
        # SOURCE: vllm/model_executor/layers/linear.py:L216-L220
        return

    # SOURCE: vllm/model_executor/layers/linear.py:L222-L233 apply —— 减法子集
    #   （batch invariant 分支删除——ch20 域，envs 默认 False 不进；
    #   dispatch 主路径逐字）
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SUBTRACTED: envs.VLLM_BATCH_INVARIANT 的 linear_batch_invariant
        #   （linear.py:L223-L225）——batch invariant GEMM 归 ch20 域
        # SOURCE: vllm/model_executor/layers/linear.py:L222-L233 apply —— 减法子集
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)


# SOURCE: vllm/model_executor/layers/linear.py:L233 LinearBase
class LinearBase(PluggableLayer):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: Prefix for parameter names.
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, tensor parallelism will be disabled for this layer.
    """

    # SOURCE: vllm/model_executor/layers/linear.py:L247-L294 __init__
    #   tp_rank/tp_size 决策面是 m7 分片算术的基准秩来源
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py:L247-L294 __init__
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
        self.allow_fp8_block_shape_mismatch = False
        self.quant_method: QuantizeMethodBase
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
            # SUBTRACTED: 量化 get_quant_method 分支的本体（linear.py:L278-L279）
            #   ——ch27 域
            self.quant_method = quant_method
        else:
            raise ValueError("All linear layers should support quant method.")
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1
        else:
            self.tp_rank = (
                tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
            )
            self.tp_size = (
                tp_size
                if tp_size is not None
                else get_tensor_model_parallel_world_size()
            )

    # SOURCE: vllm/model_executor/layers/linear.py:L296-L309 update_param_tp_status
    #   （逐字——param.tp_rank 与层 tp_rank 的调和位）
    def update_param_tp_status(self):
        # Single source of truth for a parameter's TP state. BasevLLMParameter
        # stamps self.tp_rank with the *global* rank in __init__; this reconciles
        # every child parameter to the *layer's* tp_rank/tp_size (which correctly
        # accounts for disable_tp -> replicated weights with tp_rank == 0).
        #
        # Must be re-run whenever parameters are (re-)created after construction,
        # e.g. after quant_method.process_weights_after_loading() swaps in fresh
        # Parameters. Otherwise a later load_weights()/weight-refit would narrow a
        # replicated weight at global_rank * shard_size and overflow.
        # SOURCE: vllm/model_executor/layers/linear.py:L296-L309 update_param_tp_status
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size


# SUBTRACTED: ReplicatedLinear 全类（linear.py:L312-L414）——复制型线性非本章
#   面（weight_loader 纯拷贝、无分片算术）；m7 三型只讲 Column 系/Row 系


# SOURCE: vllm/model_executor/layers/linear.py:L419 ColumnParallelLinear
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
        tp_rank: Override the tensor-parallel rank used for sharding. Defaults to
            the global TP rank. Used to shard at a coarser granularity than one
            shard per rank (see ``DCPGroupColumnParallelLinear``).
        tp_size: Override the tensor-parallel world size used for sharding.
            Defaults to the global TP world size.
    """

    # SOURCE: vllm/model_executor/layers/linear.py:L450-L531 __init__
    #   output_partition_sizes 的均分账：divide(output_size, tp_size)
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ):
        # Divide the weight matrix along the last dimension.
        # SOURCE: vllm/model_executor/layers/linear.py:L450-L531 __init__
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1
        else:
            self.tp_rank = (
                tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
            )
            self.tp_size = (
                tp_size
                if tp_size is not None
                else get_tensor_model_parallel_world_size()
            )
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size) for output_size in self.output_sizes
            ]

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
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )

        self._maybe_allow_fp8_block_shape_mismatch()
        self.gather_output = gather_output

        # SUBTRACTED: weight_loader 选路三元（linear.py:L511-L518：v2 若
        #   quant_method 类名在 WEIGHT_LOADER_V2_SUPPORTED 表中则走
        #   self.weight_loader_v2）——delete[7]：未量化路径恒 v1，直供
        #   self.weight_loader
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    # SOURCE: vllm/model_executor/layers/linear.py:L533-L558
    #   _maybe_allow_fp8_block_shape_mismatch（逐字——FP8 块对齐探测：未量化
    #   路径 quant_config=None → weight_block None → 早退无操作，故原样保留）
    def _maybe_allow_fp8_block_shape_mismatch(self) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:L533-L558
        quant_config = getattr(self, "quant_config", None)
        weight_block = getattr(quant_config, "weight_block_size", None)
        if (
            weight_block is None
            or len(weight_block) < 1
            or len(self.output_partition_sizes) <= 1
        ):
            return

        try:
            block_n = int(weight_block[0])
        except (ValueError, TypeError):
            return

        if block_n <= 0:
            return

        if any(size % block_n != 0 for size in self.output_partition_sizes):
            self.allow_fp8_block_shape_mismatch = True
            logger.debug(
                "Allowing FP8 block shape mismatch for %s (block_n=%d, partitions=%s)",
                getattr(self, "prefix", "<unknown>"),
                block_n,
                self.output_partition_sizes,
            )

    # SOURCE: vllm/model_executor/layers/linear.py:L560-L581 weight_loader
    #   减法子集（delete[7]：bnb/is_sharded_weight 分支删除；output_dim narrow
    #   主干与 scale reshape 特例逐字）
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # SOURCE: vllm/model_executor/layers/linear.py:L560-L581 weight_loader
        output_dim = getattr(param, "output_dim", None)

        # SUBTRACTED: is_sharded_weight / use_bitsandbytes_4bit 判整
        #   （linear.py:L563-L567）——delete[7]
        param_data = param.data
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    # SOURCE: vllm/model_executor/layers/linear.py:L583-L589 weight_loader_v2
    #   —— SUBTRACTED（delete[7]，量化参数路径归 ch27）

    # SOURCE: vllm/model_executor/layers/linear.py:L591-L609 forward（逐字）
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

    # SOURCE: vllm/model_executor/layers/linear.py:L611-L617 extra_repr（逐字）
    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


# SUBTRACTED: DCPGroupColumnParallelLinear 全类（linear.py:L620-L658）——DCP
#   （Decode Context Parallelism）域归 ch34


# SOURCE: vllm/model_executor/layers/linear.py:L661 MergedColumnParallelLinear
class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on the output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.gate_up_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, all weights matrix won't be sharded, this layer
                    will be treated as a "Replicated" MergedLinear.
    """

    # SOURCE: vllm/model_executor/layers/linear.py:L687-L717 __init__
    #   output_sizes 声明 + 整除断言 + sum(output_sizes) 融合账
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py:L687-L717 __init__
        self.output_sizes = output_sizes
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0

        assert all(output_size % self.tp_size == 0 for output_size in output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    # SOURCE: vllm/model_executor/layers/linear.py:L719-L744 validate_shard_id
    #   （逐字——int/None/tuple 三支；tuple 多段支的消费位在下方 fused-on-disk
    #   分支（delete[7] 豁免保留），随之保留）
    def validate_shard_id(self, shard_id: Any) -> TypeIs[int | tuple[int, ...] | None]:
        # SOURCE: vllm/model_executor/layers/linear.py:L719-L744 validate_shard_id
        if isinstance(shard_id, int):
            if shard_id < 0 or shard_id >= len(self.output_sizes):
                raise ValueError(
                    f"Shard id should be between 0 and {len(self.output_sizes) - 1}. "
                    f"Got shard id {shard_id}."
                )
            return True
        if shard_id is None:
            return True
        if isinstance(shard_id, tuple):
            for idx in shard_id:
                if not (0 <= idx < len(self.output_sizes)):
                    raise ValueError(
                        f"Shard id index {idx} should be between 0 and "
                        f"{len(self.output_sizes) - 1}. Got shard id {shard_id}."
                    )
            if len(shard_id) > 1 and any(
                b - a != 1 for a, b in zip(shard_id[:-1], shard_id[1:])
            ):
                raise ValueError(
                    "Shard id with multiple indices should be consecutive. "
                    f"Got shard id {shard_id}."
                )
            return True
        raise ValueError("This line should not be reached")

    # SOURCE: vllm/model_executor/layers/linear.py:L746-L889 weight_loader
    #   减法子集（delete[7]：量化/低比特块删除——bnb raise 整块 L779-L787、
    #   循环内量化块 L794-L810/L812-L821、单段路径 L836-L840/L842-L852/
    #   L854-L868，均连 if 头删不留悬空块；is_sharded_weight 分支头 L871
    #   随 getattr 判整删、narrow 落成无条件。豁免保留（checkpoint 格式面
    #   非量化面，不删）：fused-on-disk 骨架 L759-L778 + shard_offsets 构造
    #   与循环头 L788-L793 + narrow/递归装载尾 L823-L827 + 需要与 None/
    #   警告两支尾部 L873-L886；单段 shard_id 的段定位主干——m7 的 Merged
    #   算术：shard_offset=sum(output_sizes[:id])、//tp、tp_rank narrow）
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py:L746-L889 weight_loader
        self.validate_shard_id(loaded_shard_id)

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for per-tensor scale to load scalar into fused array.
        # SOURCE: vllm/model_executor/layers/linear.py:L757（逐字）
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        # SOURCE: vllm/model_executor/layers/linear.py:L759-L778 fused-on-disk
        #   分支头 + output_dim None 直拷 + shard_offsets 前置（delete[7] 豁免：
        #   checkpoint 格式面非量化面，不删）
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            # Loaded weight is already fused on disk (mlp).
            # (e.g., Phi-3's gate_up_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return

            output_sizes = (
                self.output_sizes[loaded_shard_id[0] : loaded_shard_id[-1] + 1]
                if loaded_shard_id is not None
                else self.output_sizes
            )
            current_shard_offset = 0
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            # SUBTRACTED: bnb raise 整块（linear.py:L779-L787，含 if 头与
            #   raise 闭括号）——delete[7]，连头删不留悬空块
            # SOURCE: vllm/model_executor/layers/linear.py:L788-L793 shard_offsets
            #   构造 + 循环头（delete[7] 豁免保留——对齐 QKV 侧同类豁免）
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # SUBTRACTED: 循环内量化块（BlockQuantScale/packed-Marlin
                #   L794-L810 与 bnb L812-L821，均连 if 头删）——delete[7]，
                #   量化装载归 ch27
                # SOURCE: vllm/model_executor/layers/linear.py:L823-L826 narrow+
                #   递归装载尾（delete[7] 豁免保留）
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
            shard_offset //= self.tp_size
            shard_size //= self.tp_size

            # SUBTRACTED: BlockQuantScaleParameter 调整（L836-L840）与 packed/
            #   Marlin 块（L842-L852）、bnb/is_sharded_weight getattr 与 bnb 块
            #   （L854-L868）——delete[7]，均连 if 头删不留悬空块；is_sharded_
            #   weight 分支头 L871 随之删，narrow 落成无条件（未量化恒 False）
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for per-tensor scales in fused case.
        # SOURCE: vllm/model_executor/layers/linear.py:L873-L877（逐字）
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )

        else:
            # SOURCE: vllm/model_executor/layers/linear.py:L879-L886 无维度
            #   警告支（逐字——assume same for all partitions）
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    # SUBTRACTED: _load_fused_module_from_checkpoint / weight_loader_v2
    #   两方法（linear.py:L891-L994）——weight_loader_v2 族与磁盘已融合特例，
    #   delete[7]

    # SOURCE: vllm/model_executor/layers/linear.py:L996-L1019 load_weights
    #   （逐字——模块级入口：shard_id 三参派发；logger.debug 观测行删除）
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        # SOURCE: vllm/model_executor/layers/linear.py:L996-L1019 load_weights
        for name, loaded_weight in weights:
            shard_id = getattr(loaded_weight, "shard_id", None)
            self.validate_shard_id(shard_id)
            # Load into self if name is not an attr of self or its submodules
            param: Parameter
            if "." in name:
                submodule, _, attr = name.rpartition(".")
                param = getattr(self.get_submodule(submodule), attr, self)
            else:
                param = getattr(self, name, self)
            if param is None and name == "bias":
                continue
            param.weight_loader(param, loaded_weight, shard_id)
            yield name


# SOURCE: vllm/model_executor/layers/linear.py:L1022 QKVParallelLinear
class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """

    # SOURCE: vllm/model_executor/layers/linear.py:L1050-L1101 __init__
    #   m2 的融合账本体：num_heads/num_kv_heads divide、num_kv_head_replicas
    #   二分支、output_sizes 三段总量）
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py:L1050-L1101 __init__
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]
        output_size = sum(self.output_sizes)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    # SOURCE: vllm/model_executor/layers/linear.py:L1103-L1109 validate_shard_id
    #   （逐字——shard_id 只认 'q'/'k'/'v'/None）
    def validate_shard_id(self, shard_id: Any) -> TypeIs[str | None]:
        # SOURCE: vllm/model_executor/layers/linear.py:L1103-L1109 validate_shard_id
        if shard_id in {"q", "k", "v"} or shard_id is None:
            return True
        raise ValueError(
            "Shard id for QKVParallelLinear should be 'q', 'k', or 'v', "
            f"got shard id {shard_id}."
        )

    # SOURCE: vllm/model_executor/layers/linear.py:L1111-L1119 _get_shard_offset_mapping
    #   （逐字——q/k/v 三段 offset 定位表）
    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        # SOURCE: vllm/model_executor/layers/linear.py:L1111-L1119 _get_shard_offset_mapping
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
            + self.num_kv_heads * self.v_head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    # SOURCE: vllm/model_executor/layers/linear.py:L1121-L1127 _get_shard_size_mapping
    #   （逐字）
    def _get_shard_size_mapping(self, loaded_shard_id: str):
        # SOURCE: vllm/model_executor/layers/linear.py:L1121-L1127 _get_shard_size_mapping
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.v_head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    # SUBTRACTED: _load_fused_module_from_checkpoint / weight_loader_v2
    #   （linear.py:L1129-L1220）——磁盘已融合与 v2 参数路径，delete[7]

    # SOURCE: vllm/model_executor/layers/linear.py:L1222-L1400 weight_loader
    #   减法子集（delete[7]：循环内量化块 L1266-L1307（BlockQuantScale
    #   L1266-L1274 + packed/Marlin L1276-L1283 + bnb L1285-L1307）与 q/k/v
    #   路径 L1329-L1374 删除、is_sharded_weight 分支头 L1382 随 getattr 判整
    #   删（narrow 落成无条件）。豁免保留（checkpoint 格式面非量化面，不删）：
    #   fused-on-disk 骨架 L1236-L1265（Phi-3 类已融合 checkpoint 的
    #   shard_offsets 构造）+ narrow/递归装载尾 L1309-L1313 + 尾部 needs/
    #   警告两支 L1385-L1397；单段 q/k/v 的段定位 + 秩算术主干——m7 的
    #   QKV worked example 本体）
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py:L1222-L1400 weight_loader
        self.validate_shard_id(loaded_shard_id)

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        # Special case for per-tensor scales in fused case.
        # SOURCE: vllm/model_executor/layers/linear.py:L1233-L1234（逐字）
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        # SOURCE: vllm/model_executor/layers/linear.py:L1236-L1265 磁盘已融合
        #   分支头 + output_dim None 直拷 + shard_offsets 构造（delete[7] 豁免：
        #   Phi-3 类已融合 checkpoint 的段位账，checkpoint 格式面非量化面）
        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv).
            # (e.g., Phi-3's qkv_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.v_head_size,
                ),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # SUBTRACTED: 循环内量化块（BlockQuantScale L1266-L1274 +
                #   packed/Marlin L1276-L1283 + bnb L1285-L1307，均连 if 头删）
                #   ——delete[7]，量化装载归 ch27
                # SOURCE: vllm/model_executor/layers/linear.py:L1309-L1313 narrow+
                #   递归装载尾（delete[7] 豁免保留）
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.v_head_size

            # SUBTRACTED: BlockQuantScaleParameter 调整（linear.py:L1329-L1333）
            #   与 packed_dim/Marlin/bnb 调整段（L1335-L1373）、bnb/is_sharded_
            #   weight getattr（L1348-L1352）——delete[7]，均连 if 头删不留
            #   悬空块；is_sharded_weight 分支头 L1382 随之删，narrow 落成
            #   无条件（未量化恒 False）
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_rank = self.tp_rank
            else:
                shard_rank = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for per-tensor scales in fused case.
        # SOURCE: vllm/model_executor/layers/linear.py:L1385-L1389（逐字）
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )
        else:
            # SOURCE: vllm/model_executor/layers/linear.py:L1390-L1397 无维度
            #   警告支（逐字——assume the weight is the same for all partitions）
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    # SOURCE: vllm/model_executor/layers/linear.py:L1402-L1425 load_weights
    #   （逐字——模块级入口：从张量属性读 shard_id（WeightsMapper.apply 盖的
    #   章）、三参派发 weight_loader；logger.debug 观测行删除）
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        # SOURCE: vllm/model_executor/layers/linear.py:L1402-L1425 load_weights
        for name, loaded_weight in weights:
            shard_id = getattr(loaded_weight, "shard_id", None)
            self.validate_shard_id(shard_id)
            # Load into self if name is not an attr of self or its submodules
            param: Parameter
            if "." in name:
                submodule, _, attr = name.rpartition(".")
                param = getattr(self.get_submodule(submodule), attr, self)
            else:
                param = getattr(self, name, self)
            if param is None and name == "bias":
                continue
            param.weight_loader(param, loaded_weight, shard_id)
            yield name


# SUBTRACTED: MinimaxM3QKVParallelLinearWithIndexer（linear.py:L1428-L1612）
#   ——闪电索引器特例归 ch26


# SOURCE: vllm/model_executor/layers/linear.py:L1613 RowParallelLinear
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
                       which is Y_i = X_iA_i
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.down_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """

    # SOURCE: vllm/model_executor/layers/linear.py:L1648-L1715 __init__
    #   input 维行切账：divide(input_size, tp_size)
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # Divide the weight matrix along the first dimension.
        # SOURCE: vllm/model_executor/layers/linear.py:L1648-L1715 __init__
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

        # SUBTRACTED: weight_loader_v2 选路三元（linear.py:L1692-L1696）——
        #   delete[7]
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    # SOURCE: vllm/model_executor/layers/linear.py:L1717-L1737 weight_loader
    #   减法子集（delete[7]：bnb/is_sharded_weight 判整删除；input_dim 行切
    #   主干与 scale reshape 特例逐字）
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # SOURCE: vllm/model_executor/layers/linear.py:L1717-L1737 weight_loader
        input_dim = getattr(param, "input_dim", None)
        # SUBTRACTED: use_bitsandbytes_4bit / is_sharded_weight 判整
        #   （linear.py:L1719-L1723）——delete[7]

        param_data = param.data
        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    # SOURCE: vllm/model_executor/layers/linear.py:L1739-L1746 weight_loader_v2
    #   —— SUBTRACTED（delete[7]，量化参数路径归 ch27）

    # SOURCE: vllm/model_executor/layers/linear.py:L1748-L1774 forward
    #   行切的计算侧对偶：input_is_parallel 直入、bias 只 rank0 加、
    #   reduce_results 时 all_reduce）
    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # SOURCE: vllm/model_executor/layers/linear.py:L1748-L1774 forward
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

    # SOURCE: vllm/model_executor/layers/linear.py:L1776-L1782 extra_repr（逐字）
    def extra_repr(self) -> str:
        s = f"in_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
