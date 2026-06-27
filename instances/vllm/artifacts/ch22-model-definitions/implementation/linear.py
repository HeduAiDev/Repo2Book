"""ch22 companion — TP 切分线性层（只做减法）。

对应 vllm/model_executor/layers/linear.py。保留非量化主路径：
  - LinearBase: 持 quant_method（精简版恒 UnquantizedLinearMethod），记 tp_rank/tp_size。
  - ColumnParallelLinear: 沿 output 维按 tp_size 列切分。
  - MergedColumnParallelLinear: 把 [gate, up] 沿 output 维 fuse，按 shard_id 各自切分。
  - QKVParallelLinear: 把 q/k/v fuse，GQA 下 KV 头切分或复制（num_kv_head_replicas）。
  - RowParallelLinear: 沿 input 维行切分，forward 末 all_reduce 归约。

删除项见各处 # SUBTRACTED（量化/GGUF/bitsandbytes/weight_loader_v2/Phi-3 已 fuse 等分支）。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import Parameter

from ._runtime import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    set_weight_attrs,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)


# SOURCE: vllm/model_executor/layers/linear.py:183 (class UnquantizedLinearMethod)
class UnquantizedLinearMethod:
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:186 (UnquantizedLinearMethod.create_weights)
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0, "weight_loader": weight_loader})
        layer.register_parameter("weight", weight)

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/linear.py:224 (UnquantizedLinearMethod.apply)
        # SUBTRACTED: 真实经 dispatch_unquantized_gemm / batch-invariant 分派；精简版直调 F.linear。
        return torch.nn.functional.linear(x, layer.weight, bias)


# SOURCE: vllm/model_executor/layers/linear.py:235 (class LinearBase)
class LinearBase(nn.Module):
    """Base linear layer。持 quant_method 统一经其 apply 做 GEMM，记 tp_rank/tp_size。"""

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
    ) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:249 (LinearBase.__init__)
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        # SUBTRACTED: 真实 quant_config 非空时取 quant_config.get_quant_method；精简版恒非量化。
        self.quant_method = UnquantizedLinearMethod()
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1


# SOURCE: vllm/model_executor/layers/linear.py:414 (class ColumnParallelLinear)
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism。权重沿 output(列)维按 tp_size 切，input 不切。

    Y = XA + b，A 沿第二维切成 A = [A_1, ..., A_p]，每 rank 产 Y_i = XA_i（已沿 out 维切）。
    """

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
    ) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:440 (ColumnParallelLinear.__init__)
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
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
        )

        # SUBTRACTED: _maybe_allow_fp8_block_shape_mismatch（仅 fp8 block 量化用）。
        self.gather_output = gather_output

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            # SUBTRACTED: 真实按 WEIGHT_LOADER_V2_SUPPORTED 在 weight_loader/weight_loader_v2
            # 间二选一（v2 仅量化方法走）；精简版非量化恒选 v1 weight_loader。
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:537 (ColumnParallelLinear.weight_loader)
        output_dim = getattr(param, "output_dim", None)
        # SUBTRACTED: GGUF（is_gguf_weight*/materialize）、bitsandbytes/is_sharded_weight 分支。
        param_data = param.data
        if output_dim is not None:
            # 列切分：按 tp_rank 在磁盘全量上 narrow 出本 rank 的列切片。
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        # SOURCE: vllm/model_executor/layers/linear.py:582 (ColumnParallelLinear.forward)
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


# SOURCE: vllm/model_executor/layers/linear.py:611 (class MergedColumnParallelLinear)
class MergedColumnParallelLinear(ColumnParallelLinear):
    """把多段(gate, up)沿 output 维 fuse 的列并行层，按 output_sizes 各自切分。"""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:637 (MergedColumnParallelLinear.__init__)
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

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int | None = None
    ) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:696 (MergedColumnParallelLinear.weight_loader)
        # SUBTRACTED: GGUF / BlockQuantScale / packed_dim / bitsandbytes / needs_scalar_to_array
        # 分支，以及 loaded_shard_id is None（磁盘已 fuse 的 Phi-3 式 gate_up）的拆分递归路径。
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        assert loaded_shard_id is not None and loaded_shard_id < len(self.output_sizes)
        # 按 shard_id 算 fused 参数内的段 offset/size（已按 tp 切分）。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        # 再按 tp_rank narrow 出本 rank 的磁盘切片。
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


# SOURCE: vllm/model_executor/layers/linear.py:979 (class QKVParallelLinear)
class QKVParallelLinear(ColumnParallelLinear):
    """把 q/k/v fuse 成一个列并行权重。GQA 下 KV 头数<tp 时复制(num_kv_head_replicas)否则切分。"""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:1007 (QKVParallelLinear.__init__)
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
            # KV 头数 <= tp：每 rank 持 1 个 KV 头，多个 q-rank 共享同一份 KV（复制）。
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            # KV 头数 > tp：正常按 tp 切分。
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]
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

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None = None
    ) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:1189 (QKVParallelLinear.weight_loader)
        assert loaded_shard_id in ["q", "k", "v"]
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # SUBTRACTED: GGUF / BlockQuantScale / packed_dim / bitsandbytes 分支，以及
        # loaded_shard_id is None（磁盘已 fuse 的 Phi-3 式 qkv）的拆分递归路径。
        # 按 shard_id 算 fused 参数内 q/k/v 段的 offset/size（本 rank 视角）。
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

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            # q 用 tp_rank；k/v 用 tp_rank//num_kv_head_replicas（GQA 复制时多 q-rank 共享 KV）。
            if loaded_shard_id == "q":
                shard_rank = self.tp_rank
            else:
                shard_rank = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


# SOURCE: vllm/model_executor/layers/linear.py:1396 (class RowParallelLinear)
class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism。权重沿 input(行)维按 tp_size 切，forward 末 all_reduce。

    A 沿第一维切成 [A_1; ...; A_p]，X 沿第二维切成 [X_1, ..., X_p]，Y_i = X_iA_i 是部分和。
    """

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
    ) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:1431 (RowParallelLinear.__init__)
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
            weight_loader=self.weight_loader,
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )
        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        # SOURCE: vllm/model_executor/layers/linear.py:1500 (RowParallelLinear.weight_loader)
        input_dim = getattr(param, "input_dim", None)
        # SUBTRACTED: GGUF / bitsandbytes / is_sharded_weight 分支。
        param_data = param.data
        if input_dim is not None:
            # 行切分：沿 input 维按 tp_rank narrow 出本 rank 的行切片。
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        # SOURCE: vllm/model_executor/layers/linear.py:1544 (RowParallelLinear.forward)
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # SUBTRACTED: input 非并行时 split_tensor_along_last_dim 取本 rank 段；
            # Llama 里 o_proj/down_proj 的 input 来自列并行已切好的输出，恒走 input_is_parallel。
            input_parallel = input_
        # bias 只在 rank0 加（其余 rank 加完 all_reduce 会重复）。
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
