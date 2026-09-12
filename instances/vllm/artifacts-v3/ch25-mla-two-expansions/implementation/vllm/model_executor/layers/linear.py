# SOURCE: vllm/model_executor/layers/linear.py
# HOST SEAM：TP 线性族的单进程退化承载。真实 LinearBase/ColumnParallel/
# MergedColumnParallel/RowParallel/Replicated 是 ch23 域的承重墙（分片
# weight_loader 算术、fused-on-disk 骨架全在彼章）；ch25 只消费其构造面与
# forward（x @ weight.T）+ disable_tp/return_bias 形参位——TP=1 下分片恒为
# 全量，语义即真实退化形态。weight 以 [out, in] 朝向持有（与真实一致——
# nn.Linear 同型初始化）。
from __future__ import annotations

import torch
from torch import nn


# SOURCE: vllm/model_executor/layers/linear.py LinearBase —— HOST SEAM 载体
class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config=None,
        return_bias: bool = True,
        prefix: str = "",
        **kwargs,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py LinearBase.__init__
        #   —— HOST SEAM：TP=1 全量位（真实按 shard_sizes 切 weight）
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.return_bias = return_bias
        self.skip_bias_add = False
        self.quant_config = quant_config
        self.params_dtype = torch.get_default_dtype()
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter("bias", None)
        # [out, in] 朝向（真实 UnquantizedLinearMethod 同型——nn.Linear 初始化；
        # requires_grad=False 与真实加载后权重一致——推理承载）
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=self.params_dtype),
            requires_grad=False,
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        # SUBTRACTED: weight_loader/shard_sizes/quant_method 装配面——ch23 域

    def forward(self, x: torch.Tensor):
        # SOURCE: vllm/model_executor/layers/linear.py UnquantizedLinearMethod
        #   .apply —— HOST SEAM：bias=None 恒走 F.linear
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        output_bias = self.bias if not self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias


# SOURCE: vllm/model_executor/layers/linear.py ColumnParallelLinear —— HOST
#   SEAM：TP=1 全量直通（真实按 TP 切输出维）
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, *, bias=False, quant_config=None,
                 gather_output=False, prefix="", return_bias=True, **kwargs):
        # SOURCE: vllm/model_executor/layers/linear.py ColumnParallelLinear
        #   —— HOST SEAM
        super().__init__(input_size, output_size, bias, quant_config,
                         return_bias, prefix, **kwargs)


# SOURCE: vllm/model_executor/layers/linear.py:L788-L834 MergedColumnParallel
#   Linear —— HOST SEAM：多段融合位（fused_qkv_a_proj 的 [Lq | Lkv+R] 段账）
class MergedColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
        disable_tp: bool = False,  # HOST SEAM：disable_tp 位（真实禁分片）
        **kwargs,
    ):
        # SOURCE: vllm/model_executor/layers/linear.py MergedColumnParallelLinear
        #   —— HOST SEAM：段拼接 [out, in]（disable_tp/TP=1 均为全量）
        self.output_sizes = output_sizes
        super().__init__(
            input_size, sum(output_sizes), bias, quant_config, True, prefix,
            **kwargs,
        )
        self.disable_tp = disable_tp


# SOURCE: vllm/model_executor/layers/linear.py ReplicatedLinear —— HOST SEAM
class ReplicatedLinear(LinearBase):
    def __init__(self, input_size, output_size, *, bias=False, quant_config=None,
                 prefix="", **kwargs):
        # SOURCE: vllm/model_executor/layers/linear.py ReplicatedLinear —— HOST SEAM
        super().__init__(input_size, output_size, bias, quant_config, True,
                         prefix, **kwargs)


# SOURCE: vllm/model_executor/layers/linear.py RowParallelLinear —— HOST SEAM：
#   TP=1 无 all-reduce（reduce_results 位保形）
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, *, bias=False, quant_config=None,
                 reduce_results=True, prefix="", return_bias=True, **kwargs):
        # SOURCE: vllm/model_executor/layers/linear.py RowParallelLinear
        #   —— HOST SEAM
        super().__init__(input_size, output_size, bias, quant_config,
                         return_bias, prefix, **kwargs)
        self.reduce_results = reduce_results
        # SUBTRACTED: TP>1 的 all_reduce 位——分布式域
