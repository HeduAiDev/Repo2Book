# SOURCE: vllm/model_executor/layers/linear.py
# HOST SEAM（ch25 同款·TP=1 退化）：本章消费的线性族——
#   ReplicatedLinear（wq_b「复制不切 TP」）/ MergedColumnParallelLinear
#   （wk_weights_proj 一枪 GEMM 出 [head_dim, n_head] 两片 / fused_wkv_wgate）/
#   ColumnParallelLinear / RowParallelLinear。
# weight 形状一律 [out, in]（torch.nn.Linear 朝向——测试按此断言防转置手滑）；
# TP=1 退化下 forward = F.linear（分片 weight_loader 算术归 ch23）。
from __future__ import annotations

import torch
from torch import nn


class _LinearBase(nn.Module):
    # SOURCE: vllm/model_executor/layers/linear.py LinearBase —— HOST SEAM
    #   TP=1 退化位（[out, in] 朝向 + F.linear）
    # SOURCE: vllm/model_executor/layers/linear.py —— HOST SEAM TP=1（锚点双置）
    def __init__(self, input_size, output_size, bias=False, prefix="",
                 return_bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.prefix = prefix
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.bias = None
        self.return_bias = return_bias

    def _init_weight(self, out_features, in_features):
        # SOURCE: vllm/model_executor/layers/linear.py —— [out, in] 朝向位
        w = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(w, mean=0.0, std=0.02)
        return w

    def forward(self, x):
        # SOURCE: vllm/model_executor/layers/linear.py forward —— HOST SEAM
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.return_bias:
            return out, self.bias
        return out


# SOURCE: vllm/model_executor/layers/linear.py ReplicatedLinear —— HOST SEAM
#   TP=1 退化（wq_b 的「no tensor parallel, just replicated」）
class ReplicatedLinear(_LinearBase):
    # SOURCE: vllm/model_executor/layers/linear.py —— HOST SEAM TP=1（锚点双置）
    def __init__(self, input_size, output_size, bias=False, quant_config=None,
                 prefix="", return_bias=True):
        # SOURCE: vllm/model_executor/layers/linear.py ReplicatedLinear.__init__
        super().__init__(input_size, output_size, bias, prefix, return_bias)
        self.weight = self._init_weight(output_size, input_size)


# SOURCE: vllm/model_executor/layers/linear.py ColumnParallelLinear —— HOST
#   SEAM TP=1 退化（整列本地）
# SOURCE: vllm/model_executor/layers/linear.py —— HOST SEAM TP=1（锚点双置）
class ColumnParallelLinear(_LinearBase):
    # SOURCE: vllm/model_executor/layers/linear.py —— HOST SEAM TP=1（锚点双置）
    def __init__(self, input_size, output_size, bias=False, quant_config=None,
                 prefix="", return_bias=True):
        super().__init__(input_size, output_size, bias, prefix, return_bias)
        self.weight = self._init_weight(output_size, input_size)


# SOURCE: vllm/model_executor/layers/linear.py MergedColumnParallelLinear
#   —— HOST SEAM TP=1 退化（多片拼接成单个 [Σout, in]——wk_weights_proj 的
#   「一枪 GEMM」几何；shard 切分按 output_sizes 算术承载）
class MergedColumnParallelLinear(_LinearBase):
    # SOURCE: vllm/model_executor/layers/linear.py —— HOST SEAM TP=1（锚点双置）
    def __init__(self, input_size, output_sizes, bias=False, quant_config=None,
                 prefix="", return_bias=True, disable_tp=False):
        super().__init__(input_size, sum(output_sizes), bias, prefix,
                         return_bias)
        self.output_sizes = list(output_sizes)
        self.disable_tp = disable_tp
        self.weight = self._init_weight(sum(output_sizes), input_size)

    def shard(self, shard_id):
        # SOURCE: vllm/model_executor/layers/linear.py weight_loader 的分片
        #   算术位——HOST SEAM（[shard_offset:shard_offset+size] 切行）
        start = sum(self.output_sizes[:shard_id])
        return slice(start, start + self.output_sizes[shard_id])


# SOURCE: vllm/model_executor/layers/linear.py RowParallelLinear —— HOST SEAM
#   TP=1 退化（无 all-reduce）
# SOURCE: vllm/model_executor/layers/linear.py —— HOST SEAM TP=1（锚点双置）
class RowParallelLinear(_LinearBase):
    # SOURCE: vllm/model_executor/layers/linear.py —— HOST SEAM TP=1（锚点双置）
    def __init__(self, input_size, output_size, bias=False, quant_config=None,
                 prefix="", reduce_results=True, return_bias=True):
        super().__init__(input_size, output_size, bias, prefix, return_bias)
        self.reduce_results = reduce_results
        self.weight = self._init_weight(output_size, input_size)
