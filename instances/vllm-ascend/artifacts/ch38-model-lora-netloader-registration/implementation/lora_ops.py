# ch30 变体(2) LoRA 算子薄壳 —— subtract-only 精简版
#
# 真实源码 vllm_ascend/lora/lora_ops.py（122 行）：6 个 Python 薄壳，每个只做参数顺序
# 适配后转调 C++ 注册的 torch.ops._C_ascend.{bgmv,sgmv}_* NPU kernel（真 kernel 要
# NPU/CANN，host 不真跑——本章只读控制流与签名）。
#
# 按 subtraction_plan.delete 批准项：6 个薄壳同构，只内嵌 bgmv_shrink / bgmv_expand 两个
# 代表（must_keep），其余 4 个（bgmv_expand_slice / sgmv_shrink / sgmv_expand /
# sgmv_expand_slice）保留签名、删去同构函数体（都是参数 reorder 后转调 torch.ops._C_ascend.*）。

import torch


# SOURCE: vllm_ascend/lora/lora_ops.py:L19-L32
def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    return torch.ops._C_ascend.bgmv_shrink(
        inputs,
        lora_a_weights,
        lora_indices_tensor,
        output_tensor,
        scaling,
    )


# SOURCE: vllm_ascend/lora/lora_ops.py:L35-L49
def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    # 注意：薄壳丢弃 add_inputs 形参、固定传 offset=0 / size=output_tensor.size(1)，
    # 是与 C++ 签名对齐。
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
    )


# SOURCE: vllm_ascend/lora/lora_ops.py:L52-L63
def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    # SUBTRACTED: 函数体与 bgmv_expand 同构 —— 转调 torch.ops._C_ascend.bgmv_expand(
    #   inputs, lora_b_weights, lora_indices_tensor, output_tensor, slice_offset, slice_size)。
    ...


# SOURCE: vllm_ascend/lora/lora_ops.py:L66-L80
def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    # SUBTRACTED: 函数体与 bgmv_shrink 同构 —— 转调 torch.ops._C_ascend.sgmv_shrink(
    #   inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor, output_tensor, scaling)。
    ...


# SOURCE: vllm_ascend/lora/lora_ops.py:L83-L103
def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
):
    # SUBTRACTED: 函数体与 bgmv_expand 同构 —— 转调 torch.ops._C_ascend.sgmv_expand(
    #   inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor,
    #   0, output_tensor.size(1))。
    ...


# SOURCE: vllm_ascend/lora/lora_ops.py:L106-L122
def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    # SUBTRACTED: 函数体与 sgmv_expand 同构 —— 转调 torch.ops._C_ascend.sgmv_expand(
    #   ..., output_tensor, slice_offset, slice_size)。
    ...
