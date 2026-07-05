# 精简版（只做减法）— 对照真实源码 vllm_ascend/ops/register_custom_ops.py
#
# 第三条注册线（纯 Python）：用基座 vLLM 的 direct_register_custom_op 把 10 个
# torch.ops.vllm.* 各包成「真实现(op_func, dispatch PrivateUse1, 真跑) + _fake(Python 版 meta,
# 只推形状)」。本章只关心「注册范式」，故各 _xxx_impl 的通信/预取/MoE 业务分支按 dossier
# 批准裁到单一主路径并标 # SUBTRACTED（这些 NPU 真算逻辑见 ch20~ch23）；
# 所有 _xxx_fake 与 10 处 direct_register_custom_op 调用原样保留——它们是本章主角。
import torch
import torch.nn.functional as F  # 仅被已 SUBTRACTED 的 F.pad 分支使用
import torch_npu
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.rotary_embedding import rope_forward_oot
from vllm_ascend.ops.triton.muls_add import muls_add_triton
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.utils import enable_sp_by_pass, is_vl_model, npu_stream_switch, prefetch_stream


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L23
def _maybe_chunk_residual_impl(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        # SUBTRACTED: pad_size>0 时先 F.pad(residual, ...) 补齐再 chunk — flash_comm 对齐细节(ch20)；
        #             删后主路径(直接按 TP 切分)仍正确  原 vllm_ascend/ops/register_custom_ops.py:L30-L32
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]

    return residual


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L40
def _maybe_all_gather_and_maybe_unpad_impl(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()  # noqa: F841（保留以镜像源码；其消费分支已 SUBTRACTED）
    except AssertionError:
        return x

    # SUBTRACTED: flash_comm_v1 开启时的 all_gather + DP/EP unpad 多分支重排(ch20~ch22 通信)；
    #             本章只看注册范式，未开启主路径直接返回 x  原 vllm_ascend/ops/register_custom_ops.py:L46-L68
    return x


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L73
def _maybe_pad_and_reduce_impl(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    # SUBTRACTED: flash_comm_v1 下 pad + reduce_scatter 的 DP/EP 多分支(ch20~ch22)；
    #             主路径回退 all_reduce  原 vllm_ascend/ops/register_custom_ops.py:L79-L105
    return tensor_model_parallel_all_reduce(x)


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L108
def _maybe_all_gather_and_maybe_unpad_fake(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    if _EXTRA_CTX.flash_comm_v1_enabled and label:
        return torch.empty(
            (x.shape[0] * get_tensor_model_parallel_world_size(), *x.shape[1:]), device=x.device, dtype=x.dtype
        )

    return x


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L117
def _maybe_pad_and_reduce_fake(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    if _EXTRA_CTX.flash_comm_v1_enabled or enable_sp_by_pass():
        return torch.empty(
            (x.shape[0] // get_tensor_model_parallel_world_size(), *x.shape[1:]), device=x.device, dtype=x.dtype
        )

    return x


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L126
def _prefetch_preprocess_impl(weight: torch.Tensor, start_flag: torch.Tensor, max_weight_size: int) -> None:
    # SUBTRACTED: NPU 双流(计算流/预取流)切换 + maybe_npu_prefetch 提交权重预取；
    #             host 无 NPU 流，注册范式不依赖此真算  原 vllm_ascend/ops/register_custom_ops.py:L127-L131
    return


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L134
def _prefetch_preprocess_impl_fake(weight: torch.Tensor, start_flag: torch.Tensor, max_weight_size: int) -> None:
    return


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L138
def _prefetch_postprocess_impl(stop_flag: torch.Tensor) -> None:
    # SUBTRACTED: 计算流 wait 预取流的同步；host 无 NPU 流  原 vllm_ascend/ops/register_custom_ops.py:L139-L141
    return


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L144
def _prefetch_postprocess_impl_fake(stop_flag: torch.Tensor) -> None:
    return


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L148
def _maybe_all_reduce_tensor_model_parallel_impl(final_hidden_states: torch.Tensor) -> torch.Tensor:
    # SUBTRACTED: moe_comm_type∈{ALLTOALL,MC2,FUSED_MC2} 或 flash_comm 时跳过 all_reduce 的分支(ch22 MoE 通信)；
    #             主路径回退 all_reduce  原 vllm_ascend/ops/register_custom_ops.py:L149-L154
    return tensor_model_parallel_all_reduce(final_hidden_states)


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L159
def _matmul_and_reduce_impl(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.custom_op is not None
    # SUBTRACTED: bias_ 选择(tp_rank>0 / skip_bias_add 时为 None，否则 self.bias)；真算交给 custom_op
    #             原 vllm_ascend/ops/register_custom_ops.py:L163
    output = self.custom_op.matmul_and_reduce(input_parallel, None)

    return output


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L169
def _matmul_and_reduce_impl_fake(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    num_tokens = input_parallel.size(0)
    if _EXTRA_CTX.flash_comm_v1_enabled:
        num_tokens = num_tokens // self.tp_size
    output = torch.empty(
        size=(num_tokens, self.output_size_per_partition), device=input_parallel.device, dtype=input_parallel.dtype
    )

    return output


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L188
def _quantize_impl(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L194
def _quantize_impl_fake(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L200
def _rope_forward_oot_impl_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


# SOURCE: vllm_ascend/ops/register_custom_ops.py:L212
def _muls_add_impl_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="maybe_chunk_residual",
    op_func=_maybe_chunk_residual_impl,
    fake_impl=lambda x, residual: torch.empty_like(x),
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_gather_and_maybe_unpad",
    op_func=_maybe_all_gather_and_maybe_unpad_impl,
    fake_impl=_maybe_all_gather_and_maybe_unpad_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_pad_and_reduce",
    op_func=_maybe_pad_and_reduce_impl,
    fake_impl=_maybe_pad_and_reduce_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="prefetch_preprocess",
    op_func=_prefetch_preprocess_impl,
    fake_impl=_prefetch_preprocess_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="prefetch_postprocess",
    op_func=_prefetch_postprocess_impl,
    fake_impl=_prefetch_postprocess_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_reduce_tensor_model_parallel",
    op_func=_maybe_all_reduce_tensor_model_parallel_impl,
    fake_impl=lambda x: x,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="matmul_and_reduce",
    op_func=_matmul_and_reduce_impl,
    fake_impl=_matmul_and_reduce_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="quantize",
    op_func=_quantize_impl,
    fake_impl=_quantize_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="npu_rotary_embedding",
    op_func=rope_forward_oot,
    fake_impl=_rope_forward_oot_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="muls_add",
    op_func=muls_add_triton,
    fake_impl=_muls_add_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
