"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/batch_invariant.py

昇腾对『可复现推理』的额外保证，两步合力消除批内非确定性：
  (1) override_envs_for_invariance —— 关漂移源：weight_nz_mode=0 / matmul_allreduce off /
      HCCL_DETERMINISTIC=strict / LCCL_DETERMINISTIC=1（强制集合通信走确定性 reduce 顺序）。
  (2) enable_batch_invariant_mode —— 用 torch.library.Library 把 aten::mm/matmul/addmm/bmm/
      softmax/sum 在 NPU 上替换成固定分块、reduce 顺序固定的 batch-invariant 实现
      （AscendC 优先、triton 回退）；torch.sum/npu_add_rms_norm 这类非 dispatch 算子直接猴补。
"""
import os

import torch
import torch_npu
import vllm.envs as envs
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

# in case recursive call in reduce_sum.
torch_sum = torch.sum


if HAS_TRITON:
    from vllm_ascend.ops.triton.batch_invariant.matmul import (
        addmm_batch_invariant,
        bmm_batch_invariant,
        linear_batch_invariant,
        matmul_batch_invariant,
        mm_batch_invariant,
    )
    from vllm_ascend.ops.triton.batch_invariant.softmax import softmax_batch_invariant


try:
    import batch_invariant_ops  # type: ignore[import-not-found] # noqa

    HAS_ASCENDC_BATCH_INVARIANT = True
except ImportError:
    HAS_ASCENDC_BATCH_INVARIANT = False


# SOURCE: vllm_ascend/batch_invariant.py:L50
def add_rms_norm(x, residual, weight, eps):
    """AclnnAddRmsNorm can't ensure batch invariant, so split into add + rms_norm."""
    x_ = x + residual
    residual_ = x_
    x_, _ = torch_npu.npu_rms_norm(x_, weight, eps)
    return x_, None, residual_


# SOURCE: vllm_ascend/batch_invariant.py:L65
def reduce_sum(x: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    """torch.sum 的确定性替换——批不变的 reduce 锚点。npu_reduce_sum_batch_invariant 要求显式 dim。"""
    dim = -1 if dim is None and x.dim() == 1 else dim
    if x.device.type == "npu" and dim is not None:
        return torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x, dim, keepdim)
    # cpu tensor can't use npu kernel, fall back to torch.sum.
    return torch_sum(x, dim, keepdim)


# SOURCE: vllm_ascend/batch_invariant.py:L76
def override_envs_for_invariance():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    ascend_config.weight_nz_mode = 0          # 关 NZ 重排
    ascend_config.enable_matmul_allreduce = False  # 关 matmul-allreduce 融合

    os.environ["HCCL_DETERMINISTIC"] = "strict"  # 集合通信强制确定性 reduce 顺序
    os.environ["LCCL_DETERMINISTIC"] = "1"


_batch_invariant_LIB = None


# SOURCE: vllm_ascend/batch_invariant.py:L90
def enable_batch_invariant_mode():
    global _batch_invariant_LIB
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    # Register operators only implemented in triton.
    if HAS_TRITON:
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, "NPU")

    # Register AscendC batch-invariant ops in priority.
    if HAS_ASCENDC_BATCH_INVARIANT:
        _batch_invariant_LIB.impl("aten::mm", torch.ops.batch_invariant_ops.npu_mm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::matmul", torch.ops.batch_invariant_ops.npu_matmul_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::sum", torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant, "NPU")
        # 非 dispatch 算子直接猴补函数指针。
        torch_npu.npu_fused_infer_attention_score = (
            torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
        )
        torch_npu.npu_add_rms_norm = add_rms_norm
        torch.sum = reduce_sum
    # register triton implementations if ascendc is not available.
    elif HAS_TRITON:
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "NPU")


# SOURCE: vllm_ascend/batch_invariant.py:L126
def init_batch_invariance():
    """VLLM_BATCH_INVARIANT=1 时启动期一次性：关漂移源 env + 替换 aten 算子为确定性实现。"""
    if envs.VLLM_BATCH_INVARIANT:
        if HAS_TRITON or HAS_ASCENDC_BATCH_INVARIANT:
            logger.info("Enabling batch-invariant mode for vLLM on Ascend NPU.")
            override_envs_for_invariance()
            enable_batch_invariant_mode()
        else:
            logger.warning(
                "Batch-invariant mode requested but Triton or AscendC batch-invariant "
                "ops is not available.skipping batch-invariant initialization."
            )
