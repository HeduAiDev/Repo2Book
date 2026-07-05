"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/ops/triton/batch_invariant/matmul.py

batch-invariant 的 triton matmul：BLOCK_M/N/K 固定常量、float32 累加、allow_tf32=False，
使 K 维累加的分块数与顺序只取决于 K（与 batch 中 M 怎么拼无关）——同一行无论和谁组 batch
都走相同 reduce 顺序、逐位可复现。aten::mm/matmul/addmm/bmm 经 batch_invariant.py 替换到这里。
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def matmul_bias_persistent_kernel(
    x_ptr, y_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_yk, stride_yn, stride_bias, stride_outm, stride_outn,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # SOURCE: vllm_ascend/ops/triton/batch_invariant/matmul.py:L24
    # SUBTRACTED: kernel 指针运算全体（rm/rn/rk 索引、分块 load、acc）（原 matmul.py:L50-L98）——
    #            host 无 NPU/triton 编译环境不真跑。批不变的关键就在这被省略的循环里：
    #              acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # float32 累加
    #              for k in range(0, tl.cdiv(K, BLOCK_K)):                # 固定 BLOCK_K 分块
    #                  acc += tl.dot(x_chunk, y_chunk, allow_tf32=False)  # 关 TF32
    #            BLOCK_K 为常量 ⇒ K 维累加分块数/顺序只取决于 K，与 batch 维 M 的切分无关。
    raise NotImplementedError("triton kernel body, compiled on NPU only")


# SOURCE: vllm_ascend/ops/triton/batch_invariant/matmul.py:L101
def matmul_persistent(x, y, bias=None):
    """x @ y + bias（可选 bias），固定分块的确定性 matmul。

    x: [M, K]; y: [K, N]; bias: [N] or None → output: [M, N]
    """
    assert x.dim() == 2, "x must be a 2D tensor"
    assert y.dim() == 2, "y must be a 2D tensor"
    assert x.shape[1] == y.shape[0], f"Matrix dimension mismatch: x.shape[1]={x.shape[1]}, y.shape[0]={y.shape[0]}"

    # 转 contiguous，防转置张量的 stride 导致后续算错搬运量。
    x = x.contiguous()
    y = y.contiguous()
    M, K = x.shape
    _, N = y.shape
    if bias is not None:
        assert bias.dim() == 1, "bias must be a 1D tensor"
        assert y.shape[1] == bias.shape[0], (
            f"Bias dimension mismatch: y.shape[1]={y.shape[1]}, bias.shape[0]={bias.shape[0]}"
        )

    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    # 固定分块常量——批不变的核心：分块大小与 batch 无关。
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    if bias is None:
        dummy_bias = torch.empty(0, dtype=x.dtype, device=x.device)
        has_bias = False
        bias_stride = 0
        bias_to_pass = dummy_bias
    else:
        has_bias = True
        bias_stride = bias.stride(0)
        bias_to_pass = bias
    matmul_bias_persistent_kernel[grid](
        x, y, bias_to_pass, output,
        M, N, K,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1), bias_stride, output.stride(0), output.stride(1),
        has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return output


# SUBTRACTED: linear_persistent_kernel / linear_persistent（原 matmul.py:L178-L350）——固定网格的
#            linear 变体，与 matmul_persistent 同构、同批不变设计；保留 matmul_persistent 已足以说明原理。


# SOURCE: vllm_ascend/ops/triton/batch_invariant/matmul.py:L351
def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


# SOURCE: vllm_ascend/ops/triton/batch_invariant/matmul.py:L355
def bmm_batch_invariant(a, b, *, out=None):
    # (B, M, K) x (B, K, N) -> (B, M, N)：逐 batch 用 persistent kernel。
    if a.ndim == 3 and b.ndim == 3:
        results = [matmul_persistent(a[i], b[i]) for i in range(a.shape[0])]
        result = torch.stack(results, dim=0)
        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(f"bmm_batch_invariant expects 3D tensors, got shapes {a.shape} and {b.shape}")


# SOURCE: vllm_ascend/ops/triton/batch_invariant/matmul.py:L373
def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


# SOURCE: vllm_ascend/ops/triton/batch_invariant/matmul.py:L377
def matmul_batch_invariant(a, b, *, out=None):
    # SUBTRACTED: 3Dx2D / 2Dx3D / 4Dx4D 的 reshape 分发分支（原 matmul.py:L385-L430）——
    #            维度适配样板，保留 2Dx2D 与 3Dx3D 主路即可呈现批不变 matmul 的接入。
    if a.ndim == 2 and b.ndim == 2:
        result = matmul_persistent(a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 3 and b.ndim == 3:
        return bmm_batch_invariant(a, b, out=out)
    else:
        raise ValueError(
            f"matmul_batch_invariant (精简版) only keeps 2Dx2D / 3Dx3D, got shapes {a.shape} and {b.shape}"
        )


# SOURCE: vllm_ascend/ops/triton/batch_invariant/matmul.py:L434
def linear_batch_invariant(input_, weight, bias=None):
    # SUBTRACTED: 原走 linear_persistent；本精简版已删 linear_persistent，改走 matmul_persistent
    #            以保持可读（昇腾真实路径用专门的 linear kernel，数值等价）。
    output = matmul_persistent(input_, weight)
    if bias is not None:
        output = output + bias
    return output
