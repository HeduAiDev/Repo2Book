"""arXiv:2606.19348 §5.2.1 "FP4 Quantization-Aware Training" —— indexer 的 QK 路径
量化到 MXFP4（block size 32，per-block 共享 scale），index 分数从 FP32 量化到 BF16；
论文报告"2x top-k 选择器加速、99.7% 召回率"。参考实现聚焦其中可数值验证的部分：量化
引入的近似误差，以及它对 top-k 选择结果的影响（召回率）——不模拟真实硬件加速比，
那是本章参考实现无法验证的工程量。
"""
import numpy as np

from lightning_indexer import topk_select

# PAPER: §5.2.1 / 落地对应 vllm sparse_attn_indexer.py 的 MXFP4_BLOCK_SIZE = 32
MXFP4_BLOCK_SIZE = 32


# "the Query-Key (QK) path in the indexer of CSA ... cached, loaded, and
# multiplied entirely in FP4" —— MXFP4 每个 block（32 值）共享一个 scale，块内
# 每个值量化到 4-bit（仅 2^4=16 个可表示电平，比 int8/FP8 粗得多）。不做位级编码
# （2 值/字节打包见 index_cache.py），显式建模"电平数骤减"到 [-7,7] 的定点。
# PAPER: §5.2.1 "FP4 Quantization-Aware Training"
def quantize_mxfp4(x: np.ndarray, block_size: int = MXFP4_BLOCK_SIZE):
    n = x.shape[-1]
    assert n % block_size == 0
    shape = x.shape[:-1] + (n // block_size, block_size)
    blocks = x.reshape(shape)
    scale = np.maximum(np.abs(blocks).max(axis=-1), 1e-12) / 7.0
    q = np.clip(np.round(blocks / scale[..., None]), -7, 7).astype(np.int8)
    return q, scale


# PAPER: §5.2.1 "FP4 Quantization-Aware Training" —— MXFP4 反量化（供 IndexCache
# 读取/相乘前还原成浮点，对应"QK activations are cached, loaded"这句里的 loaded）。
def dequantize_mxfp4(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    deq = q.astype(np.float32) * scale[..., None]
    return deq.reshape(deq.shape[:-2] + (deq.shape[-2] * deq.shape[-1],))


# BF16 = 1 符号位 + 8 指数位 + 7 尾数位，与 FP32 共享指数范围，只是尾数被截断——用
# "截断 float32 位模式的低 16 位尾数"模拟，不引入指数范围损失（BF16 相对 FP16 的
# 设计取舍：牺牲尾数精度换指数范围）。
# PAPER: §5.2.1 —— "we further quantize the index scores I_{:,:} from FP32 to BF16"
def quantize_scores_bf16(scores: np.ndarray) -> np.ndarray:
    as_bits = scores.astype(np.float32).view(np.uint32)
    truncated = as_bits & np.uint32(0xFFFF0000)
    return truncated.view(np.float32)


# 可数值验证的部分是召回率：比较量化前后 top-k 选择集合的重合度。recall = 量化后仍
# 被选中的原 top-k 条目数 / k。不复刻具体的 99.7% 数字（真实训练权重上的经验值），
# 只验证"量化引入误差越大，召回率越低"这个方向性结论。
# PAPER: §5.2.1 —— "2x speedup for the top-k selector, ... 99.7% recall rate"
def topk_recall(
    scores_reference: np.ndarray, scores_quantized: np.ndarray, k: int
) -> float:
    idx_ref = topk_select(scores_reference, k)
    idx_q = topk_select(scores_quantized, k)
    hits = 0
    for row_ref, row_q in zip(idx_ref, idx_q):
        hits += len(set(row_ref.tolist()) & set(row_q.tolist()))
    return hits / (idx_ref.shape[0] * k)
