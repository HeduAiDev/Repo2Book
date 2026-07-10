"""落地对应 vllm/model_executor/layers/mla.py:L168-169（先调用 indexer 写 top-k
buffer，稀疏 MLA 后端再从同一 buffer 取索引做数值计算）与
vllm/model_executor/layers/sparse_attn_indexer.py 的 SparseAttnIndexer CustomOp——
本文件不复刻这些具体算子（FP8 kernel/CUDA 分派），而是用最小的"调用顺序 + 共享
buffer"拓扑复现同一条协议：indexer 打分/选 top-k 是纯副作用（写 buffer，
indexer 本身无有意义返回值被下游消费），下游的主注意力再从这同一个 buffer 读索引。
"""
import numpy as np

from lightning_indexer import index_score, topk_select


# PAPER: §2.1 "Instantiate DSA Under MLA" —— 对应 mla.py:L168-169 消费的共享 buffer
class TopkIndicesBuffer:
    """对应 vllm 全模型共享的 topk_indices_buffer：indexer 只写，MLA 后端只读。
    未选中/超出边界的位置用 -1 填充，与 vllm 的填充约定一致。
    """

    # PAPER: §2.1 Eq.(2) —— buffer 以 -1 填充初始化，等待 top-k 写入
    def __init__(self, num_tokens: int, topk: int):
        self.data = np.full((num_tokens, topk), -1, dtype=np.int64)

    # PAPER: §2.1 Eq.(2) —— 把 Top-k(I_{t,:}) 的结果写进共享 buffer（纯副作用）
    def write(self, token_start: int, indices: np.ndarray) -> None:
        n, k = indices.shape
        self.data[token_start : token_start + n, :k] = indices


# PAPER: §2.1 "Instantiate DSA Under MLA" —— 对应 mla.py:L168-169 的调用序：先让
# indexer（独立小头，见 lightning_indexer.IndexerConfig）对 hidden_states 打分
# （Eq.1）并把 top-k（Eq.2）写进共享 buffer，这是一次纯副作用调用。
def v32_indexer_step(
    q: np.ndarray,
    k: np.ndarray,
    w: np.ndarray,
    buffer: TopkIndicesBuffer,
    topk: int,
    token_start: int = 0,
) -> np.ndarray:
    scores = index_score(q, k, w)  # Eq.(1)
    idx = topk_select(scores, topk)  # Eq.(2)
    buffer.write(token_start, idx)  # 副作用：写共享 buffer
    return scores  # 仅供示教打印，真正被下游消费的是 buffer.data，不是这个返回值


# PAPER: §2.1 "Instantiate DSA Under MLA" —— 稀疏主注意力从共享 buffer 读 top-k
# 索引，只对选中的 latent KV 条目（c_main，MLA 的 latent 向量）做数值计算——不接触
# indexer 的 q/k/权重，两者除了这个共享 buffer 之外没有别的耦合。
def main_attention_from_buffer(
    q_main: np.ndarray,
    c_main: np.ndarray,
    buffer: TopkIndicesBuffer,
    token_start: int,
    scale: float,
) -> np.ndarray:
    t = q_main.shape[0]
    out = np.zeros_like(q_main)
    for i in range(t):
        idx = buffer.data[token_start + i]
        idx = idx[idx >= 0]  # 去掉 -1 填充
        kv = c_main[idx]
        logits = (q_main[i] @ kv.T) * scale
        m = logits.max()
        attn = np.exp(logits - m)
        attn = attn / attn.sum()
        out[i] = attn @ kv
    return out
