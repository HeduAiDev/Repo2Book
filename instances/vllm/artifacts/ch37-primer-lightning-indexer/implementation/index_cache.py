"""落地对应：vllm/model_executor/models/deepseek_v2.py 的 DeepseekV32IndexerCache 与
vllm/model_executor/layers/deepseek_v4_attention.py 的 DeepseekV4IndexerCache——lightning
indexer 自己的 KV 缓存（IndexCache），与主 KV cache 完全独立分配、独立张量布局。
arXiv:2606.19348 §2.3.1 "IndexCache 要点"（见 paper-v4.md）给出布局依据：K^IComp 的
列宽是 indexer 专属头维 c^I，与主注意力头维 c 无关。本文件用最小可跑的数组建模这份独立
缓存的写入/读取，不接触、不依赖任何"主 KV cache"对象。
"""
import numpy as np


# K^IComp ∈ R^{n/m x c^I}：形状与主压缩 KV cache C^Comp 独立（仅"块数"n/m 同步增长，
# 列宽 c^I 是 indexer 专属、与主头维 c 无关）。这里用一个不持有任何"主缓存"引用的类
# 显式建模这份独立性——它自己决定何时扩容、如何写入，读取时也只从自己的存储里取。
# PAPER: arXiv:2606.19348 §2.3.1 "IndexCache 要点"
class IndexCache:
    """indexer 专属的 KV 缓存：量化写入 + 收集读取。

    对应 vllm sparse_attn_indexer.py 的字面布局（FP8 路径）：每条 = head_dim 个
    量化字节 + head_dim/quant_block_size 个 fp32 scale。参考实现不复刻 FP8 位编码，
    而是用"定点整数 + per-block scale"模拟同一件事——量化引入的误差有界、由
    quant_block_size 控制粒度，这是 IndexCache 存在量化损失这条数值行为的忠实体现。
    """

    # PAPER: arXiv:2606.19348 §2.3.1 "IndexCache 要点" —— 独立分配、独立布局的构造
    def __init__(self, head_dim: int, quant_block_size: int = 128):
        assert head_dim % quant_block_size == 0 or quant_block_size >= head_dim
        self.head_dim = head_dim
        self.quant_block_size = min(quant_block_size, head_dim)
        self._entries: list[np.ndarray] = []  # 已写入的量化整数行
        self._scales: list[np.ndarray] = []  # 每行对应的 per-block fp32 scale

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    # PAPER: 对应"K 的量化与缓存插入融合"（vllm sparse_attn_indexer.py 的
    # indexer_k_quant_and_cache）：key 按 quant_block_size 分组量化，量化值与 scale
    # 一起写入缓存。
    def write(self, key: np.ndarray) -> None:
        assert key.shape == (self.head_dim,), "一次写入一条 head_dim 维的 key"
        q, scale = _quantize_blockwise(key, self.quant_block_size)
        self._entries.append(q)
        self._scales.append(scale)

    # PAPER: 对应 prefill 路径的 cp_gather_indexer_k_quant_cache——从已写入的历史
    # 条目里收集出反量化后的 key，供 index_score / csa_index_score 使用。
    def gather(self, start: int, end: int) -> np.ndarray:
        assert 0 <= start <= end <= self.num_entries
        rows = [
            _dequantize_blockwise(
                self._entries[i], self._scales[i], self.quant_block_size
            )
            for i in range(start, end)
        ]
        if not rows:
            return np.zeros((0, self.head_dim), dtype=np.float32)
        return np.stack(rows, axis=0)


# PAPER: arXiv:2606.19348 §2.3.1 "IndexCache 要点" —— IndexCache 量化写入的辅助函数
# （非位级 FP8 编码，用定点整数+per-block scale 模拟同一数值行为）。
def _quantize_blockwise(
    x: np.ndarray, block: int, levels: int = 127
) -> tuple[np.ndarray, np.ndarray]:
    """把 x（[head_dim]）按 block 大小分组，每组一个 scale = max(|x|)/levels，
    量化到 [-levels, levels] 的定点整数。返回 (量化整数数组, 每块 scale 数组)。
    """
    n = x.shape[0]
    assert n % block == 0
    blocks = x.reshape(n // block, block)
    scale = np.maximum(np.abs(blocks).max(axis=-1), 1e-12) / levels  # [n//block]
    q = np.clip(np.round(blocks / scale[:, None]), -levels, levels).astype(np.int16)
    return q.reshape(-1), scale


# PAPER: arXiv:2606.19348 §2.3.1 "IndexCache 要点" —— gather() 收集时的反量化辅助函数。
def _dequantize_blockwise(
    q: np.ndarray, scale: np.ndarray, block: int
) -> np.ndarray:
    blocks = q.reshape(-1, block).astype(np.float32)
    return (blocks * scale[:, None]).reshape(-1)
