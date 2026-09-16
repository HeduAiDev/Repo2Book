"""Lightning Indexer（压缩块上的稠密打分 + top-k 选择）—— 论文忠实的小型参考实现。

论文出处（真相源）：
- arXiv:2606.19348 §2.3.1 Eq.(13)(14)(15)(16)(17)：低秩 q（c^Q → q^I）、逐头权重 w、
  `I_{t,s} = Σ_h w^I_{t,h} · ReLU(q^I_{t,h} · K^IComp_s)` 与 top-k 选择
  `C^SprsComp_t = {C^Comp_s | I_{t,s} ∈ Top-k(I_{t,:})}`；因果条件 `s < Floor(t/m)`。
- 谱系锚 arXiv:2512.02556 §2.1 Eq.(1)(2)：同一打分函数在 V3.2 里的形态——对象是 token 的
  k^I_s，V4 换成压缩后的 K^IComp_s（打分成本随 n/m 而非 n 增长）。
- 索引器自己的压缩键 K^IComp：论文原话 "CSA performs the same compression operation used
  for C^Comp to get compressed indexer keys"，即复用 compressor.py 的同一函数，只是把
  头维换成 c^I。

实现层补充（论文 Eq.(16) 未写、两侧实现都有）：打分带两个缩放
`softmax_scale = c^I^{-1/2}` 与 `weights_scaling = n_h^I^{-1/2}` —— 官方参考实现
（DeepSeek-V4-Pro inference/model.py:L395 `self.softmax_scale = self.head_dim ** -0.5`，
L418 `weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)`）；
本仓 pin 同口径（vllm/models/deepseek_v4/attention.py:L766 与 L877-L878）。
默认不启用（按论文原式），由调用方显式传入。
"""
import numpy as np

from compressor import csa_compress


# PAPER: arXiv:2606.19348 Eq.(13)(14) —— 低秩 q：c^Q = h·W^{DQ} → q^I = c^Q·W^{IUQ}
def indexer_queries(h, W_DQ, W_IUQ, index_head_dim):
    """Eq.(13)(14)：`c_t^Q = h_t·W^{DQ}`（低秩潜向量，**与主注意力的 q 共用**，见 Eq.18）、
    `[q^I_{t,1}; ...; q^I_{t,n_h^I}] = c_t^Q·W^{IUQ}`（升维成 n_h^I 个索引头）。

    h 形状 (d,)、W_DQ (d, d_c)、W_IUQ (d_c, c^I·n_h^I)；`index_head_dim` = c^I（论文的
    索引头维，config 口径 128，与主注意力 head_dim=c 无关）。返回 (c_q (d_c,),
    q_I (n_h^I, c^I))。n_h^I 由 W_IUQ 的列数除以 c^I 反推。
    """
    c_q = h @ W_DQ
    q_I = (c_q @ W_IUQ).reshape(-1, index_head_dim)
    return c_q, q_I


# PAPER: arXiv:2606.19348 Eq.(15) —— 逐头权重 w^I_t = h_t·W^w
def indexer_head_weights(h, W_w):
    """Eq.(15)：`[w^I_{t,1}; ...; w^I_{t,n_h^I}] = w^I_t = h_t·W^w`。

    每个索引头一票的标量权重——要学的就是"哪票重要"（论文：`W^w ∈ R^{d×n_h^I}`）。
    """
    return h @ W_w


# PAPER: arXiv:2606.19348 Eq.(16) —— I_{t,s} = Σ_h w^I_{t,h}·ReLU(q^I_{t,h}·K^IComp_s)
def index_scores(q_I, k_icomp, w, softmax_scale=None, weights_scaling=None):
    """Eq.(16)：单 query 对每个压缩块打分（与 V3.2 Eq.(1) 同形，对象从 token 换成压缩块）。

    q_I 形状 (n_h^I, c^I)、k_icomp 形状 (n_blocks, c^I)、w 形状 (n_h^I,)。
    返回 I 形状 (n_blocks,)。

    ReLU 的作用：负相关不打负分，top-k 只被正相关推动。两个可选缩放是**实现层补充**
    （论文未写）：softmax_scale 乘在 ReLU 后的内积上、weights_scaling 乘在逐头权重上。
    """
    dots = np.einsum("hc,sc->hs", np.asarray(q_I, dtype=np.float64), np.asarray(k_icomp, dtype=np.float64))
    relu = np.maximum(dots, 0.0)
    if softmax_scale is not None:
        relu = relu * softmax_scale
    weights = np.asarray(w, dtype=np.float64)
    if weights_scaling is not None:
        weights = weights * weights_scaling
    return (weights[:, None] * relu).sum(axis=0)


# PAPER: arXiv:2606.19348 Eq.(16) §2.3.3 —— 因果候选数：paper 口径 s < Floor(t/m)
def causal_candidate_count(t, m, rule="paper"):
    """因果允许的压缩块个数（唯一实现方式：**候选集合本身就是因果的**）。

    - `rule="paper"`（论文 Eq.(16) 的措辞）：`s < Floor(t/m)` ⇒ 只能选**严格在自己所在块
      之前**的完整块，个数 = `t // m`。
    - `rule="impl"`（pin 与官方参考实现统一口径）：`num_compressed = (t + 1) // m`（已完成
      的块数；官方 DeepSeek-V4-Pro inference/model.py:L424-L426 与 L271、pin
      vllm/models/deepseek_v4/attention.py:L71-L88）——块尾 token 能看自己那块，仍严格因果
      （不含任何未来 token）。

    两者在"块尾 token 能否看自己的块"上差一格（dossier m06）。
    """
    if rule == "paper":
        return t // m
    if rule == "impl":
        return (t + 1) // m
    raise ValueError(f"unknown causal rule: {rule!r}")


# PAPER: arXiv:2606.19348 Eq.(17) —— Top-k 选择 + 越界位置的 −1 哨兵
def select_topk_compressed(I, topk, causal_threshold=None):
    """Eq.(17)：`C^SprsComp_t = {C^Comp_s | I_{t,s} ∈ Top-k(I_{t,:})}`。

    I 形状 (B, n_blocks)。`causal_threshold`（若给）把下标 ≥ 阈值的块置 −∞（论文的
    `s < Floor(t/m)` 约束）；候选不足 topk 的位置填 **−1 哨兵**（pin 与官方参考实现同款：
    官方 DeepSeek-V4-Pro inference/model.py:L428-L430 用 `torch.where(mask, -1, ...)`，
    pin 见 vllm/models/deepseek_v4/attention.py:L71-L88）。

    返回 `(indices (B, topk) int，valid (B, topk) bool)`。
    """
    I = np.atleast_2d(np.asarray(I, dtype=np.float64))
    B, n_blocks = I.shape
    scores = I.copy()
    if causal_threshold is not None:
        thresh = np.asarray(causal_threshold)
        if thresh.ndim == 0:
            thresh = np.full(B, int(thresh))
        for b in range(B):
            scores[b, int(thresh[b]) :] = -np.inf
    k = min(topk, n_blocks)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    taken = np.take_along_axis(scores, order, axis=1)
    valid = np.isfinite(taken)
    indices = np.where(valid, order, -1).astype(np.int64)
    if k < topk:  # 候选连 topk 都不到 ⇒ 其余位置补哨兵
        pad = np.full((B, topk - k), -1, dtype=np.int64)
        indices = np.hstack([indices, pad])
        valid = np.hstack([valid, np.zeros((B, topk - k), dtype=bool)])
    return indices, valid


# PAPER: arXiv:2606.19348 §2.3.1 —— 短上下文快路径：候选 ≤ topk ⇒ 一个不落地全选
def short_context_select_all(max_seq_len, m, topk):
    """论文侧没有这一条——它是**候选不足**时的必然形态，pin 与官方参考实现都显式短路：

    - pin（vllm/models/deepseek_v4/attention.py:L71-L88 的
      `_fill_short_context_topk_indices`）：候选数 = `(position+1)//COMPRESS_RATIO`，
      不足 topk 时把 0..n−1 全写进 topk 缓冲；
    - 官方参考实现（DeepSeek-V4-Pro inference/model.py:L409-L410）：
      `max_seq_len // compress_ratio <= index_topk` 时"每个候选都被选中"，但仍然要建 K
      cache（注释原话 candidates num smaller than topk, every candidate is selected）。

    返回 True 表示走全选快路径。
    """
    return max_seq_len // m <= topk


# PAPER: arXiv:2606.19348 §2.3.1 —— K^IComp 用"同一套压缩操作"，只是头维是 c^I
def compressed_index_keys(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m):
    """索引器的压缩键：论文原话 "performs the same compression operation used for
    C^Comp to get compressed indexer keys" —— 直接复用 CSA 的压缩函数（列宽换成 c^I）。

    这就是"索引器有自己的小头缓存、行数与主压缩 KV 同步增长"（都以 n/m 条为单位）的来历。
    """
    C_comp, _ = csa_compress(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m, return_weights=True)
    return C_comp
