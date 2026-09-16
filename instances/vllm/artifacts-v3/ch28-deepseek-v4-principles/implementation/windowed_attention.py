"""核心注意力（共享 KV 的 MQA）+ 滑窗支路 + 位置细节 —— 论文忠实的小型参考实现。

论文出处（真相源）：
- arXiv:2606.19348 §2.3.1 Eq.(18)(19)：选完条目后做**标准 MQA**——`q_t = c_t^Q·W^{UQ}`，
  每条压缩条目**同时当 key 和 value**（共享 KV），`o_{t,i} = CoreAttn(q_{t,i},
  C^SprsComp_t, C^SprsComp_t)`；§2.3.2 Eq.(24)(25)(26) 是 HCA 侧的同构式（key/value 换成
  整份 C^Comp，"压完不挑"）。
- arXiv:2606.19348 §2.3.3：严格因果（`each query attends to only preceding compressed KV
  blocks`、`a query cannot access information from other tokens within its own compressed
  block`）；近处更相关 ⇒ 额外产 n_win 条**未压缩** KV 与压缩条目并用；RoPE 只作用最后
  64 维、核心注意力**输出侧**按 position −i 反向旋转；HCA 层另有可学习 attention sink
  （`Exp(z'_h) will be added to the denominator of the attention score`）。

实现层口径（非论文新机制）：
- 滑窗的精确式子 `start_pos = max(pos − window_size + 1, 0)` 来自 pin
  （vllm/v1/attention/backends/mla/sparse_swa.py 的 _compute_swa_indices_and_lens_kernel）；
  滑窗条目与压缩条目在**同一次**注意力里合成（pin 的 flash_mla_with_kvcache 一核两源）。
- 压缩条目的位置口径 `(positions // compress_ratio) * compress_ratio` 来自 pin 的压缩机
  注释（vllm/models/deepseek_v4/compressor.py:L403）——条目代表整块，所以它带的是块级位置；
  query 带自己的位置，输出侧必须转回来。
- attention sink 的落法：官方参考实现的 `sparse_attn` 在线 softmax 收尾处
  `sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`（DeepSeek-V4-Pro
  inference/kernel.py:L345-L346）——**只进分母、分子（`acc_o`）里没有对应的 value 项**，
  这正是论文那句话的字面形态；pin 的可读版是 ROCm/aiter 路径的
  `l_final = l_i * alpha + exp(sink - m_final)`
  （vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:L1194-L1201）。本实现按"末列追加 sink
  logit → softmax → 该列不参与分子"的等价写法落成朴素形式。
"""
import numpy as np


# PAPER: arXiv:2606.19348 Eq.(18) —— 主注意力的 q 也从 c^Q 升维（与索引器共用同一份潜向量）
def core_queries(c_q, W_UQ, n_heads, head_dim):
    """Eq.(18)：`[q_{t,1}; ... ; q_{t,n_h}] = q_t = c_t^Q · W^{UQ}`。

    入参 `c_q` 就是 `indexer.indexer_queries` 返回的**同一个**低秩潜向量——论文原话
    `Note that the latent query vector c_t^Q is shared with that used for the indexer
    queries`。这是本章的隐藏主线：一次降维两处用（往左接索引器的 64×128 小头 q、
    往右接主注意力的 n_h×512 大 q）。

    `W_UQ` 形状 (d_c, c·n_h)（`W^{UQ} ∈ R^{d_c × c n_h}`，c = head_dim）；返回 (n_h, c)。
    官方侧同一条链在 `Attention.forward`（DeepSeek-V4-Pro inference/model.py:L496-L499：
    `qr = q = q_norm(wq_a(x))` → `wq_b(q).unflatten(...)`），且 `qr` 同一个张量被传进
    `Indexer.forward`（L411 `q = self.wq_b(qr)`）——**两份 q 同源**在代码里就是传了同一个
    `qr`。pin 侧是 fused_wqa_wkv 的低秩瓶颈一次降维、两处消费
    （vllm/models/deepseek_v4/attention.py:L226-L262 的 q 低秩链）。
    """
    q = np.asarray(c_q, dtype=np.float64) @ np.asarray(W_UQ, dtype=np.float64)
    assert q.shape[0] == n_heads * head_dim, (q.shape, n_heads, head_dim)
    return q.reshape(n_heads, head_dim)


# PAPER: arXiv:2606.19348 Eq.(19)(26) —— CoreAttn：每条条目同时当 K 和 V
def core_attention_mqa(q, kv, mask=None, sink=None, scale=None):
    """Eq.(19)/(26) 的 CoreAttn，按共享 KV 的 MQA 形态落成朴素 softmax 注意力。

    `q` 形状 (n_h, d)、`kv` 形状 (T, d)——**同一份张量既是 key 又是 value**（这就是
    "共享 KV"四个字的字面含义：head 之间共享同一份压缩条目，query 侧才多头）。
    `mask` 形状 (n_h, T)（True = 可见）承载因果与"哪些条目被选中"。
    `sink` 形状 (n_h,)：每头一个可学习 sink logit（见 softmax_with_sink）。
    `scale` 默认取 1/√d（标准注意力缩放；论文只写 CoreAttn(·)，缩放是标准惯例，
    pin 侧 `softmax_scale = head_dim ** -0.5`）。

    返回 `(o (n_h, d), weights (n_h, T))`；weights 的行和 ≤ 1（有 sink 时 < 1）。
    """
    q = np.asarray(q, dtype=np.float64)
    kv = np.asarray(kv, dtype=np.float64)
    d = q.shape[-1]
    scale = d**-0.5 if scale is None else scale
    logits = (q @ kv.T) * scale

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        logits = np.where(mask, logits, -np.inf)

    if sink is None:
        s = np.full(q.shape[0], -np.inf)  # 无 sink ⇒ exp(−inf) = 0，退化成普通 softmax
    else:
        s = np.atleast_1d(np.asarray(sink, dtype=np.float64))
        if s.shape[0] == 1 and q.shape[0] > 1:
            s = np.repeat(s, q.shape[0])

    m = np.maximum(logits.max(axis=1), s)  # 每头的 softmax 平移量（含 sink）
    # 整行被 mask 掉又没有 sink（m = −inf）时，平移量取 0：下面 e 全为 0、权重全为 0
    # （这一头这一拍什么都不看），而不是除出 NaN。
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(logits - m[:, None])
    e_sink = np.exp(s - m)
    denom = e.sum(axis=1) + e_sink
    weights = e / np.where(denom > 0, denom, 1.0)[:, None]
    o = weights @ kv
    return o, weights


# PAPER: arXiv:2606.19348 §2.3.3 —— Exp(z′) 只进 softmax 分母、分子里没有对应 value
def softmax_with_sink(logits, sink=None):
    """§2.3.3 的 attention sink 字面式子：

        `Exp(z'_h) will be added to the denominator of the attention score`

    返回 `(probs, mass)`：probs 是候选条目的注意力权重、mass 是它们的总和（= 1 − sink
    拿走的份额）。sink=None 或 −inf 时退化成普通 softmax（mass = 1）。

    **两侧初始化不可比、都不是语义**（dossier difference_list 第 2 条）：pin 的 attn_sink
    初值 −inf（exp(−inf)=0，等价"无 sink"，vllm/models/deepseek_v4/attention.py:L221-L224，
    权重装载按 TP 窄切填前 n_local_heads 槽）；官方的 `self.attn_sink =
    nn.Parameter(torch.empty(...))` 是**未初始化**的垃圾值
    （DeepSeek-V4-Pro inference/model.py:L456）。⇒ 谈默认行为只能写"取决于检查点权重"。

    论文原句：`allows each query head to adjust its total attention scores to be not equal
    to 1, and even to be near 0`。
    """
    logits = np.asarray(logits, dtype=np.float64)
    if sink is None or (np.isscalar(sink) and not np.isfinite(sink) and sink < 0):
        s = -np.inf
    else:
        s = float(np.atleast_1d(sink)[0])
    shift = max(logits.max(), s)
    e = np.exp(logits - shift)
    e_sink = np.exp(s - shift)
    denom = e.sum() + e_sink
    return e / denom, float(e.sum() / denom)


# PAPER: arXiv:2606.19348 §2.3.3 —— n_win 条未压缩 KV：start_pos = max(pos − n_win + 1, 0)
def sliding_window_slice(pos, n_win):
    """滑窗支路的可见范围：**最近的 n_win 个 token**（含 query 自己），返回 `(start, count)`。

    论文只给机制（`we additionally produce n_win uncompressed KV entries corresponding
    to the recent n_win tokens`）、不给数值；窗口式子来自 pin
    （sparse_swa.py: `start_pos = max(pos − window_size + 1, 0)`、`swa_len = end_pos −
    start_pos`）。窗口长度只看 n_win 与 t，与压缩率 m 无关。
    """
    start = max(pos - n_win + 1, 0)
    return start, pos - start + 1


# PAPER: arXiv:2606.19348 §2.3.1 Eq.(16) —— 候选块数（因果的载体）
def compressed_entry_count(pos, m, rule="impl"):
    """query 位置 `pos` 能看到的**已完成**压缩块数。

    `rule="impl"`：`(pos + 1) // m`（pin 与官方参考实现统一口径，见 indexer.causal_candidate_count）；
    `rule="paper"`：`pos // m`（论文 Eq.(16) 的 `s < Floor(t/m)` 措辞）。
    """
    if rule == "paper":
        return pos // m
    if rule == "impl":
        return (pos + 1) // m
    raise ValueError(f"unknown causal rule: {rule!r}")


# PAPER: arXiv:2606.19348 §2.3.3 —— 滑窗条目与压缩条目并进同一次注意力
def compose_attention_inputs(pos, m, n_win, rule="impl", has_compressor=True, topk=None):
    """把一拍里两路 KV 摆到一起：`KV = [最近 n_win 条未压缩 KV] + [选中的压缩条目]`。

    返回 dict：`swa_start` / `n_swa`（滑窗那一段）、`n_compressed`（压缩条目数，
    `topk` 给定时按 `min(topk, 候选数)` 截断——CSA 层；不给就是全看——HCA 层）、
    `compressed_indices` 与 `kv_len`（合成后的 KV 轴长度）。
    `has_compressor=False` 对应 `compress_ratios` 里取 0 的层：没有压缩机，滑窗支路就是
    该层的全部注意力（pin：只有 `compress_ratio > 1` 才建 DeepseekCompressor）。
    """
    start, n_swa = sliding_window_slice(pos, n_win)
    candidates = compressed_entry_count(pos, m, rule=rule) if has_compressor else 0
    n_compressed = candidates if topk is None else min(topk, candidates)
    return {
        "pos": pos,
        "swa_start": start,
        "n_swa": n_swa,
        "candidates": candidates,
        "n_compressed": n_compressed,
        "compressed_indices": list(range(n_compressed)),
        "kv_len": n_swa + n_compressed,
    }


# PAPER: arXiv:2606.19348 §2.3.3 —— 条目带块级位置：(positions // compress_ratio) × compress_ratio
def compressed_entry_positions(n_entries, m):
    """第 i 条压缩条目携带的绝对位置 = `i × m`（pin 压缩机注释的口径
    `(positions // compress_ratio) * compress_ratio`）：条目代表整块，所以它记的是块首位置。"""
    return np.arange(n_entries) * m


# PAPER: arXiv:2606.19348 §2.3.3 —— RoPE 只作用最后 64 维（前 448 维是 NoPE）
def rope_inv_freq(rope_dim, theta):
    """RoPE 的频率表：`θ_j = theta^{-2j/rope_dim}`，j = 0..rope_dim/2−1（每对共享一个 θ）。"""
    j = np.arange(rope_dim // 2, dtype=np.float64)
    return theta ** (-2.0 * j / rope_dim)


# PAPER: arXiv:2606.19348 §2.3.3 —— 正向 RoPE（偶数位：x·cos − partner·sin）
def rope_forward(x, positions, rope_dim, theta):
    """对**最后 rope_dim 维**做正向 RoPE（前 d−rope_dim 维不动，即 NoPE 段）。

    配对口径 = pin 核的 `offsets ^ 1`（相邻成对）：
    偶数维 `x·cos − partner·sin`、奇数维 `x·cos + partner·sin`。
    论文口径：RoPE 只作用最后 64 维。
    """
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    if rope_dim <= 0:
        return out
    nope = x.shape[-1] - rope_dim
    seg = x[..., nope:].reshape(*x.shape[:-1], rope_dim // 2, 2)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
    ang = pos * rope_inv_freq(rope_dim, theta)[None, :]
    c, s = np.cos(ang), np.sin(ang)
    even, odd = seg[..., 0], seg[..., 1]
    out[..., nope::2] = even * c - odd * s
    out[..., nope + 1 :: 2] = odd * c + even * s
    return out


# PAPER: arXiv:2606.19348 §2.3.3 —— 输出侧按 position −i 反向旋转（与正向反号）
def rope_inverse(x, positions, rope_dim, theta):
    """核心注意力**输出侧**的反向旋转（论文：`we also apply RoPE with position −i on the
    last 64 dimensions of each o_{t,i}`）。

    与 rope_forward 只差两个符号（pin 的 fused_inv_rope 即此式：even `x·cos + partner·sin`、
    odd `x·cos − partner·sin`）——乘回去等于没转，所以压缩条目里的块级位置信息与 query
    自己的位置能对上（不反转就错位）。
    """
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    if rope_dim <= 0:
        return out
    nope = x.shape[-1] - rope_dim
    seg = x[..., nope:].reshape(*x.shape[:-1], rope_dim // 2, 2)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
    ang = pos * rope_inv_freq(rope_dim, theta)[None, :]
    c, s = np.cos(ang), np.sin(ang)
    even, odd = seg[..., 0], seg[..., 1]
    out[..., nope::2] = even * c + odd * s
    out[..., nope + 1 :: 2] = odd * c - even * s
    return out
