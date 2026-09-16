"""MoE 与 hash 路由 —— 论文忠实的小型参考实现。

论文出处（真相源）：
- arXiv:2401.06066 §3.1 Eq.(3)(4)(5)：标准 MoE 层（gate 打分 → top-k 稀疏激活 → 加权和 + 残差）；
  §3.2 Eq.(6)(7)(8)：**板斧一**细粒度切分（中间维砍到 1/m、专家数与激活数乘 m，算力不变）；
  §3.2 Eq.(9)(10)(11)：**板斧二**共享专家隔离（前 K_s 个**无条件计算**，式 (9) 第一项没有
  gate；路由只在 K_s 之后挑 mK−K_s 个）；§3.3 Eq.(12)-(17)：专家级/设备级均衡损失（谱系背景，
  V4 不再用它作主力）。
- arXiv:2106.04426（Hash Layers）：路由函数按**原始输入 token** x_t 查表、no routing
  parameters or extra terms in the objective function such as a load balancing loss。
- arXiv:2606.19348 §2.1（V4 口径）：亲合分从 Sigmoid 换成 **Sqrt(Softplus(·))**；
  均衡改用 **auxiliary-loss-free**（偏置只进选择、不进权重）；最前 `num_hash_layers` 层
  改 hash 路由（按 input token ID 的预定义 hash 函数定专家）。

实现层口径（两侧代码事实，讲代码时与公式分开说）：pin 与官方参考实现里的 hash 路由都落成
一张 `[vocab_size, topk]` 的整数表（`tid2eid`：官方 DeepSeek-V4-Pro inference/model.py:
L559 与 L576-L577；pin vllm/models/deepseek_v4/nvidia/model.py:L571-L590，
vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:L100-L106），运行时
只做一次 gather；hash **只决定"选谁"**，权重仍从**无偏**的 scores gather；
e_score_correction_bias 同理只进 `scores_for_choice`（注释点破：bias≈8.08 近均匀时用带偏
分数当权重会把分布压平）。这些是 §2.1 两条口径的字面落法，不是新机制。
"""
import numpy as np


# PAPER: arXiv:2401.06066 Eq.(5) —— 标准 MoE 的 gate 分：s = Softmax_i(uᵀe_i)
def moe_gate_scores(u, E):
    """Eq.(5)：`s_{i,t} = Softmax_i(u_tᵀ e_i)`——token 与专家中心的内积过 softmax。

    u 形状 (d,)、E 形状 (N, d)（每个专家一行中心向量）。返回 (N,)。
    """
    logits = np.asarray(E, dtype=np.float64) @ np.asarray(u, dtype=np.float64)
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()


# PAPER: arXiv:2401.06066 Eq.(4) —— top-k 稀疏激活：只留 top-k 的分值，其余置 0
def topk_gate(scores, k, start=0):
    """Eq.(4) 的 `g_{i,t}`：`s_{i,t} ∈ Topk(·)` 时取该分值、否则 0。

    `start` 对应 Eq.(10) 的口径：路由只在**第 K_s 个之后**的专家里挑（前 K_s 个是共享专家，
    不参与路由）。返回与 scores 同形的门控向量（不重新归一——论文的 g 就是 top-k 原分值）。
    """
    scores = np.asarray(scores, dtype=np.float64)
    g = np.zeros_like(scores)
    pool = np.arange(start, scores.shape[0])
    if k <= 0 or pool.size == 0:
        return g
    picked = pool[np.argsort(-scores[pool])[:k]]
    g[picked] = scores[picked]
    return g


# PAPER: arXiv:2401.06066 Eq.(3) —— 加权和 + 残差
def standard_moe_forward(u, expert_outs, g):
    """Eq.(3)：`h_t^l = Σ_i g_{i,t} FFN_i(u_t^l) + u_t^l`。

    `expert_outs` 是各专家的输出 `FFN_i(u)`（列表或 (N, d)），`g` 是 Eq.(4) 的门控。
    """
    outs = np.asarray(expert_outs, dtype=np.float64)
    return (np.asarray(g, dtype=np.float64)[:, None] * outs).sum(axis=0) + np.asarray(u, dtype=np.float64)


# PAPER: arXiv:2401.06066 Eq.(6)(7)(8) —— 板斧一：mN 个专家、激活 mK 个
def fine_grained_counts(N, K, m):
    """Eq.(6)(7)：`we segment each expert FFN into m smaller experts ... and increase the
    number of activated experts to m times to keep the same computation cost`
    ⇒ 专家数 mN、激活数 mK（返回 `(mN, mK)`）。"""
    return N * m, K * m


# PAPER: arXiv:2401.06066 §3.2 —— 组合数：切得越细，可选的"专家组合"越多
def segment_counts(N, K):
    """从 N 个专家里挑 K 个的可能组合数（摘要口径：a more flexible combination of
    activated experts）。math.comb 的直给版。"""
    import math

    return math.comb(N, K)


# PAPER: arXiv:2401.06066 Eq.(9)(10)(11) —— 板斧二：共享专家无条件计算
def shared_expert_output(shared_outs):
    """Eq.(9) 的第一项 `Σ_{i=1}^{K_s} FFN_i(u_t^l)`：**没有 gate 的求和**——共享专家的
    全部意义（capturing and consolidating common knowledge）就在这个"无条件"里。"""
    outs = np.asarray(shared_outs, dtype=np.float64)
    return outs.sum(axis=0)


# PAPER: arXiv:2401.06066 Eq.(9) —— 完整式：共享项 + 路由项 + 残差
def deepseekmoe_forward(u, shared_outs, routed_outs, routed_gates):
    """Eq.(9)：`h_t^l = Σ_{i≤K_s} FFN_i(u) + Σ_{i>K_s} g_{i,t} FFN_i(u) + u_t^l`。

    共享项（第一项）与门控无关——这是"无条件计算"的可观察后果。
    """
    routed = np.asarray(routed_outs, dtype=np.float64)
    gates = np.asarray(routed_gates, dtype=np.float64)
    return shared_expert_output(shared_outs) + (gates[:, None] * routed).sum(axis=0) + np.asarray(
        u, dtype=np.float64
    )


# PAPER: arXiv:2401.06066 §3.1 Eq.(5) 的 V3 激活 —— Sigmoid（V4 前的口径）
def affinity_scores_v3(logits):
    """V3/DeepSeekMoE 时代的亲合分：`Sigmoid(·)`——会把大 logits 压平到 1 附近。"""
    z = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


# PAPER: arXiv:2606.19348 §2.1 —— V4 换的激活：Sqrt(Softplus(·))
def affinity_scores_v4(logits):
    """§2.1 原文：`we change the activation function that computes the affinity scores
    from Sigmoid(·) into Sqrt(Softplus(·))`。

    softplus 不饱和上界（sigmoid 会把大 logits 压平到 1），开根号把分布拉平一些但不至于
    均匀。pin 的字面落法：`scores = torch.sqrt(F.softplus(gating_output))`。
    """
    z = np.asarray(logits, dtype=np.float64)
    return np.sqrt(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0))


# PAPER: arXiv:2606.19348 §2.1 —— auxiliary-loss-free：偏置只进选择、不进权重
def topk_with_correction_bias(scores, bias, k, renormalize=True, use_biased_weights=False):
    """auxiliary-loss-free 的落地形态（V4 `topk_method=noaux_tc`）：

    选择用 `scores + bias` 排序；**权重仍从无偏的 scores gather**（pin 的注释原话：用带偏
    分数当权重会在 bias 近均匀时把分布压平）。`use_biased_weights=True` 是对照组——正是
    注释点破的那个错误做法，用来看分布被压平的样子。

    返回 `(indices (k,), weights (k,))`。
    """
    scores = np.asarray(scores, dtype=np.float64)
    idx = np.argsort(-(scores + np.asarray(bias, dtype=np.float64)), kind="stable")[:k]
    if use_biased_weights:
        w = scores[idx] + np.asarray(bias, dtype=np.float64)[idx]
    else:
        w = scores[idx]
    if renormalize:
        w = w / w.sum()
    return idx, w


# PAPER: arXiv:2106.04426 路由式 + arXiv:2606.19348 §2.1 —— hash 路由：按 token id 查表
def hash_route(tid2eid, token_ids, scores, renormalize=True):
    """arXiv:2106.04426 的路由：`h^l_t = FFN_hash(x_t)(h̄^l_t)`——**按原始输入 token** 查表
    定专家；V4 §2.1 的落法是"预定义 hash 函数 + input token ID"。

    实现层形态（pin 与官方参考实现同款）：`tid2eid` 是 `[vocab_size, topk]` 的整数表，选择
    就是一次 gather；**选谁定死、给多少分现算**——权重仍从无偏的 gate 分数 gather（这也是"hash 层
    并不是没有 gate，只是 gate 不参与选择"的字面含义）。返回 `(indices (T,k), weights)`。

    `scores` 传**已过亲和激活**的分数（V4 口径 = `sqrt(softplus(gate logits))`，见
    `affinity_scores_v4`），不是原始 logits——pin 的调用点也是先算 scores 再 gather。
    """
    table = np.asarray(tid2eid)
    tokens = np.asarray(token_ids, dtype=np.int64)
    idx = table[tokens]
    s = np.asarray(scores, dtype=np.float64)
    rows = np.arange(tokens.shape[0])[:, None]
    w = s[rows, idx]
    if renormalize:
        w = w / w.sum(axis=1, keepdims=True)
    return idx, w


# PAPER: arXiv:2401.06066 Eq.(12)(13)(14) —— 专家级均衡损失（谱系背景）
def expert_balance_loss(selections, probs, alpha=1.0):
    """Eq.(12)：`L_ExpBal = α₁ Σ_i f_i P_i`，其中

        f_i = N'/(K'T)·Σ_t 𝟙(token t selects expert i)        （Eq.13）
        P_i = (1/T)·Σ_t s_{i,t}                                （Eq.14）

    归一化约定：f_i 的定义使 Σ_i f_i = N'（测试里核过）。`selections` 是每个 token 选中的
    专家下标列表、`probs` 形状 (T, N')。
    """
    s = np.asarray(probs, dtype=np.float64)
    T, Np = s.shape
    Kp = np.mean([len(x) for x in selections])
    f = np.zeros(Np)
    for sel in selections:
        for i in sel:
            f[i] += 1.0
    f = f * (Np / (Kp * T))
    P = s.mean(axis=0)
    return float(alpha * (f * P).sum())


# PAPER: arXiv:2401.06066 Eq.(15)(16)(17) —— 设备级均衡损失（谱系背景）
def device_balance_loss(f, P, groups, alpha=1.0):
    """Eq.(15)：`L_DevBal = α₂ Σ_i f'_i P'_i`，其中 `f'_i = (1/|E_i|)Σ_{j∈E_i} f_j`（Eq.16）、
    `P'_i = Σ_{j∈E_i} P_j`（Eq.17）——把专家按设备分组后再算一遍同样的均衡量。"""
    f = np.asarray(f, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    fp = np.array([f[g].mean() for g in groups])
    Pp = np.array([P[g].sum() for g in groups])
    return float(alpha * (fp * Pp).sum())
