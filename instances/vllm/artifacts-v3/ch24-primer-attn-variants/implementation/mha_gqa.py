"""MHA 基线与分组谱系 —— arXiv:2405.04434 §2.1.1 Eq.(1)-(8)(标准 MHA)+
arXiv:2305.13245 §2.2(GQA-g 分组插值:GQA-1=MQA、GQA-H=MHA)+ §2.1
(mean-pool 转换,uptraining 第一步)。

本章把「注意力变体」读成「KV cache 每 token 付多少元素」的账:MHA 每 token
每层缓存 2·n_h·d_h 个元素(§2.1.1 尾句),MQA 把 H 个 KV 头砍到 1(cache 与
加载量降 H 倍,§2.2 原句),GQA-g 在两者间插值。三者在代码里是同一前向的
num_kv_heads 一根参数轴——对应 vLLM QKVParallelLinear 的 total_num_kv_heads
参数(vllm/model_executor/layers/linear.py:L1022-L1101:docstring 自述「KV 头
少于 query 头(multi-query/grouped-query)时 KV 头复制、query 头切分」,
=None 默认等于 total_num_heads 即 MHA)。

权重方向一律按论文:W ∈ R^{out×in}(与 nn.Linear.weight 同向),前向算
x @ W.T。Eq.(7) 求和上限 j≤t(因果解码);论文基线式(1)-(8) 不含 RoPE
(RoPE 在 §2.1.3 才为 MLA 引入——vLLM llama.py 在式(7) 之前对 q/k 施
rotary,与本式正交)。

「trace」可选参数只逐格记录算法自身中间量的快照(供示教轨迹用),不是
论文之外的新机制。
"""
import numpy as np


# PAPER: §2.1.1 Eq.(7) —— Softmax_j 算子(对 j 维归一化;减行最大值的数值
# 稳定版,与 §2.1.1 的 Softmax 定义逐点相等)
def softmax_lastdim(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# PAPER: §2.1.1 Eq.(4)-(6) —— 逐头切分的向量化形式:[q_{t,1};…;q_{t,n_h}]=q_t
# 的逆操作:把 (T, n·d_h) 的最后一维按头切开成 (T, n, d_h)
def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    T, _ = x.shape
    d_h = x.shape[1] // num_heads
    return x.reshape(T, num_heads, d_h)


# PAPER: §2.2 —— query 头分 G 组、每组共享单个 K/V 头(连续分组,Fig.2 画法);
# 第 idx 个 query 头所在组 = idx // (H/G)。端点:G=H → 恒等(MHA)、G=1 → 全
# 共享(MQA)。与 vLLM/HF 的组映射约定一致(kv 头按 query 头 idx // 组宽取用)
def kv_head_of_query_head(num_heads: int, num_kv_heads: int) -> np.ndarray:
    assert num_heads % num_kv_heads == 0, "query 头数须能被 KV 组数整除(§2.2 分组)"
    group_width = num_heads // num_kv_heads
    return np.arange(num_heads) // group_width


# PAPER: §2.1.1 Eq.(1)-(8) + §2.2 —— MHA/GQA/MQA 同一前向:
# Eq.(1)-(3) h_t 三投影(W^Q/W^K/W^V ∈ R^{d_h·n×d})→ Eq.(4)-(6) 切头
# (K/V 按 num_kv_heads 切、组内共享 = §2.2 分组)→ Eq.(7) 逐头因果
# softmax(q_{t,i}ᵀk_{j,i}/√d_h)·v_{j,i} → Eq.(8) u_t=W^O[o_{t,1};…;o_{t,n_h}]。
# num_kv_heads=None 默认 = num_heads(即 MHA)—— 对应 vLLM QKVParallelLinear
# 的 total_num_kv_heads=None 默认(vllm/model_executor/layers/linear.py:
# L1070-L1072);num_kv_heads=1 即 MQA。
# PAPER: §2.1.1 Eq.(1)-(8)
def attention_forward(
    h: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    num_heads: int,
    num_kv_heads: int = None,
    trace: dict = None,
):
    if num_kv_heads is None:
        num_kv_heads = num_heads          # MHA 默认(=QKVParallelLinear None 语义)
    T, d = h.shape
    d_h = W_Q.shape[0] // num_heads
    q = h @ W_Q.T                          # Eq.(1):q_t = W^Q h_t
    k = h @ W_K.T                          # Eq.(2):k_t = W^K h_t
    v = h @ W_V.T                          # Eq.(3):v_t = W^V h_t
    q_heads = split_heads(q, num_heads)    # Eq.(4):(T, n_h, d_h)
    k_heads = split_heads(k, num_kv_heads)  # Eq.(5):(T, n_g, d_h) §2.2 按 KV 头数切
    v_heads = split_heads(v, num_kv_heads)  # Eq.(6)

    kv_map = kv_head_of_query_head(num_heads, num_kv_heads)   # §2.2 组映射
    # 每个 query 头实际吃到的 K/V = 所在组的共享头(广播)
    k_per_q = k_heads[:, kv_map, :]        # (T, n_h, d_h)
    v_per_q = v_heads[:, kv_map, :]

    # Eq.(7):逐头分数 q_{t,i}ᵀk_{j,i}/√d_h(因果:j<=t 才可见),布局 (n_h, T, T)
    scale = d_h ** -0.5
    scores = np.einsum("tid,sid->its", q_heads, k_per_q) * scale
    causal = np.tril(np.ones((T, T)))                       # j<=t
    scores = np.where(causal[None, :, :] > 0, scores, -np.inf)
    probs = softmax_lastdim(scores)                         # Softmax_j
    o_heads = np.einsum("its,sid->tid", probs, v_per_q)     # Eq.(7) 加权和
    u = o_heads.reshape(T, num_heads * d_h) @ W_O.T          # Eq.(8):W^O 拼回

    if trace is not None:
        trace.update(
            q=q, k=k, v=v,
            q_heads=q_heads, k_heads=k_heads, v_heads=v_heads,
            kv_head_map=kv_map.copy(),
            k_per_query_head=k_per_q, v_per_query_head=v_per_q,
            scores=scores, probs=probs, o_heads=o_heads,
            scale=scale, num_kv_heads=num_kv_heads,
        )
    return u, (k_heads, v_heads)


# PAPER: §2.1 Uptraining 第一步 + §2.2 —— MHA checkpoint → GQA 的转换:
# 「The projection matrices for key and value heads are mean pooled into
# single projection matrices」/「construct each group key and value head by
# mean-pooling all the original heads within that group」(§2.1/§2.2:优于
# 选单头或随机初始化)。线性:组头矩阵 = 组内成员头矩阵的算术平均。
# PAPER: §2.1 mean-pool 转换 + §2.2 组构造句
def mha_to_gqa_mean_pool(
    W_K: np.ndarray,
    W_V: np.ndarray,
    num_heads: int,
    head_dim: int,
    num_groups: int,
):
    assert num_heads % num_groups == 0, "头数须能被组数整除(§2.2 分组)"
    W_K_heads = W_K.reshape(num_heads, head_dim, -1)   # (H, d_h, d) 逐头切片
    W_V_heads = W_V.reshape(num_heads, head_dim, -1)
    W_K_g = W_K_heads.reshape(num_groups, num_heads // num_groups, head_dim, -1).mean(axis=1)
    W_V_g = W_V_heads.reshape(num_groups, num_heads // num_groups, head_dim, -1).mean(axis=1)
    return (
        W_K_g.reshape(num_groups * head_dim, -1),
        W_V_g.reshape(num_groups * head_dim, -1),
    )
