"""MLA —— arXiv:2405.04434 §2.1.2 Eq.(9)-(13)(低秩联合压缩 + query 侧同构)、
§2.1.3 Eq.(14)-(19)(解耦 RoPE)、§3.1.2(超参与潜向量后 RMSNorm/scale 注)、
App B.1(V2-Lite 不压 query)、App C Eq.(37)-(47)(计算序全公式 + naive 展开/
离线吸收两形态);V3 §2.1.1 重述(V3-1..V3-11)与 V2 数学一字不差。

两条等价前向(App C 尾段原句):
- naive 展开形态:"During inference, the naive formula needs to recover k^C
  and v^C from c^KV for attention"—— 用 W^UK/W^UV 把潜向量上投影回每头
  K/V(MHA 形状)再算注意力;
- 吸收形态:"we can absorb W^UK into W^UQ, and W^UV into W^O. Since this
  optimization is related to only model parameters, it can be completed
  offline at once"—— 结合律把上投影折进参数侧,c^KV 直接当「单头 K/V」参与
  attention(MQA 形状、潜向量内容;vLLM 落地脚印 = mla_attention.py
  process_weights_after_loading 的 W_UK_T/W_UV 副本,vllm/.../mla_attention.py:
  L1027-L1035/L1092-L1100)。

KV cache 侧只存蓝框两向量(App C Eq.(41)/V3-1/V3-3):c^KV(d_c 维潜向量,
§3.1.2 的 RMSNorm 之后入缓存——对应 vLLM 缓存 kv_c_normed)与共享 k^R
(d_h^R 维解耦 RoPE key,不分头)——每 token 每层 d_c+d_h^R 个元素
(§2.1.3 尾句;DSV2/V3 超参 512+64=576)。

RoPE(·) 是论文 Eq.(14)-(15) 引用的标准算子(Su et al. 2022,§3.1.4 取
θ=10000);此处按 vLLM/Llama 的半分式配对约定实现(前半×cos−后半×sin、
后半×cos+前半×sin),旋转对具体配对布局不敏感的性质(同位置内积不变)两种
约定下均成立。

「trace」可选参数只逐格记录算法自身中间量的快照(供示教轨迹用),不是
论文之外的新机制。
"""
from dataclasses import dataclass

import numpy as np

from mha_gqa import softmax_lastdim


# PAPER: §3.1.2 ——「we employ additional RMS Norm layers after the compressed
# latent vectors」:压缩潜向量(c^Q/c^KV)下投影后接 RMSNorm(x/√(mean(x²)+eps),
# 权重取单位 γ;vLLM 侧 = q_a_layernorm/kv_a_layernorm,vllm/model_executor/
# models/deepseek_v2.py:L1035/L1051)
# PAPER: §3.1.2 —— RMS Norm after the compressed latent vectors
def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)


# PAPER: §2.1.3 Eq.(14)-(15) —— RoPE(·) 算子:逐头 q^R 与共享 k^R 的位置旋转
# (θ=10000 出自 §3.1.4)。半分式配对:第 j 对 (x_j, x_{half+j}) 旋转角
# pos·base^{-2j/D}——vLLM RotaryEmbedding 同款(vllm/model_executor/layers/
# rotary_embedding.py);x 形状 (T, D) 或 (T, n_h, D) 均沿最后一维旋。
# PAPER: §2.1.3 Eq.(14)-(15) —— RoPE(·) 算子
def rope_rotate(
    x: np.ndarray, positions: np.ndarray, base: float = 10000.0
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    D = x.shape[-1]
    assert D % 2 == 0, "RoPE 维度须为偶数(2 元素配对旋转)"
    assert len(positions) == x.shape[0], "positions 逐 token(第一维)"
    half = D // 2
    pos = np.asarray(positions, dtype=np.float64)
    inv_freq = base ** (-np.arange(half) * 2.0 / D)          # θ_j = base^{-2j/D}
    angles = pos.reshape(-1, *([1] * (x.ndim - 1))) * inv_freq
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


# PAPER: App C Eq.(37)-(47) —— 论文八矩阵的形状账本:W^DQ∈R^{d_c'×d}、
# W^UQ∈R^{d_h·n_h×d_c'}、W^QR∈R^{d_h^R·n_h×d_c'}、W^DKV∈R^{d_c×d}、
# W^UK/W^UV∈R^{d_h·n_h×d_c}、W^KR∈R^{d_h^R×d}、W^O∈R^{d×d_h·n_h}(与
# vllm/model_executor/layers/attention/mla_attention.py:L44-L63 的 W_DQ…W_O
# 逐一同名同形)。q_lora_rank=None 即 App B.1 的 V2-Lite 分支:不压 query,
# W^DQ/W^UQ 缺席、由直投 W^QC∈R^{d_h·n_h×d} 代替(W^QR 改吃 h)。
# PAPER: App C Eq.(37)-(47) —— 八矩阵形状账本
@dataclass
class MLAWeights:
    d: int
    n_h: int
    d_h: int
    d_c: int
    d_h_R: int
    q_lora_rank: int | None
    W_DQ: np.ndarray | None      # (d_c', d)         Eq.(12)/(37)
    W_UQ: np.ndarray | None      # (n_h·d_h, d_c')   Eq.(13)/(42)
    W_QR: np.ndarray             # (n_h·d_h_R, d_c' 或 d)  Eq.(14)/(39)
    W_DKV: np.ndarray            # (d_c, d)          Eq.(9)/(38)
    W_UK: np.ndarray             # (n_h·d_h, d_c)    Eq.(10)/(43)
    W_KR: np.ndarray             # (d_h_R, d)        Eq.(15)/(40)
    W_UV: np.ndarray             # (n_h·d_h, d_c)    Eq.(11)/(44)
    W_O: np.ndarray              # (d, n_h·d_h)      Eq.(19)/(47)
    W_QC: np.ndarray | None = None  # (n_h·d_h, d)   V2-Lite 直投(App B.1)

    # PAPER: §2.1.2 Eq.(12)-(13)(c^Q 在)vs App B.1(缺席)
    @property
    def q_compressed(self) -> bool:
        return self.q_lora_rank is not None


# PAPER: §2.1.2 Eq.(9)-(13) + §3.1.2 —— 权重工厂:固定种子生成八矩阵
# (论文超参 d=5120/n_h=128/d_h=128/d_c=512/d_c'=1536/d_h^R=64,或缩小教具);
# q_lora_rank=None → DeepSeek-V2-Lite 不压 query 分支(App B.1:16 头/27 层/
# 不压 query,对应 vllm/.../deepseek_v2.py:L1019-L1024 的 q_lora_rank=None
# 退化装配 kv_a_proj_with_mqa + q_proj)。
# PAPER: §2.1.2 Eq.(9)-(13)+§3.1.2 —— 固定种子权重工厂
def make_mla_weights(
    d: int,
    n_h: int,
    d_h: int,
    d_c: int,
    d_h_R: int,
    q_lora_rank: int = 1536,
    seed: int = 0,
) -> MLAWeights:
    rng = np.random.default_rng(seed)
    if q_lora_rank is None:
        return MLAWeights(
            d=d, n_h=n_h, d_h=d_h, d_c=d_c, d_h_R=d_h_R, q_lora_rank=None,
            W_DQ=None, W_UQ=None,
            W_QR=rng.standard_normal((n_h * d_h_R, d)),
            W_DKV=rng.standard_normal((d_c, d)),
            W_UK=rng.standard_normal((n_h * d_h, d_c)),
            W_KR=rng.standard_normal((d_h_R, d)),
            W_UV=rng.standard_normal((n_h * d_h, d_c)),
            W_O=rng.standard_normal((d, n_h * d_h)),
            W_QC=rng.standard_normal((n_h * d_h, d)),
        )
    return MLAWeights(
        d=d, n_h=n_h, d_h=d_h, d_c=d_c, d_h_R=d_h_R, q_lora_rank=q_lora_rank,
        W_DQ=rng.standard_normal((q_lora_rank, d)),
        W_UQ=rng.standard_normal((n_h * d_h, q_lora_rank)),
        W_QR=rng.standard_normal((n_h * d_h_R, q_lora_rank)),
        W_DKV=rng.standard_normal((d_c, d)),
        W_UK=rng.standard_normal((n_h * d_h, d_c)),
        W_KR=rng.standard_normal((d_h_R, d)),
        W_UV=rng.standard_normal((n_h * d_h, d_c)),
        W_O=rng.standard_normal((d, n_h * d_h)),
    )


# PAPER: §2.1.3 Eq.(18) —— 因果掩码(求和上限 j<=t)→ Softmax_j → 加权和:
# 两条前向共用的 Eq.(47)/(18) 收尾三拍,同一 Softmax 算子(§2.1.1 Eq.(7));
# scores 布局 (n_h, T, T)、v 布局 (T, n_h, d_h) → o (T, n_h, d_h)
def _causal_softmax_weighted_sum(
    scores: np.ndarray, v: np.ndarray, T: int
) -> np.ndarray:
    causal = np.tril(np.ones((T, T)))
    scores = np.where(causal[None, :, :] > 0, scores, -np.inf)
    probs = softmax_lastdim(scores)
    # 头下标在 probs 与 v 中须同名(这里叫 i)且出现在输出里——只对 key 位置 s
    # 求和;若误写成 "hts,sid->tid",头下标 h 会被当成自由指标求和掉,
    # 输出整体放大 n_h 倍(n_h=2 时恰为 2x,实测踩过)
    return np.einsum("its,sid->tid", probs, v), probs, scores


# PAPER: App C Eq.(37)-(47) —— naive 展开形态前向(计算序与附录 C 一致):
# (37)/(38) 双下投影(+§3.1.2 RMSNorm)→ (39)/(40) RoPE 只旋 R 段 → (41)
# 蓝框两向量入缓存 → (42)/(43)/(44) 三上投影(把 k^C/v^C 从 c^KV 恢复出来)
# → (45)/(46) 拼接 [C;R] → (47) 逐头 Softmax(qᵀk/√(d_h+d_h^R))·v^C →
# W^O 拼回。返回 (u, (c^KV, k_R)):后者即每 token 写进 KV cache 的全部内容。
# PAPER: App C Eq.(37)-(47)
def mla_naive_forward(
    h: np.ndarray, positions: np.ndarray, w: MLAWeights, trace: dict = None
):
    T = h.shape[0]
    if w.q_compressed:
        c_Q = rms_norm(h @ w.W_DQ.T)                    # (37) c^Q=W^DQ h + RMSNorm
        q_in = c_Q
        q_C_flat = q_in @ w.W_UQ.T                      # (42) q^C=W^UQ c^Q
    else:   # App B.1:V2-Lite 不压 query,q^C 直接从 h 投
        c_Q = None
        q_in = h
        q_C_flat = h @ w.W_QC.T
    c_KV = rms_norm(h @ w.W_DKV.T)                      # (38) c^KV=W^DKV h + RMSNorm
    q_R = rope_rotate(                                  # (39) 逐头 q^R=RoPE(W^QR c^Q)
        (q_in @ w.W_QR.T).reshape(T, w.n_h, w.d_h_R), positions
    )
    k_R = rope_rotate(h @ w.W_KR.T, positions)          # (40) 共享单份 k^R=RoPE(W^KR h)
    cache = (c_KV, k_R)                                 # (41) 蓝框:仅此两向量需缓存
    q_C = q_C_flat.reshape(T, w.n_h, w.d_h)             # (42) [q^C_1;…;q^C_{n_h}]
    k_C = (c_KV @ w.W_UK.T).reshape(T, w.n_h, w.d_h)    # (43) k^C=W^UK c^KV(逐头恢复)
    v_C = (c_KV @ w.W_UV.T).reshape(T, w.n_h, w.d_h)    # (44) v^C=W^UV c^KV(逐头恢复)
    q = np.concatenate([q_C, q_R], axis=-1)             # (45) q_i=[q^C_i;q^R_i]
    k = np.concatenate(                                 # (46) k_i=[k^C_i;k^R](k^R 所有头共享)
        [k_C, np.broadcast_to(k_R[:, None, :], (T, w.n_h, w.d_h_R))], axis=-1
    )
    scale = 1.0 / np.sqrt(w.d_h + w.d_h_R)              # (47) 分母 √(d_h+d_h^R)
    scores = np.einsum("tid,sid->its", q, k) * scale    # (n_h, T, T) 逐头分数
    o_heads, probs, scores_masked = _causal_softmax_weighted_sum(scores, v_C, T)
    u = o_heads.reshape(T, w.n_h * w.d_h) @ w.W_O.T     # (47) u=W^O[o_1;…;o_{n_h}]

    if trace is not None:
        trace.update(
            c_Q=c_Q, c_KV=c_KV, k_R=k_R, q_R=q_R,
            q_C=q_C, k_C=k_C, v_C=v_C, q=q, k=k,
            scores=scores_masked, probs=probs, o_heads=o_heads,
            scale=scale, cache=cache,
        )
    return u, cache


# PAPER: App C 吸收句(结合律,离线一次)—— 吸收后的参数形态:
# 逐头 B_i=(W^UK_i)^T·W^UQ_i ∈ R^{d_c×d_c'}(「W^UK absorbed into W^UQ」,
# 把 query 侧投进潜空间;vLLM 脚印 = W_UK_T 重排副本 L1092-L1100)、
# W_O_absorbed=W^O·blockdiag(W^UV_1..H) ∈ R^{d×(n_h·d_c)}(「W^UV absorbed
# into W^O」,vLLM 脚印 = W_UV 转置副本)。下投影/共享 RoPE 路不动。
# PAPER: App C 吸收句 —— 吸收后的参数形态
@dataclass
class MLAWeightsAbsorbed:
    d: int
    n_h: int
    d_h: int
    d_c: int
    d_h_R: int
    q_lora_rank: int | None
    W_DQ: np.ndarray | None      # (d_c', d)  q 压缩分支保留(37)
    W_QR: np.ndarray             # (n_h·d_h_R, d_c' 或 d)  (39)
    W_DKV: np.ndarray            # (d_c, d)   (38)
    W_KR: np.ndarray             # (d_h_R, d) (40)
    B: np.ndarray                # (n_h, d_c, d_c' 或 d)  逐头吸收矩阵
    W_O_absorbed: np.ndarray     # (d, n_h·d_c)           W^UV 已折入 W^O

    # PAPER: App C(吸收不动下投影;query 压缩旋钮与吸收正交)
    @property
    def q_compressed(self) -> bool:
        return self.q_lora_rank is not None


# PAPER: App C(「related to only model parameters, it can be completed
# offline at once」)—— 离线吸收:纯参数侧变换,一次完成;q^C 路的 W^UK 与
# W^UQ 结合、输出路的 W^UV 与 W^O 结合,前向从此不必物化 k^C/v^C。
def absorb_weights(w: MLAWeights) -> MLAWeightsAbsorbed:
    W_UK_heads = w.W_UK.reshape(w.n_h, w.d_h, w.d_c)     # (n_h, d_h, d_c) 逐头
    if w.q_compressed:
        W_up_heads = w.W_UQ.reshape(w.n_h, w.d_h, w.q_lora_rank)
    else:   # App B.1:直投 W^QC 也能吸(query 压缩是独立旋钮)
        W_up_heads = w.W_QC.reshape(w.n_h, w.d_h, w.d)
    B = np.einsum("hpc,hpq->hcq", W_UK_heads, W_up_heads)  # B_i=(W^UK_i)^T W^UQ_i ∈ R^{d_c×d_c'}
    # W^UV 吸进 W^O:o_i=W^UV_i·e_i ⇒ u=W^O·blockdiag(W^UV)·[e_1;…;e_H]
    W_UV_heads = w.W_UV.reshape(w.n_h, w.d_h, w.d_c)
    blockdiag = np.zeros((w.n_h * w.d_h, w.n_h * w.d_c))
    for i in range(w.n_h):
        blockdiag[i * w.d_h:(i + 1) * w.d_h, i * w.d_c:(i + 1) * w.d_c] = W_UV_heads[i]
    return MLAWeightsAbsorbed(
        d=w.d, n_h=w.n_h, d_h=w.d_h, d_c=w.d_c, d_h_R=w.d_h_R,
        q_lora_rank=w.q_lora_rank,
        W_DQ=w.W_DQ, W_QR=w.W_QR, W_DKV=w.W_DKV, W_KR=w.W_KR,
        B=B, W_O_absorbed=w.W_O @ blockdiag,
    )


# PAPER: §2.1.2 吸收论断 + App C —— 吸收形态前向:c^KV 直接当单头 K/V 参与
# attention(MQA 形状、潜向量内容)。分数 = [q̂_i·c^KV + q^R_i·k^R]/√(d_h+d_h^R)
# —— 缩放保持 Eq.(47) 分母不变(标量乘与结合次序交换);注意力权重直接作用
# 于潜向量得 e_i=Σ_j p_j·c^KV_j,u=W_O_absorbed·[e_1;…;e_H]。缓存侧与 naive
# 完全一致(同一路 (38)/(40) 的蓝框两向量)。
# PAPER: App C 吸收形态 + Eq.(47) 收尾
def mla_absorbed_forward(
    h: np.ndarray,
    positions: np.ndarray,
    wa: MLAWeightsAbsorbed,
    trace: dict = None,
):
    T = h.shape[0]
    if wa.q_compressed:
        q_in = rms_norm(h @ wa.W_DQ.T)                  # (37)+RMSNorm,同 naive
    else:
        q_in = h                                        # App B.1 分支
    c_KV = rms_norm(h @ wa.W_DKV.T)                     # (38)+RMSNorm
    q_R = rope_rotate(                                  # (39)
        (q_in @ wa.W_QR.T).reshape(T, wa.n_h, wa.d_h_R), positions
    )
    k_R = rope_rotate(h @ wa.W_KR.T, positions)         # (40)
    cache = (c_KV, k_R)                                 # (41) 蓝框两向量(与 naive 同)
    q_latent = np.einsum("hcp,tp->thc", wa.B, q_in)     # q̂_i=B_i·q_in ∈ R^{d_c}(query 进潜空间)
    scale = 1.0 / np.sqrt(wa.d_h + wa.d_h_R)            # Eq.(47) 分母不变
    scores = (
        np.einsum("thc,sc->hts", q_latent, c_KV)        # 吸收后的 C 段分数
        + np.einsum("thr,sr->hts", q_R, k_R)            # R 段旁路(不参与吸收)
    ) * scale
    causal = np.tril(np.ones((T, T)))
    scores = np.where(causal[None] > 0, scores, -np.inf)   # 因果:j<=t(Eq.(47))
    probs = softmax_lastdim(scores)                        # Softmax_j
    e_latent = np.einsum("hts,sc->thc", probs, c_KV)       # e_i=Σ_j p_j·c^KV_j
    u = e_latent.reshape(T, wa.n_h * wa.d_c) @ wa.W_O_absorbed.T   # W^UV 已折入 W^O

    if trace is not None:
        trace.update(
            c_KV=c_KV, k_R=k_R, q_R=q_R, q_latent=q_latent,
            scores=scores, probs=probs, e_latent=e_latent,
            scale=scale, cache=cache,
        )
    return u, cache


# PAPER: App C(结合律)+ §2.1.2(「W^UK can be absorbed into W^Q」)——
# 同一个注意力分数 q^Cᵀk^C = q^Cᵀ(W^UK c^KV) 的三种结合次序:
# ① 朴素:k=W^UK·c^KV、q=W^UQ·c^Q 都物化再点积(naive 形态,逐头 K/V 恢复);
# ② 吸收:q̂=(W^UK)^T·q 只把 query 投进潜空间,与 c^KV 直接点积(decode 形态);
# ③ 参数折叠:B=(W^UK)^T·W^UQ 离线一次,前向只剩 (B·c^Q)·c^KV。
# 结合律保证三者逐位相等——这是「甚至不必把 K/V 算出来」的全部数学。
# PAPER: §2.1.2 吸收论断 + App C 结合律句
def absorption_score_three_orders(
    W_UQ_head: np.ndarray,
    W_UK_head: np.ndarray,
    c_Q: np.ndarray,
    c_KV: np.ndarray,
):
    q = W_UQ_head @ c_Q                                  # ① 的 query
    k = W_UK_head @ c_KV                                 # ① 的 key(物化)
    s_naive = q @ k
    q_latent = W_UK_head.T @ q                           # ② query 进潜空间
    s_absorbed = q_latent @ c_KV
    B = W_UK_head.T @ W_UQ_head                          # ③ 离线折叠的参数
    s_folded = (B @ c_Q) @ c_KV
    return s_naive, s_absorbed, s_folded


# PAPER: §2.1.3 ——「a RoPE matrix related to the currently generating token
# will lie between W^Q and W^UK and matrix multiplication does not obey a
# commutative law」:解耦 RoPE 动机的不交换律演示(2×2 手算教具):
# ① 直接证据:两个具体 2×2 矩阵 AB≠BA;
# ② 灾难版:若对 k^C 施 RoPE(q/k 各旋 R(m)/R(t)),本可离线折叠成单一矩阵的
#    B=(W^UK)^T·W^UQ 变成 F(m,t)=(W^UK)^T·R(m)^T·R(t)·W^UQ——随位置对变化,
#    不存在单一离线形态(吸收失效);只有 R=I(位置 0)时退化回固定 B。
#    这正是论文把位置信息解耦成旁路共享 k^R(不参与吸收)的原因。
# PAPER: §2.1.3 不交换律论证
def noncommutativity_and_rope_coupling(
    W_UQ_head: np.ndarray, W_UK_head: np.ndarray, theta: float = 1.0
) -> dict:
    AB = W_UQ_head @ W_UK_head
    BA = W_UK_head @ W_UQ_head

    # PAPER: §2.1.3 RoPE(·) 的 2 维特例(旋转矩阵,角 pos·θ)
    def rot2(pos: float) -> np.ndarray:      # 2 维 RoPE 旋转矩阵(角 pos·θ)
        c, s = np.cos(pos * theta), np.sin(pos * theta)
        return np.array([[c, -s], [s, c]])

    # PAPER: §2.1.3 —— RoPE 夹在 W^UQ 与 W^UK 之间时的「折叠矩阵」
    def folded(m: float, t: float) -> np.ndarray:
        # RoPE 夹在 W^UQ 与 W^UK 之间时的「折叠矩阵」
        return W_UK_head.T @ rot2(m).T @ rot2(t) @ W_UQ_head

    B_fixed = W_UK_head.T @ W_UQ_head        # 无 RoPE:单一固定离线折叠(App C)
    F_00, F_01 = folded(0.0, 0.0), folded(0.0, 1.0)
    return {
        "AB": AB,
        "BA": BA,
        "not_commutative": not np.allclose(AB, BA),
        "B_fixed": B_fixed,
        "F_00": F_00,                        # R(0)=I → 退化为固定 B
        "F_01": F_01,                        # 位置对一变,折叠矩阵就变
        "fold_is_position_dependent": not np.allclose(F_00, F_01),
    }
