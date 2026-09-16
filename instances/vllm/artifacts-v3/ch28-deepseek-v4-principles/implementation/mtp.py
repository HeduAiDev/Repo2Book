"""MTP（多 token 预测）—— 论文忠实的小型参考实现。

论文出处（真相源）：arXiv:2412.19437（DeepSeek-V3 技术报告）§2.2 Eq.(21)-(25)：

    h'^k_i = M_k[RMSNorm(h^{k-1}_i) ; RMSNorm(Emb(t_{i+k}))]      (21)
    h^k_{1:T-k} = TRM_k(h'^k_{1:T-k})                             (22)
    P^k_{i+k+1} = OutHead(h^k_i)                                  (23)
    L^k_MTP = CrossEntropy(P^k_{2+k:T+1}, t_{2+k:T+1})            (24)
    L_MTP = (λ/D)·Σ_{k=1}^{D} L^k_MTP                             (25)

短句口径（原文摘录）：embedding 层与输出头**与主模型共享**；用 D 个**顺序**模块预测 D 个
额外 token，每档保持完整因果链（并且论文明确与"n 个独立输出头并行预测"区分）；`D = 1`、
`λ = 0.3`；推理期两条路——直接丢弃、或 repurpose 成投机解码的 drafter。
V4 的沿用声明见 arXiv:2606.19348 §2.1：`we adopt the same strategy for DeepSeek-V4 series
without modification`（config 口径 `num_nextn_predict_layers = 1` 与 D=1 一致）。

对齐口径（0 基下标，T 个 token）：第 k 档在位置 i 吃 `(h^{k-1}_i, Emb(t_{i+k}))`、预测
`t_{i+k+1}`；有监督的位置是 `i + k + 1 ≤ T − 1`（论文的 1 基写法 `t_{2+k:T+1}` 即此区间）。
`mtp_alignment(T, k)` 把这段下标算术显式化——它是本章最容易读错的地方（dossier m19）。

「推理期两条路」的一手对照物（dossier m19，两侧都要读）：
- **丢弃**：官方参考实现**有** MTPBlock（DeepSeek-V4-Pro inference/model.py:L738-L766，
  `n_mtp_layers=1` 见 L48，构建与权重共享见 L789-L793，式 (21) 的拼接在 L763 落成
  `e_proj(e).unsqueeze(2) + h_proj(x)`），但**推理脚本不用它**——generate.py:L51 只调主干
  `model.forward`，MTP 只在 model.py:L824-L827 的 smoke test 里被单独驱动。这正是 V3 原话
  「推理期可以直接丢弃 MTP 模块」那条路。
- **转投机解码**：pin 把 MTP 当 drafter 用（vllm/models/deepseek_v4/nvidia/mtp.py:
  L156-L188 + 主干尾部把 pre-hc_head 残差拷进 `_mtp_hidden_buffer`）——**另一条路**，别混。
- 形态差（讲代码时说清，别混进论文公式）：V4 把 V3 的融合 eh_proj 拆成 e_proj/h_proj 两段
  （官方 L742-L743 定义、L763 相加）且各自量化；MTP 的输入不是主模型的 hidden，而是主干
  尾部 **hc_head 之前**的残差（hc_mult 条流的展平态，`unsqueeze(2)` 把单流补成多流）；
  hc_head 推迟到 compute_logits（pin）里做（多流信息在这一档不丢）。
"""
import numpy as np


# PAPER: arXiv:2412.19437 Eq.(21) —— RMSNorm(h) 与 RMSNorm(Emb(t)) 两个归一
def rms_norm(x, weight=None, eps=1e-6):
    """Eq.(21) 里的两个 RMSNorm：按均方根归一（可选增益 `weight`）。

    输入形状 (..., d)，返回同形。无增益时对缩放不变。
    """
    x = np.asarray(x, dtype=np.float64)
    y = x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return y if weight is None else y * np.asarray(weight, dtype=np.float64)


# PAPER: arXiv:2412.19437 Eq.(21) —— h'^k_i = M_k[RMSNorm(h); RMSNorm(Emb(t))]
def mtp_input(h_prev, emb_next, W_mix, hnorm_w=None, enorm_w=None, eps=1e-6):
    """Eq.(21)：把「上一档隐状态」与「第 i+k 个 token 的 embedding」各自 RMSNorm 后**拼接**，
    再过投影 `M_k`（`W_mix` 形状 (2d, d)）。

    注意论文写的是**拼接**（`[· ; ·]`，M_k ∈ R^{2d×d}）——V4 的实现把这一步换成两个投影相加
    （e_proj/h_proj），是可读性更好的等价工程形态，但**不是**同一个式子（讲代码时要分开说）。
    返回 `(h_prime, concat)`：concat 是拼接后的中间量，便于逐段核对。
    """
    a = rms_norm(h_prev, hnorm_w, eps)
    b = rms_norm(emb_next, enorm_w, eps)
    concat = np.hstack([a, b])
    return concat @ np.asarray(W_mix, dtype=np.float64), concat


# PAPER: arXiv:2412.19437 Eq.(22) —— TRM_k：一个 Transformer 块（保持完整因果链）
def trm_block(h, W_q, W_k, W_v, W_o, eps=1e-6):
    """Eq.(22) 的 `TRM_k`：一个**因果** Transformer 块。

    因果性是这一档的关键（论文原话 `keep the complete causal chain at each prediction
    depth`）：位置 i 的输出只依赖 ≤ i 的输入——否则"预测第 i+k+1 个 token"就偷看了未来。
    论文只把它写成黑盒（`一个 Transformer 块`，不给内部结构），本实现取最小形态：
    RMSNorm + 单头因果自注意力 + 残差 + 一次 RMSNorm 收尾——只保留能验证因果性的那部分，
    FFN 子层不做（不发明论文没写的东西）。
    """
    h = np.asarray(h, dtype=np.float64)
    T = h.shape[0]
    x = rms_norm(h, None, eps)
    q, k, v = x @ W_q, x @ W_k, x @ W_v
    scale = q.shape[-1] ** -0.5
    logits = (q @ k.T) * scale
    causal = np.tril(np.ones((T, T), dtype=bool))
    logits = np.where(causal, logits, -np.inf)
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    w = w / w.sum(axis=-1, keepdims=True)
    attn = w @ v
    h = h + attn @ W_o
    x = rms_norm(h, None, eps)
    return h + x


# PAPER: arXiv:2412.19437 Eq.(23) —— OutHead：与主模型共享的输出头
def out_head(h, W_out):
    """Eq.(23) 的 `OutHead(h^k_i)`：输出头**与主模型共享**（论文短句 `its output head is
    shared with the main model`）——`W_out` 就是主模型 compute_logits 用的那张矩阵。"""
    return np.asarray(h, dtype=np.float64) @ np.asarray(W_out, dtype=np.float64).T


# PAPER: arXiv:2412.19437 Eq.(23) —— P^k_{i+k+1} = OutHead(h^k_i)
def mtp_predict(h, W_out):
    """Eq.(23)：第 k 档的隐状态过共享输出头得到下一个位置的 logits `P^k_{i+k+1}`。

    返回形状 (T, vocab)；**下标含义与隐状态差 k+1**（`mtp_alignment` 负责把对齐显式化）。
    """
    return out_head(h, W_out)


# PAPER: arXiv:2412.19437 §2.2（Eq.21-24 的下标算术）—— 对齐窗口
def mtp_alignment(T, k):
    """把 Eq.(23)(24) 的下标算术显式化（0 基下标，T 个 token，第 k 档）：

    - `hidden_indices`：模块有全部输入的位置 i（要求 `i + k ≤ T − 1`），
    - `targets`：参与损失的位置对 `(i, i+k+1)`（要求预测位置 `i+k+1 ≤ T − 1`），
    - `n_loss_positions` = len(targets)。

    T=5、k=1 ⇒ hidden [0,1,2,3]、targets [(0,2),(1,3),(2,4)]——即论文 1 基写法的
    `t_{3:T+1}`。注意模块算出的最后一个位置（i = T−1−k）没有目标（论文 Eq.(24) 的区间
    比 Eq.(22) 的区间短一格），这是记号边界不是实现缺陷。
    """
    hidden = [i for i in range(T) if i + k <= T - 1]
    targets = [(i, i + k + 1) for i in range(T) if i + k + 1 <= T - 1]
    return {
        "T": T,
        "k": k,
        "hidden_indices": hidden,
        "targets": targets,
        "n_loss_positions": len(targets),
    }


# PAPER: arXiv:2412.19437 Eq.(24) —— 逐深度交叉熵
def mtp_depth_loss(logits, targets):
    """Eq.(24)：`L^k_MTP = CrossEntropy(P^k_{2+k:T+1}, t_{2+k:T+1})`——对**本档有目标的
    位置**取平均交叉熵（对齐由 `mtp_alignment` 给出）。"""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    z = logits - logits.max(axis=-1, keepdims=True)
    logprob = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(-logprob[np.arange(targets.shape[0]), targets].mean())


# PAPER: arXiv:2412.19437 Eq.(25) —— 总损失 (λ/D)·Σ_k
def mtp_total_loss(losses, lam=0.3, D=1):
    """Eq.(25)：`L_MTP = (λ/D)·Σ_{k=1}^{D} L^k_MTP`——默认 `D=1`、`λ=0.3`（论文超参）。"""
    return float(lam / D * np.sum(losses))
