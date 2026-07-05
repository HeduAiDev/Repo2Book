"""ch36 §2.2 Eq.(1)-(8) (paper.md, arXiv:2606.19348) -- Manifold-Constrained
Hyper-Connections(mHC):把标准 Hyper-Connections 的残差映射矩阵 B_l 约束到双随机
矩阵流形(Birkhoff polytope),用 Sinkhorn-Knopp 迭代实现这个约束。

为什么要约束:标准 HC 把残差流宽度从 d 扩到 n_hc*d(Eq.1),但堆叠多层后训练容易数值
不稳定。把 B_l 约束成双随机矩阵后,‖B_l‖_2 <= 1(非扩张),且双随机矩阵集合对乘法封闭,
所以深层堆叠也稳定(Eq.2)。A_l、C_l 各自过 Sigmoid 保证非负有界,避免信号被直接置零。

落地:DeepseekV2DecoderLayer.forward(vllm_ascend/models/deepseek_v4.py:L984-1003)
用融合算子 npu_hc_pre/npu_hc_post(内含 hc_sinkhorn_iters 次迭代)包裹 attn 与 mlp
两个子层——本文件把这套融合算子在做什么,用可读的 NumPy 拆开还原。
"""
import numpy as np


# PAPER: §2.2 Eq.(1) —— X_{l+1} = B_l.X_l + C_l.F_l(A_l.X_l)。F_l_out 由调用方传入
# (F_l 是真正的 transformer 子层——attn 或 mlp,本文件不关心它的内部实现,只关心
# mHC 怎么把它的输出重新混回残差流)
def hc_residual_update(X_l: np.ndarray, A_l: np.ndarray, B_l: np.ndarray, C_l: np.ndarray,
                        F_l_out: np.ndarray) -> np.ndarray:
    """X_l:(n_hc,d);A_l:(1,n_hc);B_l:(n_hc,n_hc);C_l:(n_hc,1);F_l_out:(1,d)
    (F_l 作用在 d 维的 layer_input=A_l@X_l 上得到的输出)。返回 X_{l+1}:(n_hc,d)。"""
    return B_l @ X_l + C_l @ F_l_out


# PAPER: §2.2 "actual layer input A_l.X_l" —— F_l 真正吃进去的 d 维输入
def layer_input(X_l: np.ndarray, A_l: np.ndarray) -> np.ndarray:
    return A_l @ X_l   # (1,d)


# PAPER: §2.2 Eq.(2) —— 双随机矩阵流形 M 的定义(Birkhoff polytope):行和列和都是 1、非负
def is_doubly_stochastic(M: np.ndarray, atol: float = 1e-6) -> bool:
    return bool(
        np.all(M >= -atol)
        and np.allclose(M.sum(axis=0), 1.0, atol=atol)
        and np.allclose(M.sum(axis=1), 1.0, atol=atol)
    )


# ---------------------------------------------------------------------------
# 动态参数化(Eq.3-5)
# ---------------------------------------------------------------------------


# PAPER: §2.2 "hat X_l = RMSNorm(vec(X_l))" —— flatten 后做 RMSNorm(不是逐行,是整个
# n_hc*d 向量一起归一化)
def rms_norm_flatten(X_l: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    v = X_l.reshape(1, -1)
    return v / np.sqrt(np.mean(v ** 2) + eps)


# PAPER: §2.2 Eq.(3)-(5) —— 三个未约束的原始参数,由"输入相关的动态分量"(alpha * hat_X @ W)
# 与"输入无关的静态偏置"S 两部分相加得到
def dynamic_raw_params(X_l: np.ndarray, W_pre: np.ndarray, W_res: np.ndarray, W_post: np.ndarray,
                        S_pre: np.ndarray, S_res: np.ndarray, S_post: np.ndarray,
                        alpha_pre: float, alpha_res: float, alpha_post: float, eps: float = 1e-6):
    """W_pre,W_post:(n_hc*d, n_hc);W_res:(n_hc*d, n_hc**2)。返回 (A_tilde,B_tilde,C_tilde)。"""
    n_hc, d = X_l.shape
    x_hat = rms_norm_flatten(X_l, eps)                       # (1, n_hc*d)
    A_tilde = alpha_pre * (x_hat @ W_pre) + S_pre             # (1, n_hc)
    B_tilde = alpha_res * (x_hat @ W_res).reshape(n_hc, n_hc) + S_res   # (n_hc, n_hc)
    C_tilde = alpha_post * (x_hat @ W_post).T + S_post        # (n_hc, 1)
    return A_tilde, B_tilde, C_tilde


# ---------------------------------------------------------------------------
# 约束施加(Eq.6-8)
# ---------------------------------------------------------------------------


# PAPER: §2.2 Eq.(6)-(7) sigma(.) 算子 —— 标准 sigmoid,供 apply_sigmoid_constraints 复用
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# PAPER: §2.2 Eq.(6)-(7) —— A_l=sigmoid(A_tilde) 非负有界;C_l=2*sigmoid(C_tilde) 同理
# (乘 2 让 C_l 的值域是 [0,2) 而非 [0,1),论文原文如此)
def apply_sigmoid_constraints(A_tilde: np.ndarray, C_tilde: np.ndarray):
    return _sigmoid(A_tilde), 2.0 * _sigmoid(C_tilde)


# PAPER: §2.2 Eq.(8) —— M^(0)=exp(B_tilde);M^(t)=T_r(T_c(M^(t-1)))(先列归一化再行归一化),
# 迭代 t_max 次收敛到双随机矩阵;论文取 t_max=20 作为实用值
def sinkhorn_knopp(B_tilde: np.ndarray, iters: int = 20) -> np.ndarray:
    M = np.exp(B_tilde)
    for _ in range(iters):
        M = M / M.sum(axis=0, keepdims=True)   # T_c: 列归一化
        M = M / M.sum(axis=1, keepdims=True)   # T_r: 行归一化
    return M


# PAPER: §2.2 Eq.(6)-(8) 打包 —— 从三个原始参数一次性算出施加约束后的 A_l,B_l,C_l
def apply_constraints(A_tilde: np.ndarray, B_tilde: np.ndarray, C_tilde: np.ndarray,
                       sinkhorn_iters: int = 20):
    A_l, C_l = apply_sigmoid_constraints(A_tilde, C_tilde)
    B_l = sinkhorn_knopp(B_tilde, sinkhorn_iters)
    return A_l, B_l, C_l
