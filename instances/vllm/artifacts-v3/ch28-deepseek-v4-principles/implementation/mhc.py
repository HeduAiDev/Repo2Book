"""mHC（Manifold-Constrained Hyper-Connections）—— 论文忠实的小型参考实现。

论文出处（真相源）：
- arXiv:2606.19348 §2.2 Eq.(1)-(8)（**本章正文采用的口径**，符号是 A/B/C）：
  `X_{l+1} = B_l X_l + C_l F_l(A_l X_l)`（Eq.1）、`B_l` 落在 Birkhoff 多面体（Eq.2）、
  三条 raw 映射由展平 RMSNorm 态一次生成（Eq.3)(4)(5)）、`A_l = σ(Ã_l)`（Eq.6）、
  `C_l = 2σ(C̃_l)`（Eq.7）、Sinkhorn 迭代 `M^(t) = T_r(T_c(M^(t-1)))`（Eq.8）、
  `t_max = 20 as a practical value`。
- arXiv:2512.24880（mHC 本尊）Eq.(3)(5)(6)(7)(8)(9) + §4.1：三映射 H^pre/H^post/H^res、
  输入相关参数化、Birkhoff 投影、σ/2σ/Sinkhorn-Knopp 三件套、三条性质（‖H^res‖₂ ≤ 1 /
  恢复恒等映射 / 对乘法封闭）。
- arXiv:2409.19606（前身 HC）：多流残差与「恒等映射被破坏」的病根——本文件用
  `--` 不用它做实现，只在 docstring 里点动机。

实现层口径（非论文新机制，讲代码时要与公式分开说）：
- 展平态 RMSNorm **无权**：官方参考实现 DeepSeek-V4-Pro inference/model.py:L676-L678 与
  pin 的 torch 回退（vllm/model_executor/kernels/mhc/torch.py:L62-L65）都是
  `rsqrt(均方 + eps)` 直接乘，没有增益。两者把 rsqrt 折进 GEMM **之后**
  （`mixes = x·fnᵀ` 再乘 `rsqrt`），数学等价于先归一再多乘（本实现按论文写：先 RMSNorm
  再乘）。
- raw 参数三块的**打包顺序是 (pre, post, res)**：官方 kernel 的切法（`pre` = 前 hc 个数、
  `post` = 中 hc 个、`comb` = 末尾 hc² 个，DeepSeek-V4-Pro inference/kernel.py:L391-L396，
  非官方 model.py 侧）；pin 同序（torch.py:L67-L78、nvidia/model.py 的 `mix_hc`）。与论文
  Eq.(3)(4)(5) 的列举顺序 (pre, res, post) 不同——不是同一个东西的两种写法，而是"谁放第
  几段"的工程约定，别倒着读（测试里有对账）。
- 矩阵作用方向：论文写 `B_l X_l`。**两侧实现算的是 `combᵀ X`**——官方
  `mhc_post`（DeepSeek-V4-Pro inference/model.py:L685）的 `torch.sum(comb.unsqueeze(-1) *
  residual.unsqueeze(-2), dim=2)` 收缩掉的是 comb 的**首**下标（即 `Σ_j comb[j,k]·res[j]`，
  与 pin 的 vllm/model_executor/kernels/mhc/torch.py:L94-L106 的 einsum
  `"...ij,...ih->...jh"` 同一收缩）；pin 的 tilelang 核写成字面式
  `x[o,j] += comb[i,o] * res[i,j]`（对 i 求和，vllm/model_executor/kernels/mhc/
  tilelang_kernels.py:L690-L691）——同样是 `combᵀ X`。故两侧存的 `comb` 与其转置是同一族
  里的两个点（双随机集合对转置封闭），本实现按论文原式写 `B X`，讲代码时不要断言方向相同。
- σ 之后加 eps（hc_eps=1e-6）是数值保护：给门控一个正值下界——官方 kernel
  （DeepSeek-V4-Pro inference/kernel.py:L391-L392）的 `pre = sigmoid(...) + eps`；pin 的
  torch 回退同款（vllm/model_executor/kernels/mhc/torch.py:L68 `+ hc_pre_eps`）。注意
  **post 不加 eps**（kernel L393-L394 是裸 2σ）。
- C 的 2σ 里那个 2 在 pin 里是硬编码 `hc_post_alpha = 2.0`
  （vllm/models/deepseek_v4/nvidia/model.py:L852，同一段 L849-L865 是 mHC 的参数账
  `mix_hc = (2 + hc_mult) * hc_mult`）。
- 出口 hc_head（把 n_hc 条流压回单流）是 **V4 自造件**：mHC 论文没有这一步（论文只在
  残差空间里工作）。两侧都有：官方 `ParallelHead.hc_head`（DeepSeek-V4-Pro
  inference/model.py:L728-L735，**只有 σ、没有 Sinkhorn**）；pin 的 `HCHeadOp`
  （vllm/model_executor/layers/mhc.py）走 tilelang 核，其注释原话是 "apply sigmoid-gated
  weighted sum of residual channels to output"
  （vllm/model_executor/kernels/mhc/tilelang_kernels.py:L901）。
"""
import numpy as np


# PAPER: arXiv:2606.19348 Eq.(3) —— X̂ = RMSNorm(vec(X_l))（展平、无权）
def rms_norm_flat(x, eps=1e-6, weight=None):
    """论文 Eq.(3) 的 `X̂_l = RMSNorm(vec(X_l))`：把 n_hc×d 展平后按均方根归一。

    `weight=None` 即两侧实现的无权 RMSNorm（V4 的 mHC 输入归一没有增益）；给 weight 时
    乘上去（Eq.(21) 的 MTP 那两个 RMSNorm 是带增益的形态）。缩放不变性：RMSNorm(c·x) =
    RMSNorm(x)。
    """
    x = np.asarray(x, dtype=np.float64)
    ms = (x**2).mean(axis=-1, keepdims=True)
    y = x / np.sqrt(ms + eps)
    return y if weight is None else y * weight


# PAPER: arXiv:2606.19348 Eq.(3)(4)(5) —— 一次 GEMM 出 A/B/C 的 raw 参数
def mhc_raw_mappings(X, fn, base, scale, eps=1e-6):
    """Eq.(3)(4)(5)：三张映射都是「输入相关的动态项 + 静态偏置」：

        Ã_l = α^{pre}·(X̂_l W^{pre}) + S^{pre}
        B̃_l = α^{res}·Mat(X̂_l W^{res}) + S^{res}
        C̃_l = α^{post}·(X̂_l W^{post})ᵀ + S^{post}

    X 形状 (T, n_hc, d)；`fn` (mix, n_hc·d)、`base` (mix,)、`scale` (3,) 是两侧实现共有的
    **一次 GEMM** 形态（`mix = (2 + n_hc)·n_hc`；官方 DeepSeek-V4-Pro inference/model.py:
    L663 与 L673-L681、pin vllm/models/deepseek_v4/nvidia/model.py:L853），三段的顺序是
    **(pre, post, res)** —— 与论文列举顺序 (pre, res, post) 不同，见模块 docstring。返回
    `(Ã (T,n_hc), B̃ (T,n_hc,n_hc), C̃ (T,n_hc,1))`。
    """
    hc = X.shape[1]
    flat = rms_norm_flat(X.reshape(X.shape[0], hc * X.shape[2]), eps=eps)
    mixes = flat @ fn.T
    pre = mixes[:, :hc] * scale[0] + base[:hc]
    post = mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    res = mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]
    return pre, res.reshape(X.shape[0], hc, hc), post[:, :, None]


# PAPER: arXiv:2606.19348 Eq.(6)(7)(8) —— A = σ(·)、C = 2σ(·)、B = Sinkhorn(·)
def mhc_gates(A_tilde, B_tilde, C_tilde, sinkhorn_iters=20, eps=1e-6, post_mult=2.0):
    """Eq.(6)(7)(8) 的投影三件套：

        A_l = σ(Ã_l)（读入，值域 (0,1)）
        C_l = 2σ(C̃_l)（写回，值域 (0,2)；pin 里 2 是硬编码 hc_post_alpha=2.0）
        B_l = Sinkhorn-Knopp(B̃_l)（流间混合，投影进 Birkhoff 多面体）

    返回 `(A (T,n_hc), B (T,n_hc,n_hc), C (T,n_hc,1))`；σ 后加 eps 是数值保护
    （给门控一个正值下界，对应 pin 的 hc_eps）。
    """
    A = 1.0 / (1.0 + np.exp(-np.asarray(A_tilde, dtype=np.float64))) + eps
    C = post_mult / (1.0 + np.exp(-np.asarray(C_tilde, dtype=np.float64)))
    B = np.stack([sinkhorn_knopp(b, iters=sinkhorn_iters, eps=eps) for b in np.asarray(B_tilde, dtype=np.float64)])
    return A, B, C


# PAPER: arXiv:2512.24880 Eq.(8)(9) / arXiv:2606.19348 Eq.(8) —— Sinkhorn-Knopp 迭代
def sinkhorn_knopp(raw, iters=20, eps=1e-6, start="exp", order="row-first", trace=False):
    """把非负矩阵投影进 Birkhoff 多面体（双随机）：交替行/列归一，迭代 `t_max` 轮。

    - `start="exp"`：论文的起点 `M^(0) = exp(H̃^res)`（Eq.8/9 的原设定）；
    - `start="softmax"`：两侧实现的起点 `softmax(dim=-1)`（= exp + 行归一，把第一步吸收掉；
      官方 DeepSeek-V4-Pro inference/kernel.py:L401-L408，pin
      vllm/model_executor/kernels/mhc/torch.py:L78）—— 与 "exp" 起步的差别只在 eps 落点
      （~1e-5 量级），20 轮内不改变结论；
    - `start="raw"`：直接从原始非负矩阵起步（教科书式的交替归一，用于手算 worked example）。

    - `order="row-first"`：每轮先行归一后列归一（**两侧实现都是这个顺序**，pin 的 torch
      回退路径 `softmax → 列归一 → 重复 (行,列)`）；
    - `order="col-first"`：论文 Eq.(8)(9) 的书面顺序 `M^(t) = T_r(T_c(M^(t-1)))`（先列后行）。

    极限相同（都收敛到双随机），`t_max=20` 轮内的定点略有差别（dossier m14）。**20 是论文
    原话的 `as a practical value`——精度与开销的折中，不是"20 次就精确了"。**
    """
    M = np.asarray(raw, dtype=np.float64)
    M0, ops = _start_matrix_and_ops(M, start, order, iters, eps)
    M = M0
    for op in ops:
        M = _normalize(M, -1 if op == "row" else -2, eps)
    if trace:
        return M, sinkhorn_trace(raw, iters=iters, start=start, eps=eps, order=order)
    return M


# PAPER: arXiv:2512.24880 Eq.(9) —— 起点 M^(0) 与交替归一的算子序列
def _start_matrix_and_ops(raw, start, order, iters, eps):
    """把「起点 + iters 轮行/列交替」展成一个算子序列（三个起点约定共用一条执行路径）：

    - `exp`：`M^(0) = exp(H̃)`（论文 Eq.8/9 的起点），序列 = iters × (行/列)；
    - `softmax`：两侧实现的起点（`softmax(dim=-1)` = exp + 行归一），把第一步行归一吸收掉；
    - `raw`：直接以原始非负矩阵起步（教科书式交替归一，手算 worked example 用）。
    """
    pair = ["row", "col"] if order == "row-first" else ["col", "row"]
    if start == "exp":
        return np.exp(raw), pair * iters
    if start == "raw":
        return raw, pair * iters
    if start == "softmax":
        return _normalize(np.exp(raw), -1, eps), (pair * iters)[1:]
    raise ValueError(f"unknown start: {start!r}")


# PAPER: arXiv:2512.24880 Eq.(9) —— T_r / T_c：行/列归一化算子
def _normalize(M, axis, eps):
    """Eq.(9) 的 T_r（axis=−1，行）与 T_c（axis=−2，列）：`M / (Σ + eps)`。

    分母加 eps 是数值保护，同时给矩阵一个正值下界（不会出现 0 行 / 0 列）。
    """
    return M / (M.sum(axis=axis, keepdims=True) + eps)


# PAPER: arXiv:2512.24880 Eq.(9) —— 逐轮行/列和偏差（worked example 的收敛轨迹）
def sinkhorn_trace(raw, iters=20, start="raw", eps=1e-6, order="row-first"):
    """逐步记录每轮的行和/列和偏差——「交替归一为什么收敛」的可示教轨迹。

    每轮一项：`row_sums_after_row` / `col_sums_after_row`（行归一之后）、
    `row_sums_after_col` / `col_sums_after_col`（列归一之后）、以及
    `row_dev` / `col_dev`（该轮结束时的最大偏差）。
    """
    M = np.asarray(raw, dtype=np.float64)
    M, ops = _start_matrix_and_ops(M, start, order, iters, eps)
    rows = []
    cursor = 0
    for _ in range(iters):
        rec = {}
        applied = 0
        while cursor < len(ops) and applied < 2:
            op = ops[cursor]
            cursor += 1
            applied += 1
            M = _normalize(M, -1 if op == "row" else -2, eps)
            rec[f"row_sums_after_{op}"] = M.sum(axis=-1)
            rec[f"col_sums_after_{op}"] = M.sum(axis=-2)
        rec["row_dev"] = float(np.abs(M.sum(axis=-1) - 1).max())
        rec["col_dev"] = float(np.abs(M.sum(axis=-2) - 1).max())
        rows.append(rec)
    return rows


# PAPER: arXiv:2606.19348 Eq.(2) / arXiv:2512.24880 Eq.(6) —— Birkhoff 多面体判定
def is_doubly_stochastic(M, atol=1e-8):
    """Eq.(2)/(6) 的三个条件：非负、每行和 = 1、每列和 = 1（Birkhoff 多面体的定义）。"""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    if np.any(M < -atol):
        return False
    return bool(np.allclose(M.sum(axis=1), 1.0, atol=atol) and np.allclose(M.sum(axis=0), 1.0, atol=atol))


# PAPER: arXiv:2512.24880 §4.1 —— ‖H^res‖₂ ≤ 1（论文原短语 "is bounded by 1"）
def spectral_norm(M):
    """谱范数 = 最大奇异值；双随机矩阵的 ‖·‖₂ ≤ 1（不放大信号）。"""
    return float(np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)[0])


# PAPER: arXiv:2606.19348 Eq.(1) —— 残差自带流间混合：B_l X_l
def doubly_stochastic_residual(B, X):
    """Eq.(1) 的第一项：`B_l X_l`。B 形状 (n_hc, n_hc)、X 形状 (T, n_hc, d)。

    单流的 `x + F(x)` 在这里变成「n_hc 条流互相混合」——B 被关进 Birkhoff 多面体后，
    这一步是**非扩张**的（‖B X‖ ≤ ‖X‖），所以深堆叠不会放大信号（§4.1 第一条性质）。
    """
    return np.einsum("ij,tjd->tid", np.asarray(B, dtype=np.float64), np.asarray(X, dtype=np.float64))


# PAPER: arXiv:2606.19348 Eq.(1) —— 读入：A_l X_l（n_hc 条流聚成子层输入）
def mhc_layer_input(A, X):
    """Eq.(1) 里 `F_l(A_l X_l)` 的 `A_l X_l`：A 形状 (n_hc,)，把 n_hc 条流加权求和成
    一条子层输入（pin 里这个中间量叫 `layer_input = sum_i pre_mix_i * residual_i`，
    vllm/model_executor/kernels/mhc/torch.py:L84-L86；官方是
    `y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)`，
    DeepSeek-V4-Pro inference/model.py:L680）。"""
    A = np.asarray(A, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    return (A[None, :, None] * X).sum(axis=1)


# PAPER: arXiv:2606.19348 Eq.(1) —— X_{l+1} = B_l X_l + C_l F_l(·)
def mhc_update(X, B, C, sublayer_out):
    """Eq.(1) 全式：`X_{l+1} = B_l X_l + C_l F_l(A_l X_l)`。

    `X` (T,n_hc,d)、`B` (n_hc,n_hc)、`C` (n_hc,1)——C 把子层输出**散回** n_hc 条流、
    B 让残差在流间混合（pin 的 `mhc_post_torch` 两项相加即此式，
    vllm/model_executor/kernels/mhc/torch.py:L104-L106 `mixed_residual + post_term`）。
    """
    C = np.asarray(C, dtype=np.float64).reshape(-1)
    f = np.asarray(sublayer_out, dtype=np.float64)
    return doubly_stochastic_residual(B, X) + np.einsum("i,tj->tij", C, f)


# PAPER: arXiv:2606.19348 §2.2 —— 出口 hc_head：sigmoid 门控加权求和压回单流（V4 自造件）
def hc_head(x, hc_fn, hc_base, hc_scale, eps=1e-6):
    """把 n_hc 条流按 **sigmoid 门控加权求和**压回 (T, d)。

    mHC 论文没有这一步（论文只在残差空间里工作）；V4 用它把多流出口接回共享 RMSNorm 与
    输出头。两侧都有：官方 `ParallelHead.hc_head`（DeepSeek-V4-Pro inference/model.py:
    L728-L735，`pre = sigmoid(mixes * hc_scale + hc_base) + hc_eps`，**只有 σ、没有
    Sinkhorn**）；pin 走 tilelang 核，注释原话 "apply sigmoid-gated weighted sum of
    residual channels to output"（vllm/model_executor/kernels/mhc/tilelang_kernels.py:L901）。
    门控在 (0,1) ⇒ 输出是各条流的凸组合。
    """
    x = np.asarray(x, dtype=np.float64)
    hc = x.shape[1]
    flat = rms_norm_flat(x.reshape(x.shape[0], -1), eps=eps)
    mixes = flat @ np.asarray(hc_fn, dtype=np.float64).T
    pre = 1.0 / (1.0 + np.exp(-(mixes * float(hc_scale[0]) + np.asarray(hc_base, dtype=np.float64)))) + eps
    return (pre[:, :, None] * x).sum(axis=1)
