"""GPTQ —— arXiv:2210.17323 §3-§4:逐层重建目标 Eq.1(argmin ‖WX − ŴX‖²)、
OBQ 前身 Eq.2-Eq.3(贪心选权重 + 逆 Hessian 一步高斯消元)、GPTQ 三步优化
(Step 1 任意固定列序——H 只依赖层输入;Step 2 lazy batch B=128;Step 3
Cholesky 重构 + dampening 1% 平均对角元)、Algorithm 1 主循环;§5 的
per-row 非对称 min-max 网格(Setup)与 RTN 对照基线(Baselines:同一副网格
上直接取整)。对应 vLLM 侧:gptq_quantize_weights 是测试用 RTN 参考实现,
二阶补偿全在离线 autogptq。

布局约定:W (d_row, d_col) = (输出维, 输入维)(HF Linear 权重即此布局),
X (n_samples, d_col)。论文记 X 为 (d_col, m)、层输出 WX;这里等价地以
X @ W^T 计算,‖WX − ŴX‖²_F = ‖X W^T − X Ŵ^T‖²_F(转置不变)。
「trace」可选参数只逐格记录算法自身循环变量的快照(供示教轨迹用),不是
论文之外的新机制。np.round 为银行家舍入(0.5 取偶),论文未指定、不影响性质。
"""
import numpy as np


# PAPER: arXiv:2210.17323 §3 Eq.2 —— H_F = 2 X_F X_F^T:目标函数(WX − ŴX)²
# 对权重行的 Hessian;只依赖层输入 X、与权重无关(Step 1 全行同序的合法性来源)。
# 论文 X 记 (d_col, m),此处 X 为 (n_samples, d_col),故 H = 2 X^T X。
def layer_hessian(X):
    X = np.asarray(X, dtype=float)
    return 2.0 * X.T @ X


# PAPER: arXiv:2210.17323 §4 Step 3 —— dampening:对角加小常数 λ
# ("we always choose 1% of the average diagonal value"),防数值问题。
def dampen_hessian(H, damp=0.01):
    lam = damp * np.mean(np.diag(H))
    return H + lam * np.eye(H.shape[0])


# PAPER: arXiv:2210.17323 §4 Algorithm 1 前置行 —— H^{-1} = (2XX^T + λI)^{-1},
# 再取 Cholesky(H^{-1})^T(上三角):预取全部所需行的数值稳定替代
# (反复应用 Eq.5 的增量求逆会累积误差把 H^{-1} 推成不定阵)。
# 注:NumPy 无 cholesky_inverse,H^{-1} 直接求逆(数学同值);算法内容
# (一次 Cholesky 预取全部行)不变。
# PAPER: arXiv:2210.17323 §4 Algorithm 1 前置行(H^{-1} <- Cholesky(H^{-1})^T)
def inverse_hessian_cholesky(X, damp=0.01):
    H = dampen_hessian(layer_hessian(X), damp=damp)
    H_inv = np.linalg.inv(H)
    return np.linalg.cholesky(H_inv).T  # 上三角 U,U^T U = H^{-1}


# PAPER: arXiv:2210.17323 §5 Setup —— "standard uniform per-row asymmetric
# quantization on the min-max grid":逐行 scale = (xmax-xmin)/(qmax-qmin)、
# zp = qmin - round(xmin/scale),xmin->qmin、xmax->qmax 精确落格。
def row_grid_params(w_rows, num_bits=4):
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -(2 ** (num_bits - 1))
    xmax = w_rows.max(axis=1)
    xmin = w_rows.min(axis=1)
    scale = np.maximum((xmax - xmin) / (qmax - qmin), 1e-12)  # 数值护栏:常数行
    zp = qmin - np.round(xmin / scale)
    return scale, zp.astype(np.int64)


# PAPER: arXiv:2210.17323 §5 Setup —— 在给定 per-row 网格上取整并夹回码域。
def quantize_with_grid(w_col, scale, zp, num_bits=4):
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -(2 ** (num_bits - 1))
    return np.clip(
        np.round(w_col / scale) + zp, qmin, qmax
    ).astype(np.int64)


# PAPER: arXiv:2210.17323 §5 Setup —— 反量化 w_hat = (q - zp)·scale。
def dequantize_with_grid(q, scale, zp):
    return (q - zp) * scale


# PAPER: arXiv:2210.17323 §3 Eq.1 + §5 Baselines —— RTN:在与 GPTQ 完全相同的
# per-row(可分组)非对称网格上把原始权重直接取整,无任何补偿。
def rtn_quantize(W, num_bits=4, group_size=None):
    W = np.asarray(W, dtype=float)
    d_row, d_col = W.shape
    gs = group_size or d_col
    Q = np.zeros((d_row, d_col), dtype=np.int64)
    W_hat = np.zeros((d_row, d_col))
    for g1 in range(0, d_col, gs):
        g2 = min(g1 + gs, d_col)
        scale, zp = row_grid_params(W[:, g1:g2], num_bits)
        Q[:, g1:g2] = quantize_with_grid(W[:, g1:g2], scale[:, None], zp[:, None], num_bits)
        W_hat[:, g1:g2] = dequantize_with_grid(Q[:, g1:g2], scale[:, None], zp[:, None])
    return Q, W_hat


# PAPER: arXiv:2210.17323 §3 Eq.1 —— 层重建误差 ‖WX − ŴX‖²_F(转置不变地
# 以 X @ W^T 计算):GPTQ 的 argmin 目标,也是与 RTN 对比的记分板。
def layer_output_error(W_orig, W_hat, X):
    diff = X @ W_orig.T - X @ W_hat.T
    return float(np.sum(diff * diff))


# PAPER: arXiv:2210.17323 §3 Eq.2 + Eq.3 —— OBQ 单行:贪心选
# (quant(w_q) − w_q)² / [H^{-1}_F]_qq 最小的权重,量化后用 δ_F 等比例调整
# 未量化权重吸收误差;H^{-1} 经 Eq.3(高斯消元一步)移除该行列。行网格取自
# 原始行(§3:网格在过程开始前固定)。复杂度 O(d_row·d_col³) 的那一半就在
# 每权重一次的 Eq.3 上。注意:§3 的 OBQ 没有 dampening(那是 GPTQ §4
# Step 3 才加的),故 H 须正定可逆——校准样本数少于特征数时 H=2X^T X 奇异,
# 可改传 dampen_hessian(layer_hessian(X)) 或用 gptq_quantize。
# PAPER: arXiv:2210.17323 §3 Eq.2 + Eq.3
def obq_quantize_row(w, H, num_bits=4, trace=None):
    d = len(w)
    Hinv = np.linalg.inv(np.asarray(H, dtype=float))
    w = np.asarray(w, dtype=float).copy()
    scale_arr, zp_arr = row_grid_params(w[None, :], num_bits)
    scale, zp = float(scale_arr[0]), int(zp_arr[0])
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -(2 ** (num_bits - 1))
    q = np.zeros(d, dtype=np.int64)
    w_hat = np.zeros(d)
    done = np.zeros(d, dtype=bool)
    for _ in range(d):
        F = np.flatnonzero(~done)
        # Eq.2 第一式:逐候选算补偿代价(标量版 quantize_with_grid)。
        codes = np.clip(np.round(w[F] / scale) + zp, qmin, qmax)
        costs = (dequantize_with_grid(codes, scale, zp) - w[F]) ** 2 / Hinv[F, F]
        k = int(np.argmin(costs))
        pick = F[k]
        q[pick] = codes[k]
        w_hat[pick] = dequantize_with_grid(codes[k], scale, zp)
        if trace is not None:
            trace.append((int(pick), float(costs[k])))
        # Eq.2 第二式:δ_F = −(w_q − quant(w_q))/[H^{-1}]_qq · (H^{-1})_{:,q}
        err = (w[pick] - w_hat[pick]) / Hinv[pick, pick]
        rest = F[F != pick]
        w[rest] -= err * Hinv[rest, pick]
        done[pick] = True
        # Eq.3:H^{-1}_{-q} = (H^{-1} − H^{-1}_{:,q} H^{-1}_{q,:}/[H^{-1}]_qq)_{-q}
        Hinv -= np.outer(Hinv[:, pick], Hinv[pick, :]) / Hinv[pick, pick]
    return q, w_hat


# PAPER: arXiv:2210.17323 §4 Algorithm 1 —— GPTQ 主循环:H^{-1} 先整体
# Cholesky;外层按 B 列分块,块内逐列「量化→记误差→即时补偿块内」,块末把
# 累计误差 E 一次性传播给块外全部剩余列(lazy batch:不减计算量、只攒大矩阵
# 乘治 GPU 计算访存比)。组网格参数取自当前最新权重(§5 Additional Tricks:
# "always using the most current updated weights")。
# PAPER: arXiv:2210.17323 §4 Algorithm 1(主循环)
def gptq_quantize(
    W, X, num_bits=4, group_size=None, block_size=128, damp=0.01, trace=None
):
    W_orig = np.asarray(W, dtype=float).copy()  # Eq.1 记分板用原始权重
    W = W_orig.copy()  # 工作副本:补偿会改写未量化列
    d_row, d_col = W.shape
    gs = group_size or d_col
    U = inverse_hessian_cholesky(X, damp=damp)  # Algorithm 1 前置行
    Q = np.zeros((d_row, d_col), dtype=np.int64)
    W_hat = np.zeros((d_row, d_col))
    for i1 in range(0, d_col, block_size):
        i2 = min(i1 + block_size, d_col)
        E = np.zeros((d_row, i2 - i1))  # 块内量化误差(÷[H^{-1}]_jj)
        for j in range(i1, i2):
            if j % gs == 0:
                g2 = min(j + gs, d_col)
                scale, zp = row_grid_params(W[:, j:g2], num_bits)
            q = quantize_with_grid(W[:, j], scale, zp, num_bits)
            q_real = dequantize_with_grid(q, scale, zp)
            Q[:, j] = q
            W_hat[:, j] = q_real
            E[:, j - i1] = (W[:, j] - q_real) / U[j, j]
            W[:, j:i2] -= np.outer(E[:, j - i1], U[j, j:i2])
            W[:, j] = q_real  # 已量化列定格(该列此后不再被读)
            if trace is not None:
                trace.append(
                    {
                        "col": j,
                        "U_jj": float(U[j, j]),
                        "q": q.copy(),
                        "err": E[:, j - i1].copy(),
                    }
                )
        W[:, i2:] -= E @ U[i1:i2, i2:]  # lazy batch:块末一次性全局补偿
    return Q, W_hat, layer_output_error(W_orig, W_hat, X)


# PAPER: arXiv:2210.17323 §4 Step 1 —— 任意固定列序的等价参考实现:每列一次
# Eq.2(全行共享列序)+ Eq.3 全矩阵消元,无 Cholesky、无 lazy batch。
# 与 Algorithm 1 在精确算术下结果相同(Step 2/3 只改执行方式不改数学),
# 也是「OBQ 立方复杂度 -> GPTQ」的中间形态:H^{-1} 更新次数 d_row·d_col -> d_col。
# PAPER: arXiv:2210.17323 §4 Step 1(每列一次 Eq.2 + Eq.3)
def gptq_naive_inverse_updates(W, X, num_bits=4, group_size=None, damp=0.01):
    W = np.asarray(W, dtype=float).copy()
    d_row, d_col = W.shape
    gs = group_size or d_col
    Hinv = np.linalg.inv(dampen_hessian(layer_hessian(X), damp=damp))
    Q = np.zeros((d_row, d_col), dtype=np.int64)
    W_hat = np.zeros((d_row, d_col))
    for j in range(d_col):
        if j % gs == 0:
            g2 = min(j + gs, d_col)
            scale, zp = row_grid_params(W[:, j:g2], num_bits)
        q = quantize_with_grid(W[:, j], scale, zp, num_bits)
        q_real = dequantize_with_grid(q, scale, zp)
        Q[:, j] = q
        W_hat[:, j] = q_real
        err = (W[:, j] - q_real) / Hinv[j, j]
        W[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
        W[:, j] = q_real
        # Eq.3:移除第 j 行列(后续只读 j 之后的子块,数学等价于删行列)。
        Hinv -= np.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]
    return Q, W_hat


# PAPER: arXiv:2210.17323 §4 Step 1 —— 复杂度账:OBQ O(d_row·d_col³) ->
# GPTQ O(max{d_row·d_col², d_col³}),提速 min{d_row, d_col} 倍
# (H^{-1} 更新从每权重一次降到每列一次)。
def hessian_update_flops(d_row, d_col):
    obq = d_row * d_col ** 3
    gptq = max(d_row * d_col ** 2, d_col ** 3)
    return obq, gptq
