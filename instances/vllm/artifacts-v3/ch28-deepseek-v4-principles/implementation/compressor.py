"""CSA / HCA 的 KV 压缩器 —— 论文忠实的小型参考实现（NumPy，纯 CPU）。

论文出处（真相源）：
- arXiv:2606.19348 §2.3 引言 + §2.3.1 Eq.(9)(10)(11)(12)：CSA 的「软池化」压缩——
  四组投影出 C^a/C^b（KV 序列）与 Z^a/Z^b（权重序列），对**堆叠后的 2m 个元素**做一次
  行 softmax，再加权求和成一条 C^Comp；i=0 时 Z^b 填 −∞、C^b 填 0；相邻条目共享一半
  输入 ⇒ 序列恰好压到 1/m。
- arXiv:2606.19348 §2.3.2 Eq.(20)(21)(22)(23)：HCA 用**同一套机制**，差别只有三条：
  只有一组 C/Z、窗宽是 m'（≫ m）、没有重叠。
- 推理期形态（谁产出条目、什么时候产出）在 pin 代码里：块尾 token 才干活
  （`(position+1) % COMPRESS_RATIO == 0`，vllm/v1/attention/backends/mla/compressor_utils.py
  的 is_valid），未满一窗的 token 暂存在压缩机的 state_cache 里。本文件用
  WindowAccumulator 表达同一件事（推理期缓存形态，非新增机制）。

标签注意（dossier m04 的三处独立佐证）：**官方参考实现（DeepSeek-V4-Pro inference/model.py）
的两半区标签与论文相反**——它的前半区 `[..., :head_dim]` 装的是"前一个窗"，后半区
`[..., head_dim:]` 是"当前窗"（model.py:L296 的注释原话 "the first half of dims is for
overlapping compression, second half for normal"、L302 的状态缓冲注释、L312-L313 的
`overlap_transform` 同序；本仓 pin 的压缩核 `head_offset = (tokens >= CR) * HEAD_SIZE`
也是这个顺序）。论文 Eq.(11) 的 a 半区是当前窗、b 半区是前一个窗。本文件按**论文原式**
实现（a=当前窗、b=前一个窗），并另给 `csa_compress_reference_layout` 让读者看到两种标签
下是同一套数学。

标记法：每个 def/class 用 `# PAPER: <arXiv id> §x Eq.y` 锚定论文出处（本实现的验收门禁）。
"""
import numpy as np


# PAPER: arXiv:2606.19348 Eq.(11) —— Softmax_row(·)：沿堆叠轴归一，跨 2m 个元素
def softmax_row(x, axis=0):
    """Eq.(11) 的 Softmax_row：`normalization across the total of 2m elements from
    both Z^a and Z^b`。逐列（沿堆叠轴）归一 ⇒ 每一列的权重和 = 1。

    数值上按行减最大值再取指数（等价变形，防溢出）。
    """
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# PAPER: arXiv:2606.19348 Eq.(9) Eq.(10) —— 四组可学习投影
def csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b):
    """Eq.(9)(10)：`C^a = H·W^{aKV}, C^b = H·W^{bKV}`、`Z^a = H·W^{aZ}, Z^b = H·W^{bZ}`。

    H 形状 (n, d)，四组 W 形状 (d, c)（c = head_dim）。
    """
    return H @ W_kv_a, H @ W_kv_b, H @ W_z_a, H @ W_z_b


# PAPER: arXiv:2606.19348 Eq.(11) —— 第 i 条条目的 2m 个输入（含 i=0 的 −∞/0 填充）
def csa_entry_inputs(C_a, C_b, Z_a, Z_b, B_a, B_b, m, i, prior_a=None):
    """把第 i 条压缩条目的 2m 个输入按论文顺序堆起来：

        [Z^a_{mi:m(i+1)-1} + B^a ; Z^b_{m(i-1):mi-1} + B^b]      （Eq.11）

    a 半区 = **当前窗**（第 i 段的 m 个 token）、b 半区 = **前一个窗**（第 i−1 段）。
    `i = 0` 时 b 半区的 Z^b 填 −∞、C^b 填 0（论文唯一的特例）。

    `prior_a` 用于跨 forward 的流式续窗：形如 `(C_prev, Z_prev)`，装的是**上一个 forward 里
    最后那个完整窗的 b 投影**（`C^b/Z^b`，即"贡献给下一个窗"的那一半）——它正是官方参考实现
    状态缓冲里保存的那半区（DeepSeek-V4-Pro inference/model.py:L302-L304 的 `kv_state`/
    `score_state` 前 `ratio` 行；官方把它叫 a 半区，因为它的命名与论文的 a/b 相反，见模块
    docstring），也对应 pin 里压缩机 state_cache 中已经落好的那半区 state。
    **口径提醒**：这里收的是**未折位置偏置的原始投影**（偏置由 `B_b` 在本函数里加）；pin 的
    state 是在**写**的时候就折进了 `ape[position % compress_ratio]`（save_partial_states），
    所以 pin 的 state 不能原样当 `prior_a` 传进来（会重复加偏置）。
    给了它，第 0 条条目的 b 半区就不再是 −∞/0 填充。行数必须是 m、位置偏置按窗内槽位对齐。

    返回 `(Z_stack, C_stack, info)`；info 记录两半区各带哪些 token 下标（越界位置为空）。
    """
    idx_a = slice(m * i, m * (i + 1))
    Za = Z_a[idx_a] + B_a
    Ca = C_a[idx_a]
    a_tokens = list(range(m * i, m * (i + 1)))

    if i > 0:
        idx_b = slice(m * (i - 1), m * i)
        Zb = Z_b[idx_b] + B_b
        Cb = C_b[idx_b]
        b_tokens = list(range(m * (i - 1), m * i))
    elif prior_a is not None:
        C_prev, Z_prev = prior_a
        Zb = np.asarray(Z_prev) + B_b
        Cb = np.asarray(C_prev)
        b_tokens = ["prior"] * m  # 上一个 forward 的末窗（跨调用续窗，pin 的 state_cache）
    else:
        Zb = np.full_like(Za, -np.inf)
        Cb = np.zeros_like(Ca)
        b_tokens = []

    info = {"i": i, "a_tokens": a_tokens, "b_tokens": b_tokens, "padded": i == 0 and prior_a is None}
    return np.vstack([Za, Zb]), np.vstack([Ca, Cb]), info


# PAPER: arXiv:2606.19348 Eq.(12) —— 加权和：C^Comp_i = Σ S^a_j ⊙ C^a_j + Σ S^b_j ⊙ C^b_j
def csa_compress(
    H,
    W_kv_a,
    W_kv_b,
    W_z_a,
    W_z_b,
    B_a,
    B_b,
    m,
    prior_a=None,
    return_weights=True,
):
    """Eq.(9)-(12) 全式：投影 → 拼 2m 窗 → 一次 softmax → 逐元素加权求和。

    `B_a/B_b` 形状 (m, c)——可学习位置偏置（行 = 窗内槽位，pin 里是 `ape`，在**写 state**
    时按 `position % compress_ratio` 折进去）。`m` 是压缩率（config 口径 CSA 层 m=4）。
    返回 `(C_comp, S)`：C_comp 形状 (n//m, c)、S 形状 (n//m, 2m, c)。

    注意：只产**完整窗**的条目——不整除的尾巴不产条目（推理期它留在压缩机 state 里等
    下一拍，pin 的 `(position+1) % COMPRESS_RATIO == 0` 即此）。
    """
    C_a, C_b, Z_a, Z_b = csa_project(H, W_kv_a, W_kv_b, W_z_a, W_z_b)
    n_win = n_entries(H.shape[0], m)
    c = W_kv_a.shape[1]
    C_comp = np.zeros((n_win, c), dtype=np.float64)
    S_all = np.zeros((n_win, 2 * m, c), dtype=np.float64)
    for i in range(n_win):
        Z_stack, C_stack, _ = csa_entry_inputs(
            C_a, C_b, Z_a, Z_b, B_a, B_b, m, i, prior_a=prior_a
        )
        S = softmax_row(Z_stack, axis=0)
        C_comp[i] = np.sum(S * C_stack, axis=0)
        S_all[i] = S
    return (C_comp, S_all) if return_weights else C_comp


# PAPER: arXiv:2606.19348 Eq.(20)(21)(22)(23) —— HCA：单序列、窗宽 m'、无重叠
def softpool_single_series(C, Z, B, m_prime):
    """单序列软池化 = Eq.(22)(23)：

        S_{m'i:m'(i+1)-1} = Softmax_row(Z + B)          （Eq.22）
        C^Comp_i = Σ_j S_j ⊙ C_j（j 遍历窗内 m' 个）      （Eq.23）

    没有 a/b 两半、没有重叠 ⇒ 条目数 = n/m'（不是 n/2m'）。B 形状 (m', c)。
    """
    n = C.shape[0]
    n_win = n_entries(n, m_prime)
    C_comp = np.zeros((n_win, C.shape[1]), dtype=np.float64)
    for i in range(n_win):
        idx = slice(m_prime * i, m_prime * (i + 1))
        S = softmax_row(Z[idx] + B, axis=0)
        C_comp[i] = np.sum(S * C[idx], axis=0)
    return C_comp


# PAPER: arXiv:2606.19348 Eq.(20)-(23) —— 完整投影版
def hca_compress(H, W_kv, W_z, B, m_prime, return_weights=True):
    """Eq.(20)(21)：`C = H·W^{KV}`、`Z = H·W^Z`，随后 Eq.(22)(23) 的窗内 softmax 加权和。

    `m_prime` 是重压缩率（config 口径 HCA 层 m'=128）。返回 `(C_comp, S)`，
    S 形状 (n//m', m', c)——每条的归一化只跨自己窗内的 m' 个元素。
    """
    C = H @ W_kv
    Z = H @ W_z
    n_win = n_entries(H.shape[0], m_prime)
    c = C.shape[1]
    C_comp = np.zeros((n_win, c), dtype=np.float64)
    S_all = np.zeros((n_win, m_prime, c), dtype=np.float64)
    for i in range(n_win):
        idx = slice(m_prime * i, m_prime * (i + 1))
        S = softmax_row(Z[idx] + B, axis=0)
        C_comp[i] = np.sum(S * C[idx], axis=0)
        S_all[i] = S
    return (C_comp, S_all) if return_weights else C_comp


# PAPER: arXiv:2606.19348 §2.3.1 —— "CSA in fact compresses the sequence length to 1/m times"
def n_entries(n, m):
    """完整窗的条目数 = n // m（论文原话：压到 1/m 倍；重叠窗不改变这个数）。"""
    return n // m


# PAPER: arXiv:2606.19348 Eq.(11) —— 相邻条目共享的那半窗（重叠的来源）
def csa_entry_input_index_sets(m, count):
    """每条条目的输入 token 下标：[a 半区（当前窗）, b 半区（前一个窗）]。

    论文原话：`the indexes of C^b used for C^Comp_i and the indexes of C^a used for
    C^Comp_{i-1} are overlapped` —— 所以相邻条目共享 m 个输入、条目数是 n/m 而非 n/(2m)。
    i=0 的 b 半区为空（−∞/0 填充）。
    """
    out = []
    for i in range(count):
        a = list(range(m * i, m * (i + 1)))
        b = list(range(m * (i - 1), m * i)) if i > 0 else []
        out.append([a, b])
    return out


# PAPER: arXiv:2606.19348 Eq.(11)(12) —— 官方参考实现的两半区标签
def csa_compress_reference_layout(H, kv_proj_w, gate_proj_w, position_bias, m):
    """**官方参考实现口径**（译自 DeepSeek-V4-Pro inference/model.py 的 `Compressor`，
    投影在 L297-L298、窗内加工在 L337-L342）：`wkv`/`wgate` 一枪出 `2c`（`coff * head_dim`，
    coff = 1 + overlap = 2），**前半区 `[..., :c]` 是"前一个窗"、后半区 `[..., c:]` 是"当前
    窗"**（L296 注释：the first half of dims is for overlapping compression, second half for
    normal），`position_bias` 形状 (m, 2c) 就是官方的 `ape`（L294）。

    与 `csa_compress` 是**同一套数学、标签相反**（测试里逐位对账）。放这里是为了让
    「公式的 a/b」与「代码的两半区」的对应关系一眼可查——讲代码时不要照抄论文的 a/b。
    """
    kv = H @ kv_proj_w  # (n, 2c)
    gate = H @ gate_proj_w  # (n, 2c)
    c = kv.shape[1] // 2
    n_win = n_entries(H.shape[0], m)
    C_comp = np.zeros((n_win, c), dtype=np.float64)
    S_all = np.zeros((n_win, 2 * m, c), dtype=np.float64)
    for w in range(n_win):
        idx_cur = slice(m * w, m * (w + 1))
        gate_cur = gate[idx_cur] + position_bias  # 逐 token：gate[pos] + B[pos % m]
        kv_cur = kv[idx_cur]
        stack_z = np.vstack([np.full((m, c), -np.inf), gate_cur[:, c:]])
        stack_c = np.vstack([np.zeros((m, c)), kv_cur[:, c:]])
        if w > 0:
            idx_prev = slice(m * (w - 1), m * w)
            gate_prev = gate[idx_prev] + position_bias
            kv_prev = kv[idx_prev]
            stack_z[:m] = gate_prev[:, :c]
            stack_c[:m] = kv_prev[:, :c]
        S = softmax_row(stack_z, axis=0)
        C_comp[w] = np.sum(S * stack_c, axis=0)
        S_all[w] = S
    return C_comp, S_all


# PAPER: arXiv:2606.19348 §2.3.1 —— 推理期：未满一窗的 token 暂存、块尾才产出条目
class WindowAccumulator:
    """压缩机的推理期状态（非新增机制）：与 pin 的 state_cache 同一件事。

    pin 口径（vllm/v1/attention/backends/mla/compressor_utils.py 的 is_valid）：
    只有 `(position + 1) % compress_ratio == 0` 的**块尾 token** 触发压缩，
    产出条目的序号是 `pos_after_compress = pos // compress_ratio`。

    这里把同一件事写成可跑的形态：逐 token 攒，攒满 m 个就交出一个完整窗
    （对应官方参考实现的 prefill 分支：`cutoff = seqlen - seqlen % ratio` 切成
    [窗对齐前缀, 余数] 两段，前缀进压缩、余数留在 state 里等下一拍 —— DeepSeek-V4-Pro
    inference/model.py:L327-L336）。
    """

    # PAPER: arXiv:2606.19348 §2.3.1 —— 压缩坐标系：条目序号 = pos // m
    def __init__(self, m, width):
        self.m = m
        self.width = width
        self.buffer_kv = np.zeros((0, width), dtype=np.float64)
        self.buffer_gate = np.zeros((0, width), dtype=np.float64)
        self.entry_count = 0

    # PAPER: arXiv:2606.19348 §2.3.1 —— 攒满一窗即交出一条（块尾 token 触发）
    def push(self, kv, gate, pos):
        """把 token `pos` 的 (kv, gate) 折进 buffer；若攒满整数个窗就交出来。

        返回 `(chunk_kv, chunk_gate, first_window_position, complete)`：
        chunk 是窗对齐的前缀（可能 0 行或多窗），`first_window_position = entry_count * m`
        给压缩机定 RoPE 位置用（pin: `(positions // compress_ratio) * compress_ratio`）。
        """
        self.buffer_kv = np.vstack([self.buffer_kv, np.atleast_2d(kv)])
        self.buffer_gate = np.vstack([self.buffer_gate, np.atleast_2d(gate)])
        usable = (self.buffer_kv.shape[0] // self.m) * self.m
        first_window_position = self.entry_count * self.m
        chunk_kv, chunk_gate = self.buffer_kv[:usable], self.buffer_gate[:usable]
        self.buffer_kv, self.buffer_gate = self.buffer_kv[usable:], self.buffer_gate[usable:]
        n_new = usable // self.m
        self.entry_count += n_new
        self.last_pos = pos
        return chunk_kv, chunk_gate, first_window_position, n_new > 0

    # PAPER: arXiv:2606.19348 §2.3.1 —— 条目落在哪个槽位（pos_after_compress）
    def entry_slot(self, pos):
        """压缩坐标系：条目序号 = pos // m（pin: `pos_after_compress = pos // block_size`）。"""
        return pos // self.m

    @property
    # PAPER: arXiv:2606.19348 §2.3.1 —— 还没攒满一窗的 token 数（state_cache 的存量）
    def buffered(self):
        return self.buffer_kv.shape[0]
