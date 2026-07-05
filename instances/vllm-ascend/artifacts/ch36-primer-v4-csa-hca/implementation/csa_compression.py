"""ch36 §2.3.1 Eq.(9)-(12) / §2.3.2 Eq.(20)-(23) (paper.md, arXiv:2606.19348) --
CSA/HCA 共用的 KV 压缩算子。

CSA(compress_ratio=m=4,重叠 2m 窗口)与 HCA(compress_ratio=m'=128,不重叠)在论文里
分别是 Eq.9-12 与 Eq.20-23——本质是同一个"按位置软选择"算子的两个参数化:每个压缩块
i 的输出 C_i^Comp 是块内(CSA 还借用上一块)若干 token 的 C 值,按 Z 值(加位置偏置 B
后)在"位置"这一维做逐通道 softmax 得到的权重加权求和(注意 Eq.11-12 的 S⊙C 是
Hadamard 积,softmax 是逐通道各自归一化,不是普通的标量注意力)。

落地:vllm_ascend/models/deepseek_v4.py:L598-L674(Compressor 类同时实现两种参数化,
coff=1+overlap 决定 wkv/wgate 输出打包成 1 份还是 2 份)。overlap_transform 就是本文件
`overlap_transform` 的落地对应——两者的『本块自己的 a 值 + 上一块借来的 b 值』错位拼接
逻辑完全一致,只是落地版把 a/b 打包进同一个线性层的列空间里。
"""
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# 投影:H -> (C, Z)
# ---------------------------------------------------------------------------


# PAPER: §2.3.1 Eq.(9)-(10) —— C^a=H.W^aKV, C^b=H.W^bKV, Z^a=H.W^aZ, Z^b=H.W^bZ
def csa_project_kv_entries(H: np.ndarray, W_a_kv: np.ndarray, W_b_kv: np.ndarray,
                            W_a_z: np.ndarray, W_b_z: np.ndarray):
    """H:(n,d)。返回 (C_a,Z_a,C_b,Z_b),每个 (n,c)。"""
    return H @ W_a_kv, H @ W_a_z, H @ W_b_kv, H @ W_b_z


# PAPER: §2.3.2 Eq.(20)-(21) —— C=H.W^KV, Z=H.W^Z(HCA 只有一路,无 a/b 之分)
def hca_project_kv_entries(H: np.ndarray, W_kv: np.ndarray, W_z: np.ndarray):
    """H:(n,d)。返回 (C,Z),每个 (n,c)。"""
    return H @ W_kv, H @ W_z


# ---------------------------------------------------------------------------
# 重叠窗口拼接(CSA 专属)——落地 overlap_transform 的对应物
# ---------------------------------------------------------------------------


# PAPER: §2.3.1 Eq.(11) 方括号里的 2m 窗口拼接;落地对照
# vllm_ascend/models/deepseek_v4.py:L668-674 overlap_transform
def overlap_transform(a_seq: np.ndarray, b_seq: np.ndarray, m: int, pad_value: float = 0.0) -> np.ndarray:
    """a_seq,b_seq:(n,c) 论文的 Z^a/Z^b 或 C^a/C^b 全序列,n 须能被 m 整除。

    返回窗口张量 (num_blocks, 2m, c):每块 i 的窗口 = [本块自己的 a_seq 值(m 个);
    上一块的 b_seq 值(m 个)]——块 i 的下标 [0:m) 借自块 i-1,[m:2m) 是块 i 自己的。
    i=0 没有"上一块",按论文规定用 pad_value 填充(对 Z 传 -inf,对 C 传 0.0)。
    """
    n, c = a_seq.shape
    if n % m != 0:
        raise ValueError(f"序列长度 {n} 必须能被压缩率 m={m} 整除")
    num_blocks = n // m
    a_blocks = a_seq.reshape(num_blocks, m, c)
    b_blocks = b_seq.reshape(num_blocks, m, c)
    windows = np.empty((num_blocks, 2 * m, c), dtype=np.result_type(a_seq, b_seq, float))
    windows[:, m:] = a_blocks
    windows[0, :m] = pad_value
    if num_blocks > 1:
        windows[1:, :m] = b_blocks[:-1]
    return windows


# ---------------------------------------------------------------------------
# 逐通道 softmax(Eq.11 的 Softmax_row)+ 加权求和(Eq.12/23 的 Hadamard 积求和)
# ---------------------------------------------------------------------------


# PAPER: §2.3.1 Eq.(11) "Softmax_row" —— 沿"位置"这一维(axis=1)逐通道独立归一化,
# 允许出现 -inf(i=0 时的 padding)而不产生 NaN(只要同一通道内不是全 -inf)
def softmax_over_positions(Z_windows: np.ndarray) -> np.ndarray:
    """Z_windows:(num_blocks,P,c),P=2m(CSA)或 m'(HCA)。返回同形状的权重,沿 axis=1 求和为 1。"""
    m = np.max(Z_windows, axis=1, keepdims=True)
    e = np.exp(Z_windows - m)
    return e / np.sum(e, axis=1, keepdims=True)


# PAPER: §2.3.1 Eq.(12) —— C_i^Comp = sum_j S_j ⊙ C_j(Hadamard 积后按位置求和)
def weighted_pool(S_windows: np.ndarray, C_windows: np.ndarray) -> np.ndarray:
    """S_windows,C_windows:(num_blocks,P,c)。返回 (num_blocks,c)。"""
    return np.sum(S_windows * C_windows, axis=1)


# ---------------------------------------------------------------------------
# 组装:整段序列的 CSA / HCA 压缩
# ---------------------------------------------------------------------------


# PAPER: §2.3.1 Eq.(9)-(12) —— CSA 整段压缩:重叠 2m 窗口,序列长压到 1/m
def csa_compress_sequence(H: np.ndarray, W_a_kv, W_b_kv, W_a_z, W_b_z, B_a: np.ndarray,
                           B_b: np.ndarray, m: int) -> np.ndarray:
    """返回 C^Comp:(n/m, c)。B_a,B_b:(m,c) 可学习位置偏置(Eq.11 的 B^a/B^b)。"""
    C_a, Z_a, C_b, Z_b = csa_project_kv_entries(H, W_a_kv, W_b_kv, W_a_z, W_b_z)
    C_win = overlap_transform(C_a, C_b, m, pad_value=0.0)
    Z_win = overlap_transform(Z_a, Z_b, m, pad_value=-np.inf)
    num_blocks = Z_win.shape[0]
    B_win = np.concatenate([np.broadcast_to(B_b, (num_blocks, m, B_b.shape[-1])),
                             np.broadcast_to(B_a, (num_blocks, m, B_a.shape[-1]))], axis=1)
    S = softmax_over_positions(Z_win + B_win)
    return weighted_pool(S, C_win)


# PAPER: §2.3.2 Eq.(20)-(23) —— HCA 整段压缩:不重叠,序列长压到 1/m'
def hca_compress_sequence(H: np.ndarray, W_kv, W_z, B: np.ndarray, m_prime: int) -> np.ndarray:
    """返回 C^Comp:(n/m', c)。B:(m',c) 可学习位置偏置。"""
    C, Z = hca_project_kv_entries(H, W_kv, W_z)
    n, c = C.shape
    if n % m_prime != 0:
        raise ValueError(f"序列长度 {n} 必须能被压缩率 m'={m_prime} 整除")
    num_blocks = n // m_prime
    C_blk = C.reshape(num_blocks, m_prime, c)
    Z_blk = Z.reshape(num_blocks, m_prime, c) + B[None, :, :]
    S = softmax_over_positions(Z_blk)
    return weighted_pool(S, C_blk)


# ---------------------------------------------------------------------------
# coff:CSA/HCA 在落地代码里合并成同一个 Compressor 类的关键量
# ---------------------------------------------------------------------------


# PAPER: §2.3.1 Eq.(9)-(12)(CSA,overlap=True,coff=2)/ §2.3.2 Eq.(20)-(23)
# (HCA,overlap=False,coff=1)——coff=1+overlap 决定落地 wkv/wgate 的输出维度是
# head_dim(HCA)还是 2*head_dim(CSA,打包 W^aKV/W^bKV 两段);落地对照
# vllm_ascend/models/deepseek_v4.py:L618-621
# PAPER: §2.3.1 Eq.(9)-(12) / §2.3.2 Eq.(20)-(23)(见上)
@dataclass
class CompressorShape:
    compress_ratio: int
    overlap: bool = field(init=False)
    coff: int = field(init=False)

    # PAPER: 同上(CompressorShape 类注释)—— overlap=(compress_ratio==4),coff=1+overlap
    def __post_init__(self):
        if self.compress_ratio not in (4, 128):
            raise ValueError(f"Only support compress_ratio in [4, 128]. Got: {self.compress_ratio}")
        self.overlap = self.compress_ratio == 4
        self.coff = 1 + int(self.overlap)
