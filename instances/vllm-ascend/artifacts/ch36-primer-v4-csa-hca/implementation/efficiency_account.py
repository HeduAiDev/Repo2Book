"""ch36 §1 / §2.3.4 (paper.md, arXiv:2606.19348) -- 数值推演:把『FLOPs 27% / KV 10%』
逐项算出来的账本模型。

论文给的是结论性数字(§1:"DeepSeek-V4-Pro requires only 27% of single-token inference
FLOPs and 10% of KV cache compared with DeepSeek-V3.2"),没有给出能重新推导出这两个
具体百分比的完整配置(逐层 compress_ratios 数组、每层 k、indexer 头数等发行版配置文件
里的数字,不在这份论文摘录里)。本文件因此做两件诚实的事,不做第三件:

  1. 把"KV 存多少"与"单 token 算多少 FLOPs"这两本账,按 CSA/HCA/稠密三种层各自的
     复杂度公式(动机段的 O(L) 账、CSA 的 O(k) 账、HCA 的 O(L/m') 账)精确建模;
  2. 用示意性的参数(数量级与论文/落地代码给出的线索一致,但不是 DeepSeek 未公开的
     确切配置)跑一遍这套账,得到"同一量级"的压缩比,验证账本模型本身是对的;
  3. **不**假装用这套模型复现出论文原文的 27%/10% 这两个具体数字——那需要 DeepSeek
     未公开的完整配置,伪造出来会违反"不发明论文没有的数字"的铁律。

CSA 退化验证:令 m=1(不压缩)时,csa_layer_cost 退化为"直接对原始 token 做 top-k
稀疏注意力"——这正是论文对照基线 DeepSeek-V3.2 的 DSA(§2.3 引言:"CSA...then applies
DeepSeek Sparse Attention (DSA)")。因此本文件不需要另造一个独立的 baseline 函数,
用 m=1 复用同一个 csa_layer_cost 即可得到"论文口径下的 DSA-only 基线"。
"""
from dataclasses import dataclass


# PAPER: §1 + §2.3 —— KV 存量与单 token FLOPs 的账本容器(动机段的核心度量)
@dataclass
class LayerCost:
    kv_entries_stored: float   # 该层单 token 需要保留的 KV 条目数(压缩后的"存量"账)
    flops_per_token: float     # 该层单 token(decode)核心注意力的点积计数代理(FLOPs 账)


# PAPER: §2.3.1 —— CSA:KV 压到 L/m 条(Eq.9-12),核注意力只对 k 个选中的压缩块 + n_win
# 个滑窗 raw token 做点积(Eq.19,§2.3.3),但 lightning indexer 仍要扫过全部 L/m 个候选块
# 打分(Eq.16)——indexer 的开销与"主注意力降了多少倍"是两本不同的账,必须分开算再相加,
# 否则会像 ch32 的 cost_model 强调的那样高估加速比。令 m=1 时退化为 DSA-only 基线(见文件
# 顶部说明)。
# PAPER: §2.3.1 Eq.(9)-(12)/Eq.(16)(见上)
def csa_layer_cost(seq_len: int, m: int, k: int, n_win: int, head_dim: int,
                    indexer_heads: int = 1, indexer_dim: int | None = None) -> LayerCost:
    kv_entries = seq_len / m
    core_attn_flops = (k + n_win) * head_dim
    indexer_flops = kv_entries * indexer_heads * (indexer_dim if indexer_dim is not None else head_dim)
    return LayerCost(kv_entries_stored=kv_entries, flops_per_token=core_attn_flops + indexer_flops)


# PAPER: §2.3.2 —— HCA:KV 压到 L/m' 条(Eq.20-23,m' >> m),不做稀疏选择,直接对全部
# L/m' 个压缩块 + n_win 个滑窗 raw token 做稠密核注意力(Eq.26);没有 indexer 开销。
def hca_layer_cost(seq_len: int, m_prime: int, n_win: int, head_dim: int) -> LayerCost:
    kv_entries = seq_len / m_prime
    core_attn_flops = (kv_entries + n_win) * head_dim
    return LayerCost(kv_entries_stored=kv_entries, flops_per_token=core_attn_flops)


# PAPER: §1 "vanilla attention mechanism" 的对照组(引言"quadratic computational
# complexity of the vanilla attention mechanism") —— 不压缩、不稀疏,KV 存满 L 条,
# 单 token FLOPs 与 L 成正比
def dense_baseline_layer_cost(seq_len: int, head_dim: int) -> LayerCost:
    return LayerCost(kv_entries_stored=float(seq_len), flops_per_token=float(seq_len * head_dim))


# PAPER: §2.3 首段"employ their interleaved hybrid configuration" —— 按逐层 compress_ratios
# (混合开关表,见 hybrid_layer.get_dsv4_compress_ratio)把每层的账算出来再取平均,得到
# "整个模型平均单 token 需要多少 KV / 多少 FLOPs"
def hybrid_stack_average_cost(compress_ratios: list, seq_len: int, k: int, n_win: int,
                               head_dim: int, indexer_heads: int = 1,
                               indexer_dim: int | None = None) -> LayerCost:
    kv_sum, flops_sum = 0.0, 0.0
    for ratio in compress_ratios:
        if ratio == 4:
            c = csa_layer_cost(seq_len, m=4, k=k, n_win=n_win, head_dim=head_dim,
                                indexer_heads=indexer_heads, indexer_dim=indexer_dim)
        elif ratio == 128:
            c = hca_layer_cost(seq_len, m_prime=128, n_win=n_win, head_dim=head_dim)
        elif ratio <= 1:
            c = dense_baseline_layer_cost(seq_len, head_dim)
        else:
            raise ValueError(f"Only support compress_ratio in [4, 128] (or <=1 for dense). Got: {ratio}")
        kv_sum += c.kv_entries_stored
        flops_sum += c.flops_per_token
    n = len(compress_ratios)
    return LayerCost(kv_entries_stored=kv_sum / n, flops_per_token=flops_sum / n)


# PAPER: §2.3.4 "DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs
# and 10% of KV cache compared with DeepSeek-V3.2" —— 把两个账本按比例对比(hybrid
# vs baseline),返回 (flops_ratio, kv_ratio),数值应读作"账本模型算出的相对比例",
# 不是对论文原文百分比的复现(见文件顶部说明)。
# PAPER: §2.3.4(同上)
def relative_efficiency(hybrid: LayerCost, baseline: LayerCost) -> tuple:
    return hybrid.flops_per_token / baseline.flops_per_token, hybrid.kv_entries_stored / baseline.kv_entries_stored


# ---------------------------------------------------------------------------
# 精度账(§2.3.4 第一段:混合精度存储 / indexer 低精度计算)
# ---------------------------------------------------------------------------


# PAPER: §2.3.4 "BF16 precision is used for the rotary positional embedding (RoPE)
# dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid
# representation reduces the KV cache size by nearly half compared with pure BF16
# storage." —— 按维度拆分精度算 KV 字节账
# PAPER: §2.3.4(同上,混合精度存储)
def mixed_precision_kv_bytes(kv_entries: float, rope_dims: int, other_dims: int,
                              bytes_bf16: int = 2, bytes_fp8: int = 1) -> float:
    return kv_entries * (rope_dims * bytes_bf16 + other_dims * bytes_fp8)


# PAPER: §2.3.4 —— 对照组:纯 BF16 存储(未采用混合精度前的基线,用来量化"近乎减半")
def pure_bf16_kv_bytes(kv_entries: float, total_dims: int, bytes_bf16: int = 2) -> float:
    return kv_entries * total_dims * bytes_bf16


# ---------------------------------------------------------------------------
# 示意性数值推演(不是论文数字的复现,只验证账本模型量级自洽)
# ---------------------------------------------------------------------------


# PAPER: §1 + §2.3.4(数值推演结果容器,见 worked_example_efficiency)
@dataclass
class WorkedExampleResult:
    hybrid: LayerCost
    baseline_dsa: LayerCost         # csa_layer_cost(m=1) 退化出的 DSA-only 基线
    baseline_dense: LayerCost       # 完全不压缩不稀疏的稠密基线
    flops_ratio_vs_dsa: float
    kv_ratio_vs_dsa: float
    flops_ratio_vs_dense: float
    kv_ratio_vs_dense: float
    kv_bytes_mixed_precision: float
    kv_bytes_pure_bf16: float


# PAPER: §1 + §2.3.4 —— 把 CSA/HCA 混合交错 + 混合精度存储这两笔账串起来跑一遍,
# 用示意性参数(数量级参考落地代码 code_spine 给出的线索,如 window_size/index_topk
# 的存在性,但具体数值不是 DeepSeek 未公开的确切配置)
def worked_example_efficiency(seq_len: int = 1_000_000, compress_ratios: list | None = None,
                               k: int = 2048, n_win: int = 1024, head_dim: int = 128,
                               indexer_heads: int = 4, indexer_dim: int = 64,
                               rope_dims: int = 64) -> WorkedExampleResult:
    if compress_ratios is None:
        # 示意性交错模式:3 个 CSA 层配 1 个 HCA 层(呼应 dossier design_decisions 里
        # "两者互补,交错让每种缺陷被另一种补上"的定性描述;不是发行版配置文件里的真实数组)
        compress_ratios = ([4, 4, 4, 128] * 9)
    hybrid = hybrid_stack_average_cost(compress_ratios, seq_len, k, n_win, head_dim,
                                        indexer_heads, indexer_dim)
    baseline_dsa = csa_layer_cost(seq_len, m=1, k=k, n_win=n_win, head_dim=head_dim,
                                   indexer_heads=indexer_heads, indexer_dim=indexer_dim)
    baseline_dense = dense_baseline_layer_cost(seq_len, head_dim)
    flops_r_dsa, kv_r_dsa = relative_efficiency(hybrid, baseline_dsa)
    flops_r_dense, kv_r_dense = relative_efficiency(hybrid, baseline_dense)
    other_dims = head_dim - rope_dims
    kv_mixed = mixed_precision_kv_bytes(hybrid.kv_entries_stored, rope_dims, other_dims)
    kv_pure = pure_bf16_kv_bytes(hybrid.kv_entries_stored, head_dim)
    return WorkedExampleResult(
        hybrid=hybrid, baseline_dsa=baseline_dsa, baseline_dense=baseline_dense,
        flops_ratio_vs_dsa=flops_r_dsa, kv_ratio_vs_dsa=kv_r_dsa,
        flops_ratio_vs_dense=flops_r_dense, kv_ratio_vs_dense=kv_r_dense,
        kv_bytes_mixed_precision=kv_mixed, kv_bytes_pure_bf16=kv_pure,
    )
