"""ch31 §2.1.4 —— KV cache 元素数对比（Table 1）+ DeepSeek-V2 真实数字代入。

DeepSeek-V2 论文 arXiv:2405.04434 Table 1：MHA=2*n_h*d_h*l，GQA=2*n_g*d_h*l，MQA=2*d_h*l，
MLA=(d_c+d_h^R)*l ≈ 4.5*d_h*l（当 d_c=4*d_h, d_h^R=d_h/2 时）。
"""
from dataclasses import dataclass


# PAPER: §2.1.1 Eq.7-8 讨论段 —— 标准 MHA 的 KV cache：每 token 每层 2*n_h*d_h 个元素
def kv_cache_elements_mha(n_h: int, d_h: int, l: int) -> int:
    return 2 * n_h * d_h * l


# PAPER: §2.1.4 Table 1 —— GQA：n_g 个分组，每组仍产出满维 K/V
def kv_cache_elements_gqa(n_g: int, d_h: int, l: int) -> int:
    return 2 * n_g * d_h * l


# PAPER: §2.1.4 Table 1 —— MQA：只有 1 组（n_g=1 的 GQA 特例）
def kv_cache_elements_mqa(d_h: int, l: int) -> int:
    return 2 * d_h * l


# PAPER: §2.1.4 Table 1 —— MLA：只缓存潜向量 c^{KV}（d_c 维）与解耦 key k^R（d_h^R 维）
def kv_cache_elements_mla(d_c: int, d_h_r: int, l: int) -> int:
    return (d_c + d_h_r) * l


# PAPER: §2.1.4 Table 1 —— 四种注意力机制 KV cache 元素数的汇总容器
@dataclass
class KvCacheComparison:
    mha: int
    gqa: int
    mqa: int
    mla: int
    mla_equivalent_gqa_groups: float  # (d_c+d_h_r) / (2*d_h)——MLA 的 KV cache 相当于几组 GQA
    mla_compression_ratio_vs_mha: float  # mha / mla


# PAPER: §2.1.4 Table 1 —— 一次性算出 MHA/GQA/MQA/MLA 四行 + 表格脚注里的等效 GQA 组数
def compare_kv_cache(n_h: int, d_h: int, l: int, d_c: int, d_h_r: int, n_g: int) -> KvCacheComparison:
    mha = kv_cache_elements_mha(n_h, d_h, l)
    gqa = kv_cache_elements_gqa(n_g, d_h, l)
    mqa = kv_cache_elements_mqa(d_h, l)
    mla = kv_cache_elements_mla(d_c, d_h_r, l)
    return KvCacheComparison(
        mha=mha, gqa=gqa, mqa=mqa, mla=mla,
        mla_equivalent_gqa_groups=(d_c + d_h_r) / (2 * d_h),
        mla_compression_ratio_vs_mha=mha / mla,
    )


# PAPER: §3.1.2 模型超参 —— DeepSeek-V2 真实维度代入 Table 1
def deepseek_v2_numbers() -> KvCacheComparison:
    """n_h=128, d_h=128, l=60, d_c=4*d_h=512, d_h^R=d_h/2=64（论文 §3.1.2、Table 1 脚注）。
    n_g 取论文举的对照例子——GQA 若要匹配 MLA 的 cache 量，只需 2.25 组。
    """
    n_h, d_h, l = 128, 128, 60
    d_c, d_h_r = 4 * d_h, d_h // 2
    return compare_kv_cache(n_h, d_h, l, d_c, d_h_r, n_g=1)


# PAPER: §2.1.4 Table 1 公式代入 —— 小维度版，与 mla_reference.MLAConfig 的默认玩具维度对齐
def toy_numbers() -> KvCacheComparison:
    n_h, d_h, l = 4, 8, 2
    d_c, d_h_r = 12, 4  # 刻意选得比 d_h*n_h=32 小，体现"压缩"而非精确复刻论文的 4x/0.5x 比例
    return compare_kv_cache(n_h, d_h, l, d_c, d_h_r, n_g=1)
