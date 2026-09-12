"""KV cache 元素总账 —— arXiv:2405.04434 §2.1.4 Table 1(四机制每 token 元素数,
按元素计、不分精度)+ §2.1.1 尾句(MHA 2n_h·d_h·l)+ §2.1.3 尾句(MLA
(d_c+d_h^R)l)+ §3.1.2 超参(n_h=128/d_h=128/d_c=512/d_c'=1536/d_h^R=64,
V3 §4.2 同值——576 口径跨代稳定)+ arXiv:2305.13245 §2.2(MHA→MQA 降 H 倍)。

代码对照(引擎账本侧的接入点,叙事引用时用):普通注意力层的页字节公式
2×block_size×num_kv_heads×head_dim×dtype(vllm/v1/kv_cache_interface.py:
L212-L226,那个 2 = K/V 各一份)vs MLAAttentionSpec 去掉 2 因子的
storage_block_size×num_kv_heads×head_dim×dtype(L388-L426,单份潜向量、
head_size=576、num_kv_heads=1——vllm/.../mla_attention.py:L393/L397/L1140-L1152)。
"""


# PAPER: §2.1.1 尾句 ——「MHA needs to cache 2n_h·d_h·l elements for each
# token」:K、V 各 n_h 份逐头向量,每 token 每层 2·n_h·d_h 个元素
def mha_kv_cache_per_token(num_heads: int, head_dim: int, num_layers: int = 1) -> int:
    return 2 * num_heads * head_dim * num_layers


# PAPER: §2.1.4 Table 1 MQA 行 —— 2·d_h·l:单 KV 头(=GQA-1 特例单列,便于
# 与 Table 1 四行逐行对齐)
def mqa_kv_cache_per_token(head_dim: int, num_layers: int = 1) -> int:
    return 2 * head_dim * num_layers


# PAPER: §2.1.4 Table 1 GQA 行 —— 2·n_g·d_h·l:query 头分 G 组、每组共享一份
# K/V(§2.2);G=H 即 MHA、G=1 即 MQA——谱系端点在账上合拢
def gqa_kv_cache_per_token(
    num_groups: int, head_dim: int, num_layers: int = 1
) -> int:
    return 2 * num_groups * head_dim * num_layers


# PAPER: §2.1.3 尾句 ——「DeepSeek-V2 requires a total KV cache containing
# (d_c+d_h^R)l elements」:单份 d_c 维潜向量 c^KV(式 9)加单份 d_h^R 维共享
# 解耦 key k^R(式 15),无 K/V 之分、无 2 因子
def mla_kv_cache_per_token(d_c: int, d_h_R: int, num_layers: int = 1) -> int:
    return (d_c + d_h_R) * num_layers


# PAPER: §2.2 ——「Going from MHA to MQA reduces H key and value heads to a
# single key and value head, reducing the size of the key-value cache and
# therefore amount of data that needs to be loaded by a factor of H」:
# 缩减倍数就是头数 H(cache 与加载量同步降 H 倍)
# PAPER: §2.2 factor-H 句
def mha_to_mqa_cache_reduction_factor(num_heads: int) -> int:
    return num_heads


# PAPER: §2.1.4 Table 1 caption ——「d_c is set to 4d_h and d_h^R is set to
# d_h/2. So, its KV cache is equal to GQA with only 2.25 groups」:
# 等效组数 = (d_c+d_h^R)/(2·d_h) = (4d_h+d_h/2)/(2d_h) = 2.25——把 MLA 放回
# GQA 谱系定位的换算
# PAPER: §2.1.4 Table 1 caption(2.25 groups 句)
def equivalent_gqa_groups(d_c: int, d_h_R: int, head_dim: int) -> float:
    return (d_c + d_h_R) / (2.0 * head_dim)


# PAPER: §3.1.2 —— MLA 超参:「we set the number of attention heads n_h to 128
# and the per-head dimension d_h to 128. The KV compression dimension d_c is
# set to 512, and the query compression dimension d_c' is set to 1536. For the
# decoupled queries and key, we set the per-head dimension d_h^R to 64」
# (V3 §4.2 同值:576=512+64 跨代稳定,不是 V2 的一次性配置)
# PAPER: §3.1.2 MLA 超参段
def deepseek_v2_mla_hyperparams() -> dict:
    return {
        "n_h": 128,        # 注意力头数
        "d_h": 128,        # 每头维度
        "d_c": 512,        # KV 压缩维度(= 4·d_h)
        "d_c_prime": 1536,  # query 压缩维度(压训练激活,不压 cache)
        "d_h_R": 64,       # 解耦 RoPE 每 head 维度(= d_h/2)
    }


# PAPER: §2.1.4 Table 1 —— 四机制同口径总账(每 token 元素数 × 层数 l,
# regardless of the storage precision)。代 DSV2/V3 超参(l=1)即
# MHA 32768 / MQA 256 / GQA-8 2048 / MLA 576;代 GQA 论文场景
# (H 头→G 组)即 §2.2 的 factor-H 缩减账
# PAPER: §2.1.4 Table 1 四行
def table1_kv_cache_per_token(
    num_heads: int,
    head_dim: int,
    num_groups: int,
    d_c: int,
    d_h_R: int,
    num_layers: int = 1,
) -> dict:
    return {
        "MHA": mha_kv_cache_per_token(num_heads, head_dim, num_layers),
        "MQA": mqa_kv_cache_per_token(head_dim, num_layers),
        "GQA": gqa_kv_cache_per_token(num_groups, head_dim, num_layers),
        "MLA": mla_kv_cache_per_token(d_c, d_h_R, num_layers),
    }
