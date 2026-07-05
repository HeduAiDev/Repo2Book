"""ch36 落地对照(vllm_ascend/worker/kvcomp_utils.py) -- KVComp:用 LSH 哈希 + 汉明距离
近似 lightning indexer 的 top-k 选块打分,是 §2.3.1 Eq.(16)-(17) 的一种运行期落地方案
(不是论文正文公式,论文没有描述这一具体工程手段;这里按 code_spine 明确标注的落地代码
做"论文思想 -> 工程近似"的桥接,豁免范围内的落地讲解)。

思路:把 §2.3.1 的 "ReLU(q^I.k^I) 加权求和"打分近似成"汉明距离检索"——每个 query/KV
块先投影到 hash_bits 维再取符号位打包成 uint8,检索时只需按位异或+popcount(汉明距离),
比稠密浮点内积省得多的内存与算力,代价是打分变粗糙(LSH 近似);must_select_blocks 用来
保证 sink(首块)与 recent(最近几块)不会被这个近似检索意外漏选。

落地:KVCompConfig(L144-194)、HashEncoder.compute_hash(L423-458)、KVCompMetaData
(L491-580,initialize_kvcomp_metadata 固定 sink=1, recent=4)。
"""
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# KVCompConfig —— 落地配置真相源
# ---------------------------------------------------------------------------


# PAPER: 落地 vllm_ascend/worker/kvcomp_utils.py:L150-194(仅保留本章要讲的字段子集)
# —— KVComp 落地配置真相源,近似 §2.3.1 Eq.(16)-(17) 的运行期工程实现
@dataclass
class KVCompConfig:
    chunk_size: int = 128                                              # 须能被 128 整除
    chunk_repre_method: str = "max"                                    # "max" | "min" | "sum"
    top_k_ratio_per_layer: list = field(default_factory=lambda: [0.3])
    # 非负下标=从头数,负下标=从尾数;默认 [0,-2,-1] = 强制保留首块(呼应 sink)+ 最近两块(呼应 recent)
    must_select_blocks: list = field(default_factory=lambda: [0, -2, -1])
    seq_len_threshhold: int = 2048                                     # 触发 KVComp 的最小 seq_len
    rollback_layers: list = field(default_factory=list)
    skip_layers: list = field(default_factory=list)


# PAPER: 落地 kvcomp_utils.py chunk_repre_method 字段的具体计算 —— 把整块压成单个代表向量
# (供 hash 编码前先降维,减少要打包/检索的候选数量)
def chunk_representative(chunk: np.ndarray, method: str = "max") -> np.ndarray:
    """chunk:(chunk_size, dim)。返回 (dim,)。"""
    if method == "max":
        return np.max(chunk, axis=0)
    if method == "min":
        return np.min(chunk, axis=0)
    if method == "sum":
        return np.sum(chunk, axis=0)
    raise ValueError(f"unknown chunk_repre_method: {method}")


# ---------------------------------------------------------------------------
# HashEncoder —— LSH 式哈希编码
# ---------------------------------------------------------------------------


# PAPER: 落地 kvcomp_utils.py _init_hash_weights(elide 注释:"torch.linalg.qr 对随机高斯
# 权重做 QR 正交化(Q*sign(diag(R)))") —— 正交化随机投影权重,减小哈希碰撞的系统性偏差
def qr_orthogonal_random_weights(dim_in: int, hash_bits: int, seed: int = 0) -> np.ndarray:
    """要求 dim_in >= hash_bits(reduced QR 才能给出 (dim_in,hash_bits) 的正交列)。"""
    if dim_in < hash_bits:
        raise ValueError(f"dim_in={dim_in} 必须 >= hash_bits={hash_bits}(reduced QR 的前提)")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dim_in, hash_bits))
    Q, R = np.linalg.qr(raw)
    Q = Q * np.sign(np.diag(R))[None, :]   # 消除 QR 分解的符号歧义
    return Q


# PAPER: 落地 vllm_ascend/worker/kvcomp_utils.py:L368-458(HashEncoder)——仅保留
# compute_hash 这条主链路,hash_weight_type="fixed" 的分支/持久化方法省略
@dataclass
class HashEncoder:
    hash_weights: np.ndarray   # (dim_in, hash_bits),hash_bits 须能被 8 整除(按字节打包)

    # PAPER: 同上(HashEncoder 构造约束)—— hash_bits 须能被 8 整除才能按字节打包
    def __post_init__(self):
        if self.hash_weights.shape[1] % 8 != 0:
            raise ValueError("hash_bits(hash_weights 的列数)必须能被 8 整除,才能按字节打包")

    # PAPER: 同上(HashEncoder.hash_bits)
    @property
    def hash_bits(self) -> int:
        return self.hash_weights.shape[1]

    @property
    def hash_numbers(self) -> int:
        return self.hash_bits // 8

    # PAPER: 落地 compute_hash(L423-458) —— xW=x@hash_weights,取符号位,
    # npu_sign_bits_pack 打包成 uint8;这里用 np.packbits 对符号位(xW>0)做等价的按位打包
    def compute_hash(self, x: np.ndarray) -> np.ndarray:
        """x:(...,dim_in)。返回 (...,hash_numbers) uint8 打包哈希码。"""
        xW = x @ self.hash_weights
        sign_bits = (xW > 0).astype(np.uint8)
        return np.packbits(sign_bits, axis=-1)


# PAPER: 落地 compute_hash 的逆运算(elide 注释提到的 "_unpack_hash 把 uint8 解回 ±1"),
# 供测试/数值推演核对打包前后语义一致
def unpack_hash(packed: np.ndarray, hash_bits: int) -> np.ndarray:
    bits = np.unpackbits(packed, axis=-1)[..., :hash_bits]
    return np.where(bits > 0, 1, -1)


# ---------------------------------------------------------------------------
# 汉明距离 top-k 选块(运行期落地的核心检索)
# ---------------------------------------------------------------------------


# PAPER: 落地近似 §2.3.1 Eq.(16)-(17) 的打分/选块 —— 汉明距离越小代表打包前的浮点向量
# 越相似(近似"点积越大"),用它替代 ReLU 加权内积做 top-k 检索
def hamming_distance_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a,b:(...,n_bytes) uint8 打包码(可广播)。返回逐元素对的汉明距离。"""
    x = np.bitwise_xor(a, b)
    return np.unpackbits(x, axis=-1).sum(axis=-1)


# PAPER: 落地 KVCompConfig.must_select_blocks + initialize_kvcomp_metadata(sink=1,recent=4)
# —— 把 sink/recent 两个语义量映射成 must_select_blocks 风格的下标列表
def must_select_indices_for(sink: int, recent: int, num_blocks: int) -> list:
    idx = list(range(min(sink, num_blocks)))
    idx += list(range(max(0, num_blocks - recent), num_blocks))
    return sorted(set(idx))


# PAPER: 落地近似 §2.3.1 Eq.(17) 的运行期版本 —— 汉明距离 top-k,加 must_select_blocks 强制并入
def select_topk_blocks_by_hamming(query_hash: np.ndarray, key_hashes: np.ndarray, top_k: int,
                                   must_select_blocks: list | None = None) -> np.ndarray:
    """query_hash:(n_bytes,);key_hashes:(num_blocks,n_bytes)。must_select_blocks 支持负下标
    (从末尾数,如 -1 = 最后一块)。返回升序排列、去重后的块下标数组。"""
    num_blocks = key_hashes.shape[0]
    dists = hamming_distance_packed(query_hash[None, :], key_hashes)
    k_eff = min(top_k, num_blocks)
    topk_idx = np.argsort(dists)[:k_eff]
    forced = set()
    if must_select_blocks:
        for idx in must_select_blocks:
            forced.add(idx if idx >= 0 else num_blocks + idx)
    selected = sorted(set(topk_idx.tolist()) | forced)
    return np.array(selected, dtype=int)


# PAPER: 落地 vllm_ascend/worker/kvcomp_utils.py:L491-513(仅保留 sink/recent + 配置引用)
@dataclass
class KVCompMetaData:
    """落地 vllm_ascend/worker/kvcomp_utils.py:L491-513(仅保留 sink/recent + 配置引用,
    运行期的张量集合按需在测试里现造小规模替身,不引入未使用的字段)。"""
    kvcomp_config: KVCompConfig
    sink: int = 1
    recent: int = 4
    num_actual_tokens: int = 0
