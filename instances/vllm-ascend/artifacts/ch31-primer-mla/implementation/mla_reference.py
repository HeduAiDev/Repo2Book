"""ch31 —— MLA 参考实现装配：把 Eq.9-19 拼成一个可跑的前向，并验证"decode 吸收路径 ==
prefill 物化路径"这条贯穿全文的恒等式，逐步落到 vllm_ascend/attention/mla_v1.py 的真实分流。

两条路径，和落地代码的 forward() 一一对应：
  - forward_full(prefill 物化路径，对应 mla_preprocess_prefill + _forward_prefill)：
    对整段序列一次性算出 c^{KV}/c^Q/q^C/k^C/v^C/q^R/k^R，注意力用物化的 k^C、v^C 直接算——
    简单、但需要显式产出满维 key/value。
  - decode_step(decode 吸收路径，对应 mla_preprocess_decode + _q_proj_and_k_up_proj +
    _forward_decode + _v_up_proj)：只缓存 c^{KV} 历史与 k^R 历史（对应 exec_kv_decode 写
    进 KV cache 的两个张量），nope 打分用预先吸收好的静态矩阵 W~ 直接在潜空间做，
    从不重新物化任何历史 token 的 k^C/v^C。

本文件的核心测试（tests/test_mla_reference.py）：把同一份权重、同一段输入逐 token 喂给
decode_step，得到的输出应与一次性跑 forward_full 完全一致——这就是"权重吸收 + 解耦 RoPE
永远正确"的端到端数值证明。
"""
from dataclasses import dataclass

import numpy as np

from mha_baseline import merge_heads
from low_rank_mla import (
    KVCompressionWeights,
    QCompressionWeights,
    init_kv_compression_weights,
    init_q_compression_weights,
    kv_joint_compression,
    q_joint_compression,
    precompute_absorbed_query_weights,
    precompute_uv_head_slices,
    split_kv_heads,
)
from decoupled_rope import (
    DecoupledRopeWeights,
    init_decoupled_rope_weights,
    decoupled_query_rope,
    decoupled_key_rope,
    concat_nope_rope_query,
    concat_nope_rope_key,
    decoupled_attention_scores,
    rope_rotation_matrix,
)
from numerics import softmax


# PAPER: §2.1.2 Eq.9-13 + §2.1.3 Eq.14-15 —— 装配全部维度记号的配置容器
@dataclass
class MLAConfig:
    d: int          # 模型维
    n_h: int        # 头数
    d_h: int        # 每头维（nope 部分；论文里 k^C/v^C/q^C 都是这个维度）
    d_c: int        # KV 压缩维（缓存的就是这个）
    d_c_q: int      # query 压缩维（只降训练激活显存，不影响 KV cache）
    d_h_r: int      # 解耦 RoPE 每头维


# PAPER: §2.1.2 Eq.9-13 + §2.1.3 Eq.14-15 + Eq.8/19 的 W^O —— 全套 MLA 权重容器
@dataclass
class MLAWeights:
    kv: KVCompressionWeights
    q: QCompressionWeights
    rope: DecoupledRopeWeights
    W_O: np.ndarray  # (d, n_h*d_h)


# PAPER: 同上 —— 随机初始化全套 MLA 权重
def init_mla_weights(cfg: MLAConfig, seed: int = 0) -> MLAWeights:
    rng = np.random.default_rng(seed)
    return MLAWeights(
        kv=init_kv_compression_weights(cfg.d, cfg.n_h, cfg.d_h, cfg.d_c, seed=seed),
        q=init_q_compression_weights(cfg.d, cfg.n_h, cfg.d_h, cfg.d_c_q, seed=seed + 1),
        rope=init_decoupled_rope_weights(cfg.d, cfg.n_h, cfg.d_h_r, cfg.d_c_q, seed=seed + 2),
        W_O=rng.normal(scale=1.0 / np.sqrt(cfg.n_h * cfg.d_h), size=(cfg.d, cfg.n_h * cfg.d_h)),
    )


# PAPER: §2.1.2 Eq.9 + §2.1.3 Eq.15 文字 "the decoupled key should also be cached" —— 推理期唯二要缓存的量
@dataclass
class DecodeCache:
    """decode 吸收路径实际需要缓存的东西——只有这两样，对应 vllm_ascend 里
    exec_kv_decode 写入 KV cache 的 c^{KV}（kv_lora_rank 维）与 k^R（qk_rope_head_dim 维）。
    从始至终不出现任何 (T, n_h*d_h) 形状的物化 key/value。
    """
    c_kv_history: np.ndarray = None   # (t, d_c)
    k_r_history: np.ndarray = None    # (t, d_h_r)


# PAPER: Eq.9-19 端到端装配 —— prefill 物化路径 / decode 吸收路径二选一的壳
class MLAReference:
    # PAPER: §2.1.2 文字（Eq.11 之后）—— 加载后一次性算好吸收矩阵，此后不再重算
    def __init__(self, cfg: MLAConfig, seed: int = 0):
        self.cfg = cfg
        self.w = init_mla_weights(cfg, seed)
        # 权重吸收在"加载后"算一次，之后推理期反复复用——对应 process_weights_after_loading。
        self.w_tildes = precompute_absorbed_query_weights(self.w.q, self.w.kv, cfg.n_h, cfg.d_h)
        self.w_uv_heads = precompute_uv_head_slices(self.w.kv, cfg.n_h, cfg.d_h)

    # ---------------------------------------------------------------- prefill（物化）路径 ---
    # PAPER: Eq.9-13 压缩 + Eq.14-18 解耦注意力，全部物化后一次性对整段序列计算
    def forward_full(self, h_seq: np.ndarray, positions) -> np.ndarray:
        cfg, w = self.cfg, self.w
        c_kv_seq, k_c_seq, v_c_seq = kv_joint_compression(h_seq, w.kv)
        c_q_seq, q_c_seq = q_joint_compression(h_seq, w.q)
        k_c_heads, v_c_heads, q_c_heads = split_kv_heads(k_c_seq, v_c_seq, q_c_seq, cfg.n_h, cfg.d_h)

        q_r_heads = decoupled_query_rope(c_q_seq, w.rope.W_QR, positions, cfg.n_h, cfg.d_h_r)
        k_r_seq = decoupled_key_rope(h_seq, w.rope.W_KR, positions)

        q_full = concat_nope_rope_query(q_c_heads, q_r_heads)
        k_full = concat_nope_rope_key(k_c_heads, k_r_seq)
        attn_w = decoupled_attention_scores(q_full, k_full)  # (n_h, T, T)

        o_heads = np.stack([attn_w[i] @ v_c_heads[i] for i in range(cfg.n_h)])  # 物化 v^C 直接加权求和
        return merge_heads(o_heads) @ w.W_O.T

    # ---------------------------------------------------------------- decode（吸收）路径 ---
    # PAPER: §2.1.2 文字（权重吸收）+ §2.1.3 Eq.14-15（解耦 rope 现场旋转）—— 增量单步、只用缓存
    def decode_step(self, h_t: np.ndarray, pos_t: int, cache: DecodeCache):
        cfg, w = self.cfg, self.w
        c_kv_t = h_t @ w.kv.W_DKV.T   # (d_c,) —— 唯一要写入 cache 的 KV 侧张量
        c_q_t = h_t @ w.q.W_DQ.T      # (d_c_q,)

        c_kv_hist = c_kv_t[None, :] if cache.c_kv_history is None else np.concatenate([cache.c_kv_history, c_kv_t[None, :]], axis=0)

        # nope 打分：q~ = W~_i @ c_q_t，直接和整段 c^{KV} 历史内积——不物化任何历史 key
        nope_scores = np.zeros((cfg.n_h, c_kv_hist.shape[0]))
        for i in range(cfg.n_h):
            q_tilde = self.w_tildes[i] @ c_q_t          # (d_c,)
            nope_scores[i] = c_kv_hist @ q_tilde         # (t,)

        # rope 打分：q^R_t 现场旋转（只有 d_h_r 维，代价与历史长度无关）；k^R 历史缓存/现场旋转均可，
        # 这里选择“写 cache 时就旋转好”，与 exec_kv_decode 把 RoPE 融进写 cache 那一步一致。
        pre_qr = c_q_t @ w.rope.W_QR.T                    # (n_h*d_h_r,)
        pre_qr_heads = pre_qr.reshape(cfg.n_h, cfg.d_h_r)
        q_r_t = np.stack([rope_rotation_matrix(pos_t, cfg.d_h_r) @ pre_qr_heads[i] for i in range(cfg.n_h)])  # (n_h, d_h_r)

        k_r_t = rope_rotation_matrix(pos_t, cfg.d_h_r) @ (h_t @ w.rope.W_KR.T)  # (d_h_r,)
        k_r_hist = k_r_t[None, :] if cache.k_r_history is None else np.concatenate([cache.k_r_history, k_r_t[None, :]], axis=0)

        rope_scores = np.zeros((cfg.n_h, k_r_hist.shape[0]))
        for i in range(cfg.n_h):
            rope_scores[i] = k_r_hist @ q_r_t[i]

        scale = np.sqrt(cfg.d_h + cfg.d_h_r)
        combined = (nope_scores + rope_scores) / scale  # (n_h, t)——因果性天然满足：只对已缓存的历史打分

        attn_w = softmax(combined, axis=-1)  # 单个新 query，对已有全部历史做 softmax（无需再掩码）

        o_heads = np.zeros((cfg.n_h, cfg.d_h))
        for i in range(cfg.n_h):
            latent_out = attn_w[i] @ c_kv_hist            # (d_c,) —— 在潜空间里加权求和
            o_heads[i] = self.w_uv_heads[i] @ latent_out  # 吸收 W^{UV}：还原到 value 空间

        u_t = o_heads.reshape(-1) @ w.W_O.T
        return u_t, DecodeCache(c_kv_history=c_kv_hist, k_r_history=k_r_hist)
