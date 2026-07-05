"""ch32 §2.3 Inference Costs(paper-dsa.md) —— 成本模型 O(L.d_idx + k.d) 与加速账。

论文原句:"DSA reduces the core attention complexity of the main model from O(L^2) to O(Lk)
...Although the lightning indexer still has a complexity of O(L^2), it requires much less
computation compared with MLA...Combined with our optimized implementation, DSA achieves a
significant end-to-end speedup"(§2.3)。§2.3 的 Figure 3 展示的是"token 成本随其在序列中
位置变化"——即单个 decode 步(一个 query 对增长中的 KV cache 打分)的成本,而不是整条
prefill 序列的成本总和。本文件按这个"单 decode 步、query 位于满上下文末尾(context_len=L)"
的框架建模:

  (a) 单步稠密主注意力成本 = context_len * per_kv_dim               (对全部前驱打分)
  (b) 单步稀疏主注意力成本 = k * per_kv_dim                          (只对 top-k 个前驱打分)
  (c) 单步 indexer 成本    = context_len * indexer_heads * indexer_dim (indexer 仍要扫全部前驱)

这样"主注意力降幅"精确等于 context_len/k(与 dossier 数值推演的 256x/64x 直接对应);
再把 indexer 开销并入总成本,得到更保守、更诚实的端到端加速账——避免读者误以为总复杂度
变成了线性。若把这套单步成本对 t=1..L 累加(对应整条 prefill 序列),就得到论文所说
"indexer 仍是 O(L^2)"的整体标度(见 prefill_total_* 系列函数)。
"""
from dataclasses import dataclass


# PAPER: §2.3 文字("reduces the core attention complexity...from O(L^2) to O(Lk)") ——
# 单个 decode 步:query 对 context_len 个前驱做稠密打分,每次点积维度 per_kv_dim
def decode_step_main_cost_dense(context_len: int, per_kv_dim: int) -> int:
    return context_len * per_kv_dim


# PAPER: §2.3 —— 单个 decode 步:query 只对 top-k 个选中前驱做打分,与 context_len 无关
def decode_step_main_cost_sparse(k: int, per_kv_dim: int) -> int:
    return k * per_kv_dim


# PAPER: §2.3 文字("the lightning indexer still has a complexity of O(L^2)...requires much
# less computation compared with MLA") —— 单个 decode 步:indexer 仍要对 context_len 个
# 前驱打分,但每次点积维度只是 indexer_heads*indexer_dim(远小于主注意力的 per_kv_dim)
def decode_step_indexer_cost(context_len: int, indexer_heads: int, indexer_dim: int) -> int:
    return context_len * indexer_heads * indexer_dim


# PAPER: §2.3 —— 加速账数值容器,汇总下面 speedup_accounting() 算出的四个量
@dataclass
class SpeedupAccounting:
    dense_main: int
    sparse_main: int
    indexer: int
    main_only_speedup: float   # 只看主注意力:dense_main / sparse_main == context_len/k(精确)
    end_to_end_speedup: float  # 算上 indexer:dense_main / (sparse_main + indexer),更保守


# PAPER: §2.3 —— 把"主注意力降 X 倍"与"indexer 仍是 O(L^2) 但常数小"两笔账一次性算清,
# 单个 decode 步、context_len=L(满上下文,worst case)
def speedup_accounting(
    context_len: int, k: int, per_kv_dim: int, indexer_heads: int, indexer_dim: int
) -> SpeedupAccounting:
    dense_main = decode_step_main_cost_dense(context_len, per_kv_dim)
    sparse_main = decode_step_main_cost_sparse(k, per_kv_dim)
    idx_cost = decode_step_indexer_cost(context_len, indexer_heads, indexer_dim)
    total_sparse = sparse_main + idx_cost
    return SpeedupAccounting(
        dense_main=dense_main,
        sparse_main=sparse_main,
        indexer=idx_cost,
        main_only_speedup=dense_main / sparse_main,
        end_to_end_speedup=dense_main / total_sparse,
    )


# 数值锚说明(结合落地代码) —— vllm_ascend 默认 index_n_heads=64、index_head_dim=128
# (dsa_v1.py L829-831 注释)、index_topk=512(dsa_v1.py L831 "# 512")。主注意力维度
# per_kv_dim 取 ch31 primer 复用的 DeepSeek-V2 MLA 数字:n_h=128 个 query 头共享同一份
# 潜向量(d_c=512)+ 解耦 key(d_h^R=64),per_kv_dim = n_h*(d_c+d_h^R) = 128*(512+64) = 73728。
# PAPER: §2.3 —— L=131072(V3.1-Terminus 128K 续训长度,paper-dsa §2.1.1)代入成本模型
def vllm_ascend_deployment_numbers(k: int = 512) -> SpeedupAccounting:
    seq_len = 131072
    n_h, d_c, d_h_r = 128, 512, 64
    per_kv_dim = n_h * (d_c + d_h_r)
    return speedup_accounting(
        context_len=seq_len, k=k, per_kv_dim=per_kv_dim, indexer_heads=64, indexer_dim=128
    )


# PAPER: §2.1.1 文字("select 2048 key-value tokens for each query token") —— 论文训练用的
# k=2048,与 vllm_ascend 落地默认 k=512 对照,数值推演段用来展示"更保守的 k 换更大加速"
def paper_training_numbers() -> SpeedupAccounting:
    return vllm_ascend_deployment_numbers(k=2048)


# --- 整条 prefill 序列的累加账(对应论文"indexer 仍是 O(L^2)"里的 O(L^2) 是怎么来的) ---


# PAPER: §2.3 —— 把 decode_step_main_cost_dense 对全部 query 位置 t=1..L 累加,
# 即因果注意力对整条序列的总代价:Sum_{t=1}^{L} t * per_kv_dim = L(L+1)/2 * per_kv_dim
def prefill_total_main_cost_dense(seq_len: int, per_kv_dim: int) -> int:
    return (seq_len * (seq_len + 1) // 2) * per_kv_dim


# PAPER: §2.3 —— 稀疏版:每个 query 只对 k 个前驱打分,总代价随 L 线性增长(O(Lk))
def prefill_total_main_cost_sparse(seq_len: int, k: int, per_kv_dim: int) -> int:
    return seq_len * k * per_kv_dim


# PAPER: §2.3 —— indexer 对每个 query 仍要扫其全部前驱,累加后是 O(L^2)(论文原句的
# 复杂度标度就是这样得出的),只是常数项 indexer_heads*indexer_dim 远小于 per_kv_dim
def prefill_total_indexer_cost(seq_len: int, indexer_heads: int, indexer_dim: int) -> int:
    return (seq_len * (seq_len + 1) // 2) * indexer_heads * indexer_dim
