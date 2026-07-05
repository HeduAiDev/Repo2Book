"""ch32 §2.1 Eq.(1)(paper-dsa.md,arXiv:2512.02556) —— Lightning Indexer 打分函数。

DSA 是 NSA 的简化后裔:三支路(压缩+选择+滑窗)+ 块级重要性打分被替换为一个更简单的
token 级打分函数——lightning indexer。H^I 个 indexer 头各自对 query token h_t 与前驱
token h_s 做 q^I.k^I 点积、ReLU 激活(不是 softmax——论文明说是为了吞吐:头少、可 FP8
实现),再按标量权重 w^I 加权求和,得到 I_{t,s}:query token t 对前驱 token s 的相关性
代理分数。

落地(vllm_ascend/attention/dsa_v1.py L1443-1462、L2735):indexer_heads=H^I、
inderxer_dim=d^I、weights_proj 产出 w^I;indexer key k^I 每个 token 只有一份(不像主
attention 那样每头独立),被全部 H^I 个 query 头共享——这正是 DSA "instantiate under MLA
的 MQA 模式"(paper-dsa §2.1 脚注)在 indexer 侧的体现。sfa_v1.py 里 indexer_select_pre_process
只产出一份 k_li(head_dim 维,unsqueeze 后广播),indexer_select_post_process 产出 (n_head,
head_dim) 形状的 q_li——形状恰好印证这一点。
"""
import numpy as np


# 单对 (t,s) 的 indexer 打分。q_t:(H^I,d^I) 每头 query;k_s:(d^I,) 该 token 单份共享 key
# (全部 H^I 个头共享同一个 k_s——见 indexer_select_pre_process 只产出 k_li 一份);
# w_t:(H^I,) 每头标量权重。
# PAPER: §2.1 Eq.(1) —— I_{t,s} = Sum_{j=1}^{H^I} w_{t,j}^I . ReLU(q_{t,j}^I . k_s^I)
def indexer_score(q_t: np.ndarray, k_s: np.ndarray, w_t: np.ndarray) -> float:
    dots = q_t @ k_s          # (H^I,) 每头 q^I.k^I 点积,同一个 k_s 被所有头共享
    relu = np.maximum(dots, 0.0)
    return float(w_t @ relu)


# PAPER: §2.1 Eq.(1) —— 对整条前驱序列批量算 I_{t,:}(query t 对全部 s<=t 的打分),
# 与逐对调用 indexer_score 等价,但走矩阵乘更贴近落地代码里融合算子的批处理形态
def indexer_scores_for_query(q_t: np.ndarray, k_seq: np.ndarray, w_t: np.ndarray) -> np.ndarray:
    """q_t:(H^I,d^I);k_seq:(t,d^I)(indexer key 每 token 只有一份,不像主注意力那样
    每头独立);w_t:(H^I,)。返回 I_{t,:} 形状 (t,)。"""
    dots = q_t @ k_seq.T   # (H^I, t)
    relu = np.maximum(dots, 0.0)
    return w_t @ relu      # (t,)
