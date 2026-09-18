# -*- coding: utf-8 -*-
"""DSpark 半自回归 drafter —— arXiv:2607.05147 §2.2(Eq.(2)(3):DFlash 骨干的
上下文 KV 注入)+ §3.1(Eq.(4)(5):半自回归两阶段/Markov 头/anchor-as-first)
+ §3.2.1(Eq.(7):置信度头)。

⚠ 论文包两处 LaTeXML 截断(写作引用须带口径):
- Eq.(4)(softmax 形式)在 paper.md:L95 句中断;p_k(x) ∝ softmax(U_k(x) +
  B(x_{k−1},x)) 的形式由 Eq.(5)(低秩 B=W_1W_2 的偏置以加法注入)与 vLLM 落地
  (vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:L118-L148
  `logits_i = base_logits[:, i] + bias`)双向重构确认;
- RNN 头的门控更新公式整体缺失(§4.3.2:仅长块边际增益、默认 Markov 头;
  vLLM 也只有 Markov 头)——本参考实现不实现 RNN 头。

块内层结构(残差/单头/无 FFN)是论文包未给细节的最小形态(细节在外部论文
DFlash);论文写明的部分——Eq.(2) 的 W_c 投影+RMSNorm、Eq.(3) 的同 W^K/W^V
序列维拼接、块内双向注意、共享 target embed/lm_head(冻结)、γ 输入(anchor +
γ−1 mask)出 γ logits——逐条实现。
"""
import numpy as np

from spec_decode import softmax_lastdim, analytical_acceptance_rate

MASK = -1  # 输入串的 mask 槽约定(非词表 id;forward 时走 mask_embed)


# PAPER: §2.2 Eq.(2) —— RMSNorm 算子(x/√(mean(x²)+eps);论文只引用算子未给
#   γ/eps,取 γ=1、eps=1e-6 的标准退化形态,ch24 参考实现同款口径)
def rms_norm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


# PAPER: §2.2 Eq.(2) —— H_ctx = RMSNorm(W_c[H^{(l1)};…;H^{(lm)}]):m 层 target
#   隐状态沿特征维拼接(md 维)后由 W_c∈R^{d×md} 投到 draft 隐空间,prefill 一次
def project_context(concat_hiddens, W_c, eps=1e-6):
    return rms_norm(np.asarray(concat_hiddens, dtype=float) @ np.asarray(W_c).T, eps)


# PAPER: §2.2 Eq.(3) —— 单层注入式注意:K_i = [W_i^K H_ctx; W_i^K H_d]、V_i 同构
#   (上下文与块用同一 W^K/W^V,沿序列维拼接);块内查询双向注意上下文+全块。
#   返回残差形态 h + Attn@W_O(softmax(qKᵀ/√d) 标准缩放算子)
def bidirectional_attention_with_context(h_d, K_ctx, V_ctx, W_q, W_k, W_v, W_o):
    h_d = np.asarray(h_d, dtype=float)
    d_k = W_q.shape[0]
    q = h_d @ W_q.T
    K = np.vstack([K_ctx, h_d @ W_k.T])
    V = np.vstack([V_ctx, h_d @ W_v.T])
    scores = q @ K.T / np.sqrt(d_k)
    weights = softmax_lastdim(scores)
    return h_d + (weights @ V) @ W_o.T


# PAPER: §2.2 + §3.1 —— 并行骨干(DFlash 式):上下文 KV 预计算 + 查询块单次
#   前向出 h_1..h_γ 与 base logits U_1..U_γ;embed/lm_head 共享自 target(冻结)
class ParallelBackbone:
    def __init__(self, embed_matrix, lm_head, mask_embed, W_c, layers):
        # layers: [(W_q, W_k, W_v, W_o), …](单头;层结构为论文未细说的最小形态)
        self.embed_matrix = embed_matrix
        self.lm_head = lm_head
        self.mask_embed = np.asarray(mask_embed, dtype=float)
        self.W_c = np.asarray(W_c, dtype=float)
        self.layers = [tuple(np.asarray(w, dtype=float) for w in layer) for layer in layers]

    # PAPER: §2.2 Eq.(2) + Eq.(3) 左半 —— prefill 一次:上下文隐状态 → 每层
    #   K/V_ctx(vLLM precompute_and_store_context_kv 同构:预写进 draft KV cache)
    def precompute_context(self, concat_hiddens):
        H_ctx = project_context(concat_hiddens, self.W_c)
        kv = []
        for _, W_k, W_v, _ in self.layers:
            kv.append((H_ctx @ W_k.T, H_ctx @ W_v.T))
        return {"H_ctx": H_ctx, "kv": kv}

    # PAPER: §3.1 —— 单次前向:γ 个输入位(anchor/mask 经嵌入)→ h 与 base
    #   logits U = h @ lm_headᵀ(共享、冻结);拼接=注意时 vstack,与 γ 无关的
    #   单次前向即 T_draft 近独立于块长的结构性来源
    def forward(self, input_ids, context):
        rows = [self.mask_embed if t == MASK else self.embed_matrix[t] for t in input_ids]
        h = np.asarray(rows, dtype=float)
        for (W_q, W_k, W_v, W_o), (K_ctx, V_ctx) in zip(self.layers, context["kv"]):
            h = rms_norm(bidirectional_attention_with_context(h, K_ctx, V_ctx, W_q, W_k, W_v, W_o))
        return h, h @ self.lm_head.T


# PAPER: §2.2 + §3.1 —— 玩具实例化(示教轨迹用):小参数随机初始化,结构即论文形态
def make_toy_backbone(vocab_size, d, num_layers, num_target_layers, seed=0):
    rng = np.random.default_rng(seed)
    return ParallelBackbone(
        embed_matrix=rng.normal(0.0, 0.5, size=(vocab_size, d)),
        lm_head=rng.normal(0.0, 0.5, size=(vocab_size, d)),
        mask_embed=rng.normal(0.0, 0.5, size=d),
        W_c=rng.normal(0.0, 0.5, size=(d, num_target_layers * d)),
        layers=[
            tuple(rng.normal(0.0, 0.5, size=(d, d)) for _ in range(4))
            for _ in range(num_layers)
        ],
    )


# PAPER: §3.1 Eq.(5) —— Markov 头:一阶转移偏置低秩分解 B = W_1W_2
#   (W_1∈R^{V×r} 嵌入查表、W_2∈R^{r×V} logit 投影;r=256 默认)。
#   vLLM 同构:DSparkMarkovHead(vllm/model_executor/models/qwen3_dspark.py:L36-L78,
#   markov_w1=nn.Embedding(V,r) + markov_w2=ParallelLMHead(V,r))
# PAPER: §3.1 Eq.(5) —— Markov 头(低秩 B=W_1W_2)
class MarkovHead:
    # PAPER: §3.1 Eq.(5) —— 持有低秩因子 W_1/W_2
    def __init__(self, w1, w2):
        self.w1 = np.asarray(w1, dtype=float)
        self.w2 = np.asarray(w2, dtype=float)
        assert self.w1.shape[1] == self.w2.shape[0], "rank 不一致"

    # PAPER: §3.1 Eq.(5) —— W_1 嵌入查表([B] → [B,r];speculator 串行循环第一步)
    def embed(self, token_ids):
        return self.w1[np.asarray(token_ids)]

    # PAPER: §3.1 Eq.(5) —— B(x_{k−1},·) = W_1[x_{k−1}]W_2 ∈ R^V([B,r] → [B,V])
    def bias(self, markov_embed):
        return markov_embed @ self.w2


# PAPER: §3.1 Eq.(5) + §4.3.2 —— 低秩参数账:全表 V×V vs 低秩 2·V·r;
#   V=129280、r=256 → 各 ≈33.1M、共 ≈66M,V/(2r) ≈252.5x 省(论文口径 ≈253x);
#   每步代价 = O(r) 查表 + O(rV) GEMV(两个 GEMV 就是全部序列性)
def markov_head_param_account(vocab_size, rank):
    full = vocab_size * vocab_size
    low_rank = 2 * vocab_size * rank
    return {"full": full, "low_rank": low_rank, "saving_factor": full / low_rank}


# PAPER: §3.2.1 Eq.(7) —— 置信度头 c_k = σ(w·[h_k; W_1[x_{k−1}]]):
#   轻量线性投影 + sigmoid,建模「前缀全被接受时位 k 存活」的条件概率。
#   vLLM 边界:confidence_head 权重随 checkpoint 到货但推理未接线
#   (qwen3_dspark.py:L184-L188 显式 skip)——本实现是论文侧蓝图
# PAPER: §3.2.1 Eq.(7) —— 置信度头
class ConfidenceHead:
    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    # PAPER: §3.2.1 Eq.(7) —— h_k 骨干隐状态、W_1[x_{k−1}] Markov 嵌入(拼接后投影)
    def confidence(self, h_k, markov_embed_prev):
        z = float(np.concatenate([np.asarray(h_k), np.asarray(markov_embed_prev)]) @ self.w)
        return 1.0 / (1.0 + np.exp(-z))


# PAPER: §3.2.1 Eq.(8) —— 监督标签 c* = 1 − ½‖p_d − p_t‖₁(解析接受率;
#   与 §2.1 接受准则同一条 TV 恒等式——同一组数字 (0.5,0.5)vs(0.7,0.3) → 0.8)
def analytical_confidence_label(p_d, p_t):
    return analytical_acceptance_rate(p_t, p_d)


# PAPER: §3.1 —— anchor-as-first("minor modification"):γ 个输入(anchor +
#   γ−1 mask)产出 γ 个 draft logits;每查询位都是预测位(sample_off=0、
#   sample_pos=query_pos+1)。vLLM:v1/worker/gpu/spec_decode/dspark/speculator.py:L43-L52
def anchor_as_first_inputs(anchor_id, gamma):
    return [int(anchor_id)] + [MASK] * (gamma - 1)


# PAPER: §2.2 —— DFlash 原布局:anchor + γ 个 mask(1+γ 输入,只 mask 位出
#   logits、预测自身位置;anchor 是 bonus 槽)——对照物
def fill_in_inputs(anchor_id, gamma):
    return [int(anchor_id)] + [MASK] * gamma


# PAPER: §3.1 —— 查询位布局小账:DSpark 恰 N 查询/全位采样/预测位=查询位+1
#   vs DFlash N+1 查询/anchor 位不采样/偏移 1 起采样(布局 kernel 的
#   SAMPLE_FROM_ANCHOR 分叉,vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:L574-L585)
def query_layout(method, num_speculative_tokens):
    N = int(num_speculative_tokens)
    if method == "dspark":
        records = [
            {"query_off": k,
             "input": "anchor" if k == 0 else "mask",
             "sampled": True,
             "target_off": k + 1}
            for k in range(N)
        ]
        return {"num_queries": N, "records": records}
    if method == "dflash":
        records = [{"query_off": 0, "input": "anchor", "sampled": False, "target_off": None}]
        records += [
            {"query_off": k, "input": "mask", "sampled": True, "target_off": k}
            for k in range(1, N + 1)
        ]
        return {"num_queries": N + 1, "records": records}
    raise ValueError(f"unknown method: {method}")


# PAPER: §3.1 —— 调度器侧 lookahead 槽位账:DSpark 恰 N(anchor 是预测位)、
#   DFlash N+1(in-fill:最后已采 token 的查询 + 每 draft 位查询)。
#   vLLM:vllm/v1/core/sched/scheduler.py:L261-L270
def num_lookahead_slots(method, num_speculative_tokens):
    if method == "dspark":
        return int(num_speculative_tokens)
    if method == "dflash":
        return int(num_speculative_tokens) + 1
    raise ValueError(f"unknown method: {method}")


# PAPER: §3.1 Eq.(4)(重构口径)—— 序列阶段:左到右按 p_k(x) ∝ softmax(U_k(x) +
#   B(x_{k−1},x)) 采样;prev 链 x_0=anchor、x_k=已采 draft token(串行依赖的全部
#   实现就是这一行 prev 赋值)。vLLM _sample_sequential 同构(dspark/speculator.py:
#   L100-L149);greedy 模式 = argmax(q 退化为 one-hot)
# PAPER: §3.1 Eq.(4) —— 序列阶段左到右采样(prev 链)
def sample_sequential(base_logits, anchor_token, markov_head, rng=None, greedy=False):
    base_logits = np.asarray(base_logits, dtype=float)
    gamma, vocab = base_logits.shape
    tokens, dists, steps = [], [], []
    prev = int(anchor_token)
    for k in range(gamma):
        bias = markov_head.bias(markov_head.embed([prev]))[0]
        logits_k = base_logits[k] + bias
        p_k = softmax_lastdim(logits_k)
        if greedy:
            x = int(np.argmax(logits_k))
        else:
            assert rng is not None, "采样模式需要 rng"
            x = int(rng.choice(vocab, p=p_k))
        steps.append({"k": k, "prev": prev, "bias": bias, "logits": logits_k,
                      "p": p_k, "token": x})
        tokens.append(x)
        dists.append(p_k)
        prev = x
    return tokens, np.asarray(dists), steps


# PAPER: §3.1 + §3.2.1 —— DSpark 完整 draft:并行骨干出 h/U(单次前向)→
#   序列 Markov 采样(Eq.4/5)→ 逐位置信度 c_k(Eq.7;prev 链与采样共用)。
#   返回 dict(含全部中间量,供示教轨迹:base_logits/hiddens/steps/confidences)
def dspark_draft(backbone, context, anchor_token, gamma, markov_head, conf_head,
                 rng=None, greedy=False):
    inputs = anchor_as_first_inputs(anchor_token, gamma)
    h, U = backbone.forward(inputs, context)
    tokens, p_d, steps = sample_sequential(U, anchor_token, markov_head, rng=rng, greedy=greedy)
    confidences = []
    prev = int(anchor_token)
    for k in range(gamma):
        confidences.append(float(conf_head.confidence(h[k], markov_head.embed([prev])[0])))
        prev = tokens[k]
    return {
        "draft_tokens": tokens,
        "draft_probs": p_d,
        "confidences": confidences,
        "base_logits": U,
        "hiddens": h,
        "steps": steps,
    }


# PAPER: §3.1 —— 并行位对前驱边缘化:p(x_k) = Σ_{x_{k−1}} p(x_{k−1})·p(x_k|x_{k−1})
#   (多模态碰撞的形式化:上下文容多个 mode 时,边缘分布是各 mode 的混合)
def marginal_over_predecessors(p_prev, cond_next):
    return np.asarray(p_prev, dtype=float) @ np.asarray(cond_next, dtype=float)


# PAPER: §3.1 —— 多模态碰撞 worked example:双 mode 上下文('of course'/'no
#   problem'),并行 drafter 每位独立取边缘 argmax → 'of problem'(跨 mode 拼接);
#   DSpark 序列阶段:位 1 采出 'of' 后,Markov 头抬 'course' 压 'problem' →
#   'of course'(论文开篇例子的可运行版)。base logits = 边缘分布的对数
#   (并行 drafter 每位学到的东西就是边缘);Markov 偏置取 rank-1 分解
#   B = W_1W_2(行 [of,no,anchor] 方向 × 列 [course,problem] 方向)
# PAPER: §3.1 —— 多模态碰撞 worked example
def multimodal_collision_demo(greedy=True):
    vocab = ["<anchor>", "of", "no", "course", "problem"]
    OF, NO, COURSE, PROBLEM = 1, 2, 3, 4
    p_x1 = np.array([0.0, 0.6, 0.4, 0.0, 0.0])  # 上下文略偏 'of'
    cond_x2 = np.array([
        [0.20, 0.20, 0.20, 0.20, 0.20],  # x_1='<anchor>' 占位(条件于 x_0=anchor 的位 1 不取 x_2)
        [0.05, 0.05, 0.00, 0.60, 0.30],  # x_1='of'  → 'course' 主导
        [0.01, 0.00, 0.02, 0.02, 0.95],  # x_1='no'  → 'problem' 极尖
        [0.20, 0.20, 0.20, 0.20, 0.20],  # 占位
        [0.20, 0.20, 0.20, 0.20, 0.20],
    ])
    marginal_x2 = marginal_over_predecessors(p_x1, cond_x2)  # 'problem' 0.56 > 'course' 0.368
    U = np.log(np.clip(np.vstack([p_x1, marginal_x2]), 1e-12, None))  # base logits = 边缘
    # rank-1 Markov 偏置:B(of,course)=+3、B(of,problem)=−3、B(no,·) 镜像、B(anchor,·)=0
    w1 = np.array([[0.0], [1.0], [-1.0], [0.0], [0.0]])
    w2 = np.array([[0.0, 0.0, 0.0, 3.0, -3.0]])
    markov = MarkovHead(w1, w2)
    parallel_block = [int(np.argmax(U[0])), int(np.argmax(U[1]))]  # 逐位独立 → 'of problem'
    dspark_tokens, dspark_dists, steps = sample_sequential(U, 0, markov, greedy=greedy)
    # 并行独立采样的连贯率(解析):p(x_1)⊗p(x_2) 落在两个真 mode 上的概率
    joint = np.outer(p_x1, marginal_x2)
    parallel_coherent_prob = float(joint[OF, COURSE] + joint[NO, PROBLEM])
    return {
        "vocab": vocab,
        "p_x1": p_x1,
        "cond_x2": cond_x2,
        "marginal_x2": marginal_x2,
        "base_logits": U,
        "markov_w1": w1,
        "markov_w2": w2,
        "parallel_block": parallel_block,
        "dspark_block": dspark_tokens,
        "dspark_probs": dspark_dists,
        "steps": steps,
        "parallel_coherent_prob": parallel_coherent_prob,
    }
