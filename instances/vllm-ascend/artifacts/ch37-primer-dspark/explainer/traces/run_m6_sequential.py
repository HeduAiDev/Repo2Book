#!/usr/bin/env python3
"""m6 — 序列 Markov 采样循环的教学玩具轨迹（host 纯算术，无 vLLM/CUDA）。

复现 speculator.py:L74-L113 _sample_sequential 的控制流：
  base_logits 只算一次（骨干一次前向的产物，此处直接给定）；
  for i in range(N): markov_embed(prev) -> markov_bias -> logits_i = U_i + bias -> 采样 -> prev=draft_i
采样用 argmax（对应源码 else 分支 draft_i = logits_i.argmax(dim=-1)，避开 gumbel 随机性，
玩具例可心算复现）。V/r/N 为便于心算的玩具值，非 checkpoint 真值（真值 r=256 见论文摘要）。
"""

# 玩具词表 V=4：A=0 B=1 C=2 D=3
VOCAB = ["A", "B", "C", "D"]
N = 3          # num_speculative_steps（玩具）
R = 2          # markov_rank（玩具；真值 r=256，来源 ai-infrastructure.net 摘要）

# 骨干一次前向出的基础 logits U_k（每行一个块内位置 k，每列一个词表 token）
U = [
    [1.0, 0.5, 0.0, 0.0],   # U_0
    [0.0, 1.5, 0.5, 0.0],   # U_1
    [1.5, 0.0, 0.0, 0.5],   # U_2
]

# markov_w1（V x r）：前驱 token -> r 维嵌入 e
W1 = {
    0: [1.0, 0.0],   # W1[A]
    1: [0.0, 1.0],   # W1[B]
    2: [1.0, 1.0],   # W1[C]
    3: [-1.0, 0.0],  # W1[D]
}

# markov_w2（V x r），充当伪 lm-head：bias_v = W2[v] . e
W2 = {
    0: [0.0, 0.0],   # W2[A]
    1: [2.0, 0.0],   # W2[B]
    2: [0.0, 2.0],   # W2[C]
    3: [1.0, 1.0],   # W2[D]
}

ANCHOR = 0   # 锚点(bonus) token = query offset 0 的 input id（玩具设为 A）


def markov_embed(tok):
    return W1[tok]


def markov_bias(e):
    return [sum(W2[v][j] * e[j] for j in range(R)) for v in range(len(VOCAB))]


def argmax(xs):
    best, bi = xs[0], 0
    for i, x in enumerate(xs):
        if x > best:
            best, bi = x, i
    return bi


prev = ANCHOR
print(f"anchor(prev0) = {VOCAB[prev]}({prev})")
draft = []
for i in range(N):
    e = markov_embed(prev)
    bias = markov_bias(e)
    logits = [U[i][v] + bias[v] for v in range(len(VOCAB))]
    base_argmax = argmax(U[i])
    d = argmax(logits)
    draft.append(d)
    print(f"i={i}: prev={VOCAB[prev]} e={e} bias={bias} "
          f"U_{i}={U[i]} logits={logits} "
          f"base_argmax={VOCAB[base_argmax]} draft_{i}={VOCAB[d]}({d}) "
          f"flipped={base_argmax != d}")
    prev = d

print("draft_tokens =", [VOCAB[d] for d in draft])
