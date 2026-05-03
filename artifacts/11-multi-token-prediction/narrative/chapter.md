# 第10章：Multi-Token Prediction — 一次 decode，多个 token

> 标准自回归 decode 每步 1 个 token。MTP 每步预测 M 个——但只有第一个是被"验证"的，
> 其余是 draft。接受率决定实际加速比。**加速比 = 1 + Σ acceptance_rate_i。**

---

## 这章要做什么？

前九章建立了 vLLM 的完整推理管线：Attention 怎么算、KV Cache 怎么存、怎么调 schedule、怎么管理显存、怎么跨 GPU 并行。所有这些都有一个共同假设：**每个 decode step 生成 1 个 token。**

Multi-Token Prediction 打破了这个假设。它在最后几层 Transformer 之上加额外的"预测头"——每个头预测一个 future token。一次 forward pass 产出 M 个候选 token，接受率决定实际加速比。

MTP 不是 speculative decoding——它不需要 draft model。它是同一个模型在做预测，只是加了几个轻量级的额外头。

学完这章你能：
- 推导 MTP 的加速比公式：$S = 1 + \sum_{i=1}^{M-1} a_i$，其中 $a_i$ 是第 i 个 draft token 的接受率
- 解释为什么接受率随预测距离递减——从 ~85%（+1 token）到 ~58%（+3 tokens）

---

## 10.1 MTP 的数学

### Theory: 加速比公式

标准自回归 decode：每步生成 1 个 token，执行 1 次 transformer forward。生成 N 个 token 需要 N 次 forward。

MTP：每步生成 M 个候选 token（1 个被模型自身"验证" + M-1 个 draft）。**只有第一个 token 是确定正确的**（和标准 decode 一样）。其余 M-1 个是 draft——需要用标准 decode 重新验证。

如果第 i 个 draft token 的接受率为 $a_i$（与标准 decode 输出一致的概率），一次 MTP forward 的期望产出：

$$
\mathbb{E}[\mathrm{tokens\_per\_step}] = 1 + a_1 + a_2 + ... + a_{M-1}
$$

加速比 = 期望产出 / 1 = $S = 1 + \sum_{i=1}^{M-1} a_i$。

**为什么接受率递减？** 预测 token_{t+1} 时，模型看到了确切的 token_t 的 hidden state——条件信息完整。预测 token_{t+3} 时，模型不知道 token_{t+1} 和 token_{t+2} 是什么——它只能"猜测"。不确定性随预测距离累积，所以 $a_1 > a_2 > a_3$。

**实证数据（DeepSeek V3 技术报告）：**
- $a_1$ ≈ 85-90%（预测 1 步前——模型很确定）
- $a_2$ ≈ 70-75%（预测 2 步前——不确定性开始累积）
- $a_3$ ≈ 55-65%（预测 3 步前——边际收益递减）
- $a_4$+ ≈ <50%（几乎不值得额外计算）

这就是为什么大多数模型只用 1-3 个 MTP head——超过 3 个，边际加速比 <0.5×，不值得额外的计算和参数。

**为什么接受率递减——形式化推导：**

直觉：预测"下一个词"比"下下个词"容易，就像猜"他说了__"比猜"他说了__然后__"容易。多预测一步，不确定性相乘。

设真实下一个 token 分布为 $P(t_{k+1} | t_{1..k})$。MTP head $i$ 预测的分布为 $Q_i(t_{k+i} | t_{1..k})$——注意 $Q_i$ 不能条件于未知的 $t_{k+1}..t_{k+i-1}$。

标准 decode 的 top-1 准确率：$a_0 = \mathbb{E}[\mathbb{1}_{t_{k+1} = \arg\max P}] \approx 1$（模型几乎总是对下一个 token 正确）。

MTP head 1 的 top-1 准确率：

$$
a_1 = \mathbb{E}[\mathbb{1}_{t_{k+1} = \arg\max Q_1}] \approx 0.85
$$

MTP head 2 需要预测 $t_{k+2}$，但不能条件于 $t_{k+1}$ 的真实值（因为 $t_{k+1}$ 也是被预测的）。它只能基于**边缘分布**：

$$
a_2 = \mathbb{E}[\mathbb{1}_{t_{k+2} = \arg\max Q_2}] \approx a_1 \cdot c
$$

其中 $c < 1$ 是条件不确定性衰减因子（~0.85 for language modeling）。递归：$a_i \approx a_1 \cdot c^{i-1}$。

**数值 trace（以 DeepSeek V3 的 a₁=0.85, c=0.85 为例）：**

| Head i | 推导 | 接受率 | 边际加速比 |
|--------|------|--------|-----------|
| 1 | $a_1$ | 0.850 | +0.85 |
| 2 | $a_1 \cdot c$ | 0.722 | +0.72 |
| 3 | $a_1 \cdot c^2$ | 0.614 | +0.61 |
| 4 | $a_1 \cdot c^3$ | 0.522 | +0.52 |

总加速比 $S = 1 + 0.85 + 0.72 + 0.61 + 0.52 = 3.70$×（4 heads）。但每个 head 增加 ~0.1% 参数和 ~2% 计算。边际分析：head 4 贡献 0.52× 加速，对应 2% 计算增加——ROI = 26×。但 head 5 只贡献 ~0.44×，ROI 降到 ~22×。模型设计者通过 ROI 阈值决定 head 数量。

### Source Trail\n\nMTP head 的架构设计来自 DeepSeek V3（`vllm/model_executor/models/deepseek_v2.py`）和 `async_scheduler.py` 的 draft 验证逻辑。

打开 `vllm/v1/core/sched/async_scheduler.py:37`——`AsyncScheduler._update_request_with_output()`：

```python
# After model forward, verify draft tokens against oracle output
# MTP draft tokens that don't match are discarded
num_rejected = compute_rejected_draft_tokens(request, new_token_ids)
```

AsyncScheduler 处理 MTP 的 output 时，需要追踪"哪些 draft token 被接受了"——被拒绝的 draft token 要减少 `num_computed_tokens`。

---

（架构设计来自 DeepSeek V3，`vllm/model_executor/models/deepseek_v2.py` 和 `async_scheduler.py`）

## 10.2 MTP Head 的架构

### Source Trail

MTP head 的设计来自 DeepSeek V3 架构。核心思想：**共享 transformer trunk，只加轻量 head。**

```python
# DeepSeek V3 MTP head (simplified)
class MTPHead:
    def forward(self, hidden_states, prev_token_embedding):
        # 1. Combine current hidden state with previous token embedding
        combined = hidden_states + prev_token_embedding

        # 2. LayerNorm (independent per head)
        normalized = layer_norm(combined)

        # 3. Transformer block (SHARED across MTP heads — not independent)
        #    This is the key design — the transformer trunk processes ALL heads
        transformed = shared_transformer_block(normalized)

        # 4. LM Head projection (SHARED with main model)
        #    Same weight matrix as the final vocab projection
        logits = lm_head(transformed)

        return logits  # token_{t+k} prediction
```

**关键设计选择：**
1. **共享 LM Head**——MTP head 的输出用同一个 `lm_head` 权重投影到 vocab。不需要学习新的 vocab 空间映射。
2. **共享 Transformer Block**——多个 MTP head 串行通过同一个 transformer block（不是每个 head 独立权重）。这让 MTP 的参数开销极小（~0.1% of model params for 1 head）。

---

（对比参考 `vllm/v1/core/sched/async_scheduler.py` 的 draft 验证逻辑）

## 10.3 MTP vs Speculative Decoding

这是两个容易混淆的技术。它们都"一次预测多个 token"，但机制根本不同：

| | MTP | Speculative Decoding |
|---|---|---|
| **需要 draft model？** | 否——同一个模型 | 是——需要独立的 draft model |
| **如何生成 draft token？** | 额外的 MTP heads | 小模型独立 forward |
| **验证机制** | 同一 forward 中验证 | 大模型 forward 验证 |
| **参数开销** | ~0.1% | 额外的 draft model（~10% 参数） |
| **延迟影响** | 每个 step 增加 MTP head compute | 每个 step 增加 draft model forward |
| **加速比** | 1.5-2.5× | 2-3×（但需要第二个模型） |

MTP 是 speculative decoding 的"free lunch"版本——不需要第二个模型，代价是加速比略低。这就是为什么 DeepSeek V3/V4 选择 MTP 而不是 speculative decoding——对于 API 部署，多维护一个 draft model 的 ops 成本不值得额外的 0.5× 加速比。

---

（接受率模型参考 DeepSeek V3 技术报告和 `async_scheduler.py` 的 rejection 计数）

## 10.4 Acceptance Rate 的决定因素

### Theory: 为什么 a_i < 1？

一个完美的语言模型会给下一个 token 分配 100% 的概率。但这和真实语言的性质冲突——在任何位置通常有多个合理的下一个 token。例如：

```
Input: "The capital of France is"
Output: " Paris" (99% probability)  ← MTP 几乎确定
Alternative: " definitely" (~0.1%)  ← 语法上可行但不常见
```

对于 token_{t+1}，模型"几乎确定"正确 token。对于 token_{t+2}，模型不知道 token_{t+1} 的确切 identity——它必须基于"可能"的 token_{t+1} 做"可能"的 token_{t+2} 预测。不确定性被乘积了。

从信息论的角度：每个 token 携带约 log_2(vocab_size) ≈ 15 bits 的信息（vocab=32000）。Token_{t+1} 的 logit 分布由当前上下文唯一决定。Token_{t+2} 的 logit 分布以 token_{t+1} 的值为条件——但这个值对于 MTP 是不确定的。**MTP 预测的是边缘分布而非条件分布——边缘分布必然有更高的熵，因此 top-1 准确率更低。**

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `mtp_speedup_analysis()` | 原创分析——基于 DeepSeek 技术报告数据 | 量化加速比和边际收益递减 |
| `MTPHead` | DeepSeek V3 architecture | 简化版—未实现 shared transformer block |
| `verify_mtp_tokens()` | `async_scheduler.py` 中的 draft rejection 逻辑 | 首次不匹配→停止验证——逻辑一致 |

---

## 验证

```bash
cd artifacts/10-multi-token-prediction && python -m pytest tests/ -q
# 8/8 passed ✅
```

---

## 总结

- **MTP = 1 次 forward → M 个候选 token。** 加速比 $S = 1 + \sum a_i$——接受率决定了实际收益。
- **接受率随预测距离递减。** 从 ~85%（+1）到 ~58%（+3）——不确定性随条件信息的减少而累积。
- **MTP ≠ Speculative Decoding。** MTP 不需要 draft model——它用轻量 MTP heads（~0.1% 额外参数）。
- **1-3 个 head 是最优的。** 超过 3 个 head（$a_3 < 60\%$）——边际加速比 <0.5×，不值得额外计算。

---

**第一部分完结。** 10 章建立了 vLLM 推理引擎的完整基础：从单个 Attention 算子到跨 GPU 并行、从 KV Cache 的显存分配到 MTP 的多 token 预测。第二部分将进入进阶主题：DCP/PCP 上下文并行、KV Offload、Prefix Cache 池化。

---

← 第9章 | 第11章 →
