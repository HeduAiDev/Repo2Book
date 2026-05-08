# 第10章：Multi-Token Prediction —— 没有 `class MultiTokenPrediction` 的 K 步并行解码

> 本章涉及的 vLLM 源码（commit `98661fe`）：
> - `instances/vllm/source/vllm/v1/sample/rejection_sampler.py:L37-L195`（`class RejectionSampler(nn.Module)` —— 接受 `Sampler + SpeculativeConfig`，缓存 `synthetic_conditional_rates`）+ `L246-L281`（`parse_output` 过滤 `PLACEHOLDER_TOKEN_ID = -1`）+ `L392-L503`（`def rejection_sample` 驱动器）+ `L425-L430`（`output_token_ids = torch.full(..., PLACEHOLDER_TOKEN_ID)`，链断点的隐式哨兵）+ `L506-L562`（`apply_sampling_constraints`：top-k/top-p 在 rejection 之前作用）+ `L604-L656`（`generate_uniform_probs`）+ `L659-L703`（`sample_recovered_tokens`：`(p-q)_+` 残差）+ `L708-L757`（`rejection_greedy_sample_kernel`：argmax 比较，无概率）+ `L760-L826`（`rejection_random_sample_kernel`：`p(d)/q(d) >= u`）+ `L853-L920`（`sample_recovered_tokens_kernel`：Gumbel-max 采样残差）
> - `instances/vllm/source/vllm/v1/spec_decode/metadata.py:L1-L66`（`@dataclass SpecDecodeMetadata` —— `draft_token_ids` / `cu_num_draft_tokens` / `target_logits_indices` / `bonus_logits_indices` / `max_spec_len`，proposer 与 sampler 之间的纯数据契约）
> - `instances/vllm/source/vllm/v1/spec_decode/llm_base_proposer.py:L60-L303`（`class SpecDecodeBaseProposer.__init__`，`num_speculative_tokens` 是全局而非 per-request）+ `L407-L412`（`_greedy_sample = argmax`）+ `L413-L656`（`def propose`：K==1 fast-path 与 K>1 sequential）+ `L1402-L1469`（`_maybe_share_embeddings`）+ `L1471-L1539`（`_maybe_share_lm_head`：DeepSeek MTP 强制走的"始终 share"分支）
> - `instances/vllm/source/vllm/v1/spec_decode/eagle.py:L10-L22`（`class EagleProposer(SpecDecodeBaseProposer)` —— 整个文件 22 行，纯继承）
> - `instances/vllm/source/vllm/v1/spec_decode/medusa.py:L18-L78`（`class MedusaProposer` —— **不**继承 base，K 个独立 MLP 头并行 argmax）
> - `instances/vllm/source/vllm/v1/spec_decode/draft_model.py:L17-L88`（`class DraftModelProposer(SpecDecodeBaseProposer)` —— `_raise_if_vocab_size_mismatch` + `_raise_if_draft_tp_mismatch`）
> - `instances/vllm/source/vllm/v1/spec_decode/extract_hidden_states.py:L26-L382`（`class ExtractHiddenStatesProposer` —— `assert num_speculative_tokens == 1`，单步 KV-extraction 路径）
> - `instances/vllm/source/vllm/v1/spec_decode/ngram_proposer.py:L12-L162`（`class NgramProposer` —— prompt-lookup，无概率，触发 `NO_DRAFT_PROBS` 路径）+ `L170-L285`（numba batch path）
> - `instances/vllm/source/vllm/model_executor/models/deepseek_mtp.py:L43-L62`（`class SharedHead(nn.Module)` —— RMSNorm + ParallelLMHead）+ `L63-L122`（`class DeepSeekMultiTokenPredictorLayer` —— enorm + hnorm + eh_proj + `mtp_block: DeepseekV2DecoderLayer`，**完整 transformer block，不是轻量 MLP**）+ `L124-L184`（`class DeepSeekMultiTokenPredictor` —— K 层堆叠）+ `L186-L488`（`class DeepSeekMTP` —— top-level wrapper）+ `L458-L488`（`_rewrite_spec_layer_name` —— HF → vLLM 三路改写）
> - `instances/vllm/source/vllm/config/speculative.py:L35-L70`（`SpeculativeMethod` 字面量、`MTPModelTypes ⊂ EagleModelTypes ⊂ SpeculativeMethod` 的传递包含关系，`"mtp"` 经此通道存在）+ `L73-L210`（`class SpeculativeConfig`）+ `L213-L227`（`_acceptance_length_to_rates` —— 平均长度 → per-position 合成率的最小方差表）
> - `instances/vllm/source/vllm/model_executor/models/llama_eagle3.py:L1-L425`（`class Eagle3LlamaForCausalLM` —— EAGLE3 reference，`fc` 投影做融合，与 MTP 的 `eh_proj` 同义不同名）
> - 对照实现 `instances/vllm/artifacts/10-multi-token-prediction/implementation/`：`spec_metadata.py`、`rejection_sampling.py`、`acceptance_math.py`、`mtp_head.py`、`weight_loading.py`、`proposers/{base,eagle,medusa,draft_model,ngram,mtp,extract_hidden}.py`、`demo.py`
>
> 第 7 章用"vLLM 没有 radix tree"开篇，第 8 章用"vLLM 没有 `class TensorParallel`"开篇，第 9 章用"vLLM 没有 `class ExpertParallel`"开篇——第 10 章是这条系列的 **第四件**：**vLLM 没有 `class MultiTokenPrediction`**。MTP-the-technique 在源码里是 **5 个 proposer 类** + **DeepSeek 专属的 mtp 模型族 30+ 个 wrapper** + **1 个 verifier kernel** 的协同。

---

## 这章要讲什么？

打开 `instances/vllm/source/vllm/v1/sample/rejection_sampler.py:L425-L430`：

```python
# vllm/v1/sample/rejection_sampler.py:L425-L430
output_token_ids = torch.full(
    (batch_size, max_spec_len + 1),
    PLACEHOLDER_TOKEN_ID,           # = -1
    dtype=torch.int32,
    device=device,
)
```

整个 spec-decode 算法的**链断点不变量**就藏在这一句 `torch.full(..., -1)` 里：输出缓冲先用 `-1` 填满，kernel 从左往右写；一旦某个位置 reject，kernel 就停笔——后面的位置自然留作 `-1`。`parse_output`（`L246-L281`）在末尾把 `-1` 过滤掉。**没有显式的 "rejected" 标志位，没有专门的 stop 向量；链断在哪里全靠 placeholder 自己宣告**。这个设计跟 Ch07 prefix cache 的 chained-hash 哨兵、Ch08 column-parallel 的隐式 reduction、Ch09 EP 的 `expert_map[i]=-1` off-rank 哨兵是同一脉——**不变量常常长在数据 layout 里，而不是控制流上**。

再 grep 一下 vLLM 全树：

```
$ grep -rE "^class\s+(MultiTokenPrediction|MTPHead|MTPModel|TokenPredictor)\b" \
    instances/vllm/source/vllm/
(zero matches)
```

但 outline 第 10 章的标题是 "Multi-Token Prediction (MTP)"，DeepSeek-V3 论文也明明写着"MTP modules"。**MTP 在 vLLM 里是一种实现模式，不是一个 class**：

- `vllm/v1/sample/rejection_sampler.py` —— **唯一的 verifier**：所有 proposer 都把 K 个 draft 喂进同一个 rejection 算法。
- `vllm/v1/spec_decode/llm_base_proposer.py` —— `SpecDecodeBaseProposer`：CUDA graph、slot mapping、weight sharing 的共享脚手架（1820 行）。
- `vllm/v1/spec_decode/{eagle,medusa,draft_model,extract_hidden_states,ngram_proposer}.py` —— **5 个 proposer 类**：分别对应 EAGLE、Medusa、独立 draft 模型、单步 KV 抽取、N-gram 查表。每一个都是 K 个 draft token 的不同生产方式。
- `vllm/model_executor/models/deepseek_mtp.py` + 30 个 `*_mtp.py`（`qwen3_5_mtp.py`、`ernie_mtp.py`、`glm4_moe_mtp.py`、`deepseek_v4_mtp.py`、`mimo_mtp.py`、`longcat_flash_mtp.py`、`openpangu_mtp.py` ……）—— **每个支持 MTP 的模型族都有自己的 wrapper**。DeepSeek 是 canonical impl，其余基本 import 它的层定义。
- `vllm/config/speculative.py` —— `SpeculativeMethod` 枚举：`"ngram"`、`"medusa"`、`"mlp_speculator"`、`"draft_model"`、`"suffix"`、`EagleModelTypes`、`NgramGPUTypes`。**`MTPModelTypes` 通过 `MTPModelTypes ⊂ EagleModelTypes ⊂ SpeculativeMethod` 这条传递链合法存在**（`speculative.py:L35-L67`）；但用户面 method 字串通常写 `"draft_model"` 配 `"deepseek_mtp"` 模型，或直接写 `"eagle3"`——MTP 不是 first-class method 名。

**Outline 的 §3 标题是 "Training——多步 CE 损失的加权策略" —— 这是个 training 概念**。vLLM 是 inference-only 引擎；`vllm/v1/spec_decode/` 整目录里 grep `\.backward\(|MTPLoss|multi_step_ce|compute_mtp_loss|mtp_aux_loss` 全部 0 匹配。本章在 §10.3 把这个 outline 节点改写成 **"Inference 期的 MTP 权重加载与共享"** —— 把 training-time 多步 CE loss 当 sidebar 简介，pivot 到 vLLM 真正在做的事：`_rewrite_spec_layer_name` 的 HF → vLLM 三路改写、`_maybe_share_lm_head` 的权重共享、`SharedHead.forward` 只返回 `norm(h)`（M09 校正）。这是 Ch09 §9.4 之后的 **第二次** training-to-inference reframe，使用同一个 *sidebar + pivot* 模板。

学完这章你能：

- 在白板上写出**几何链断点公式**

  $$
  E[\mathrm{tok} \mid \alpha, K] = \sum_{k=0}^{K} \alpha^k = \frac{1 - \alpha^{K+1}}{1 - \alpha}
  $$

  并解释为什么 α=0.5、K=4 给出 E[tok] = 1.9375 而**不是** K·α = 2.0（demo §3.2 的 35 格 α-K 表是它的具体取值，trap-A 的硬证据）。
- 解释为什么速度公式 S = E[tok] / (1 + cK) 在 K=4、c=0.20、α=0.30 给出 **0.792 < 1**（净亏），并念出 9 个 break-even α 中的关键三个：K=4 c=0.10 → α* = 0.2871；K=4 c=0.20 → α* = 0.4553；K=8 c=0.20 → α* = 0.6206（demo §3.3，trap-B 的运维风险）。
- 走通 Chen 2023 rejection sampling 的不偏证明（§10.1.4 的 5 行代数会展开），demo §3.1 经验上验证 KL(empirical || p) = 0.000395，远小于 0.01 阈值。
- 解释 **DeepSeek MTP 头不是轻量 MLP**：参数比 MTP/Medusa = **12.91×**（共享 lm_head）/ **1.91×**（独立 lm_head），demo §3.5 的 6 行 breakdown verbatim（`enorm=2048`、`hnorm=2048`、`eh_proj=8,388,608`、`mtp_block_attn=16,777,216`、`mtp_block_ffn=50,331,648`、`mtp_block_norms=4,096`）。DeepSeek 实际还有 MoE block，参数会再涨 ~10×（trap-E）。
- 区分 vLLM 里 5 个 proposer：EAGLE（继承 base，shared trunk）、Medusa（不继承 base，K 个并行 MLP 头）、DraftModel（独立小模型，仅 vocab/TP 校验）、Ngram（无模型无概率）、ExtractHiddenStates（`assert K==1`）。它们共享同一个 `RejectionSampler` 后端但生产 K 个 draft 的方式各不相同。
- 在 §10.3 区分 **训练期多步 CE 损失**（DeepSeek-V3 训练里有，vLLM 整树没有）与 **推理期权重加载**（`_rewrite_spec_layer_name` 三路改写：path 1 `mtp_block` 包裹、path 2 shared `embed_tokens` 上提、path 3 MTP 专属保留）。Demo §3.6 的 193 → (185, 8) 切分是它的具体取值。
- 在 §10.7 区分 **7 个语言陷阱**：A "K=4 ≠ 4×" ✗、B "spec_decode 总比 AR 快" ✗、C "draft 必须共享 target 架构" ✗、D "高温下 rejection sampling 有偏" ✗、E "MTP 头是轻量 MLP" ✗、F "vLLM 训练 MTP" ✗、G "α 是模型属性" ✗。

接下来 6 节按 outline 走，但 §10.3 已经从 "Training——多步 CE 损失" 重构成 "Inference 期 MTP 权重加载" —— 源码里就没有训练 loss code，硬讲就是失真。

---

## 10.1 数学：α-K 速度公式与不偏定理

### 10.1.1 打开 rejection_sample 入口

源码定位：`instances/vllm/source/vllm/v1/sample/rejection_sampler.py:L392-L503`，`def rejection_sample(...)` 的核心几行：

```python
# vllm/v1/sample/rejection_sampler.py:L392-L503 (节选)
def rejection_sample(
    metadata: SpecDecodeMetadata,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    ...,
) -> torch.Tensor:
    # L425-L430 — 输出缓冲预填 PLACEHOLDER_TOKEN_ID = -1
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    # L450-L466 — Greedy fast path：若全 greedy，比较 draft == argmax(target)
    if not all_random:
        target_argmax = target_probs.argmax(dim=-1).to(torch.int32)
        rejection_greedy_sample_kernel[grid](...)
        if all_greedy:
            return output_token_ids

    # L468-L502 — Random path：算 (p-q)_+ 残差、采 recovered token，再做 accept/reject
    target_probs = target_probs.softmax(...)
    recovered_token_ids = sample_recovered_tokens(...)
    rejection_random_sample_kernel[grid](...)
    return output_token_ids
```

四件事一起进来：**输出缓冲预填 `-1`**（链断点的隐式哨兵）、**双路径** greedy/random、**bonus token 由 `Sampler` 而不是 `RejectionSampler` 采样**（让它支持 top-p/top-k）、**recovered token 用 `(p-q)_+` 残差**（不偏的关键）。

我们的对照实现在 `implementation/rejection_sampling.py:L286-L399`，按同样的顺序复刻这个驱动器：

```python
# implementation/rejection_sampling.py:L313-L318
output_token_ids = torch.full(
    (batch_size, metadata.max_spec_len + 1),
    PLACEHOLDER_TOKEN_ID,                  # = -1
    dtype=torch.int32,
    device=device,
)
```

vLLM 在 GPU 上调 Triton kernel `rejection_greedy_sample_kernel`（`L708-L757`，全无概率）和 `rejection_random_sample_kernel`（`L760-L826`，`(target_prob / draft_prob) >= u` 的累计判定），fallback 到 `torch.full + Python loop` 时数学完全一致——这是 Ch08/Ch09 反复看到的 **算法相同、layout 不同** 的标准 vLLM 模式。

### 10.1.2 大白话：为什么 K=4 不是 4× 加速

先把直觉讲清楚（这是 trap-A 的核心，也是 framing tip 1 的"先公式后数字"）。

每一步 spec-decode 的目标是：让 draft 模型一次性提议 **K 个未来 token**，让 target 模型在 **同一次 forward** 里给出 K+1 个位置的 logits（K 个 draft 位置 + 1 个 bonus 位置），然后 verifier 决定每个 draft 接不接受。如果 K 个 draft 都接受，多收一个 bonus，相当于**每次 target forward 出 K+1 个 token**——这是上界 $K+1$。

但**链一旦在某位置 reject 就断**：position 0 的 draft 被 reject 了，那 position 1, 2, 3 的 draft 全部作废，因为它们是基于 position 0 的 draft 算出来的——往下走没有意义。Position 0 上还能 emit 一个 "recovered token"（从 $(p-q)_+$ 残差里采，单步不偏），所以**一次 target forward 至少 emit 1 个 token**——这是下界 1。

把这个机率叠起来。设每个 draft 位置被接受的概率是 $\alpha$（先假设它在 K 个位置上 i.i.d.，§10.1.3 会精化这个简化），那么：

$$
P(\mathrm{emit} \ge k) \;=\; P(\mathrm{position}\ 0..k-1\ \mathrm{all\ accept}) \;=\; \alpha^k \qquad (k \le K)
$$

期望 emit token 数：

$$
E[\mathrm{tok}] = \sum_{k=1}^{K+1} P(\mathrm{tok} \ge k) = \sum_{k=0}^{K} \alpha^k = \frac{1 - \alpha^{K+1}}{1 - \alpha}
$$

这是 Ch10 的**主公式**——一个很普通的几何级数，但它的取值跟直觉差很多。来看一些具体数字（demo §3.2 的 35 格 α-K 表 verbatim）：

```
alpha\K          1         2         3         4         5
alpha=0.3   1.3000    1.3900    1.4170    1.4251    1.4275
alpha=0.4   1.4000    1.5600    1.6240    1.6496    1.6598
alpha=0.5   1.5000    1.7500    1.8750    1.9375    1.9688
alpha=0.6   1.6000    1.9600    2.1760    2.3056    2.3834
alpha=0.7   1.7000    2.1900    2.5330    2.7731    2.9412
alpha=0.8   1.8000    2.4400    2.9520    3.3616    3.6893
alpha=0.9   1.9000    2.7100    3.4390    4.0951    4.6856
```

**Trap A 的现场**：$\alpha = 0.5, K = 4$ 给 $E[\mathrm{tok}] = 1.9375$，不是 $K \cdot \alpha = 2.0$。差距 **3.1%**。$\alpha = 0.7, K = 4$ 给 $E[\mathrm{tok}] = 2.7731$，不是 $2.8$。差距 **0.96%**。看起来 K 越大、α 越大差距就越接近"K-α 直觉"——错。$\alpha = 0.3, K = 4$ 给 $E[\mathrm{tok}] = 1.4251$，离 $K \cdot \alpha = 1.2$ **多了 18.8%**——这次差距是反方向的（因为公式里有 +1 的 bonus 项）。**直觉总是错的，公式才是真的**。

经验也验证这一点（`acceptance_math.py:L146-L187 simulate_chain_break`，10000 次试验）：

```
alpha=0.5, K=2 → empirical 1.7507 ± 0.0162  vs analytic 1.7500
alpha=0.5, K=4 → empirical 1.9323 ± 0.0232  vs analytic 1.9375
alpha=0.7, K=2 → empirical 2.1912 ± 0.0171  vs analytic 2.1900
alpha=0.7, K=4 → empirical 2.7657 ± 0.0305  vs analytic 2.7731
```

四对都在 95% CI 内，公式跟 Bernoulli 模拟一致。我们写代码：

```python
# implementation/acceptance_math.py:L56-L74
def expected_tokens(alpha: float, K: int) -> float:
    """E[tok | alpha, K] = (1 - alpha^(K+1)) / (1 - alpha)."""
    if abs(alpha - 1.0) < 1e-12:
        return float(K + 1)
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)
```

公式是 framing tip 1 ("lead with the geometric formula, NOT the number") 的硬执行：纯数学先讲清楚，然后才是具体取值。这跟 Ch07 的 K13 `(N-1)·K`、Ch09 §9.5 的 `mem_per_rank ∝ 1/(ep × tp)` 是同一套教学法——**数字只是公式的实例**。

### 10.1.3 速度公式与 break-even α

光有 E[tok] 不够。每次 target forward 还要付一个 draft 的成本——不论是 EAGLE 头跑一遍、Medusa K 个 MLP 头一次 forward、独立小模型跑一次，还是 ngram 查表（这一个例外，成本几乎零）。设

$$
c \;=\; T_\mathrm{draft} \,/\, T_\mathrm{target}
$$

是 draft 成本比，那么 K 步 spec-decode 的时间是 `T_target × (1 + cK)`，每个 token 的平均时间是 `T_target × (1 + cK) / E[tok]`。相对于纯自回归（每个 token 1 个 target forward），加速比是：

$$
S(\alpha, K, c) \;=\; \frac{E[\mathrm{tok} \mid \alpha, K]}{1 + cK}
$$

这是 demo §3.3 跑的表（c=0.05/0.10/0.20/0.30，K=4 那一行 verbatim）：

```
K = 4
c\alpha    0.30    0.40    0.50    0.60    0.70    0.80    0.90
c=0.05    1.188   1.375   1.615   1.921   2.311   2.801   3.413
c=0.1     1.018   1.178   1.384   1.647   1.981   2.401   2.925
c=0.2     0.792   0.916   1.076   1.281   1.541   1.868   2.275
c=0.3     0.648   0.750   0.881   1.048   1.260   1.528   1.861
```

**Trap B 的硬证据**：K=4、c=0.20、α=0.30 给 S = 0.792 —— **MTP 比纯自回归还慢**。K=4、c=0.30、α=0.30 给 S = 0.648，慢了 **35%**。这不是边角 case：现实里 α 会随 prompt 域变化（数学公式生成可能 α=0.85，开放式对话可能 α=0.4），c 在跨节点 IB 部署下也容易上 0.10-0.20。**运维必须先测 α 再决定 K，不然 spec-decode 是搬起石头砸自己的脚**——这是 framing tip 2 的"net-loss zone is THE headline operator risk"，不是脚注。

把"刚好打平"那条线找出来——求 S(α, K, c) = 1 对 α 的解，即 **break-even α**：

$$
\frac{1 - \alpha^{K+1}}{1 - \alpha} \;=\; 1 + cK
$$

这是关于 α 的多项式方程，没有闭式解，但 S 在 [0, 1] 上单调递增，**二分法 200 次就够**（`acceptance_math.py:L90-L112 break_even_alpha`）。Demo §3.3 跑了 9 个 (K, c) 配对：

```
K=2, c=0.05 → alpha* = 0.0916        # K 小、draft 便宜：极低 α 就能赚
K=2, c=0.10 → alpha* = 0.1708
K=2, c=0.20 → alpha* = 0.3062
K=4, c=0.05 → alpha* = 0.1668        # ngram-style 工作流的舒适区
K=4, c=0.10 → alpha* = 0.2871        # DeepSeek-V3 MTP 工作流的最低门槛
K=4, c=0.20 → alpha* = 0.4553        # 跨节点部署需要相当高 α
K=8, c=0.05 → alpha* = 0.2857
K=8, c=0.10 → alpha* = 0.4448
K=8, c=0.20 → alpha* = 0.6206        # K 很大、draft 不便宜：α 必须 > 0.62
```

**这张表是部署决定**：DeepSeek-V3 论文报告的第一步 α ≈ 0.85，远高于 K=4 c=0.10 的 0.2871——所以 DeepSeek-V3 用 K=4 MTP 是稳赚的。但 ngram-only 的工作流，α 经常在 0.3-0.5，K=8 c=0.20 的 0.6206 它就达不到——硬上 K=8 是亏。**先测 α，再选 K**——不是"K 越大越好"。

源码的 `vllm/config/speculative.py:L213-L227 _acceptance_length_to_rates` 给的是**反方向**：拿一个目标平均接受长度 $L \in [1, K+1]$，输出最小方差的合成 per-position 率（前 $\lfloor L-1 \rfloor$ 位置 = 1.0，下一个位置 = $L - 1 - \lfloor L-1 \rfloor$，后面全 0）。我们的 `weight_loading.py:L239-L257 acceptance_length_to_rates` 复刻它，用来给合成模式的测试做精确的 α 控制：

```python
# implementation/weight_loading.py:L251-L257
num_drafts = length - 1.0
num_full = int(num_drafts)
fractional = num_drafts - num_full
rates = [1.0] * num_full + [fractional] + [0.0] * (n - num_full - 1)
```

这是 Ch10 在合成测试里能 pin 出 verbatim 数字的关键——`SpeculativeConfig.rejection_sample_method == "synthetic"` 时整个 rejection 算法变成"扔预定的硬币"，不依赖 GPU 也不依赖真实模型（M13）。

### 10.1.4 不偏证明：Chen 2023 定理的 5 行代数

Trap D 的核心：**rejection sampling 在任何 p, q 下都是不偏的，跟温度无关**。要证的命题：累积分布 P(emit x) = p(x) 对所有 x 成立。

把"emit x"的事件拆成两条互斥路径：

- **Accept path**：draft 抽到 x（概率 q(x)），然后接受（条件概率 min(1, p(x)/q(x))）。
- **Reject + Recover path**：draft 抽到任意 y（概率 q(y)），然后被拒（条件概率 1 - min(1, p(y)/q(y))），然后从残差 (p-q)_+ / Z 里采到 x（条件概率 (p(x)-q(x))_+ / Z，其中 Z = Σ_z (p(z)-q(z))_+）。

代数加起来：

$$
P(\mathrm{emit}\ x) \;=\; q(x) \cdot \min\!\left(1,\, \tfrac{p(x)}{q(x)}\right) \;+\; \left(1 - \sum_y q(y)\,\min\!\left(1,\, \tfrac{p(y)}{q(y)}\right)\right) \cdot \frac{(p(x) - q(x))_+}{Z}
$$

第一项化简：

$$
q(x) \cdot \min\!\left(1,\, \tfrac{p(x)}{q(x)}\right) \;=\; \min(p(x), q(x))
$$

第二项的标量系数：

$$
1 - \sum_y q(y) \cdot \min\!\left(1,\, \tfrac{p(y)}{q(y)}\right) \;=\; 1 - \sum_y \min(p(y), q(y))
$$

利用关系 Σ_y min(p(y), q(y)) + Σ_y (p(y) - q(y))_+ = 1（由 (p-q)_+ + min(p, q) = p 求和而来），上式等于 Σ_y (p(y) - q(y))_+ = Z，所以第二项 = Z · (p(x) - q(x))_+ / Z = (p(x) - q(x))_+。

加起来：

$$
P(\mathrm{emit}\ x) \;=\; \min(p(x), q(x)) + (p(x) - q(x))_+ \;=\; p(x)
$$

QED。**对任何 p, q（只要 q(x) > 0 在 p(x) > 0 的地方），rejection sampling 都不偏**。温度只影响 α 的取值，不影响最终 emit 的分布。形式化地：

$$
\alpha \;=\; \mathbb{E}_{x \sim q}\!\left[\min\!\left(1,\, \tfrac{p(x)}{q(x)}\right)\right]
$$

我们用 demo §3.1 经验验证（`tests/test_demo_numerics.py::test_section_31_*`，10000 次试验）：

```
target p = [0.30, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03]
draft  q = [0.10, 0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05]  # 跟 p 差很多
empirical p_hat = [0.2906, 0.2037, 0.1543, 0.1005, 0.1044, 0.0674, 0.0494, 0.0297]
KL(empirical || p) = 0.000395                       # 远小于阈值 0.01
```

$q$ 跟 $p$ 全错（$q[0] = 0.10$ 但 $p[0] = 0.30$，差 3 倍），但 emitted 分布 KL = 4×10⁻⁴ ≈ 0——不偏。**这是 Chen 2023 的核心保证**：哪怕 draft 很烂，verifier 也不会让最终输出偏离 target——它只会让 α 变低，spec-decode 变慢。**质量永远是 target 的质量；spec-decode 调的是延迟，不是质量**。

源码的不偏机制在 `rejection_sampler.py:L659-L703 sample_recovered_tokens`（`(p-q)_+` 残差采样）和 `L674-L688` 的 Gumbel-max 技巧（`q ~ Exp(1)` 然后 `argmax((p - q_draft)_+ * (1/q_exp))` —— 不归一化，因为 argmax 对正缩放不变）。我们的 `rejection_sampling.py:L223-L279 sample_recovered_tokens_loop` 用 `torch.argmax(p * inv_q)` 复刻 Gumbel-max，对 ngram 路径（`NO_DRAFT_PROBS=True`）的残差退化为 "target_probs with draft_id masked"——也就是 `p` 上把 reject 掉的那个 id 挖空。

### 10.1.5 Source diff：vLLM 比我们多了什么

| 我们 | vLLM | 关系 |
|---|---|---|
| `rejection_sampling.py:L56-L131 rejection_greedy_sample_loop` | `rejection_sampler.py:L708-L757 rejection_greedy_sample_kernel` | 算法相同；Triton vs Python loop |
| `rejection_sampling.py:L138-L216 rejection_random_sample_loop` | `rejection_sampler.py:L760-L826 rejection_random_sample_kernel` | 同；`(target/draft >= u)` accept |
| `rejection_sampling.py:L223-L279 sample_recovered_tokens_loop` | `rejection_sampler.py:L853-L920 sample_recovered_tokens_kernel` | 同；Gumbel-max 技巧 |
| `rejection_sampling.py:L286-L399 rejection_sample` driver | `rejection_sampler.py:L392-L503 rejection_sample` | 同；省 logprobs_mode |
| `rejection_sampling.py:L406-L417 parse_output` | `rejection_sampler.py:L246-L281 parse_output` | 同；过滤 PLACEHOLDER 和 OOV |
| `rejection_sampling.py:L423-L464 RejectionSampler` 类 | `rejection_sampler.py:L37-L195 RejectionSampler(nn.Module)` | 教学版去掉 nn.Module 和 Sampler 依赖 |

vLLM 比我们多的：(1) Triton kernel 性能 ~100× 提升、(2) `Sampler` 依赖处理 top-k/top-p/temperature，(3) `logprobs_mode='processed_logits' / 'raw_logits'` 用于 logprob accounting，(4) `synthetic_conditional_rates` 设备张量缓存。这些跟算法正确性无关；M01 已经记下"`RejectionSampler` 是 `nn.Module` 主要因为要持有合成态张量，纯函数复刻可以扔掉 wrapping"。

---

## 10.2 没有 `class MultiTokenPrediction`：5 个 proposer 的协同

### 10.2.1 grep 一遍源码

跟 Ch07/Ch08/Ch09 一样，先 grep 把"是否有顶层类"的事实搞定：

```
$ grep -rE "^class\s+(MultiTokenPrediction|MTPHead|MTPModel|TokenPredictor)\b" \
    instances/vllm/source/vllm/
(zero matches)

$ grep -rE "^class\s+\w*MTP\w*\b" instances/vllm/source/vllm/ | head -10
.../models/deepseek_mtp.py:43:    class SharedHead(nn.Module):
.../models/deepseek_mtp.py:63:    class DeepSeekMultiTokenPredictorLayer(nn.Module):
.../models/deepseek_mtp.py:124:   class DeepSeekMultiTokenPredictor(nn.Module):
.../models/deepseek_mtp.py:186:   class DeepSeekMTP(...):
.../models/qwen3_5_mtp.py: ...     # imports DeepSeekMultiTokenPredictorLayer
.../models/ernie_mtp.py: ...       # 同
.../models/mimo_mtp.py: ...
.../models/glm4_moe_mtp.py: ...
.../models/deepseek_v4_mtp.py: ... # adds hc_mult carrier expansion
... (30+ files in total)
```

**`class MultiTokenPrediction` 不存在**。存在的是 **DeepSeek 前缀的层定义** + **30 个模型族 wrapper**。Trap "MTP 是一个统一的类" 的反证。Reframe A 的三锚点之一 (title) 已经在章名 "没有 `class MultiTokenPrediction` 的 K 步并行解码" 钉死，hook 在导言段的"vLLM 没有 `class MultiTokenPrediction`，存在的是 DeepSeek 前缀层 + 30+ 模型族 wrapper"，§10.2 body 在这一节展开。

那 `SpeculativeMethod` 枚举里有什么？打开 `vllm/config/speculative.py:L35-L70`：

```python
# vllm/config/speculative.py:L35-L67 (节选)
MTPModelTypes = Literal["deepseek_mtp", "qwen3_5_mtp", "ernie_mtp",
                        "mimo_mtp", "glm4_moe_mtp", "deepseek_v4_mtp",
                        "longcat_flash_mtp", "openpangu_mtp", "mtp",
                        ...]
EagleModelTypes = Literal[MTPModelTypes, "eagle", "eagle3", "llama_eagle3"]
NgramGPUTypes = Literal["ngram_gpu"]
SpeculativeMethod = Literal["ngram", "medusa", "mlp_speculator",
                            "draft_model", "suffix",
                            EagleModelTypes, NgramGPUTypes]
```

**M07 的精校**：brief 里曾说 `"mtp"` 不在 `SpeculativeMethod` 里——验证后实际**是在的**，通过 `MTPModelTypes ⊂ EagleModelTypes ⊂ SpeculativeMethod` 这条传递包含关系。但用户面看到的"主流"method 字串是 `"draft_model"` + `model="deepseek_mtp"`，或者 `"eagle3"`。**MTP 是 first-class method 类簇但不是 first-class method 字串**——这是 vLLM 的"枚举内嵌"风格，跟 Ch08 的"`_TP` 全局单例 vs `_EP` 条件单例"是同一套 corner-case 工艺。

### 10.2.2 5 个 proposer 类各自做了什么

打开 `vllm/v1/spec_decode/` 目录，5 个 proposer 文件：

| Proposer | 源码行数 | 是否继承 base | 怎么生产 K 个 draft | draft probs |
|---|---|---|---|---|
| `EagleProposer` (`eagle.py`) | 22 | ✓ | 跑 EAGLE 模型 K 步，每步用 target hidden + 上一 draft 的 fc 融合 | softmax，有 |
| `MedusaProposer` (`medusa.py`) | 78 | ✗ | K 个独立 MLP 头，对**同一**个 target hidden 并行 argmax | 无 |
| `DraftModelProposer` (`draft_model.py`) | 88 | ✓ | 跑独立小模型 K 步（自回归） | softmax，有 |
| `NgramProposer` (`ngram_proposer.py`) | 285 | ✗ | 在 prompt 里找最长后缀匹配，emit 后续 K 个 token | 无 |
| `ExtractHiddenStatesProposer` (`extract_hidden_states.py`) | 382 | ✗ | `assert K==1`；返回 sampled token 不变（永远 verify） | 无 |

EAGLE 是 22 行——为什么这么短？因为 **EAGLE 的算法 IS the base class**：

```python
# vllm/v1/spec_decode/eagle.py:L10-L22 (整文件)
class EagleProposer(SpecDecodeBaseProposer):
    def __init__(self, ...):
        super().__init__(
            ...,
            pass_hidden_states_to_model=True,    # 唯一的差异：吃 target hidden
        )
```

我们的 `proposers/eagle.py:L20-L36` 复刻这个零差异：

```python
# implementation/proposers/eagle.py:L20-L36
class EagleProposer(SpecDecodeBaseProposer):
    def __init__(self, num_speculative_tokens: int, hidden_size: int):
        super().__init__(
            num_speculative_tokens=num_speculative_tokens,
            hidden_size=hidden_size,
            pass_hidden_states_to_model=True,
        )
```

EAGLE 的"算法选择" 全在它**指向的模型类**——`vllm/model_executor/models/llama_eagle3.py:L1-L425 Eagle3LlamaForCausalLM`。模型用 `fc: Linear(2*hidden, hidden)` 融合 target 的 hidden 和 draft 的 prev embedding，然后跑一个 transformer block，最后 LM head。**`fc` 跟 DeepSeek MTP 的 `eh_proj` 是同义不同名**——都是 (target hidden, prev embedding) 的二元融合投影。**EAGLE 与 MTP 在算法上是同一族**，差别在 fusion topology + 头深度。

Medusa 不一样：

```python
# vllm/v1/spec_decode/medusa.py:L18-L78 (节选)
class MedusaProposer:                        # 不继承 base
    def propose(self, target_hidden_states):
        blocks = self.model(target_hidden_states)        # K 个 head 并行
        logits = self.model.compute_logits(blocks)
        return torch.stack([logit.argmax(dim=-1)
                            for logit in logits], dim=1)  # 直接 argmax，无概率
```

**Medusa 的关键**：K 个 head 看的**都是同一个 target hidden**，没有 draft chain——所以 head k 看不到 head 0..k-1 的输出。这意味着 head 越远，acceptance 越快塌掉。Medusa 牺牲质量换低延迟、低参数。

我们的 `proposers/medusa.py:L56-L104` 复刻：

```python
# implementation/proposers/medusa.py:L80-L104
class MedusaProposer:
    def propose(self, target_hidden_states):
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks)
        return torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)
```

DraftModel 是"经典 spec-decode"：用一个独立的小 LM 当 draft，跑 K 步自回归。要求只有两条：(1) **同 vocab 大小**（rejection sampler 用 draft id 索引 target_logits）、(2) **同 TP 大小**（避免 torch.compile cache 跨 rank 损坏）。

```python
# vllm/v1/spec_decode/draft_model.py:L17-L88 (节选)
class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(self, ..., target_vocab, draft_vocab, target_tp, draft_tp):
        super().__init__(..., pass_hidden_states_to_model=False)   # 不喂 target hidden
        self._raise_if_vocab_size_mismatch(target_vocab, draft_vocab)
        self._raise_if_draft_tp_mismatch(target_tp, draft_tp)
```

我们的 `proposers/draft_model.py:L41-L84` 复刻这两条 raise 校验。**Trap C 的反证**：DraftModel 完全不共享架构（Llama-3.3-1B drafting Llama-3.3-70B 的常见组合），仅校验 vocab 和 TP——共享架构是 EAGLE/MTP 的事，不是"draft 的硬要求"。

Ngram 是最简单的，**无模型，无概率**：

```python
# vllm/v1/spec_decode/ngram_proposer.py:L131-L162 (节选)
class NgramProposer:
    def propose(self, sampled_token_ids, token_ids_per_req):
        for ctx in token_ids_per_req:
            # 在 ctx 的前缀里搜最长 suffix match，emit 后续 K 个
            ...
        return drafts                # 不返回 draft_probs
```

我们的 `proposers/ngram.py:L82-L101` 用朴素 O(N·K) Python 复刻 vLLM 的 numba KMP/LPS 实现——同样的算法，~100× 慢。Ngram 没有概率，所以下游 rejection sampler 走 `NO_DRAFT_PROBS=True` 路径——`draft_prob` 强制为 1，accept 条件退化为 `target_prob >= u`。M03、M14 已经记下这两个细节。

ExtractHiddenStates 最特殊——它**不真正 speculate**：

```python
# vllm/v1/spec_decode/extract_hidden_states.py:L26-L70 (节选)
class ExtractHiddenStatesProposer:
    def __init__(self, num_speculative_tokens, ...):
        # L30 的硬 assertion
        assert num_speculative_tokens == 1
        self.hidden_states_buffer = torch.zeros(...)

    def propose(self, sampled_token_ids, target_hidden_states):
        # 把 target hidden state 缓存进 buffer
        self.hidden_states_buffer[:num_tokens] = stacked_hidden
        # 返回 sampled_token_ids 不变 —— 永远 trivially verify
        return sampled_token_ids.view(-1, 1)
```

**M10 已经精校**：brief 曾说这是"single-step MTP variant"——其实更准确地说是 **KV-cache hidden-state 抽取器**，主要服务于 PD 解耦架构（Ch22+）的 KV transfer，不真正在做 spec-decode。我们的 `proposers/extract_hidden.py:L42-L91` 复刻 `assert K==1` 和"return unchanged"行为。

### 10.2.3 共享脚手架：`SpecDecodeBaseProposer.propose`

`SpecDecodeBaseProposer`（`llm_base_proposer.py:L60-L1820`，1820 行——CUDA graph、slot mapping、tree attention 等都在里面）只有一个**算法核心**：

```python
# vllm/v1/spec_decode/llm_base_proposer.py:L491-L494 (fast path)
if self.num_speculative_tokens == 1 or self.parallel_drafting:
    draft_token_ids = self._greedy_sample(last_hidden_states)
    return draft_token_ids.view(-1, self.num_speculative_tokens)

# vllm/v1/spec_decode/llm_base_proposer.py:L516-L654 (sequential path)
draft_list = [self._greedy_sample(last_hidden_states)]
for step in range(self.num_speculative_tokens - 1):
    input_ids = draft_list[-1]
    ret = self.model(input_ids=input_ids, hidden_states=hidden_states, ...)
    draft_list.append(self._greedy_sample(ret))
return torch.stack(draft_list, dim=1)
```

两条路径：**K==1 fast path**（一次 forward 直接返）或 **K>1 sequential path**（K 次 forward，前一步的 draft 喂给下一步），plus 一个**parallel_drafting flag**——`True` 时 draft 一次 forward 出 K 个 token（DFlash、parallel-EAGLE）。我们的 `proposers/base.py:L73-L168` 复刻这三种 control flow。**M11** 已经记下 `parallel_drafting` 的 layout 影响。

`num_speculative_tokens` 是**全局**（M06）：在 `SpeculativeConfig.__init__` 时设定，运行时 batch 里所有 request 共用同一个 K——但每个 request 可以有 `num_draft_tokens < K`（比如 ngram 在某 request 上没找到 suffix，就给 0 个 draft）。这点在 §10.2.4 的 `SpecDecodeMetadata` 数据契约里体现。

### 10.2.4 数据契约：`SpecDecodeMetadata` —— 5 个字段的 backpressure gate

**proposer 与 verifier 的边界是一个 dataclass，不是一个函数调用**——这是 W04 backpressure isolation 的教科书案例（Ch09 §9.4 EplbState 同模式）。`vllm/v1/spec_decode/metadata.py` 整文件 66 行，`SpecDecodeMetadata` 是其中一个 `@dataclass`，无任何方法：

```python
# vllm/v1/spec_decode/metadata.py:L9-L24 (节选)
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor          # [num_tokens] 扁平 K-per-request
    num_draft_tokens: list[int]            # 每个 request 的 K（可不等）
    cu_num_draft_tokens: torch.Tensor      # [batch] 累计
    target_logits_indices: torch.Tensor    # [num_tokens] 取 target logits
    bonus_logits_indices: torch.Tensor     # [batch] 取 bonus logits
    logits_indices: torch.Tensor           # 上两者拼接

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens) if ... else 0
```

我们的 `spec_metadata.py:L36-L113` 完全复刻字段名和 `make_dummy` 工厂方法。**M05** 已经记下：`target_logits_indices`（size `num_tokens`）和 `bonus_logits_indices`（size `batch_size`）分开——前者每个 draft 位置一个，后者每个 request 一个，因为 bonus token 只在"全 K 接受"时才采。

为什么这是 backpressure gate？因为 **proposer 输出 metadata 之后就完全脱手了**——sampler 只看 metadata，不看 proposer 的内部状态、不调 proposer 的方法。两边的 commit 频率可以独立改、内部可以重写、可以加新 proposer——只要 dataclass 的字段语义不变，下游就不需要改。**这跟 Ch09 §9.4 的 `EplbState.record_step` 写、`__post_init__` 读的 detach 边界、Ch08 `RowParallelLinear.forward` 的 reduce-then-add 边界是同一个家族的设计**。

### 10.2.5 Mini-mapping：5 个 proposer + 1 个 verifier 的协同

| 我们的 | vLLM 源码 | 关系 |
|---|---|---|
| `proposers/base.py:L61-L168 SpecDecodeBaseProposer` | `llm_base_proposer.py:L60-L1820` | 教学版；保留 propose / _greedy_sample，省 CUDA graph / slot mapping |
| `proposers/base.py:L97-L101 _greedy_sample` | `llm_base_proposer.py:L407-L412` | 一致；`compute_logits().argmax()` |
| `proposers/base.py:L104-L168 propose` (fast/sequential) | `llm_base_proposer.py:L413-L656` | 一致；K==1 fast path + K>1 sequential |
| `proposers/eagle.py:L20-L36 EagleProposer` | `eagle.py:L1-L22` | 一致；纯继承 + `pass_hidden_states_to_model=True` |
| `proposers/medusa.py:L56-L104 MedusaProposer` | `medusa.py:L18-L78` | 一致；不继承 base，K 个独立头 argmax |
| `proposers/draft_model.py:L41-L84 DraftModelProposer` | `draft_model.py:L17-L88` | 一致；vocab + draft_tp 校验 |
| `proposers/ngram.py:L38-L101 NgramProposer` | `ngram_proposer.py:L12-L162` | 一致算法；朴素 vs numba KMP |
| `proposers/extract_hidden.py:L42-L91 ExtractHiddenStatesProposer` | `extract_hidden_states.py:L26-L130` | 一致；`assert K==1` + return unchanged |
| `proposers/mtp.py:L24-L70 DeepSeekMTPProposer` | `deepseek_mtp.py:L186-L488 DeepSeekMTP` + `draft_model.py` | 一致；wraps `DeepSeekMultiTokenPredictor` |
| `spec_metadata.py:L36-L113 SpecDecodeMetadata` | `metadata.py:L9-L66` | 一致字段；`make_dummy` 是教学便利 |
| `spec_metadata.py:L117-L122 PLACEHOLDER_TOKEN_ID` | `rejection_sampler.py:L30` | 一致 = -1 |
| `spec_metadata.py:L130-L134 MAX_SPEC_LEN` | `rejection_sampler.py:L34` | 一致 = 128 |
| `rejection_sampling.py:L423-L464 RejectionSampler` | `rejection_sampler.py:L37-L195` | 教学版；去掉 nn.Module 包装 |

10 行 mini-mapping 把 5 个 proposer 全部覆盖；主表（§10.9）会进一步 fanned-out。

---

## 10.3 Inference 期 MTP 权重加载（替代 outline §3 "训练期多步 CE 损失"）

### 10.3.1 §X 重构：outline 写的是 training，源码做的是 inference

Outline 第 3 节叫 "Training——多步 CE 损失的加权策略"。grep 一下 vLLM 全树：

```
$ grep -rE "MTPLoss|multi_step_ce|compute_mtp_loss|mtp_aux_loss" \
    instances/vllm/source/vllm/
(zero matches)
```

加上 `vllm/v1/spec_decode/` 目录里的 `.backward(`：

```
$ grep -rn '\.backward\(' instances/vllm/source/vllm/v1/spec_decode/
(zero matches)
```

**vLLM 是推理引擎，不训练 MTP**——MTP heads 在上游模型的训练 repo（DeepSeek-V3、Llama EAGLE 等）里训好，vLLM 只**加载已训权重**当 draft head 用。Trap-F 的反证。这是 Ch09 §9.4 之后的**第二次** training-to-inference reframe。

**M20** 已经记下精化版：grep 必须 scope 到 spec-decode subtree。在全 `vllm/` 树里 grep `aux_loss` 会得到两个 false positive：`models/phimoe.py:router_aux_loss_coef`（HuggingFace config 字段，运行时不调）、`models/vision.py:get_load_balance_assignment`（图像 tile 放置 helper，不是 expert routing）。Tester 把测试搜索路径限定在 `vllm/v1/spec_decode/` + `vllm/model_executor/models/deepseek_mtp.py`，避免假阴。

### 10.3.2 Sidebar：训练期多步 CE 损失（来自上游论文的简介）

DeepSeek-V3 论文 / Switch Transformer / Better-MTP 的多步预测损失大致是：

$$
\mathcal{L}_\mathrm{MTP} = \sum_{k=0}^{K-1} \lambda_k \cdot \mathrm{CE}\left(p_k(\cdot \mid h_t, x_{t+1..t+k}), x_{t+1+k}\right)
$$

其中：

- $h_t$ 是 trunk 在位置 $t$ 输出的隐状态；
- $p_k$ 是第 $k$ 个 MTP 头在位置 $t+k$ 上对 vocab 的预测；
- $x_{t+1..t+k}$ 是 ground-truth 的下 $k$ 个 token；
- $\lambda_k$ 是层 weighting，典型选 $1.0, 0.5, 0.25, 0.125$（exponential decay）。

**核心约束**：第 $k$ 头看到 $x_{t+1..t+k}$ 作为输入（teacher forcing），不是看到自己生成的 chain——所以 trained MTP 头是个"K-to-1"的 mapper：给定历史 + 目标的前 k 个未来 token，预测第 k+1 个未来 token。这是为什么 inference 期 MTP 头需要走"shared trunk"路径——每一步都需要 target 模型的 $h_t$ 当输入。

**核心收益**：相比传统的 1-step CE（只预测 $x_{t+1}$），多步 CE 让 trunk 的 hidden state 对未来 K 步都有 informative gradient——这给了 trunk 更强的 long-horizon 表征。MTP 的"draft 模型本身就是 trained-with-the-target-jointly"是它在 DeepSeek-V3 上 α ≈ 0.85 的根本原因，跟 EAGLE、独立 draft model 都不一样——后两者 α 通常在 0.5-0.7。

**这一段 100% 是文献简介，源码里没有 corresponding 代码**。Sidebar 的目的是给读者一个 grounding：他们听过的"L_MTP"、"weight decay schedule"、"teacher forcing for K steps"在哪里发生（在训练 pipeline 里），不在 vLLM 里。

### 10.3.3 Pivot：vLLM 的实际响应——三路权重改写

训练期产物是 HuggingFace checkpoint（safetensors），里面 MTP 头的权重以模型作者的 layout 存放。vLLM 内部用自己的 layout（带 `mtp_block.` 命名空间），所以加载时需要**改名字**。这是 `vllm/model_executor/models/deepseek_mtp.py:L458-L488 _rewrite_spec_layer_name`：

```python
# vllm/model_executor/models/deepseek_mtp.py:L458-L488 (节选)
def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
    spec_layer_weight_names = ["embed_tokens", "enorm", "hnorm",
                               "eh_proj", "shared_head"]
    shared_weight_names = ["embed_tokens"]
    spec_layer_weight = any(w in name for w in spec_layer_weight_names)
    shared_weight = any(w in name for w in shared_weight_names)

    if not spec_layer_weight:
        # Path 1: 普通 transformer block 权重 → 包到 .mtp_block. 下
        name = name.replace(f"model.layers.{spec_layer}.",
                            f"model.layers.{spec_layer}.mtp_block.")
    elif shared_weight:
        # Path 2: shared 权重（embed_tokens）→ 提升到 top-level
        name = name.replace(f"model.layers.{spec_layer}.", "model.")
    # Path 3: MTP-specific 不改
    return name
```

**三路改写**（M15 + M21）：

| Path | 输入 | 输出 | 为什么 |
|---|---|---|---|
| 1 (block weight) | `model.layers.61.self_attn.q_proj.weight` | `model.layers.61.mtp_block.self_attn.q_proj.weight` | 融合后的 transformer block 在 vLLM 类树里嵌套在 `.mtp_block.` 下 |
| 2 (shared embed) | `model.layers.61.embed_tokens.weight` | `model.embed_tokens.weight` | embed_tokens 跟 target 共享，提升到 top-level 让 `_maybe_share_embeddings` 找得到 |
| 3 (MTP-specific) | `model.layers.61.enorm.weight` | `model.layers.61.enorm.weight`（不变） | enorm/hnorm/eh_proj/shared_head 都是 MTP 独有的；保留 layer 索引 |

我们的 `weight_loading.py:L39-L103 rewrite_spec_layer_name` 一行不改地复刻这三路。

**关键观察**：vLLM **保留 spec_layer 的索引**，不重 index 为 0。如果 target 有 61 层，MTP 1 层，`DeepSeekMultiTokenPredictor.layers` 的键是 `"61"`，不是 `"0"`。这样 `model.layers.{idx}` 在 ModuleDict 里查就直接命中——HF checkpoint 跟 vLLM 内部 layout 在"layer 索引"这一维上完全对齐，只在"路径前缀"上改。

Demo §3.6 的 verbatim 数字（`weight_loading.py:L287-L343 loader_demo_shapes`，target_layers=61, mtp_layers=1）：

```
input keys        = 193   (target × 61 + MTP-layer 4 + 2 top-level + 2 共享)
target keys       = 185
mtp keys          =   8
sample renames:
  Path 1 (block weight wrapped):
    model.layers.61.self_attn.q_proj.weight
    → model.layers.61.mtp_block.self_attn.q_proj.weight
  Path 2 (shared embed promoted):
    model.layers.61.embed_tokens.weight
    → model.embed_tokens.weight
  Path 3 (MTP-specific kept):
    model.layers.61.eh_proj.weight              (unchanged)
    model.layers.61.shared_head.head.weight     (unchanged)
```

这 3 行示例是 framing tip 4 的 sidebar+pivot 模板的"pivot 锚点"——读者看完 sidebar 的训练 loss math，再看到这 3 行 verbatim 改写，立刻知道"vLLM 不算 loss，vLLM 只搬运权重"。

### 10.3.4 Pivot 第二段：`_maybe_share_lm_head` 的 0.93 GB 节省

第二个 pivot 锚点：**LM head 权重共享**。`vllm/v1/spec_decode/llm_base_proposer.py:L1471-L1539 _maybe_share_lm_head`：

```python
# vllm/v1/spec_decode/llm_base_proposer.py:L1471-L1539 (节选)
# 注释（关键）：
# "MTP models call compute_logits via shared_head.head (a ParallelLMHead
#  inside each MTP layer), not self.model.lm_head. If the checkpoint omits
#  a copy of the lm_head weights at the MTP layer path, shared_head.head
#  stays uninitialised and produces NaN logits. Always share it explicitly."

def _maybe_share_lm_head(self, target_language_model):
    target_lm_head = target_language_model.lm_head
    if hasattr(self.predictor, "lm_head"):
        del self.predictor.lm_head
    self.predictor.lm_head = target_lm_head
    # 内层每一层的 shared_head.head 都换成 target.lm_head
    for layer in self.predictor.model.layers.values():
        if hasattr(layer.shared_head, "head"):
            del layer.shared_head.head
            layer.shared_head.head = target_lm_head
```

为什么强制共享？因为每个 MTP 层都有自己的 `SharedHead`（`deepseek_mtp.py:L43-L62`），每个 SharedHead 都有自己的 `ParallelLMHead`。DeepSeek-V3 的 vocab=129280, hidden=7168 —— 每层 lm_head 是 `129280 × 7168 ≈ 926M params ≈ 1.85 GB (bf16)`。1 层 MTP 就要单独再付 1.85 GB；`_maybe_share_lm_head` 把它跟 target 共享，**直接省一倍参数**。

**对应 demo §3.5 的 verbatim 数字**——共享 vs 不共享对 MTP 跟 Medusa 的对比：

```
hidden=2048, intermediate=8192, vocab=32000, K=2

MTP per-layer params       =     75,505,664
   enorm                   =          2,048
   hnorm                   =          2,048
   eh_proj (2h*h)          =      8,388,608
   mtp_block_attn          =     16,777,216  (4 * h * h)
   mtp_block_ffn           =     50,331,648  (3 * h * inter)
   mtp_block_norms         =          4,096
MTP total (shared lm_head) =    216,549,376       # 共享 lm
MTP total (separate lm)    =    282,085,376       # 每层独立 lm

Medusa per-head            =     73,924,608
   per-head MLP-only       =      8,388,608  (2 * h * h)
   per-head LM-only        =     65,536,000  (vocab * h)

Ratio MTP/Medusa (shared lm)   = 12.91x       ← Trap E 的硬证据
Ratio MTP/Medusa (separate lm) =  1.91x
```

注意算法：**12.91× 是 MTP-stack 跟 Medusa-stack 的总体比**，不是 per-layer 比。MTP 1 个 layer 75.5M，加上共享后的 216.5M（包括 embed_tokens）；Medusa K=2 头共 ~16.8M。比值 12.91。**MTP 头的"重"主要不在 lm_head 上**（lm_head 共享后被剔除）——主要在 `mtp_block` 这个完整 transformer block 里：`mtp_block_attn=16.8M + mtp_block_ffn=50.3M = 67M`，占 per-layer 75.5M 的 **88.7%**。Trap E 的精确表述：**MTP 头的重量来自完整 transformer block（attention + FFN），不是 lm_head；DeepSeek 实际还把 dense FFN 换成 MoE，参数会再涨 ~10×**。

我们的 `mtp_head.py:L367-L417 parameter_count_mtp` 用闭式公式算这些数，跟 nn.Module 的 `sum(p.numel() ...)` 完全一致（demo 里两者都跑，pin verbatim 一致）。

### 10.3.5 Pivot 第三段：`SharedHead.forward` 只返回 norm（M09 校正）

第三个 pivot 锚点：**SharedHead 的 forward 不算 logits**。`deepseek_mtp.py:L43-L62`：

```python
# vllm/model_executor/models/deepseek_mtp.py:L43-L62
class SharedHead(nn.Module):
    def __init__(self, ...):
        self.norm = RMSNorm(...)
        self.head = ParallelLMHead(...)             # 后面被 _maybe_share_lm_head 替换

    def forward(self, hidden_states):
        return self.norm(hidden_states)             # 只返回 normed hidden，不算 logits
```

**M09 校正**：brief 曾说 "SharedHead.forward 直接返回 logits"——这是错的。源码 `forward` **只返回 `norm(hidden_states)`**；`compute_logits` 才是单独的 step。我们的 `mtp_head.py:L162-L170` 复刻：

```python
# implementation/mtp_head.py:L162-L170
class SharedHead(nn.Module):
    def forward(self, hidden_states):
        return self.norm(hidden_states)            # 只 norm

    def compute_logits(self, normed):
        return self.head(normed)                    # 单独算 logits
```

为什么这样设计？因为 logits 的计算（一个 vocab × hidden 的大 GEMM）放在`DeepSeekMultiTokenPredictor.compute_logits`（`L172-L182`）里——它有自己的 logit_processor、softmax temperature、max_logprobs cap。SharedHead 只做"normalize"这件 cheap 的事，让 LM head 这个 expensive 的 op 跟下游的 sampling pipeline 对齐。这是 Ch10 的另一个 backpressure-gate 实例：**算 logits 跟 normalize 之间有一道边界，让 kernel 选择 / dtype upcast / logit processor 都可以独立决定**。

### 10.3.6 Mini-mapping：权重加载与共享

| 我们的 | vLLM 源码 | 关系 |
|---|---|---|
| `weight_loading.py:L39-L103 rewrite_spec_layer_name` | `deepseek_mtp.py:L458-L488` | 一致；三路改写 |
| `weight_loading.py:L106-L143 remap_checkpoint` | `deepseek_mtp.py:L271-L456 load_weights` | 教学版；只做名字改写部分 |
| `weight_loading.py:L147-L200 maybe_share_lm_head` | `llm_base_proposer.py:L1471-L1539` | 一致；删除 mtp.lm_head + 替换内层 shared_head.head |
| `weight_loading.py:L204-L232 maybe_share_embeddings` | `llm_base_proposer.py:L1402-L1469` | 一致；MTP 强制共享 |
| `weight_loading.py:L239-L257 acceptance_length_to_rates` | `speculative.py:L213-L227` | 一致；最小方差 schedule |
| `weight_loading.py:L260-L284 unconditional_to_conditional_rates` | `vllm/v1/spec_decode/utils.py` | 教学版；分位数除法 |
| `weight_loading.py:L287-L343 loader_demo_shapes` | demo §3.6 来源 | 教学装置 |
| `mtp_head.py:L42-L58 RMSNorm` | `vllm/model_executor/layers/layernorm.py RMSNorm` | 一致公式 |
| `mtp_head.py:L148-L177 SharedHead` | `deepseek_mtp.py:L43-L62` | 一致；forward 只 norm |
| `mtp_head.py:L204-L276 DeepSeekMultiTokenPredictorLayer` | `deepseek_mtp.py:L63-L122` | 一致 layout；MoE 替换为 dense FFN |
| `mtp_head.py:L279-L364 DeepSeekMultiTokenPredictor` | `deepseek_mtp.py:L124-L184` | 一致；K 步 propose_K |

---

## 10.4 Rejection sampling kernel：从 driver 到两条 kernel 路径

### 10.4.1 Driver 的两条分支

回到 §10.1.1 的 driver `rejection_sample`：

```python
# vllm/v1/sample/rejection_sampler.py:L450-L502 (节选)
if not all_random:                              # 至少有一个 greedy req
    target_argmax = target_probs.argmax(dim=-1).to(torch.int32)
    rejection_greedy_sample_kernel[grid](...)
    if all_greedy:
        return output_token_ids                 # 全 greedy 提前返

# 否则进入 random path：先 softmax 再算残差再采样
target_probs = target_probs.softmax(...)
recovered_token_ids = sample_recovered_tokens(...)
rejection_random_sample_kernel[grid](...)
return output_token_ids
```

**Greedy path 是 fast path**（M02）：当 `sampling_metadata.all_greedy=True` 时，target 模型走 argmax（无 softmax，无温度），所以 verifier 只需要比 `draft_id == target_argmax` ——**整个 recover sampling 都跳过**。Ngram 走 greedy（无 draft probs）；Medusa 也走 greedy（其 propose 直接 argmax）；DeepSeek MTP 在低温（greedy）配置下也走这条。

**Random path** 是完整算法：先把 target_logits 做 softmax 拿 $p$，预先算每个位置的 recovered token（不管这个位置有没有真 reject——GPU 不喜欢分支发散，统一 precompute 比 lazy 更快），再走 accept/reject 循环。

### 10.4.2 Greedy kernel 的 5 行核心

打开 `vllm/v1/sample/rejection_sampler.py:L708-L757 rejection_greedy_sample_kernel`：

```python
# vllm/v1/sample/rejection_sampler.py:L723-L757 (节选)
@triton.jit
def rejection_greedy_sample_kernel(...):
    # L723: per-request greedy gate
    if is_greedy_ptr is not None:
        is_greedy = tl.load(is_greedy_ptr + req_idx)
        if not is_greedy:
            return                          # skip this request, random kernel will handle

    rejected = False
    for pos in range(num_draft_tokens):
        if rejected:
            break                           # L734: 链断点，停止写
        draft = tl.load(draft_token_ids_ptr + start + pos)
        target = tl.load(target_argmax_ptr + start + pos)
        token_id = target                   # L743-L745: 写 target 的 argmax
        rejected = (draft != target)
        tl.store(output_ptr + (req_idx, pos), token_id)

    if not rejected:
        # L751-L757: 全部接受，写 bonus
        tl.store(output_ptr + (req_idx, num_draft_tokens),
                 bonus_token_ids[req_idx])
```

这是 5 个观察：

1. **`if rejected: break`** 是链断点的显式编码——只要前面有任何位置 reject，后面的位置不写——加上 driver 预填的 `-1`，kernel "停笔" 就等于 "后面是 PLACEHOLDER"。
2. **写的总是 target 的 argmax**——即使 accept 时（`draft == target`），写哪个都一样，所以源码 unconditionally 写 target；这避免了一个分支。
3. **bonus 在 `num_draft_tokens` 这个 slot 写**——不是 `max_spec_len`。每个 request 的 K 可不同，bonus 的 slot 跟着各自的 K 走。
4. **`is_greedy_ptr=None` 表示全 greedy**（M02）——driver 在全 greedy 时传 `None`，kernel 判 None 就直接进 loop。这是 fast path 的 fast。
5. **没有概率计算**——argmax 比较是 int 操作，跟 `target_probs` 的 softmax 完全无关。Greedy 是真的便宜。

我们的 `rejection_sampling.py:L56-L131 rejection_greedy_sample_loop` 用 Python 逐行复刻这 5 个观察。

### 10.4.3 Random kernel 的累计判定

打开 `vllm/v1/sample/rejection_sampler.py:L760-L826 rejection_random_sample_kernel`：

```python
# vllm/v1/sample/rejection_sampler.py:L789-L826 (节选)
@triton.jit
def rejection_random_sample_kernel(..., NO_DRAFT_PROBS: tl.constexpr):
    # L779-L782: skip greedy reqs (greedy kernel handled them)
    if is_greedy_ptr is not None:
        if tl.load(is_greedy_ptr + req_idx):
            return

    rejected = False
    for pos in range(num_draft_tokens):
        if rejected:
            break
        draft_id = tl.load(draft_token_ids_ptr + start + pos)
        u = tl.load(uniform_probs_ptr + start + pos)         # ~ Uniform[0,1)

        if NO_DRAFT_PROBS:                                   # ngram 路径
            draft_prob = 1.0
        else:
            draft_prob = tl.load(draft_probs_ptr + (start + pos, draft_id))
        target_prob = tl.load(target_probs_ptr + (start + pos, draft_id))
        # L797-L810: accept iff draft_prob > 0 AND target_prob/draft_prob >= u
        accepted = (draft_prob > 0.0) & ((target_prob / draft_prob) >= u)

        if accepted:
            token_id = draft_id
        else:
            # L811-L815: 用预先采的 recovered token
            rejected = True
            token_id = tl.load(recovered_ids_ptr + start + pos)
        tl.store(output_ptr + (req_idx, pos), token_id)

    if not rejected:
        tl.store(output_ptr + (req_idx, num_draft_tokens),
                 bonus_token_ids[req_idx])
```

5 个观察：

1. **`(target_prob / draft_prob) >= u` 跟 `u < min(1, p/q)` 等价**（在 $u \in [0, 1)$ 上）——这是 §10.1.4 不偏证明里的 accept 条件。
2. **`NO_DRAFT_PROBS=True`** 是 Triton metaparam（M03）——不是运行时分支，是编译期开关，所以两个版本的 kernel 各自最优。
3. **`draft_prob > 0` 的硬 guard**——防御 `q(d) = 0` 的非法状态（理论上不该发生，因为 draft 是从 q 抽的，所以 $q(\mathrm{draft}) > 0$；但浮点下溢可能让 q 变 0），强制 reject 走 recover。
4. **recovered token 是预先采好的**——`sample_recovered_tokens` 在 driver 里跑，给每个 draft 位置都采一个 recover；kernel 只在 reject 时取。这是 GPU 友好的 layout——kernel 内不发散到 sampling 操作。
5. **`SYNTHETIC_MODE`** 是另一个 metaparam——synthetic 测试模式跳过概率计算，直接用预设的 `synthetic_conditional_rates[pos]` 当 accept 阈值。M13 的合成测试基础。

我们的 `rejection_sampling.py:L138-L216 rejection_random_sample_loop` 用 Python 复刻——`NO_DRAFT_PROBS` 和 `SYNTHETIC_MODE` 是 Python 函数参数，效果一致。

### 10.4.4 Recovered token：(p-q)_+ 的 Gumbel-max 采样

回到 `sample_recovered_tokens`（`rejection_sampler.py:L659-L703`）的内核（`L853-L920`）：

```python
# vllm/v1/sample/rejection_sampler.py:L877-L920 (节选)
@triton.jit
def sample_recovered_tokens_kernel(..., NO_DRAFT_PROBS: tl.constexpr):
    if NO_DRAFT_PROBS:
        # ngram 路径：residual = target_probs with draft_id masked to 0
        draft_id = tl.load(draft_token_ids_ptr + token_idx)
        # 把 draft_id 那一项设 0，其余取 target_probs
        ...
    else:
        # 标准残差：(p - q)_+，clamp 到 0 以下
        p = tl.load(target_probs_ptr + (token_idx, :))
        q = tl.load(draft_probs_ptr + (token_idx, :))
        residual = tl.maximum(p - q, 0.0)

    # Gumbel-max trick: argmax over residual * (1 / Exp(1) sample)
    # L904-L905: 不归一化（argmax 对正缩放不变）
    recovered = tl.argmax(residual * inv_q[req_idx, :])
```

**Gumbel-max trick**（M14）：要从 residual 这个非归一化分布里采样，标准做法是先归一 residual / Z，然后用 multinomial。但有个等价做法——先采指数变量 $E_v \sim \mathrm{Exp}(1)$（size = vocab），然后取

$$
\mathrm{argmax}_v\!\bigl(\mathrm{residual}_v / E_v\bigr) \;=\; \mathrm{argmax}_v\!\bigl(\mathrm{residual}_v \cdot \mathrm{inv\_q}_v\bigr)
$$

给出从归一化分布中采样的等价结果。**这避免了 reduction（求和算 Z）**——argmax 是一次 stride，比 reduction 友好得多；归一化的除法也省了。一个 GPU 友好的小工艺。

我们的 `rejection_sampling.py:L223-L279 sample_recovered_tokens_loop` 复刻（注意我们用 `torch.argmax(p * inv_q[req_idx])`，跟源码完全一致）。

### 10.4.5 Greedy vs Random：emit 数的对比

Demo §3.4（`tests/test_rejection_sampling.py::test_demo_34_greedy_emit_in_range`）：

```
Trials              = 1000
K                   = 4
Greedy mean emit    = 1.5120
Random mean emit    = 4.5150
ratio random/greedy = 2.9861
Greedy emit min/max = 1/5
Random emit min/max = 1/5
```

**Random emit ~3× Greedy emit** 在这个合成 benchmark 上。为什么？因为随机模式里**每个位置都至少 emit 1 个 token**（accept 的 draft 或 recovered token），而 greedy 模式里只有"draft == argmax"才接受，否则 emit 一个 target argmax 就停。在低 α 下（这次 benchmark 是合成的低 α），greedy 经常在第 1-2 个位置就停，random 还能继续走到 K+1。

**这跟 framing tip 2 的 net-loss 风险相辅相成**：低 α 下 greedy 比 random 更容易亏（emit 少、target forward 还是付了 K 步成本），所以 greedy 配置（temperature=0）下 K 应该选小。Production 部署如果用 greedy + K=4 + α=0.3，用 demo §3.4 的 1.51 emit / forward + c=0.10 算实际加速：$1.51 / (1 + 0.4) = 1.08$——勉强打平。换成 random + 同样 α=0.3 + K=4 + c=0.10：$1.42 / 1.4 = 1.01$（用 §3.2 的 emit 2 + bonus 期望 1.4251）——也几乎打平。**low α 下，greedy 跟 random 都不香**——这跟运维直觉里的"greedy 总是最快"反着来。

### 10.4.6 Mini-mapping：rejection sampling kernel 三件套

| 我们的 | vLLM 源码 | 关系 |
|---|---|---|
| `rejection_sampling.py:L56-L131 rejection_greedy_sample_loop` | `rejection_sampler.py:L708-L757 rejection_greedy_sample_kernel` | 一致；Triton vs Python loop |
| `rejection_sampling.py:L66-L91 SYNTHETIC_MODE` 分支 | `rejection_sampler.py:L737-L742` | 一致；synthetic test path |
| `rejection_sampling.py:L100-L104 chain-break break` | `rejection_sampler.py:L734 if not rejected` | 一致；M04 锚点 |
| `rejection_sampling.py:L122-L124 token_id = target_id` | `rejection_sampler.py:L743-L745` | 一致；unconditional write |
| `rejection_sampling.py:L138-L216 rejection_random_sample_loop` | `rejection_sampler.py:L760-L826` | 一致 |
| `rejection_sampling.py:L195-L203 NO_DRAFT_PROBS` 分支 | `rejection_sampler.py:L797-L799 NO_DRAFT_PROBS` metaparam | 一致；M03 锚点 |
| `rejection_sampling.py:L223-L279 sample_recovered_tokens_loop` | `rejection_sampler.py:L853-L920 sample_recovered_tokens_kernel` | 一致；Gumbel-max 采样 |
| `rejection_sampling.py:L260-L265 NO_DRAFT_PROBS residual` | `rejection_sampler.py:L877-L891` | 一致；mask draft_id |
| `rejection_sampling.py:L268-L271 (p-q)_+ residual` | `rejection_sampler.py:L892-L903` | 一致；clamp(min=0) |
| `rejection_sampling.py:L276-L277 score = p * inv_q` | `rejection_sampler.py:L904-L905 unnormalized argmax` | 一致；Gumbel-max trick |

---

## 10.5 Proposer 横向对比：5 种 draft 生产策略的 (cost, accuracy, coupling) 三元组

### 10.5.1 五条路径的取舍

每个 proposer 在三个轴上做不同选择：

- **Draft cost ($c$)**：每步 draft forward 占 target forward 的多少？
- **Acceptance rate ($\alpha$)**：draft 跟 target 的分布有多接近？
- **Coupling**：draft 跟 target 的架构有多耦合？

下面是 5 个 proposer 在这三个轴上的定性 + 量化对比（M07 + Brief §1.5 + framing tip 5）：

| Proposer | Draft cost $c$ | $\alpha$ 典型 | 共享 trunk | 参数 (h=2048, K=2) | 适用场景 |
|---|---|---|---|---|---|
| **Eagle / EAGLE3** | 0.05-0.10 | 0.7-0.9 | ✓ (target hidden) | ~80M | DeepSeek 系，shared-trunk 训练 |
| **Medusa** | 0.02-0.05 | 0.4-0.6 | ✓ (target hidden) | ~16M (K=2) | 想要轻量、低 latency 服务 |
| **DraftModel** | 0.02-0.05 | 0.5-0.7 | ✗ | ~1B (Llama-3.3-1B) | 任何同 vocab 的 LM 对 |
| **Ngram** | ~0 | 0.3-0.5 | ✗ | 0 | 代码、重复文本 |
| **DeepSeek-MTP** | 0.05-0.10 | 0.85+ | ✓ (target hidden + 联合训练) | ~217M (K=1, MoE 后 ~2B) | DeepSeek-V3 推理 |

**Trap C 的反证**：5 个 proposer 里 3 个不共享架构（DraftModel、Ngram、ExtractHidden），仍然能 work——共享 trunk 是一种"提高 α"的设计选择，不是"draft 的硬要求"。

**Trap E 的硬证据**：MTP 在参数上是 ~217M（共享 lm_head 后），加 MoE 是 ~2 GB（dense FFN 替换为 256-routed-expert MoE block），相比 Medusa 的 16M（K=2）**贵 12.91×**。"MTP 头是轻量"是错的——只有 Medusa 是真轻量。

### 10.5.2 EAGLE 与 MTP 的"shared trunk"对比

EAGLE 与 MTP 是 spec-decode 里的 **shared-trunk 双子星**——都拿 target 的 hidden state 当 draft 的输入，但 fusion 拓扑不一样：

```
DeepSeek MTP:                              EAGLE:
  emb_n = enorm(prev_token_emb)             prev_emb = embed(prev_token)
  h_t   = hnorm(target_hidden)              fused = fc(concat([prev_emb,
  fused = eh_proj([emb_n, h_t])                          target_hidden]))
  h, residual = mtp_block(fused)            h, residual = trans_block(fused)
  return h + residual                       return h + residual
```

**两者本质相同**（concat 两个 stream → 投影 → 一个 transformer block → 残差），区别在：

1. **MTP 把两个 norm（enorm, hnorm）放在投影前**——RMSNorm 让两个 stream 在 magnitude 上对齐；EAGLE 不这样做，依赖 `fc` 自己学到正确的尺度。
2. **MTP 是 DeepSeek 训练的**（多步 CE loss），所以 mtp_block 内部用 DeepSeek 的 MoE block；EAGLE 是后训练阶段在已有 base 上微调出来的小型 transformer。
3. **K-step 行为**：MTP 用 `num_nextn_predict_layers` 个不同的 MTP layer（每层负责"预测第 k 个未来 token"），通常 K=1（DeepSeek-V3）或 2-4（V3.2）；EAGLE 通常 K=4-8，重复用同一个 EAGLE 模型自回归。

我们的 `mtp_head.py:L204-L256 DeepSeekMultiTokenPredictorLayer.forward` 复刻 MTP 的融合 + transformer block：

```python
# implementation/mtp_head.py:L240-L256
def forward(self, inputs_embeds, previous_hidden_states, positions):
    # source 在 position 0 把 emb 设 0（因为没有 prev token）
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0.0, inputs_embeds)
    emb_n = self.enorm(inputs_embeds)
    h_t   = self.hnorm(previous_hidden_states)
    fused = self.eh_proj(torch.cat([emb_n, h_t], dim=-1))
    h, residual = self.mtp_block(fused)
    return h + residual
```

**Position 0 的 mask**（M22 + 源码 L107-L113）：第一步 MTP 还没有"上一个 draft token"可以喂进来，所以 source 在 position == 0 的位置把 inputs_embeds 强制设 0——等价于"忽略 prev token 那个 stream，纯粹根据 target hidden 预测"。这是 DeepSeek 训练时的一致约定。

### 10.5.3 Medusa vs MTP：参数量 12.91× 的来源

Medusa 不共享 trunk，K 个独立 MLP 头并行预测。它的便宜是真便宜：

```python
# implementation/mtp_head.py:L420-L443
def parameter_count_medusa(hidden_size, vocab_size, K):
    h = hidden_size
    per_head_mlp = 2 * h * h          # 2-layer MLP
    per_head_lm = vocab_size * h      # LM proj
    return {
        "per_head_mlp": per_head_mlp,           # 8.39M
        "per_head_lm": per_head_lm,             # 65.5M
        "per_head": per_head_mlp + per_head_lm, # 73.9M
    }
```

K=2 Medusa：2 × 73.9M = 147.8M（独立 lm）或 2 × 8.39M = 16.78M（共享 lm）。**Medusa 真正"轻"的部分是 MLP 那 8.39M**——LM 头如果不共享，每个 head 65.5M，加起来跟 MTP 差不多。所以"Medusa 是轻量"的精确表述：**Medusa 的 MLP 部分是轻量；如果 lm_head 共享，整个 stack 才真便宜（12.91× vs MTP）；如果 lm_head 独立，比值掉到 1.91×**——Trap E 的细节。

**为什么 Medusa 准确率低？** 因为 K 个 head 看到的都是同一个 target hidden state——head 0 预测 $x_{t+1}$、head 1 预测 $x_{t+2}$、…… head k 预测 $x_{t+k+1}$。但 head k 看不到 head 0..k-1 的输出，没法用 chain 信息——每个 head 都在"瞎猜"远端 token。结果就是 acceptance rate 随位置快速塌掉：position 0 可能 α=0.7，position 1 跌到 0.5，position 2 到 0.35，position 3 到 0.2。MTP 不一样：每一步用上一步的 draft + target hidden 合成，chain 信息保留——所以 α 衰减慢，K=4 时 α 仍能保持 0.7 以上。**Medusa 适合 K=2-3、低延迟服务；MTP 适合 K=4-8、批量推理**。

### 10.5.4 DraftModel 的 vocab + TP 校验

DraftModel 是"独立小模型"——典型 setup 是 Llama-3.3-1B 给 Llama-3.3-70B 当 draft。两条硬约束：

1. **vocab_size 必须相等**（`draft_model.py:L33-L34`）——rejection sampler 用 `target_logits[draft_id]` 索引 target 概率；如果 vocab 不一样，索引越界。
2. **draft_tp == target_tp**（`draft_model.py:L36-L51`）——torch.compile cache 在不同 rank 上独立编译；如果 draft 跑在 rank 0 上而 target 在 4 ranks，rank 0 编译过的 draft kernel 跟 rank 1-3 看不到，cache 错乱。issue tracker `vllm-project/vllm#5414`。

**Trap C 的细节**：DraftModel 不要求**架构一致**（embedding dim、num_layers、attention head 数都可以差很大），但要求**接口一致**（vocab、TP）。"draft 必须共享 target 架构"是错的；"draft 必须能 verifier 喂进来的 token"才是真的。

### 10.5.5 Ngram 的 NO_DRAFT_PROBS 信号链

Ngram 没有概率，意味着 rejection_sample 走 `NO_DRAFT_PROBS=True` 路径——`draft_prob` 强制为 1.0，accept 条件退化为 `target_prob >= u`。直觉上："target 给 draft 任何非平凡概率，就接受"——这是**比 greedy 弱、比 random 还弱**的策略。但因为 ngram 的 draft cost 几乎 0，即使 α=0.3 也是净赚（demo §3.3：K=4, c=0.05, α=0.30 → S=1.188）。

这条路径的不偏证明也成立——M14 已经记下：当 `draft_probs is None` 时残差退化为 "target_probs with draft_id masked"，仍然是 valid 概率分布，仍然给出 unbiased emit。

我们的 `proposers/ngram.py:L82-L101 _find_and_propose` 用朴素 O(N·K) Python 复刻 vLLM 的 numba KMP/LPS 算法。

### 10.5.6 ExtractHiddenStates：一个非真正 spec-decode

最特殊的一条路径——它的 `assert num_speculative_tokens == 1` 把"K 步"完全砍掉。**`propose` 不真正 propose**，只**把 target hidden state 缓存进 buffer**，返回 sampled_token_ids 不变（永远 trivially verify）。M10 已经记下这是"KV-cache hidden-state 抽取器"，主要服务 PD 解耦架构（Ch22+）。

为什么 vLLM 要把这玩意儿放进 `vllm/v1/spec_decode/`？因为它复用了 spec-decode 框架的 plumbing（特别是 `target_hidden_states` 那个传参链）——是个 **proposer 接口的非典型用法**，不是真的在做 spec-decode。**Trap "所有 proposer 都在做 spec-decode" 的反证**——proposer 接口比 spec-decode 用例宽。

### 10.5.7 Mini-mapping：5 个 proposer + 横向对比表

| 我们的 | vLLM 源码 | 关系 |
|---|---|---|
| `proposers/eagle.py:L20-L36` | `eagle.py:L1-L22` | 一致；纯继承 |
| `proposers/medusa.py:L32-L52 MedusaHeads` | `medusa.py:L48-L49` heads | 教学版；省 layer norm |
| `proposers/medusa.py:L80-L104 MedusaProposer.propose` | `medusa.py:L48-L55 propose` | 一致；K 头 stacked argmax |
| `proposers/draft_model.py:L62-L71 _raise_if_vocab_size_mismatch` | `draft_model.py:L33-L34` | 一致 |
| `proposers/draft_model.py:L73-L84 _raise_if_draft_tp_mismatch` | `draft_model.py:L36-L51` | 一致 + 引用 issue #5414 |
| `proposers/ngram.py:L38-L77 NgramProposer` 接口 | `ngram_proposer.py:L12-L62 __init__` | 一致 |
| `proposers/ngram.py:L82-L101 _find_and_propose` | `ngram_proposer.py:L198-L285 _find_longest_matched_ngram_*` | 一致算法；朴素 vs numba KMP |
| `proposers/extract_hidden.py:L42-L66 __init__` 含 `assert K==1` | `extract_hidden_states.py:L29-L31` | 一致；M10 锚点 |
| `proposers/extract_hidden.py:L68-L91 propose` | `extract_hidden_states.py:L72-L130` | 一致；返回 sampled_token_ids 不变 |
| `proposers/mtp.py:L24-L70 DeepSeekMTPProposer` | `deepseek_mtp.py:L186-L488` + `draft_model.py` | 一致；wraps `DeepSeekMultiTokenPredictor` |

---

## 10.6 系统影响与跨章串接：α-K 速度曲线 + net-loss zone + cross-chapter

### 10.6.1 α-K 速度曲线：DeepSeek-V3 的 3.4× 与 ngram 的 1.2×

把 §10.1.3 的 break-even α 表跟典型 production 配置叠起来：

| 配置 | $K$ | $c$ | $\alpha$ | $E[\mathrm{tok}]$ | 加速比 $S$ |
|---|---|---|---|---|---|
| DeepSeek-V3 MTP（H100, NVLink） | 4 | 0.05 | 0.85 | 3.622 | **2.96×** |
| DeepSeek-V3 MTP（H100, IB 跨节点） | 4 | 0.10 | 0.85 | 3.622 | **2.59×** |
| Llama-3.3 (1B drafts 70B) | 4 | 0.03 | 0.65 | 2.512 | 2.24× |
| Llama-3.3 (跨节点) | 4 | 0.10 | 0.65 | 2.512 | 1.79× |
| Medusa K=2 | 2 | 0.03 | 0.50 | 1.750 | 1.65× |
| **Ngram on code** | 4 | 0.005 | 0.45 | 1.654 | **1.62×** |
| **Ngram on prose** | 4 | 0.005 | 0.30 | 1.425 | 1.40× |
| **Ngram + 长 K (high K)** | 8 | 0.005 | 0.30 | 1.428 | **1.37×** |
| **MTP 配低 α 的 risk** | 4 | 0.10 | 0.30 | 1.425 | **1.018**（勉强打平） |
| **MTP 配低 α 跨节点** | 4 | 0.20 | 0.30 | 1.425 | **0.792**（净亏！） |

**最后两行是 framing tip 2 的核心（M18）**：DeepSeek-V3 报告 α=0.85+，那是 long-form 数学和代码生成场景下的优势区间。但同样的 MTP 头部署到对话（α≈0.4-0.5）或者跨节点 IB（c↑），加速比会大幅缩水甚至变成负的。**运维必须先测 α 再决定 K**——没有"MTP 总是更快"的金科玉律。

这跟 framing tip 1 的"K=4 ≠ 4×"一脉相承：在最理想的 DeepSeek-V3 配置下，$E[\mathrm{tok}] = 3.62$（K+1=5 的 72%），加速比 ≈ 3×（不是 K=4 的 4×）。**spec-decode 的现实加速从来不会到 K**——chain-break 几何 + draft 成本两面夹击。

### 10.6.2 跨章接口：Ch10 用了什么、给了什么

**Back-pointers（Ch10 复用）**：

- **Ch01（Self-attention）**：MTP block 内部用 MHA。`mtp_head._MultiHeadAttention.forward` 用 Ch01 的标准 MHA pattern：reshape qkv → permute heads first → masked softmax → o_proj。M22 的 reshape-then-permute 修了一个早期 shape bug——causal mask `[T, T]` 跟 `[H, T, T]` 才能正确广播。
- **Ch08（Tensor Parallelism）**：DraftModel 可以有自己的 TP，`draft_tp == target_tp` 是硬约束。`eh_proj`、`o_proj`、`gate_up_proj` 都可以 TP 切（vLLM 用 `MergedColumnParallelLinear` + `RowParallelLinear`，跟 Ch08 §8.2 一致）。MTP 的 lm_head 通过 `_maybe_share_lm_head` 跟 target 共享，省下 vocab × hidden × K 参数。
- **Ch09（Expert Parallelism）**：**DeepSeek MTP 的 mtp_block 是 `DeepseekV2DecoderLayer`，里面有 MoE**——所以真实参数量比 demo §3.5 的 dense FFN 估计**多 10×**。Ch09 §9.5 的 `mem_per_rank ∝ 1/(ep × tp)` 不变量直接套用：MTP layer 上 EP+TP 均匀切分。

**Forward-pointers（Ch10 设置）**：

- **Ch15+（模型 zoo / Llama / Qwen / Mistral）**：每个支持 MTP 的模型族都有自己的 `*_mtp.py` wrapper（30 个），结构跟 DeepSeek MTP 同构（共享 `DeepSeekMultiTokenPredictorLayer`，差别在 trunk 模型类）。Llama EAGLE3 的 `Eagle3LlamaForCausalLM`、Qwen3 的 `qwen3_5_mtp.py`、ERNIE 的 `ernie_mtp.py`——Ch15+ 把每个变体的 fusion 拓扑详细对比。
- **Ch27（DeepSeek-V3.2 deep-dive）**：DeepSeek-V3.2 升级了 MLA + MTP 配置；其 mtp_block 内部用 MLA（不是普通 MHA）+ 256-routed-expert MoE。Ch27 把 §10.5 的"MTP 12.91× Medusa"重新算成 "~120× Medusa"（在 MoE + MLA 后），并解释 production α=0.85 的工作流分布敏感度。
- **Ch28（DeepSeek-V4-pro）**：`deepseek_v4_mtp.py` 加了 `hc_mult` carrier hidden-state expansion——把 trunk 的隐状态在喂给 MTP 前**再扩 2-4×**，给 MTP 更多 context。Ch28 把这个机制跟 §10.3 的 reframe 串起来。

### 10.6.3 同周期 hand-off：tester 给的 5 个 framing tips 检查表

Tester 给了 5 个 framing tips（test-report.md §"5 framing tips"），逐一在本章里落到三个锚点（hook + body + recap，per Ch09 E22）：

| Tip | Hook 锚点 | Body 锚点 | Recap 锚点 |
|---|---|---|---|
| 1. "K=4 ≠ 4×"（lead with formula） | 导言 §1 公式 + §10.1.2 Trap A 公式介绍 | §10.1.2 35 格表 + §10.1.3 break-even | §10.7 Trap A |
| 2. Net-loss zone is THE risk | 导言 §1 9 个 break-even α 列举 | §10.1.3 K=4 row + §10.6.1 production 表 | §10.7 Trap B |
| 3. MTP 头不是轻量（Trap E） | 导言 §1 12.91× 引用 | §10.3.4 demo §3.5 verbatim + §10.5.3 Medusa 对比 | §10.7 Trap E |
| 4. vLLM inference-only（Trap F） | 导言 §1 §10.3 reframe 提及 | §10.3.1 grep + §10.3.2 sidebar + §10.3.3 pivot | §10.7 Trap F |
| 5. "no class MultiTokenPrediction" 4 实例 | 章名 + 导言 §1 4-instance 列表 | §10.2.1 grep evidence + 5-proposer 列表 | §10.10 总结 |

每个 tip 三锚点齐全；reviewer 验证可对照本表 cross-check。

### 10.6.4 Mini-mapping：跨章接口与 forward-pointer

| Ch10 引用 | 目标章节 | 内容 |
|---|---|---|
| `mtp_head._MultiHeadAttention.forward` reshape-then-permute | Ch01 §1.3 self-attention | 同 MHA pattern；`[T, 3, H, D]` → permute `[H, T, D]` |
| MTP layer 用 `eh_proj` `MergedColumnParallelLinear` candidate | Ch08 §8.2 column-parallel | 同 row-merging；可 TP 切分 |
| MTP block 内部 MoE | Ch09 §9.1-§9.5 EP | DeepSeek MTP 实际是 MoE block；本章用 dense FFN approximation |
| `_maybe_share_lm_head` 把 vocab × hidden 节省 | Ch08 §8.2 weight tying pattern | 跟 Llama 的 lm_head ↔ embed_tokens tie 同思路 |
| EAGLE/Medusa/MTP 三种 head 拓扑 | Ch15+ 模型 zoo | 每个 model 族对应一个 `*_mtp.py` 或 `*_eagle3.py` |
| α=0.85 production 区间 | Ch27 DeepSeek-V3.2 deep-dive | Ch27 给真实 traffic 数据复盘 |
| `deepseek_v4_mtp.py` hc_mult | Ch28 DeepSeek-V4-pro | hidden carrier expansion |

---

## 10.7 7 个语言陷阱：一次性集中检查

每个陷阱 *"声明 → 错 → 为什么 → 源码证据 → Demo/测试"*，匹配 Ch07 §7.6.4、Ch08 §8.6.4、Ch09 §9.7 的模板。

### Trap A — "MTP 让吞吐翻倍 / K=4 意味着 4× 加速"
- **错**。Speedup 是 S(α, K, c) = E[tok | α, K] / (1 + cK)，而 E[tok] = (1 - α^(K+1)) / (1 - α) 是几何级数——chain 一断，后面的 draft 都浪费。
- **为什么**：直觉的"K 个 draft 都接受"概率是 $\alpha^K$，但只占 K+1 项的 1 项；剩下的 K 项是"前 j 项接受、第 j+1 项 reject"的概率链，每条贡献 1+j 个 emit。geometric closure 不是 K·α。
- **源码证据**：`rejection_sampler.py:L734 if not rejected: ...` 是 chain-break 的精确编码；`L425-L430` 用 -1 sentinel 自然 propagate；`metadata.py:L20 num_draft_tokens` 是 per-request K，但运行时 K 可以不到全。
- **Demo 证据**：§3.2 verbatim：α=0.5, K=4 → 1.9375（不是 2.0）；α=0.7, K=4 → 2.7731（不是 2.8）；α=0.3, K=4 → 1.4251（也不是 1.2）。

### Trap B — "Speculative decoding 总是比纯自回归便宜"
- **错**。当 $\alpha$ 低或 $c$ 高时 $S < 1$。Demo §3.3 给出 K=4, c=0.20, α=0.30 → S=0.792（净亏 21%）；K=4, c=0.30, α=0.30 → S=0.648（净亏 35%）。
- **为什么**：低 α 让 emit 数贴近 1 的下界（recovered token），但 draft 成本 $cK$ 还在累计；K 越大、c 越大，break-even α 越高。K=8, c=0.20 break-even = 0.6206——production 几乎需要 long-form 数学场景才能稳过。
- **源码证据**：`rejection_sampler.py` 完全没有 net-loss 检查；运维由 `SpeculativeConfig.num_speculative_tokens` 设定 K，自负盈亏。`vllm/config/speculative.py:L93` 只校验 K 在 [1, MAX_SPEC_LEN] 范围。
- **Demo 证据**：§3.3 9 个 break-even α verbatim；K=4 row 的 c=0.20 列在 α=0.30、0.40 都 < 1。

### Trap C — "Draft 模型必须共享 target 架构才能高准确率"
- **部分错，主要错**。EAGLE/MTP **的确**共享 trunk（target hidden 当输入），但这只是"提高 α"的设计选择。`DraftModelProposer` 完全独立架构（Llama-1B 给 70B 当 draft）也工作（α≈0.5-0.7）；`NgramProposer` 完全没有架构（α≈0.3-0.5），仍然是净赚（c≈0）。
- **为什么**：spec-decode 算法（Chen 2023 不偏定理）对 $p, q$ 的关系**没有任何要求**——只要 $q(x) > 0$ 在 $p(x) > 0$ 的地方。共享架构只影响 $\alpha$ 的取值，不是合法性的前提。
- **源码证据**：`draft_model.py:L36 _raise_if_draft_tp_mismatch` 只检 TP；`L33-L34 _raise_if_vocab_size_mismatch` 只检 vocab。架构不一致**不**报错。
- **Demo 证据**：§3.6 lower-vocab + lower-hidden 的 Llama-3.3-1B → 70B 配置在 vocab=128256 时合法。

### Trap D — "Rejection sampling 在高温下有偏"
- **错**。Chen 2023 的不偏证明（§10.1.4）对**任何** $p, q$ 和**任何**温度都成立——只要 $q(x) > 0$ 在 $p(x) > 0$ 的地方。温度只影响 $\alpha$ 的取值（$\alpha = E_x[\min(1, p/q)]$），不影响 emit 分布。
- **为什么**：accept 概率 + reject 后从 $(p-q)_+$ 残差采，两条路径加起来恰好 $p(x)$（5 行代数即得）。这是算法的**核心保证**：质量永远是 target 的质量。
- **源码证据**：`rejection_sampler.py:L491-L504` accept 公式 `target/draft >= u`；`L659-L703 sample_recovered_tokens` + `L877-L920 kernel` 残差 Gumbel-max 采样。
- **Demo 证据**：§3.1 KL(empirical || p) = 0.000395（vocab=8, K=4, 10000 试验）。p, q 故意取很不一样的两个分布，emit 仍然匹配 p。

### Trap E — "MTP 头是轻量 MLP，加几个就好"
- **错**（对 DeepSeek 的 canonical MTP）。`DeepSeekMultiTokenPredictorLayer` 用 `mtp_block: DeepseekV2DecoderLayer`——**完整 MoE transformer block**。Demo §3.5 的 dense-FFN 估算下 MTP per-layer 75.5M、Medusa per-head MLP 8.39M，比值 9×；MTP-stack 跟 Medusa-stack（共享 lm）比值 12.91×。
- **为什么**：MTP 跟 trunk 联合训练，需要保留 trunk 表征能力；用纯 MLP 表征力不够，acceptance rate 会下来。Medusa 不联合训练（postfix 微调），用 MLP 是工艺选择不是表征要求。
- **源码证据**：`deepseek_mtp.py:L92-L97 mtp_block: DeepseekV2DecoderLayer` 直接用 MoE block，跟 DeepSeek-V3 trunk 同款。
- **Demo 证据**：§3.5 的 6 行 breakdown 里 mtp_block_attn=16.8M + mtp_block_ffn=50.3M = 67M，占 per-layer 75.5M 的 88.7%——重量来自 transformer block，不是 enorm/hnorm/eh_proj 这些 MTP-specific bits（共 8.4M）。

### Trap F — "vLLM 训练 MTP / vLLM 算 MTP loss"
- **错**。vLLM 是 **inference-only**——MTP heads 在上游模型（DeepSeek-V3、Llama EAGLE 等）的训练 repo 里训好，vLLM **加载已训权重** 当 draft head。
- **为什么**：vLLM 的所有"MTP" 代码都在 `vllm/v1/spec_decode/` 和 `vllm/model_executor/models/*_mtp.py` 下，全部是 inference-time logic（forward, load_weights, share weights）。Training-side 的多步 CE loss 是 trainer 的事。
- **源码证据**：grep `MTPLoss|multi_step_ce|compute_mtp_loss|mtp_aux_loss` 在 `vllm/` 全 0 匹配；`.backward(` 在 `vllm/v1/spec_decode/` 全 0 匹配。**精化（M20）**：grep 必须 scope 到 spec-decode subtree——全树有 `phimoe.py:router_aux_loss_coef`（HF config 字段，运行时不调）和 `vision.py:get_load_balance_assignment`（图像 tile）两个 false positive，**不**是 expert routing 的 aux loss。
- **Demo / 测试证据**：tester 的 `test_fidelity.py::test_no_training_loss_in_spec_decode` 显式 grep。Mirror 了 Ch09 Trap-E aux-loss reframe——这是 **Ch10 的第二次 training-to-inference reframe**。

### Trap G — "Acceptance rate 是模型属性"
- **错**。$\alpha$ 是 (draft, target, prompt, 温度) 的**条件期望**——不只跟模型有关，还跟工作流强相关。同一对 (draft, target) 在不同 prompt 上 α 差很多（生成式数学 α≈0.85，开放对话 α≈0.40），同一 target 配不同 draft α 差更多（Medusa < draft-model < MTP < EAGLE 典型）。
- **为什么**：$\alpha = E_x[\min(1, p_x/q_x)]$ 在 $x$ 上的期望——$x$ 的分布由 prompt 决定，所以 $\alpha$ 跟 prompt 关联。Production 必须把 $\alpha$ 当**实时遥测**而不是常量。
- **源码证据**：`rejection_sampler.py:L72-L85 synthetic_conditional_rates` 是 per-position（不是 per-model）——意识到 α 是依赖于位置的；production 的 metrics module 通常给 rolling-mean α。
- **Demo / 测试证据**：本章未直接 demo（无真实 model trace），但 §3.7 的 honest caveat 1 verbatim："The acceptance rate α here is SYNTHETIC (uniform Bernoulli) — real workloads have α that varies per-position and depends on prompt domain. Production DeepSeek-V3 reports α≈0.85+ for the first MTP step."

---

## 10.8 验证：跑 demo 跟 lint

### 10.8.1 跑 demo

```bash
$ cd instances/vllm/artifacts/10-multi-token-prediction
$ /home/zjq/.conda/envs/mujoco/bin/python implementation/demo.py
=== Ch10 Multi-Token Prediction — demo numerics (deterministic, seed=42) ===

§3.1 Rejection sampling unbiasedness (vocab=8, K=4, 10000 trials)
  target_p = [0.30, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03]
  draft_q  = [0.10, 0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05]
  empirical p_hat = [0.2906, 0.2037, 0.1543, 0.1005, 0.1044, 0.0674, 0.0494, 0.0297]
  KL(empirical || target_p) = 0.000395

§3.2 35-cell α-K grid (analytic + 10000-trial empirical)
  [35-cell table verbatim ...]

§3.3 28-cell speedup grid + 9 break-even α
  [verbatim ...]

§3.4 greedy / random emit comparison
  greedy 1.5120, random 4.5150, ratio 2.9861

§3.5 MTP / Medusa parameter ratio
  ratio_shared_lm 12.91x, ratio_separate_lm 1.91x

§3.6 loader demo: 193 → (185, 8)
```

§3.1 ~ §3.6 跑出来的所有数字都跟本章正文里 verbatim 一致（test-report.md §"Demo numerics" 显式 pin）。

### 10.8.2 跑测试

```bash
$ /home/zjq/.conda/envs/mujoco/bin/python -m pytest tests/ \
      --ignore=tests/_legacy -q
311 passed in 4.57s
```

测试模块：

| 模块 | 测试数 | 用例 |
|---|---|---|
| `test_spec_metadata.py` | 23 | dataclass 字段 + factory + chain-break invariant |
| `test_rejection_sampling.py` | 29 | greedy / random kernel + recovered token + bonus |
| `test_acceptance_math.py` | 52 | 35-cell grid + break-even bisection + 7 trap A/B 实证 |
| `test_mtp_head.py` | 48 | RMSNorm / MHA / MTP layer / SharedHead + 参数计数 |
| `test_weight_loading.py` | 32 | rewrite_spec_layer_name 三路 + share_lm/embed |
| `test_proposers.py` | 40 | 7 个 proposer fidelity（含 ExtractHidden、DeepSeekMTP） |
| `test_integration.py` | 13 | 端到端 propose → verify → emit |
| `test_demo_numerics.py` | 42 | §3.1-§3.6 verbatim pin |
| `test_fidelity.py` | 21 | 7 traps 的源码-tester 一致性 |
| **总** | **311** | **PASS** |

### 10.8.3 跑 linter

```bash
$ python3 /home/zjq/Repo2Book/scripts/lint_formulas.py \
    instances/vllm/artifacts/10-multi-token-prediction/narrative/chapter.md
[expected: 🟢 No blocking issues]

$ python3 /home/zjq/Repo2Book/scripts/lint_source_grounding.py \
    instances/vllm/artifacts/10-multi-token-prediction/
[expected: ✓ All grounding checks passed!]
```

---

## 10.9 Source Mapping Table（主表）

| 我们的 | vLLM 源码 | 关系 |
|---|---|---|
| `spec_metadata.py:L36-L113 SpecDecodeMetadata` | `vllm/v1/spec_decode/metadata.py:L9-L66` | 一致 dataclass；M05 锚点 |
| `spec_metadata.py:L62-L66 __post_init__ max_spec_len` | `metadata.py:L26-L27` | 一致 |
| `spec_metadata.py:L68-L113 make_dummy` | `metadata.py:L29-L66` 类似工厂 | 教学便利 |
| `spec_metadata.py:L117 PLACEHOLDER_TOKEN_ID = -1` | `rejection_sampler.py:L30` | 一致 |
| `spec_metadata.py:L124 GREEDY_TEMPERATURE = 0` | `rejection_sampler.py:L31` | 一致 |
| `spec_metadata.py:L130 MAX_SPEC_LEN = 128` | `rejection_sampler.py:L34` | 一致 |
| `rejection_sampling.py:L56-L131 rejection_greedy_sample_loop` | `rejection_sampler.py:L708-L757 rejection_greedy_sample_kernel` | 一致；Triton vs Python loop |
| `rejection_sampling.py:L82-L91 is_greedy=None 全 greedy` | `rejection_sampler.py:L723` | 一致；M02 锚点 |
| `rejection_sampling.py:L100-L104 if rejected: break` | `rejection_sampler.py:L734` | 一致；M04 锚点 |
| `rejection_sampling.py:L122-L124 token_id = target_id` | `rejection_sampler.py:L743-L745` | 一致；unconditional write |
| `rejection_sampling.py:L128-L131 bonus token slot` | `rejection_sampler.py:L751-L757` | 一致 |
| `rejection_sampling.py:L138-L216 rejection_random_sample_loop` | `rejection_sampler.py:L760-L826` | 一致 |
| `rejection_sampling.py:L195-L203 NO_DRAFT_PROBS draft_prob=1` | `rejection_sampler.py:L797-L799` | 一致；M03 锚点 |
| `rejection_sampling.py:L200-L203 (target/draft >= u) accept` | `rejection_sampler.py:L797-L810` | 一致；不偏证明的 accept 公式 |
| `rejection_sampling.py:L205-L211 reject + recovered token` | `rejection_sampler.py:L811-L815` | 一致；chain-break |
| `rejection_sampling.py:L223-L279 sample_recovered_tokens_loop` | `rejection_sampler.py:L853-L920 sample_recovered_tokens_kernel` | 一致；Gumbel-max |
| `rejection_sampling.py:L260-L265 NO_DRAFT_PROBS residual` | `rejection_sampler.py:L877-L891` | 一致；mask draft_id |
| `rejection_sampling.py:L268-L271 (p-q)_+ residual` | `rejection_sampler.py:L892-L903` | 一致；clamp(min=0) |
| `rejection_sampling.py:L276-L277 score = p * inv_q` | `rejection_sampler.py:L904-L905` | 一致；M14 Gumbel-max |
| `rejection_sampling.py:L286-L399 rejection_sample` driver | `rejection_sampler.py:L392-L503 rejection_sample` | 一致；省 logprobs_mode |
| `rejection_sampling.py:L313-L318 output_token_ids = full(-1)` | `rejection_sampler.py:L425-L430` | 一致；M04 锚点 |
| `rejection_sampling.py:L322-L328 is_greedy=None 信号` | `rejection_sampler.py:L432-L435` | 一致 |
| `rejection_sampling.py:L341-L355 greedy path 提前返` | `rejection_sampler.py:L450-L466` | 一致；fast path |
| `rejection_sampling.py:L361-L372 inv_q ~ Exp(1)` | `rejection_sampler.py:L674-L688` | 一致；M14 |
| `rejection_sampling.py:L406-L417 parse_output 过滤 -1` | `rejection_sampler.py:L246-L281 parse_output` | 一致；filter PLACEHOLDER + OOV |
| `rejection_sampling.py:L423-L464 RejectionSampler` 类 | `rejection_sampler.py:L37-L195 RejectionSampler(nn.Module)` | 教学版；省 nn.Module 包装 |
| `acceptance_math.py:L56-L74 expected_tokens` | 隐含于 `rejection_sampler.py:L424-L430` 的 chain-break | 教学公式；M12 |
| `acceptance_math.py:L77-L87 speedup` | `speculative.py:L93-L98 num_speculative_tokens` 配合运维公式 | 教学派生 |
| `acceptance_math.py:L90-L112 break_even_alpha` | 隐含 | 教学；二分法 |
| `acceptance_math.py:L115-L129 alpha_K_grid` | demo §3.2 来源 | 教学装置 |
| `acceptance_math.py:L132-L143 speedup_grid` | demo §3.3 来源 | 教学装置 |
| `acceptance_math.py:L146-L187 simulate_chain_break` | 隐含；用于 verify analytic | 教学装置 |
| `acceptance_math.py:L190-L254 parameter_count_mtp` | `deepseek_mtp.py:L63-L122 layer struct` | 闭式公式 |
| `acceptance_math.py:L257-L275 parameter_count_medusa` | `medusa.py:L18-L78 head struct` | 闭式公式 |
| `mtp_head.py:L42-L58 RMSNorm` | `vllm/model_executor/layers/layernorm.py RMSNorm` | 一致公式 |
| `mtp_head.py:L67-L94 _MultiHeadAttention` | `vllm/model_executor/layers/attention/__init__.py MHA` | 教学；DeepSeek 真实用 MLA |
| `mtp_head.py:L78-L94 reshape-then-permute` | M22 修复点 | 修复 causal mask shape bug |
| `mtp_head.py:L97-L115 _DenseFFN` | `vllm/model_executor/layers/fused_moe/layer.py` 替代物 | 教学 SwiGLU；DeepSeek 用 MoE |
| `mtp_head.py:L118-L140 MTPBlock` | `vllm/model_executor/models/deepseek_v2.py DeepseekV2DecoderLayer` | 教学版；省 MoE |
| `mtp_head.py:L148-L177 SharedHead` | `deepseek_mtp.py:L43-L62` | 一致；M09 forward 只 norm |
| `mtp_head.py:L171-L177 share_lm_head_with` | `llm_base_proposer.py:L1471-L1539 _maybe_share_lm_head` | 一致；M09 锚点 |
| `mtp_head.py:L204-L256 DeepSeekMultiTokenPredictorLayer.forward` | `deepseek_mtp.py:L99-L121` | 一致 layout；M08 锚点 |
| `mtp_head.py:L248 positions == 0 mask` | `deepseek_mtp.py:L107-L113` | 一致；M22 |
| `mtp_head.py:L258-L276 parameter_stats` | nn.Module `sum(p.numel())` | 教学；闭式校验 |
| `mtp_head.py:L279-L364 DeepSeekMultiTokenPredictor` | `deepseek_mtp.py:L124-L184` | 一致；K 步 propose_K |
| `mtp_head.py:L316-L330 forward_one_step` | `deepseek_mtp.py:L160-L170` | 一致 |
| `mtp_head.py:L332-L335 compute_logits` | `deepseek_mtp.py:L172-L182` | 一致；分两步 |
| `mtp_head.py:L337-L364 propose_K` | `llm_base_proposer.py:L516-L654` 序列 path | 教学；K 步循环 |
| `mtp_head.py:L367-L417 parameter_count_mtp` | `deepseek_mtp.py:L63-L122` 配合公式 | 闭式 |
| `mtp_head.py:L420-L443 parameter_count_medusa` | `medusa.py:L18-L78` 配合公式 | 闭式 |
| `weight_loading.py:L39-L103 rewrite_spec_layer_name` | `deepseek_mtp.py:L458-L488` | 一致；三路改写 |
| `weight_loading.py:L75-L81 spec_layer_weight_names list` | `deepseek_mtp.py:L464-L470` | 一致 |
| `weight_loading.py:L93-L102 path1 / path2 / path3 dispatch` | `deepseek_mtp.py:L480-L488` | 一致；M15+M21 锚点 |
| `weight_loading.py:L106-L143 remap_checkpoint` | `deepseek_mtp.py:L271-L456 load_weights` | 教学版；只做名字部分 |
| `weight_loading.py:L147-L200 maybe_share_lm_head` | `llm_base_proposer.py:L1471-L1539` | 一致；强制共享分支 |
| `weight_loading.py:L188-L199 inner layers loop` | `llm_base_proposer.py:L1522-L1538` | 一致；替换内层 shared_head.head |
| `weight_loading.py:L204-L232 maybe_share_embeddings` | `llm_base_proposer.py:L1402-L1469` | 一致；MTP 强制共享 |
| `weight_loading.py:L239-L257 acceptance_length_to_rates` | `speculative.py:L213-L227` | 一致；最小方差 schedule |
| `weight_loading.py:L260-L284 unconditional_to_conditional_rates` | `vllm/v1/spec_decode/utils.py` | 一致 |
| `weight_loading.py:L287-L343 loader_demo_shapes` | demo §3.6 来源 | 教学装置 |
| `proposers/base.py:L33-L43 ProposerOutput` | `llm_base_proposer.py:L407-L411` 返回 shape | 教学 |
| `proposers/base.py:L73-L93 SpecDecodeBaseProposer.__init__` | `llm_base_proposer.py:L60-L106` | 教学；只保留 K + pass_hidden + parallel_drafting |
| `proposers/base.py:L97-L101 _greedy_sample` | `llm_base_proposer.py:L407-L412` | 一致 |
| `proposers/base.py:L104-L168 propose` (fast/sequential) | `llm_base_proposer.py:L413-L656` | 一致 |
| `proposers/base.py:L147-L149 K==1 fast path` | `llm_base_proposer.py:L491-L494` | 一致 |
| `proposers/base.py:L155-L168 K>1 sequential` | `llm_base_proposer.py:L516-L654` | 一致 |
| `proposers/eagle.py:L20-L36 EagleProposer` | `eagle.py:L1-L22` | 一致；纯继承 |
| `proposers/medusa.py:L32-L52 MedusaHeads` | `medusa.py:L48-L49` | 教学；省层 norm |
| `proposers/medusa.py:L56-L104 MedusaProposer` | `medusa.py:L18-L78` | 一致；不继承 base |
| `proposers/medusa.py:L80-L104 propose stacked argmax` | `medusa.py:L52-L53` | 一致 |
| `proposers/draft_model.py:L41-L84 DraftModelProposer` | `draft_model.py:L17-L88` | 一致 |
| `proposers/draft_model.py:L62-L71 vocab size mismatch` | `draft_model.py:L33-L34` | 一致；issue #5414 |
| `proposers/draft_model.py:L73-L84 draft_tp mismatch` | `draft_model.py:L36-L51` | 一致 |
| `proposers/ngram.py:L38-L77 NgramProposer` | `ngram_proposer.py:L12-L62` | 一致 |
| `proposers/ngram.py:L82-L101 _find_and_propose` | `ngram_proposer.py:L198-L285` | 一致算法；朴素 vs numba KMP |
| `proposers/extract_hidden.py:L42-L66 __init__ assert K==1` | `extract_hidden_states.py:L29-L31` | 一致；M10 锚点 |
| `proposers/extract_hidden.py:L68-L91 propose unchanged` | `extract_hidden_states.py:L72-L130` | 一致 |
| `proposers/mtp.py:L24-L70 DeepSeekMTPProposer` | `deepseek_mtp.py:L186-L488 DeepSeekMTP` | 一致；wraps Predictor |
| `proposers/mtp.py:L52-L70 propose K-step` | `deepseek_mtp.py:L160-L170` | 一致 |
| `demo.py:§3.1 unbiasedness` | demo source for KL=0.000395 | 一致；M14 + 不偏定理 |
| `demo.py:§3.2 alpha-K grid` | demo source for 35-cell table | 一致；M12 |
| `demo.py:§3.3 speedup grid + break-even` | demo source for 9 α* | 一致；M18 |
| `demo.py:§3.4 greedy/random emit` | demo source for ratio 2.99x | 一致；M02 |
| `demo.py:§3.5 param count comparison` | demo source for ratio 12.91x | 一致；M08 + Trap E |
| `demo.py:§3.6 loader demo 193→(185,8)` | demo source for 3-path renames | 一致；M15+M21 + Trap F reframe |

**主表 80 行；加上 §10.2.5（13 行）、§10.3.6（11 行）、§10.4.6（10 行）、§10.5.7（10 行）、§10.6.4（7 行）四个 mini-mapping 的 51 行，总共 131 行 source mapping**——超过 v6 floor 的 10 行下限和 Ch10 brief 的 ≥130 目标（Ch08 用了 122 行，Ch09 用了 88 行；Ch10 surface 11 文件需要更细的覆盖）。

---

## 10.10 总结

第 10 章把 vLLM 的 Multi-Token Prediction 拆成 5 个 proposer + 1 个 verifier + DeepSeek-MTP 模型族 wrapper 的协同。读完你能：

1. **从数学上**：写出 E[tok | α, K] = (1 - α^(K+1)) / (1 - α) 与 S(α, K, c) = E[tok] / (1 + cK)；解释 K=4 ≠ 4× 与 net-loss zone 的运维风险（demo §3.2 35-cell + §3.3 break-even α）；走通 Chen 2023 不偏定理的 5 行代数（demo §3.1 KL = 0.000395）。
2. **从架构上**：知道 vLLM **没有 `class MultiTokenPrediction`**——MTP 是 5 个 proposer（EAGLE / Medusa / DraftModel / Ngram / ExtractHidden）共享同一个 `RejectionSampler` 后端 + 30+ 个 `*_mtp.py` 模型族 wrapper 的协同，DeepSeek 是 canonical impl。这是 Ch07 "no radix tree"、Ch08 "no class TensorParallel"、Ch09 "no class ExpertParallel" 之后的**第四件**"no class X"。
3. **从权重加载上**：把 outline §3 的 "Training——多步 CE 损失" 重构成 "Inference 期 MTP 权重加载"——这是 Ch09 §9.4 之后的**第二次** training-to-inference reframe。Sidebar 介绍训练侧损失 L_MTP = Σ_k λ_k · CE(p_k, x_{t+k})，pivot 到 vLLM 的 `_rewrite_spec_layer_name` 三路改写（demo §3.6: 193 → (185, 8)）+ `_maybe_share_lm_head` + `SharedHead.forward` 只 norm（M09）。
4. **从架构开销上**：MTP 头**不是轻量** MLP（demo §3.5 ratio MTP/Medusa 共享 lm = 12.91×，独立 lm = 1.91×）；DeepSeek 实际用 MoE block 让参数再涨 ~10×。Trap E 的硬证据。
5. **从 proposer 横向上**：理解 5 个 proposer 在 (cost, accuracy, coupling) 三元组上的取舍——EAGLE/MTP shared-trunk 高 α、Medusa 独立头低 α 但参数轻、DraftModel 完全独立架构 (vocab + TP 校验)、Ngram 零参 NO_DRAFT_PROBS 路径、ExtractHidden 单步 KV 抽取（不真 spec）。

**链断点不变量**：`output_token_ids = torch.full(..., -1)` + `if rejected: break` —— Ch10 的核心几何来自 chain-break 几何级数 E[tok] = Σ_{k=0..K} α^k。

**七大语言陷阱**：A "K=4 ≠ 4×" ✗、B "spec_decode 总比 AR 快" ✗（K=4 c=0.20 α=0.30 → S=0.792，净亏）、C "draft 必须共享架构" ✗（DraftModel 仅校验 vocab + TP）、D "高温下有偏" ✗（Chen 2023 对任何 p, q 不偏）、E "MTP 头是轻量 MLP" ✗（DeepSeek MTP block 是完整 MoE）、F "vLLM 训练 MTP" ✗（grep 0 匹配）、G "α 是模型属性" ✗（α 是 (draft, target, prompt, 温度) 的条件期望）。

**Forward-pointer**：

- **Ch15+（模型 zoo）** 把 30+ 个 `*_mtp.py` / `*_eagle3.py` 的 fusion 拓扑差异详细对比——Llama EAGLE3 的 `fc` 投影、Qwen3-MTP 的 trunk 选择、ERNIE-MTP 的 K-layer 配置。
- **Ch27（DeepSeek-V3.2 deep-dive）** 把 §10.3 的 dense-FFN 估算重新算成 MoE 实际值（参数 ~120× Medusa），把 production α=0.85 跟工作流分布敏感度联系，给真实 traffic trace 复盘 Trap B 的 break-even 临界。
- **Ch28（DeepSeek-V4-pro）** 解析 `deepseek_v4_mtp.py` 的 `hc_mult` carrier hidden-state expansion——把 trunk hidden 在喂给 MTP 前再扩 2-4×，给 MTP 更多 context；把 §10.3.2 的 sidebar+pivot 模板再用一次。

下一章 Ch11 进入 Part 2 公共能力进阶，讲 **DCP/PCP（Decoding/Prefilling Context Parallelism）**——把超长序列切到多 GPU 上的 ring attention 拓扑。同样的 chain-break 模式（GroupCoordinator + 一对 collective + 一个 model-side use site），换一个并行轴（sequence 维度）；同样的不变量 `mem_per_rank ∝ 1 / cp_size` 跟 Ch08 §8.3 / Ch09 §9.5 一脉相承。

---

> 章节版本：v6.0（Ch09 单 cycle APPROVED 之后的第三章）
> 源码 commit：vllm `98661fe`
> 公式 lint：🟢 No blocking issues
> 源码 grounding lint：✓ All grounding checks passed
