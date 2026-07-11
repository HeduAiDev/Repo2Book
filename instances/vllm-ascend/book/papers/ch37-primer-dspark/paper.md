# DSpark：半自回归投机解码——并行骨干 + 序列 Markov 头（前瞻）

> **本章为前瞻 primer**：DSpark 尚未合入 vllm-ascend（RFC [#11126](https://github.com/vllm-project/vllm-ascend/issues/11126) "[RFC]: Add DSpark speculative decoding support for DeepSeek-V4"），本包据上游 vLLM 主线 PR [#46995](https://github.com/vllm-project/vllm/pull/46995)（"[Spec Decode] DSpark"，已合入 vLLM 主线，MERGED 2026-07-01，merge commit `f5a8d73377d0f0a4e00cba172f9fbd0d50471b07`）与 DeepSeek 相关技术报告/评测整理。凡引用上游代码处均标注「来自 vLLM 主线 PR #46995 @f5a8d73，尚未合入本书 pin 的 v0.21.0——前瞻解读」；数字表全部标来源、未独立复现。

## 来源说明

本包综合三处公开材料，无单一 arXiv 论文可锚（DeepSeek 尚未为 DSpark 发布独立 arXiv 报告，机制描述分散在技术博客、上游 lineage 博客与 vllm-ascend 的 RFC issue 里）：

1. **机制最全**——ai-infrastructure.net 技术摘要（半自回归架构、Markov 头低秩分解、置信度头公式、Algorithm 1 调度、DeepSeek-V4 生产数字）：<https://ai-infrastructure.net/dspark-speculative-decoding/>
2. **DFlash 谱系**——LMSYS 博客，DSpark 的并行骨干（context-KV 预计算 + 非因果 query-block 前向）直接继承自 DFlash：<https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/>
3. **vllm-ascend RFC**——`AscendDsparkProposer` / Markov 头 / confidence 头 / checkpoint 的落地提案：<https://github.com/vllm-project/vllm-ascend/issues/11126>

外部上游代码快照见 `instances/vllm-ascend/book/external-source/dspark-pr46995/`（`PROVENANCE.md` 锁定来源仓/PR/commit/拉取日），本章数学推导与之逐点对应。

## 一、动机与总览：为什么是"半自回归"

投机解码的草稿模型有两条路数，各有一个硬伤：

- **纯序列（EAGLE / MTP）**：草稿逐 token 自回归采样，块内依赖精确，但 γ 个草稿 token 要 γ 次串行前向——小模型也躲不开延迟的"骨牌效应"。
- **纯并行（DFlash）**：把上下文 KV 一次性注入草稿模型的 KV cache（"跳过从零建模完整上下文"），一次非因果前向出整块草稿——硬件友好，但代价是块内 token 之间的条件依赖被丢弃：块内第 `k` 个位置的草稿分布不再以"块内前 `k-1` 个已采样草稿 token"为条件，只以"目标模型的上下文表示"为条件。

DSpark 的选择：**复用 DFlash 的并行骨干做"重"计算，只在骨干之上补一个极轻的序列头找回块内依赖**——重的部分（一整层 Transformer 解码器栈）保持一次非因果并行前向；轻的部分（块内 `N` 步的从左到右修正）只需一次向量内积规模的低秩偏置计算，不需要重跑骨干。这就是"半自回归"名字的来源：宏观并行、微观（块内）序列。

## 二、并行骨干：一次非因果前向出一整块

DSpark 的并行骨干直接复用 DFlash 的 Qwen3 解码器栈（`Qwen3DSparkModel` 继承 `DFlashQwen3Model`），未做架构改动；DeepSeek-V4 版本（`DSparkDeepseekV4Model`）额外复用了目标模型的超连接 MLA 解码层。骨干输出块内每个位置的 hidden state，经 `compute_logits` 得到该位置的**基础 logits** `U_k`（下标 `k` 记块内位置，`k = 0, ..., N-1`）：

$$
U_k = \mathrm{lm\_head}(\mathrm{norm}(h_k))
$$

其中 `h_k` 是骨干在块内位置 `k` 的输出 hidden state（Qwen3DSparkModel 用 `DFlashQwen3ForCausalLM.compute_logits`；DeepSeek-V4 版本用 `DSparkDeepseekV4ForCausalLM.compute_logits`，见下方源码，`self.model.norm` 是 head 前的最终归一化，与骨干内部逐层 norm 是两回事）。

```python
# 来自 vLLM 主线 PR #46995 @f5a8d73，尚未合入本书 pin 的 v0.21.0——前瞻解读
# 见 external-source/dspark-pr46995/vllm_models_deepseek_v4_nvidia_dspark.py
def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Base logits U_k = lm_head(norm(head_hidden))."""
    return self.logits_processor(self.lm_head, self.model.norm(hidden_states))
```

关键的架构细节是**"锚点即首个预测位"**（anchor-as-first-prediction）。DFlash 的输入是 `1 + N` 个 query token（第 0 个是"锚点"，只提供上下文，不产生预测；后 `N` 个才各自预测一个草稿 token）。DSpark 改成恰好 `N` 个 query token（锚点 + `N-1` 个占位噪声 token），**每个 query 位置都是一次预测**——锚点自己预测块内第一个草稿 token：

```python
# 来自 vLLM 主线 PR #46995 @f5a8d73，尚未合入本书 pin 的 v0.21.0——前瞻解读
# 见 external-source/dspark-pr46995/vllm_v1_worker_gpu_spec_decode_dspark_speculator.py
class DSparkSpeculator(DFlashSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # Anchor-first: N query tokens per request (anchor + N-1 noise), not 1+N.
        self.num_query_per_req = self.num_speculative_steps
        ...
        self.dflash_causal = False
        # The anchor query position is itself a prediction (see module docstring).
        self.sample_from_anchor = True
```

`self.dflash_causal = False` 对应到注意力后端是**非因果**索引：块内每个 query 位置除了看滑窗内的历史上下文，还要看**块内其他所有 query 位置（含未来位置）**——这是"并行骨干一次出整块"在注意力层面的实现。`vllm_v1_attention_backends_mla_sparse_swa.py` 里的 `_compute_dspark_noncausal_swa_indices_kernel` 专门为此新增了一条非因果索引路径（`is_dspark` 分支），把窗口宽度从"仅历史 `window_size`"扩到"历史窗口 + 整个草稿块"：

```python
# 来自 vLLM 主线 PR #46995 @f5a8d73，尚未合入本书 pin 的 v0.21.0——前瞻解读
# 见 external-source/dspark-pr46995/vllm_v1_attention_backends_mla_sparse_swa.py
self.is_dspark = spec_config is not None and spec_config.use_dspark()
self.noncausal_index_width = (
    cdiv(self.window_size + self.num_speculative_tokens, 128) * 128
    if self.is_dspark
    else 0
)
```

这一步产出的 `U_k` 就是"块内位置 `k` 只看得到上下文、看不到块内其他位置**已采样的草稿 token**"的基础分布——它天然独立于块内相邻位置的采样结果，块内依赖要靠第三节的序列头找回来。

## 三、序列 Markov 头：低秩转移偏置里的精确 softmax

DSpark 不重跑骨干来补依赖，而是给基础 logits 加一个**只依赖"前一个已采样 token"的转移偏置**——即一阶 Markov 假设：块内位置 `k` 的草稿分布，只以位置 `k-1` 采样出的具体 token `x_{k-1}` 为条件，不看更早的块内 token：

$$
p_k(v \mid x_0, x_{<k}) = \mathrm{softmax}\big(U_k(v) + B_k(x_{k-1}, v)\big)
$$

其中 `B_k` 是转移偏置，因为要对每个（前驱 token，候选 token）配对存一个数，朴素实现是一张 `V x V` 矩阵——词表 `V` 常有数万到十万量级，`V x V` 存不起也学不动。`DSparkMarkovHead` 把它低秩分解成两段：

```python
# 来自 vLLM 主线 PR #46995 @f5a8d73，尚未合入本书 pin 的 v0.21.0——前瞻解读
# 见 external-source/dspark-pr46995/vllm_model_executor_models_qwen3_dspark.py
class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V).

    ``markov_w1[token]`` is an r-dim embedding of the previously sampled token;
    ``markov_w2`` projects it back to a vocab-size bias added to the base logits.
    """

    def __init__(self, vocab_size: int, markov_rank: int, prefix: str) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w1")
        )
        self.markov_w2 = ParallelLMHead(
            vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w2")
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of ``token_ids`` ([B] -> [B, r])."""
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        return logits_processor(self.markov_w2, markov_embed)
```

`markov_w1` 是 `V x r` 的嵌入表（rank `r = 256`，即 ai-infrastructure.net 摘要给出的 rank 数值），把前驱 token 映到一个 `r` 维向量 `e = W1[x_{k-1}]`；`markov_w2` 是 `V x r` 的另一张表，当作 lm-head 用，把 `e` 投回词表维度。写成矩阵形式：

$$
B(v, x') = \big(W_2\, W_1[x']^\top\big)_v
$$

也就是等价于学一张秩 `\le r` 的 `V x V` 转移矩阵 `B = W_2 W_1^\top`，但存储和计算量都只有 `O(V r)`，不是 `O(V^2)`——这正是"低秩 V×r / r×V 转移偏置"的字面含义（用户摘要里的 `B = W1 ⊗ W2`，在此实现里精确对应 `W1: V×r` 嵌入表与 `W2: V×r` 权重表的秩-`r` 乘积）。

**为什么这个偏置不破坏投机采样的精确性**：注意 `p_k` 依然是对**加了偏置之后的完整 logits** 做一次标准 `softmax`——它仍然是词表上的一个合法的、逐点显式的概率分布，不是某种近似或采样内插。既然 `p_k(v)` 和目标模型的验证分布 `q_k(v)` 都是词表上的合法分布，接受率判据 `\min(1, p_k(v)/q_k(v))` 的比值逐点良定义，下游的拒绝采样验证器不需要为 DSpark 做任何特殊改动——这也是骨干本身丢弃块内依赖、但仍能用一个廉价加性偏置"精确"找回一部分依赖的关键：偏置只改变了 softmax 之前的分数，不改变"过完 softmax 得到合法分布"这件事。

块内采样是一个 `for` 循环，从锚点的采样结果开始，逐步把上一步采样到的 token 喂给 Markov 头算偏置：

```python
# 来自 vLLM 主线 PR #46995 @f5a8d73，尚未合入本书 pin 的 v0.21.0——前瞻解读
# 见 external-source/dspark-pr46995/vllm_v1_worker_gpu_spec_decode_dspark_speculator.py
def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
    n_spec = self.num_speculative_steps
    ...
    base_logits = self.model.compute_logits(sample_hidden)
    ...
    # Anchor (bonus) token per request = the input id at query offset 0.
    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

    for i in range(n_spec):
        # Sequential stage: Markov bias from the previously sampled token.
        markov_embed = self.model.markov_embed(prev)
        bias = self.model.markov_bias(markov_embed)
        logits_i = base_logits[:, i] + bias
        ...
        draft_i = gumbel_sample(logits_i, ...)
        self.draft_tokens[:num_reqs, i] = draft_i
        prev = draft_i
```

这个循环里唯一"重复计算"的是 `markov_embed` / `markov_bias`（各自 `O(r)` / `O(Vr)`），骨干的一整层 Transformer 前向只在第二节做过一次——这就是"半自回归"省下的那部分算力：把 `N` 次完整前向压成 1 次完整前向 + `N` 次向量级修正。

## 四、置信度头与温度校准（论文机制；本 PR 快照未接入推理路径）

ai-infrastructure.net 摘要描述的置信度头，用于**在验证之前预测每个草稿位置的存活概率**，公式为：

$$
c_k = \mathrm{sigmoid}\big(w^\top [\,h_k\,;\, W_1[x_{k-1}]\,]\big)
$$

即把骨干 hidden state `h_k` 与该位置的 Markov 嵌入 `W_1[x_{k-1}]` 拼接后过一个线性层加 sigmoid，逼近**解析接受率**：

$$
c_k^{*} = 1 - \mathrm{TV}(p_{\mathrm{draft}}, p_{\mathrm{target}})
$$

（`TV` 是全变差距离，取归一化定义：逐点差绝对值和的一半，值域 `[0,1]`。这与第三节"接受率 `\min(1,p/q)` 的期望等于 `1 - D_{LK}(p,q)`"不是同族近亲而是**同一个恒等式**——在此归一化下 `\mathrm{TV}(p,q)` 与 `D_{LK}(p,q)=\sum_v|p(v)-m(v)|`（`m=(p+q)/2`）逐点严格相等，故 `c_k^* = 1 - \mathrm{TV} = 1 - D_{LK}`，即 `\beta`。反例 `p=(1,0)`、`q=(0,1)`：`TV=1`、`c_k^*=0`，与真实接受率 `\sum\min(p,q)=0` 吻合；若误带 `1/2` 会算出 `0.5`）。论文侧还提到 **Sequential Temperature Scaling（STS）**：用一维网格搜索校准 `c_k`，把期望校准误差（ECE）从 3–8% 压到约 1%，使得"累计存活概率"（连乘 `c_i`）在数值上可信，能真正用来做第五节的调度判据。

**代码现状的诚实标注**：本包所据的 PR #46995 快照里，`confidence_head` 的权重被**显式跳过**，尚未接入前向/采样路径：

```python
# 来自 vLLM 主线 PR #46995 @f5a8d73，尚未合入本书 pin 的 v0.21.0——前瞻解读
# 见 external-source/dspark-pr46995/vllm_model_executor_models_qwen3_dspark.py
# mask_embedding is an unused placeholder param; DSpark masks via the vocab row.
# confidence_head is not wired into inference yet; skip its weights.
skip_substrs = ["mask_embedding", "confidence_head"]
```

```python
# 来自 vLLM 主线 PR #46995 @f5a8d73，尚未合入本书 pin 的 v0.21.0——前瞻解读
# 见 external-source/dspark-pr46995/vllm_models_deepseek_v4_nvidia_dspark.py
# The confidence head is not wired into inference yet; drop its weights.
if rest.startswith("confidence_head."):
    return None
```

也就是说：**checkpoint 里确实存在 `confidence_head.*` 权重（说明训练侧已经产出这个头），但当前这版 vLLM 主线代码只加载并丢弃它，第四节的公式目前停留在"论文/checkpoint 有、上游推理代码暂未使用"的阶段**。这不是本包的推测，而是逐字引用的源码注释——按前瞻 primer 纪律，这里如实标注，不把"论文机制"包装成"已落地机制"。

## 五、硬件感知动态调度 Algorithm 1（论文机制；本 PR 快照未包含调度器代码）

有了第四节的逐位存活概率，DeepSeek 报告描述了一个**贪心动态调度器**，目标是最大化吞吐量：

$$
\Theta = \tau \cdot \mathrm{SPS}(B)
$$

其中 `τ` 是（给定批大小 `B` 下）期望被接受的 token 数，`SPS(B)`（steps-per-second）是针对批大小 `B` 离线 profile 好的、可 `O(1)` 查表的硬件吞吐曲线。调度过程（论文 Algorithm 1，按摘要复述）：

1. 把候选草稿 token 按"累计前缀存活概率"（即 `\prod_{i \le k} c_i`，块内从锚点往后逐位相乘）排序；
2. 逐个贪心录取，每录取一个 token 就更新有效批大小 `B` 与期望 `τ`；
3. 每一步都用 `SPS(B)` 查表评估吞吐量是否提升；
4. **一旦吞吐量不再提升就立即停止**——这是保证"不破坏因果"的早停闸门。

第 4 步的早停不是性能上的权宜之计，而是正确性要求：论文附录 A 给出一个反例，如果允许"回看全局"式的搜索（而不是贪心、单调、一旦劣化立刻停），会把一个 `(0.7, 0.3)` 的目标分布搜偏成 `(0.85, 0.15)`——即"提前窥视未来验证结果再决定接受多少"会系统性偏置输出分布。贪心早停等价于"只用已经产生的、单调不减的信息做决策，不回溯",这与拒绝采样"不能用目标模型的验证结果反过来影响草稿采样"的因果约束是一回事。

**代码现状的诚实标注**：本包所据的 PR #46995 快照里没有实现这套基于置信度的自适应调度器——`DSparkSpeculator` 每步固定生产 `self.num_speculative_steps`（即配置里的 `num_speculative_tokens`）个草稿 token，循环体（第三节的 `for i in range(n_spec)`）没有提前退出的分支。上游现有的、粒度最接近的"动态"机制是 `SpeculativeConfig.num_speculative_tokens_per_batch_size`——一个按**批大小区间**（而非逐请求置信度）查表选定草稿长度的粗粒度调度：

```python
# 来自 vLLM 主线 PR #46995 @f5a8d73，尚未合入本书 pin 的 v0.21.0——前瞻解读
# 见 external-source/dspark-pr46995/vllm_config_speculative.py
num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None
"""Batch-size schedule used to dynamically choose speculative-token count.

Each entry is ``(range_start, range_end, num_speculative_tokens)`` with an
inclusive batch-size range.
"""
```

这与 Algorithm 1"逐请求、逐位置、按累计存活概率贪心录取"是两个不同粒度的机制：前者是部署时的静态查表（对所有请求生效同一个草稿长度），后者是运行时按每个请求的实时置信度动态决定录取多少——第五节描述的是后者，代码里目前只有前者的雏形。

## 六、数字表（据来源，未独立复现）

以下数字均引自 ai-infrastructure.net 摘要与 LMSYS DFlash 博客，本书未接触 DeepSeek-V4 生产环境或对应硬件复现，仅作为"论文/厂方报告的数量级参考"呈现。

**DeepSeek-V4 生产部署（据 ai-infrastructure.net 摘要）**：

| 目标模型 | 匹配吞吐下的单用户加速 | 中等 SLA 下的聚合吞吐提升 |
|---|---|---|
| V4-Flash | 60–85% | +51%（SLA 80 tok/s/user） |
| V4-Pro | 57–78% | +52%（SLA 35 tok/s/user） |

**离线基准（Qwen3 4B/8B/14B 目标模型，据同一摘要）**：

| 对比对象 | 宏平均接受长度提升 |
|---|---|
| vs. EAGLE-3 | +30.9% / +26.7% / +30.0%（4B/8B/14B） |
| vs. DFlash | +16.3% / +18.4% / +18.3%（4B/8B/14B） |

**逐位置条件接受率（Qwen3-4B，Math 任务，位置 1，据同一摘要）**：DSpark 0.93，DFlash 0.88，EAGLE-3 0.81；Chat 任务位置 1 接受率在三种方法间落在 0.72–0.93 区间（摘要未给出逐方法拆分数值，仅给区间）。

**DFlash 自身数字（据 LMSYS 博客，作为 DSpark 并行骨干的血缘参照，非 DSpark 本身数字）**：

| 任务 | EAGLE-3（5 层） | DFlash |
|---|---|---|
| GSM8K | 接受长度 4.2 / 加速 2.1x | 接受长度 4.2 / 加速 **3.3x** |
| HumanEval | 接受长度 4.3 / 加速 2.2x | 接受长度 4.0 / 加速 **3.2x** |
| MT-Bench | 接受长度 3.1 / 加速 1.4x | 接受长度 3.0 / 加速 **2.2x** |

以上所有数字未独立复现，来源与抓取日期见本包 `meta.json` 的 `papers` 字段。

## 七、术语与符号表

| 符号/术语 | 含义 | 对应代码 |
|---|---|---|
| `N` / `num_speculative_tokens` | 每块草稿 token 数（= query 数，锚点+N-1 噪声） | `DSparkSpeculator.num_query_per_req` |
| `U_k` | 位置 `k` 的基础 logits（并行骨干输出，不含块内依赖） | `compute_logits` |
| `B_k` / `B(v, x')` | 转移偏置，一阶 Markov，低秩 `V x r` / `r x V` 分解 | `DSparkMarkovHead.bias` |
| `r` / `markov_rank` | Markov 头低秩维度（摘要给出数值 256） | `DSparkMarkovHead.__init__` |
| `W1`（`markov_w1`） | `V x r` 前驱 token 嵌入表 | `VocabParallelEmbedding` |
| `W2`（`markov_w2`） | `V x r` 投影表，充当伪 lm-head | `ParallelLMHead` |
| `c_k` | 置信度头输出，位置 `k` 的预测存活概率（**PR 快照未接入**） | `confidence_head`（权重被跳过） |
| `Θ` | 调度器优化目标：吞吐量 | 论文 Algorithm 1（**PR 快照未实现**） |
| `SPS(B)` | 批大小 `B` 下 profile 好的每秒步数 | 同上 |
| `dflash_causal` / `is_dspark` | 非因果注意力开关，块内位置互相可见 | `DSparkSpeculator` / `DeepseekSparseSWAMetadataBuilder` |

## 附：与 external-source 的对应关系一览

| 本章机制 | external-source 文件 |
|---|---|
| 并行骨干继承 DFlash Qwen3 栈 + Markov 头挂载 | `vllm_model_executor_models_qwen3_dspark.py` |
| DeepSeek-V4 版并行骨干（超连接 MLA 解码层）+ Markov 头 + 权重映射 | `vllm_models_deepseek_v4_nvidia_dspark.py` |
| 序列采样循环 / 锚点即首预测位 / CUDA graph 捕获整个草稿步 | `vllm_v1_worker_gpu_spec_decode_dspark_speculator.py` |
| 草稿模型加载/权重别名（与目标模型共享 embed/lm_head） | `vllm_v1_worker_gpu_spec_decode_dspark_utils.py` |
| 非因果滑窗索引 kernel（块内位置互相可见） | `vllm_v1_attention_backends_mla_sparse_swa.py` |
| `method="dspark"` 的配置分支 / 批大小动态草稿长度 | `vllm_config_speculative.py` |
