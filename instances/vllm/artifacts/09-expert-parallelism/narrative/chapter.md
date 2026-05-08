# 第9章：Expert Parallelism — 没有 `class ExpertParallel` 的专家并行

> 本章涉及的 vLLM 源码（commit `98661fe`）：
> - `instances/vllm/source/vllm/distributed/parallel_state.py:L1261-L1283`（`_EP` / `_EPLB` 模块级单例 + `get_ep_group()` / `get_eplb_group()` accessor，dense 模型时 `_EP is None`）+ `L1670-L1696`（EP group 构造，**只在 `model_config.is_moe` 时建组**，mesh 公式 `all_ranks.transpose(1, 2).reshape(-1, dp × pcp × tp).unbind(0)`）+ `L1700-L1719`（EPLB **单独的进程组**，源码注释 *"to prevent deadlocks"*）+ `L1797-L1801`（DeepEP buffer hook）+ `L1891-L1896`（EP group teardown）
> - `instances/vllm/source/vllm/model_executor/layers/fused_moe/layer.py:L70-L157`（`determine_expert_map(ep_size, ep_rank, global_num_experts, expert_placement_strategy)` — 返回 `(local_num_experts, expert_map, expert_mask)`，`expert_map[i] = -1` 是 off-rank 哨兵）+ `L160-L193`（`determine_expert_placement_strategy` — `linear` vs `round_robin`，4 个 fallback 条件）+ `L196-L214`（`get_compressed_expert_map` 日志 helper）+ `L219-L290`（`class FusedMoE(PluggableLayer)` 的声明，**这是组合层，不是 EP 算法**）+ `L290-L605`（`__init__`：`FusedMoEParallelConfig.make`、quant_method、router、`expert_map` 构造）+ `L548-L557`（EPLB 与 quant method 的 `supports_eplb` 检查）+ `L660-L685`（`@property ep_size` / `use_ep` / `ep_rank`）+ `L1543-L1649`（`forward` — quant_method.apply → prepare_finalize → dispatch/local FFN/combine）
> - `instances/vllm/source/vllm/model_executor/layers/fused_moe/config.py:L998-L1209`（`class FusedMoEParallelConfig` dataclass + `make()`：**EP 把 TP 收缩到 `tp_size=1`** 的 collapse 规则在 `L1192-L1208`；`use_all2all_kernels = dp_size > 1 and use_ep` 在 `L1019-L1020`）+ `L1077`（`flatten_tp_size = dp × pcp × tp`）
> - `instances/vllm/source/vllm/distributed/device_communicators/all2all.py:L40-L139`（`class AgRsAll2AllManager` — 教学 baseline backend：**`dispatch = all_gatherv`，`combine = reduce_scatterv`**，不是真正的对称 all-to-all）+ `L142-L195`（`DeepEPAll2AllManagerBase` 公共基类）+ `L196-L256`（`DeepEPHTAll2AllManager` 高吞吐跨节点）+ `L257-L325`（`DeepEPLLAll2AllManager` 低延迟跨节点）+ `L327-L440`（`NixlEPAll2AllManager` CPU staged）+ `L442-L670`（FlashInfer NVLink 单/双边）+ `L671+`（`MoriAll2AllManager`）
> - `instances/vllm/source/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L69-L113`（`def fused_topk(...)` — Mixtral 路径：**先 softmax 后 topk**，`L94-L100` 是 trap-G 关键行）+ `L116-L167`（`class FusedTopKRouter`）
> - `instances/vllm/source/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L81-L162`（`def grouped_topk(...)` — DeepSeek 路径：先按 group_score 选 `topk_group` 组，再在组内 topk，`e_score_correction_bias` 是 V3 noaux_tc 的偏置）+ `L247-L353`（`class GroupedTopKRouter`）
> - `instances/vllm/source/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L71-L168`（`MoEPrepareAndFinalizeNaiveDPEPModular` — dispatch → expert exec → combine 的 wiring，`L125-L168 .apply()` 是 5-step 锚点）
> - `instances/vllm/source/vllm/distributed/eplb/eplb_state.py:L62-L210`（`EplbStats` / `EplbModelState`）+ `L210-L920`（`class EplbState` — **运行期统计 rebalance**，1166 行总长，本章只取它的 public 接口）+ `L925-L944`（`class EplbLayerState`：per-FusedMoE 层的 `expert_load_view`）
> - `instances/vllm/source/vllm/model_executor/models/mixtral.py:L77-L154`（`class MixtralMoE`：reference 现场 #1 — `gate=ReplicatedLinear`，E=8 K=2，无 shared expert）
> - `instances/vllm/source/vllm/model_executor/models/deepseek_v2.py:L244-L386`（`class DeepseekV2MoE`：reference 现场 #2 — `GateLinear` + 可选 `e_score_correction_bias`，grouped Top-K，`shared_experts` 在 FusedMoE 之外构造）
> - 对照实现 `instances/vllm/artifacts/09-expert-parallelism/implementation/`：`routing.py`、`expert_map.py`、`ep_groups.py`、`all2all_baseline.py`、`fused_moe_block.py`、`eplb.py`、`mixtral_vs_deepseek.py`、`demo.py`
>
> 第 7 章用"vLLM 没有 radix tree"开篇，第 8 章用"vLLM 没有 `class TensorParallel`"开篇，第 9 章是这条系列的第三件——**vLLM 没有 `class ExpertParallel`，也没有 `class MoEParallel`，更没有 `class TopKGate`**。EP 在源码里是 5 个文件 + 2 个 reference 现场 + 1 个 1166 行的状态机的 **组合**。

---

## 这章要讲什么？

打开 `instances/vllm/source/vllm/distributed/parallel_state.py:L1264`：

```python
def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, (
        "expert parallel group is not initialized. "
        "EP group is only created for MoE models with num_experts > 0."
    )
    return _EP
```

`_EP` 是模块级单例。只有 `model_config.is_moe` 为真时（`L1672`）才会被构建——dense 模型如 Llama-3、Qwen3-base 跑起来时 `_EP is None`，调 `get_ep_group()` 会立刻断言失败。这跟 `_TP` 不一样：第 8 章里 `_TP` 是无条件存在的，哪怕 `tp_size=1` 也存在（`world_size==1` 时 `all_reduce` 在 `L518-L519` 被 bypass，但 group 还在）。**EP 是有条件存在的**——这是 EP 跟 TP 的第一处不对称。

再 grep 一下 vLLM 全树：

```
$ grep -rE "^class\s+(ExpertParallel|MoEParallel|TopKGate)\b" \
    instances/vllm/source/vllm/
(zero matches)
```

但 outline 第 9 章的标题是"MoE 专家并行"，业界论文也常说"EP framework"。**EP 是一种实现模式，不是一个 class**。在 vLLM 里它是 **5 个文件 + 2 个 reference 现场** 的协同：

- `parallel_state.py` — 进程组（`_EP` + `_EPLB`，**两个独立 group**）。
- `fused_moe/layer.py` — 组合层 `class FusedMoE`（gate + experts + routing + EP/TP plumbing 一站式）。
- `fused_moe/config.py` — EP 与 TP 的 collapse 规则 `class FusedMoEParallelConfig`。
- `device_communicators/all2all.py` — 7 个 all-to-all backend manager（`AgRs` 是教学 baseline，`DeepEPHT/LL`、`Nixl`、`FlashInfer`、`Mori` 是生产替代）。
- `fused_moe/prepare_finalize/naive_dp_ep.py` — `dispatch → expert exec → combine` 的 wiring。

加上两个真实使用现场：`models/mixtral.py:L77 MixtralMoE`（E=8、K=2、softmax、无 shared expert）和 `models/deepseek_v2.py:L244 DeepseekV2MoE`（E=64-256、grouped TopK、`noaux_tc`、shared experts）。再加一个 1166 行的状态机：`distributed/eplb/eplb_state.py`。

**outline 第 4 节叫"Expert Load Balancing Loss 的梯度回传"——这是个 training 概念**。vLLM 是 inference-only 引擎，`vllm/distributed/eplb/` 整目录里 grep `\\.backward\\(|compute_aux_loss|router_aux_loss_coef|aux_loss\\s*=` 全部 0 匹配（trap-E 的硬证据）。本章在 §9.4 把这个 outline 节点改写成 **"运行期负载均衡：EPLB 与冗余专家"**——把 training-time aux loss 当 sidebar 简介一下，然后 pivot 到 vLLM 真正在做的事：`EplbState` + 冗余专家槽 + 周期性 logical→physical 重排。这是 Ch07 "no radix tree"、Ch08 "no class TensorParallel" 之后的第三次 outline-vs-source reframe。

学完这章你能：

- 在白板上写出 Mixtral 的 `softmax → topk → renormalize` 三步走（demo §3.1：E=8 K=2 经过 1024 token 后每个 expert 的命中数 `[250, 285, 277, 243, 253, 272, 247, 221]`，coverage=1.000），并解释 trap-G —— `softmax → topk` 与 `topk → softmax` 在 `renormalize=True` 下**代数等价**、在 `renormalize=False` 下不等价（demo §3.1：`renormalize=False → sum range [0.2730, 0.6171]`）。
- 解释 vLLM 的 7 个 all-to-all backend 共享同一个 `dispatch/combine` 接口，而 `AgRsAll2AllManager` 用 `all_gatherv + reduce_scatterv` 而不是真正的 `dist.all_to_all`——同一个 end state，更简单的依赖面（trap-B 的源码证据）。
- 用 `mem_per_rank ∝ 1 / (ep × tp)` 这条不变量解释为什么 `(ep=4, tp=2)` 和 `(ep=8, tp=1)` 在 DeepSeek-V2-Lite 块上的每卡显存都是 132 MiB —— demo §3.4 表 6 行（trap-D 与 §9.5 主线）。
- 解释为什么 linear 放置 + 偏斜路由在 ep=8 给出 max/mean=3.251，而 round-robin 放置降到 1.196（demo §3.3，trap-A 的硬证据）。
- 在 §9.4 区分 EPLB（**inference 期统计 rebalance**，1166 行状态机，没有梯度）跟 Switch Transformer 的 aux loss（**训练期梯度信号**，vLLM 整树没这个东西）——demo §3.5 的时间线 `2.523 → 2.529 → 1.203` 是 EPLB 在 step 50 触发的 2.1× 改善。
- 在 §9.6 区分 **7 个语言陷阱**：A "EP=N 给 N× 容量" ✗、B "all-to-all 是对称的" ✗、C "experts 独立所以 EP 自由扩" ✗、D "EPLB 是免费的运行期 bolt-on" ✗、E "aux loss 在 vLLM 里平衡专家" ✗、F "FusedMoE.forward 总是 dispatch→experts→combine" ✗、G "softmax→topk = topk→softmax" ✗。

接下来 6 节按 outline 走，但 §9.4 已经从"Expert Load Balancing Loss 的梯度回传"重构成"运行期负载均衡：EPLB 与冗余专家"——源码里就没有 aux loss code，硬讲就是失真。

---

## 9.1 数学：Top-K 路由的两条路径——Mixtral 与 DeepSeek

### 9.1.1 打开 fused_topk 入口

源码定位：`instances/vllm/source/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L69-L113`，`def fused_topk(...)` 的核心几行：

```python
# vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L69-L113 (节选)
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "softmax",
):
    # L77 assert
    assert hidden_states.size(0) == gating_output.size(0)
    M, E = gating_output.shape
    # L94-L100 — softmax-first 然后 topk
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output.to(torch.float32), dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.to(torch.float32).sigmoid()
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids, token_expert_indices
```

四件事一起进来：**先 softmax 后 topk**（这是 trap-G 的现场）、**默认 fp32 算分**（数值稳定）、**可选 renormalize 让 K 个权重和为 1**（Mixtral / Switch 路径开）、**返回 `token_expert_indices`** 是给 Triton 的 scatter buffer。

我们的对照实现在 `implementation/routing.py:L26-L97`，原样复刻这四步：

```python
# implementation/routing.py:L65-L91
M, E = gating_output.shape
if scoring_func == "softmax":
    scores = torch.softmax(gating_output.to(torch.float32), dim=-1)
elif scoring_func == "sigmoid":
    scores = gating_output.to(torch.float32).sigmoid()
topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
token_expert_indices = (
    torch.arange(topk, dtype=torch.int32, device=gating_output.device)
    .unsqueeze(0).expand(M, topk).contiguous()
)
if renormalize:
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
```

vLLM 的 `fused_topk` 在 GPU 上调 Triton kernel `ops.topk_softmax`（softmax 路径）或 `ops.topk_sigmoid`（sigmoid 路径），但 fallback 到 `torch.softmax + torch.topk` 时数学完全一致——这是**演算法相同、layout 不同**的标准 vLLM 模式（也是 Ch08 里 column-parallel 数学一致但 GEMM 调度不同的一脉相承）。

### 9.1.2 大白话：为什么是 softmax 然后 topk，不是反过来

先把直觉讲清楚。"路由"问题的输入是 **每个 token 的 expert 偏好向量** $g \\in \\mathbb{R}^E$（来自 `g = h W_g^T`，`gate_weight` 在所有 EP rank 上**复制**——Mixtral 的 `ReplicatedLinear` 直接说明了这点，`models/mixtral.py:L123`）。要做的事是从 E 个 expert 里选 K 个，并给这 K 个一个加权系数，最后让每个 token 的输出是

$$
y = \\sum_{k=1}^{K} w_k \\cdot \\mathrm{Expert}_k(h)
$$

直觉：要让权重 $w$ 是一个非负且和为 1 的**概率分布**，最自然的做法是先把 logits 过 softmax 拿到全 E 维的概率 $s$，再选最大的 K 个。"先选 K 个再 softmax"也是一种做法——但它跟前者**不等价**，因为后者的 softmax 只在 K 个 logit 上做，分母只看那 K 个。

形式化下来，vLLM 的路径是：

$$
g = h W_g^T, \\qquad s_i = \\frac{e^{g_i}}{\\sum_{j=1}^E e^{g_j}}, \\qquad \\mathcal{T} = \\arg\\mathrm{top}_K(s)
$$

如果 `renormalize=True`：

$$
w_i = \\frac{s_i}{\\sum_{j \\in \\mathcal{T}} s_j} \\quad \\mathrm{for}\\ i \\in \\mathcal{T}, \\qquad w_i = 0 \\quad \\mathrm{otherwise}
$$

如果 `renormalize=False`，`w_i = s_i` 直接保留 softmax 尾部质量——`sum(w)` 就是落在 top-K 上的概率质量，**严格小于 1**（因为还有 E-K 个非选中的 expert 分走了概率）。

trap-G 在哪里？**`renormalize=True` 下两条路径代数等价**：

$$
\\frac{s_i}{\\sum_{j \\in \\mathcal{T}} s_j} = \\frac{e^{g_i} / Z}{\\sum_{j \\in \\mathcal{T}} e^{g_j} / Z} = \\frac{e^{g_i}}{\\sum_{j \\in \\mathcal{T}} e^{g_j}} = \\mathrm{softmax}_{\\mathcal{T}}(g)_i
$$

这跟 "topk-then-softmax" 完全一样。**`renormalize=False` 下两条路径才不等价**——前者给 softmax 尾部质量（`sum < 1`），后者总是给和为 1 的概率分布。所以 trap-G 的精确表述是 *"在 `renormalize=False` 下，softmax→topk 与 topk→softmax 不交换"*——demo §3.1 验证了这一点：

```
Renormalize on/off (Mixtral, K=2):
  renormalize=True  → sum range [1.0000, 1.0000]  mean 1.0000
  renormalize=False → sum range [0.2730, 0.6171]  mean 0.3899
```

`[0.2730, 0.6171]` 是 K=2 个 softmax 尾部概率的和——E=8 时随机 logit 下平均落在 K=2 的概率质量大约 0.39（如果 logits 是均匀的，K/E = 2/8 = 0.25，但实际 softmax 是有偏的，所以略高一点）。

**为什么 Mixtral 默认 `renormalize=True`？** 训练期 aux loss 学的是 `s_i / sum(s_in_topk)` 的概率分布，推理期保持一致，权重和为 1，与 `Expert_k(h)` 求和后量纲不变。**为什么 DeepSeek-V2 默认 `renormalize=True` 但 V3 用 `noaux_tc` 时不一定**？因为 V3 引入 `e_score_correction_bias`，把"分组评分"和"路由权重"分开（详见 9.1.4）。

### 9.1.3 Numerics：Mixtral 与 DeepSeek 的路由分布

跑 `python implementation/demo.py` 得到 §3.1 的 verbatim 数字。我们用两个尺度做 fingerprint：

**Mixtral**（E=8, K=2，1024 token，seed=7）：

```
per_expert_count = [250, 285, 277, 243, 253, 272, 247, 221]
max=285  min=221  mean=256.00  coverage=1.000
per-token weight sum: min=1.0000  max=1.0000  mean=1.0000
```

总命中数 = sum(per_expert_count) = `250+285+277+243+253+272+247+221 = 2048` = `1024 tokens × K=2`。每个 expert 的命中数应该围绕 `2048/8 = 256` 分布——确实如此（mean=256.00）。max/min 比是 `285/221 ≈ 1.29`——随机 logit + 单层 MLP 的偏置很轻。

**DeepSeek-V2 grouped**（E=64, K=6, n_group=8, topk_group=3，1024 token）：

```
max=131  min=78  mean=96.00  coverage=1.000
per-token weight sum: min=1.0000  max=1.0000  mean=1.0000
```

总命中数 = `1024 × 6 = 6144`，平均 `6144/64 = 96`——确实 mean=96.00。max/min 比 `131/78 ≈ 1.68`——比 Mixtral 略偏，因为 grouped 路由先选 3/8 组、再在选中的 24 个 expert 里挑 6 个，受组内分布影响更大。

**`coverage=1.000` 表示什么？** 每个 expert 至少被一个 token 选中过——这是 *path coverage* 而不是 *load balance*。trap-A 的提醒：`coverage=1.000` 不代表均衡。下一节我们看 `per_rank_count` 在 EP 切分下会怎么样——一旦 P>1，路由的细微偏斜会被放大成 GPU 的 idle/overload。

### 9.1.4 第二条路径：DeepSeek 的 grouped Top-K

DeepSeek-V2 / V3 用一种叫 *grouped top-K* 的路由——E 个 expert 先按 `num_expert_group` 切成几组（比如 E=64 切成 8 组，每组 8 个 expert），然后两阶段选：

1. 给每组打分（默认是组内 expert 的 max；noaux_tc 路径下是组内 top-2 之和）。
2. 选 `topk_group` 个最高分的组（比如选 3 个）。
3. 把没被选中的组里的 expert 分数 mask 成 `-inf`。
4. 在剩下的（3 × 8 = 24 个 expert 上）做标准 top-K。

源码在 `grouped_topk_router.py:L81-L162`：

```python
# vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L113-L155 (节选)
if scoring_func == "softmax":
    scores = torch.softmax(gating_output.to(torch.float32), dim=-1)
if e_score_correction_bias is not None:
    original_scores = scores
    scores = scores + e_score_correction_bias.unsqueeze(0).to(scores.dtype)
    # noaux_tc: 组分数 = 组内 top-2 之和
    group_scores = (scores.view(num_token, num_expert_group, -1)
                    .topk(2, dim=-1)[0].sum(dim=-1))
else:
    # 默认：组分数 = 组内最大值
    group_scores = (scores.view(num_token, num_expert_group, -1)
                    .max(dim=-1).values)
group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
# 把 group_mask 展平到 expert 维度，未选中组 → -inf
score_mask = group_mask.unsqueeze(-1).expand(...).reshape(num_token, -1)
tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
if e_score_correction_bias is not None:
    topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
    topk_weights = original_scores.gather(1, topk_ids)
else:
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
```

为什么这么复杂？因为 EP 把 expert 切到 GPU 上时，**如果 8 个 GPU 各拿 1 组 expert**，grouped Top-K 保证每个 token 只命中 `topk_group` 张 GPU，而不是任意 `K` 张 GPU——**all-to-all 的扇出从 K 降到 topk_group**。DeepSeek-V3 拿 E=256, K=8, topk_group=4：每 token 最多 dispatch 到 4 张 GPU，不是 8 张，all-to-all 流量减半。

`e_score_correction_bias`（noaux_tc）是 V3 的额外 trick：**用一个外部偏置 b 调整 group score**，但**返回原始 scores 当 weight**——选哪几个 expert 受偏置影响，但选完后的权重还是 unbiased softmax。这是为了让"用什么标准选"和"用什么权重组合"解耦——选的标准可以加 V3 的训练期反馈信号，权重组合保持原概率语义。

我们的 `implementation/routing.py:L100-L178 grouped_topk` 同样复刻了这一切；fingerprint 给出的 `max=131 min=78 coverage=1.000` 是 grouped 路径在我们 demo 设定下的 ground truth。

### 9.1.5 Source diff：vLLM 比我们多了什么

我们的 `routing.py` 用 `torch.softmax + torch.topk` 直接算；vLLM 的 fast path 是 Triton kernel：

| 我们的 | vLLM 的 fast path | 区别 |
|---|---|---|
| `torch.softmax(g, dim=-1)` | `ops.topk_softmax(g, ...)` | Triton fused 一步出 (weights, ids)，省掉中间 buffer |
| `torch.topk(scores, k)` | 同上 | Triton kernel 内部按 K 个最大值并行 reduction |
| Python loop 走 group mask | `ops.fused_grouped_topk(...)` | CUDA kernel 在 register 里把 mask + topk 一并做 |
| `scoring_func` 字符串 dispatch | 编译期 macro / branch | 运行时省掉 if 分支 |
| 4 行 noaux_tc | 同步进 fused kernel | 同上 |

数学完全一致——这是 vLLM 的标准模式：**算法在 PyTorch 里讲清楚，Triton/CUDA 只是把 layout 改一下让 GPU 跑得快**。整本书都在做的事是把 Triton 这层揭开，让你看见底下的 PyTorch math。

---

## 9.2 没有 `class ExpertParallel`：5 个文件的协同

### 9.2.1 grep 一遍源码

第 7 章的开篇是 *"vLLM 没有 radix tree"*，第 8 章是 *"vLLM 没有 `class TensorParallel`"*。第 9 章是这条系列的第三件：

```bash
$ grep -rE "^class\s+(ExpertParallel|MoEParallel|TopKGate)\b" \
    instances/vllm/source/vllm/
(zero matches)
```

EP 不是一个 class。Outline 说"MoE 专家并行"，但源码里 EP 是 5 个文件的组合：

| 文件 | 角色 | 关键行 |
|---|---|---|
| `vllm/distributed/parallel_state.py` | 进程组（`_EP` + `_EPLB`，**两个独立 group**） | L1261-L1283（singleton）、L1670-L1719（构造） |
| `vllm/model_executor/layers/fused_moe/layer.py` | 组合层 `class FusedMoE` + `determine_expert_map` | L70-L157、L219-L290、L1543-L1649 |
| `vllm/model_executor/layers/fused_moe/config.py` | EP-vs-TP collapse 规则 `FusedMoEParallelConfig` | L998-L1209、L1192-L1208 |
| `vllm/distributed/device_communicators/all2all.py` | 7 个 all-to-all backend manager | L40-L139（`AgRs` baseline）、L196-L325（DeepEP）、L327+（Nixl/FlashInfer/Mori） |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` | dispatch / combine wiring | L71-L168 |

加上两个 reference 现场（`models/mixtral.py:L77 MixtralMoE` 和 `models/deepseek_v2.py:L244 DeepseekV2MoE`），加上一个 1166 行的状态机（`distributed/eplb/eplb_state.py`），就是 EP 的全部源码面。

### 9.2.2 `_EP` 与 `_TP` 的两处不对称

第 8 章 `_TP` 在 `parallel_state.py:L1494-L1599 initialize_model_parallel` 里**无条件**构造——哪怕 `tp_size=1`，group 都存在；只是 `world_size==1` 时 `all_reduce` 在 `L518-L519` 直接 bypass。`_EP` 不是这样。

源码 `parallel_state.py:L1670-L1696`（节选，行号经校对）：

```python
# vllm/distributed/parallel_state.py:L1670+ (节选)
if model_config is not None and model_config.is_moe:
    # only build EP group for MoE models
    group_ranks = (all_ranks.transpose(1, 2)
                   .reshape(-1, dp * pcp * tp).unbind(0))
    _EP = init_model_parallel_group(group_ranks, ..., group_name="ep")
    if parallel_config.enable_eplb:
        _EPLB = init_model_parallel_group(group_ranks, ..., group_name="eplb")
```

两处不对称：

1. **`_EP` 只为 MoE 模型构造**。Llama/Qwen3-base 这种 dense 模型 `_EP is None`；调 `get_ep_group()` 立即断言"EP group is only created for MoE models with num_experts > 0"。
2. **如果开启 EPLB，单独再建一个 `_EPLB` group**——同一组 rank list、不同的 group object、不同的 NCCL communicator。源码 `L1700` 的注释一字不漏：*"to prevent deadlocks when using torch.distributed in execution with torch.distributed in EPLB"*。

第二点是 trap-D 的硬证据。EPLB 的 rebalance broadcast 跟 forward-pass 的 dispatch all-to-all **不能共用一个 NCCL stream**——共用的话，一个 in-flight 的 EPLB 广播会 block 一个正在等待的 dispatch，反过来也一样。**两个 group 是两个独立的 heap object**，它们的 NCCL handle 互不阻塞——这是 W04 backpressure isolation pattern 的教科书例子。

我们的 `implementation/ep_groups.py:L259-L267` 复刻这一点：

```python
# implementation/ep_groups.py:L247-L267 (节选)
_EP = EPGroup(world_size=len(my_group), rank_in_group=rank_in_group,
              group_name="ep", rank_list=my_group)
if enable_eplb:
    _EPLB = EPGroup(world_size=len(my_group), rank_in_group=rank_in_group,
                    group_name="eplb", rank_list=list(my_group))  # SAME ranks, NEW object
```

测试 `test_eplb_group_is_distinct_object_from_ep` 用 `assert ep is not eplb` 校验**对象身份不等**——共享 rank list 是必要条件、不是充分条件。

### 9.2.3 mesh 公式：EP 是 TP × DP × PCP 的补轴

这是 §9.5 主线的预热（那里会用 demo §3.4 的内存表证明），先把公式写清楚。`parallel_state.py:L1670-L1696` 的 mesh 构造法：

```python
# vllm/distributed/parallel_state.py:L1670-L1683 (节选)
all_ranks = torch.arange(world_size).view(pp, pcp_size, dp_size, tp_size)
# (pp, pcp, dp, tp) → 把 (pcp, dp) 对换 → 展平 (dp*pcp*tp) 维
group_ranks = (all_ranks.transpose(1, 2)
               .reshape(-1, dp * pcp * tp).unbind(0))
```

逐步看：

1. `all_ranks` 是一个 4D tensor，shape `(pp, pcp, dp, tp)`，每个位置是一个 rank id。
2. `transpose(1, 2)` 把 axis-1 (pcp) 和 axis-2 (dp) 对换，得到 `(pp, dp, pcp, tp)`。
3. `reshape(-1, dp * pcp * tp)` 把 axis-1, axis-2, axis-3 三个一起展平成大小 `dp * pcp * tp` 的一维——这就是每个 EP group 的大小。
4. `unbind(0)` 把外层 axis（pp）拆开，得到 `pp` 个 EP group（每个 PP stage 一个 EP group）。

**EP group size 公式**：

$$
\\mathrm{ep\\_size} = \\frac{\\mathrm{world\\_size}}{\\mathrm{pp\\_size}} = \\mathrm{dp\\_size} \\times \\mathrm{pcp\\_size} \\times \\mathrm{tp\\_size}
$$

——对应 `config.py:L1077` 的 `flatten_tp_size = dp_size * pcp_size * tp_size`，再到 `L1192` 的 `ep_size = flatten_tp_size`。**EP 是 TP×DP×PCP 的"补轴"**，不是一个独立的 hyperparameter——operator 设 `tp_size, dp_size, pcp_size, enable_expert_parallel`，`ep_size` 是被算出来的。

trap-D 的另一面在这里：把 EP 当成一个独立的 hyperparameter（"我想要 ep=8 同时 tp=4 同时 dp=2 同时 world=8"）会立刻被 config.py 的 assert 拦下来——8 ≠ 8×4×2 = 64。EP 是导出量。

### 9.2.4 5 文件协同：一次 forward 走过的路

打开 `prepare_finalize/naive_dp_ep.py:L125-L168` 看 `MoEPrepareAndFinalizeNaiveDPEPModular.apply` 的核心：

```python
# vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L127+ (节选)
def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
          activation, global_num_experts, expert_map, ...):
    # L127-L132 — dispatch
    hs_disp, tw_disp, ti_disp = get_ep_group().dispatch(
        hidden_states, topk_weights, topk_ids, ...)
    # L150-L161 — local expert FFN（per-rank pass）
    expert_out = self.fused_experts(hs_disp, w1, w2, tw_disp, ti_disp,
                                     activation=activation,
                                     global_num_experts=global_num_experts,
                                     expert_map=expert_map, ...)
    # L166-L168 — combine
    out = get_ep_group().combine(expert_out, ...)
    output.copy_(out)
```

5 文件协同的"一次 forward":

```
hidden_states [M, hidden]
   │
   │ ┌── (1) gate(h) → router_logits [M, E]              gate.weight 在 EP 内 REPLICATED
   │ │       Mixtral: ReplicatedLinear; DeepSeek: GateLinear (+e_score_correction_bias)
   │ │       源码：models/mixtral.py:L123 / models/deepseek_v2.py:L272
   │ ▼
   │ ┌── (2) Top-K 路由 → (topk_weights [M,K], topk_ids [M,K])
   │ │       Mixtral: fused_topk           (router/fused_topk_router.py:L69)
   │ │       DeepSeek: grouped_topk         (router/grouped_topk_router.py:L81)
   │ ▼
   │ ┌── (3) EP-axis all-to-all DISPATCH    ← get_ep_group().dispatch(...)
   │ │       AgRs: all_gatherv 把 hidden_states/topk_meta 收齐到所有 rank
   │ │       DeepEP: 一次 fused IB+NVLink kernel
   │ │       源码：device_communicators/all2all.py:L40-L139 (AgRs)
   │ ▼
   │ ┌── (4) 本地 expert FFN（仅本 rank 拥有的 experts）
   │ │       gate|up = h @ w13.T          (MergedColumnParallelLinear, w13)
   │ │       act     = SiluAndMul(gate|up)
   │ │       down    = act @ w2.T          (RowParallelLinear, w2, 内部 TP)
   │ │       out += weight[k] * down
   │ │       源码：layer.py:L1543+ → quant_method.apply → fused_experts
   │ ▼
   │ ┌── (5) EP-axis all-to-all COMBINE   ← get_ep_group().combine(...)
   │ │       AgRs: reduce_scatterv 把求和后的 [total, hidden] 切回每 rank 的 chunk
   │ │       DeepEP: 一次 fused combine kernel
   │ │       源码：all2all.py:L130-L135 (AgRs.combine)
   │ ▼
output_states [M, hidden]
```

**一次 MoE forward 有两次 all-to-all**（dispatch + combine）。第 8 章 Llama transformer block **有两次 all-reduce**（attn 的 o_proj + MLP 的 down_proj）。Ch08 的 chain-break 是 *"一次 col→row pair 一次 all-reduce"*；Ch09 的 chain-break 是 *"一次 MoE block 两次 all-to-all，且 EP 与 TP 正交"*。

这里我们对照 `implementation/fused_moe_block.py:L226-L267 forward`：

```python
# implementation/fused_moe_block.py:L244-L266 (节选)
if router_logits is None:
    router_logits = hidden_states @ self.gate_weight.T
topk_weights, topk_ids = self._route(hidden_states, router_logits)
contributions = []
for r in range(self.ep_size):
    contributions.append(
        self._run_local_experts(hidden_states, topk_weights, topk_ids, rank=r)
    )
combined = torch.stack(contributions, dim=0).sum(dim=0)
return combined
```

教学 vs 生产差异：单进程模拟里"all-to-all"被压扁成"对每个 rank 的 expert 跑一遍"，因为我们没有真 NCCL。对应的 chain-break 不变：每个 rank 只激活自己拥有的 expert，最后再求和。tester 的 `test_forward_invariance_ep1_vs_ep4` 验证 `ep ∈ {1, 2, 4, 8}` 在两条路由路径下都给出 atol=1e-6 一致的输出——**EP 是 expert-sum 的 partition，不是不同的数学**。

### 9.2.5 Mini-mapping：EP 五文件 + reference 现场

| 角色 | 源码路径 | 核心行 | 我们的对照 |
|---|---|---|---|
| 进程组 (`_EP`) | `parallel_state.py` | L1261-L1283, L1670-L1696 | `implementation/ep_groups.py:L188-L269` |
| 进程组 (`_EPLB`) | `parallel_state.py` | L1700-L1719 | `implementation/ep_groups.py:L259-L267` |
| 组合层 | `fused_moe/layer.py` | L219-L290 | `implementation/fused_moe_block.py:L63-L267` |
| 配置 collapse | `fused_moe/config.py` | L998-L1209, L1192-L1208 | `implementation/ep_groups.py:L36-L150` |
| 路由 (Mixtral) | `router/fused_topk_router.py` | L69-L113 | `implementation/routing.py:L26-L97` |
| 路由 (DeepSeek) | `router/grouped_topk_router.py` | L81-L162 | `implementation/routing.py:L100-L178` |
| 全局→本地 map | `fused_moe/layer.py` | L70-L157 | `implementation/expert_map.py:L27-L88` |
| 放置策略 | `fused_moe/layer.py` | L160-L193 | `implementation/expert_map.py:L66-L86` |
| AgRs backend | `all2all.py` | L40-L139 | `implementation/all2all_baseline.py:L65-L115` |
| DeepEP HT | `all2all.py` | L196-L256 | (生产替代，未复刻) |
| DeepEP LL | `all2all.py` | L257-L325 | (生产替代，未复刻) |
| Nixl backend | `all2all.py` | L327-L440 | (生产替代，未复刻) |
| FlashInfer NVLink | `all2all.py` | L442-L670 | (生产替代，未复刻) |
| Mori backend | `all2all.py` | L671+ | (生产替代，未复刻) |
| Dispatch wiring | `prepare_finalize/naive_dp_ep.py` | L71-L168 | `implementation/fused_moe_block.py:L162-L220` |
| EPLB 状态机 | `eplb/eplb_state.py` | L210-L920 | `implementation/eplb.py:L34-L152` |
| Reference #1 (Mixtral) | `models/mixtral.py` | L77-L154 | `implementation/mixtral_vs_deepseek.py:L48-L57` |
| Reference #2 (DeepSeek) | `models/deepseek_v2.py` | L244-L386 | `implementation/mixtral_vs_deepseek.py:L64-L75` |

**18 行 mini-mapping**——这是 §9.6 主表的预热。

---

## 9.3 Expert Placement：linear vs round_robin 与 -1 哨兵

### 9.3.1 打开 determine_expert_map 入口

源码定位：`instances/vllm/source/vllm/model_executor/layers/fused_moe/layer.py:L70-L157`，`def determine_expert_map(...)`。把核心几行抠出来：

```python
# vllm/model_executor/layers/fused_moe/layer.py:L70+ (节选)
def determine_expert_map(
    ep_size: int, ep_rank: int, global_num_experts: int,
    expert_placement_strategy: str = "linear",
):
    # L107-L109 — ep_size==1 短路
    if ep_size == 1:
        return global_num_experts, None, None  # 3-tuple in source
    # L112-L114
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts
    # L117 — 全局 -1 哨兵
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    if expert_placement_strategy == "linear":
        # L119-L123
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = (
            torch.arange(0, local_num_experts, dtype=torch.int32))
    elif expert_placement_strategy == "round_robin":
        # L124-L131
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int64)
        expert_map[local_log_experts] = (
            torch.arange(0, local_num_experts, dtype=torch.int32))
    return local_num_experts, expert_map, expert_mask
```

**两条放置策略，一个 -1 哨兵，一个 ep_size==1 短路**。我们的对照 `implementation/expert_map.py:L27-L88` 复刻三件事：

```python
# implementation/expert_map.py:L46-L86 (节选)
if ep_size == 1:
    return (global_num_experts, None)             # 2-tuple, 见 E13
base_experts = global_num_experts // ep_size
remainder = global_num_experts % ep_size
local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts
expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
if expert_placement_strategy == "linear":
    start_idx = ep_rank * base_experts + min(ep_rank, remainder)
    expert_map[start_idx : start_idx + local_num_experts] = (
        torch.arange(0, local_num_experts, dtype=torch.int32))
elif expert_placement_strategy == "round_robin":
    local_log_experts = torch.arange(
        ep_rank, global_num_experts, ep_size, dtype=torch.int64)
    expert_map[local_log_experts] = (
        torch.arange(0, local_num_experts, dtype=torch.int32))
```

**`ep_size==1` 我们返回 2-tuple `(E, None)`，源码返回 3-tuple `(E, None, None)`**——这是 E13 知识点。源码的第三项是给 ROCm AITER quant 路径的 `expert_mask`，本章不涉及 AITER，所以删掉了。tester 的 `test_ep1_returns_E_and_None_map` pin 这个 2-tuple 形状。读者把我们的代码跟源码逐行 diff 时一定会注意到这个差——是有意的简化，不是 bug。

### 9.3.2 大白话：linear 是顺序、round_robin 是交错

E=8、P=4 时：

```
linear:        rank 0: [0, 1]   rank 1: [2, 3]   rank 2: [4, 5]   rank 3: [6, 7]
round_robin:   rank 0: [0, 4]   rank 1: [1, 5]   rank 2: [2, 6]   rank 3: [3, 7]
```

E=8、P=3 时（remainder=2）：

```
linear:        rank 0: [0, 1, 2]  rank 1: [3, 4, 5]  rank 2: [6, 7]   ← 前 2 rank 多拿 1 个
round_robin:   rank 0: [0, 3, 6]  rank 1: [1, 4, 7]  rank 2: [2, 5]
```

**为什么有两种？** 因为路由分布**不是均匀的**。一个训练好的 MoE 模型，相邻 expert id 的 firing pattern 是**相关的**（trained 模型常常把语义相近的概念聚到相邻 expert 上）。这意味着：

- `linear`: rank 0 拿 expert [0..local-1]——这些 expert 通常一起活跃。**locality-friendly** 但容易让某一个 rank 持续 hot。
- `round_robin`: rank 0 拿 expert [0, P, 2P, ...]——交错放置把 hot 的几个 expert 散到不同 rank。**load-friendly** 但牺牲 locality（如果两个相邻 expert 经常一起被同一个 token 选中，现在这两个 expert 在不同 rank，dispatch 流量更大）。

`-1` 哨兵的作用：`expert_map` 是一个长 E 的 int32 张量，`expert_map[i] = local_idx_or_-1`。本地 forward pass 拿到 `topk_ids[m, k] = global_id`，查一下 `expert_map[global_id]`——如果 `>= 0`，是这张 rank 的本地 idx；如果 `== -1`，跳过这个 (token, slot)。`prepare_finalize/naive_dp_ep.py:L104-L132` 用这个掩码决定哪些 token-slot 在本 rank 跑 expert FFN。

**为什么用 -1 而不是另一个 list？** 因为 GPU 上索引一个固定大小的张量是一次内存读，比维护一个 hashmap 快。trade-off：每个 token 多一次分支（`if local_idx == -1: skip`），但分支可预测（绝大多数 token 在同一个或几个 expert 上集中）。

### 9.3.3 Demo §3.3：偏斜路由放大放置差异

跑 demo 用 32 个 expert、4096 token、K=2、人造 Pareto-skewed 路由（hot 20% 的 expert 拿到 60% 的 token）：

```
E=32, K=2, tokens=4096, hot 20% of experts received 4915/8192 routed pairs (0.600)

placement      ep_size                 rank loads                  max/mean
linear           1                     [8192]                       1.000
linear           4                     [5175, 980, 1017, 1020]      2.527
linear           8                     [3329, 1846, 483, 497, 458,  3.251
                                        559, 515, 505]
round_robin      4                     [2350, 2420, 1695, 1727]     1.182
round_robin      8                     [1199, 1195, 1186, 1205,     1.196
                                        1151, 1225, 509, 522]
```

读这张表：

- **`ep_size=1` 时无所谓放置**——只有一张 rank，所有 token 都在这一张上，max/mean=1.000（没有比较对象）。
- **linear + ep=4**：rank 0 拿 expert [0..7]——hot expert 都在这块——5175/2048=2.53× 平均。rank 1/2/3 各拿 [8..15]/[16..23]/[24..31]——cold expert，远低于平均。
- **linear + ep=8**：放置更细，rank 0 拿 [0..3]——但 hot 全在 [0..6]——rank 0 仍然 3329/1024 ≈ 3.25× 平均。rank 1 拿 [4..7]—— [4, 5, 6] 也是 hot——也偏高（1846/1024 ≈ 1.80×）。max/mean=3.251 是这张表的核心数字。
- **round_robin + ep=4**：rank r 拿 [r, r+4, r+8, ..., r+28]。Hot 的 [0..5] 散到 rank 0,1,2,3,0,1——还是有点偏（前两 rank 多）但整体 max/mean=1.18，比 linear 好得多。
- **round_robin + ep=8**：hot 的 [0..5] 散到 rank 0..5 各 1 个 expert——每个 rank 拿到 1 个 hot。max/mean=1.196。

**trap-A 的硬证据**：`linear + ep=8 + 偏斜路由 → max/mean=3.251`。EP 把内存切了 8 倍（每张 GPU 拿 4 个 expert 的权重，1/8 的总参数），但**吞吐没切到 8 倍**——因为吞吐被最慢的 rank 限制——慢 rank 的工作量是平均的 3.25 倍，所以**有效吞吐只有理论峰值的 1/3.25 ≈ 31%**。Round-robin 把这个比例从 1/3.25 提到 1/1.196 ≈ 84%——一个 2.7× 的实际改善。

**所以 EP 不是 free 的。EP 切内存，throughput 要靠放置 + EPLB 维持**。这是 §9.5 的基础。

### 9.3.4 Trap-A 与 Trap-C 的源码评注

`determine_expert_placement_strategy` 在 `layer.py:L160-L193` 还有 **4 个 round_robin → linear 的回退条件**——也就是说你设了 `round_robin` 不一定真的拿到 round_robin：

| 条件 | 源码 | 影响 |
|---|---|---|
| 模型只有一组 expert（`num_expert_group <= 1`） | L168 附近 | 没有 group 概念，round_robin 退化 |
| `num_redundant_experts > 0` | L169 附近 | EPLB 路径要求 linear（冗余 slot 复用 logical 编号） |
| `enable_eplb=True` | L170 附近 | 同上，EPLB 自带放置策略 |
| backend 不是 DeepEP-LL 或 NIXL | L181-L191 附近 | AgRs / DeepEP-HT 的 kernel 假定 owned block 连续 |

**knowledge E09**：写测试要验证 round_robin 必须满足全部 4 条，否则**会被默默回退到 linear，logger 给 warning**。这是 trap-A 的延伸——不仅 EP 不 free，连放置策略本身也不是想用就能用。

**Trap-C 的另一面：DeepSeek 的 shared experts 不参与 EP 切分**。`models/deepseek_v2.py:L295-L317`：

```python
# vllm/model_executor/models/deepseek_v2.py:L295+ (节选)
if config.n_shared_experts is not None:
    intermediate_size = (config.moe_intermediate_size *
                          config.n_shared_experts)
    self.shared_experts = DeepseekV2MLP(
        hidden_size=config.hidden_size,
        intermediate_size=intermediate_size,
        ...)
self.experts = FusedMoE(num_experts=config.n_routed_experts, ...)
```

`shared_experts` 在 `FusedMoE` **之外** 构造——它走标准的 TP 分片（`MergedColumnParallelLinear` + `RowParallelLinear`），但不走 EP——**每个 rank 都有完整的 shared experts**。`n_shared_experts=2` 的 DeepSeek-V2 上，每张 GPU 拿到 64 个 routed experts 的 1/ep_size + 全部 2 个 shared experts。EP 给的内存节省**只对 routed experts 有效，shared experts 完全复制**——这是 trap-C 的硬证据。

测试 `test_trap_C_shared_experts_DO_NOT_appear_in_FusedMoEBlock_memory` 显式校验这一点：我们的 `memory_per_rank_MiB` 函数只对 `num_experts` 求和，**不**包含 shared experts——因为 shared experts 不是 EP 的事情，它们是 TP 的事情。

### 9.3.5 Mini-mapping：放置与哨兵

| 我们的 | 源码 | 区别 |
|---|---|---|
| `expert_map.py:L46-L50` 返回 2-tuple | `layer.py:L107-L109` 返回 3-tuple | 我们删掉 AITER `expert_mask`（E13） |
| `expert_map.py:L64` 用 `-1` int32 哨兵 | `layer.py:L117` 同 | 一致 |
| `expert_map.py:L70` `start_idx = r·base + min(r, rem)` | `layer.py:L119-L123` 同 | 一致 |
| `expert_map.py:L77` `arange(r, E, P)` round_robin | `layer.py:L124-L131` 同 | 一致 |
| 没有 `determine_expert_placement_strategy` 4 个回退 | `layer.py:L160-L193` 有 | 我们简化为直接传字符串（E09 注解） |
| `expert_map.py:L91-L103 get_compressed_expert_map` | `layer.py:L196-L214` 同 | 教学 helper，一致 |
| 没有 ROCm AITER `expert_mask` | `layer.py:L116` 旁边 | E13 显式简化 |

---

## 9.4 运行期负载均衡：EPLB 与冗余专家（替代 outline §4 "梯度回传"）

### 9.4.1 §X 重构：outline 写的是 training，源码做的是 inference

Outline 第 4 节叫"Expert Load Balancing Loss 的梯度回传"。这是一个 **training 概念**——Switch Transformer 的 $L_\\mathrm{balance}$、GShard 的 capacity factor、DeepSpeed-MoE 的 aux loss——它们都是训练时往 loss 加一个正则项，让 expert 命中分布趋于均匀。

vLLM 是 inference-only 引擎。`vllm/distributed/eplb/` 整目录里 grep `\\.backward\\(|compute_aux_loss|router_aux_loss_coef|aux_loss\\s*=` ——**0 匹配**。`vllm/model_executor/` 下也是 0 匹配（**Tester 精化 E10**：除 `phimoe.py` 把 `router_aux_loss_coef` 当成 stored constant 存下来、`vision.py` 有个名字撞车的 `get_load_balance_assignment`——这两个都不是 expert routing 的 aux loss）。

所以 **outline §4 的题面问的是一个 vLLM 不实现的概念**。我们要照实把这件事说清楚，然后 pivot 到 vLLM 真正在做的事——**运行期统计 rebalance**。

历史背景（一段 sidebar，给读者一个锚）：训练期用 aux loss 平衡专家是 Switch Transformer (Fedus et al., 2022) 提出的，配方是在 cross-entropy loss 之外加：

$$
L_\\mathrm{balance} = \\alpha \\cdot E \\cdot \\sum_{i=1}^{E} f_i \\cdot P_i
$$

其中 $f_i$ 是分配给 expert $i$ 的 token 比例，$P_i$ 是 expert $i$ 的平均 gate 概率，$\\alpha$ 是系数。直觉是 *expert 命中频率 × 平均概率 → 越偏越大*。这个 loss 让训练期权重往均衡方向移。**但训练好后部署时，aux loss 已经走完使命了**——模型权重定下来了，给定输入数据分布的命中分布也定了。

**问题是：deployed 流量 ≠ training 流量**。即便训练期 aux loss 让分布平衡了，真实推理流量上还是会偏。EPLB 是 vLLM 对这个偏的**运行期回应**——不动权重、不算 loss、不回传梯度，只**改 logical → physical 的映射**。

### 9.4.2 EPLB 的核心：冗余专家 + 周期性重排

`distributed/eplb/eplb_state.py:L210-L920` 是 `class EplbState`——1166 行的状态机。我们只取它的两个核心组件：

1. **冗余专家**（`num_redundant_experts > 0`）。比方说 logical 32 个 expert，加 4 个冗余 → 36 个 physical slot。**多个 physical slot 可以映到同一个 logical expert**——hot expert 在多张 rank 上同时存在，并行处理。
2. **周期性重排**（`rearrangement_step_interval`）。每 N 个 forward 跑一次重排：根据滑动窗口的 per-expert load 重排 logical → physical 映射，把 hot logical expert 派到冗余 slot 上。

我们的 `implementation/eplb.py:L34-L152` 复刻这两件事（没有 async worker、没有真 collective broadcast——这是教学版）：

```python
# implementation/eplb.py:L55-L102 (节选)
@dataclass
class EplbState:
    num_logical_experts: int
    num_redundant_experts: int = 0
    ep_size: int = 1
    rearrangement_step_interval: int = 50
    window_size: int = 50
    _step: int = 0
    _last_rearrangement_step: int = 0
    _load_history: List[torch.Tensor] = field(default_factory=list)
    _physical_to_logical: torch.Tensor | None = None

    def __post_init__(self):
        # 初始 layout: [0..L-1, redundant round-robin]
        layout = list(range(self.num_logical_experts))
        layout += [i % self.num_logical_experts
                   for i in range(self.num_redundant_experts)]
        self._physical_to_logical = torch.tensor(layout, dtype=torch.int64)

    def record_step(self, per_expert_load: torch.Tensor) -> bool:
        self._step += 1
        self._load_history.append(per_expert_load.detach().clone())
        if len(self._load_history) > self.window_size:
            self._load_history.pop(0)
        if self._step - self._last_rearrangement_step < self.rearrangement_step_interval:
            return False
        self._rearrange()
        self._last_rearrangement_step = self._step
        return True
```

`record_step` 把当前 forward 的 per-logical-expert load 加入窗口。每 `interval` 次 forward 触发 `_rearrange()`：取窗口平均、按 hot 程度排序、把 hottest 几个 logical expert 抢占冗余 slot。

`_rearrange` 的策略很简单（生产版在 `vllm/distributed/eplb/policies.py` 里有更复杂的 bin-packing）：

```python
# implementation/eplb.py:L108-L136 (节选)
def _rearrange(self):
    windowed = torch.stack(self._load_history, dim=0).float().mean(dim=0)
    n_log = self.num_logical_experts
    n_phys = self.num_physical_experts
    order = torch.argsort(windowed, descending=True).tolist()
    layout = list(range(n_log))   # 每个 logical 至少 1 个 physical slot
    redundant = n_phys - n_log
    for i in range(redundant):
        layout.append(order[i % len(order)])  # 冗余 slot 拿 hot expert
    self._physical_to_logical = torch.tensor(layout, dtype=torch.int64)
```

### 9.4.3 Demo §3.5：100 步重排时间线

跑 100 步、E=32 logical、num_redundant=4 → 36 physical、ep=4，K=2。每步用 Pareto-skewed 路由（hot 20% 拿 60% token）。重排间隔=50。每步 seed=`100 + step`（**E14 知识点**：tester 提醒，重现这一时间线必须 mirror 这个 seed 模式）。结果：

```
step    placement       per-rank load                              max/mean
   0    linear          [1292, 246, 257, 253]                       2.523
  25    linear          [1295, 230, 261, 262]                       2.529
  50    round_robin     [591, 616, 423, 418]                        1.203  ← EPLB triggered
  51    round_robin     [575, 593, 463, 417]                        1.158
  75    round_robin     [594, 629, 391, 434]                        1.229
  99    round_robin     [582, 611, 434, 421]                        1.193
```

EPLB 在 step 50 触发——`max/mean` 从 2.529 跳到 1.203，**2.1× 改善**。后续 step 51, 75, 99 上下波动在 1.15-1.23 之间，因为 hot 集合每步种子不同有微抖动。

底层 layout 变化：

```
EplbState: num_logical=32, num_redundant=4, num_physical=36
physical_to_logical[0:8] = [0, 1, 2, 3, 4, 5, 6, 7]   ← 初始
physical_to_logical[-4:] = [5, 2, 0, 4]                ← 重排后冗余 slot
```

冗余 slot 上是 logical expert `[5, 2, 0, 4]`——这四个是窗口里命中数最高的（按 windowed load 排序）。它们现在在 physical slot 32, 33, 34, 35 各有一份，加上原本的 physical slot 5, 2, 0, 4——**hot expert 在 2 张物理 slot 上同时存在**，路由可以并行打到任一个。

**`max/mean` 与 `imbalance_ratio` 的边界注意**（E16）：`imbalance_ratio` 的实现把 `mean` clamp 到 1（避免除零）。当 per-rank load 全 0 时（routing 还没发生），`mean=1 max=0 → ratio=0.0`，**不是 1.0**。当 `numel==0` 时返回 1.0 哨兵。两种边界情况返回不同的哨兵——读者要注意 `imbalance=1.0` 不一定意味着均衡，可能是"还没数据"。这个微妙在 `implementation/eplb.py:L142-L152` 有显式的注释。

### 9.4.4 EPLB ≠ aux loss：三条硬证据

tester 写了 4 个 negative 测试 pin trap-E：

1. **我们的 EPLB 模块**（AST-strip docstrings 后）：grep `.backward(`、`torch.optim`、`Optimizer(`、`compute_aux_loss` —— 0 匹配。
2. **autograd**：`record_step` 显式 `.detach()` load tensor；保存的 history `requires_grad == False`。
3. **vLLM 源码**：grep `vllm/distributed/eplb/` 5 个 `.py` 文件 —— `.backward(|compute_aux_loss|compute_balance_loss|router_aux_loss_coef|aux_loss\\s*=` 全部 0 匹配。
4. **vLLM 全树**：trap-E 的精化（E10）—— 名字撞车的两个文件 `phimoe.py:router_aux_loss_coef` 是 HuggingFace config 的 stored constant、`vision.py:get_load_balance_assignment` 是图像 tile 放置的 helper（不是 expert 路由）。

**所以精确表述是**：vLLM 是 inference-only 的，**MoE 推理路径里没有 aux-loss 计算**。Trained 模型的 expert 分布是训练期 aux loss 沉淀下来的，但部署时 aux loss 这个概念不再生效。EPLB 是 vLLM 部署时的运行期回应——不动权重、不回传梯度，**只改 logical→physical 映射**。

### 9.4.5 EPLB 不是 free 的：三道侧门

trap-D 的精化版（E02、E15）：

1. **必须 `num_redundant_experts > 0`**——否则 `_rearrange` 有 0 个冗余 slot 可分，重排退化为恒等。
2. **`_EP` 与 `_EPLB` 是不同的 Python heap object**（`id(_EP) != id(_EPLB)`），共享 rank list 但 NCCL communicator 不同——这是反 deadlock 的核心机制（`parallel_state.py:L1700-L1719` 注释）。
3. **quant method 必须声明 `supports_eplb`**。`fused_moe/layer.py:L548-L557`：

```python
# vllm/model_executor/layers/fused_moe/layer.py:L548-L557 (节选)
if self.enable_eplb and not self.quant_method.supports_eplb:
    raise NotImplementedError(
        f"EPLB is not supported with {self.quant_method.__class__.__name__}")
```

这是个硬 gate。比如某些早期 FP8 量化 method 不支持 EPLB——开 EPLB 直接报错。**EPLB 不是免费的运行期 bolt-on**——它跟 quant method、placement strategy（不能 round_robin，会被 fallback 到 linear）、`num_redundant_experts > 0` 这三个东西都耦合。

测试 `test_trap_D_eplb_separate_group_prevents_aliasing` 用 `assert ep is not eplb` 验证对象身份不等。`test_eplb_not_created_when_disabled` 验证 `enable_eplb=False` 时 `_EPLB is None`。

### 9.4.6 Mini-mapping：EPLB 教学版与生产版

| 角色 | 我们的 (`eplb.py`) | 源码 (`eplb_state.py`) | 区别 |
|---|---|---|---|
| state machine | `EplbState` 156 行 | `class EplbState` 1166 行的子集 | 我们留 record_step + _rearrange，去掉 async worker |
| 重排策略 | `_rearrange` 贪心：hot 进冗余 | `policies.py` bin-packing solver | 一致的目标，更复杂的策略 |
| 进程组 | `_EPLB` toy object | `_EPLB` 单独 NCCL communicator | 单进程模拟，没有真 collective |
| 重排间隔 | 固定 `interval=50` | 可配 `expert_rearrangement_step_interval` | 一致 |
| 窗口 | `window_size=50` 列表 | 滑动窗口（同形 pop） | 一致 |
| autograd | `.detach()` 保证不挂 | `.detach()` 同 | E10：trap-E 的 negative test 锚点 |
| imbalance metric | `max/mean` | 同 | E16：边界 corner-case |

---

## 9.5 EP+TP 的 device mesh：`mem_per_rank ∝ 1 / (ep × tp)`

### 9.5.1 把不变量放在最前面

§9.5 主线一句话：

$$
\\mathrm{mem\\_per\\_rank}(E, h, F, ep, tp) = \\frac{E \\cdot 3 F h \\cdot \\mathrm{bytes\\_per\\_param}}{ep \\cdot tp}
$$

其中 $E$ 是全局 expert 数、$h$ 是 hidden size、$F$ 是 intermediate size、$3 F h$ 是单个 expert 的总参数数（gate + up + down，每个 $F \\times h$ 矩阵）。这个不变量是 §9.5 的 load-bearing claim——**任何 demo 数字都是这条公式的实例**（这是 K13 的 N-1×K 模式：先把公式说清楚，再用数字验证）。

为什么 $3 F h$？标准 SwiGLU expert 有三个矩阵：

- `w_gate` ∈ ℝ^{F × h}（gate projection，输入 h 维，输出 F 维 intermediate）。
- `w_up` ∈ ℝ^{F × h}（up projection，同上）。
- `w_down` ∈ ℝ^{h × F}（down projection，输入 F 维，输出 h 维）。

vLLM 把 `gate` 和 `up` 横向 concat 成 `w13` ∈ ℝ^{2F × h}（`MergedColumnParallelLinear`），`down` 是 `w2` ∈ ℝ^{h × F}（`RowParallelLinear`）。两个矩阵加起来 `(2F + F) × h = 3 F h` 个参数（per-expert）。

```python
# implementation/fused_moe_block.py:L302 (节选)
def memory_per_rank_MiB(num_experts, hidden, intermediate, ep_size, tp_size,
                         bytes_per_param=2):
    params_per_expert = 3 * intermediate * hidden
    total = num_experts * params_per_expert * bytes_per_param
    return total / (ep_size * tp_size) / (1024**2)
```

EP 切 expert 维度（每张 rank 拿 $E / \\mathrm{ep\\_size}$ 个 expert），TP 切每个 expert 内的矩阵（$2F$ 维和 $F$ 维各切 tp 份）。这两条**正交**，乘起来就是总缩减比。

### 9.5.2 Demo §3.4：6 行内存表

DeepSeek-V2-Lite 块尺寸：E=64、hidden=2048、intermediate=1408、bf16（2 byte/param）。

```
Per-expert params: 3·1408·2048 = 8,650,752
Total params:       E · 3·intermediate·hidden = 553,648,128
Bytes per param:    2 (bf16)

  ep  tp    mem/rank (MiB)    reduction vs (1,1)
   1   1            1056.00         1.00x
   4   1             264.00         4.00x
   4   2             132.00         8.00x
   8   2              66.00        16.00x
  16   1              66.00        16.00x
   8   4              33.00        32.00x
```

逐行看：

- **(1, 1)**：每张 rank 全部 64 expert × 全部矩阵 = `64 × 3 × 1408 × 2048 × 2 = 553,648,128 bytes ≈ 528 MB ≈ 1056 MiB`。
- **(4, 1)**：64 个 expert / 4 rank = 每张 rank 16 expert，每个 expert 完整矩阵。`1056/4 = 264 MiB`。
- **(4, 2)**：每张 rank 16 expert，每个 expert 内部矩阵切 2 份。`264/2 = 132 MiB`。
- **(8, 2)**：每张 rank 8 expert，矩阵切 2 份。`1056/(8×2) = 66 MiB`。
- **(16, 1)**：每张 rank 4 expert，矩阵不切。`1056/(16×1) = 66 MiB`——**与 (8, 2) 同**。
- **(8, 4)**：每张 rank 8 expert，矩阵切 4 份。`1056/(8×4) = 33 MiB`。

**关键校验**：`(4, 2)` 和 `(8, 1)` 都给 132 MiB，`(8, 2)` 和 `(16, 1)` 都给 66 MiB——`mem_per_rank` 只看 `ep × tp` 这个乘积，不看分别的值。这是不变量在 6 个数据点上的 **存在证明**（test `test_memory_inverse_proportional_to_ep_times_tp` pin 这点）。

### 9.5.3 EP-vs-TP 的 collapse 规则

`config.py:L1192-L1208` 的核心：

```python
# vllm/model_executor/layers/fused_moe/config.py:L1162+ (节选)
use_ep = (dp_size_ * pcp_size_ * tp_size_ > 1) and enable_expert_parallel
flatten_tp_size = dp_size_ * pcp_size_ * tp_size_   # L1077
if not use_ep:
    return FusedMoEParallelConfig(
        tp_size=flatten_tp_size, ep_size=1, ...)        # L1175-L1189
# use_ep == True:
ep_size = flatten_tp_size                                 # L1192
return FusedMoEParallelConfig(
    tp_size=1, ep_size=ep_size, ...)                       # L1192-L1208
```

读这段代码：

1. `use_ep = (flatten_tp > 1) and enable_expert_parallel`——只要任一并行轴 > 1 且 operator 设了 `enable_expert_parallel=True`，就开 EP。
2. **关键**：开 EP 后，`tp_size_inner = 1`（在每个 expert 内部不再做 TP），整个 TP×DP×PCP 的预算**全部塌缩到 EP**。

**为什么 collapse？** 因为 expert 们在功能上**独立**——一个 token 路由到 expert i，**不需要** expert j 的权重。把 TP 留在 expert 内部意味着每个 expert 内部都要 all-reduce（如同 Ch08 的 col→row pair），但 expert 与 expert 之间在做不相关的事——**对每个 expert 内部 TP 不会让它们彼此协作**。所以宁可把 TP 预算转到 EP（每张 rank 拿不同的 expert），节省一次 all-reduce，代价是 expert 的矩阵不再切分。

但是！production 仍然可以同时用 EP 和 TP——`config.py` 的 collapse 规则是 *默认* 的，但 `FusedMoE` 的 weight loader 还允许进一步在 expert 内做 TP（`MergedColumnParallelLinear` + `RowParallelLinear`）。**Demo §3.4 的 (8, 4) 行就是这种情况**——8-way EP 把 64 个 expert 切到 8 张 rank（每张 8 个 expert），又在每个 expert 内 4-way TP（每个 $1408 \\times 2048$ 矩阵切 4 份）。这种用法在 DeepSeek-V3 / Qwen3-MoE 实际部署里常见——大模型既要 EP 省专家显存、又要 TP 省每个 expert 的矩阵显存。

### 9.5.4 关于 EP+TP 的 chain-break

第 7 章的 chain-break 是 *"vLLM 没有 radix tree，是 chain hash + flat dict"*。第 8 章的 chain-break 是 *"col→row pair 一次 all-reduce"*。第 9 章的 chain-break：

> **`_EP` 与 `_TP` 正交。一次 MoE forward 是：**
>
> 1. **gate(h)** → router_logits（gate.weight 在 EP 内 replicated）
> 2. **Top-K 路由** → (topk_weights, topk_ids)
> 3. **EP-axis dispatch all-to-all**
> 4. **本地 expert FFN**（内部 TP-sharded，1 次 intra-expert all-reduce）
> 5. **EP-axis combine all-to-all**
>
> **2 次 all-to-all（沿 EP 轴）+ K × 1 次 all-reduce（沿 TP 轴，每个被激活的 expert 内一次）**——但因为 collapse，绝大多数 production deployment 把 tp_size 设为 1，所以 step 4 的 all-reduce 也是 1 次（intra-expert TP）或 0 次（无 intra-expert TP，即纯 EP 路径）。

结合 Ch08：MLP block 一次 col→row pair 一次 all-reduce。Ch09 把 MLP 替换成 MoE block——多了 2 次 all-to-all、加上 K 个被激活的 expert 各自 (可能的) 一次 intra-expert all-reduce。**通信成本曲线**变成：

$$
T_\\mathrm{comm}^\\mathrm{MoE\\;block} = 2 \\cdot T_\\mathrm{all\\_to\\_all} + K \\cdot \\mathbf{1}_{\\mathrm{tp\\_inner} > 1} \\cdot T_\\mathrm{all\\_reduce}^\\mathrm{tp\\_inner}
$$

跟 dense MLP block 的通信开销

$$
T_\\mathrm{comm}^\\mathrm{dense} = T_\\mathrm{all\\_reduce}^\\mathrm{tp}
$$

比，多了 2 次 all-to-all（沿 EP 轴）。代价是 expert 数增加 / 每 token 计算量减少。

### 9.5.5 Mini-mapping：EP×TP composition

| 我们的 | 源码 | 区别 |
|---|---|---|
| `ep_groups.py:L88-L150 .make()` | `config.py:L1082-L1209 .make()` | 我们简化 sp/quant 路径 |
| `ep_groups.py:L75-L86 flatten_tp_across_dp_and_pcp` | `config.py:L1071-L1079` 同 | 一致公式 |
| `ep_groups.py:L70-L72 use_all2all_kernels` | `config.py:L1019-L1020` 同 | 一致 gate（Trap F 的源码点） |
| `fused_moe_block.py:L286-L304 memory_per_rank_MiB` | 隐式（推导自 weight shapes） | 我们显式公式化 |
| 没有 quant method weight 路径 | `layer.py:L548+` 4 个 quant 路径 | 我们只走 unquantized |
| `(ep, tp) = (8, 4)` demo 单进程模拟 | `parallel_state.py:L1670+ + linear.py` 真 NCCL | 教学版 |

---

## 9.6 All-to-All 通信：AgRs 与 7 个 backend

### 9.6.1 打开 AgRsAll2AllManager 入口

源码定位：`instances/vllm/source/vllm/distributed/device_communicators/all2all.py:L40-L139`。这是 vLLM 的 7 个 all-to-all backend 里**最简单的一个**——也是教学 baseline：

```python
# vllm/distributed/device_communicators/all2all.py:L40+ (节选)
class AgRsAll2AllManager(All2AllManagerBase):
    """all_gatherv + reduce_scatterv 的最简实现"""
    def dispatch(self, hidden_states, topk_weights, topk_ids, ...):
        # L108-L121
        sizes = self.dp_metadata.get_chunk_sizes_across_dp_rank()
        gathered = self.dp_group.all_gatherv(
            [hidden_states, topk_weights, topk_ids], dim=0, sizes=sizes)
        return gathered
    def combine(self, summed_full, ...):
        # L130-L135
        sizes = self.dp_metadata.get_chunk_sizes_across_dp_rank()
        return self.dp_group.reduce_scatterv(summed_full, dim=0, sizes=sizes)
```

**3 件事在这 100 行里**：

1. **`dispatch = all_gatherv`**（不是真正的对称 `all_to_all`）。每张 rank 把自己的 hidden_states 加上 topk meta 全部 gather 到所有 rank——所有 rank 都看到完整的 [total_tokens, hidden] 张量。
2. **`combine = reduce_scatterv`**。每张 rank 算完自己拥有的 expert 的输出后，把全部 expert 的输出加和，然后按"每张 rank 原来贡献的 token 数"切回去。
3. **`_v` 后缀**——`all_gatherv` 不是 `all_gather`，`reduce_scatterv` 不是 `reduce_scatter`。每张 rank 贡献的 chunk 大小可能**不同**（每张 rank 的 token 数 = `dp_metadata.get_chunk_sizes_across_dp_rank()`）。`_v`（variable size）是这种不均匀情况的标准 NCCL API。

trap-B 的硬证据就在这里：vLLM 的"all-to-all"其实是 **all_gatherv + reduce_scatterv**——end state 跟真正的 all-to-all 一样，但 kernel sequence 不一样。这有什么关系？

- **真正的对称 `dist.all_to_all`**：每张 rank 直接给每张 rank 发自己拥有的那部分。一次 P×(P-1) 路通信。如果路由偏（rank 0 的 token 都路由到 expert 在 rank 7 上），rank 0 → rank 7 这条路要发很多东西，其他路几乎空闲——**通信不均衡**。
- **AgRs 路径**：先 all-gather 把 hidden_states 散到所有 rank（每张 rank 都拿到完整数据），各自做 expert FFN，再 reduce-scatter 把结果按原始 token-rank 关系切回去。**通信流量是固定的**（每张 rank 都拿到 P 倍的 token），不受路由偏影响——但 kernel 内存使用变大。

DeepEP HT/LL 走的是真正的对称 path（IBGDA + NVLink fused），它们效率更高但**对路由偏更敏感**——这是 production 部署需要 EPLB 的另一个原因。Trap-B 的精确表述：*"all-to-all 在算法层是对称的，cost ≈ all-reduce / 2；但在 vLLM 的 AgRs baseline 上 cost = all_gather + reduce_scatter，每张 rank 看到完整数据；DeepEP 是真对称但路由偏惩罚更大"*。

### 9.6.2 α-β cost 模型（与一个 caveat）

经典 Hockney α-β 模型：

$$
T_\\mathrm{AR} = 2 \\cdot \\frac{P-1}{P} \\cdot \\left( \\alpha + \\frac{S}{\\beta} \\right)
$$

$$
T_\\mathrm{A2A} = \\frac{P-1}{P} \\cdot \\left( \\alpha + \\frac{S}{\\beta} \\right)
$$

其中 $\\alpha$ 是 per-message latency floor（μs），$\\beta$ 是 per-rank 网络带宽（GB/s），$S$ 是 payload bytes。比值：

$$
\\frac{T_\\mathrm{AR}}{T_\\mathrm{A2A}} = 2
$$

**E12 的 caveat**：这个比值 **2.000** 是模型的恒等式，不是测量结果——只要 `alpha_beta_cost` 函数的两个常数（`2 * (P-1)/P` vs `(P-1)/P`）一摆，比值必然是 2。所以 demo §3.2 的 `ratio=2.000` 列：

```
NVLink (alpha=5μs, beta=250 GB/s, hidden=4096 bf16, P=8):
       128 tokens →  T_AR=  16.09μs   T_A2A=   8.05μs   ratio=2.000
      1024 tokens →  T_AR=  67.47μs   T_A2A=  33.74μs   ratio=2.000
      8192 tokens →  T_AR= 478.51μs   T_A2A= 239.26μs   ratio=2.000
     65536 tokens →  T_AR=3766.85μs   T_A2A=1883.42μs   ratio=2.000

IB (alpha=8μs, beta=50 GB/s):
       128 tokens →  T_AR=    50.70μs T_A2A=    25.35μs ratio=2.000
     65536 tokens →  T_AR= 18804.48μs T_A2A=  9402.24μs ratio=2.000
```

**writer 必须按 verbatim 复述这条 caveat**：`ratio=2.000` 是**模型恒等式**的展示，不是真实测量。真实 NCCL/DeepEP all-to-all 在偏斜路由下会偏离 2× —— hot rank 的 outbound queue 会被堵住。Trap-B 的精确表述要包含这个 caveat。

那为什么还要拿这个公式？因为**绝对值有意义**：NVLink 上 8192 token 的 dispatch 大约 240μs，IB 上同样大小是 1180μs——**5× 慢**。这是真实部署里跨节点 EP（IB）跟节点内 EP（NVLink）的量级对比，能让读者直观感受为什么 grouped Top-K 要把每 token 的 dispatch 扇出从 K 降到 topk_group。

### 9.6.3 7 个 backend 共享同一个接口

vLLM 有 7 个 all-to-all backend，全部继承自 `All2AllManagerBase`，全部实现同样的 `dispatch / combine` 接口：

| Backend | 源码行 | 用途 | Kernel |
|---|---|---|---|
| `AgRsAll2AllManager` | L40-L139 | 教学/兼容 baseline | `all_gatherv + reduce_scatterv`（NCCL 标准） |
| `DeepEPHTAll2AllManager` | L196-L256 | 跨节点高吞吐 | DeepEP IBGDA fused |
| `DeepEPLLAll2AllManager` | L257-L325 | 跨节点低延迟 | DeepEP LL fused |
| `NixlEPAll2AllManager` | L327-L440 | CPU staged path | NIXL transport library |
| `FlashInferNVLinkOneSidedManager` | L442+ | 节点内 NVLink 单边 | FlashInfer custom |
| `FlashInferNVLinkTwoSidedManager` | L500+ | 节点内 NVLink 双边 | FlashInfer custom |
| `MoriAll2AllManager` | L671+ | AMD/ROCm | Mori library |

**为什么这么多？** 因为 EP 部署场景太多样：

- 同节点 NVLink → FlashInfer 单/双边或 DeepEP HT。
- 跨节点 IB 高吞吐 → DeepEP HT 的 IBGDA。
- 跨节点 IB 低延迟（小 batch、低延迟服务）→ DeepEP LL。
- 没有 DeepEP 编译的环境 → AgRs 兜底。
- ROCm/AMD GPU → Mori。
- CPU offload 路径 → Nixl。

所有这些 backend 对 `FusedMoE` 来说是**透明的**——`get_ep_group().dispatch(...)` 一行调用，backend 由 `select_prepare_finalize_modular`（`fused_moe/all2all_utils.py:L60-L210`）在 init 时根据 `moe.moe_parallel_config.use_all2all_kernels` 和 `device_communicator` 选好。这是教科书的 strategy pattern——一个接口、多个实现、运行时选择。

**Trap F 的源码点**：`use_all2all_kernels = dp_size > 1 AND use_ep`（`config.py:L1019-L1020`）。**`dp_size==1` 时 fast path 是 `MoEPrepareAndFinalizeNoDPEP`**——`dispatch = identity`、`combine = identity`，**不 call backend manager**。所以 *"FusedMoE.forward 总是 dispatch→experts→combine"* 这句话是错的——`ep_size=1` 或 `dp_size==1` 时根本没有 all-to-all。

测试 `test_use_all2all_kernels_requires_dp_and_ep` pin 这点：`use_all2all_kernels` 当且仅当 `dp_size > 1 AND use_ep` 都为真。

### 9.6.4 Cross-chapter：EP 通信跟 Ch08 / Ch11 / Ch15+ / Ch27 怎么连

**与 Ch08（Tensor Parallelism）**：
- `_TP` group 在每个 expert 内部仍然可能 active（如果 `tp_size_inner > 1`）。两个 group 是 **正交** 的——`_EP` 和 `_TP` 的 rank list 互补，由 mesh `(pp, pcp, dp, tp)` 推出来。
- `MergedColumnParallelLinear` 和 `RowParallelLinear` 在 expert 内部还是 Ch08 的语义——一对 col→row 一次 all-reduce。

**与 Ch11（DCP/PCP）**：
- `parallel_state.py:L1670` 的 mesh 公式 `(pp, pcp, dp, tp)` 在 Ch11 里会扩展为 `(pp, pcp, dcp, dp, tp)`——多一个 DCP 轴（context parallelism）。EP group 仍然是补轴：`ep_size = world / (pp × pcp × dcp × dp × tp)`。Ch11 会扩这个 mesh 到 5D。
- `_EP` 与 `_DCP` 也是正交的——dcp 是 sequence 维度的切分，跟 expert 维度不冲突。

**与 Ch15+（Llama / 模型 zoo）**：
- Llama-3 / Qwen3-base 是 dense 模型，**`_EP is None`**。它们走 Ch08 的纯 TP 路径，没有 expert 概念。
- Mixtral-8x7B 是 §9.1 的 reference 现场——`MixtralMoE` 直接用 `FusedMoE(num_experts=8, top_k=2, use_grouped_topk=False)`。
- DeepSeek-V2 / V3 用 grouped Top-K 加 shared experts——§9.1 的另一个 reference 现场。

**与 Ch27（DeepSeek-V3.2 deep dive）**：
- `e_score_correction_bias` (`noaux_tc`) 的训练动机和数值表现会在 Ch27 详细分析。
- DeepEP 的 IBGDA fused kernel 内部细节也在 Ch27——这章只列出 `DeepEPHTAll2AllManager` 的存在和接口，不展开 kernel。
- EPLB 的 production policy（`vllm/distributed/eplb/policies.py` 的 bin-packing）——Ch27 的 case study 会拿真实 traffic 数据复盘。

### 9.6.5 Mini-mapping：all-to-all backend zoo

| Backend | 源码 | 用例 | 我们的对照 |
|---|---|---|---|
| AgRs | `all2all.py:L40-L139` | 兜底/教学 | `implementation/all2all_baseline.py:L65-L115` |
| DeepEP HT | `all2all.py:L196-L256` | 跨节点高吞吐 | (生产替代) |
| DeepEP LL | `all2all.py:L257-L325` | 跨节点低延迟 | (生产替代) |
| Nixl | `all2all.py:L327-L440` | CPU staged | (生产替代) |
| FlashInfer 单边 | `all2all.py:L442+` | NVLink 单边 | (生产替代) |
| FlashInfer 双边 | `all2all.py:L500+` | NVLink 双边 | (生产替代) |
| Mori | `all2all.py:L671+` | ROCm/AMD | (生产替代) |
| Backend 选择器 | `all2all_utils.py:L60-L210 select_prepare_finalize_modular` | 运行时 dispatcher | `implementation/fused_moe_block.py:L226-L267` 单进程模拟 |

---

## 9.7 7 个语言陷阱：一次性集中检查

每个陷阱 *"声明 → 错 → 为什么 → 源码证据"*，匹配 Ch07 §7.6.4、Ch08 §8.6.4 的模板。

### Trap A — "EP=N 给 N× 的容量，同样的 compute"
- **错**。EP 切的是**参数存储**，不是 compute-per-token。每 token 仍然只激活 K 个 expert，FLOPs = `K × per_expert_FLOPs`，跟 E 和 ep_size 无关。
- **为什么**：EP scales **memory**（每张 rank 的权重显存 = 总参数 / ep_size），latency **变大**（多 2 次 all-to-all），throughput-per-token 不变；只有总吞吐能因为模型变大、能 fit 进多 GPU 的总 HBM 而变高。
- **源码证据**：`fused_moe/layer.py:L378+` 把 expert 切到 rank；`router/fused_topk_router.py:L77` 选 K 与 E 无关。
- **Demo 证据**：§3.3 linear + ep=8 + 偏斜路由 → max/mean=3.251；round-robin 改善到 1.196 还是 < 1。EP 不能单独给 N× 吞吐。

### Trap B — "All-to-all 是对称的，所以 cost = all-reduce / 2"
- **错**。算法层是对称的，但 vLLM 的 AgRs baseline 是 `all_gatherv + reduce_scatterv`——end state 等价但 kernel 不同；DeepEP 是真对称 fused kernel 但**对路由偏更敏感**。
- **为什么**：α-β 模型的 ratio=2.000 是**模型恒等式**（demo §3.2 的 `ratio=2.000` 在所有 payload 上都是 2.000，因为公式里就是 2× 关系）。真实 all-to-all 在偏斜路由下偏离 2×——hot rank 的 outbound queue 堵塞。
- **源码证据**：`all2all.py:L99-L102` AgRs 的 dispatch 用 `all_gatherv` 配 per-rank `sizes`——`_v` 后缀表明 per-rank chunk 大小不同。
- **Demo 证据**：§3.2 `ratio=2.000` 在 4 个 payload size 上一字不差——这是公式的特性，不是测量。

### Trap C — "Experts 是独立的，所以 EP 扩展是 free 的"
- **错**。两个耦合效应破坏 free scaling：(1) 路由偏导致 hot expert 让某 rank 过载（trap-A）；(2) shared experts 在 `FusedMoE` 之外构造，**每张 rank 都有完整一份**——它们不参与 EP 切分。
- **为什么**：DeepSeek 的 `n_shared_experts > 0` 在 `models/deepseek_v2.py:L295-L317` 显式构造在 `FusedMoE` 之前，作为 `DeepseekV2MLP` 直接用 TP 切分（`MergedColumnParallelLinear` + `RowParallelLinear`）。EP 只对 routed experts 节省内存。
- **源码证据**：`models/deepseek_v2.py:L295-L317`；test `test_trap_C_shared_experts_DO_NOT_appear_in_FusedMoEBlock_memory` pin 内存模型不含 shared experts。

### Trap D — "EPLB 是免费的运行期 bolt-on"
- **错**。三道侧门：(1) `_EPLB` 是单独的 NCCL communicator（防 deadlock）；(2) quant method 必须声明 `supports_eplb`；(3) round-robin placement + EPLB 不兼容（fallback 到 linear）。
- **为什么**：EPLB 的 rebalance 广播跟 forward 的 dispatch all-to-all 共用 NCCL stream 会彼此 block；不同 quant method 的 weight loader 假设不同的 layout。
- **源码证据**：`parallel_state.py:L1700-L1719` 注释 *"to prevent deadlocks"*；`fused_moe/layer.py:L548-L557` `if not supports_eplb: raise NotImplementedError`；`fused_moe/layer.py:L168-L171` round_robin gating。
- **Demo 证据**：tester 的 `test_trap_D_eplb_separate_group_prevents_aliasing` pin `id(_EP) != id(_EPLB)`。

### Trap E — "Aux loss 在 vLLM 里平衡专家"
- **错**。vLLM 是 **inference-only**——MoE 推理路径里**没有 aux-loss 计算**。Aux loss 是 training 概念（Switch Transformer L_balance）；trained 模型的分布是训练期 aux loss 沉淀下来的，但部署时 aux loss 不再生效。
- **为什么**：`vllm/distributed/eplb/` 整目录 grep `\\.backward(|compute_aux_loss|router_aux_loss_coef|aux_loss\\s*=` 全部 0 匹配。EPLB 是**运行期统计 rebalance**，没有梯度。
- **精化（E10）**：vLLM 全树有两处名字撞车：`phimoe.py:router_aux_loss_coef` 是 HuggingFace config 的 stored constant、`vision.py:get_load_balance_assignment` 是图像 tile 的 helper——**不是** expert routing 的 aux loss。
- **源码证据**：tester 的 4 个 negative 测试覆盖 (a) 我们的 EPLB 模块 AST-strip 后没有 backward/optim、(b) `record_step` 显式 `.detach()`、(c) `vllm/distributed/eplb/` 5 个 .py 文件 0 匹配、(d) trap-E 名字撞车的两个 false positive 不在 MoE 推理路径里。

### Trap F — "FusedMoE.forward 总是 dispatch→experts→combine"
- **错**。`use_all2all_kernels = dp_size > 1 AND use_ep`——`dp_size==1` 时 `MoEPrepareAndFinalizeNoDPEP` 被选中，**dispatch 和 combine 都是 identity**，根本没有 all-to-all。
- **为什么**：单 DP replica + 单 rank（`ep_size=1`）就是普通 dense MLP 的等价形式——没必要走 all-to-all。
- **源码证据**：`config.py:L1019-L1020`；`prepare_finalize/no_dp_ep.py` 整文件实现 identity dispatch。
- **测试**：`test_use_all2all_kernels_requires_dp_and_ep`。

### Trap G — "Top-K 然后 softmax = softmax 然后 Top-K"
- **错** **当且仅当 `renormalize=False`**。在 `renormalize=True` 下两条路径**代数等价**：`softmax(g_i) / Σ_topk softmax(g_j) == softmax_topk(g)_i`。在 `renormalize=False` 下 vLLM 路径返回 softmax 尾部质量（`sum < 1`），而 topk-first-then-softmax 总是 `sum = 1`。
- **为什么**：renormalize 把 K 个权重重新归一化为概率分布；不开 renormalize 时它们就是 softmax 在 K 个 expert 上的概率质量——是全 E 概率的子集，<1。
- **源码证据**：`router/fused_topk_router.py:L94-L100`。
- **Demo 证据**：§3.1 `renormalize=False → sum range [0.2730, 0.6171]  mean 0.3899`——这正是 softmax 尾部质量，跟 topk-then-softmax 的 `sum=1` 是不同的数值。

---

## 9.8 验证：跑 demo 跟 lint

### 9.8.1 跑 demo

```bash
$ cd instances/vllm/artifacts/09-expert-parallelism
$ /home/zjq/.conda/envs/mujoco/bin/python implementation/demo.py
Ch09 Expert Parallelism — demo numerics (deterministic, seed=42)

========================================================================
§1 Top-K routing distributions (Mixtral scale, DeepSeek scale)
========================================================================
Mixtral (E=8, K=2):
  per_expert_count = [250, 285, 277, 243, 253, 272, 247, 221]
  ...
DeepSeek-V2 grouped (E=64, K=6, n_group=8, topk_group=3):
  max=131  min=78  mean=96.00  coverage=1.000
  ...
[full demo output ≈ 60 lines, see tests/demo-output.txt]
```

§3.1 ~ §3.5 跑出来的所有数字都跟本章正文里 verbatim 一致。

### 9.8.2 跑测试

```bash
$ /home/zjq/.conda/envs/mujoco/bin/python -m pytest tests/ \
      --ignore=tests/_legacy -q
204 passed in 2.24s
```

204 个测试全过。覆盖（按 module）：

| Module | 测试数 | 覆盖 |
|---|---|---|
| `test_routing.py` | 28 | fused_topk shape/dtype、renormalize 等价、grouped_topk mask、E11 trap-G |
| `test_expert_map.py` | 24 | linear/round_robin、-1 哨兵、E13 ep=1 短路 2-tuple |
| `test_ep_groups.py` | 27 | EP-vs-TP collapse、`use_all2all_kernels` gate、E15 _EP/_EPLB 对象身份不等 |
| `test_all2all_baseline.py` | 20 | all_gatherv/reduce_scatterv、α-β、AR/A2A 比 2.000、E12 model-tautology |
| `test_fused_moe_block.py` | 30 | SiluAndMul、ep=1 vs ep=4 等价、forward shape、Mixtral/DeepSeek 路由 |
| `test_eplb.py` | 32 | 初始 layout、record_step interval、hot 进冗余 slot、trap-E 4 个 negative |
| `test_mixtral_vs_deepseek.py` | 16 | MIXTRAL_8x7B/DEEPSEEK_V2_LITE config pinning、routing fingerprint |
| `test_integration.py` | 18 | §3.1~§3.5 verbatim、`mem ∝ 1/(ep×tp)`、ep ∈ {1,2,4,8} forward 等价 |
| `test_smoke.py` | 9 | 实现期 sanity（保留） |

### 9.8.3 跑 linter

```bash
$ python3 scripts/lint_formulas.py \
      instances/vllm/artifacts/09-expert-parallelism/narrative/chapter.md
🟢 No blocking issues
$ python3 scripts/lint_source_grounding.py \
      instances/vllm/artifacts/09-expert-parallelism/
✓ All grounding checks passed!
```

两个 linter 都过。

---

## 9.9 Source Mapping Table

| 我们的 | vLLM 源码 | 关系 |
|---|---|---|
| `routing.py:L26-L97 fused_topk` | `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L69-L113` | 复刻 softmax→topk→renormalize；fast path 是 Triton kernel |
| `routing.py:L65-L74 scoring_func` | `fused_topk_router.py:L94-L100` | 一致；softmax/sigmoid 两条路径 |
| `routing.py:L77 torch.topk(scores, k)` | `fused_topk_router.py:L100` | 一致；fast path 用 `ops.topk_softmax` |
| `routing.py:L89-L91 renormalize` | `fused_topk_router.py:L106-L110` | 一致；E11 trap-G 锚点 |
| `routing.py:L100-L178 grouped_topk` | `grouped_topk_router.py:L81-L162` | 复刻 group_score → topk_group → mask → topk |
| `routing.py:L141-L147 e_score_correction_bias` | `grouped_topk_router.py:L121-L132` | 复刻 V3 noaux_tc：选用 biased，权重用 unbiased |
| `routing.py:L181-L193 expert_load_counts` | `eplb_state.py:L210` per-expert load | 教学 helper，原版集成在 EplbState |
| `expert_map.py:L27-L88 determine_expert_map` | `fused_moe/layer.py:L70-L157` | 复刻 placement；返回 2-tuple vs 3-tuple（E13） |
| `expert_map.py:L46-L50 ep=1 短路` | `layer.py:L107-L109` | E12/E13：vLLM 返 3-tuple `(E, None, None)` |
| `expert_map.py:L64 -1 哨兵` | `layer.py:L117` | 一致；off-rank 标记 |
| `expert_map.py:L70 linear start_idx` | `layer.py:L119-L123` | 一致；包括 remainder 处理 |
| `expert_map.py:L77 round_robin arange` | `layer.py:L124-L131` | 一致 |
| `expert_map.py:L91-L103 get_compressed_expert_map` | `layer.py:L196-L214` | 一致；日志 helper |
| `ep_groups.py:L36-L150 FusedMoEParallelConfig` | `fused_moe/config.py:L998-L1209` | 复刻 dataclass + .make() collapse 规则 |
| `ep_groups.py:L70-L72 use_all2all_kernels` | `config.py:L1019-L1020` | 一致；trap-F 锚点 |
| `ep_groups.py:L75-L86 flatten_tp_across_dp_and_pcp` | `config.py:L1071-L1079` | 一致 |
| `ep_groups.py:L116-L132 use_ep=False 路径` | `config.py:L1175-L1189` | 一致 |
| `ep_groups.py:L133-L150 use_ep=True 路径` | `config.py:L1192-L1208` | 一致；EP 把 TP 收缩为 1 |
| `ep_groups.py:L188 _EP singleton` | `parallel_state.py:L1261` | 一致 |
| `ep_groups.py:L190 _EPLB singleton` | `parallel_state.py:L1273` | 一致 |
| `ep_groups.py:L218-L223 dense 模型 _EP=None` | `parallel_state.py:L1672-L1673` | E01：dense vs MoE 不对称 |
| `ep_groups.py:L233-L242 mesh transpose+reshape` | `parallel_state.py:L1670-L1696` | 一致；EP 是 dp×pcp×tp 补轴 |
| `ep_groups.py:L259-L267 _EPLB 单独 group` | `parallel_state.py:L1700-L1719` | 一致；W04 backpressure isolation 锚点 |
| `ep_groups.py:L272-L279 get_ep_group()` | `parallel_state.py:L1264-L1270` | 一致；MoE 断言 |
| `all2all_baseline.py:L36-L47 all_gatherv` | `parallel_state.py` GroupCoordinator.all_gatherv | 单进程模拟 |
| `all2all_baseline.py:L50-L62 reduce_scatterv` | `parallel_state.py` GroupCoordinator.reduce_scatterv | 单进程模拟 |
| `all2all_baseline.py:L65-L97 AgRsAll2AllManager.dispatch` | `device_communicators/all2all.py:L83-L121` | 复刻 baseline backend |
| `all2all_baseline.py:L99-L115 AgRsAll2AllManager.combine` | `all2all.py:L123-L136` | 一致；E13 caller 预求和 contract |
| `all2all_baseline.py:L123-L162 alpha_beta_cost` | `all2all.py` 没有显式公式 | 教学 helper；E12 model-tautology |
| `all2all_baseline.py:L165-L194 measure_dispatch_combine` | 集成测试 | 单进程 shape demo |
| `fused_moe_block.py:L34-L50 ExpertFFNWeights` | `fused_moe/layer.py:L222-L223` w13/w2 layout | 一致 layout |
| `fused_moe_block.py:L53-L60 silu_and_mul` | `vllm/model_executor/layers/activation.py SiluAndMul` | 一致 |
| `fused_moe_block.py:L63-L126 FusedMoEBlock.__init__` | `fused_moe/layer.py:L290-L605` | 教学版；省略 quant_method、router_factory 间接 |
| `fused_moe_block.py:L132-L156 _route` | `router/fused_topk_router.py:L149-L165` + `grouped_topk_router.py:L341-L351` | 复刻路由 dispatch |
| `fused_moe_block.py:L162-L220 _run_local_experts` | `prepare_finalize/naive_dp_ep.py:L104-L168` | 教学版；显式 expert_map 掩码 |
| `fused_moe_block.py:L226-L267 forward` | `fused_moe/layer.py:L1543-L1649` | 教学版；省略 quant_method.apply 间接 |
| `fused_moe_block.py:L286-L304 memory_per_rank_MiB` | 推自 weight shape 公式 | §9.5 不变量 |
| `eplb.py:L34-L66 EplbState 字段` | `eplb_state.py:L210-L300` | 教学版；省 async worker |
| `eplb.py:L55-L65 __post_init__ 初始 layout` | `eplb_state.py:L286-L304` | 一致 |
| `eplb.py:L81-L102 record_step` | `eplb_state.py:L223-L249` | 一致；E10 显式 .detach() |
| `eplb.py:L108-L136 _rearrange` | `eplb_state.py:L920+` rearrangement entrypoint | 教学贪心；生产用 `policies.py` bin-packing |
| `eplb.py:L142-L152 imbalance_ratio` | `eplb_state.py` 类似 metric | E16：边界 corner-case |
| `eplb.py:L155-L178 per_rank_load_from_logical_load` | `expert_map.determine_expert_map` 同逻辑 | 教学 helper |
| `eplb.py:L181-L211 make_skewed_routing` | (实验装置) | E14：seed=100+step pin |
| `mixtral_vs_deepseek.py:L48-L57 MIXTRAL_8x7B` | `models/mixtral.py:L132-L145` | 一致 config |
| `mixtral_vs_deepseek.py:L64-L75 DEEPSEEK_V2_LITE` | `models/deepseek_v2.py:L319-L341` | 一致 config |
| `mixtral_vs_deepseek.py:L78-L91 build_block` | (装载 helper) | 教学 |
| `mixtral_vs_deepseek.py:L94-L130 routing_fingerprint` | (诊断 helper) | demo §3.1 数据来源 |
| `demo.py:L65-L139 demo_routing` | demo §3.1 来源 | 一致；命中数 verbatim |
| `demo.py:L147-L188 demo_alpha_beta` | demo §3.2 来源 | E12 model tautology |
| `demo.py:L196-L236 demo_placement` | demo §3.3 来源 | trap-A 硬证据 |
| `demo.py:L244-L260 demo_mesh_memory` | demo §3.4 来源 | §9.5 不变量 |
| `demo.py:L268-L333 demo_eplb` | demo §3.5 来源 | E14 seed pattern |

**这张主表 49 行；加上前面 §9.2、§9.3、§9.5、§9.6 的四个 mini-mapping（18 + 7 + 6 + 8 = 39 行），总共 88 行 source mapping**——超过 v6 floor 的 10 行下限和 Ch08 baseline 的 122 行接近（Ch08 用了 75 行主 + 47 行 mini = 122 行，Ch09 是 49 + 39 = 88 行；考虑到 Ch09 surface 是 5 文件但 backend 多 7 个 manager，整体覆盖完整）。

---

## 9.10 总结

第 9 章把 vLLM 的 EP 拆成 5 文件协同 + 2 reference 现场 + 1 状态机。读完你能：

1. **从 router 数学上**：解释 vLLM 走 `softmax → topk → renormalize` 的两条路径（Mixtral fused_topk + DeepSeek grouped_topk），并知道 trap-G 在 `renormalize=False` 下才不交换。
2. **从架构上**：知道 vLLM 没有 `class ExpertParallel`——EP 是 `parallel_state.py` 的 `_EP` 单例 + `fused_moe/layer.py:L70-L157` 的 `determine_expert_map` + `fused_moe/config.py` 的 collapse 规则 + `device_communicators/all2all.py` 的 7 个 backend + `prepare_finalize/naive_dp_ep.py` 的 wiring。
3. **从内存模型上**：用 `mem_per_rank ∝ 1/(ep×tp)` 解释 demo §3.4 的 6 行内存表，理解 `(4, 2)` 与 `(8, 1)` 等价的根源。
4. **从负载均衡上**：知道 outline §4 的"梯度回传"是 training 概念（Switch Transformer L_balance），vLLM 没有；vLLM 真正在做的是 EPLB 运行期统计 rebalance（demo §3.5：`max/mean` 从 2.529 跳到 1.203 的 2.1× 改善）；EPLB 不 free，三道侧门（quant supports_eplb、单独 _EPLB group、与 round_robin 不兼容）。
5. **从通信上**：知道 vLLM 的 7 个 all-to-all backend 共享接口、AgRs 用 `all_gatherv + reduce_scatterv` 不是真对称、`use_all2all_kernels = dp_size > 1 AND use_ep` 才走 backend manager。

**chain-break**：`_EP` 与 `_TP` 正交。一次 MoE forward = gate（replicated）→ Top-K → all-to-all dispatch → 本地 expert FFN（内部 TP 一次 all-reduce 或 0 次） → all-to-all combine。两次 all-to-all 沿 EP 轴。

**七大语言陷阱**：A "EP=N 给 N× 容量" ✗、B "all-to-all = all-reduce/2" ✗、C "experts 独立所以 EP free" ✗、D "EPLB 是免费 bolt-on" ✗、E "aux loss 在 vLLM 平衡专家" ✗、F "FusedMoE.forward 总是 dispatch→experts→combine" ✗、G "softmax→topk = topk→softmax" ✗（仅 `renormalize=False` 下不等价）。

**Forward-pointer**：
- **Ch11（DCP/PCP）** 把 mesh 扩到 5D `(pp, pcp, dcp, dp, tp)`；EP 仍是补轴。
- **Ch15+（模型 zoo）** 把 Mixtral / DeepSeek-V2 / Qwen3-MoE 的 `MoE.__init__` 配置展开看每个变体怎么把 §9.1 的 `use_grouped_topk` flag、§9.3 的 placement strategy、§9.5 的 EP+TP 数字定下来。
- **Ch27（DeepSeek-V3.2 deep dive）** 把本章只列名字的 `e_score_correction_bias`、DeepEP IBGDA、`vllm/distributed/eplb/policies.py` 的 production rebalance solver 一一展开。

下一章（Ch10）继续 Part 2 的并行性篇章，讲 sequence parallelism 与 DeepSpeed-Ulysses 的关系——同样的 chain-break 模式（GroupCoordinator + 一对 collective + 一个 model-side use site），换一个并行轴。

---

> 章节版本：v6.0（Ch08 单 cycle APPROVED 之后的第二章）
> 源码 commit：vllm `98661fe`
> 公式 lint：🟢 No blocking issues
> 源码接地 lint：✓ All grounding checks passed!
> 测试：204 passed in 2.24s
> Mapping rows：49 主 + 39 mini = 88
> 字数：约 6500 字
