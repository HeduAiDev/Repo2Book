# 第9章：Expert Parallelism — MoE 的专家分布式

> 打开 `vllm/model_executor/layers/fused_moe/layer.py:71`。`determine_expert_map()` 回答
> 了 EP 的核心问题：**256 个 expert，8 张 GPU，每张 GPU 负责哪几个 expert？**
> 答案是 `round(n/8)` 个 contiguous block——或者 interleaved round-robin。两种映射策略引发
> 完全不同的负载均衡行为。

---

## 这章要做什么？

第 8 章的 TP 解决了"矩阵太大"的问题——把权重沿输入/输出维切分。但 MoE 带来一个新问题：**不是每个 token 都用所有 expert。** 只 touch top-k 个 expert，其他 expert 的计算可以跳过（稀疏激活）。

Expert Parallelism 利用这个稀疏性：把 256 个 expert 分布到 8 张 GPU 上，每张管 32 个。一个 token 需要 expert #17 → AllToAll 通信把它的 hidden_states 发到 GPU 2（管 #16-31）→ GPU 2 算完 → AllToAll 把结果发回来。

**核心 trade-off：TP 切权重（每张 GPU 参十几分之一），EP 切 expert（每张 GPU 算几分之一的 expert）。**

学完这章你能：
- 追踪 MoE 的完整数据流：Router → Dispatch → Expert Compute → Combine
- 理解为什么 EP 启用时 TP 被禁用——`FusedMoEParallelConfig.make()` 中 `tp_size = 1`
- 区分 AllToAll 的六种通信后端——从纯 NCCL 的 all_gatherv 到 DeepEP 的 RDMA kernel

---

## 9.1 Mixture of Experts：稀疏激活的计算

### Theory: 为什么不用一个大 FFN？

标准 Transformer 的 FFN layer：
$$
y = W_{\mathrm{down}} \cdot \mathrm{SiLU}(W_{\mathrm{up}} \cdot x)
$$

一个 FFN 要服务所有类型的 token——名词、动词、介词、代码、数学公式。用一个 FFN 处理所有这些 → 需要极大容量（intermediate_size / d_model 比例，通常是 8/3）。

MoE 的 insight：**用多个小的 FFN（expert），每个专门处理一类 token。** 通过 Router 选择 top-k 个 expert，计算稀疏化：

$$
y = \sum_{i \in \mathrm{topk}} g_i(x) \cdot \mathrm{Expert}_i(x)
$$

其中 $g_i(x)$ 是 Router 的输出权重（softmax 概率），$\mathrm{Expert}_i(x)$ 是第 i 个 expert 的 FFN。

**实际效果：** 同样的参数总量，MoE 的活跃参数只有 dense model 的 k/E 比例（例如 DeepSeek V3：256 expert，top-8 → 活跃 8/256 = 3.1% 参数）。

### Source Trail

打开 `vllm/model_executor/layers/fused_moe/layer.py:219`。`FusedMoE.__init__`：

```python
class FusedMoE(nn.Module):
    def __init__(self, num_experts, top_k, hidden_size, intermediate_size, ...):
        self.router = create_fused_moe_router(top_k=top_k, ...)
        self.experts = ...  # W1, W2 per expert
```

Router 的创建在 `layer.py:467`：

```python
self.router = create_fused_moe_router(
    top_k=top_k,
    global_num_experts=self.global_num_experts,
    ...
)
```

---

## 9.2 Router：Top-K 门控

### Source Trail

打开 `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:116`。

标准 Top-K Router 的逻辑：

```python
class FusedTopKRouter(BaseRouter):
    def _compute_routing(self, hidden_states, router_logits):
        router_probs = softmax(router_logits, dim=-1)       # [N, E]
        topk_weights, topk_ids = torch.topk(router_probs, self.top_k)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        return topk_weights, topk_ids
```

三个步骤：
1. **Softmax:** 把 `router_logits` 归一化——每个 expert 获得一个"这个 token 需要我"的概率
2. **Top-K:** 取概率最大的 k 个 expert——稀疏化
3. **Renormalize:** 选中的 k 个权重重新归一化，和为 1

**DeepSeek 的 Grouped Top-K（`grouped_topk_router.py:81`）：** 标准 top-k 在 256 个 expert 中选 8 个——这需要一个 256 路的 top-k。DeepSeek 把它分解为两步：先在 8 个 group 中选 top-k_g 个 group，再从每个选中的 group 中选 expert。两步筛选减少了路由开销。

### Theory: Load Balancing 的必要性

如果 Router 总是把 token 路由到相同的 8 个 expert——其他 248 个 expert 白占了显存。**Auxiliary loss** 惩罚这种不均衡：

$$
L_{\mathrm{aux}} = \alpha \cdot \sum_{e} f_e \cdot P_e
$$

其中 $f_e$ = expert $e$ 被路由到的 token 比例，$P_e$ = softmax 给出的平均概率。这个 loss 鼓励所有 expert 被均匀使用。

vLLM 还实现了 **EPLB（Expert Parallel Load Balancing）**——不是靠 loss，而是动态迁移热门 expert 的物理副本（`vllm/distributed/eplb/eplb_state.py:210`）。用到 `num_redundant_experts`——物理 expert 比逻辑 expert 多几个，最热门的 expert 复制。

---

## 9.3 Dispatch → Compute → Combine

### Source Trail

MoE 的完整通信流在 `FusedMoEKernelModularImpl.apply()`（`modular_kernel.py:1332`）中：

**Step 1 — Dispatch（AllToAll）：**

打开 `vllm/distributed/device_communicators/all2all.py:83`——`AgRsAll2AllManager.dispatch()`：

```python
# Naive path: all_gatherv of hidden_states + router data across EP group
def dispatch(self, hidden_states, topk_weights, topk_ids, ...):
    # 1. Gather all hidden_states to every GPU
    hidden_states = all_gatherv(hidden_states, group=self.ep_group)

    # 2. Gather router data (which tokens go to which experts)
    topk_weights = all_gatherv(topk_weights, group=self.ep_group)
    topk_ids = all_gatherv(topk_ids, group=self.ep_group)

    # 3. Permute: reorder tokens by target expert
    permuted_tokens = permute_by_expert(hidden_states, topk_ids, ...)
    return permuted_tokens
```

**Step 2 — Expert Compute（本地）：** 每个 GPU 拿到路由到它的 expert 的所有 token → 计算本地 expert FFN。

**Step 3 — Combine（AllToAll reverse）：** `all2all.py:123`——`AgRsAll2AllManager.combine()`：

```python
# reduce_scatterv: send results back to the GPU that originated each token
def combine(self, expert_output):
    return reduce_scatterv(expert_output, group=self.ep_group)
```

### Theory: 通信量分析

对于 $N$ 个 token、hidden_size=$d$、$E$ 个 expert、$K$ 路 top-k、$P$ 路 EP：

**Dispatch（AllGatherV）：** 每个 GPU 向所有其他 GPU 广播它的 token 的 hidden_states + router 数据。通信量 = $O(N \cdot d \cdot P)$——每个 GPU 看到 $N_{total} = N_{per\_GPU} \cdot P$ 的完整数据。

**Combine（ReduceScatterV）：** 计算完的结果被 scatter 回原来的 GPU。通信量 = $O(N \cdot d \cdot P)$。

**总量：** $O(2 \cdot N \cdot d \cdot P)$。对比 TP 的 AllReduce 是 $O(2 \cdot N \cdot d)$。EP 通信量是 TP 的 $P$ 倍——但 EP 让每张 GPU 的 compute 减少了 $K/P$ 比例（只有 $K/P$ 的 expert 需要算）。

`★ Insight ─────────────────────────────────────`
EP 的 trade-off 可以用一个简单的公式概括：如果 $K \cdot d_{expert} > d_{model}$（每个 token 的 expert compute 比模型维度大），EP 的通信开销是值得的——减少的 compute 大于增加的 AllToAll。对于典型 MoE（DeepSeek V3：d=7168, K=8, d_expert=2048——K·d_expert=16K > d=7K），EP 是净赢。对于小 MoE（K=2, d_expert=d/2）——TP 可能更好。
`─────────────────────────────────────────────────`

---

## 9.4 EP + TP 的组合

### Source Trail

打开 `vllm/model_executor/layers/fused_moe/config.py:1082`——`FusedMoEParallelConfig.make()`。

当 EP 启用时，关键赋值（`config.py:1194-1198`）：

```python
ep_size = tp_size   # EP 使用 TP group 的全部设备
ep_rank = tp_rank
tp_size = 1         # MoE 层不用 TP！
tp_rank = 0
```

**EP 禁用了 MoE 层的 TP。** 因为 Expert weight 在 EP 下不被切分——每张 GPU 拥有完整（所有参数）的少量 expert（比例 1/P），而不是部分参数的全体 expert（比例 tp_s/P）。两种策略分配的是不同的资源维度：TP 切参数，EP 切责任。

### Device Mesh 拓扑

打开 `vllm/distributed/parallel_state.py:1569-1575`——5D 设备网格：

```
all_ranks = [ExternalDP, DP, PP, PCP, TP]
```

EP group 通过 transpose DP 和 PP 维度来创建（`parallel_state.py:1673-1683`）：

```python
group_ranks = all_ranks.transpose(1, 2).reshape(-1, DP*PCP*TP).unbind(0)
```

EP group 大小 = `DP * PCP * TP`——在一个 PP stage 内，所有跨 DP 和 TP 的设备都属于同一个 EP group。这意味着 expert 不仅跨 TP rank 分布，也跨 DP rank——EP 是一种全局分布。

---

## 9.5 AllToAll 通信后端

### Source Trail

打开 `vllm/distributed/device_communicators/all2all.py`。

| 后端 | 类 | 机制 | 适用场景 |
|------|-----|------|---------|
| `allgather_reducescatter` | `AgRsAll2AllManager` (L40) | NCCL all_gatherv + reduce_scatterv | 默认；通用 NVLink/RDMA |
| `deepep_high_throughput` | `DeepEPHTAll2AllManager` (L196) | DeepEP NVLink Buffer | prefill（大量 token） |
| `deepep_low_latency` | `DeepEPLLAll2AllManager` (L257) | DeepEP batched dispatch | decode（低延迟） |
| `flashinfer_nvlink_two_sided` | `FlashInferNVLinkTwoSidedManager` (L442) | FlashInfer MnnvlMoe | NVLink-only 拓扑 |
| `mori` | `MoriAll2AllManager` (L671) | MoRI dispatch+combine fused | 交换优化 |
| `nixl_ep` | `NixlEPAll2AllManager` (L327) | NIXL RDMA buffer | 跨节点 RDMA |

**为什么需要多个后端？** AllToAll 通信模式在不同硬件上差异巨大。NVLink 的带宽是双向对称的——适合 all_gatherv 然后 reduce_scatterv。InfiniBand RDMA 的单向带宽可能不对称——需要不同的调度。DeepEP 用专门的 CUDA kernel 让 dispatch 和 combine 与 expert compute 重叠，把通信隐藏到计算下面。

---

## 9.6 EPLB：动态 Expert 负载均衡

### Source Trail

打开 `vllm/distributed/eplb/eplb_state.py:210`——`EplbState`。

**逻辑 vs 物理 Expert：**
- **逻辑 Expert：** 模型结构中的概念 Expert——256 个
- **物理 Expert：** 实例化在特定 GPU 上的 Expert——数量 = 256 + N_redundant

**记录负载：** 在 routing 时，Triton kernel `_eplb_map_and_record_i32_kernel`（`base_router.py:18`）用 `tl.atomic_add` 记录每个 expert 在每个 device 上的 token 数。

**重新排列（`eplb_state.py:657`——`EplbState.rearrange()`）：**
1. 每 `step_interval` 步（默认 3000），收集所有 rank 的负载窗口
2. AllReduce 得到全局负载指标
3. 策略层（`DefaultEplbPolicy`）调用 `rebalance_experts()`
4. Weight transfer——把热门 expert 复制到多个物理副本，淘汰冷门 expert
5. Commit 新的 logical→physical 映射

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `TopKRouter` | `fused_topk_router.py:L116` | Softmax + topk + renormalize；教育版 |
| `determine_expert_map()` | `layer.py:L71` | Linear + round_robin 策略一致 |
| `SimpleMoELayer` | `layer.py:L219` `FusedMoE` | 简化版：PyTorch loop vs vLLM fused CUDA kernel |
| `simulate_ep_dispatch_combine()` | `all2all.py:L83-L136` | 通信量分析；非真实 NCCL |
| `ep_tp_tradeoff_analysis()` | `config.py:L1082` `FusedMoEParallelConfig.make()` | 原创量化分析 |

---

## 验证

```bash
cd artifacts/09-expert-parallelism && python -m pytest tests/ -q
# 9/9 passed ✅
```

---

## 总结

- **MoE = 稀疏激活的 FFN。** Router 选 top-k expert → 只计算 k/E 的活跃参数。
- **EP 的通信流：Dispatch（AllToAll）→ Expert Compute → Combine（AllToAll reverse）。** 每个 GPU 广播其 token 并接收路由到其本地 expert 的 token。
- **EP 启用时 TP 被禁用。** Expert 权重不被切分——每张 GPU 只有 1/P 的 expert，但每个都完整的。
- **六种 AllToAll 后端——为了效率。** NVLink ↔ RDMA ↔ 纯 NCCL 的 trade-off 取决于拓扑。
- **EPLB 动态迁移热门 expert。** 物理 expert 比逻辑 expert 多——复制被高频使用的，释放被忽略的。

---

**下一章：** 第10章 — Multi-Token Prediction

MoE 用稀疏激活减少 compute。MTP 用一个相反的策略：用额外计算增加吞吐。通过一次 decode 预测多个 future token（MTP draft heads），接受率足够高以产生 net speedup。第 10 章将分析 MTP 的架构和 acceptance rate 的数学。

---

← 第8章 | 第10章 →
