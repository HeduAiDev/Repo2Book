# Chunked Prefill — 长 Prompt 的切片调度

> 打开 `vllm/v1/core/sched/scheduler.py:678`。三行代码定义了整个 chunked prefill 机制：
> ```python
> if 0 < threshold < num_new_tokens:
>     num_new_tokens = threshold
> ```
> 一个 128K token 的 prompt 被切成 64 个 2K chunk，**和其他请求的 decode token 交叉调度**。
> 长 prompt 的 TTFT 只多了一点点，短请求几乎不被阻塞——这就是 chunked prefill 的全部价值。

---

## 这章要做什么？

第 4 章讲了 continuous batching 的 scheduling 循环——running 优先、waiting 填剩余 budget。但有一个问题悬而未决：**如果一个 waiting 请求的 prompt 是 128K token，token budget 只有 2048——怎么调度？**

不做 chunked prefill：这个请求的 prefill 放不下 token budget → 它一直被跳过 → **starvation**。

做 chunked prefill：prompt 被切成 64 个 chunk，每个 step 处理 2K token。长 prompt 和其他请求的 decode 交替执行。

这章从 `scheduler.py:678` 的三行阈值 cap 出发，推导 chunked prefill 如何影响系统的 latency-throughput Pareto frontier。

学完这章你能：
- 解释 `long_prefill_token_threshold` 的三个触发点和 `enable_chunked_prefill` 的 guard 逻辑
- 量化 TTFT vs throughput 的 trade-off——长 prompt 的 TTFT 涨了多少，短请求的 TTFT 降了多少
- 理解 `scheduler_reserve_full_isl` 为什么是 chunked prefill 的安全阀——防止 over-admission

---

## 三个决策点

### Source Trail

打开 `vllm/config/scheduler.py:70-84`，chunked prefill 的全部配置：

```python
enable_chunked_prefill: bool = True       # L84 — 主开关
long_prefill_token_threshold: int = 0     # L80 — 0=disabled; >0=每步上限
max_num_scheduled_tokens: int = 2048      # token budget 总量
scheduler_reserve_full_isl: bool = True   # L140 — 准入检查
```

在 `scheduler.py` 中有三个决策点使用这些参数：

**决策点 1 — Running 请求（L413-L414）：**
```python
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold
```
一个已经在 running 的请求如果还有大量未计算的 prefill token——限制它这次只推进 threshold 个。

**决策点 2 — Waiting 请求（L678-L680）：**
```python
threshold = long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
```
新请求的 prompt 超过阈值——切成 chunk，只 admission threshold 个 token。

**决策点 3 — Chunked prefill guard（L684-L690）：**
```python
if not enable_chunked_prefill and num_new_tokens > token_budget:
    break  # 不做 chunk → 整个 prompt 放不下 → 拒绝 admission
```
如果 chunked prefill 关了，prompt 超过 token budget → 这个请求永远进不来。如果开了，prompt 会被 cap 到 budget。

### Theory: 三个决策点的语义

这三个点分别保护三种不同的资源：
1. **L413-L414 保护 token budget**——running 请求不能垄断 budget（给其他 running 和 waiting 留空间）
2. **L678-L680 保护延迟公平性**——新请求的 TTFT 被 cap 到 threshold 步以内
3. **L684-L690 保护 availability**——没有 chunk 的话大 prompt 直接 starve

---

## TTFT vs Throughput 的量化

### Theory: 一个具体例子

**场景：** 1 个 128K token 的 prompt + 8 个 128 token 的 prompt。每个输出 256 token。Token budget = 2048。

**不做 chunked prefill：**
```
Step 1-63: 长 prompt prefill (128K / 2K = 64 steps，但 decoder-only 需要 63 完整 prefill steps)
           8 个短请求在 waiting 中空等 — 0 throughput
Step 64+:  所有请求 decode (每步 9 tokens)
Total: ~320 steps
短请求 TTFT = 63 steps ← 等了 63 步才开始生成第一个 token
```

**做 chunked prefill（threshold=2048）：**
```
Step 1:   长 chunk(2048) + 8 短 prefill(8×128=1024) → budget 几乎用完
Step 2:   长 chunk(2048) + 8 decode(8×1=8) → budget 充足
...
Step 64:  长 chunk(最后一批) + 8 decode(8×1=8)
Step 65+: 所有请求 decode
Total: ~320 steps（总步数相近）
短请求 TTFT = 1 step ← 第一个 step 就被 admit 并开始 prefill！
```

**核心观察：** Total steps 几乎一样（~320），但短请求的 TTFT 从 63 步降到 1 步——**降低了 63×**。长请求的 TTFT 从 63 步涨到 64 步——**涨了不到 2%**。

`★ Insight ─────────────────────────────────────`
Chunked prefill 是一个近乎 Pareto-optimal 的优化：它大幅改善了短请求的延迟（TTFT 降低 10-100×），而对长请求的延迟影响微乎其微（TTFT 增加 <5%）。原因在于长请求的 prefill 本身就是 O(seq²) 的 compute——切成 chunk 后每个 chunk 的 compute 是 O(chunk²)，总 compute 几乎不变（Σ chunk² ≈ seq² 因为交叉项被消除）。Compute 没增加，只是分布更均匀了。
`─────────────────────────────────────────────────`

---

## scheduler_reserve_full_isl：防止 Over-Admission

### Source Trail

打开 `scheduler.py:753`：

```python
full_sequence_must_fit = (
    scheduler_reserve_full_isl
    and request.num_computed_tokens == 0
)
kv_cache_blocks = kv_cache_manager.allocate_slots(
    request, num_new_tokens, ...,
    full_sequence_must_fit=full_sequence_must_fit,
)
```

`full_sequence_must_fit=True` 意味着：**在 admission 第一个 chunk 之前，验证整个 prompt 的 KV Cache 需求能被满足。** 不是只检查第一个 chunk——而是整个 prompt。

### Theory: 为什么需要这个？

不做这个检查的后果：Scheduler 持续 admit 新请求的 chunk。KV Cache 逐渐被填满——但每个请求只完成了 prompt 的一部分。当 KV Cache 满了，**所有正在进行的 chunked prefill 都无法继续**——它们都需要新 block。

这就像餐厅不检查总人数就不断让客人入座——每个人都点了一部分菜，但厨房满了之后所有人都吃不到下一道菜。`scheduler_reserve_full_isl=True` 确保在客人入座前检查"整桌人的菜能不能放进厨房"。

---

## discard_request_mask：非最终 Chunk 的 Dummy Token

### Source Trail

打开 `vllm/v1/worker/gpu_model_runner.py:1928-1933`：

```python
self.discard_request_mask[:num_reqs] = (
    optimistic_seq_lens_cpu[:num_reqs] < num_tokens_np
)
```

对于非最终 prefill chunk 的请求（prompt 还没全部处理完），这个 mask 被设为 True。意味着：**这个请求这一步的 sampled token 是 dummy——丢弃它。**

Model runner 仍然对所有请求做 sampling——为了代码简洁——但 generator offset 在采样后被回滚（L3418-3424）：

```python
gen.set_offset(gen.get_offset() - 4)
```

这只是工程上的一个简化——统一处理所有请求比在 sampling 时分叉更容易维护。代价是几微秒的额外 sampling 计算——完全被 GPU 的 attention compute 主导。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `ChunkedPrefillConfig` | `config/scheduler.py:L70-L84` | 保留核心参数；未实现 `max_num_partial_prefills`（v1 scheduler 不用） |
| `ChunkedPrefillScheduler.schedule()` | `scheduler.py:L388-L846` | 三个决策点的阈值 cap 逻辑一致；简化了 KV Cache allocation |
| `ttft_vs_throughput_analysis()` | 原创量化分析 | 演示长/短请求在 chunked prefill 下的 TTFT 差异 |

---

## 验证

```bash
cd artifacts/04-chunked-prefill && python -m pytest tests/ -q
# 7/7 passed ✅
```

---

## 总结

- **Chunked prefill = 三个决策点的阈值 cap。** Running 请求的 L413-L414、waiting 请求的 L678-L680、guard 的 L684-L690。
- **TTFT 几乎免费的优化。** 短请求 TTFT 降低 10-100×，长请求 TTFT 增加 <5%。Compute 总量不变——只是分布更均匀。
- **scheduler_reserve_full_isl 是安全阀。** 防止 over-admission——在 admit 第一个 chunk 前验证整个 prompt 能否放进 KV Cache。
- **非最终 chunk 的 token 是 dummy。** discard_request_mask 标记——model runner 采样后回滚 generator offset。

---

← 第4章 Continuous Batching | 第5章 GPU Memory Management →
