# 第4章：Continuous Batching — vLLM 的动态调度系统

> 打开 `vllm/v1/core/sched/scheduler.py:352`。`schedule()` 方法的注释里有一句话是全章的钥匙：
> *"There's no 'decoding phase' nor 'prefill phase' in the scheduler."*
> 这句话定义了 Continuous Batching 的本质。本章就是解释它为什么是对的——以及它如何工作。

---

## 这章要做什么？

前三章讲了 Attention 怎么算（FlashAttention）、KV Cache 怎么存（PagedAttention）、KV Cache 怎么管理（KVCacheManager）。但这些部件需要一个人来指挥——在 GPU 上把有限的 token budget 分配给几十个并发的请求，同时管理它们的显存。

这个指挥就是 `Scheduler`。它的核心决策：**在每个 forward pass 中，调度哪些请求、每个请求推进多少个 token。**

这不是一个简单的 round-robin。不同请求在不同的状态——有的刚进来需要 prefill 整个 prompt，有的在逐 token decode，有的在 prefill 的中途（chunked prefill），还有的被特殊指令阻塞（structured output grammar, remote KV transfer）。Scheduler 要在 token budget 和显存的约束下最大化吞吐量。

学完这章你能：
- 打开 `scheduler.py:352`，对着源码解释 `schedule()` 的两个 phase 为什么先调度 running 再调度 waiting
- 从 bubble analysis 的数学出发，证明 continuous batching 比 static batching 的吞吐量提升
- 理解 chunked prefill ——如何把一个长 prompt 切成多个 step，与其他 decode 请求混合调度
- 解释 `RequestStatus` 的五种状态及其转换条件（`request.py:L310`）

---

## 4.1 Static Batching 的 Bubble 问题

### Theory: Bubble Analysis

在讲 vLLM 怎么做之前，先理解旧方法为什么不好。Static batching 把推理分成两个独立的阶段：

```
Phase 1 — Prefill:   所有请求的 prompt 一起处理
                      GPU 利用率高（大 batch 做 attention）
                      但：短 prompt 等长 prompt → BUBBLE

Phase 2 — Decode:    每个请求一次生成一个 token
                      GPU 利用率低（每次只处理 batch_size 个 token）
                      但：prefill 已经全做完了 → 大量算力空闲
```

用一个具体的例子量化。假设 8 个并发请求：

```
2 个长 prompt (2048 tokens) → 每个输出 128 tokens
6 个短 prompt (128 tokens)  → 每个输出 256 tokens
```

**Static batching:**
- Prefill phase: 等待最长的 prompt → 2048 steps。GPU 同时处理 8 个请求的 prefill——利用率尚可。
- Decode phase: 在生成阶段，每个 request 每次只处理 1 个 token → 8 tokens/step。而 GPU 可以处理 2048+ tokens/step。**GPU 利用率暴跌至 <1%。**
- 总 steps ≈ 2048 (prefill) + 256 (longest output) = 2304

**Continuous batching:**
- 没有独立的 prefill decode 阶段。每个 step 可以同时包含：
  - 新请求的 prefill chunk（例如 512 tokens）
  - 部分完成请求的另一个 prefill chunk
  - 正在 decode 的请求（每次 1 token）
- Token budget（典型值 2048）被动态分配：短 prefill 和 decode tokens 先占位，剩余的 budget 留给长 prefill。没有 bubble——GPU 几乎满负荷运作。
- 总 steps ≈ 400-800（取决于 chunk size）

**Bubble 的本质：** Static batching 的 GPU 利用率呈 U 型曲线——prefill 阶段高，transition 阶段骤降，decode 阶段长期低迷。Continuous batching 把这个 U 型曲线拉平——通过把 prefill 和 decode 混合在同一 step，消除阶段切换带来的空闲。

### Source Trail

vLLM 的 Scheduler 通过一个简单的抽象来消除 bubble：**每个请求不是"在 prefill"或"在 decode"，而是有一个 `num_computed_tokens` 计数器。** 这个计数器告诉 Scheduler "这个请求还有多少 token 没算完"。

打开 `scheduler.py:353-362`，看注释：

> *"There's no 'decoding phase' nor 'prefill phase' in the scheduler. Each request just has num_computed_tokens and num_tokens_with_spec."*

这不是宣传口号——这是实现细节。`schedule()` 遍历 `self.running` 列表，对每个请求问同一个问题：**需要推进多少个 token？** 然后检查 token budget 和 KV cache。如果够，就推进。如果不够——对于 running 请求，preempt；对于 waiting 请求，不 admission。

---

## 4.2 Request 状态机

### Source Trail

打开 `vllm/v1/request.py:310`。`RequestStatus` 是一个 `IntEnum`：

```python
class RequestStatus(IntEnum):
    WAITING = 0       # 等待被调度
    RUNNING = 1       # 正在被处理
    PREEMPTED = 2     # 被驱逐，等待重新调度
    # -- PREEMPTED 是分界线 --
    FINISHED_STOPPED = 3         # 正常完成 (EOS/stop token)
    FINISHED_LENGTH_CAPPED = 4   # 达到 max_tokens
    FINISHED_ABORTED = 5         # 被客户端终止
    FINISHED_IGNORED = 6
    FINISHED_ERROR = 7
```

注意 `PREEMPTED = 2` 是"活着"和"死了"的边界——`is_finished()` 用 `status > PREEMPTED` 判断（`request.py:L332`）。PREEMPTED 的请求不是 finished——它们会被重新放回 waiting 队列，从零开始重新计算。

### Theory: 状态转换图

```
add_request()
    │
    ▼
WAITING ────────────────────────────────────────────────┐
    │                                                    │
    │ schedule() 分配成功                                  │
    ▼                                                    │
RUNNING                                                  │
    │                                                    │
    ├── 正常完成 ──→ FINISHED_STOPPED                     │
    ├── 达到上限 ──→ FINISHED_LENGTH_CAPPED               │
    ├── KV Cache OOM ──→ PREEMPTED ──→ [重新进入等待] ──┘
    └── 客户端终止 ──→ FINISHED_ABORTED
```

**PREEMPTED 的区别对待：** 被驱逐的请求的 `num_computed_tokens` 被重置为 0——之前算的 K、V 全部丢弃。当它重新被调度时，从头 prefill。这在显存极度紧张的情况下是一种无奈的策略。

vLLM 的 preempt 有两种策略（从 `scheduler.py:L478-511`）：
- **FCFS:** 驱逐 running 列表中的最后一个请求——即最新加入的那个
- **PRIORITY:** 驱逐优先级最低的请求

---

## 4.3 schedule() 的两个 Phase

### Source Trail

打开 `scheduler.py:388-846`。`schedule()` 把调度分成两个 phase：

**Phase 1 (L388-556): 调度 RUNNING 请求**

```python
for req in self.running:
    num_new_tokens = req.num_new_tokens  # total - computed
    if num_new_tokens <= 0:
        continue

    # Cap at token budget
    num_new_tokens = min(num_new_tokens, token_budget)

    # Allocate KV cache
    blocks = self.kv_cache_manager.allocate_slots(req, num_new_tokens)
    if blocks is None:
        self._preempt_request(req)  # ← 驱逐这个请求
        continue

    scheduled[req.request_id] = num_new_tokens
    token_budget -= num_new_tokens
```

**Phase 2 (L568-846): 调度 WAITING 请求**

```python
while self.waiting and token_budget > 0:
    req = self.waiting[0]
    num_new_tokens = req.num_new_tokens

    if self.enable_chunked_prefill:
        num_new_tokens = min(num_new_tokens, token_budget)
    elif num_new_tokens > token_budget:
        break  # 不做 chunked prefill, 整 prompt 放不下 → 停

    blocks = self.kv_cache_manager.allocate_slots(req, num_new_tokens)
    if blocks is None:
        break  # KV Cache 不够 → 停

    self.waiting.pop(0)
    self.running.append(req)
    req.status = RequestStatus.RUNNING
    scheduled[req.request_id] = num_new_tokens
    token_budget -= num_new_tokens
```

### Theory: 为什么先 Running 后 Waiting？

调度顺序对应两个优先级：

**Running 优先：** 如果一个请求已经在处理中——它已经分配了 KV Cache blocks，它的部分 prefill 已经完成——让它继续推进的成本更低。推迟它意味着它的 KV Cache 白占了（不能释放给其他请求），而它本身没有产出。**Running 请求有最高的"周转优先级"——投入了资源就尽快产出。**

**Waiting 在后：** 新请求还没占用 KV Cache。推迟它没有浪费——只是增加延迟。**Waiting 请求用剩余的 token budget 和 KV Cache 空间——running 吃不下的才给 waiting。**

这个优先顺序可以用排队论来形式化：如果 running 请求的处理时间（从 admission 到 completion）服从参数 $\mu$ 的指数分布，waiting 请求的到达率服从 $\lambda$ 的泊松过程，则 running-first 调度最小化系统的平均 KV Cache 占用 $\mathbb{E}[C] = \lambda / (\mu(\mu - \lambda))$——因为它确保 KV Cache 只在请求被积极处理时被占用。

`★ Insight ─────────────────────────────────────`
Running-first 的精妙之处在于它自然地处理了预抢占。如果一个 running 请求在 phase 1 被 preempt（因为 KV Cache 不够），schedule() 不会回退到 waiting 队列去填充这个空位——phase 2 只在 phase 1 没有 preemption 时才执行（`if preempted is not None: break`）。这意味着**一个 preempt 事件会创建一个"空白 step"——只有少量请求被调度，其余 token budget 被浪费。** 这给了被驱逐的请求一个机会在下个 step 中被重新 admission，而不需要和 waiting 队列中的新请求竞争。这本质上是 preempt 之后的"快速重试"机制。
`─────────────────────────────────────────────────`

---

## 4.4 Chunked Prefill

### Source Trail

打开 `vllm/v1/config/scheduler.py:84`：

```python
enable_chunked_prefill: bool = True
max_num_partial_prefills: int = 1      # 最多几个请求同时部分 prefill
long_prefill_token_threshold: int = 0  # 超过这个长度算"长 prefill"
```

在 `scheduler.py:684-690` 的应用：

```python
if self.enable_chunked_prefill:
    num_new_tokens = min(num_new_tokens, token_budget)
elif num_new_tokens > token_budget:
    break  # 不做 chunked prefill → 不能拆分 → 这个请求永远进不来!
```

### Theory: 为什么要 Chunk？

一个用户的 128K 超长 prompt 到达。不做 chunked prefill：必须一次性完成整个 prefill → 需要 128K 的 token budget → token budget 只有 2048 → **请求永远无法被调度。**

Chunked prefill 把这个 128K 切成 64 个 2K-token 的 chunk，分布在 64 个 step 中。每个 step 中，除了这个长 prefill 的一个 chunk，Scheduler 还同时调度其他请求的 decode token 或短 prefill。

**代价：** 长 prompt 的 time-to-first-token (TTFT) 增加了——切成 64 步意味着 64 个 forward pass 后才开始生成第一个 token。**收益：** 在相同的时间内，GPU 同时服务了几十个其他请求——整体吞吐量大幅提升。

**Chunk size 的控制：** `max_num_partial_prefills=1` 限制了同时进行 chunked prefill 的请求数。为什么要限制？每个 chunked prefill 请求的 attention 是 compute-heavy 的（长序列的 QK^T 计算）。如果有 10 个请求同时在 chunked prefill，GPU 的 compute-bound——会饿死 decode 请求（每个只需要 1 token 的 attention）。这是 latency-throughput trade-off 在调度参数上的体现。

---

## 4.5 Token Budget 的计算

### Source Trail

`scheduler.py:L107-111`：

```python
self.max_num_scheduled_tokens = (
    scheduler_config.max_num_batched_tokens  # 典型值 2048-8192
)
```

`max_num_batched_tokens` 是 GPU 一次能处理的 token 总数的硬件上限——受 model 的 max_position_embeddings 和 GPU 显存的共同约束。Scheduler 的 token_budget 从这个值开始，每调度一个请求就减去它需要的 token 数，直到 budget 耗尽或 KV Cache 满。

### Theory: Token Budget 的经济学

Token budget 定义了 GPU 一次 forward pass 的"购买力"。每个请求"购买"一定数量的 token 处理能力：

- Decode 请求买 1 token（最便宜）
- 短 prefill chunk 买 128-512 tokens
- Chunked prefill 买 `min(剩余prompt, token_budget)` tokens

Scheduler 的目标是最大化每个 step 的"有效产出"：优先把 budget 花在低边际成本的 decode 请求上（每个 1 token，但释放一个请求的完成路径），剩余的预算填给 prefill chunks。

**为什么不是先填 prefill？** 因为 prefill 比 decode 贵得多——prefill 需要 O(seq²) 的 attention 计算（或 O(seq) with FlashAttention），而 decode 只需要 O(seq) 的 KV 加载 + O(1) 的新 token 计算。优先 decode 意味着用固定的 token budget 推进更多请求的完成——通过释放 KV Cache 占用（完成的请求释放 blocks），间接为 waiting 请求腾出空间。

---

## 4.6 完整调度循环

(本节综合 `scheduler.py:352`, `scheduler.py:974`, `scheduler.py:1290` 的逻辑，展示完整循环。)

Scheduler 在 vLLM 的运行时循环中的位置：

```
1. Scheduler.schedule()
   → 决定哪些请求推进多少个 token
   → 返回 SchedulerOutput

2. ModelRunner 执行 forward pass
   → 构建 batch (attention metadata + block_table + token IDs)
   → GPU forward

3. Scheduler.update_from_output()
   → 处理新生成的 token
   → 检查 stop conditions
   → 释放完成请求的 KV Cache
   → 更新 num_computed_tokens

4. 回到步骤 1
```

一个请求的完整生命周期：

```
add_request("r1", prompt=4096 tokens, max_tokens=256)
  │
  ├─ Step 1:  waiting → schedule() 分配 2048 tokens prefill chunk → RUNNING
  ├─ Step 2:  running → schedule() 分配剩余 2048 tokens → prefill 完成
  ├─ Step 3:  running → schedule() 分配 1 token (decode)
  ├─ Step 4:  running → schedule() 分配 1 token (decode)
  │  ...
  ├─ Step 258: running → schedule() 分配 1 token (decode)
  │            → update_from_output() 检测到 max_tokens=256
  │            → status = FINISHED_LENGTH_CAPPED
  │            → free KV Cache blocks
```

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `RequestStatus` | `vllm/v1/request.py:L310` | 核心状态枚举完整保留；未实现 streaming/remote KV 等高级状态 |
| `Request` | `vllm/v1/request.py` | 保留核心字段（num_computed_tokens, status）；简化了 spec decode/sampling params |
| `ContinuousBatchingScheduler.schedule()` | `scheduler.py:L352` | 两个 phase 调度逻辑一致；简化了 encoder/spec decode/preempt 策略 |
| `_preempt_request()` | `scheduler.py:L952` | 核心 preempt 逻辑保留（free blocks, reset tokens, remove from running） |
| `update_after_step()` | `scheduler.py:L974 + L1290` | 保留核心更新逻辑；简化了 spec decode rejection 和 stop condition |
| Chunked prefill cap | `scheduler.py:L684-L690` | 逻辑一致——cap at token_budget |
| `bubble_analysis()` | 原创分析——量化 static vs continuous 的差距 | 不直接对应 vLLM 源码 |
| `SimpleKVCacheManager` | `vllm/v1/core/kv_cache_manager.py:L225` | 简化版；无 block pool、LRU、prefix cache |

---

## 验证

```bash
cd artifacts/04-continuous-batching && python -m pytest tests/ -q
# 12/12 passed ✅
```

---

## 总结

- **Continuous Batching 消除了 static batching 的 bubble。** 不存在独立的 prefill/decode 阶段——每个请求用 `num_computed_tokens` 追踪进度。Scheduler 每 step 决定每个请求推进多少 token。
- **Running-first 调度最小化 KV Cache 占用。** 已经投入了显存的请求优先推进；新请求用剩余 budget。
- **Chunked Prefill 让长 prompt 不阻塞短请求。** 切成多个 step 的 chunk，和 decode token 混合调度。
- **Preempt 是必要之恶。** 当 KV Cache 耗尽时，驱逐请求是最干净的回退路径——代价是驱逐的请求重算所有 K、V。

---

**下一章：** 第5章 — GPU 显存管理系统

Scheduler 的 `allocate_slots()` → `None` 的路径告诉我们显存不够了。第 5 章将深入 GPU 显存的三级分配器——PyTorch CachingAllocator → vLLM BlockPool → KVCacheManager——理解每一级如何保护下一级不超支。

---

← 第3章 | 第5章 →
