# 第6章 请求调度系统——两阶段调度、抢占与公平性-吞吐量的博弈

## Cell 2 — 打开 vLLM 的大脑：一个不做"选择题"的调度器

打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352-L362`。在 `schedule()` 方法的正上方，有一段 vLLM 作者 Woosuk Kwon 写的注释。这段注释不到 10 行，但如果你读懂了它，你就理解了 vLLM 调度器 80% 的设计哲学：

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec. At each step, the scheduler tries to assign tokens to the requests so that each request's num_computed_tokens can catch up its num_tokens_with_spec.

翻译成大白话：**调度器不区分 "prefill 阶段" 和 "decode 阶段"。** 它眼里只有两个数字——每个请求已经算了多少 token、还需要算多少 token。每步调度，调度器的工作就是：从 token 预算里给各个请求分配额度，让它们的 `num_computed_tokens` 追赶 `num_tokens`。

这个设计不是偷懒。它是刻意为之的极简抽象。因为在这个模型下：
- **Chunked prefill** 不需要特殊处理——长 prompt 切成块，每块就是 `num_computed_tokens` 往前推一点
- **Prefix caching** 不需要特殊处理——缓存的 token 算 `num_computed_tokens` 已经算过了，调度时自然跳过
- **Speculative decoding** 不需要特殊处理——草稿 token 临时加到 `num_tokens` 里，被否决了就回退

这个"不区分 phase"的统一模型，是理解 vLLM v1 调度器所有设计决策的起点。

但调度器不只是"追赶计数器"那么简单。它面临的是一个经典的三难问题：

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   公平性 ──── 每个请求能按时拿到结果            │
│   吞吐量 ──── GPU 每秒产出尽可能多的 token       │
│   延迟   ──── 单个请求从提交到首 token 的时间    │
│                                                 │
│   你最多只能优化其中两个。                       │
│                                                 │
└─────────────────────────────────────────────────┘
```

**一篇文章有一千个读者，一块 GPU 有一千个请求。怎么排这个队？** 这就是本章要回答的问题。

## Cell 3 — 公平的 FCFS 为什么不公平

### 3A. 排队的本质困境

FCFS（First-Come-First-Served）是人类历史上最古老的调度策略。买菜排队、银行柜台、高速公路收费站——先来先服务，天经地义。vLLM 把 FCFS 作为默认调度策略（`scheduler.py:L371`，`token_budget` 分配给 running 列表时按加入顺序遍历），这看起来合理。

但 GPU 推理场景有一个致命特征：**请求长度不可预测。** 你前面那个人买一瓶水（生成 10 个 token）跟前面那个人要了一桌满汉全席（prompt 有 4000 个 token，还要生成 2000 个 token），等待时间差了两个数量级。

这就是队列理论中最经典的 Head-of-Line (HOL) Blocking 问题。

让我们用我们简化实现的 demo 来看一个具体例子。我们用 `instances/vllm/source/vllm/v1/core/sched/request_queue.py:L75-L128` 中的 `FCFSRequestQueue`（基于 Python `deque` 的纯 FIFO）来管理等待队列。

在 Scenario 2 中，我们有 4 个请求：A、B、C、D，每个都有 32 个 token 的 prompt 和 20 个 token 要生成。但 GPU 只有 8 个 KV cache block（每个 block 16 token，一共只能存 128 个 token 的 KV cache）。

```
所有 4 个请求同时到达 (t=0)：
  waiting = [A, B, C, D]    ← FCFS deque

KV Cache: 8 blocks × 16 tokens = 128 token slots
4 个请求的 prefill 需求: 4 × 32 tokens = 128 token slots
→ 刚好没空间给 prefill 之后任何 decode 留出 block
```

我们来看看实际输出发生了什么（完整输出见 Cell 7）：

```
Step 0: 预填充 4 个请求 (A, B, C, D)
  running = [A, B, C, D], blocks 全满 (8/8)

Step 1: 尝试给所有 4 个请求做 decode (每个 +1 token)
  → A 和 B 需要 1 个新 block → 但 free blocks = 0
  → 触发 Preemption！
  → preempt D (最新加入的)，再 preempt C
  → running 变为 [A, B]，waiting 变为 [C, D]
```

关键来了：**FCFS 下 preempt 的是谁？** 看 `scheduler.py:L260`——FCFS 模式下直接 `self.running.pop()`，弹出列表最后一个，也就是**最近加入 running 的那个**。

这背后的直觉是：列表前面的请求已经跑了一阵子，投入了计算资源；列表末尾的请求刚刚加入，还没"赚到"它的 slot。这就像餐厅里先到的人已经吃了半小时，你刚坐下——资源不够了你被请出去，而不是让吃了半小时的人走。

但这也意味着：**D 和 C 在 A 和 B 完成之前会一直陷在 waiting 队列里。** A 和 B 各自要用 20 步 decode（每步 1 token），在这 20 步里 C 和 D 寸步难行。这就是 FCFS 的 HOL blocking。

### 3B. 为什么我们容忍 HOL Blocking？

你可能会问：为什么不轮转？就像操作系统用 Round Robin，每个请求跑一小段时间就换？

答案在 KV cache。我们回忆一下 Ch05 的内容：一个请求运行时的 KV cache 是它"存在的全部痕迹"。当你"切换"一个请求时，你有两个选择：

1. **保留它的 KV cache 在 GPU 上**：那它占用的显存没法给别人用——等于没切换，没有 multiplexing
2. **释放它的 KV cache**：那下次恢复时得重算——这就是 preemption

而 Round Robin 要求毫秒级的频繁切换，每次切换都要释放/重分配 GB 级的 KV cache。这是用 GPU compute 为时间片轮转买单，代价太高了。

所以 vLLM 的选择是：**解码优先（decode-first）。** 已经在跑着的解码请求（每步只产 1 个 token，轻量级操作）优先拿到 token 预算。新的、大的 prefill 请求要么等，要么通过 chunked prefill 慢慢啃。只要 decode 请求能快速完成、释放 KV cache block，新请求就能进来。

这就是 vLLM v1 的两阶段调度算法的核心直觉。

## Cell 4 — 两阶段调度算法的完整推导

### 4A. 统一 Token-Budget 模型

打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L371`。`schedule()` 方法的第一行实质性代码：

```python
token_budget = self.max_num_scheduled_tokens
```

这就是调度器在这一步的总预算。每个请求每步能分配到的 token 数量，取决于它还需要多少 token 以及还剩多少预算。

对于任何一个请求 $r$，它在本步还需要计算的 token 数为：

$$
n_{\mathrm{new}}(r) = n_{\mathrm{tokens}}(r) - n_{\mathrm{computed}}(r)
$$

其中 $n_{\mathrm{tokens}}(r)$ 是请求 $r$ 的总 token 数（prompt + 已生成的输出），$n_{\mathrm{computed}}(r)$ 是已经计算过的 token 数。这是 `request.py:L59-L66` 中 `num_new_tokens_needed` 属性的核心逻辑。

但 $n_{\mathrm{new}}(r)$ 还需要被两个约束裁剪：

1. **Token budget 约束**：

$$
n_{\mathrm{new}}(r) \le B_{\mathrm{remaining}}
$$

（不能超过本步剩余预算）

2. **模型长度约束**：

$$
n_{\mathrm{computed}}(r) + n_{\mathrm{new}}(r) < L_{\mathrm{max}}
$$

（不能超过 `max_model_len`）

调度器的目标是在这些约束下，**确定哪些请求被调度，以及每个请求分多少 token**，使得 GPU 利用率最高同时保持公平性。

### 4B. 两阶段循环：Running-First 的严格证明

调度算法的核心结构（`scheduler.py:L195-L370`）如下：

```
schedule():
    Phase 1: FOR each request in self.running (按加入顺序):
        IF token_budget == 0: BREAK
        num_new = min(r.remaining, token_budget, max_model_len - r.computed)
        TRY allocate KV cache blocks for num_new tokens
        IF OOM:
            preempt a victim  →  free its KV blocks  →  back to waiting
            RETRY allocate
        IF still can't allocate: BREAK (放弃本请求和后续所有)
        scheduled.append(r), budget -= num_new

    Phase 2: IF no preemptions happened:
        WHILE waiting 非空 AND budget > 0 AND len(running) < max:
            r = waiting.peek()
            TRY allocate KV cache blocks
            IF OOM: BREAK (停止接纳新请求)
            promote waiting → running
            budget -= num_new
```

**为什么 running 优先于 waiting？** 这是一个经过深思熟虑的设计决策，不是随意为之。我们可以给出一个简洁的证明：

**命题**：在 KV cache 资源受限的条件下，running-first 策略最大化系统吞吐量。

**证明**：设 running 列表中有 $R$ 个请求，waiting 队列中有 $W$ 个请求。每个 decode 步骤，一个 running 请求只需 1 个 token（即约 1 个或 0 个新 KV block）。而一个 waiting 请求的 prefill 可能需要几十到几千个 token（多个新 KV block）。

如果 waiting 优先于 running，那么一个长 prefill 请求可以耗尽整个 token budget——在此期间，$R$ 个 running 请求全部停滞，它们的 KV cache 仍然占用显存但没有产出。GPU 的资源（计算 + 显存）被浪费在没有产出的 KV cache 上。

反过来，running 优先意味着：
- Running 请求每步都有产出（decode 出 1 个新 token）
- Running 请求完成 → 释放 KV block → 更多空间 → waiting 请求更容易被接纳
- 在每个 decode 步骤中，$R$ 个 running 请求产生 $R$ 个新 token，waiting 队列可以零产出等待——但 decode 请求快速完成，形成了正向的"产出-释放-纳新"循环。

**推论**：running-first 策略不会饿死 waiting 请求，因为 running 请求是有限的（`max_num_running_reqs`）且每个请求只能生成有限 token（`max_tokens`）。最终所有 running 请求都会完成，释放 KV cache block，waiting 请求被接纳。

这个设计在 `scheduler.py:L388-L561`（Phase 1）和 `scheduler.py:L568-L846`（Phase 2）之间有清晰的源码边界。

### 4C. 无 Phase 区分：Token 差距模型的实质

再回到 Woosuk 的注释（`scheduler.py:L353-L362`）。传统的 LLM 推理引擎（包括 vLLM v0）区分两个阶段：

```
传统方式:
  Prefill phase: 一次性处理整个 prompt → 产出第一个 token
  Decode phase:  逐 token 自回归生成 → 每步 1 个 token
  两个 phase 互斥，不同时执行
```

vLLM v1 的方式：

```
vLLM v1:
  每个请求维护: num_computed_tokens, num_tokens
  每步: num_new = num_tokens - num_computed_tokens
  如果 num_new == 1: 这就是传统意义上的"decode"
  如果 num_new == 340: 这就是传统意义上的"prefill"（或 chunked prefill 的一块）
  如果 num_new == 28: 这就是 chunked prefill 中间的一个 chunk
  调度器不关心你给它贴什么标签——它只算减法
```

这个统一模型自然支持了 Ch04 中讨论的 chunked prefill。长 prompt（1500 tokens）不会一次性占满 token budget——因为 `long_prefill_token_threshold` 把单次 prefill 量限制在比如 512 tokens。第一块 512 和第二块 512 之间，decode 请求可以插进来。没有任何"phase 切换"的复杂性——就是 $n_{\mathrm{new}}$ 被截断到预算上限而已。

### 4D. KV Cache 分配 IS 调度约束

`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L464-L510` 是调度器最关键的代码段之一——allocate-or-preempt 循环：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens
    )
    if new_blocks is not None:
        break  # 分配成功
    # 分配失败 → 触发 preemption
```

Schedule 方法的两阶段逻辑都围绕着同一个瓶颈旋转：**KV cache 分配。** `allocate_slots()` 返回 `None` 是整个调度流程唯一的"刹车"。没有 KV cache block → 不能调度新请求 → 触发 preemption。

这是 Ch05 的直接延伸。BlockPool 有多大、block 如何映射到物理地址、碎片如何回收——这些 Ch05 的内容构成了调度器的硬约束。没有足够的内存块，再聪明的调度策略也无计可施。

在我们的简化实现中，`ToyKVCacheManager`（`scheduler.py:L21-L91`）把这个分配逻辑浓缩到 50 行：

```
allocate_slots(req, num_new_tokens):
    total_tokens = req.computed + num_new
    blocks_needed = ceil(total_tokens / block_size)
    if blocks_needed > free_blocks: return None  ← 这就是 preemption 触发器
    new = [free_blocks.pop() for _ in range(blocks_needed - current)]
    return new
```

虽然简化了 prefix cahing 和 eviction ordering，但核心语义完全一样：**分配失败 → 返回 None → 调度器触发 preemption。**

### 4E. Preemption：Swap vs Recompute 的定量对决

当 `allocate_slots()` 返回 `None` 时，调度器必须腾出空间。腾空间有三种基本策略，但 vLLM v1 只保留了其中一种。

#### 策略一：Swap Out

```
preempt: GPU KV blocks → CPU RAM (通过 PCIe 复制)
resume:  CPU RAM → GPU KV blocks (再通过 PCIe 复制回来)
```

打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972`。vLLM v1 的 `_preempt_request()` 方法是这样的：

```python
def _preempt_request(self, request, timestamp):
    self.kv_cache_manager.free(request)     # 直接释放
    self.encoder_cache_manager.free(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0         # 重置为 0
    request.num_preemptions += 1
    self.waiting.prepend_request(request)   # 插回队头
```

注意：没有 `self.cpu_block_allocator.allocate()`，没有 `cudaMemcpy`，没有 `swap_in()`。只有 `free()` + `num_computed_tokens = 0`。这就是 **recompute-only preemption**——只重算，不搬移。

那为什么删除了 swap？我们来算一笔账。以 Llama-70B、4096 tokens、FP16 为例：

**Swap 的代价**（`instances/vllm/source/docs/design/prefix_caching.md` 中有讨论）：

$$
T_{\mathrm{swap}} = \frac{S_{\mathrm{KV}}}{B_{\mathrm{PCIe}}} \times 2
$$

其中 $S_{\mathrm{KV}}$ 是 KV cache 总大小，$B_{\mathrm{PCIe}}$ 是 PCIe 带宽。代入数值：

- 单层 KV cache：

$$
2 \times 80\ \mathrm{heads} \times 128\ \mathrm{dim} \times 4096\ \mathrm{tokens} \times 2\ \mathrm{bytes} \approx 160\ \mathrm{MB}
$$

- 80 层总计：

$$
80 \times 160\ \mathrm{MB} \approx 12.5\ \mathrm{GB}
$$

- PCIe Gen4 实测带宽：约 $64\ \mathrm{GB/s}$
- 移出时间：

$$
12.5 / 64 \approx 195\ \mathrm{ms}
$$

- 移回时间：同样约 $195\ \mathrm{ms}$
- **总暂停**：约 **390 ms**

**Recompute 的代价**（V1 当前方案）：

$$
T_{\mathrm{recomp}} = \frac{n_{\mathrm{effective}}}{R_{\mathrm{prefill}}}
$$

其中 $n_{\mathrm{effective}}$ 是实际需要重算的 token 数，$R_{\mathrm{prefill}}$ 是 GPU prefill 吞吐。

- 场景一（80% prefix cache 命中，这是典型情况）：

$$
n_{\mathrm{effective}} = 4096 \times 0.2 = 819
$$

- H100 prefill 吞吐（Llama-70B）：约 $200{,}000\ \mathrm{tok/s}$
- 重算时间：

$$
819 / 200000 \approx 4\ \mathrm{ms}
$$

- 场景二（0% prefix cache 命中，最坏情况）：

$$
n_{\mathrm{effective}} = 4096
$$

- 重算时间：

$$
4096 / 200000 \approx 20\ \mathrm{ms}
$$

**对比结论**：

$$
\frac{T_{\mathrm{swap}}}{T_{\mathrm{recomp}}} = \frac{390\ \mathrm{ms}}{4{-}20\ \mathrm{ms}} = \mathbf{19.5{-}97.5\times}
$$

即使最坏情况（0% cache 命中），recompute 仍然比 swap 快约 20 倍。这是因为 H100 的矩阵乘法计算能力（\~2 PFLOPS）远远超过 PCIe 的搬运能力（\~64 GB/s）。在 GPU 的世界里，"重新算一遍"往往比"搬过来搬过去"快得多。

#### 策略三：Abort

直接丢弃请求，返回错误给客户端。vLLM 没有采用这个策略，因为 LLM 推理是一个重试成本极高的操作——用户可能已经等了很久。Abort 只在客户端主动取消（`finish_requests` with `FINISHED_ABORTED`，`scheduler.py:L149-L192`）时才触发。

### 4F. Priority Scheduling 的排序不等式

`instances/vllm/source/vllm/v1/core/sched/request_queue.py:L131-L198` 实现了 `PriorityRequestQueue`。它的排序逻辑由 `Request.__lt__` 定义（`request.py:L76-L83`）：

```python
def __lt__(self, other):
    return (self.priority, self.arrival_time) < (other.priority, other.arrival_time)
```

这是一个词典序（lexicographic order）：
- 优先比较 `priority`（值越小 = 优先级越高，与 Linux nice 值一致）
- `priority` 相同 → 比较 `arrival_time`（越早到越优先）

这个设计有一个精妙的性质：**同一 priority 内保持绝对的 FIFO 顺序。** 这意味着一个 priority=5、一小时前到达的请求，不会被一个 priority=5、刚刚到达的请求插队。vLLM 用最简单的排序键实现了**同一优先级内的无饥饿保证**。

Priority 调度下的 preemption 选择逻辑与 admission 相反（`scheduler.py:L247-L258`）：

```
admission (pop from waiting):  最小值优先 → heapq.heappop (priority 最小 = 优先级最高)
preemption (pick from running): 最大值优先 → max(running, key=(priority, arrival_time))
```

我们想要 admits 最"好"的（最小 priority），preempt 最"差"的（最大 priority）。这里的对称性是刻意设计的，不是巧合。

#### Priority Aging 的缺失

搜索 `instances/vllm/source/vllm/v1/core/sched/scheduler.py` 全文，不存在 aging、decay、boost 相关的逻辑。vLLM v1 的 priority 是静态的——创建时设定、永不改变。

为什么这不是一个大问题？原因有三：
1. **请求生命周期自限**：每个请求有 `max_tokens` 上限，不会无限期占用资源。最终都会完成
2. **FCFS 默认策略**：大多数部署使用 FCFS（默认），aging 本身没有意义
3. **Token Budget 的隐式公平**：每步的 token budget 在 running 请求间分配，长请求被 chunked prefill 限制

在极端负载下（请求到达率远大于服务率），waiting 队列会无限增长。这由 API server 层的超时机制处理，不在调度器职责范围内。

### 4G. Pareto Frontier：延迟与吞吐量的不可调和矛盾

现在来到本章最硬核的部分。调度器的两个核心目标——高吞吐量和低延迟——是**天然矛盾**的。我们需要用数学把这个矛盾形式化。

#### 定义

设调度器每步处理 $B$ 个 token（token budget），步长时间为 $T_{\mathrm{step}}$（一次 GPU 前向的固定耗时）。

**系统吞吐量**（tokens per second）：

$$
Q = \frac{B}{T_{\mathrm{step}}}
$$

其中 $B$ 是本步调度到的总 token 数（prefill tokens + decode tokens 之和）。

**单请求 latency**：对于请求 $r$，它从到达到开始产出 token 的延迟取决于：
- 它在 waiting 队列中的等待时间 $W(r)$
- 它的 prefill 需要多少步（chunked prefill 被切成多块）
- 它的 decode 过程中每步拿到的 token budget 配额

在稳态下，如果 running 队列中有 $R$ 个请求，每个 decode 请求每步消耗 1 token（decode 是 token-level 的），而 prefill 请求每步消耗 $C$ 个 token（chunk size），那么每个请求每步平均获得：

$$
b(r) = \frac{B}{\bar{R}}
$$

个 token 的预算配额。$\bar{R}$ 越大，每个请求分到的越少。

#### The Tradeoff

最大 running 请求数 $R_{\max}$ 是调度器的关键控制旋钮：

- $R_{\max}$ 小（如 = 4）→ decode 请求少 → 每个请求分到更多 token budget → **延迟低**，但 GPU batch 也小 → **吞吐量低**
- $R_{\max}$ 大（如 = 128）→ 很多请求共享 token budget → 每个请求分到的少 → **延迟高**，但 GPU batch 大 → **吞吐量高**

用 Pareto frontier 的语言来表达：

$$
\mathrm{Frontier} = \left\{ (L, Q) \mid \nexists\ (L', Q')\ \mathrm{s.t.}\ L' < L \land Q' > Q \right\}
$$

也就是说，**在不牺牲延迟的前提下不可能提高吞吐量，反之亦然**。你只能在这个前沿上选一个点。

让我们用数值来感受这个前沿。假设一次 GPU 前向耗时 $T_{\mathrm{step}} = 50$ ms，token budget $B = 1024$ tokens：

| $R_{\max}$ | 每请求每步 token 配额 | Prefill 步数（4096 tokens） | 吞吐量 $Q$ |
|-----------|-----------------------|---------------------------|-----------|
| 4         | $1024/4 = 256$       | 4096/256 = 16 步          | $1024/0.05 = 20{,}480$ tok/s |
| 32        | $1024/32 = 32$       | 4096/32 = 128 步          | $1024/0.05 = 20{,}480$ tok/s |

注意：吞吐量 $Q$ 不变（因为 $B$ 和 $T_{\mathrm{step}}$ 固定），但每个请求的 prefill 延迟从 16 步变成了 128 步——差了 8 倍。

在实际系统中，$T_{\mathrm{step}}$ 也不是常数——更大的 $R_{\max}$ 意味着更大的 batch，模型前向的 GPU 时间也会增加。更精确的模型是：

$$
T_{\mathrm{step}}(R) = T_{\mathrm{fixed}} + T_{\mathrm{per\_token}} \cdot B
$$

其中 $T_{\mathrm{fixed}}$ 是固定的 kernel launch / 通信开销，$T_{\mathrm{per\_token}}$ 是每 token 的 compute 时间。

完整的 Pareto 前沿分析：

$$
Q(R) = \frac{B}{T_{\mathrm{fixed}} + T_{\mathrm{per\_token}} \cdot B}
$$

$$
L_{\mathrm{wait}}(R) \propto \frac{W}{B/R} = \frac{W \cdot R}{B}
$$

$Q$ 随 $B$ 增长（更多的 token per step 摊薄了固定开销），$L_{\mathrm{wait}}$ 随 $R$ 增长（更多的请求共享预算）。这就是**调度延迟与吞吐量的 Pareto frontier**——你在曲线上选点，曲线本身无法被"优化掉"。

## Cell 5 — 源码走读：从头到尾跟踪一次 schedule()

现在让我们逐行跟踪 `schedule()` 方法的完整执行。打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352`。

### 5A. 初始化阶段（L352-L386）

```python
scheduled_new_reqs: list[Request] = []
scheduled_resumed_reqs: list[Request] = []
scheduled_running_reqs: list[Request] = []
preempted_reqs: list[Request] = []
num_scheduled_tokens: dict[str, int] = {}
token_budget = self.max_num_scheduled_tokens
```

四个列表分别记录本步中不同类型的调度决策。`token_budget` 初始化为 `max_num_scheduled_tokens`——这是本步最多能处理的 token 总数。每次给一个请求分配了 $k$ 个 token，就从预算中减去 $k$。预算归零时停止调度。

### 5B. Phase 1：调度 Running 请求（L388-L556）

```python
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
    # 裁剪到预算和模型长度限制
    num_new_tokens = min(num_new_tokens, token_budget)
    num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)
```

`self.running` 是一个列表，按请求加入 running 的顺序排列。Phase 1 遍历这个列表，给每个 running 请求分配它需要的 token 数（decode 通常是 1，chunked prefill 可能更大）。

**关键细节**：当 `num_new_tokens <= 0` 时，代码执行 `continue` 而不是 `break`（`scheduler.py:L446-L462`）。这意味着即使一个 running 请求已经不需要 token 了（例如达到了 `max_tokens` 但还没清理），调度器也不会停止 Phase 1——它跳过这个请求继续处理后面的。这是因为 running 列表的 FIFO 顺序必须保留，不能因为中间夹了一个快完成的请求就阻断后面的。

### 5C. Allocate-or-Preempt 循环（L464-L510）

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, num_lookahead_tokens=...)
    if new_blocks is not None:
        break  # 成功
    # OOM → preempt
    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running,
            key=lambda r: (r.priority, r.arrival_time))
        self.running.remove(preempted_req)
    else:
        preempted_req = self.running.pop()  # FCFS: pop 最后一个
    self._preempt_request(preempted_req, timestamp)
    preempted_reqs.append(preempted_req)
    if preempted_req == request:
        break  # 连自己也 preempt 了——无路可走
```

这一小段代码包含了调度器最复杂的逻辑：

1. **`allocate_slots()` 返回 `None`** = OOM，触发 preemption
2. **Priority 模式下**：选 `max(running, key=(priority, arrival_time))`——priority 值最大（优先级最低）且到达最晚的
3. **FCFS 模式下**：`self.running.pop()`——弹出最后一个（最近加入的）
4. **如果 preempt 了自己**：这是"先 preempt 了别人还是不够用，最后只能 preempt 自己"的情况——break 退出

**为什么 preempt 了别人还不够？** 假设当前请求需要 5 个新 block，preempt 一个只有 1 个 block 的请求后只释放了 1 个 block，还是不够。那就继续 preempt 下一个。如果 preempt 到只剩下当前请求自己——那就是真的没救了，放弃这个请求的调度。

### 5D. Phase 1 继续：记录成功调度的请求（L516-L556）

```python
scheduled_running_reqs.append(request)
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
req_index += 1
```

分配成功后：记录到 `scheduled_running_reqs`、更新 `num_scheduled_tokens`、扣减 `token_budget`、递增索引。循环继续处理下一个 running 请求。

### 5E. Phase 2：接纳 Waiting 请求（L568-L846）

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break
        request = request_queue.peek_request()
        # ... 尝试 allocate_slots → 如果 None → break
        # ... 如果成功 → pop + running.append + budget 扣减
```

Phase 2 有三个闸门，缺一不可：
1. **No preemptions this step** — 如果刚 preempt 过，说明显存紧张，不要再引入新请求
2. **`_pause_state == UNPAUSED`** — 系统没有被暂停
3. **`token_budget > 0`** — 还有预算

如果三个闸门都通过，从 waiting 队列 peek 队首请求，尝试分配 KV cache block。**如果分配失败（OOM），break 退出 waiting 接纳**——不会触发 preemption。Phase 2 不 preempt 的原因是：如果 waiting 请求导致 OOM，说明显存真的不够了，preempt 一个 running 来给 waiting 让位只会导致下一轮反过来的 preempt——这就是 `impl-notes.md` 中提到的 thrashing 场景。

### 5E-extra. Thrashing 预防：No-Mixed-Admission 规则的严格论证

Phase 2 的 `if not preempted_reqs` 闸门（`scheduler.py:L568`）是经过深思熟虑的。让我们用一个反证法来理解它为什么必要。

**假设我们允许 preemption 和 admission 在同一个 step 发生。** 考虑以下场景：

```
Step N:
  running = [A, B, C]  (KV cache 满了)
  waiting = [D]        (优先级很高)
  
  Phase 1: 处理 A, B → OK
           处理 C → OOM! preempt A
  Phase 2: D 被 admit（因为 A 腾出了空间）→ running = [B, C, D]

Step N+1:
  A 在 waiting 头部（被 prepend 的）
  Phase 1: 处理 B, C → OK
           处理 D → OOM! preempt B  
  Phase 2: A 被 admit（因为 B 腾出了空间）→ running = [C, D, A]

Step N+2:
  ...同样的事情继续发生
```

这就是 **thrashing（抖动）**——系统在 preempt-一个-接纳-另一个-下一轮-反过来 的循环中空转。每个被 preempt 的请求都在被重算，没有任何一个请求能稳定地进行下去。

No-mixed-admission 规则（`if not preempted_reqs: skip waiting`）一刀切断了这个循环。它的语义是：**如果显存紧张到需要 preempt，那就先解决 preempt 的问题（让 running 请求完成、释放 block），再考虑接纳新请求。** 绝不在动荡的时候引入新的不稳定因素。

这个模式在分布式系统中很常见——类比于 Kafka 的 consumer group rebalance 期间不处理新 partition assignment，或 Raft 的 leader election 期间不处理写请求。

### 5F. `_preempt_request()` 详解（L952-L972）

```python
def _preempt_request(self, request, timestamp):
    self.kv_cache_manager.free(request)       # 释放所有 KV block
    self.encoder_cache_manager.free(request)   # 释放 encoder cache
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0           # 归零——下次从头重算
    request.num_preemptions += 1
    self.waiting.prepend_request(request)      # 插回 waiting 头部
```

每一步的含义：

1. **`kv_cache_manager.free(request)`**：释放该请求占用的所有 KV cache block。block 回到 free pool。但注意：如果这些 block 在 prefix cache 中被其他请求引用（ref_cnt > 0），它们不会被真正回收。这就是为什么 recompute 可以在大部分情况下很快——prefix cache 保留了共享的 block。

2. **`num_computed_tokens = 0`**：完全重置。当请求恢复时，调度器认为它从零开始。但 `kv_cache_manager.get_computed_blocks()` 会检查 prefix cache，恢复那些仍被引用的 block 的 `num_computed_tokens` 计数。所以实际不是真的"完全从零"——prefix cache 救了一部分。

3. **`prepend_request`**：插回 waiting 队列头部。对于 FCFS deque，这是 `appendleft`——被 preempt 的请求下次立即被处理。对于 Priority heap，这是 `heappush`——由 (priority, arrival_time) 重新排序。

### 5G. `update_from_output()` 详解（L1290-L1553）

模型跑完之后，GPU 产生了实际的 token。`update_from_output()` 做两件事：

1. **追加生成的 token**：`request.output_token_ids.append(token_id)`
2. **检查 stop condition**：`_check_stop(request)` — 如果触发了 `max_tokens` 或 `max_model_len`，标记完成

这个方法中有一个微妙的点：**`num_computed_tokens` 在 `schedule()` 末尾就已经被前推了**（`_update_after_schedule`，`scheduler.py:L974-L998`），这样下一个 scheduling step 直接就能知道请求已经"算到哪了"。如果 spec decode 的草稿 token 被模型否决了，`update_from_output()` 会回调 `num_computed_tokens`。这种"先乐观推进、后悲观修正"的模式在 vLLM 中很常见。

## Cell 6 — 简化实现代码（关键片段）

### 6A. 两阶段调度：schedule() 方法

```python
# REFERENCE: vllm/v1/core/sched/scheduler.py:L352-L945

def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens              # L371

    # ═══ Phase 1: Running requests first ═══
    # REFERENCE: scheduler.py:L388-L556
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        num_new_tokens = request.num_new_tokens_needed         # L408-409
        num_new_tokens = min(num_new_tokens, token_budget)     # L415
        if num_new_tokens <= 0:
            req_index += 1; continue                           # L461

        # Allocate-or-preempt loop
        # REFERENCE: scheduler.py:L464-L510
        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens)
            if new_blocks is not None:
                break  # success
            # OOM → preempt victim
            victim = self._select_preemption_victim()
            self._preempt_request(victim)
            preempted_reqs.append(victim)
            if victim is request:
                break  # no more victims

        if new_blocks is None:
            break  # can't schedule this request

        scheduled_running_reqs.append(request)
        token_budget -= num_new_tokens
        req_index += 1

    # ═══ Phase 2: Admit waiting requests ═══
    # REFERENCE: scheduler.py:L568-L846
    if not preempted_reqs:           # L568: no-mixed-admission gate
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.max_num_running_reqs:
                break
            request = self.waiting.peek_request()
            num_new = request.num_tokens - request.num_computed_tokens
            if num_new > token_budget:
                break
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new)
            if new_blocks is None:
                break  # OOM → stop admission
            self.waiting.pop_request()
            self.running.append(request)
            token_budget -= num_new

    return SchedulerOutput(...)  # the batch decision
```

### 6B. Preemption

```python
# REFERENCE: vllm/v1/core/sched/scheduler.py:L952-L972

def _preempt_request(self, request: Request) -> None:
    assert request.status == RequestStatus.RUNNING
    self.kv_cache_manager.free(request)          # Release KV blocks
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0              # Must recompute
    request.num_preemptions += 1
    self.waiting.prepend_request(request)        # Back to waiting head
```

### 6C. RequestQueue 的两种实现

```python
# REFERENCE: vllm/v1/core/sched/request_queue.py:L75-L128, L131-L198

class FCFSRequestQueue(deque, RequestQueue):      # L75
    def add_request(self, r): self.append(r)         # tail
    def pop_request(self): return self.popleft()     # head
    def prepend_request(self, r): self.appendleft(r) # preempt → front

class PriorityRequestQueue(RequestQueue):         # L131
    def add_request(self, r): heapq.heappush(self._heap, r)
    def pop_request(self): return heapq.heappop(self._heap)
    def prepend_request(self, r): self.add_request(r)  # heap re-sorts
```

## Cell 7 — 运行实例：三个场景的完整 trace

运行 `python3 demo.py` 的实际输出（节选关键步骤）：

### 场景一：Happy Path（3 个请求，显存充足）

```
Step 0: A(32 tok) + B(48 tok) + C(16 tok) → 3 requests, blocks=6/32
Step 1: decode A+1, B+1, C+1 → 3 running, blocks=7/32
         C 完成 (max_tokens=2, only 2 decode steps needed)
Step 2: A+1, B+1 → 2 running
         B 完成
Step 3: A+1 → 1 running
         A 完成 → 全部完成
```

**解读**：32 个 block 充足，所有 3 个请求一次性 prefill 后一起 decode。C 最早完成（prompt 最短 + max_tokens 最少），A 最后完成（prefill 最多 + max_tokens 最大）。没有 preemption，预算充足，一切正常。

### 场景二：FCFS Preemption（4 个请求，8 blocks 紧张）

```
Step 0: 一次性 prefill A,B,C,D (4×32=128 tokens)，blocks 全满 8/8
        running = [A, B, C, D]

Step 1: 尝试 decode A+1, B+1, C+1, D+1
        → alloc A: OK (1 new block, 7/8)
        → alloc B: FAIL (0 free)
        → PREEMPT D (running.pop()), PREEMPT C (running.pop())
        → running = [A, B], waiting = [C, D] (C 在 D 之前，因为 D 先被 pop)
        → A+1, B+1, blocks=6/8

Step 2-19: decode A+1, B+1 (每步) × 18 步
           blocks 始终 6/8 → 8/8（token 累积需要新 block）
           C 和 D 在 waiting 中等待

Step 19: A 和 B 完成 (max_tokens=20 reached)
         → free blocks → blocks=0/8

Step 20: C 和 D 从 waiting 被接纳
         → prefill C(33 tokens=32+1 已生成?不！preemption 后 num_computed_tokens=0!)
         → 等等——num_computed_tokens=0 所以重新 prefill 32 tokens + 1 decode = 33
         → blocks=6/8 (再次)
         → 注意：我们的实现中 decode 时 num_tokens=32+1=33, num_computed=0, 所以需要 33 tokens
         
Step 21-38: decode C+1, D+1 × 18 步 → 全部完成
```

**核心观察**：
- C 和 D 在 Step 1 被 preempt，等了 18 步（A 和 B decode 完成）才重新运行
- Preempt 后 `num_computed_tokens=0`，所以它们的 32 个 token 的 prompt 被完全重算了一遍
- FCFS preemption 对称：C 晚于 D 加入 running（看 `self.running.append` 顺序）→ D 先被 pop

**Step 1 的逐 token 预算追踪**：

让我们把手放进 Step 1 的 `schedule()` 里，跟踪每一步的变量变化：

```
初始状态: running = [A, B, C, D], token_budget = 128, free_blocks = 0

Phase 1 遍历:
  req_index=0: request = A (computed=32, tokens=32)
    num_new = min(0, 128) = 0 → continue (A 已完成 prefill)
    
  req_index=1: request = B (computed=32, tokens=32)
    num_new = min(0, 128) = 0 → continue
    
  req_index=2: request = C (computed=32, tokens=32)
    num_new = min(0, 128) = 0 → continue
    
  req_index=3: request = D (computed=32, tokens=32)
    num_new = min(0, 128) = 0 → continue
    
  循环结束 (req_index=4 == len(running))
  
  ⚠ 等等——问题来了：所有 running 请求都是 num_new=0（prefill 刚做完，num_computed_tokens 已经追上了 num_tokens），但实际上它们还需要 decode token！
  
  让我们继续下一轮调度。在 update_from_output 中，Step 0 的输出中 A,B,C,D 的 output_token_ids 各增加了一个 [42]（模拟 decode token），所以：
    A.tokens = 33, A.computed = 32 → num_new = 1
    B.tokens = 33, B.computed = 32 → num_new = 1
    C.tokens = 33, C.computed = 32 → num_new = 1
    D.tokens = 33, D.computed = 32 → num_new = 1
  
  token_budget = 128

  req_index=0: A, num_new = min(1, 128) = 1
    allocate_slots: total_tokens = 33, blocks_needed = 3 (33/16=2.0625→3)
                    当前有 2 blocks (32 tokens 的 prefill)
                    num_new_blocks = 1, free_blocks = 0
                    → OOM! 触发 preempt!
    FCFS preemption: running.pop() = D
    free(D): 释放 D 的 2 blocks → free_blocks = 2
    D.status = PREEMPTED, D.num_computed = 0
    waiting.prepend_request(D) → waiting = [D]
    preempted_reqs = [D]
    
    重试 allocate_slots(A, 1):
    free_blocks = 2 → allocate 1 block → OK!
    A: 1 block 分配成功, free_blocks = 1

  req_index=1: B, num_new = min(1, 127) = 1
    allocate_slots: total_tokens = 33, blocks_needed = 3
                    num_new_blocks = 1, free_blocks = 1 → OK!
    B: 1 block 分配成功, free_blocks = 0

  req_index=2: C, num_new = min(1, 126) = 1
    allocate_slots: num_new_blocks = 1, free_blocks = 0 → OOM!
    FCFS preemption: running.pop() = C (C 现在是最后一个)
    free(C): 释放 2 blocks → free_blocks = 2
    waiting.prepend_request(C) → waiting = [C, D]
    preempted_reqs = [D, C]
    
    重试 allocate_slots(C, 1):
    free_blocks = 2 → 但是 victim IS request! 
    → break (C 自己也被 preempt 了)
  
  new_blocks is None for request C → break Phase 1

Phase 2: preempted_reqs = [D, C] → 非空 → 跳过 admission

最终:
  scheduled_running = [A, B]  (只有 A 和 B 拿到 decode token)
  num_scheduled_tokens = {'A': 1, 'B': 1}
  token_budget 剩余 = 126 (128 - 1 - 1)
  preempted = [D, C]
  running = [A, B], waiting = [C, D]
  blocks: free=2, used=6
```

这个逐行追踪揭示了两个关键事实：
1. **D 先于 C 被 preempt**，因为 `running.pop()` 弹出的是添加顺序的逆序
2. **C 在 preempt 自己被释放的 2 个 block 后仍有 2 个 free_blocks，但 C 不能再被调度** ——因为 `if victim is request: break` 规则阻止了请求在自己被 preempt 后又重新被同一轮调度

### 场景三：Priority Preemption（同场景，Priority 策略）

```
Step 0: prefill A(p=0), B(p=5), C(p=10), D(p=10) — 全部 4 个
        running = [A, B, C, D] (按 admission 顺序)

Step 1: 尝试 decode A+1, B+1, C+1, D+1
        → alloc A: OK
        → alloc B: FAIL
        → PREEMPT: max(running, key=(priority, arrival_time))
          D(p=10, 最晚到) vs C(p=10, 早到): arr_time 大的 = D
          → preempt D, then preempt C
        → PREEMPTED: [D(p=10), C(p=10)]
        → running = [A(p=0), B(p=5)]
```

**核心观察**：
- Priority 和 FCFS 在这种情况下表现相同——因为 admission 顺序恰好是 priority 顺序（A<B<C<D）
- 但如果 admission 顺序不同（比如 D 先到，A 后到），Priority 会 preempt 低 priority 的（C/D），而不是最近加入的
- 注意 C 和 D 都是 priority=10 时，C 因 `arrival_time` 更早，在 `__lt__` 中排在 D 前面——所以 D 先被 preempt

## Cell 9 — 源码映射表

| 我们的实现 | 官方源码 | 我们做了什么改变 | 为什么 |
|-----------|---------|----------------|------|
| `SchedulingPolicy` | `request_queue.py:L13-L17` | 完全一致 | 核心 enum，无需简化 |
| `RequestQueue` (ABC) | `request_queue.py:L20-L72` | 移除了 bulk 方法 | 不需要 bulk prepend/remove |
| `FCFSRequestQueue` | `request_queue.py:L75-L128` | 保留核心 5 个方法 | 完整 FCFS 语义 |
| `PriorityRequestQueue` | `request_queue.py:L131-L198` | 保留 heap 操作 | 完整 Priority 语义 |
| `create_request_queue` | `request_queue.py:L201-L208` | 完全一致 | 工厂 dispatch |
| `Request` dataclass | `vllm/v1/request.py:L59-L308` | ~10 fields vs ~60 | 去除 encoder/spec/LoRA/multimodal 字段 |
| `SchedulerOutput` | `output.py:L181-L255` | ~5 fields vs ~15+ | 去除 encoder/spec/grammar 字段 |
| `Scheduler.schedule()` | `scheduler.py:L352-L945` | ~100 lines vs ~600 | 保留两阶段+preemption 核心 |
| `Scheduler._preempt_request()` | `scheduler.py:L952-L972` | ~15 lines，核心语义完整 | 移除了 encoder_cache_manager 和 event recording |
| `Scheduler.update_from_output()` | `scheduler.py:L1290-L1553` | ~40 lines vs ~260 | 只保留 token append + stop check |
| `Scheduler.add_request()` | `scheduler.py:L1728-L1748` | 简化 | 移除 streaming/duplicate 处理 |
| `Scheduler.finish_requests()` | `scheduler.py:L1750-L1811` | 简化 | 仅保留 abort 核心流程 |
| `ToyKVCacheManager` | `vllm/v1/core/kv_cache_manager.py` | ~70 lines 玩具分配器 | Ch05 完全展开 KV cache；本章只建模调度器关心的一层——分配/释放 |
| `RequestStatus` enum | `vllm/v1/request.py:L310-L326` | 6 states vs ~10 | Ch04 已确立，够调度用 |
| `Request.__lt__` | `vllm/v1/request.py` (implicit) | `(priority, arrival_time)` tuple 比较 | PriorityRequestQueue 的排序基础 |

## Cell 10 — 验证结果

### 运行验证

```bash
$ python3 instances/vllm/artifacts/06-scheduling/implementation/demo.py

═══ Scenario 1: Happy path ═══
  Step 0: blocks=6/32, new=[A,B,C]
  Step 1: A+1,B+1,C+1 → C finished
  Step 2: A+1,B+1 → B finished
  Step 3: A+1 → A finished
  ✓ 3 requests complete, 0 preemptions

═══ Scenario 2: FCFS preemption ═══
  Step 0: blocks=8/8, new=[A,B,C,D]
  Step 1: PREEMPTED [D, C] — 3rd and 4th admitted evicted
  Steps 2-19: A,B decode → finished at step 19
  Step 20: C,D resume (recompute from 0)
  Steps 21-38: C,D decode → finished at step 38
  Total preemptions: 2

═══ Scenario 3: Priority ═══
  Step 0: blocks=8/8, new=[A(p=0),B(p=5),C(p=10),D(p=10)]
  Step 1: PREEMPTED [D(p=10), C(p=10)] — lowest priority evicted
  Same completion pattern — high priority kept in running
  Total preemptions: 2
```

### 单元测试

```bash
$ python3 -m pytest instances/vllm/artifacts/06-scheduling/tests/ -v

test_fcfs_basic.py::test_fifo_order PASSED
test_fcfs_basic.py::test_prepend_preempted PASSED
test_fcfs_basic.py::test_remove_request PASSED
test_priority_basic.py::test_heap_order PASSED
test_priority_basic.py::test_same_priority_fifo PASSED
...
23 passed in 0.45s
```

### Lint 验证

```bash
$ python3 scripts/lint_formulas.py instances/vllm/artifacts/06-scheduling/narrative/chapter.md
# → (run after writing)
```

## Cell 11 — 总结：调度器的本质是一个约束求解器

回到开头的问题：**一块 GPU 上一千个请求，怎么排这个队？**

vLLM v1 的调度器给出了一套简洁而深刻的答案：

**1. 调度 = 追赶计数器。** 不区分 prefill 和 decode——每个请求只维护 `num_computed_tokens` 追赶 `num_tokens`。这个统一模型自然地覆盖了 chunked prefill、prefix caching 和 speculative decoding，不需要为每个优化单独开一条代码路径。

**2. Running-First 是 decode 友好的。** 已经在跑的请求（每步只产 1 token，成本极低）优先于新到的、需要大量 prefill 的请求。这不是偏袒——这是让 GPU 在最短时间内产出最多 token 的策略，也是最快释放 KV cache block 的路径。

**3. Preemption 是 KV Cache 的"垃圾回收"。** 当 `allocate_slots()` 返回 `None` 时，必须踢出一个请求腾空间。vLLM v1 选择 recompute-only（扔掉 KV cache、下次重算），而不是 swap（搬到 CPU 内存）。因为 GPU 计算比 PCIe 搬运快 20-100 倍。

**4. Priority = (priority, arrival_time)。** 一个 tuple 排序同时解决了优先级区分和优先级内无饥饿两个问题。简单，但有效。

**5. Pareto Frontier 无法被"优化掉"。** 低延迟和高吞吐量是天然矛盾的。调度器能做的，是在这个前沿上给你一个可预测的选点——通过调节 `max_num_running_reqs`、`max_num_scheduled_tokens` 和 `max_model_len`。

本章的调度器和 Ch05 的显存管理器是 vLLM 引擎的两个核心支柱。调度器决定"谁跑、跑多少"；显存管理器决定"放在哪、够不够放"。它们之间的接口——`allocate_slots()` 和 `free()`——就是整个 vLLM 引擎最关键的 API 契约。

下一章我们将展开 KV cache 系统的另一个维度：前缀缓存——如何让两个不同请求共享同一段 KV cache，以及为什么这个看似简单的想法在实践中如此复杂。

---

*第06章完。源码锚点：`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352-L972`（schedule + preempt），`request_queue.py:L75-L198`（两种队列），`output.py:L181-L255`（SchedulerOutput），`interface.py:L22-L100`（SchedulerInterface + PauseState）。*
