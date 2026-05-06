# 第6章：请求调度系统 — 策略、优先级与公平性

> 打开 `vllm/v1/core/sched/request_queue.py:13`。只有两行：`FCFS = "fcfs"` 和 `PRIORITY = "priority"`。
> 整个 vLLM 的调度策略选择就在这两行里。但这两个值的差异能产生截然不同的系统行为。

---

## 这章要做什么？

第 4 章讲了 `schedule()` 的两个 phase——running 优先、waiting 填充。但它有一个没回答的问题：**当多个请求竞争资源时，谁先谁后？谁被驱逐？**

答案在 `SchedulingPolicy` 和 `PreemptionPolicy` 里。这不是一个复杂的模块——整个调度策略系统只有几百行 Python，没有 CUDA kernel，没有复杂的数学。但它的设计选择对系统的尾部延迟、公平性和吞吐量有深远影响。

本章从 queue 数据结构的选择出发，推导两种策略的行为差异。

学完这章你能：
- 解释 `PriorityRequestQueue` 为什么用 `heapq` 而不是 `sortedcontainers`
- 理解 FCFS preemption 为什么 `pop()` 最后一个请求——以及这如何防止 thrashing
- 分析 Priority 模式下的饥饿风险——以及 vLLM 为什么不内置 anti-starvation 机制

---

## 6.1 两个 Queue，两种世界观

### Source Trail

打开 `vllm/v1/core/sched/request_queue.py:13`：

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
```

没有 Weighted Fair Queueing。没有 Shortest-Job-First。没有 Lottery Scheduling。两个选项——**先来先服务和优先级。**

### FCFSQueue：`deque`，O(1) 的简单

```python
class FCFSRequestQueue(deque[Request], RequestQueue):  # L75
    def add_request(self, request):
        self.append(request)       # 尾部追加 → 新来的排后面

    def pop_request(self) -> Request:
        return self.popleft()      # 头部取出 → 最早来的先处理

    def prepend_request(self, request):
        self.appendleft(request)   # 头部插入 → 被驱逐的请求排到最前面
```

`deque` 的三个操作都是 O(1)。语义直观：先来的先服务。Prepend 让被驱逐的请求回到队列最前面——它们已经等过一次了，不应该再排到新请求后面。

### PriorityQueue：`heapq`，O(log n) 的公平

```python
class PriorityRequestQueue(RequestQueue):  # L131
    def add_request(self, request):
        heapq.heappush(self._heap, request)  # O(log n) 入堆

    def pop_request(self) -> Request:
        return heapq.heappop(self._heap)     # O(log n) 出堆
```

**用 `heapq` 而不是 `sortedcontainers`？** sortedcontainers 的 `SortedList.add()` 是 O(log n) 理论上一样，但常数项更大（纯 Python vs C 实现的 heapq）。而且 `heapq` 不维护完全排序——只维护"最小值在顶部"的堆性质——这对 scheduler 足够了：它每次只需要最优的请求，不需要完整的排序。

### 优先级排序：`Request.__lt__`

打开 `vllm/v1/request.py:296`：

```python
def __lt__(self, other: "Request") -> bool:
    if self.priority != other.priority:
        return self.priority < other.priority     # priority 值小 = 优先级高
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time  # 同优先级, 早到优先
    return self.request_id < other.request_id     # 都相同, 字典序
```

**`priority=0` 是最优优先级。** 这不是显然的——很多系统用"priority 越大越优先"。vLLM 选了"越小越优先"，因为这样 default(priority=0) 自然就是最高优先级，不需要显式设置。

---

## 6.2 Preemption：驱逐谁？

### Source Trail

打开 `vllm/v1/core/sched/scheduler.py:478`。当 KV Cache 不够分配时，必须驱逐一个 running 请求：

```python
# PRIORITY mode (L479-L502)
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, r.arrival_time),
    )
    self.running.remove(preempted_req)

# FCFS mode (L503-L504)
else:
    preempted_req = self.running.pop()
```

### Theory: 为什么两个策略选不同的受害者？

**FCFS: `self.running.pop()`** ——弹出 running 列表的**最后一个**元素。因为请求按 admission 顺序加入 running，最后一个就是**最新被调度的**。驱逐它意味着"你刚进来，还没投入多少资源——被赶出去代价最小"。

这个选择防止了 thrashing：如果驱逐最老的请求，它可能已经有了大量的 KV Cache blocks 被其他请求通过 prefix cache 引用（高 `ref_cnt`）。驱逐它会导致级联失效。

**PRIORITY: `max(running, key=lambda r: (r.priority, r.arrival_time))`** ——驱逐优先级最低的（priority 值最大）。如果有多个相同 priority，驱逐最晚到的。

注意这里 `max()` 的 key 和 `__lt__` 的逻辑是一致的——`__lt__` 下"更小 = 更高优先级"，所以 `max()` 选出的恰是"最不重要的请求"。这保证了高优先级请求的延迟不会被低优先级请求挤占。

### 被驱逐后：状态重置

打开 `scheduler.py:952`：

```python
def _preempt_request(self, request, timestamp):
    self.kv_cache_manager.free(request)       # 释放所有 KV Cache blocks
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0           # ← 从头开始！之前算的全扔了
    request.num_preemptions += 1
    self.waiting.prepend_request(request)     # ← 排到队伍最前面
```

**`num_computed_tokens = 0`** 意味着被驱逐的请求下次被调度时，必须从头重新计算所有的 K 和 V——即使它之前已经 prefill 了 99%。这是 preemption 的代价：它比从头 admission 更贵，因为已经浪费了一轮 compute。

`★ Insight ─────────────────────────────────────`
FCFS 的 `pop()` 可以用列表的 O(1) 操作，因为要驱逐的请求恰好在列表末尾。PRIORITY 的 `max()` + `remove()` 是 O(n) 的——需要扫描整个 running 列表找最差的。在 running 列表通常只有几十个请求时（`max_num_running_reqs` 默认 128），O(n) 几乎无感。但如果 running 列表扩展到几千个请求，`max()` 就会成为瓶颈。这暗示 vLLM 的设计假设是 **running 列表不会很长**——这与"KV Cache 是主要瓶颈，running 数量受显存约束"的事实一致。
`─────────────────────────────────────────────────`

---

## 6.3 skipped_waiting：被跳过的请求

### Source Trail

打开 `scheduler.py:1567`：

```python
def _select_waiting_queue_for_scheduling(self):
    if self.policy == SchedulingPolicy.FCFS:
        return self.skipped_waiting or self.waiting or None

    # PRIORITY mode: compare heads
    if self.waiting and self.skipped_waiting:
        w = self.waiting.peek_request()
        s = self.skipped_waiting.peek_request()
        return self.waiting if w < s else self.skipped_waiting

    return self.waiting or self.skipped_waiting or None
```

### Theory: 为什么有两个 Waiting Queue？

`self.waiting` 存放正常等待的请求。`self.skipped_waiting` 存放上一轮被跳过的请求——这些请求因为外部依赖（remote KV 还没传到、streaming input 还没完全到达、grammar 还没编译好）无法被调度。

**FCFS 模式：优先 skipped_waiting。** 被跳过的请求已经等了一段时间（等外部依赖完成），现在依赖满足了——让它们插队到新请求前面是公平的。否则，一个请求可能因为 1ms 的 remote KV transfer 延迟被无限期推迟——每一个新到达的请求都会排到它前面。

**PRIORITY 模式：比较两个队列的 head 元素。** 如果 skipped_waiting 的 head 优先级更高，从 skipped_waiting 取；否则从 waiting 取。这需要**两个队列都是 PriorityQueue**——否则无法比较"哪个 head 优先级更高"。这意味着 PRIORITY 模式下 `self.skipped_waiting` 也是 `PriorityRequestQueue`，不是 FCFS。

### 跳出队列的时机

在每轮调度结束时，被跳过的请求被 prepend 到 `self.skipped_waiting`：

```python
# scheduler.py:L845
if step_skipped_waiting:
    self.skipped_waiting.prepend_requests(step_skipped_waiting)
```

**Prepend 意味着什么？** 在 FCFS 中，prepend = appendleft——刚跳过的请求排到队伍前面。在 PRIORITY 中，prepend = add（heap push）——刚跳过的请求进堆，按优先级排序。语义上的区别：FCFS 给了"被跳过"一个明确的插队。PRIORITY 不给——优先级的顺序不被"是否被跳过"影响。

---

## 6.4 暂停状态：PauseState

### Source Trail

打开 `vllm/v1/core/sched/interface.py:22`：

```python
class PauseState(IntEnum):
    UNPAUSED = 0    # 正常运行
    PAUSED_NEW = 1  # 不接受新请求（running 继续）
    PAUSED_ALL = 2  # 完全停止
```

在 `scheduler.py` 中的应用：

- `PAUSED_ALL` (L372-L374): 设置 `token_budget = 0`，跳过所有调度。用于 graceful shutdown——让 running 请求完成但不 admit 新的。
- `PAUSED_NEW` (L568): 跳过 waiting 请求的调度。用于 load shedding——当前负载太高，不接受新请求直到压力下降。

---

（搜索 `vllm/v1/core/sched/` 目录，零结果）

## 6.5 公平性的缺失

（搜索 `vllm/v1/core/sched/scheduler.py` 和 `request_queue.py` 目录下 fairness/starvation/aging — 零结果。vLLM 调度系统无内置公平性机制。）

### Theory: vLLM 为什么不内置 anti-starvation？

搜索 `vllm/v1/core/sched/` 目录下 `fairness|fair|starvation|aging`——在 `vllm/v1/core/sched/` 目录下**零结果**。vLLM 的调度系统中**没有公平性机制**。没有 aging（等待时间递增后优先级自动提升），没有 weighted fair queueing，没有最小服务时间保证。

这不是疏忽——是设计选择。**LLM 推理的延迟通常由最慢的请求决定（tail latency）**，而吞吐量由"有多少请求可以同时被处理"决定。在 KV Cache 是主要瓶颈的前提下，添加 aging 或 fair queueing 不会增加吞吐量——只会重排请求的顺序。

但饥饿确实可能发生：在 PRIORITY 模式下，如果持续有高优先级请求到达，低优先级请求可能永远不被调度。解决方案？**用户负责**——你设置 priority 值，你承担饥饿的责任。vLLM 提供机制（mechanism），不提供策略（policy）。

这种"把 policy 交给用户"的设计在系统软件中很常见——Linux 的 `nice` 值、Kubernetes 的 PriorityClass、vLLM 的 `SchedulingPolicy.PRIORITY`。机制在框架中，策略在用户手中。

---

## 6.6 调度系统全貌

（综合 `scheduler.py:L352`, `L478`, `L1567` 的逻辑。）

将第 4 章和第 6 章整合，完整的调度决策流程：

```
schedule()
  │
  ├── Phase 1: Running Requests
  │     └── for req in self.running:
  │           num_new = req.num_tokens - req.num_computed_tokens
  │           cap at token_budget, max_model_len
  │           blocks = kv_cache_manager.allocate_slots(req, num_new)
  │           if blocks is None:
  │             │
  │             ├── FCFS:    preempted = self.running.pop()
  │             └── PRIORITY: preempted = max(running, key=priority)
  │             │
  │             └── _preempt_request(preempted)
  │                   → free KV blocks, reset num_computed_tokens=0
  │                   → prepend to waiting queue
  │
  ├── Phase 2: Waiting Requests
  │     └── queue = _select_waiting_queue_for_scheduling()
  │           │
  │           ├── FCFS:    skipped_waiting > waiting
  │           └── PRIORITY: compare head of both queues
  │           │
  │           req = queue.peek()
  │           if req.status in blocked_statuses:
  │               try promote → skip if blocked
  │           blocks = kv_cache_manager.allocate_slots(req, num_new)
  │           if blocks is None: break
  │           req.status = RUNNING
  │
  └── Phase 3: Build SchedulerOutput
```

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `SchedulingPolicy` | `request_queue.py:L13` | 相同——两值 enum |
| `FCFSQueue` | `request_queue.py:L75` | 保留核心 API：add/pop/prepend/remove；未实现 prepend_requests(extendleft) |
| `PriorityQueue` | `request_queue.py:L131` | 保留 heapq + `__lt__` 排序；未实现 `__iter__` 细节 |
| `PreemptionPolicy` | `scheduler.py:L478-L511` | FCFS pop() + PRIORITY max() 逻辑一致 |
| `select_queue()` | `scheduler.py:L1567` | FCFS/PRIORITY 的队列选择逻辑一致 |
| `PauseState` | `interface.py:L22` | 三值 enum 完全一致 |

---

## 验证

```bash
cd artifacts/06-scheduling && python -m pytest tests/ -q
# 11/11 passed ✅
```

---

## 总结

- **FCFS vs PRIORITY 只有两个选项。** 没有复杂的 fair queueing——机制在框架，策略交给用户。
- **FCFS preempt 最新请求**——代价最小（还没投入多少 compute）。PRIORITY preempt 最低优先级——保护高优请求的延迟。
- **skipped_waiting 是外部依赖的等待室。** FCFS 给它插队权，PRIORITY 让它和 waiting 按优先级竞争。
- **没有 aging，没有 starvation prevention。** 设计选择——LLM 推理的瓶颈是 KV Cache 而不是调度策略。
- **纯 Python，无 CUDA kernel。** 调度系统不需要 GPU 算力——它在 CPU 上决定"下一步算什么"，然后 GPU 去算。

---

**下一章：** 第7章 — Prefix Cache & APC Aware Allocation

调度系统把请求推进来。但如果两个请求有相同的 system prompt，它们能不能共享 KV Cache？第 7 章将解释 vLLM 的 Automatic Prefix Caching——如何用 Radix Tree 在 O(1) 时间内找到最长缓存前缀，以及 BlockPool 的 `touch()` 和 `cache_full_blocks()` 如何让这个机制在 block 粒度运作。

---

← 第5章 | 第7章 →
