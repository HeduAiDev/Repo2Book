# 第6章：请求调度策略 — 当 schedule() 必须做选择题

> 本章涉及的 vLLM 源码：
> - `instances/vllm/source/vllm/v1/core/sched/request_queue.py:L1-L208`（RequestQueue ABC + FCFS + PRIORITY + factory）
> - `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L478-L504`（preempt 受害者选择）+ `L1567-L1577`（waiting/skipped 选择）+ `L372-L374`（pause-state gating）
> - `instances/vllm/source/vllm/v1/core/sched/interface.py`（PauseState）
> - `instances/vllm/source/vllm/v1/request.py:L296-L307`（`__lt__` 四级排序）
> - `instances/vllm/source/vllm/config/scheduler.py:L22, L63, L109-L117`（policy 字段、默认值、文档）
> - `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972`（recompute）+ `kv_offload/cpu/gpu_worker.py:L319`（swap，注意：不是 preempt 路径）+ `scheduler.py:L1750-L1811`（abort）
>
> 本章源码 commit：`98661fe`。
>
> **第 4 章把 `schedule()` 怎么跑讲完了——两个 phase、preempt 队尾、chunked prefill 的 continue/break。但那个循环里有几个看起来是"if/else"的语句——哪个 running 请求被 preempt？哪个 waiting 队列先被服务？budget 怎么受 pause-state 影响？这些不是机械的算法，是策略选择题。本章把它们抠出来，逐个回答。**

---

## 这章要讲什么？

策略和机械之分：

- **机械层**（第 4 章）：`schedule()` 的两个 phase、`_preempt_request` 释放 KV、Phase 2 跳过条件——所有"必然要发生"的事。
- **策略层**（本章）：在每个机械分支里，**选谁**。受害者选择、waiting 队列选择、pause-state 解释、preemption 策略对比、并发吞吐 vs 尾延迟的 Pareto 取舍。

打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L478-L504` 和 `L1567-L1577`——这两段就是本章的"心脏"。Ch04 走读 `schedule()` 时跳过了它们；本章把它们从机械循环里**提取成纯函数**，独立分析。

学完这章你能：

- 写出 FCFS 和 PRIORITY 两个 policy 在三件事上的差异：（a）preempt 受害者怎么挑、（b）waiting vs skipped_waiting 哪个先用、（c）`prepend_request` 在两种 queue 下的语义。
- 用 demo 的 8.00x head-of-line factor 说清楚 FCFS 为什么会让短请求等到长请求 prefill 完——并知道为什么这是结构性的，不是 bug。
- 在白板上对比 8K 请求的 recompute (164 ms) / swap (62 ms) / abort (5000 ms)；解释为什么 vLLM v1 选了延迟最差但**简单**的那个；并解释 P02 的"crossover length-independent"——L 不影响选择。
- 用 K18 不变量（"PRIORITY queue 没有 front 概念"）一句话回答"为什么 priority 模式下被 preempt 的请求可能永远 stuck"。
- 看懂 demo 第 [5] 段的 sweep——"16x p95 TTFT 改善"是**两组配置之间**的比值，不是某个开关单方面带来的；并解释 `long_prefill_token_threshold` 怎么让小引擎匹配大引擎的尾延迟。

---

## 6.1 Ch04 的循环里那些 if 语句到底在选什么？

第 4 章 4.3 节走读 `schedule()` 时，我们写过这一段（简化版）：

```python
# 来自 Ch04 §4.3.2 的 Phase 1 摘录
while new_blocks is None:
    preempted_req = self.running.pop()      # ← 这里
    ...
```

`self.running.pop()` 这个看起来人畜无害的一行，藏着一个策略决定：**FCFS 下 pop 队尾就是"踢最晚到达的人"，但 PRIORITY 模式下应该踢谁？** 不是队尾——而是**优先级最差**的那个，无论它是不是最后到的。

打开真实 vLLM 源码 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L478-L504` —— 完整的 preempt-victim 选择：

```python
# vllm/v1/core/sched/scheduler.py:L478-L484 (PRIORITY branch)
if self.scheduler_config.policy == "priority":
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, r.arrival_time),
    )
    ...
# vllm/v1/core/sched/scheduler.py:L504 (FCFS branch)
else:
    preempted_req = self.running.pop()
```

看似两行 if/else，背后是两套完全不同的"公平观":
- FCFS 的公平 = **先来的先享受**。后来的应该先牺牲——所以踢队尾。
- PRIORITY 的公平 = **优先级低的先牺牲**——和到达顺序无关。

同样的故事还出现在另外两处：
1. **`waiting` vs `skipped_waiting` 的选择**（`scheduler.py:L1567-L1577`）：FCFS 永远先看 skipped；PRIORITY peek 两个 head 比 `__lt__`。
2. **`prepend_request` 的语义**（`request_queue.py:L160-L165`）：FCFS 是 `appendleft`（真的把请求塞到队首）；PRIORITY 是 `add_request`（heap push 回它本来的优先级位置）。

把这三处合起来看，"policy" 不是一个枚举字段——是**贯穿调度全流程的一组等价的取舍**。Ch06 的 5 节就是按这条主线展开。

接下来 5 节的安排：
- **6.2** FCFS 的代价：head-of-line blocking 用 demo 的 8.00x 量化。
- **6.3** Preempt 的策略：recompute / swap / abort，三条路，为什么 v1 选了延迟最差的那条。
- **6.4** PRIORITY queue：K18 不变量、`max(priority, arrival_time)` 受害者选择、`aged_priority` 为什么不是真特性。
- **6.5** Pareto 前沿：`max_num_seqs`、`max_num_batched_tokens`、`long_prefill_token_threshold` 三个旋钮的取舍，以及 16x p95 TTFT 的 sweep-pair 解释。

每节都有 5-step rhythm：源码定位 → 桥接 → 理论 → 实现 → 差异分析。

---

## 6.2 FCFS 的代价：head-of-line blocking

### 6.2.1 一个长请求能让多少短请求陪跑？

打开 `instances/vllm/artifacts/06-scheduling/implementation/starvation_analysis.py:L41-L96` —— `WorkloadProfile` 用一个 2-class workload 量化 FCFS 的尾延迟问题。

跑 demo（`python3 -m instances.vllm.artifacts.06-scheduling.implementation.demo`）的第 [2] 段：

```
[2] FCFS head-of-line blocking — long prompt starves short
    Workload: 2 long(4K) + 8 short(64)
    Token budget B = 512
    Long-request prefill steps:    8
    Short-request worst-case wait: 8 steps
    Head-of-line blocking factor:  8.00x
    FCFS starves under this profile?    True
    PRIORITY (well-assigned) starves?   False
```

读这个数字：**短请求的等待时间是它自己 prefill 时间的 8 倍**——它的 prompt 只要 1 步（64 token / 512 budget），但要排队等长请求那 8 步 prefill 跑完才能挤进 running。

公式（`starvation_analysis.py:L60-L96`）：

$$
\mathrm{factor} = \frac{\lceil P_{\mathrm{long}} / B \rceil}{\max(1, \lceil P_{\mathrm{short}} / B \rceil)}
$$

代入 demo：分子 $\lceil 4096 / 512 \rceil = 8$，分母 $\max(1, 1) = 1$（因为 $\lceil 64/512 \rceil = 1$），factor = 8.00。

**factor > 1 就是结构性的尾延迟**——短请求的 P95 时间被长请求拉长，越长越糟。`has_starvation(profile, FCFS)` 返回 True 这个判断（`starvation_analysis.py:L138-L153`）就是这个不等式。

### 6.2.2 为什么 vLLM 默认 FCFS？

明知道 FCFS 有 head-of-line 问题，vLLM 默认还是 `policy = "fcfs"`（`config/scheduler.py:L109-L117`）。原因有三个：

1. **PRIORITY 也有 starvation**——只是发生在不同情形。如果所有请求 priority 都相等，PRIORITY 退化成 FCFS（`__lt__` 第二级 tier 是 `arrival_time`，见 `vllm/v1/request.py:L301-L302`）。如果某个低优先级请求被一连串高优先级新到达者持续盖过，它**永远拿不到 token**。FCFS 至少保证"等够久就能跑"。
2. **Priority 是 user signal，不是自动机制。** vLLM 不会自动把等久了的请求 priority 往上调（D1 in impl-notes）。priority 字段是**调用者写的**（API 请求时传），它意味着 SLA 合同——付费用户 priority=0、免费用户 priority=10——任何"自动 aging"都会偷偷违反这个合同。
3. **多数 workload 不需要 PRIORITY**。同一推理服务里大部分请求形状接近（同一个模型、同一个用户群）。FCFS 在这种"同质"workload 上几乎无 starvation——head-of-line factor 接近 1。

### 6.2.3 PRIORITY policy 的入场条件

那什么情况该开 PRIORITY？三个典型场景：

- **多租户付费层**：付费 priority=0，免费 priority=10。绝不允许免费请求 preempt 付费请求。
- **延迟敏感与吞吐敏感混部**：交互聊天（priority=1）与离线批量（priority=5）混着跑——交互的尾延迟要稳。
- **请求形状显著不齐**：比如长 prompt 偶尔到达但占比 < 5%——给长 prompt 高 priority value（低优先级），让短的优先跑。

注意第三种场景里"给长 prompt 低优先级"是**人工预先分流**——vLLM 不会自动检测请求长度然后调 priority。这是 `config/scheduler.py:L113-L117` 反复强调的点：priority 是 caller 责任，不是 vLLM 责任。

### 6.2.4 我们的实现 vs vLLM

`starvation_analysis.py` 是个**分析模型**——`WorkloadProfile` 在 vLLM 源码里没有对应类。`fcfs_short_request_latency_steps`、`fcfs_long_request_completion_steps`、`head_of_line_blocking_factor` 都是从 `scheduler.py:L387-L556`（Phase 1 RUNNING 循环）+ `L568-L846`（Phase 2 WAITING 循环）的语义反推出来的封闭形式。

`aged_priority`（`starvation_analysis.py:L114-L135`）干脆是个**反例**——它故意展示一个 vLLM **不**会做的事，配合 docstring 解释为什么不做（D1 三条理由：a）priority 是 SLA、b）aging rate 难调、c）破坏 heap 不变量）。这是教学手法：先展示一个"看起来合理"的反方案，再解释 vLLM 为什么拒绝它。

---

## 6.3 Preempt 的三条路：recompute / swap / abort

### 6.3.1 把第 5 章的 trade-off 表展开成三选一

第 5 章 5.6 节我们对比过 recompute vs swap。本章把 abort 加进来，凑成三选一。打开 `implementation/preemption_strategy.py:L50-L111` —— `PreemptionStrategy` 枚举和 `PreemptionScenario`：

```python
# implementation/preemption_strategy.py:L50-L53
class PreemptionStrategy(Enum):
    RECOMPUTE = "recompute"
    SWAP = "swap"
    ABORT = "abort"
```

跑 demo 的第 [3] 段：

```
[3] Preemption: recompute (v1) vs swap (v0) vs abort
    scenario                 recompute      swap     abort      winner    kv (GiB)
    short prompt (512)          10.2 ms     3.9 ms  5000.0 ms        swap       0.062
    medium prompt (2K)          41.0 ms    15.6 ms  5000.0 ms        swap       0.250
    long prompt (8K)           163.8 ms    62.5 ms  5000.0 ms        swap       1.000
    xlong prompt (32K)         655.4 ms   250.0 ms  5000.0 ms        swap       4.000
    Crossover: swap always faster regardless of prompt length
    vLLM v1's choice: recompute for ALL prompts.
```

四行四个 prompt 长度。**每一行 swap 都比 recompute 快 2-3 倍**。winner 列全部是 `swap`。

但 vLLM v1 选了 **recompute**——延迟最差的那个。为什么？

### 6.3.2 不要从 latency 入手，要从复杂度入手

注意一个语言陷阱：**不要说"recompute 更快"**——它不是。正确的说法是：

> **Swap 在 latency 上更快，但代码复杂度、CPU 内存依赖、错误恢复都更复杂。Recompute 在 latency 上更慢，但是单代码路径、零 CPU 依赖、bit-deterministic。vLLM v1 用 100 ms 的延迟惩罚换来四类复杂度的消除。**

打开 `preemption_strategy.py:L21-L37` 的 trade-off 矩阵注释：

```
                    | RECOMPUTE | SWAP      | ABORT
    ----------------+-----------+-----------+-------------
    GPU↔CPU bytes   | 0         | 2 × KV    | 0
    Compute redo    | full      | 0         | n/a
    Latency cost    | 164 ms    | 62 ms RT  | 0 ms (lost)
    User-visible    | retry-ok  | retry-ok  | error / drop
    Code complexity | 1 path    | 2 paths   | 1 path
    OOM safety      | always ok | needs CPU | always ok
    Determinism     | bit-equal | bit-equal | n/a
```

四类复杂度：
1. **代码路径**：swap 需要两条独立流程（preempt 时拷出、resume 时拷回），加 cudaMemcpy 完成同步。Recompute 复用 prefill 路径——零新代码。
2. **CPU 内存依赖**：swap 要在 CPU 端跟一个独立的 swapped-KV pool。CPU 满了就完全失败——多租户节点上 32 GB CPU 配 80 GB GPU 是常见配置，CPU 是瓶颈。Recompute 不需要任何 CPU 内存。
3. **OOM 错误恢复**：swap 失败时怎么 fallback？还要再写一条降级路径。Recompute 永远可用，没有"swap 失败"这一种错误模式。
4. **Bit-determinism**：理论上 swap+resume 也是 bit-equal（cudaMemcpy 不损失精度）。但**recompute 重做 prefill** 也是 bit-equal——只要 kernel 是确定的。两者都满足，但 recompute 还多一个免费的好处：调试时可以"重放任意中间步"，不需要 swap-state restore。

`scheduler.py:L952-L972` 的 `_preempt_request` 因此可以是 5 行：free blocks、reset `num_computed_tokens`、`num_preemptions += 1`、prepend 到 waiting——干净利落。如果加 swap，估计要 2K LoC 处理边界情况。

### 6.3.3 P02：crossover prompt length 是常数，不是函数

一个反直觉的事实——`preemption_strategy.py:L124-L153` 的 `crossover_prompt_length` 函数告诉你：**recompute 和 swap 的 latency 交叉点不依赖于 prompt 长度**。

设 recompute 和 swap 的 latency 相等：

$$
\frac{L}{\mathrm{TP}} = \frac{2 \cdot 2 \cdot N_L \cdot N_H \cdot D \cdot dt \cdot L}{\mathrm{BW}}
$$

两边除以 $L$：

$$
\frac{1}{\mathrm{TP}} = \frac{4 \cdot N_L \cdot N_H \cdot D \cdot dt}{\mathrm{BW}}
$$

L 完全消掉。所以 crossover 是 (TP, BW, model shape) 的函数，与 prompt 长度无关。代入 32 layer / 8 KV head / 128 head_size / fp16：

$$
\mathrm{bytes\_per\_token} = 4 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 262{,}144 \approx 256 \mathrm{KiB}
$$

$$
\mathrm{threshold\_TP} = \frac{32 \mathrm{GiB/s}}{256 \mathrm{KiB}} \approx 131{,}072 \mathrm{tok/s}
$$

H100 的实测 prefill throughput 是 **50K tok/s** 左右——远低于 131K threshold。所以**对所有 prompt 长度，swap 都比 recompute 快**——这就是 demo 输出 `Crossover: swap always faster regardless of prompt length` 的来源。

这就是 **P02 不变量** 的全部内容：crossover 决策与 L 无关。Tester 用一个测试把这点钉住——`test_length_independent`：assert L=128 和 L=131072 时 `crossover_prompt_length` 返回相同的 dispatch。这种"不变量测试"比"在默认配置下 swap 比 recompute 快"更结实——后者一旦有人重构引入 L-dependence 也不会失败。

### 6.3.4 Abort 是什么角色？

Abort 在表里 latency 是 0 ms（对引擎）但用户看到 5000 ms penalty——那是 SLA cost。abort 在 vLLM 里**不是 preempt 的备选**——它是另一条独立路径：

- API client disconnect → `finish_requests(req_id, FINISHED_ABORTED)` —— `scheduler.py:L1750-L1811`。
- Admin command 强制取消。

不会因为 KV-cache OOM 而 abort。这是和"preempt 三选一"的语言陷阱：**preempt 实际只有一条路（recompute）**；swap 和 abort 在源码里存在，但都不是 preempt 触发的。

`expected_latency_under_oom_rate`（`preemption_strategy.py:L156-L171`）把这层语言区分量化了。demo 在 5% OOM/step、100 step/req 时给出：

| 策略 | E[latency] |
|------|-----------:|
| recompute | 983 ms |
| swap | 476 ms |
| abort | 25,164 ms |

abort 的 25 秒是 100 步×5% 概率×5 秒 SLA penalty + base 164 ms ≈ 25,164 ms。**abort 比 recompute 慢 25 倍**——这就是它不能用于 preempt 的根本原因。

### 6.3.5 我们的实现 vs vLLM

| 项 | 我们 | vLLM |
|----|------|------|
| `PreemptionStrategy` enum | 三个变体合成 | 散落在三个文件 |
| `PreemptionScenario.recompute_seconds` | analytical model | 实际 prefill 测量 |
| `PreemptionScenario.swap_seconds` | analytical | `kv_offload/cpu/gpu_worker.py:L319` 实现，但**只用于 prefix cache offload** |
| `PreemptionScenario.abort_seconds` | analytical SLA penalty | `scheduler.py:L1750-L1811` `finish_requests` 实际终止 |
| `crossover_prompt_length` | 解析模型 | — |
| `expected_latency_under_oom_rate` | 解析模型 | — |

`kv_offload/` 子系统**做** GPU↔CPU KV 传输（`swap_blocks_batch`），但它服务的是 **prefix cache 跨步保留**，不是 preempt 时的临时换出。读源码的人很容易把这两个混淆——本章 6.3.4 的"abort 不是 preempt 的备选"也是同样的语言陷阱。

---

## 6.4 PRIORITY queue：K18 不变量先讲清

### 6.4.1 为什么从 K18 入手？

打开 `implementation/request_queue.py:L160-L165` —— `PriorityRequestQueue.prepend_request`：

```python
# implementation/request_queue.py:L160-L165
# REFERENCE: vllm/v1/core/sched/request_queue.py:L160-L165
def prepend_request(self, request: PolicyRequest) -> None:
    # Documented no-op equivalent to add_request. THE invariant.
    self.add_request(request)
```

这一行是本章最重要的不变量。它说的是：**PRIORITY queue 没有"前面"和"后面"的概念。** 一个请求被 push 进去，自动落到 `(priority, arrival_time)` 排好的位置上——你说不出"队首"是什么意思，因为它只是堆顶 invariant 维护出的一个最值。

vLLM 源码 `vllm/v1/core/sched/request_queue.py:L160-L165` 的注释一字不差地说了这点："In a priority queue, there is no concept of prepending to the front. Requests are ordered by (priority, arrival_time)."

为什么这是核心？因为它**决定了 preempt 的语义不对称**：

- **FCFS 下被 preempt 的请求**：`prepend_request` = `appendleft`——真的塞到队首。下一步就会被重新 admit，它自己造成的 OOM 已经因为别人被踢而解决了。preempt 是"暂停一下，下个 step 优先恢复"。
- **PRIORITY 下被 preempt 的请求**：`prepend_request` = `add_request`——heap push 回它本来的优先级槽位。如果它本来就是优先级最低的（4.6.2 会看到 priority queue 模式下 preempt 的恰恰就是优先级最低的），它**就在那躺着，永远轮不上**。preempt 是"你优先级低，去后面排"。

也就是说：**PRIORITY 模式下，被 preempt 的请求可能永远 stuck**——这不是 bug，是设计。如果你不希望这种事发生，那就不要给请求设很差的 priority；如果你**确实**希望低优先级请求被高优先级请求挤死，那 PRIORITY 模式就在帮你执行这个意图。

### 6.4.2 PriorityRequestQueue 的实现

打开 `implementation/request_queue.py:L147-L198` —— 整个 `PriorityRequestQueue`：

```python
# implementation/request_queue.py:L156-L184
class PriorityRequestQueue(RequestQueue):
    def __init__(self) -> None:
        self._heap: list[PolicyRequest] = []

    # REFERENCE: vllm/v1/core/sched/request_queue.py:L144-L146
    def add_request(self, request: PolicyRequest) -> None:
        heapq.heappush(self._heap, request)

    # REFERENCE: vllm/v1/core/sched/request_queue.py:L148-L152
    def pop_request(self) -> PolicyRequest:
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    # REFERENCE: vllm/v1/core/sched/request_queue.py:L154-L158
    def peek_request(self) -> PolicyRequest:
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    # REFERENCE: vllm/v1/core/sched/request_queue.py:L160-L165
    def prepend_request(self, request: PolicyRequest) -> None:
        self.add_request(request)  # K18: identical
```

四个核心方法，每个都是 `heapq` 一行——但**排序键来自 `PolicyRequest.__lt__`**。

`PolicyRequest.__lt__`（`request_queue.py:L63-L71`）严格复刻 vLLM 的 `Request.__lt__`（`request.py:L296-L307`）：

```python
def __lt__(self, other: "PolicyRequest") -> bool:
    if self.priority != other.priority:
        return self.priority < other.priority         # tier 1
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time  # tier 2
    if self.request_id != other.request_id:
        return self.request_id < other.request_id    # tier 3
    return id(self) < id(other)                       # tier 4
```

四级 tie-break。每一级都有作用：
- **tier 1 priority**：smaller is higher。这就是为什么 6.4.4 demo 输出里 priority=1 的请求排在 priority=2、priority=3 之前。
- **tier 2 arrival_time**：同 priority 时先到先排。这是把 PRIORITY 退化成 FCFS 的边界情形。
- **tier 3 request_id**：同 priority 同到达时间下的稳定排序——保证测试可重复。
- **tier 4 `id()`**：万一连 request_id 都相同（理论上不应发生），最后用对象内存地址 fallback。

K21（tester 知识）用 3 种显式 permutation 测 determinism——比 `random.seed` 更结实。

### 6.4.3 Demo §4：[B, D, C, A] 的故事

跑 demo 的第 [4] 段：

```
[4] PRIORITY policy + (pedagogical) aging compensator
    arrival order (FCFS):   ['A', 'B', 'C', 'D']
    priority queue order:   ['B', 'D', 'C', 'A']
      → B before D because earlier arrival on tied priority
    aged order at t=10.0:    ['B', 'A', 'C', 'D']
      A's effective priority at t=10: -7
      (NOTE: vLLM does NOT auto-age; this is illustrative only.)
```

Demo 创建 4 个请求（`demo.py:L155-L160`）：

| ID | priority | arrival_time |
|----|---------:|-------------:|
| A  | 3 (low)  | 0.0 |
| B  | 1 (high) | 1.0 |
| C  | 2        | 2.0 |
| D  | 1 (high) | 3.0 |

PRIORITY 排序：先按 priority 升序——B 和 D（都是 1）领先，然后 C（2），最后 A（3）。B vs D 平手——按 arrival_time，B（1.0）比 D（3.0）早。所以 **[B, D, C, A]**。

对比 FCFS 顺序 [A, B, C, D]——按到达时间。两个排序差距明显：A 在 FCFS 里第一个跑，在 PRIORITY 里**最后一个**——因为它优先级最差。

### 6.4.4 aged_priority：一个不存在的特性

demo 第三行（`aged order at t=10.0: ['B', 'A', 'C', 'D']`）展示了一个 vLLM **不**做的事：自动 aging。`aged_priority(req, now, rate=1.0)` 的公式是 `priority - rate * (now - arrival_time)`：

| ID | priority | arrival | wait at t=10 | aged_priority |
|----|---------:|--------:|-------------:|--------------:|
| A  | 3        | 0.0     | 10           | 3 - 10 = **-7** |
| B  | 1        | 1.0     | 9            | 1 - 9 = -8 |
| C  | 2        | 2.0     | 8            | 2 - 8 = -6 |
| D  | 1        | 3.0     | 7            | 1 - 7 = -6 |

aged 顺序按这个新值升序：B (-8), A (-7), C (-6, 但 C.id < D.id 排前面), D (-6)。所以 [B, A, C, D]。**A 从最后变成了第二个**——因为它等得最久。

但 vLLM **不**实现这个。三条理由（impl-notes D1 + `aged_priority` docstring）：
1. **Priority 是 user 的 SLA hint，不是引擎自治权**：付费用户给了 priority=0，他不希望某个等久了的免费请求把他挤下去。
2. **Aging rate 难调**：太大→starves 高优先级；太小→等于没开。每个 workload 都不一样，没法默认。
3. **打破 heap 不变量**：aged_priority 随时间变化——如果它已经在 heap 里，它的 key 在变；下次 `peek_request()` 时 heap top 可能不再是真正最小的。要修必须每步重建 heap，O(n log n) per step，不接受。

`aged_priority` 在 Ch06 里是**反例**——展示 vLLM 没有自动 aging 的具体后果，让读者理解这是**有意识的设计选择**而不是疏忽。

### 6.4.5 select_preemption_victim：max() 不是 min()

打开 `implementation/policy.py:L52-L89` —— 受害者选择函数：

```python
# implementation/policy.py:L80-L89
if policy is SchedulingPolicy.FCFS:
    # REFERENCE: vllm/v1/core/sched/scheduler.py:L504
    return running[-1]                                   # 队尾

# PRIORITY
# REFERENCE: vllm/v1/core/sched/scheduler.py:L480-L483
return max(running, key=lambda r: (r.priority, r.arrival_time))
```

注意 PRIORITY 路径用的是 **`max`** 不是 `min`——这是个面试题级别的细节。

vLLM 的 priority 约定：**smaller value = higher priority**（`config/scheduler.py:L114`）。所以"最大的 priority value"= 最低优先级 = 最该被踢的人。`max(running, key=(priority, arrival_time))` 在 priority 平手时取**最晚到达**的——总是踢最近来的、最低优先级的请求。

如果你看到这里"想 fix" 把 `max` 改成 `min`——别动。K19 这条不变量明确写了：**don't 'fix' this to min().**

demo §1 的输出验证：

```
running = [A(p=1), B(p=3), C(p=2)]
FCFS victim:     C  (running[-1] = last admitted)
PRIORITY victim: B  (max by (priority, arrival_time) → B has p=3, the worst)
```

FCFS 看队尾——admit 顺序最后一个是 C。PRIORITY 看 priority value——B 的 3 是最大（最差），就选 B。

### 6.4.6 select_waiting_queue：FCFS 偏 skipped，PRIORITY 比 head

打开 `implementation/policy.py:L92-L126`：

```python
# implementation/policy.py:L113-L126
if policy is SchedulingPolicy.FCFS:
    # REFERENCE: scheduler.py:L1568-L1569
    return skipped_waiting or waiting or None

if waiting and skipped_waiting:
    # REFERENCE: scheduler.py:L1572-L1575
    w = waiting.peek_request()
    s = skipped_waiting.peek_request()
    return waiting if w < s else skipped_waiting

return waiting or skipped_waiting or None
```

`skipped_waiting` 是个特殊队列：保存"上一步被 peek 但没法 admit 的请求"——比如等远端 KV 传输的请求（`WAITING_FOR_REMOTE_KVS` 状态），或者 LoRA 槽位用满。这些请求是**已经在 waiting 等过一轮**的——如果新到达的 waiting 一直把它们挤后头，会无限期 starve。

**FCFS 的解法（K20）**：永远先 drain `skipped_waiting`——它们已经付过等待时间，不能再让步。`skipped or waiting or None`——一行三选一。

**PRIORITY 的解法**：peek 两个 head 比 `__lt__`。如果新到达的请求 priority 比被卡住的高，那它**应该**插队——这正是 priority 模式的语义。所以是按 `__lt__` 比，不偏向 skipped。

demo §1 验证：FCFS 模式下，waiting 装了 `new-1` (arrival=10.0)，skipped 装了 `blocked-1` (arrival=2.0)。FCFS 选 `blocked-1`——尽管 new-1 也是 FIFO 来说"先到的"——因为 skipped 永远赢。

---

## 6.5 三个旋钮的 Pareto 前沿

### 6.5.1 三个 config 旋钮 = 一个 trade-off 平面

打开 `implementation/pareto.py:L34-L43`：

```python
@dataclass
class EngineConfig:
    max_num_seqs: int                         # vllm/config/scheduler.py:L63
    max_num_batched_tokens: int               # token budget B per step
    long_prefill_token_threshold: int = 0     # scheduler.py:L413-L414
```

三个旋钮在 vLLM 真实 config 里都有：
- **`max_num_seqs`**：同时 running 的请求上限。
- **`max_num_batched_tokens`**：每步 token budget B（第 4 章主角）。
- **`long_prefill_token_threshold`**：单请求每步最多消耗多少 token（fairness cap）。

每个组合是 **(throughput, p95 TTFT)** 平面上的一个点。改旋钮就是在这个平面上移动。Pareto 前沿是"既不能在 throughput 上更好又在 latency 上更好"的点集——任何不在前沿上的配置都被某个前沿配置严格支配。

### 6.5.2 Demo §5：sweep 7 个配置

跑 demo §5：

```
[5] schedule-latency vs throughput Pareto frontier
      max_seqs   budget   long_thr   tput (Mtok/s)    p95 TTFT    sat  pareto?
             8      512          0            0.40       800ms   0.50         
            16     1024          0            0.80       400ms   0.25         
            32     2048          0            1.60       200ms   0.12         
            64     4096          0            3.20       100ms   0.06         
           128     8192          0            6.40        50ms   0.03        ★
            32     2048        512            1.60        50ms   0.12         
            64     4096       1024            3.20        50ms   0.06         

    Pareto frontier has 1 non-dominated points (★).
```

7 行——5 行是不同 (max_seqs, budget) 比例缩放，2 行是开了 `long_prefill_token_threshold` 的同等大小。

**主对角线**（前 5 行）：`max_seqs` 和 `budget` 同步翻倍，throughput 翻倍，p95 TTFT 减半。看起来"更大就更好"——直到你注意到 `sat`（saturation，预算实际使用率）从 0.50 降到 0.03。**128 / 8192 这一档的引擎 97% 时间在 idle**——你为了 p95 TTFT 50ms 浪费了大量算力。

**threshold 行**（最后两行）：`max_seqs=32, budget=2048, threshold=512` 和 `max_seqs=64, budget=4096, threshold=1024`——这两个配置以**1/4 的引擎大小**达到了"前 5 行最大配置才有的 50ms p95 TTFT"。

### 6.5.3 16x p95 TTFT 不是单旋钮的功劳

team-lead 给的写作指导：**"16x"是 sweep-pair**，不是 threshold 单方面。

读 demo 表格：
- **最差 p95 TTFT**：max_seqs=8, budget=512, threshold=0 → **800 ms**。
- **最好 p95 TTFT**：max_seqs=128, budget=8192, threshold=0 → **50 ms**。
- **比值**：800 / 50 = **16x**。

这 16x 是把"最差配置"和"最好配置"放一起得到的。**threshold=0 时**也能 16x（前 5 行已经能跨这范围）。所以 16x 来自 (max_seqs, budget) 的 16x 缩放，不是 threshold 一刀的功劳。

那 threshold 的真正贡献是什么？看最后两行——`(32, 2048, 512)` 和 `(64, 4096, 1024)` 的 p95 TTFT 都是 **50 ms**，throughput 还是分别 1.6 / 3.2 Mtok/s——和"无 threshold 大引擎"相同。**threshold 让小引擎以小代价匹配大引擎的尾延迟**。

公式来自 `pareto.py:L101-L110`：开 threshold 时，长 prompt 单步只占 `threshold` token，剩余 `B - threshold` 留给短请求。代入 `(32, 2048, 512)`：leftover = 2048 - 512 = 1536，短 prompt 64 token 一步搞定——所以 short request wait = 1 step = 50 ms。和 budget=8192、threshold=0 的"短请求一步搞定 64 token in 8192 budget"等价——**用 threshold 在小 budget 里复制了大 budget 的 fairness 效果**。

这就是 **P05 sweep-pair**：headline 16x 是 sweep design 的自然产物，threshold 提供的是另一条 access path 而不是另一个 multiplier。

### 6.5.4 Pareto 前沿在我们的简化模型里只有 1 点

```
Pareto frontier has 1 non-dominated points (★).
```

只有 **(128, 8192, 0)** 一个点上前沿——它在 throughput 上最大、在 p95 TTFT 上最小，没人能支配它。其它 6 个都被它支配。

但**这是模型的属性，不是 vLLM 的属性**。我们的 throughput 模型很粗（`min(decode_tp * max_seqs, B * 1000)`），实际 H100 上随着 max_seqs 增加 decode throughput **不会**线性涨——会被 GPU compute bound 卡住。真实 sweep 应该出现 ~3 个 pareto 点：

- 小 `max_seqs`：低 throughput，低 latency。适合交互场景。
- 中 `max_seqs`：均衡。
- 大 `max_seqs` + threshold：高 throughput，仍然低 latency。

我们的模型把 "compute-bound"（decode_tp × max_seqs）和 "memory-bound"（budget × step rate）取 min——但 compute_bound 的 256 cap 是简化（`pareto.py:L75`），实际 H100 在 256 时已经饱和。校准生产部署时应该把 `decode_throughput_tokens_per_sec` 换成实测值。

**写作框架**：和读者说"看形状，不要看绝对数"。1 点 Pareto 前沿不是 vLLM 调度器的真实属性——是我们模型的简化产物。生产环境实测会出现完整曲线。

### 6.5.5 我们的实现 vs vLLM

vLLM 没有"sweep"或"Pareto front"的代码——这都是分析框架。但每个旋钮都对应到具体源码行：

| 我们 | vLLM 源码 |
|------|-----------|
| `EngineConfig.max_num_seqs` | `config/scheduler.py:L63` |
| `EngineConfig.max_num_batched_tokens` | `config/scheduler.py` (computed at runtime) |
| `EngineConfig.long_prefill_token_threshold` | `config/scheduler.py` + `scheduler.py:L413-L414` |
| `estimate_throughput` 的 budget cap | `scheduler.py:L848-L853` `assert total <= max_num_scheduled_tokens` |
| `estimate_p95_ttft` 的 chunked 公式 | `scheduler.py:L413-L415, L678-L680` 双相 threshold 应用 |

读源码时把这五处行号串起来，就能从 config 字段一路追到调度器的实际行为。

---

## 6.6 我们的实现 vs vLLM 源码：1:1 对照表

| 我们的代码 | vLLM 源码 | 我们改了什么 | 为什么 |
|-----------|----------|-------------|--------|
| `SchedulingPolicy` enum (`request_queue.py:L34-L39`) | `request_queue.py:L13-L17` | 一字不差 | — |
| `PolicyRequest` (`request_queue.py:L43-L71`) | `request.py:L59-L308` | 仅保留 `__lt__` 需要的字段 | Ch06 不需要完整 Request |
| `PolicyRequest.__lt__` (`request_queue.py:L63-L71`) | `request.py:L296-L307` | 一字不差（4 级 tie-break） | tie-break 行为是核心 |
| `RequestQueue` ABC (`request_queue.py:L75-L102`) | `request_queue.py:L20-L72` | 5 抽象方法集合相同 | — |
| `FCFSRequestQueue` (`request_queue.py:L106-L143`) | `request_queue.py:L75-L128` | composition 而不是继承 deque | Ch04 已用此风格 |
| `PriorityRequestQueue.add_request` | `request_queue.py:L144-L146` | 一字不差 | — |
| `PriorityRequestQueue.pop_request` | `request_queue.py:L148-L152` | 一字不差 | — |
| `PriorityRequestQueue.peek_request` | `request_queue.py:L154-L158` | 一字不差 | — |
| `PriorityRequestQueue.prepend_request` (`request_queue.py:L177-L179`) | `request_queue.py:L160-L165` | 一字不差（K18 不变量） | 核心不变量 |
| `PriorityRequestQueue.prepend_requests` | `request_queue.py:L167-L173` | 一字不差 | — |
| `PriorityRequestQueue.__iter__` (heap 复制后 pop) | `request_queue.py:L194-L198` | 一字不差 | 不破坏原 heap |
| `create_request_queue` factory | `request_queue.py:L201-L208` | 一字不差 | — |
| `PauseState` enum (`policy.py:L33-L49`) | `interface.py` PauseState | 三态匹配 | — |
| `select_preemption_victim` (`policy.py:L52-L89`) | `scheduler.py:L478-L504` | 提取为纯函数 | testability |
| `select_waiting_queue` (`policy.py:L92-L126`) | `scheduler.py:L1567-L1577` | 提取为纯函数 | testability |
| `effective_token_budget` (`policy.py:L129-L144`) | `scheduler.py:L372-L374` | 单行 primitive | — |
| `PreemptionStrategy` enum (`preemption_strategy.py:L50-L53`) | scheduler.py + kv_offload + finish_requests | 三路径合成 | analytical comparator |
| `PreemptionScenario.kv_bytes` (`preemption_strategy.py:L70-L84`) | `kv_cache_interface.py:L153-L170` | 同 Ch05 公式 | per-token 相加 |
| `PreemptionScenario.recompute_seconds` | `scheduler.py:L961-L964` (mechanics) | analytical model | 延迟比较 |
| `PreemptionScenario.swap_seconds` | `kv_offload/cpu/gpu_worker.py:L319` | analytical model | 同上（注：这条不是 preempt 路径） |
| `PreemptionScenario.abort_seconds` | `scheduler.py:L1750-L1811` (mechanics) | SLA penalty model | 同上 |
| `crossover_prompt_length` (`preemption_strategy.py:L124-L153`) | — | new (P02) | 长度独立性证明 |
| `expected_latency_under_oom_rate` | — | new | per-request E[latency] |
| `WorkloadProfile` (`starvation_analysis.py:L41-L96`) | `scheduler.py:L387-L556` (Phase 1) + L568-L846 (Phase 2) | analytical worst-case | starvation 量化 |
| `priority_ordering` (`starvation_analysis.py:L100-L110`) | `request_queue.py:L131-L198` | 调用 PriorityRequestQueue | testability |
| `aged_priority` (`starvation_analysis.py:L114-L135`) | — NOT in vLLM | new (illustrative) | D1 反例 |
| `EngineConfig` (`pareto.py:L33-L42`) | `config/scheduler.py:L60-L120` | 三旋钮子集 | sweep 方便 |
| `estimate_throughput` (`pareto.py:L55-L78`) | `scheduler.py:L848-L853` (budget cap) | min(compute, memory) 模型 | Pareto 轴 |
| `estimate_p95_ttft` (`pareto.py:L81-L110`) | `scheduler.py:L413-L415, L678-L680` | analytical with/without threshold | Pareto 轴 |
| `pareto_front` (`pareto.py:L149-L166`) | — | Pareto extractor | sweep 可视化 |

**故意砍掉的内容**（每项指向后续章节）：

- `_preempt_request` / `allocate_slots` 实际机制——Ch04 已覆盖，Ch06 只引用回去。
- streaming sessions / structured output / KV connector pause hooks——第 13-15 章生态特性。
- `id()` 第 4 级 tie-break 的显式 demo——`heapq` 通过 `__lt__` 链隐式处理；不是 Ch06 主线。
- 真实 throughput benchmark 数字——`pareto.py` 是 back-of-envelope；生产部署需要校准 `decode_throughput_tokens_per_sec` 实测值。

---

## 验证

### 跑测试

```bash
cd instances/vllm/artifacts/06-scheduling
python3 -m pytest tests/ --ignore=tests/_legacy -q
```

预期输出：

```
97 passed in 0.16s
```

97 个测试覆盖 6 个模块：

| 模块 | 测试数 | 验证什么 |
|------|------:|---------|
| `test_request_queue.py` | 22 | `__lt__` 4 级、FCFS/PRIORITY 五方法、K18 不变量、K21 determinism |
| `test_policy.py` | 16 | 受害者选择 K19、waiting 队列选择 K20、PauseState 三态 |
| `test_preemption_strategy.py` | 17 | KV bytes 公式、recompute/swap/abort 数值、P02 length-independent |
| `test_starvation_analysis.py` | 14 | head-of-line factor、`has_starvation` 谓词、aged_priority 反例 |
| `test_pareto.py` | 16 | p95_ttft 公式、`pareto_front` dominance、P05 sweep-pair 16x |
| `test_integration.py` | 12 | 端到端 demo、K22 cross-chapter import 韧性 |

### 跑 lint

```bash
python3 scripts/lint_formulas.py instances/vllm/artifacts/06-scheduling/narrative/chapter.md
python3 scripts/lint_source_grounding.py instances/vllm/artifacts/06-scheduling/
```

两个都必须 PASS。

### 跑 demo

```bash
python3 -m instances.vllm.artifacts.06-scheduling.implementation.demo
```

对照 §6.2.1 / §6.3.1 / §6.4.3 / §6.5.2 的输出。结尾应该看到 `Pareto frontier has 1 non-dominated points (★)`。

---

## 总结

1. **Policy 不是一个枚举字段，是贯穿调度全流程的取舍。** Ch04 走读 `schedule()` 时跳过了三处 if/else——preempt 受害者选择、waiting/skipped 选择、`prepend_request` 语义。这三处合起来定义了 FCFS vs PRIORITY 两种"公平观"——FCFS 的"先来的先享受"，PRIORITY 的"优先级低的先牺牲"。

2. **FCFS 有结构性的 head-of-line blocking。** Demo 量化为 8.00x——短请求等长请求 prefill 完，等待时间是它自己 prefill 的 8 倍。但 vLLM 默认仍然 FCFS：因为 PRIORITY 在 priority 平等时退化成 FCFS、低优先级请求可能永远 stuck，而且 priority 是 user 给的 SLA hint，不是引擎能自治调整的东西。

3. **Preempt 三路径里实际只有一条是 preempt：recompute。** Swap 在源码里只服务 prefix cache offload（`kv_offload/`）；abort 是 API/admin 触发的取消，不是 OOM。三个分别 164 ms / 62 ms / 5000 ms（8K prompt）——swap 最快但 vLLM v1 不用，因为简化代码、零 CPU 依赖、bit-determinism 比延迟更重要。**说"recompute 更快"是错的，"recompute 更简单"才是对的。**

4. **P02：crossover length-independent。** L 在 recompute=swap 等式两边消掉——dispatch 决策只依赖 (TP, BW, model shape)。32 layer / 8 KV head / 128 / fp16 模型上，threshold TP = 131K tok/s；H100 实测 50K tok/s 远低于此，所以"swap 永远比 recompute 快"，不分 prompt 长度。Tester 用 L=128 vs L=131072 的 dispatch equality 测试钉住这点。

5. **K18：PriorityRequestQueue 没有"front"概念。** `prepend_request` == `add_request` 一字不差。这条不变量决定了 PRIORITY 模式下 preempt 的请求**会在它本来的优先级槽位重新排队**——如果优先级低，可能永远 stuck。这不是 bug，是设计；如果你不想这样，那就**不要**给请求设很差的 priority。

6. **三个旋钮的 Pareto 前沿，16x 是 sweep design 不是 threshold 单刀。** 800 ms → 50 ms 的 16x 来自 (max_seqs, budget) 同步从 (8, 512) 到 (128, 8192) 的缩放；threshold 的真正贡献是让小引擎以小代价 (32, 2048, 512) 匹配大引擎的 50ms p95 TTFT。生产部署校准时把 `decode_throughput_tokens_per_sec` 换成实测值，会出现完整 Pareto 曲线（我们简化模型只出 1 点）。

### 下章预告

第 4-6 章把 vLLM 单实例的调度故事讲完了：机械（Ch04）+ 内存（Ch05）+ 策略（Ch06）。但生产部署里有更多约束：LoRA 适配器要按 batch 分组、多租户要做隔离、用户 token 配额要排队。第 7 章 `Multi-Tenancy & LoRA Scheduling` 把 Ch06 的 policy primitives 当作 substrate，往上叠这些生态层。

---

← 第 5 章：GPU 显存管理系统 | 第 7 章：多租户与 LoRA 调度 →
