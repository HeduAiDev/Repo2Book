# 第4章：Continuous Batching — 排队打饭与动态调度

> 打开 `vllm/v1/core/sched/scheduler.py:L352`。`schedule()` 方法开头的注释里有全章最关键的一句话：
>
> *"There's no 'decoding phase' nor 'prefill phase' in the scheduler.
> Each request just has num_computed_tokens and num_tokens_with_spec."*
>
> 这句话定义了 Continuous Batching 的本质——也是本章要讲清楚的唯一一件事。

---

## Cell 2 — 开门见山：一个排队打饭的故事

### 故事：两种食堂

想象你去一个食堂吃饭。食堂有三种菜：米饭、炒菜、汤。你有 3 个朋友（r1, r2, r3），每个人点了不同数量的菜。

**传统食堂（Static Batching）规定：**

所有人必须一起行动，分三个阶段——
1. 所有人一起打完米饭
2. 所有人一起打完菜
3. 所有人一起打完汤

r1 点了 800 碗米饭，r3 只点了 50 碗。但规则是"一起行动"，所以 r3 打完 50 碗米饭后，必须在旁边干等，直到 r1 打完 800 碗，才能一起进入"打菜"阶段。

这就是 **bubble**：GPU 有能力算更多，但被"所有人必须一起"的规则卡住了。

**自助餐厅（Continuous Batching）改了规则：**

每人拿一个盘子（能装 512 个菜 = token budget）。想装什么装什么——先装米饭、再装菜、再盛汤，随便混。盘子满了就去结账吃饭，吃完再来装。不用等任何人。

r1 第一趟先装 512 个米饭（盘子满了）→ 结账 → 回来装 288 个米饭和 20 个炒菜...
r2 第一趟装完 200 个米饭 → 结账。
r3 第一趟装了 24 个米饭（跟 r1 剩下的 288 和 r2 的 200 一起，刚好 512）→ 结账。

结果：传统食堂需要 900 趟（800 步米饭 + 100 步汤）。自助餐只需要约 3 趟（总菜数 1220 / 每盘 512 ≈ 3）。

**快了 300 倍。** 这不是魔法，是消除了"同步等待"。

> **这个类比的核心对应关系：**
> - 餐盘容量 = Token Budget $B$（`max_num_scheduled_tokens`）
> - 正在装菜的人 = Running 请求（已有 KV Cache，优先服务）
> - 排队的人 = Waiting 请求（FCFS 队列）
> - 餐盘满了结账 = budget 用完，当前 step 结束
> - 有人吃饱走人 = 请求完成，释放 KV Cache
> - 有人占着盘子不动 = Preempt，赶回队首重排

### Source Trail

打开 `vllm/v1/core/sched/scheduler.py:L67`。`Scheduler` 类是 vLLM 推理引擎的大脑。每个 step 它只做一件事：

**在 token budget 和 KV Cache 容量的约束下，决定每个请求推进多少 token。**

`vllm/v1/core/sched/scheduler.py:L353-L362` 的注释是这样写的：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has num_computed_tokens and num_tokens_with_spec.
# num_tokens_with_spec =
#   len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec. This is general enough to cover
# chunked prefills, prefix caching, speculative decoding,
# and the "jump decoding" optimization in the future.
```

woosuk（vLLM 作者之一）写这段注释时表达的核心思想就是：**不要把请求分成"prefill"和"decode"两组。** 每个请求只是一个 token 流——有"已处理游标"（`num_computed_tokens`）和"目标总量"（`num_tokens_with_spec`）。scheduler 的工作是把每个请求的游标往前推，仅此而已。

为什么这个视角如此重要？因为它让你可以在同一个 step 里混合 prefill chunk 和 decode token。就像自助餐厅里，有人在装米饭（prefill），有人在盛汤（decode），谁也没在等谁。

---

## Cell 3 — 问题演示：Static Batching 撑不住的场景

### Source Trail: 问题的数学建模

打开 `vllm/v1/core/sched/scheduler.py:L388-L556`（Phase 1: Running 请求的处理）和 `vllm/v1/core/sched/scheduler.py:L568-L846`（Phase 2: Waiting 请求的准入）。这个 two-phase 结构的存在本身，就是 static batching 的"罪证"——正是因为旧模式有同步瓶颈，才需要这个两阶段设计来打破它。

我们来看看 static batching 到底差在哪。打开我们的实现 `artifacts/04-continuous-batching/implementation/scheduler.py:L544`：

```python
# scheduler.py:L544-L575 — static_batching_simulation
# 核心逻辑：分两个阶段串行执行
# Prefill phase: 等所有人完成 prompt
max_prompt = max(prompt_lens)
steps += max_prompt   # 等最长 prompt 算完——短的在干等！

# Decode phase: 等所有人完成输出
while any(active):
    for i in range(len(requests)):
        if active[i]:
            remaining_output[i] -= 1
            if remaining_output[i] <= 0:
                active[i] = False
    steps += 1
```

### Theory: Bubble 的形式化

Static batching 的总步数由两个瓶颈决定——最长 prompt 和最长输出：

$$
T_{\mathrm{static}} = \max_i P_i + \max_i O_i
$$

其中 $P_i$ 是 prompt 长度，$O_i$ 是输出长度。

换个说法：**静态批处理的效率由"最慢的人"决定，不是由"总量"决定。** 就像传统食堂——所有人等最慢的那个人打完米饭，才能进入下一阶段。短请求的时间和 GPU 能力被白白浪费了。

**利用率的精确计算：** 设 batch 内有 $N$ 个请求，总 token 数为 $S = \sum(P_i + O_i)$。每步 GPU 一个请求一个位置，处理 $N$ 个 token。总处理能力：$T_{\mathrm{static}} \times N$。利用率：

$$
\eta_{\mathrm{static}} = \frac{\sum_i (P_i + O_i)}{(\max_i P_i + \max_i O_i) \times N}
$$

### 数值演示

用 demo 的 3 个请求做具体计算：

| 请求 | Prompt | Output | 总计 |
|------|--------|--------|------|
| r1 | 800 | 20 | 820 |
| r2 | 200 | 50 | 250 |
| r3 | 50 | 100 | 150 |

$\max P_i = 800$，$\max O_i = 100$。

$$
T_{\mathrm{static}} = 800 + 100 = 900
$$

总 token：

$$
800 + 200 + 50 + 20 + 50 + 100 = 1220
$$

GPU 总处理能力为 $900 \times 3 = 2700$（每步 3 个位置）。

$$
\eta_{\mathrm{static}} = \frac{1220}{2700} \approx 45.2\%
$$

**54.8% 的 GPU 能力在空转。** 这就是 bubble。

### Bubble 的本质：同步瓶颈

效率公式的分母是 $\max_i P_i + \max_i O_i$，分子是 $\sum_i (P_i + O_i)$。当请求之间长度差异大时：

$$
\sum_i P_i \ll N \times \max_i P_i
$$

短的被长的拖累。而且更致命的是：**batch size $N$ 远小于 GPU 的 token 处理能力 $B$。** GPU 每步能处理 2048 个 token，但 static batching 每步只处理 $N$ 个——约 680 倍的浪费，绝大部分算力在 idle。

### 我们的模拟验证

`bubble_analysis()` 函数（`scheduler.py:L635`）用 8 个请求（4 个长 prompt 2048、4 个短 prompt 128）验证：

```bash
python3 implementation/scheduler.py
```

Bubble analysis 部分输出：

```
Static batching:       2304 steps
Continuous batching:    261 steps
Speedup:               8.8x
Static GPU utilization: 58.3%
```

8 个请求时利用率 58.3% 比 3 个请求时的 45.2% 高一些——请求越多，$\max P_i$ 和 $\sum_i P_i$ 的比例差距越小。但 58.3% 仍然意味着超过 40% 的 GPU 在空转。**而且你没法控制——用户发来什么长度的 prompt 你都得处理。**

### 回到排队打饭的故事

传统食堂里，3 个朋友面对"必须一起行动"的规则：
- r1 点了 800 碗米饭 + 20 碗汤
- r2 点了 200 碗米饭 + 50 碗汤
- r3 点了 50 碗米饭 + 100 碗汤

结果：r3 在"米饭阶段"的第 51 步就打完自己那份了，但他必须等 r1 打完 800 碗——等了 750 步！期间他不仅无聊，厨房的 512 个灶台也只用了 3 个。

自助餐就没有这个问题——r3 第一趟打好米饭就开始喝汤，不用等任何人。

**关键结论：static batching 的 bubble 是结构性的——来自"所有人必须同步"这条规则。只要请求长度不均匀，就有 bubble，无法消除。**

---

## Cell 4 — 理论：Continuous Batching 的形式化模型

### 历史纵深：从 Orca 到 vLLM

Continuous batching 不是凭空出现的。它的思想可以追溯到 2022 年 OSDI 会议上发表的 **Orca 系统**（Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models"）。Orca 首次提出了 **iteration-level scheduling**——在每次迭代时动态选择哪些请求进入 batch，而不是固定 batch 然后等所有请求完成。

但 Orca 的调度粒度仍然在"请求"级别——每个请求要么被选中（执行一个完整的 attention 操作），要么不被选中。

**vLLM（Kwon et al., SOSP 2023）向前迈出了关键一步：** 将调度粒度从"请求"下沉到"token"。`num_computed_tokens` 这个追踪变量让 scheduler 不再需要知道一个请求在"prefill"还是"decode"——它只需要知道"还剩多少 token 没算"。这使得：
1. 一个请求的 prefill 可以被切成多块（chunked prefill）
2. prefill token 和 decode token 可以在同一个 forward pass 中混合
3. KV Cache 管理（PagedAttention）和 token 调度（Continuous Batching）解耦——前者只管"存哪"，后者只管"算多少"

这就是 vLLM 比其他推理框架快 2-23 倍的核心原因之一。

### Token Budget 模型

回到我们的形式化模型。Continuous batching 的核心约束只有一个：**每步最多处理 $B$ 个 token。**

打开 `vllm/v1/core/sched/scheduler.py:L371`：

```python
token_budget = self.max_num_scheduled_tokens
```

以及配置定义 `vllm/config/scheduler.py:L56-L63`：

- L56：`max_num_scheduled_tokens: int | None = None`——每步调度最多发出的 token 数。默认为 `None`，运行时取 `max_num_batched_tokens`（`DEFAULT_MAX_NUM_BATCHED_TOKENS`）
- L63：`max_num_seqs: int = DEFAULT_MAX_NUM_SEQS`——最多同时运行的请求数

$B$（token budget）由 GPU 的显存和计算能力决定，通常 2048 或 4096。

**CB 的步数下界（完美调度时）：**

$$
T_{\mathrm{CB}} \geq \left\lceil \frac{\sum_i (P_i + O_i)}{B} \right\rceil
$$

每步最多 $B$ 个 token，总 token 数为 $\sum(P_i + O_i)$——即使完美调度，步数也不能少于这个下界。

**CB 的利用率：**

$$
\eta_{\mathrm{CB}} = \frac{\sum_i (P_i + O_i)}{T_{\mathrm{CB}} \times B}
$$

当 $T_{\mathrm{CB}}$ 达到下界（完美调度）时，$\eta_{\mathrm{CB}} \approx 1$。

### 对比：为什么 CB 碾压 Static？

把两个利用率公式放在一起看：

$$
\eta_{\mathrm{static}} = \frac{\sum_i (P_i + O_i)}{(\max_i P_i + \max_i O_i) \times N}
\qquad\mathrm{vs}\qquad
\eta_{\mathrm{CB}} = \frac{\sum_i (P_i + O_i)}{T_{\mathrm{CB}} \times B} \approx 1
$$

唯一的区别在分母：
- Static：受限于**最长请求** $\times$ **请求数**（$(\max P_i + \max O_i) \times N$）
- CB：受限于**总 token 数** / **每步处理能力**（$T_{\mathrm{CB}} \times B$）

直觉：Static 的步数由"最慢的人"决定（$T = \max P_i + \max O_i$），CB 的步数由"总工作量"决定（$T \approx \sum / B$）。就像自助餐厅——一个人吃得多没关系，他多跑几趟就行，别人不用等他。

当 $N \ll B$ 时（实际场景几乎总是如此）：

$$
\frac{\eta_{\mathrm{CB}}}{\eta_{\mathrm{static}}} \approx \frac{T_{\mathrm{static}} \times N}{T_{\mathrm{CB}} \times B}
$$

代入 3 请求例子（$T_{\mathrm{static}} = 900$，$T_{\mathrm{CB}} \approx 3$，$N = 3$，$B = 512$）：

$$
\frac{\eta_{\mathrm{CB}}}{\eta_{\mathrm{static}}} \approx \frac{900 \times 3}{3 \times 512} \approx 1.76
$$

### Bubble 的形式化证明

定义 bubble ratio $R$ 为浪费的 GPU 能力比例：$R = 1 - \eta$。

**Static 的 bubble ratio 下界：**

$$
R_{\mathrm{static}} \geq 1 - \frac{\sum_i P_i}{N \cdot \max_i P_i} = \frac{N \cdot \max_i P_i - \sum_i P_i}{N \cdot \max_i P_i}
$$

这个值永远大于 0（除非所有 $P_i$ 相等）。因为最长 prompt 至少等于平均 prompt，且实际请求长度不均匀，最长远超平均：$\max_i P_i \gg \mathrm{avg}(P_i)$。

**CB 的 bubble ratio：**

$$
R_{\mathrm{CB}} = 1 - \frac{\sum_i (P_i + O_i)}{T_{\mathrm{CB}} \times B}
$$

当 $T_{\mathrm{CB}}$ 达到下界时：

$$
R_{\mathrm{CB}} \leq \frac{B - \big(\sum(P_i + O_i) \bmod B\big)}{\lceil \sum(P_i + O_i) / B \rceil \times B} < \frac{1}{\lceil \sum(P_i + O_i) / B \rceil}
$$

CB 的 bubble **仅来自最后一个不完整步的剩余空间**——当总 token 数不是 token budget 的整数倍时，最后一步有小于 $B$ 的碎片。这个值通常极小（低于 1%）。

**核心结论：Static 的 bubble 来自"所有人都要等最慢的人"（结构性的），CB 的 bubble 来自"最后一步没装满"（非结构性的、可忽略的）。**

### Running-First 设计的理论理由

为什么 vLLM 的 scheduler 先处理 running 请求，再处理 waiting 请求（two-phase 结构）？

回到排队打饭的类比：

1. **已经拿着盘子正在装菜的人（running 请求）** 已经占用了餐盘（KV Cache）。如果推迟他们，这些餐盘既不能释放也不能给别人用——资产闲置。
2. **正在排队的人（waiting 请求）** 还没拿到餐盘。让他们等一下不会造成资源浪费。
3. **如果有人盘子装不满就占着不走？** Preempt——管理员的最后手段：赶走一个已经占着盘子但进展慢的人（FCFS 最低优先级 = 最后来的），让他重新排队，腾出餐盘给前面的人。

Preempt 驱逐的是"最低优先级"的 running 请求（FCFS 顺序中的最后一个，即最后到达的）。这是最小不公平的选择——一个请求的 OOM 不会导致它自己被驱逐，只有"后面来的"被牺牲。

---

## Cell 5 — 代码走读：`schedule()` 逐行解析

> **运行我们的实现看实际输出：**
>
> ```bash
> cd artifacts/04-continuous-batching/implementation
> python3 scheduler.py
> ```
>
> 你会在终端看到 8 个 step 的调度追踪，显示了 running-first 调度、chunked prefill 的切分、以及 prefill+decode 在同一个 step 里混合执行的完整过程。

现在让我们逐行走读核心代码。打开 `artifacts/04-continuous-batching/implementation/scheduler.py:L258-L434`——`ContinuousBatchingScheduler.schedule()` 方法。

### 方法签名和初始化

```python
# scheduler.py:L258-L275
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens  # L264: B = 512
    scheduled: Dict[str, int] = {}
    finished: List[str] = []
    preempted_req_ids: List[str] = []
    newly_running_req_ids: List[str] = []
```

**REFERENCE: vllm/v1/core/sched/scheduler.py:L352-L370** —— vLLM 的 `schedule()` 有相同的 `token_budget` 和四个记录列表。差异在于 vLLM 还创建了 `scheduled_new_reqs`、`scheduled_cached_reqs` 分别追踪首次调度和命中 prefix cache 的请求。

### Phase 1: Running 请求优先

```python
# scheduler.py:L278-L350
# REFERENCE: vllm/v1/core/sched/scheduler.py:L388-L556
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    req = self.running[req_index]

    # L287: 不要超过 budget！
    num_new = min(req.num_new_tokens, token_budget)
    if num_new <= 0:
        req_index += 1
        continue

    # L294-296: 看 KV Cache 够不够
    new_blocks = self.kv_cache_manager.allocate_slots(
        req.request_id, num_new
    )

    # L301-335: OOM → preempt 循环
    while new_blocks is None:
        if len(self.running) <= 1:
            break  # 只有自己，无法驱逐

        preempted_req = self.running.pop()  # 驱逐最后一个（最低优先级）
        self._preempt_request(preempted_req)
        preempted_req_ids.append(preempted_req.request_id)

        # L325-330: 归还被驱逐请求的 token 到 budget
        if preempted_req.request_id in scheduled:
            freed_tokens = scheduled.pop(preempted_req.request_id)
            token_budget += freed_tokens

        # 重试分配
        new_blocks = self.kv_cache_manager.allocate_slots(
            req.request_id, num_new
        )

    if new_blocks is None:
        break  # 驱逐所有人仍失败，放弃此请求

    # L343-347: 分配成功
    req.block_ids.extend(new_blocks)
    scheduled[req.request_id] = num_new
    token_budget -= num_new
    req_index += 1
```

**逐行解释关键决策：**

1. **Line 287: `num_new = min(req.num_new_tokens, token_budget)`** —— 这是 token budget 约束的核心实现。对 decode 请求（`num_new_tokens = 1`），`min(1, budget) = 1` 总是成立。对 prefill 请求（可能很大），会被 budget 截断——这就是 chunked prefill 的第一步截断。

2. **Lines 301-335: Preempt 循环** —— 当 `new_blocks is None`（KV Cache 满了），不直接放弃。而是从 running 列表尾部驱逐最低优先级的请求（FCFS 顺序中最后到达的）。**驱逐 = 释放它的 KV Cache block → 重置它的 computed_tokens → 插入 waiting 队首等待重调。**

3. **Lines 325-330: 归还 budget** —— 如果被驱逐的请求已经在这个 step 获得了 token 分配，这些 token 必须归还到 budget。否则浪费了。这也是 vLLM 源码（`scheduler.py:L485-L502`）的设计。

4. **Line 304: `if len(self.running) <= 1`** —— 关键保护：不能驱逐自己。如果列表只剩当前请求，preempt 不可能成功，break。

**vLLM 的差异：** vLLM 支持 `SchedulingPolicy.PRIORITY` 和 `SchedulingPolicy.FCFS` 两种策略；在 PRIORITY 模式用 `max(self.running, key=lambda r: r.priority)` 选择被驱逐者。此外 vLLM 还要处理 spec decode token（`scheduler.py:L524-L540`）和 encoder input（`scheduler.py:L542-L555`）。

### Phase 2: Waiting 请求准入

```python
# scheduler.py:L352-L418
# REFERENCE: vllm/v1/core/sched/scheduler.py:L568-L846
if not preempted_req_ids:  # Phase 1 有 preempt → 跳过 Phase 2
    while (self.waiting and token_budget > 0
           and len(self.running) < self.max_num_running_reqs):
        req = self.waiting[0]

        num_new = req.num_new_tokens

        # L377-389: Chunked prefill 决策
        if self.enable_chunked_prefill:
            num_new = min(num_new, token_budget)  # 切！
        elif num_new > token_budget:
            break  # 不能切 → 放不下 → 停止准入

        if num_new <= 0:
            self.waiting.pop(0)
            continue

        # L395-403: KV Cache 检查
        blocks = self.kv_cache_manager.allocate_slots(
            req.request_id, num_new
        )
        if blocks is None:
            break  # Cache 满了 → 停止准入（注意这里没有 preempt！）

        # L406-414: 从 waiting 移动到 running
        req.block_ids = blocks
        req.status = RequestStatus.RUNNING
        scheduled[req.request_id] = num_new
        token_budget -= num_new
        newly_running_req_ids.append(req.request_id)
        self.waiting.pop(0)
        self.running.append(req)
```

**逐行解释关键决策：**

1. **Line 360: `if not preempted_req_ids`** —— 如果 Phase 1 发生了 preempt，整个 Phase 2 被跳过。为什么？被驱逐的请求已经回到 waiting 队首了——如果同时 admit 新请求，被驱逐的请求竞争不过，得不到快速重试。vLLM 源码 `scheduler.py:L568` 同样有 `if not preempted_reqs` 守卫。

2. **Lines 377-389: Chunked prefill** —— `enable_chunked_prefill=True` 时，`num_new = min(num_new, token_budget)` 把长 prompt 切成 token_budget 大小的块。如果关闭此开关，任何 prompt 长度超过剩余 budget 的请求都被拒绝。

3. **Line 404: `break` 而非 preempt** —— Phase 2 的 KV Cache 不够时直接停止准入，不驱逐 running 请求。这是刻意设计：running 请求已有 cache 投资，不应为 waiting 请求牺牲。

**vLLM 的差异：** vLLM 的 Phase 2 更复杂。它处理 prefix cache 命中（`get_computed_blocks()`，`scheduler.py:L614-L662`）、remote KV transfer（`WAITING_FOR_REMOTE_KVS` 状态）、`max_loras` 约束、以及 `skipped_waiting` 队列（被结构化输出或 remote KV 阻塞的请求不会卡住整个 waiting 队列）。

### 组装 SchedulerOutput

```python
# scheduler.py:L421-L434
output = SchedulerOutput(
    scheduled_requests=scheduled,
    total_scheduled_tokens=sum(scheduled.values()),
    finished_req_ids=finished,
    preempted_req_ids=preempted_req_ids,
    newly_running_req_ids=newly_running_req_ids,
)
```

**REFERENCE: vllm/v1/core/sched/output.py:L181-L200** —— vLLM 的 `SchedulerOutput` 包含更多字段：`scheduled_new_reqs`（NewRequestData 列表）、`scheduled_cached_reqs`、`num_scheduled_tokens`（字典形式）、`scheduled_spec_decode_tokens`、`scheduled_encoder_inputs`、`num_common_prefix_blocks` 等。我们的简化版只保留 narrative 和 test 需要的最小字段。

---

## Cell 6 — 实现：关键数据结构

### 状态机：从排队到吃完的完整生命周期

打开 `vllm/v1/request.py:L310-L337`。vLLM 定义了 6 个核心状态：

```
排队(WAITING) ──→ 正在打饭(RUNNING) ──→ 吃完了(FINISHED_*)
                         │
                         └──→ 被赶回去(PREEMPTED) ──→ 重新排队(WAITING)
```

在我们的实现中（`scheduler.py:L46-L57`）：

```python
class RequestStatus(IntEnum):
    """REFERENCE: vllm/v1/request.py:L310-L337"""
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2          # "活着"和"死了"的分界线
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5

    def is_finished(self) -> bool:
        return self.value > RequestStatus.PREEMPTED.value
```

PREEMPTED = 2 是"活着"和"死了"的分界线——`is_finished()` 判断 `status > PREEMPTED`（`vllm/v1/request.py:L332-L333`），所以 PREEMPTED 请求被视为"活着，等待重新调度"。

![Request 状态机图](../diagrams/state_machine.png)

> *图注：Request 状态机。WAITING → RUNNING 是准入（allocate KV blocks）；RUNNING → PREEMPTED 是 OOM 驱逐（free blocks、reset computed_tokens）；PREEMPTED → WAITING 后重新排队。三个 FINISHED_* 是终态。*

### Request 的三个核心 Property

```python
@dataclass
class Request:
    """REFERENCE: vllm/v1/request.py:L59-L308"""
    request_id: str
    prompt_token_ids: List[int]
    max_tokens: int
    arrival_time: float
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    output_token_ids: List[int] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)
```

三个 property 捕获了 continuous batching 的核心状态：

- **`num_new_tokens`**（L90-92）：`num_tokens - num_computed_tokens`——不问 prefill/decode，只问"还剩多少"。这是 scheduler 唯一关心的指标。
- **`is_prefill`**（L95-101）：`num_computed_tokens < len(prompt_token_ids)`——当 computed 追上 prompt 长度时，请求进入 decode 阶段。这是一个派生状态，不是调度决策的依据。
- **`num_tokens`**（L85-87）：`len(prompt_token_ids) + len(output_token_ids)`——请求的总 token 数。

### Preempt：驱逐的完整流程

```python
# scheduler.py:L436-L460
def _preempt_request(self, req: Request) -> None:
    """REFERENCE: vllm/v1/core/sched/scheduler.py:L952-L972"""
    self.kv_cache_manager.free(req.block_ids)  # 释放 KV Cache
    req.block_ids = []
    req.status = RequestStatus.PREEMPTED
    req.num_computed_tokens = 0  # 所有 K、V 丢了，必须从头重算
    self.waiting.insert(0, req)  # 插入队首 → 下个 step 优先重调
```

三个动作一口气完成：释放所有 KV Cache block → 重置 computed_tokens 为 0 → 插入 waiting 队首（最高优先级，保证下个 step 就被调）。

**注意：** `self.running.remove(req)` 不在 `_preempt_request` 内部——由调用者负责。这是 vLLM 的设计（`scheduler.py:L503`：调用者先 `running.pop()` 再调用 preempt）。我们的实现注释里特别强调了这一点，避免重复移除。

### update_after_step：模拟 Model Forward 之后的状态更新

```python
# scheduler.py:L462-L515
def update_after_step(self, output: SchedulerOutput) -> None:
    """REFERENCE: scheduler.py:L974-L998 + L1290-L1551"""
    for req_id, num_tokens in output.scheduled_requests.items():
        req = self.requests[req_id]
        req.num_computed_tokens += num_tokens  # 推进游标

        # 模拟 model forward: prefill 完成后产生 decode token
        if not req.is_prefill:
            if len(req.output_token_ids) < req.max_tokens:
                req.output_token_ids.append(0)  # placeholder token

        # 检查完成条件
        if len(req.output_token_ids) >= req.max_tokens:
            req.status = RequestStatus.FINISHED_LENGTH_CAPPED
            self._finish_request(req)
```

真正的 vLLM 中这一步分成了两个函数：`_update_after_schedule`（L974，只做 computed_tokens 预增）和 `update_from_output`（L1290，处理 model runner 返回的实际采样 token、stop string 检测、spec decode rejection）。我们合并简化。

### SimpleKVCacheManager：KV Cache 的简化模型

```python
# scheduler.py:L128-L177
class SimpleKVCacheManager:
    """REFERENCE: vllm/v1/core/kv_cache_manager.py"""
    def allocate_slots(self, request_id, num_new_tokens):
        blocks_needed = (num_new_tokens + block_size - 1) // block_size
        if blocks_needed > len(self._free_blocks):
            return None  # OOM!
        allocated = self._free_blocks[:blocks_needed]
        self._free_blocks = self._free_blocks[blocks_needed:]
        return allocated
```

真正的 `KVCacheManager`（`vllm/v1/core/kv_cache_manager.py`）是一个约 800 行的类——管理 block pool、prefix cache 哈希表、reference counting、lookahead slots、async free 等。我们只留了一个 free list。第 5 章和第 7 章会深入这些细节。

---

## Cell 7 — 数值示例：一步一步追踪调度决策

> **Source Trail:** 这个数值示例追踪 `ContinuousBatchingScheduler.schedule()` 的完整执行路径
> （`scheduler.py:L258-L434`），对应 vLLM 源码 `scheduler.py:L352-L945`。

现在用 demo 的实际 workload 做 step-by-step 追踪。

### 配置与请求

```
Token budget B = 512, 最大 running 请求 = 16
GPU blocks = 1000, block_size = 16, Chunked prefill = ON
```

| 请求 | Prompt | Max Output | num_new_tokens（初始） |
|------|--------|-----------|-----------------------|
| r1 | 800 | 20 | 800 |
| r2 | 200 | 50 | 200 |
| r3 | 50 | 100 | 50 |

### Step 1: 只有 waiting 请求——chunked prefill 首次生效

**调度前：** `running=[]`，`waiting=[r1, r2, r3]`，`budget=512`

**Phase 1:** 空，跳过。

**Phase 2:**
1. r1：`num_new = min(800, 512) = 512`（chunked prefill 截断）。需要 `ceil(512/16) = 32` 个 block。分配 ID 0-31。`budget = 0`，r1 → running。循环结束。

**调度结果：** `scheduled = {r1: 512}`——全是 prefill token，占满 budget。

**更新后：** `r1.num_computed_tokens = 512`，还剩 `800 - 512 = 288` 个 prompt token。`free_blocks = 968`。

### Step 2: running + waiting 混合——三种请求共处一步

**调度前：** `running=[r1]`，`waiting=[r2, r3]`，`budget=512`

**Phase 1 — Running:**
- r1：`num_new = 288`，`min(288, 512) = 288`。需要 18 个 block（ID 32-49）。`budget = 224`。

**Phase 2 — Waiting:**
1. r2：`num_new = 200`，`min(200, 224) = 200`。需要 13 个 block（ID 50-62）。`budget = 24`。
2. r3：`num_new = 50`，`min(50, 24) = 24`（chunked！）。需要 2 个 block（ID 63-64）。`budget = 0`。

**调度结果：** `scheduled = {r1: 288, r2: 200, r3: 24}`——r1 的剩余 prefill、r2 的完整 prefill、r3 的部分 prefill 在一个 step 里共存。

**更新后：**
- r1：computed = 800（prefill 完成 → 触发 1 个 decode token，output = [0]）
- r2：computed = 200（prefill 完成 → 触发 1 个 decode token，output = [0]）
- r3：computed = 24（还剩 26 个 prompt token）

### Step 3: 混合 prefill + decode——CB 的标志性场景

**调度前：** `running=[r1, r2, r3]`，`waiting=[]`，`budget=512`

**Phase 1:**
- r1：`num_new = 1`（decode）。不需要新 block。`budget = 511`。
- r2：`num_new = 1`（decode）。不需要新 block。`budget = 510`。
- r3：`num_new = 26`（剩余 prefill）。需要 2 个 block（ID 65-66）。`budget = 484`。

**调度结果：** `scheduled = {r1: 1, r2: 1, r3: 26}`——2 个 decode token + 26 个 prefill token，同一个 step！

这就是 continuous batching 的标志：**prefill 和 decode 在同一个 step 里共存，互不等待。**

**更新后：**
- r1：computed = 801，output = [0]
- r2：computed = 201，output = [0]
- r3：computed = 50（prefill 完成 → output = [0]）

### Step 4+: 全 decode 阶段

从 Step 4 开始，所有三个请求都在 decode——每步每个请求推进 1 个 token。budget 只用了 3/512。

- r1 在 Step 23 完成（20 个 output token）
- r2 在 Step 73 完成（50 个 output token）
- r3 在 Step 173 完成（100 个 output token）

**总步数约 103 步，对比 static batching 的 900 步——快了约 8.7 倍。**

### 与 demo 输出对齐

运行 `python3 implementation/scheduler.py`，可以验证前 3 步的输出：

```
Step 1
  r1: 512 tokens (prefill), computed=512/800     ← chunked prefill: 截断 800→512
  Budget: 512/512                                  ← 用满

Step 2
  r1: 288 tokens (decode), computed=800/801      ← 剩余 prefill 完成
  r2: 200 tokens (decode), computed=200/201      ← 完整 prefill
  r3: 24 tokens (prefill), computed=24/50        ← chunked prefill: 截断 50→24
  Budget: 512/512

Step 3
  r1: 1 tokens (decode), computed=801/802        ← 纯 decode
  r2: 1 tokens (decode), computed=201/202        ← 纯 decode
  r3: 26 tokens (decode), computed=50/51         ← 剩余 prefill 完成
  Budget: 28/512                                   ← 只用了 28/512
```

注意 Step 2 中 r1 和 r2 标记为 "decode"——因为 `update_after_step` 之后它们的 prefill 已经完成了。调度时它们处理的是 prefill token，但标签反映的是**更新后**的状态。

---

## Cell 9 — 源码映射表

### 我们的实现 vs vLLM 源码

| 我们的代码 | vLLM 源码 (文件:行) | 差异 | 原因 |
|-----------|-------------------|------|------|
| `ContinuousBatchingScheduler.__init__()` (L199-L223) | `vllm/v1/core/sched/scheduler.py:L67-L148` | 简化构造参数，去掉 `VllmConfig`、`KVCacheConfig`、`StructuredOutputManager` | 教学清晰；后续章节详细介绍这些配置 |
| `schedule()` Phase 1 (L278-L350) | `vllm/v1/core/sched/scheduler.py:L388-L556` | 相同 two-phase + preempt-on-OOM；简化 preempt retry | 核心算法一致；vLLM 多了 spec decode、encoder、Mamba alignment |
| `schedule()` Phase 2 (L352-L418) | `vllm/v1/core/sched/scheduler.py:L568-L846` | 相同 budget check + chunked prefill；简化 admission | vLLM 多了 prefix cache、remote KV transfer、`skipped_waiting` |
| `_preempt_request()` (L436-L460) | `vllm/v1/core/sched/scheduler.py:L952-L972` | 相同 free + reset + requeue | vLLM 额外处理 encoder cache、preemption count |
| `update_after_step()` (L462-L515) | `vllm/v1/core/sched/scheduler.py:L974-L998` + `L1290-L1551` | 合并两个函数 | 简化 spec decode rejection、stop string 检测 |
| `_finish_request()` (L517-L533) | `vllm/v1/core/sched/scheduler.py:L1813-L1834` | 相同 free + remove | vLLM 额外处理 KV connector delay-free |
| `add_request()` (L225-L234) | `vllm/v1/core/sched/scheduler.py:L1728-L1748` | 相同 add to waiting queue | vLLM 额外处理 streaming input、structured output |
| `Request.num_new_tokens` (L89-92) | `vllm/v1/request.py:L160-L162` | 相同语义 | vLLM 还考虑 spec token、placeholder |
| `RequestStatus` (L46-L57) | `vllm/v1/request.py:L310-L337` | 相同枚举核心 | 去掉了 `WAITING_FOR_*` 中间状态 |
| `SchedulerOutput` (L108-L121) | `vllm/v1/core/sched/output.py:L181-L200` | 简化字段 | 只保留 narrative + test 所需 |
| `SimpleKVCacheManager` (L128-L177) | `vllm/v1/core/kv_cache_manager.py` | Free list vs block pool + prefix cache | 第 5-7 章深入 block pool 和 prefix cache |

### 主要简化点

**Speculative Decoding（`vllm/v1/core/sched/scheduler.py:L524-L540`）：** vLLM 的 scheduler 追踪 draft model 的 token——小模型生成候选，大模型验证。处理 rejection（减少 `num_computed_tokens`）。

**Encoder-Decoder 模型（`vllm/v1/core/sched/scheduler.py:L427-L439, L542-L555`）：** 支持 T5 等多模态架构，分别追踪 encoder 和 decoder 的 token、cache、compute budget。

**Prefix Cache（`vllm/v1/core/sched/scheduler.py:L614-L662`）：** Phase 2 准入时检查 prefix cache 命中，避免重复计算公共 prompt。第 7 章的主题。

**Skipped Waiting 队列（`vllm/v1/core/sched/scheduler.py:L569-L591, L844-L846`）：** 被 blocked 的请求不卡 waiting 队列，移到 `skipped_waiting` 下次重试。

---

## Cell 10 — 验证

### Bubble 可视化

![排队打饭类比图](../diagrams/cafeteria_analogy.png)

> *图注：传统食堂（左）vs 自助餐（右）对比。传统食堂分三个阶段，每个人必须等最慢的人——产生大量红色 bubble。自助餐每个人拿一个餐盘混装，满了结账再回来——几乎满负荷运转。*

![Bubble 对比图](../diagrams/bubble_comparison.png)

> *图注：Static Batching（左）与 Continuous Batching（右）的调度对比。Static 的 prefill 阶段（蓝）中，短请求等长 prompt 产生大量 bubble（红）；decode 阶段（绿）每步只处理少量 token。CB 把 prefill 和 decode 混合在同一 step 中，GPU 利用率大幅提升。*

![Budget 分配图](../diagrams/budget_allocation.png)

> *图注：Token Budget 的分配流程——Phase 1（running-first）优先满足已有请求，Phase 2 用剩余 budget 接纳新请求。*

### 测试结果

运行测试：

```bash
docker run --rm --entrypoint bash \
  -v /mnt/e/Laboratory/vllm-from-scratch:/workspace \
  --network host vllm/vllm-openai:latest \
  -c "pip3 install pytest -q 2>&1 | tail -1 && \
      cd /workspace/artifacts/04-continuous-batching && \
      python3 -m pytest tests/ -q 2>&1"
```

预期输出：

```
............
14 passed in 0.16s
```

| 测试 | 验证点 |
|------|--------|
| `test_schedule_single_request` | 单请求 WAITING → RUNNING，正确分配 token |
| `test_chunked_prefill_splits_long_prompt` | 长 prompt 500/500 分两步处理 |
| `test_multiple_requests_interleaved` | 多请求同一个 step 被调度 |
| `test_truly_full_kv_cache_skips_waiting` | KV Cache 满时跳过 waiting |
| `test_finished_request_freed` | 完成时正确释放资源 |
| `test_status_lifecycle` | WAITING → RUNNING 状态转换 |
| `test_preempt_lowest_priority_when_oom` | OOM 时驱逐最低优先级 |
| `test_preempt_during_running_phase` | Running 请求驱逐另一个 running |
| `test_continuous_faster_than_static` | CB 比 Static 快 |
| `test_single_request_equal` | 单请求时两种调度等价 |

### Linter 检查

```bash
python3 scripts/lint_formulas.py artifacts/04-continuous-batching/narrative/chapter.md
python3 scripts/lint_source_grounding.py artifacts/04-continuous-batching/
```

---

## Cell 11 — 总结

### 你学到了什么

1. **Continuous Batching = 统一的 token 流模型。** 没有独立的 prefill/decode 阶段。每个请求只有 `num_computed_tokens` 和 `num_tokens`。Scheduler 在 token budget $B$ 的约束下推进每个请求的游标。这就是 `scheduler.py:L353-L362` 注释的核心思想。

2. **排队打饭的类比贯穿全章：** 传统食堂（static batching）的"必须一起行动"产生结构性 bubble——短请求等长请求。自助餐（continuous batching）的"每人一个餐盘，随便装"消除了同步等待——每步只被总容量限制。

3. **Bubble 的形式化证明：** Static 的步数由最长请求决定（$T = \max P_i + \max O_i$），CB 的步数由总量决定（$T = \lceil \sum / B \rceil$）。前者的 bubble 是结构性的（请求间等待），后者只有最后一步的碎片。

4. **Two-phase 调度：** Running-first（Phase 1）保护已有 KV Cache 投资——占着碗的人先吃饱。Waiting（Phase 2）用剩余 budget 填补。Preempt 是 OOM 时的最后手段——把最后来的人请出去重新排队。

5. **Chunked Prefill** 是 CB 的关键技术——把长 prompt 切成 token budget 大小的 chunk，和 decode token 在同一 step 里混合调度。prefill + decode 共存就是 CB 的核心优势。

6. **历史纵深：** Orca（OSDI 2022）首创 iteration-level scheduling，vLLM（SOSP 2023）将粒度下沉到 token 级别——`num_computed_tokens` 这个追踪变量取代了原先的 prefill/decode 阶段分治。

7. **Request 状态机：** WAITING → RUNNING → PREEMPTED → WAITING（循环）或 → FINISHED_*。PREEMPTED 是"活着"和"死了"的分界线。

### 下章预告

Scheduler 的 `allocate_slots()` → `None` 告诉我们显存不够了。但"不够"具体怎么衡量？第 5 章深入 GPU 显存的三级分配器——PyTorch `CachingAllocator` → vLLM `BlockPool` → `KVCacheManager`——理解每一级如何保护下一级不超支。

---

← 第 3 章：FlashAttention & PagedAttention | 第 5 章：GPU 显存管理系统 →
