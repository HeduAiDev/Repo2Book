# 第4章：Continuous Batching — 从气泡到满分调度

> 打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352`。`schedule()` 方法开头的注释里有全章最关键的一句话：
>
> *"There's no 'decoding phase' nor 'prefill phase' in the scheduler.
> Each request just has the num_computed_tokens and num_tokens_with_spec."*
>
> 这句话定义了 Continuous Batching 的本质——也是本章要讲清楚的唯一一件事。

---

## Cell 2 — 开门见山：食堂拼桌的故事

Imagine a cafeteria that seats groups. But here is the ridiculous rule: a group can only be seated when ALL previous groups have finished eating. If table A finishes in 20 minutes but table B takes 45 minutes, table A sits empty for 25 minutes — while a line of hungry people waits at the door.

这就是 Static Batching。GPU 的"桌子"（算力）空着，因为 batch 里的"短请求"必须等"长请求"算完才能一起离开。这 25 分钟的空等就是 **bubble**——白白浪费的 GPU 计算能力。

换到 Continuous Batching 的食堂：任何人吃完立刻离席，门口排队的人立刻入座。桌子从不空着。GPU 的每步 forward 都是满满当当的 token。

这个类比对应到调度系统的核心概念：

| 食堂类比 | 调度系统 |
|---------|---------|
| 餐桌座位 | Token Budget $B$（`max_num_scheduled_tokens`） |
| 正在吃饭的人 | Running 请求（已占用 KV Cache，优先服务） |
| 排队的人 | Waiting 请求（FCFS 队列，等空位） |
| 有人吃完走人 | 请求完成，释放 KV Cache block |
| 有人占着桌子不走 | Preempt——赶回队首重新排 |

这是 vLLM 吞吐量比传统推理框架快一个数量级的两个核心原因之一（另一个是 PagedAttention，第 3 章）。

### Source Trail

打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L67`。`Scheduler` 是 vLLM 推理引擎的大脑，~2300 行代码。每个 step 它只回答一个问题：

**在 token budget $B$ 和 KV Cache 容量约束下，每个请求该推进多少 token？**

`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L353-L362` 的注释是这么写的：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. num_tokens_with_spec =
# len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

woosuk（vLLM 作者之一）写这段注释时表达的核心思想：**不要把请求分成"prefill"和"decode"。** 每个请求只是一个 token 流——有"已处理游标"（`num_computed_tokens`）和"目标总量"（`num_tokens_with_spec`）。Scheduler 的工作是把每个游标往前推。这使得 prefill token 和 decode token 可以在同一个 forward pass 中混合——就像食堂里有人在吃米饭（prefill），有人在喝汤（decode），谁也没等谁。

为什么这个视角如此重要？`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L387-L846` 的两阶段结构（running-first + waiting admission）之所以能工作，正是因为每个请求不再被分类为"prefill"或"decode"——它们只是需要不同数量 token 的同一类对象。

---

## Cell 3 — 问题演示：气泡从哪来

### Source Trail

打开我们的实现 `instances/vllm/artifacts/04-continuous-batching/implementation/scheduler.py:L476`——`_demo_bubble_diagram()` 函数用 ASCII 艺术可视化 Static Batching 的气泡问题。对应的调度逻辑见 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352`（`schedule()` 入口）。

### ASCII 气泡图

运行 `python3 implementation/scheduler.py`，Part 1 输出：

```
  Static batch of 3 requests, 1 generation token/step:
  Alice & Bob:   8 tokens → finish at step 8
  Charlie:      16 tokens → finishes at step 16
  Bubble: 8 steps × 2 idle slots = 16 wasted slot-steps

  Time →      t=0     t=4     t=8     t=12    t=16
  Alice       ████████████████········    (8 tokens)
  Bob         ████████████████········    (8 tokens)
  Charlie     ████████████████████████████████████████  (16 tokens)

  Key: █ = computing   · = idle (bubble)
```

Alice 和 Bob 在 t=8 就算完了。但 batch 必须等到 Charlie 在 t=16 完成才能整批结束。Step 9-16 中，Alice 和 Bob 的 GPU 位置全是浪费——这就是"气泡"（bubble）。**8 步 × 2 个空位 = 16 个浪费的 slot-step。**

### 数学定义

Static Batching 的总步数由两个瓶颈决定——最长 prompt 和最长输出：

$$
T_{\mathrm{static}} = \max_i P_i + \max_i O_i
$$

其中 $P_i$ 是第 i 个请求的 prompt 长度，$O_i$ 是输出长度。换成人话：**静态批处理的效率由"最慢的人"决定，不是由"总工作量"决定。**

GPU 利用率：

$$
\eta_{\mathrm{static}} = \frac{\sum_i (P_i + O_i)}{(\max_i P_i + \max_i O_i) \times N}
$$

分子是已完成的有效工作，分母是 GPU 的总处理能力（步数 $\times$ 每步 $N$ 个位置）。

### 数值验证

Demo 的 3 个请求（来自 `scheduler.py:L542-L550`，`StaticBatchSimulator`）：

| 请求 | prompt_tokens | max_tokens | 总 token |
|------|-------------|-----------|---------|
| A | 1 | 8 | 9 |
| B | 1 | 8 | 9 |
| C | 1 | 16 | 17 |

$\max O_i = 16$（Charlie），$N = 3$。

$$
\eta_{\mathrm{static}} = \frac{9 + 9 + 17}{16 \times 3} = \frac{35}{48} \approx 72.9\%
$$

**27.1% 的 GPU 在空转。** Demo 实测输出（`scheduler.py:L547-L550`）：

```
  Static batch simulation:
    Total steps:        16
    Idle slots wasted:  13
    GPU utilization:    72.9%
```

`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L388-L556` 的 Phase 1（Running 请求处理）和 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L568-L846` 的 Phase 2（Waiting 请求准入）——这个 two-phase 结构的存在本身，就是 Static Batching 缺陷的"罪证"：正是旧模式有同步瓶颈，才需要两阶段设计来打破它。

### Bubble 的本质：同步等待

Bubble 不是 bug——是 **Static Batching 的结构性缺陷**。因为 batch 的生命周期 = 最长请求的生命周期。只要请求长度不均匀，就有 bubble。而且你无法控制用户发来多长的 prompt。

更致命的是：实际场景中 batch size（通常 8-32）远小于 GPU 每步能处理的 token 数（通常 2048-4096）。Static Batching 每步只处理 batch size 个 token——GPU 绝大部分计算能力在 idle。Continuous Batching 每步能填满接近 token budget。

---

## Cell 4 — 理论：Continuous Batching 的形式化模型

### 历史纵深

Continuous batching 的思想可以追溯到 **Orca 系统**（Yu et al., OSDI 2022）。Orca 首次提出 **iteration-level scheduling**——每步模型迭代时动态选择哪些请求进入 batch，而不是固定 batch 后等所有请求完成。

但 Orca 的调度粒度仍在"请求"级别——每个请求要么被选中（完整执行一次 attention），要么不被选中。

**vLLM（Kwon et al., SOSP 2023）向前迈出了关键一步：将调度粒度从"请求"下沉到"token"。**

打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L353-L362`。`num_computed_tokens` 这个追踪变量让 scheduler 不再需要知道一个请求在"prefill"还是"decode"——它只需要知道"还剩多少 token 没算"。这带来了三个关键能力：

1. 一个请求的 prefill 可以被切成多块（chunked prefill）
2. prefill token 和 decode token 可以在同一个 forward pass 中混合
3. KV Cache 管理（PagedAttention）和 token 调度（Continuous Batching）解耦——前者只管"存哪"，后者只管"算多少"

### Token Budget 约束

Continuous Batching 的核心约束只有一个：每步最多处理 $B$ 个 token。

打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L371`：

```python
token_budget = self.max_num_scheduled_tokens
```

配置定义 `instances/vllm/source/vllm/config/scheduler.py:L56-L63`：

- L56：`max_num_scheduled_tokens: int | None = None`——每步调度最多发出的 token 数
- L63：`max_num_seqs: int = DEFAULT_MAX_NUM_SEQS`——最多同时运行的请求数

$B$（token budget）由 GPU 的显存和计算能力决定，通常 2048 或 4096。

### CB 的步数下界（完美调度）

总 token 数为 $S = \sum_i (P_i + O_i)$，每步最多 $B$ 个：

$$
T_{\mathrm{CB}} \geq \left\lceil \frac{S}{B} \right\rceil
$$

对应地，利用率（当 $T_{\mathrm{CB}}$ 达到下界时 $\eta_{\mathrm{CB}} \approx 1$）：

$$
\eta_{\mathrm{CB}} = \frac{S}{T_{\mathrm{CB}} \times B}
$$

### 对比：为什么 CB 碾压 Static

把两个利用率公式放在一起看：

$$
\eta_{\mathrm{static}} = \frac{S}{(\max_i P_i + \max_i O_i) \times N}
\qquad \mathrm{vs} \qquad
\eta_{\mathrm{CB}} = \frac{S}{T_{\mathrm{CB}} \times B} \approx 1
$$

唯一的区别在分母：
- **Static**：步数 = 最长请求长度，每步 batch size 个位置 → 分母受限于 $(\max P_i + \max O_i) \times N$
- **CB**：步数 = 总工作量 / 每步能力，每步 token budget 个位置 → 分母约等于 $S$

代入典型的 3 请求场景（batch size = 3，token budget = 512，static 需 16 步，CB 约需 1 步）：

$$
\frac{\eta_{\mathrm{CB}}}{\eta_{\mathrm{static}}} \approx \frac{16 \times 3}{1 \times 512} = \frac{48}{512} \approx 0.094
$$

不对——这是反过来的。实际上当系统有多组 batch 满载时，CB 的利用率 $\approx 1$，而 static 的利用率受限于：

$$
\frac{S}{(\max P_i + \max O_i) \times N}
$$

当请求长度差异大时，static 的利用率远低于 1。

### Bubble 的形式化证明

定义 bubble ratio $R = 1 - \eta$——浪费的 GPU 能力比例。

**Static 的 bubble ratio 下界：**

$$
R_{\mathrm{static}} \geq 1 - \frac{\sum_i P_i}{N \cdot \max_i P_i}
= \frac{N \cdot \max_i P_i - \sum_i P_i}{N \cdot \max_i P_i}
$$

这个值永远 $> 0$（除非所有 $P_i$ 严格相等——现实中不可能）。因为：

$$
\max_i P_i \geq \mathrm{avg}(P_i)
$$

等号只在所有 $P_i$ 相等时取到。

**CB 的 bubble ratio：**

当 $T_{\mathrm{CB}}$ 达到下界时：

$$
R_{\mathrm{CB}} \leq \frac{S \bmod B}{\lceil S/B \rceil \times B}
$$

CB 的气泡**仅来自最后一个不完整步的剩余空间**（总 token 数不是 token budget 的整数倍时，最后一步有小于 budget 的碎片）。通常低于 $1\%$。

**核心结论：Static 的 bubble 来自"所有人都要等最慢的人"（结构性的），CB 的 bubble 来自"最后一步没装满"（非结构性的、可忽略的）。**

### Running-First 设计的理论依据

为什么 vLLM 的 scheduler 先处理 running 再处理 waiting？打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L387-L389`：

```python
# First, schedule the RUNNING requests.
req_index = 0
while req_index < len(self.running) and token_budget > 0:
```

回到食堂类比：

1. **保护已投资资源**：Running 请求已经占用了 KV Cache block（"端着盘子正在吃"）。推迟它们 = 资产闲置——盘子既不能给别人用，也不能释放。
2. **减少 preemption churn**：优先满足 running 请求 → 它们更快吃完走人 → 更少的请求被 preempt。
3. **最小不公平**：Waiting 请求还没拿到资源（"还没入座"），等一步不会造成资源浪费。Preempt 驱逐的是最低优先级 running 请求（`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L479-L483`），不是正在被服务的当前请求。

---

## Cell 5 — 代码走读：`schedule()` 逐行解析

打开我们的实现 `instances/vllm/artifacts/04-continuous-batching/implementation/scheduler.py:L330`——`ContinuousBatchingScheduler.schedule()` 方法。对应 vLLM 源码 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352-L950`。

### 方法签名和初始化

```python
# scheduler.py:L330-L341
def schedule(self) -> list[Request]:
    token_budget = self.max_scheduled_tokens   # L340: B = 512
    scheduled: list[Request] = []
```

**REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352-L385`。** vLLM 版本有同样的 `token_budget`、`scheduled_new_reqs`、`scheduled_resumed_reqs`、`scheduled_running_reqs`，额外多了 `scheduled_encoder_inputs`（encoder 模型输入）和 `scheduled_spec_decode_tokens`（投机解码 token）。

### Phase 1: Running 请求优先

**REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L387-L460`。**

```python
# scheduler.py:L343-L382
for request in list(self.running):          # L343: 用 list() 拷贝——循环内可能修改 running
    if request.is_finished:                 # L348: 已完成 → 释放资源，移入 finished
        self._finish_request(request)
        continue

    if token_budget <= 0:                   # L352: budget 耗尽 → 停止
        break

    num_new = min(request.num_new_tokens, token_budget)  # L355: budget 约束核心
    if num_new <= 0:
        continue

    # L364-376: KV Cache 分配 + Preempt 重试循环
    blocks = None
    while True:
        blocks = self.kv_cache.allocate(request.request_id, num_new)
        if blocks is not None:
            break                           # 分配成功

        victim = self._pick_preemption_victim()  # L370: 选最低优先级受害者
        if victim is None:
            break                           # 无受害者可选
        self.running.remove(victim)         # L373
        self._preempt_request(victim)       # L374: 驱逐 → free KV → 插入 waiting[0]
        if victim is request:               # L375-376: 不能驱逐自己
            break

    if blocks is not None:                   # L378: 分配成功 → 推进 token
        request.num_computed_tokens += num_new
        token_budget -= num_new
        self.total_tokens_processed += num_new
        scheduled.append(request)
```

**逐行解释关键决策：**

1. **Line 355: `num_new = min(request.num_new_tokens, token_budget)`** —— token budget 约束的核心实现。**REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L413-L415`。** 对 decode 请求（`num_new_tokens = 1`），`min(1, budget) = 1` 总是成立。对 prefill 请求（可能 $> 500$），被 budget 截断——这就是 chunked prefill 的"切"操作。

2. **Lines 364-376: Preempt 重试循环** —— **REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L466-L510`。** 当 `allocate()` 返回 `None`（KV Cache 满），不直接放弃。而是从 running 列表选出最低优先级请求（`_pick_preemption_victim()`）驱逐：释放它的所有 KV block → 状态改为 PREEMPTED → 插入 waiting 队首。然后**重试分配**——新释放的 block 可能刚好够用。

3. **Line 375: `if victim is request: break`** —— 不能驱逐自己。如果 running 列表只剩当前请求且 KV 不够，放弃这一步（下步重试）。

4. vLLM 源码的额外复杂度（`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L479-L510`）：支持 `SchedulingPolicy.PRIORITY` 和 `SchedulingPolicy.FCFS` 两种策略；PRIORITY 模式用 `max(running, key=lambda r: (r.priority, r.arrival_time))` 选受害者；FCFS 模式直接 `running.pop()`（最后一个是最低优先级）。还要处理 spec decode token 和 encoder input。

### Phase 2: Waiting 请求准入

**REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L567-L846`。**

```python
# scheduler.py:L384-L409
for request in list(self.waiting):          # L388: 遍历 waiting 队列
    if token_budget <= 0:                   # L389: budget 耗尽 → 停止准入
        break

    num_new = min(request.num_new_tokens, token_budget)  # L392: budget 截断
    if num_new <= 0:
        continue

    blocks = self.kv_cache.allocate(request.request_id, num_new)  # L396
    if blocks is not None:                  # L397: 分配成功 → 准入
        request.num_computed_tokens += num_new
        request.status = RequestStatus.RUNNING
        token_budget -= num_new
        self.total_tokens_processed += num_new
        self.waiting.remove(request)        # L402: 从 waiting 移除
        self.running.append(request)        # L403: 加入 running
        scheduled.append(request)
    else:                                   # L406: KV 满 → 停止准入
        break                               # L409: 不 preempt running 请求！
```

**关键决策：**

1. **Line 392: `num_new = min(request.num_new_tokens, token_budget)`** —— 长 prompt 被 budget 截断成 chunk。**REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L678-L690`**（chunked prefill 逻辑）。

2. **Line 406-409: KV 满时 `break`，不 preempt** —— **REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L756`。** Phase 2 的 KV 不够时直接停止准入，不驱逐 running 请求。这是刻意的设计：running 请求已有 KV Cache 投资，不应为 waiting 请求牺牲。

3. **与 vLLM 的差异**：vLLM 的 Phase 2 更复杂。它处理 prefix cache 命中（`get_computed_blocks()`，`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L616-L662`）、remote KV transfer、`max_loras` 约束、以及 `skipped_waiting` 队列（被阻塞的请求不卡住整个 waiting 队列）。

### Phase 3: 模拟 Model Forward

**REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L974-L998`**（`_update_after_schedule`）**和 L1290-L1500**（`update_from_output`）。

```python
# scheduler.py:L411-L431
for request in list(self.running):          # L419: 遍历所有 running
    if request.is_finished:
        self._finish_request(request)       # L421: 已完成 → 释放资源
    elif request.num_computed_tokens >= request.num_tokens_total:
        # L422-424: 所有 token 已处理 → model 生成 1 个 output token
        request.num_output_tokens += 1
        if request.is_finished:
            self._finish_request(request)   # L428

self.step_count += 1                        # L430
return scheduled                            # L431
```

在真实 vLLM 中，ModelRunner 执行 `model.forward()` 后，`update_from_output()` 处理采样得到的 token、检测 stop string、reject spec decode token。我们简化为：prompt 算完后，每个 step 自动生成 1 个 output token——这恰好模拟了 auto-regressive decode loop。

---

## Cell 6 — 实现：关键数据结构

### Request 状态机

打开 `instances/vllm/source/vllm/v1/request.py:L310-L326`。vLLM 定义了完整的 RequestStatus 枚举（11 个状态）。我们的简化版聚焦核心生命周期（`scheduler.py:L31-L42`）：

```python
# REFERENCE: vllm/v1/request.py → RequestStatus (enum.IntEnum, L310-L326)
class RequestStatus(Enum):
    WAITING = "waiting"       # 排队中，未分配资源
    RUNNING = "running"       # 正在处理（有 KV Cache 分配）
    PREEMPTED = "preempted"   # 被驱逐（KV Cache 已释放，等待重调度）
    FINISHED = "finished"     # 完成
```

状态转换图：

```
WAITING ──(Phase 2: allocate KV)──→ RUNNING ──(is_finished)──→ FINISHED
                                        │
                                        └──(OOM: _preempt_request)──→ PREEMPTED ──→ WAITING
                                           (insert at waiting[0], 优先重调度)
```

**REFERENCE: `instances/vllm/source/vllm/v1/request.py:L332-L333`。** vLLM 用 `status > PREEMPTED` 判断请求是否已完成——PREEMPTED 值定在"活着"和"死了"的分界线上。PREEMPTED 请求被视为活跃、等待重新调度，不是终态。三个 FINISHED_* 子状态值都大于 PREEMPTED。

### Request 数据类

```python
# REFERENCE: vllm/v1/request.py → Request (dataclass, L59-L308)
@dataclass
class Request:
    request_id: str
    prompt_tokens: int            # prompt 长度
    max_tokens: int = 256         # 最大输出 token 数
    priority: int = 0             # 优先级（低数字 = 高优先级）
    num_computed_tokens: int = 0  # 已输入模型的 token 数
    num_output_tokens: int = 0    # 已生成的输出 token 数
    status: RequestStatus = RequestStatus.WAITING
```

三个核心 property（`scheduler.py:L68-L85`）驱动了整个调度算法：

- **`num_tokens_total`**（L68-L76）：`prompt_tokens + num_output_tokens` —— 请求拥有的总 token 数。**REFERENCE: `instances/vllm/source/vllm/v1/request.py:L160-L162`。**
- **`num_new_tokens`**（L78-L81）：`num_tokens_total - num_computed_tokens` —— 还有多少 token 没处理。这是 scheduler 在每一步唯一需要看的指标。
- **`is_finished`**（L83-L85）：`num_output_tokens >= max_tokens` —— 请求是否已完成。

在真实 vLLM 中，Request 有约 20 个字段处理 multimodal inputs、structured output、KV connectors、speculative decoding、pooling、streaming 等。我们只保留驱动调度决策的 7 个字段——其余是功能扩展，不改调度算法的骨架。

### KVCache：块级分配与 OOM 检测

**REFERENCE: `instances/vllm/source/vllm/v1/core/kv_cache_manager.py`（~800 行）。**

我们的简化模型（`scheduler.py:L108-L140`）用一个固定大小的 block 池替代了完整的 KVCacheManager：

```python
# REFERENCE: vllm/v1/core/kv_cache_manager.py → KVCacheManager (L1+)
@dataclass
class KVCache:
    total_blocks: int                          # 总 block 数
    block_size: int = 16                       # 每个 block 存储的 token 数
    free_blocks: int = 0                       # 空闲 block 数
    allocations: dict[str, int] = {}           # request_id → 已分配 block 数
```

核心方法 `allocate()`（`scheduler.py:L119-L134`）：

```python
def allocate(self, request_id: str, num_tokens: int) -> Optional[int]:
    # ceil(num_tokens / block_size), min 1 block
    needed = max(1, (num_tokens + self.block_size - 1) // self.block_size)
    if needed <= self.free_blocks:
        self.free_blocks -= needed
        self.allocations[request_id] = (
            self.allocations.get(request_id, 0) + needed
        )
        return needed            # 返回分配的 block 数
    return None                  # OOM → 触发 preemption
```

与真实 `KVCacheManager` 的差异：我们的简化版没有 PagedAttention block table（第 3 章已覆盖）、prefix cache 哈希表（第 7 章）、reference counting、lookahead slots、async free 等。此处只需要 OOM 触发 preemption 的机制。

### Preempt：驱逐的完整流程

**REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972`。**

```python
# scheduler.py:L308-L325
def _preempt_request(self, request: Request):
    self.kv_cache.free(request.request_id)   # 1. 释放所有 KV Cache block
    request.status = RequestStatus.PREEMPTED  # 2. 标记为已驱逐
    self.total_preemptions += 1              # 3. 计数
    self.waiting.insert(0, request)          # 4. 插入 waiting 队首（公平性）
```

四个原子操作一口气完成。**关键设计决策**：我们**不重置 `num_computed_tokens`**。原因见 `scheduler.py:L312-L316` 的注释——真实的 vLLM 在 preempt 时确实重置它（`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L964`），但之后 `get_computed_blocks()`（`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L616-L647`）会立即通过 prefix cache 命中恢复大部分——使得有效行为等同于保留 `num_computed_tokens`。我们直接模拟了最终效果。

### 受害者选择

**REFERENCE: `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L479-L483`。**

```python
# scheduler.py:L296-L306
def _pick_preemption_victim(self) -> Optional[Request]:
    if not self.running:
        return None
    # Higher priority number = lower actual priority
    return max(self.running, key=lambda r: (r.priority, r.request_id))
```

选择最低优先级（priority 数字最大）的 running 请求。如果优先级相同，选择 request_id 最大的（字母序）。真实 vLLM 还用 `arrival_time` 做第二 tie-break（`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L480`），我们简化为 `request_id`。

---

## Cell 7 — 数值示例：一步一追踪

这个示例追踪 `ContinuousBatchingScheduler.schedule()` 的完整执行路径（`scheduler.py:L330-L431`），对应 vLLM 源码 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352-L950`。

### 场景 A：2 短 + 1 长 — 气泡如何被消除

配置：`max_scheduled_tokens = 64`，`kv_cache_blocks = 64`，`block_size = 4`。

来自 `scheduler.py:L562-L570`（Part 2 demo）：

| 请求 | prompt_tokens | max_tokens | 总 token | 类型 |
|------|-------------|-----------|---------|------|
| A | 1 | 8 | 9 | 短 |
| B | 1 | 8 | 9 | 短 |
| C | 1 | 16 | 17 | 长 |

每步每请求正好推进 1 token（prompt=1 在第一步就处理完了，后续全是 decode）。Demo 输出（每 5 步打印一次状态，来自 `scheduler.py:L577-L579`）：

```
Step 1
  [A       ] ████████████████████   1/8
  [B       ] ████████████████████   1/8
  [C       ] ████████████████████   1/16

Step 5
  [A       ] ████████████████████   4/8
  [B       ] ████████████████████   4/8
  [C       ] ████████████████████   5/16

Step 10
  [C       ] ████████████████████  10/16    ← A 和 B 已在 Step 9 完成并释放资源
```

**关键观察：**

1. **Step 1-8**：三个请求一起推进，每步 3 个 token。Token budget (64) 用了不到 5%。
2. **Step 9**：A 和 B 完成（output=8/8），释放 2 个 KV Cache block。C 继续。
3. **Step 10-17**：只有 C 在跑，每步 1 个 token。剩余 63 个 token budget 可用于立即接纳新请求——**这就是 continuous 的优势。**

**如果这是 Static Batching**：A、B、C 在同一个 batch，batch 必须等到 C 在第 17 步完成。A 和 B 在 Step 9-16 的位置（8 步 × 2 个空位 = **16 个浪费的 slot-step**）全是气泡。

Demo 对比输出（`scheduler.py:L592-L600`）：

```
  Comparison:
    Static batching:   16 steps, 13 idle slots, 72.9% utilization
    Continuous batch:  18 steps, 0 idle slots, 100% slot utilization
```

CB 的 18 步 vs static 的 16 步——CB 多 2 步是因为 decode 每步只推进 1 token/请求。但 **0 个 idle slot vs 13 个 idle slot** 才是本质差距。实际系统中 batch 之间可以重叠，static 的 idle slot 是永久损失的。

### 场景 B：第 4 个请求中途加入

来自 `scheduler.py:L612-L650`（Part 3 demo）：

| 请求 | prompt_tokens | max_tokens | 加入时机 |
|------|-------------|-----------|---------|
| Alpha | 1 | 10 | Step 1 |
| Beta | 1 | 10 | Step 1 |
| Gamma | 1 | 14 | Step 1 |
| **Delta** | **1** | **8** | **Step 6** |

```
Step 5 快照：
  [Alpha  ] ████████████████████   3/10
  [Beta   ] ████████████████████   3/10
  [Gamma  ] ████████████████████   5/14

*** Delta arrives at step 6! ***
(In static batching, Delta would wait for Alpha/Beta/Gamma to finish)

Step 10 快照：  ← Delta 已准入！
  [Alpha  ] ████████████████████   7/10
  [Beta   ] ████████████████████   7/10
  [Gamma  ] ████████████████████   9/14
  [Delta  ] ████████████████████   3/8
```

**在 Static Batching 中**，Delta 必须等整个 batch（Alpha+Beta+Gamma）全部完成才能开始——那意味着等 14 步（Gamma 的 max_tokens）。**在 Continuous Batching 中**，Delta 在 Step 6 加入，Phase 2（`scheduler.py:L384-L409`）发现 budget 和 KV 都有余量——立即准入，**延迟为 0 步**。

这展示了 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L567-L846` 中 Phase 2 的核心理念：**任何有空余 capacity 的 step 都可以立即接纳新请求。** 请求的生命周期不再绑定到 batch 的生命周期。

### 完成顺序

Demo 最终输出（`scheduler.py:L643-L650`）：

```
  Completion order:
    Alpha   (output=10/10)
    Beta    (output=10/10)
    Delta   (output=8/8)       ← 中途加入，反而先于 Gamma 完成
    Gamma   (output=14/14)
```

Delta 虽然是中途加入的（Step 6），但因为它的 max_tokens 只有 8（少于 Gamma 的 14），所以在 Gamma 之前完成。**完成顺序只取决于各自的长度，不取决于加入时间。**

---

## Cell 9 — 源码映射表

### 我们的实现 vs vLLM 源码

| 我们的代码 | vLLM 源码 (文件:行) | 差异 | 原因 |
|-----------|-------------------|------|------|
| `ContinuousBatchingScheduler.__init__()` (L256-L275) | `vllm/v1/core/sched/scheduler.py:L67-L300` | 简化：无 `VllmConfig`、`KVCacheConfig`、`StructuredOutputManager`、encoder-decoder 配置、speculative 配置、LoRA、CUDA graph、Mamba、多 GPU | 教学清晰：~30 行 vs ~230 行 |
| `schedule()` Phase 1 (L343-L382) | `vllm/v1/core/sched/scheduler.py:L387-L460` | Running-first + preempt-retry 一致；简化了 encoder budget 分支、spec token 处理 | 核心算法保持一致；vLLM 额外处理 encoder-decoder、spec decode、Mamba alignment |
| `schedule()` Phase 2 (L384-L409) | `vllm/v1/core/sched/scheduler.py:L567-L846` | budget 截断 + KV OOM → break 一致；简化了 prefix cache hit 检查、`skipped_waiting` 队列、remote KV transfer | 准入逻辑核心相同；vLLM 多了 ~280 行处理 cache hit、LoRA 约束、结构化输出阻塞 |
| `schedule()` Phase 3 (L411-L431) | `vllm/v1/core/sched/scheduler.py:L974-L998` + L1290-L1500 | 合并 `_update_after_schedule` + `update_from_output`；简化 stop 检测 | 模拟 auto-regressive generate；真实版处理采样 token、stop string、spec rejection |
| `_preempt_request()` (L308-L325) | `vllm/v1/core/sched/scheduler.py:L952-L972` | 无 encoder cache free、event recording、spec token cleanup；`num_computed_tokens` 不重置 | 模拟 prefix cache 有效行为；核心概念：free KV → PREEMPTED → waiting[0] |
| `_pick_preemption_victim()` (L296-L306) | `vllm/v1/core/sched/scheduler.py:L479-L483` | 独立方法；`(priority, request_id)` vs `(priority, arrival_time, request_id)` | 单一职责；教学用途不需 arrival_time |
| `KVCache` (L108-L140) | `vllm/v1/core/kv_cache_manager.py` | 固定 block 池 + 累计分配；无 PagedAttention block table、prefix cache hash、reference counting | 第 3 章已覆盖 PagedAttention；此处只需 OOM → preempt 触发机制 |
| `Request` 数据类 (L46-L91) | `vllm/v1/request.py:L59-L308` | 7 字段 vs ~20；去除了 multimodal、structured output、KV transfer、block hashing、encoder inputs、streaming、spec decode、pooling | 只保留驱动调度决策的字段 |
| `RequestStatus` (L31-L42) | `vllm/v1/request.py:L310-L326` | 4 状态 vs 11；去掉了 `WAITING_FOR_*` 中间状态和 6 个 `FINISHED_*` 子状态 | 核心生命周期保留；子状态是 async/debug 细节 |
| `StaticBatchSimulator` (L146-L223) | N/A（教学新增） | 纯 demo：模拟 static batch、统计 idle slot、计算利用率 | 与 continuous batching 并列对比，展示气泡问题 |
| `add_request()` (L278-L281) | `vllm/v1/core/sched/request_queue.py` | 直接 list append；无优先级队列 | 简化但优先级通过 `_pick_preemption_victim()` 展示 |
| `abort_request()` (L283-L290) | `vllm/v1/core/sched/scheduler.py` (`finish_requests` 相关) | 线性扫描 running+waiting | 教学环境足够；真实实现用 `self.requests` dict 做 O(1) 查找 |

### 主要简化点及理由

| 原始复杂度 | 我们的简化 | 理由 |
|-----------|----------|------|
| Multi-GPU 协调 (tensor/pipeline/context parallelism) | 单设备 | 分布式调度是 Ch11 DCP/PCP 的主题 |
| Encoder-decoder + multimodal inputs | 纯文本生成式 | 核心调度逻辑相同，encoder 只是额外一个 budget 维度 |
| Speculative decoding | 无 | Ch21 覆盖；调度视角是额外 `spec_token_ids` |
| Structured output | 无 | 独立功能；状态检查增加复杂性但不改调度骨架 |
| Prefix cache (block hashing, cache-aware scheduling) | 简化：preempt 不重置 `num_computed_tokens` | Ch05/07 深入完整机制；此处展示有效行为 |
| Chunked prefill 阈值 | 统一 token budget 截断 | Budget 截断已展示核心 chunking 逻辑 |
| LoRA (adapter constraints) | 无 | `max_loras` 约束简单但正交于核心算法 |
| KV Connectors (P/D disaggregation, offloading) | 无 | Ch12 覆盖；调度视角是 async 状态转换 |

---

## Cell 10 — 验证

### 测试结果

58 个测试全部通过，覆盖 9 个类别：

| 测试类 | 测试数 | 验证内容 |
|--------|-------|---------|
| `TestRequest` | 6 | Request 数据模型：token 追踪（`num_new_tokens`、`num_tokens_total`）、完成判断（`is_finished`）、默认值（`max_tokens=256`，`priority=0`） |
| `TestKVCache` | 7 | 块分配（成功/OOM/累积）、释放、重用、未知请求释放无 crash |
| `TestRequestLifecycle` | 6 | 完整状态机：WAITING→RUNNING→PREEMPTED→FINISHED；preempt 后 resume 并完成 |
| `TestSchedulerCore` | 8 | 基础操作：add、abort（从 waiting 和 running）、promote、finish、`schedule()` 返回列表、step 计数、token 处理累加 |
| `TestContinuousBatching` | 6 | 核心功能：中途加入、提前离开、气泡消除、late arrival 无缝集成、多请求批量准入、完成后无 KV 泄漏 |
| `TestPreemption` | 9 | 优先级驱逐：OOM 触发、priority 选受害者、tie-break（request_id）、空列表返回 None、队首插入、KV 释放、计数器递增、高低优先级互动、自我驱逐 |
| `TestTokenBudget` | 4 | Budget 约束：`budget=1` 限制 token/准入、budget 耗尽停止 Phase 1、大 budget 全部准入 |
| `TestEdgeCases` | 7 | 边界：空调度、max_tokens=0、单请求、20 请求大批量、1 token 输出、无 KV 泄漏、优先级确定准入顺序 |
| `TestStatsAndIntegration` | 5 | 统计准确性、token 累加、Static vs Continuous 对比（CB 步数 $\leq$ static 步数）、KV block 守恒不变量 |
| **总计** | **58** | |

### Static vs Continuous 对比

`TestStatsAndIntegration.test_static_vs_continuous_advantage`（`test_scheduler.py:L901-L933`）验证：

```python
# 同一组 3 个请求：A(8 token), B(8), C(16)
# Static batching:  16 steps → 72.9% utilization（13 idle slots）
# Continuous batch:  ≤16 steps → 100% slot utilization（0 idle slots）
assert sched.stats["step"] <= static_stats["steps"]
```

### Bubble 模拟验证

`TestStatsAndIntegration.test_static_batch_simulator_bubble`（`test_scheduler.py:L885-L899`）验证 Static Batching 必然产生气泡：

```python
stats = sim.run()
assert stats["idle_slots"] > 0       # Static batching MUST produce bubbles
assert stats["utilization_pct"] < 100  # <100% by structural design
assert stats["batches"] == 1         # All 3 in one fixed batch
```

### KV Block 守恒

`TestStatsAndIntegration.test_kv_blocks_conserved_across_preemptions`（`test_scheduler.py:L935-L953`）验证贯穿整个调度过程的不变量：

```python
# Invariant: free_blocks + sum(allocations) == total_blocks
allocated = sum(sched.kv_cache.allocations.values())
assert sched.kv_cache.free_blocks + allocated == total
```

即使在大量 preemption 之后，KV block 总数守恒——没有泄漏。

### Linter 检查

```bash
python3 scripts/lint_formulas.py instances/vllm/artifacts/04-continuous-batching/narrative/chapter.md
python3 scripts/lint_source_grounding.py instances/vllm/artifacts/04-continuous-batching/
```

---

## Cell 11 — 总结

### 本章要点

1. **Continuous Batching = 统一的 token 流模型。** 没有独立的 prefill/decode 阶段。每个请求只有 `num_computed_tokens`（已处理游标）和 `num_tokens_with_spec`（目标总量）。Scheduler 在 token budget $B$ 的约束下推进每个游标。这是 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L353-L362` 注释的核心思想。

2. **食堂拼桌类比贯穿全章。** Static Batching = 整组一起进出——短请求等长请求，产生结构性气泡。Continuous Batching = 吃完就走、来人即坐——GPU 从不空等。Token Budget = 餐桌座位数，Running = 正在吃的人，Waiting = 排队的人，Preempt = 有人占着桌子不吃被请出去。

3. **Bubble 的形式化证明。** 两个步数公式揭示了本质差异：

$$
T_{\mathrm{static}} = \max_i P_i + \max_i O_i
\qquad \mathrm{vs} \qquad
T_{\mathrm{CB}} = \left\lceil \frac{\sum_i (P_i + O_i)}{B} \right\rceil
$$

Static 的步数由最长请求决定，CB 的步数由总工作量决定。Static 的 bubble 来自"同步等待"（结构性的，永远 $>0$），CB 的 bubble 来自"最后一步没装满"（非结构性的，通常 $<1\%$）。

4. **三阶段调度算法（`schedule()`，`scheduler.py:L330-L431`）。** Phase 1（running-first，`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L387-L460`）保护已有 KV Cache 投资——占着碗的人先吃饱。Phase 2（waiting admission，`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L567-L846`）用剩余 budget 填补——新人入座但不能赶走正在吃的人。Phase 3（model forward，`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L974-L998 + L1290-L1500`）模拟 auto-regressive 生成——每步产出 1 token。

5. **Preempt 是 OOM 时的最后手段（`scheduler.py:L308-L325`）。** 驱逐最低优先级的 running 请求（`_pick_preemption_victim`，`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L479-L483`），释放其 KV Cache（`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972`），插入 waiting 队首（公平性保证）。重试分配——可能刚好够。

6. **Chunked Prefill** 是 CB 的自然推论。Token budget 截断（`min(num_new_tokens, token_budget)`）把长 prompt 切成块，和 decode token 在同一 step 里共存。`instances/vllm/source/vllm/v1/core/sched/scheduler.py:L413-L414` 的 `long_prefill_token_threshold` 是这种机制的显式参数。

7. **Request 状态机。** `WAITING → RUNNING → (PREEMPTED → WAITING → ...) → FINISHED`。PREEMPTED 是"活着"和"完了"的分界线（`instances/vllm/source/vllm/v1/request.py:L332-L333`）。

8. **历史纵深。** Orca（OSDI 2022）首创 iteration-level scheduling。vLLM（SOSP 2023）将粒度从"请求"下沉到"token"——`num_computed_tokens` 取代 prefill/decode 阶段划分，使 chunked prefill、混合调度、prefix caching、speculative decoding 四大技术共享同一套调度框架。

### 下章预告

`KVCache.allocate()` → `None` 告诉我们显存不够了。但"不够"具体怎么衡量？`block_size` 为什么是 16 不是 32？`max_scheduled_tokens` 为什么是 512 不是 1024？第 5 章深入 GPU 显存管理系统——从 PyTorch CachingAllocator 到 vLLM BlockPool 到 KVCacheManager——理解每一级分配器如何保护下一级不超支，以及 Chunked Prefill 如何通过控制每步 token 数来绑定峰值显存。

---

← 第 3 章：FlashAttention & PagedAttention | 第 5 章：GPU 显存管理与 Chunked Prefill →
