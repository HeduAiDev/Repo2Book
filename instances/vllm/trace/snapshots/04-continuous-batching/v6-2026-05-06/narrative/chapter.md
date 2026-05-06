# 第4章：Continuous Batching — 让 GPU 不再"等最慢的人"

> 本章涉及的 vLLM 源码：
> - `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L67-L998`（Scheduler 主体）
> - `instances/vllm/source/vllm/v1/core/sched/output.py:L181-L233`（SchedulerOutput）
> - `instances/vllm/source/vllm/v1/core/sched/request_queue.py:L75-L128`（FCFSRequestQueue）
> - `instances/vllm/source/vllm/v1/request.py:L59-L333`（Request、RequestStatus）
> - `instances/vllm/source/vllm/v1/core/kv_cache_manager.py`（call sites only）
>
> 本章源码 commit：`98661fe`。
>
> **第 3 章告诉你 attention 怎么算，KV cache 怎么存。但有了一个能算、能存的引擎以后，谁来决定"这一步算谁、算多少 token"？这就是 scheduler 的工作。它是 vLLM 推理引擎里最不像炫技、却最决定整机吞吐的那段代码。**

---

## 这章要讲什么？

打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L353` —— 这是 vLLM 主作者之一 Woosuk Kwon 留下的注释，全章最关键的一句话就在这里：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has num_computed_tokens and num_tokens_with_spec.
# ... At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

这段话翻译成大白话：**scheduler 不区分 prefill 和 decode**。每个请求都只是一个 token 流，有"已经算到第几个"（`num_computed_tokens`）、"目标到第几个"（`num_tokens`）两个指针。每一步，scheduler 在 token budget 的预算内，把这些指针往前推。仅此而已。

这一个视角的转变，就是 vLLM 比上一代框架快 2-23 倍的核心来源。本章拆解它。

学完这章你能：

- 用一张图说清楚 static batching 为什么会留下大片 GPU bubble，并写出 bubble ratio 的公式。
- 不看源码也能讲出 `schedule()` 的两阶段结构：running-first（保护已有 KV 投资）→ waiting（用剩余 budget 接纳新请求）。
- 解释 preemption 为什么"踢队尾"，为什么 OOM 时会把刚扣的 budget 还回来，以及为什么发生过 preempt 的步要直接跳过 Phase 2。
- 把 chunked-prefill 开关 ON / OFF 的语义讲清楚（`continue` vs `break`）。
- 跟着 19 步的 demo 输出走一遍，自己复现 `r1: 400-token prompt` 是怎么被切成 4 段塞进 budget=128 的。

---

## 4.1 问题：static batching 的 bubble 是结构性的

### 4.1.1 食堂打饭：传统模式 vs 自助餐

想象你和两个朋友 r1、r2、r3 去吃饭。三个人点的菜量差异巨大：

| 食客 | prompt 长度 $P_i$ | 输出长度 $O_i$ |
|------|-----:|-----:|
| r1 | 400 | 4 |
| r2 | 64 | 8 |
| r3 | 16 | 16 |

**传统食堂（static batching）的规矩是"所有人必须一起行动"**：
1. 大家一起打 prompt 阶段——必须等最慢的人打完，最长的 r1 要 400 步。
2. 然后大家一起 decode——一步推一个 token，直到 r1 的输出 4 个 token 全部生成完。

注意 r3 在 prompt 阶段第 16 步就打完自己那份了，但他必须站在那里等 r1 打到第 400 步——干等了 384 步。这 384 步里，他占着的那一份 GPU "灶台"（batch 里的一个 slot）什么也没做。

这就是 **bubble**：GPU 在等同步点，算力被浪费。

**自助餐厅（continuous batching）改了规矩**：每人拿一个固定大小的盘子（`max_num_scheduled_tokens` = 128 个 token 的预算）。盘子里想装什么菜随便装——刚开始你装 prompt，后来你装 decode token；想装 r1 的 prompt 就装 r1，盘子还有空就接着装 r2、r3 的——盘子满了才结账。

结果会差多少？运行 `python3 -m instances.vllm.artifacts.04-continuous-batching.implementation.demo` 的最后一段输出告诉我们：

```
Bubble analysis (same 3-request workload):
  static     : 416 steps
  continuous : 20 steps
  speedup    : 20.80x
```

**20.8 倍**。同样的硬件、同样的工作量、只是"调度方式"换了一种。

### 4.1.2 形式化：bubble ratio

把上面的故事写成公式。`instances/vllm/artifacts/04-continuous-batching/implementation/scheduler.py:L309-L323` 的 `static_batching_steps()` 给出的就是 static batching 的总步数：

$$
T_{\mathrm{static}} = \max_i P_i + \max_i O_i
$$

代入 demo workload：

$$
T_{\mathrm{static}} = 400 + 16 = 416
$$

与运行结果一致。

所有人都被锁定到 batch 里 $T_{\mathrm{static}}$ 步。每一步 GPU 处理 $N$ 个 token 位置（每个请求一个 slot）。但实际有用的 token 总数只有 $\sum_i (P_i + O_i)$。利用率：

$$
\eta_{\mathrm{static}} = \frac{\sum_i (P_i + O_i)}{(\max_i P_i + \max_i O_i) \cdot N}
$$

代入 demo：分子是

$$
(400+4) + (64+8) + (16+16) = 508
$$

分母是 $416 \cdot 3 = 1248$，所以

$$
\eta_{\mathrm{static}} = \frac{508}{1248} \approx 40.7\%
$$

**接近 60% 的 GPU 算力在空转。** 而且这不是工程问题，是结构问题——只要请求长度不齐，bubble 就一定存在。

continuous batching 的步数下界则只看总量：

$$
T_{\mathrm{CB}} \geq \left\lceil \frac{\sum_i (P_i + O_i)}{B} \right\rceil
$$

其中 $B$ 是 token budget。代入 demo（$B=128$，$\sum_i (P_i+O_i)=508$）：

$$
T_{\mathrm{CB}} \geq \lceil 508 / 128 \rceil = 4
$$

但 demo 实际跑了 19 步——为什么没到下界？

因为单步 decode 阶段只能每个 active 请求贡献 1 个 token，凑不满 128 的 budget。后面我们会看到，真实场景里 decode 远多于 prefill，所以 batch size $N$ 而不是 budget $B$ 才是 decode 阶段的瓶颈。但即使如此，CB 比 static 还是快了 20 倍——因为 static 的"最长 prompt 决定一切"实在太亏了。

### 4.1.3 为什么 vLLM 选择 token 粒度而不是请求粒度？

continuous batching 这个想法不是 vLLM 首创。OSDI 2022 的 **Orca**（Yu et al.）首先提出 **iteration-level scheduling**：在每次 forward 迭代时重新决定 batch 成员，而不是固定 batch 跑到底。Orca 的进步已经很大——request 级粒度。

vLLM（SOSP 2023）做的事情比 Orca 又下沉一层：**把粒度从请求降到 token**。`scheduler.py:L353` 的注释强调了这个范式——"There's no decoding phase nor prefill phase"。一旦只看 token，prefill 和 decode 就可以在同一步里混着跑，chunked prefill、prefix caching、speculative decoding 全部从这同一个 loop 里自然长出来。

这就是为什么本章的所有讨论都围着两个变量打转——`num_computed_tokens` 和 `num_tokens`。理解了这两个游标，scheduler 的所有行为都是它们的推论。

---

## 4.2 理论：token budget、running-first、preempt-tail

打开 `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L352` —— 这是 `Scheduler.schedule()` 的入口。整段函数大约 600 行，但骨架就是两个 phase。我们的简化实现把这个骨架完整保留下来：`instances/vllm/artifacts/04-continuous-batching/implementation/scheduler.py:L76-L218`。

### 4.2.1 三个不变量

每一次 `schedule()` 调用结束时必须满足：

$$
\sum_{r \in \mathrm{scheduled}} n_r \le B \qquad |\,\mathrm{running}\,| \le N_{\max} \qquad B - \sum_r n_r \ge 0
$$

其中 $B$ 是 token budget（`max_num_scheduled_tokens`，本章 demo 用 128），$N_{\max}$ 是同时 running 请求的上限（`max_num_running_reqs`，本章 demo 用 4），$n_r$ 是这一步分配给请求 $r$ 的 token 数。

vLLM 在 `scheduler.py:L848-L853` 用三个 `assert` 守住这三个不变量，我们在 `implementation/scheduler.py:L208-L210` 一字不差地照搬：

```python
total = sum(num_scheduled_tokens.values())
assert total <= self.max_num_scheduled_tokens
assert token_budget >= 0
assert len(self.running) <= self.max_num_running_reqs
```

这三行不是装饰——它们是把"调度的正确性"机械化检查的最后一道闸门。

### 4.2.2 Running-first 的理由

把 `schedule()` 拆成两个 phase 的**根本理由是 KV cache 投资保护**。

已经在 `running` 列表里的请求，意味着它们之前已经分配过 KV cache block——这些 block 装着"已经算过的 K、V 张量"。如果你这一步把它们晾在一边、转去 admit 一个新请求，那已经付出去的 KV 投资就被冻结在那儿、占着显存什么都不干。

更糟的是：要给新请求腾地方，你迟早要 preempt 一个 running 的——而 preempt 的代价是把它的 KV 全部丢掉、`num_computed_tokens` 重置为 0、下一次重新算。**preempt 等于把已经付过的 GPU 计算白白扔掉。** 所以 vLLM 的策略很朴素：**先让正在跑的人跑下去，剩多少 budget 再说接纳新人的事**。

形式化一点：设第 $k$ 步开始时 running 列表中的请求集合为 $R_k$，waiting 队列为 $W_k$，本步给 $R_k$ 中请求 $r$ 分配的 token 数为 $n_r$。vLLM 的策略是：先消耗 budget 满足 $R_k$，然后用余量

$$
B - \sum_{r \in R_k} n_r
$$

去 admit $W_k$ 队首的请求。如果 Phase 1 已经在 running 列表内部触发了 preempt（说明 KV 已经吃紧），那 Phase 2 整个跳过——下一节细说。

### 4.2.3 Preempt-tail：FCFS 下"踢队尾"是最小不公平选择

Phase 1 里如果一个 running 请求要新 KV block 但 `allocate_slots` 返回 `None`（OOM），怎么办？vLLM 的策略：

1. 从 `self.running` **弹出最后一个**（队尾），调用 `_preempt_request` 把它的 KV 释放掉、`num_computed_tokens` 清零、`prepend` 到 waiting 队首。
2. 把这个被踢出的请求"已经在本步预订的 token 数"还回 budget——因为它没机会跑了。
3. 重新尝试给原请求分配 block。如果还不够，继续踢下一个。
4. 极端情形：踢到最后只剩自己。这时 vLLM 把"自己"也 preempt 掉、整步退出 Phase 1。

为什么是**队尾**？因为 `self.running` 是按 FCFS（First-Come-First-Served）顺序排的——队首是最早进来的、优先级最高的请求。**踢队尾 = 踢最晚到达的请求**，受影响的人最少（他享受 GPU 时间最短，丢失的 KV 投资也最少）。

为什么要**还 budget**？因为 budget 是"本步的 token 预算"。如果一个被 preempt 的请求已经在前面的循环迭代里扣了 30 个 token 的 budget，它现在被踢出去这 30 个就没用上——不还回来就是浪费。这是 `scheduler.py:L485-L502` 的设计，我们的实现一字不差地复刻在 `scheduler.py:L131-L132`。

为什么 Phase 1 触发 preempt 后 **要直接跳过 Phase 2**？设想：一个 running 请求 OOM 了，我们刚踢掉一个队尾、释放了 N 个 block。如果 Phase 2 立刻拿走这 N 个 block 给新请求，那刚被踢的请求下一步还得排队等——浪费两次 KV 重算。所以 vLLM 在 `scheduler.py:L568` 写下 `if not preempted_reqs:`——出过 preempt 的步把释放出的 block 留给被踢的请求下一步重生。我们的实现：`implementation/scheduler.py:L159` `if not preempted_req_ids:`。

### 4.2.4 Chunked prefill：长 prompt 的"切片入场"

Phase 2 把 waiting 队首的请求 admit 进 running 时，如果它的 prompt 比剩余 budget 还长怎么办？两种语义：

- **chunked prefill ON**（默认）：把这一步能塞下的 token 数 `min(num_new_tokens, token_budget)` 切下来，让请求带着"半截 prefill"进 running。下一步它仍然 prefill，再切一段。这样 budget 不会浪费，prefill 长的请求不会饿死短的。
- **chunked prefill OFF**：直接 `break`——长 prompt 装不下就停，绝不切。这保留了严格的 FCFS 语义（不会因为长 prompt 装不下就跳过它去服务后面的短 prompt），代价是 budget 可能用不满、GPU 空转。

vLLM 的默认是 ON（`vllm/config/scheduler.py:L84` `enable_chunked_prefill: bool = True`），原因显然——空转太亏。我们的实现把这个开关也挂在 `Scheduler.__init__`（`implementation/scheduler.py:L45`）。

注意一个细节：Phase 1 的循环遇到 `num_new_tokens == 0` 时用 `continue`（跳过这个请求继续往下找），Phase 2 用 `break`（停止整个 admit 循环）。`scheduler.py:L446-L462` 的注释解释了 Phase 1 的取舍：*"by doing continue instead of break, we do not strictly follow the FCFS scheduling policy"*——Phase 1 容忍小幅破坏 FCFS 来榨取 GPU 利用率。Phase 2 则保留严格 FCFS。这两处的语义差异是面试题级别的细节。

---

## 4.3 走读 schedule()：从 token_budget 到 SchedulerOutput

打开 `instances/vllm/artifacts/04-continuous-batching/implementation/scheduler.py:L76` —— `Scheduler.schedule()`。我把它分成三段过：开场、Phase 1、Phase 2。

### 4.3.1 开场：初始化本步状态

```python
# implementation/scheduler.py:L76-L89
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens
    num_scheduled_tokens: dict[str, int] = {}
    preempted_req_ids: set[str] = set()
    newly_running_req_ids: list[str] = []

    self.kv_cache_manager.new_step_starts()
```

四个本步状态：
- `token_budget`：每次重置成 $B$，本步可分配的总 token 上限。
- `num_scheduled_tokens`：`req_id -> num`，本步给每个请求分了多少 token。
- `preempted_req_ids`：本步发生 preempt 的请求集合（决定 Phase 2 跳不跳）。
- `newly_running_req_ids`：本步从 waiting 升进 running 的新请求。

`kv_cache_manager.new_step_starts()` 在我们的实现里是个 no-op（`implementation/kv_cache_manager.py:L34`）；vLLM 的 `KVCacheManager.new_step_starts()` 在这里清空"本步 prefix-cache 命中统计"，详见第 12-13 章。

### 4.3.2 Phase 1：FCFS 遍历 running，OOM 时 preempt 队尾

```python
# implementation/scheduler.py:L93-L151
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # 计算这一步该给它多少 token
    num_new_tokens = request.num_new_tokens                    # 1
    if 0 < self.long_prefill_token_threshold < num_new_tokens: # 2
        num_new_tokens = self.long_prefill_token_threshold
    num_new_tokens = min(num_new_tokens, token_budget)         # 3

    if num_new_tokens == 0:
        req_index += 1
        continue

    # 试着分配 KV block；OOM 时 preempt 队尾再重试
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
    while new_blocks is None:
        preempted_req = self.running.pop()                     # 4
        if preempted_req is request:                           # 5
            self._preempt_request(preempted_req)
            preempted_req_ids.add(preempted_req.request_id)
            new_blocks = None
            break

        self._preempt_request(preempted_req)
        preempted_req_ids.add(preempted_req.request_id)
        # 把被踢请求已扣的 budget 还回来                            # 6
        returned = num_scheduled_tokens.pop(preempted_req.request_id, 0)
        token_budget += returned
        # 重算并重试
        num_new_tokens = min(request.num_new_tokens, token_budget)
        if 0 < self.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.long_prefill_token_threshold
        new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)

    if new_blocks is None:
        break                                                  # 7

    request.block_ids.extend(new_blocks)                       # 8
    num_scheduled_tokens[request.request_id] = num_new_tokens
    token_budget -= num_new_tokens
    req_index += 1
```

逐点解释：

1. **`request.num_new_tokens`**（`implementation/request.py:L91-L98`）= `num_tokens - num_computed_tokens`。这是请求"还欠模型多少 token 没算"。decode 阶段它通常是 1（每步生成 1 个）；prefill 阶段它可能是 400（剩余的 prompt token）。
2. **`long_prefill_token_threshold`**：超长 prompt 单步上限。0 表示不开启，否则即便 budget 充裕也限制单步推进，避免一个超长 prefill 把别人挤出去。`scheduler.py:L413-L414` 同名变量。
3. **`min(num_new_tokens, token_budget)`**：本步剩余 budget 是硬天花板。
4. **`self.running.pop()`**：FCFS 下队尾就是最低优先级，pop 出来的就是要 preempt 的人。
5. **被踢的恰好是自己**：发生在 running 只剩自己一个、却仍然 OOM 的极端情形——把自己 preempt 掉、走人。`scheduler.py:L508-L510` 的同款 corner case。
6. **归还 budget**：被踢的请求如果在前一轮循环已经扣过 budget，pop 出来还回 token_budget。这一步对应 vLLM `scheduler.py:L485-L502`。
7. **第 7 行的 `break`**："我已经把别人都踢光了还是不够，那本步就到此为止"。
8. **成功分配**：把新分到的 block 接到 `request.block_ids` 末尾，记账。

`_preempt_request` 长这样：

```python
# implementation/scheduler.py:L221-L241
def _preempt_request(self, request: Request) -> None:
    assert request.status == RequestStatus.RUNNING
    self.kv_cache_manager.free(request)             # 释放所有 KV block
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0                 # KV 没了，已算的进度也作废
    request.num_preemptions += 1                    # 计数器
    self.waiting.prepend_request(request)           # 插到 waiting 队首
```

注意三件事：
- **caller pop 契约**：`_preempt_request` **不**从 `self.running` 移除请求——调用方在 L117 的 `self.running.pop()` 已经做了。如果你"手贱"在这里加 `self.running.remove(request)`，会拿到 `ValueError: x not in list`。这个契约是 vLLM `scheduler.py:L955-L956` 注释明确写下的，我们的实现里也用注释强调了一遍（`implementation/scheduler.py:L223-L227`）。
- **prepend 而不是 append**：被踢的请求要在 waiting 队首，下一步第一个被重新考虑。我们的 `FCFSRequestQueue.prepend_request`（`implementation/request_queue.py:L45-L46`）是 `_dq.appendleft()`，对应 `vllm/v1/core/sched/request_queue.py:L92-L94`。
- **num_computed_tokens 归零**：KV 全释放了，下次重新调度时必须重新算 prefill。这是 preempt 真正的代价。vLLM 用 `num_preemptions` 计数器观测这个代价（`scheduler.py:L967`）。

### 4.3.3 Phase 2：FCFS 遍历 waiting，无 preempt-on-OOM

```python
# implementation/scheduler.py:L153-L204
if not preempted_req_ids:
    while (
        self.waiting
        and token_budget > 0
        and len(self.running) < self.max_num_running_reqs
    ):
        request = self.waiting.peek_request()

        num_new_tokens = request.num_new_tokens
        if 0 < self.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.long_prefill_token_threshold

        # chunked-prefill 开关
        if (
            not self.enable_chunked_prefill
            and num_new_tokens > token_budget
        ):
            break                                              # 1

        num_new_tokens = min(num_new_tokens, token_budget)
        if num_new_tokens <= 0:
            self.waiting.pop_request()
            continue

        new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
        if new_blocks is None:
            break                                              # 2

        # 提交
        self.waiting.pop_request()
        request.block_ids = list(new_blocks)
        request.status = RequestStatus.RUNNING
        self.running.append(request)
        num_scheduled_tokens[request.request_id] = num_new_tokens
        newly_running_req_ids.append(request.request_id)
        token_budget -= num_new_tokens
```

两个 `break` 是 Phase 2 与 Phase 1 在风格上的关键差异：

1. **chunked-prefill OFF 时遇到长 prompt → break**（不切、也不跳过后面的人）。FCFS 严格保证：先来的 prompt 必须先 admit，即使它装不下。
2. **OOM → break**（不 preempt running）。Phase 2 没有 preempt-on-OOM 的逻辑——running 的请求是 KV 投资的"既得利益者"，绝不为 waiting 让路。

最后是"提交"块：`peek_request()` 之前只是看了一眼队首，真正确定能 admit 才 `pop_request()` 把它从 waiting 拿走、`append` 到 running、状态改 `RUNNING`、记账。`status = RequestStatus.RUNNING` 这一行本质就是"从排队到落座"。

### 4.3.4 收尾：组装 SchedulerOutput

```python
# implementation/scheduler.py:L206-L218
total = sum(num_scheduled_tokens.values())
assert total <= self.max_num_scheduled_tokens
assert token_budget >= 0
assert len(self.running) <= self.max_num_running_reqs

return SchedulerOutput(
    num_scheduled_tokens=num_scheduled_tokens,
    total_num_scheduled_tokens=total,
    preempted_req_ids=preempted_req_ids,
    newly_running_req_ids=newly_running_req_ids,
    finished_req_ids=set(self.finished_req_ids),
)
```

三个 `assert` 是 4.2.1 的三个不变量。`SchedulerOutput` 字段名 `num_scheduled_tokens` 和 vLLM 字段一字不差（`vllm/v1/core/sched/output.py:L191-L193`），调用方代码可以照抄。

⚠️ 这里有个**简化遗留物**：`finished_req_ids`。vLLM 在跨步之间保留这个集合（`scheduler.py:L919-L923`），这样 step N 的 model runner 能拿到"step N-1 完成的请求 id"去清理 KV state。我们的简化版把"上报完成"和"清空集合"合并到了 `update_after_step` 里，于是 `SchedulerOutput.finished_req_ids` 在我们的实现中**永远是空集**——这没影响，因为 Ch04 没有 model runner，权威的"完成判断"是 `request.is_finished()`。**第 20 章引入 ModelRunner 时会修复这点**——把 `clear()` 移到 `schedule()` 开头（"上一步的 finished id 这一步消费完即清"）。

---

## 4.4 数据结构走读：Request、SchedulerOutput、KV manager

scheduler 的代码是骨架，骨架挂着的肌肉是几个数据结构。

### 4.4.1 Request：四个属性是 scheduler 的全部输入

打开 `instances/vllm/artifacts/04-continuous-batching/implementation/request.py:L41-L107`：

```python
@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    arrival_time: float

    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    num_preemptions: int = 0

    @property
    def num_new_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens
```

scheduler 全部决策只依赖这几个量。注意 `is_prefill` 是**派生**的：scheduler 不需要存"我现在在 prefill 还是 decode"这个状态——它是 `num_computed_tokens < num_prompt_tokens` 算出来的。这就是 4.1.3 那句"there's no decoding phase nor prefill phase"在数据结构层面的体现。

`RequestStatus` 是个 IntEnum，定义在同文件 L17-L36：

```python
class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED
```

5 个状态对应生命周期：

```
WAITING → RUNNING → (PREEMPTED → WAITING)* → FINISHED_*
```

`PREEMPTED` 是"活着但要重排"——通过 `> PREEMPTED` 判断 finished 这一招让代码非常紧凑：FINISHED_STOPPED、FINISHED_LENGTH_CAPPED 自动满足 finished 检查。vLLM 真实代码里有 11 个状态（多了 4 个 `WAITING_FOR_*` 和 `FINISHED_ABORTED` 等），但在 Ch04 的核心 loop 里只会出现 5 个。

### 4.4.2 SchedulerOutput：5 个字段的精简版

`implementation/output.py:L17-L44` 完整列出我们保留的字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `num_scheduled_tokens` | `dict[str, int]` | 本步每个请求分到多少 token |
| `total_num_scheduled_tokens` | `int` | 上面字典的求和（缓存供 assert 用） |
| `finished_req_ids` | `set[str]` | 已完成的请求 id（本实现下永远空，Ch20 修） |
| `preempted_req_ids` | `set[str]` | 本步发生 preempt 的请求 id |
| `newly_running_req_ids` | `list[str]` | 本步从 waiting 升进 running 的请求 id |

vLLM 真实字段有 14 个（多 `scheduled_new_reqs`、`scheduled_cached_reqs`、`scheduled_spec_decode_tokens`、`scheduled_encoder_inputs`、`num_common_prefix_blocks`、`free_encoder_mm_hashes`、`has_structured_output_requests`、`kv_connector_metadata` ……）每一个都对应一个独立特性：spec decode、encoder/multimodal、prefix cache 公共前缀、KV connector 远程传输、structured output 词表 mask。Ch12 之后会一项项展开。

### 4.4.3 SimpleKVCacheManager：把 KV 复杂性留到 Ch12-13

scheduler 只通过 4 个方法和 KV manager 打交道：`new_step_starts`、`allocate_slots`、`free`、`get_computed_blocks`（后者是 prefix cache，本章未用）。Ch04 的 `SimpleKVCacheManager`（`implementation/kv_cache_manager.py`）只是一个 free-list：

```python
# implementation/kv_cache_manager.py:L39-L55
def allocate_slots(self, request, num_new_tokens):
    blocks_needed = (num_new_tokens + self.block_size - 1) // self.block_size
    if blocks_needed > len(self._free_blocks):
        return None
    allocated = self._free_blocks[:blocks_needed]
    del self._free_blocks[:blocks_needed]
    return allocated

def free(self, request):
    self._free_blocks.extend(request.block_ids)
    request.block_ids = []
```

`(num_new_tokens + block_size - 1) // block_size` 是"向上取整除"——把要算的 token 数换算成需要的 block 数（`block_size = 16` 时，分 17 个 token 要 2 个 block）。OOM 直接返回 `None`——这就是 scheduler 那 `while new_blocks is None:` 循环的触发条件。

vLLM 真实 `KVCacheManager` 是 600+ 行的 orchestrator：BlockPool、prefix-cache 哈希表、copy-on-write 共享 block、多 group attention spec……以及 `allocate_slots` 真实签名带 7 个 kwargs（`num_new_computed_tokens`, `new_computed_blocks`, `num_lookahead_tokens`, `num_external_computed_tokens`, `delay_cache_blocks`, `num_encoder_tokens`, `full_sequence_must_fit`）。Ch12-13 完整覆盖。这里我们只关心"能不能要到 N 个 block"。

### 4.4.4 FCFSRequestQueue：deque 的薄包装

`implementation/request_queue.py:L18-L72` 的实现就一句话——拿 `collections.deque` 包了五个 vLLM 同名方法（`add_request`、`pop_request`、`peek_request`、`prepend_request`、`prepend_requests`）。

vLLM 自己的 `FCFSRequestQueue` 直接继承 `deque[Request]`（`vllm/v1/core/sched/request_queue.py:L75-L128`）；我们用 composition 而不是 inheritance，纯属代码清洁度偏好——method 接口完全一致。

vLLM 还有一个 `PriorityRequestQueue`（heap-backed）实现 priority scheduling，本章不用。

---

## 4.5 数值演示：跟着 19 步 demo 一行一行走

理论看完，跑一下真东西。`instances/vllm/artifacts/04-continuous-batching/implementation/demo.py` 配置：

```python
# implementation/demo.py:L52-L63
sched = Scheduler(
    max_num_scheduled_tokens=128,   # B
    max_num_running_reqs=4,
    num_gpu_blocks=200,
    block_size=16,
    enable_chunked_prefill=True,
)
sched.add_request(Request("1", list(range(400)), max_tokens=4, ...))  # 长 prompt
sched.add_request(Request("2", list(range(64)),  max_tokens=8, ...))  # 中 prompt
sched.add_request(Request("3", list(range(16)),  max_tokens=16, ...)) # 短 prompt
```

运行：

```bash
python3 -m instances.vllm.artifacts.04-continuous-batching.implementation.demo
```

输出（节选关键步）：

```
── step 1 ────────────────────────────────────────────
  budget=128  running=1  waiting=2  free_blocks=192
    r1:  128 tok (prefill)  computed=128/400  blocks=8
    ADMITTED:  ['1']

── step 2 ────────────────────────────────────────────
  budget=128  running=1  waiting=2  free_blocks=184
    r1:  128 tok (prefill)  computed=256/400  blocks=16

── step 3 ────────────────────────────────────────────
  budget=128  running=1  waiting=2  free_blocks=176
    r1:  128 tok (prefill)  computed=384/400  blocks=24

── step 4 ────────────────────────────────────────────
  budget=128  running=3  waiting=0  free_blocks=170
    r1:   16 tok (decode )  computed=400/401  blocks=25
    r2:   64 tok (decode )  computed=64/65  blocks=4
    r3:   16 tok (decode )  computed=16/17  blocks=1
    ADMITTED:  ['2', '3']

── step 7 ────────────────────────────────────────────
  budget=128  running=2  waiting=0  free_blocks=189
    r1:    1 tok (decode )  computed=403/404  blocks=0
    r2:    1 tok (decode )  computed=67/68  blocks=7
    r3:    1 tok (decode )  computed=19/20  blocks=4

── step 19 ────────────────────────────────────────────
  budget=128  running=0  waiting=0  free_blocks=200
    r3:    1 tok (decode )  computed=31/32  blocks=0

================================================================
Finished requests: ['1', '2', '3']
KV blocks reclaimed: 200/200
Bubble analysis: static=416, continuous=20, speedup=20.80x
```

逐步重放（关键决策标在右边）：

**Step 1**：waiting=[r1, r2, r3]，running=[]，budget=128。
- Phase 1 空。
- Phase 2：peek r1。`num_new_tokens = 400`，被 budget 截到 128（**chunked prefill 第一刀**）。需要 $\lceil 128/16 \rceil = 8$ 个 block。OK，admit。
- 步末：r1 进 running，computed=128/400，still prefill。

**Step 2-3**：r1 一直在 prefill。每步切 128 token。step 3 末 computed=384/400，仍欠 16 个 prompt token。

**Step 4**（关键步）：
- Phase 1：r1.num_new_tokens = 16（剩下的 prompt token），扣 budget 到 112。block 数 1，OK。
- Phase 2：budget 还有 112。peek r2（prompt 64）：`min(64, 112)=64`，要 4 个 block，admit。budget 剩 48。peek r3（prompt 16）：`min(16, 48)=16`，1 个 block，admit。budget 剩 32（这步用不上了）。
- 注意 r1 的输出标 `(decode)` —— 因为 `update_after_step` 把 num_computed_tokens 加到 400 之后再判断 `is_prefill`，此时 400 < 400 已不成立，所以打印时已是 decode 状态；但本步**实际算的是它最后 16 个 prompt token**。这个标签反映的是更新**后**的状态，是观察上的细节，不是 bug。
- ADMITTED 列出 ['2', '3'] — 两个新请求同步进入 running。

**Step 5-7**：三个请求都进入纯 decode，每步每个 +1 token。
- Step 5：r1 computed=401/402、r2 computed=65/66、r3 computed=17/18。注意**目标 num_tokens 也每步 +1**——因为 `update_after_step` 在 prefill 完成后会 append 一个 placeholder token 到 `output_token_ids`（`implementation/scheduler.py:L265-L267`），所以 `num_tokens = num_prompt_tokens + num_output_tokens` 每步增长 1。
- Step 7：r1 完成第 4 个 output token，触发 `FINISHED_LENGTH_CAPPED`（`implementation/scheduler.py:L271-L273`），`_free_request` 释放它的全部 26 个 block。这就是为什么 step 7 显示 `r1: blocks=0` 且 `free_blocks` 从 step 6 的 164 跳到 189（多了 25 个 block；少的那个是 r2 因为 decode 增长又申请了 1 个）。

**Step 8-19**：r2、r3 继续 decode 直到各自 max_tokens 用尽。step 11 r2 完成、free_blocks 跳；step 19 r3 完成、free_blocks=200/200，全部回收。

**bubble analysis**：static=416 步、continuous=20 步、20.8 倍。和 4.1.1 公式预期一致。

**值得注意的细节**：

- **prefill+decode 不会出现在同一步**——本 demo 里。因为 r1 从 step 1 一直独占 budget=128 直到它的 prefill 切完（step 4 的最后 16 token），等到 budget 有剩才 admit r2、r3；admit 时 r1 已经过了 prefill。这是 budget=128、prompt=400 这组特定参数下的现象，不是 CB 本身的限制。把 budget 调到 256 或 prompt 改成 200 你就能看到 r1 prefill chunk + r2 decode token 同步出现在一步里——这就是真实生产环境里 CB 让 GPU 几乎不闲着的样子。
- **decode 阶段每步只用 budget 的 3/128 ≈ 2.3%**：因为我们只有 3 个 active 请求。生产环境 batch 内通常有几十到几百个并发请求，decode 阶段才会真正吃满 budget。
- **没有发生 preempt**：200 个 block × 16 token/block = 3200 token 容量，远大于这三个请求的总 KV 需求。下一节用一个针对性 demo 看 preempt 长什么样。

---

## 4.6 Preempt 触发的条件长什么样

`tests/test_scheduler.py` 里有专门的 preempt 用例。最小的 preempt-trigger 配置是这样的：

```python
sched = Scheduler(
    max_num_scheduled_tokens=64,
    max_num_running_reqs=4,
    num_gpu_blocks=2,            # 极小 KV 池
    block_size=16,
)
sched.add_request(Request("A", list(range(16)), max_tokens=20, ...))  # prompt 16 → 1 block
sched.add_request(Request("B", list(range(16)), max_tokens=20, ...))  # prompt 16 → 1 block
```

两个 prompt 各占 1 block，一开始两个都能 admit 进 running——0 free block。

进入 decode 后，每个请求的 `num_computed_tokens` 会爬到 16（block 边界）以上。下一次 `allocate_slots(req, 1)` 要 `ceil(1/16)=1` 个新 block——但池子里 0 个 free block。`allocate_slots` 返回 `None` → 触发 preempt。

按 FCFS 队尾规则，**B**（后入队的）会被踢：`self.running.pop()` 返回 B，B 的 1 个 block 释放回池、status = PREEMPTED、num_computed_tokens=0、prepend 到 waiting 队首。下一次 `allocate_slots(A, 1)` 拿到刚释放的那个 block，A 继续 decode；B 这一步丢了。

这一步发生 preempt，`preempted_req_ids = {"B"}`，**Phase 2 跳过**（即使 waiting 队首此时就是 B 自己，也不在这一步 admit 它）——理由就是 4.2.3 那个"释放出的 block 留给 B 下一步重生"的设计。

下一步开始，B 在 waiting 队首，Phase 2 把它重新 admit、从头跑 prefill（`num_computed_tokens` 已归零）。这就是 preempt 的真实代价：**B 之前算过的 prefill 和 decode 全部白做了**，得重来。`B.num_preemptions` 加 1，留给监控告警用。

`tests/test_scheduler.py` 的 `test_preempt_triggers_on_oom`、`test_preempted_request_resumes`、`test_preempt_increments_counter`、`test_preempt_skips_phase_2` 把上述路径的每个边角都断言了一遍——感兴趣可以打开看看。

---

## 4.7 我们的实现与 vLLM 源码：1:1 对照表

| 我们的代码 | vLLM 源码 | 我们改了什么 | 为什么 |
|-----------|----------|-------------|--------|
| `Scheduler.__init__` (`scheduler.py:L39-L65`) | `vllm/v1/core/sched/scheduler.py:L67-L176` | 接受裸 int 参数，不接 `VllmConfig` | demo 没有完整 config，参数语义本章足够 |
| `Scheduler.schedule` Phase 1 (`scheduler.py:L93-L151`) | `scheduler.py:L387-L556` | 去掉 encoder/spec/structured/connector 分支 | 这些是后续章节主题 |
| `Scheduler.schedule` Phase 2 (`scheduler.py:L159-L204`) | `scheduler.py:L568-L846` | 去掉 prefix-cache hit、远程 KV 传输、`skipped_waiting` 队列 | 同上 |
| `Scheduler._preempt_request` (`scheduler.py:L221-L241`) | `scheduler.py:L952-L972` | 去掉 encoder cache 释放、event log、spec clear | 同上 |
| `Scheduler.update_after_step` (`scheduler.py:L245-L279`) | `scheduler.py:L974-L998` + `L1290-L1551` | 合并 `_update_after_schedule` 与 `update_from_output`；用 placeholder 0 模拟采样 token | Ch20 引入 ModelRunner 后会拆开 |
| `Scheduler._free_request` (`scheduler.py:L282-L295`) | `scheduler.py:L1813-L1834` | 去掉 KV connector 延迟释放、encoder cache | 同上 |
| `Scheduler.add_request` (`scheduler.py:L68-L72`) | `scheduler.py:L1728-L1748` | 去掉 streaming queue 与 structured-output gating | 同上 |
| `Request` (`request.py:L41-L107`) | `vllm/v1/request.py:L59-L308` | 7 个字段而不是 30+ | scheduler 只读这几个 |
| `RequestStatus` (`request.py:L17-L36`) | `vllm/v1/request.py:L310-L333` | 5 个状态而不是 11 | 4 个 `WAITING_FOR_*` 与 `FINISHED_ABORTED` 在 Ch04 loop 中不可达 |
| `SchedulerOutput` (`output.py:L17-L44`) | `vllm/v1/core/sched/output.py:L181-L233` | 5 个字段而不是 14 | 砍掉的每个字段对应一个独立特性 |
| `FCFSRequestQueue` (`request_queue.py:L18-L72`) | `vllm/v1/core/sched/request_queue.py:L75-L128` | composition 而非继承 deque | 风格 |
| `SimpleKVCacheManager` (`kv_cache_manager.py:L24-L70`) | `vllm/v1/core/kv_cache_manager.py` | free-list 而非 BlockPool + prefix-cache | Ch12-13 完整覆盖 |
| `static_batching_steps` / `continuous_batching_steps` (`scheduler.py:L309-L351`) | — | 教学新增 | Bubble 量化对比，不在 vLLM 中 |

**故意砍掉的内容（每一项都在源码里有 NOT IMPLEMENTED 注释）**：

- `RequestStatus` 的 `WAITING_FOR_FSM`（structured output gating）、`WAITING_FOR_REMOTE_KVS`（异步 KV 传输）、`WAITING_FOR_REMOTE_REQUESTS`（streaming inputs）、`FINISHED_ABORTED`（用户取消）—— 4 个状态。
- `SchedulerOutput` 的 `scheduled_new_reqs` / `scheduled_cached_reqs`（worker 端缓存协议，Ch20-21）、`scheduled_spec_decode_tokens`（Ch26 speculative decoding）、`scheduled_encoder_inputs`（多模态，超出本书）、`num_common_prefix_blocks`（prefix cache，Ch12-13）、`kv_connector_metadata`（异步 KV 传输，Ch15）。
- `KVCacheManager` 的 prefix-cache 哈希表与 copy-on-write 共享 block（Ch12-13）。
- `Scheduler` 的 `PauseState`（引擎暂停/恢复，运维特性，超出本书）、`PriorityRequestQueue`（priority 调度，Ch06）。

每项都在 implementation 里相应位置标注了"NOT IMPLEMENTED"或"SIMPLIFIED"注释，方便你和 vLLM 源码对照阅读。

---

## 验证

### 跑测试

```bash
cd instances/vllm/artifacts/04-continuous-batching
python3 -m pytest tests/ --ignore=tests/_legacy -q
```

预期输出：

```
48 passed in 0.05s
```

48 个测试覆盖：

| 模块 | 测试数 | 验证什么 |
|------|------:|---------|
| `test_request.py` | 7 | `num_new_tokens`、`is_prefill`、`is_finished` 派生属性 |
| `test_request_queue.py` | 7 | FCFS 顺序、prepend 进队首、remove 一致性 |
| `test_kv_cache_manager.py` | 9 | block 分配/释放、OOM 返回 `None`、向上取整 |
| `test_scheduler.py` | 25 | 不变量、admission policy、chunked prefill ON/OFF、preempt 队尾、Phase 2 跳过、长 prompt cap、整集成 demo |

### 跑 lint

```bash
python3 scripts/lint_formulas.py instances/vllm/artifacts/04-continuous-batching/narrative/chapter.md
python3 scripts/lint_source_grounding.py instances/vllm/artifacts/04-continuous-batching/
```

两个都必须 PASS（`✓ All ... checks passed!`）才算章节完成。

### 跑 demo

```bash
python3 -m instances.vllm.artifacts.04-continuous-batching.implementation.demo
```

对照 4.5 节的输出。结尾应该看到 `KV blocks reclaimed: 200/200` 和 `speedup: 20.80x`。

---

## 总结

1. **continuous batching 不是一个新算法，是一个视角的转变。** 把请求看成 token 流，scheduler 只在每步把 `num_computed_tokens` 推进到 `num_tokens`——prefill 和 decode 自然落在同一个 loop 里。这是 vLLM 比 Orca 又快了一截的根源。

2. **bubble 是结构性的。** static batching 的步数

   $$
   T_{\mathrm{static}} = \max_i P_i + \max_i O_i
   $$

   由"最慢的人"决定，请求长度不齐就一定有 bubble。CB 的步数下界

   $$
   T_{\mathrm{CB}} \ge \left\lceil \sum_i (P_i + O_i) / B \right\rceil
   $$

   由总量决定，bubble 只来自最后一步的边角碎料。本章 demo 的 3 请求工作负载，speedup 是真实的 20.8 倍。

3. **`schedule()` 的两阶段结构来自一条朴素经济学**：已经分配过 KV 的请求是"沉没投资"，先让它们跑下去比让出位置给新请求 cost-effective。Phase 1 处理 running，Phase 2 用剩余 budget admit waiting；Phase 1 内一旦发生 preempt，Phase 2 整个跳过——把释放的 block 留给被踢的请求下一步重生。

4. **Preempt 是 OOM 时的最后手段。** FCFS 下踢队尾（最晚到达 = 最低优先级 = 受影响最少的人）；被踢的请求要还 budget、要清 KV、要 `num_computed_tokens` 归零、要 prepend 到 waiting 队首——下一步就会重新 admit。`num_preemptions` 计数器是观测真实 KV 压力的单点指标。

5. **chunked-prefill 的 ON / OFF 用 `continue` vs `break` 表达：** Phase 1 的 `continue` 容忍小幅破坏 FCFS 来榨利用率，Phase 2 chunked-prefill OFF 的 `break` 保留严格 FCFS。同一份代码、两种模式，是面试题级别的细节。

6. **5 个状态、5 个字段、4 个方法。** Ch04 的所有简化都是裁掉而不是改写——`RequestStatus` 留 5 / 11，`SchedulerOutput` 留 5 / 14，`KVCacheManager` 留 4 个方法。每砍一刀都对应一个后续章节的主题（prefix cache → Ch12-13；spec decode → Ch26；ModelRunner → Ch20）。

### 下章预告

第 4 章里 `allocate_slots(req, n)` 返回 `None` 是一个关键事件——它就是 KV cache 的 OOM 信号。但这个 `None` 是怎么算出来的？GPU 显存到底有几级管理？PyTorch CachingAllocator → vLLM BlockPool → KVCacheManager 之间各管一层什么？第 5 章把这条 stack trace 完整拆开，让你看见从一个 `torch.empty` 到一块 16-token KV block 的完整旅程。

---

← 第 3 章：FlashAttention & PagedAttention | 第 5 章：GPU 显存管理系统 →
