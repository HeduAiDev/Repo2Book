# 第5章：Chunked Prefill — 长 Prompt 的切片调度

> 打开 `vllm/v1/core/sched/scheduler.py:678`。三行代码定义了整个 chunked prefill 机制：
> ```python
> if 0 < threshold < num_new_tokens:
>     num_new_tokens = threshold
> ```
> 一个 128K token 的 prompt 被切成 64 个 2K chunk，**和其他请求的 decode token 交叉调度**。
> 长 prompt 的 TTFT 只多了不到 5%，短请求几乎不被阻塞——这就是 chunked prefill 的全部价值。

---

## 这章要做什么？

第 4 章讲了 continuous batching 的 scheduling 循环——running 优先、waiting 填剩余 budget。但有一个问题悬而未决：**如果一个 waiting 请求的 prompt 是 128K token，token budget 只有 2048——怎么调度？**

不做 chunked prefill：这个请求的 prefill 放不下 token budget → 它一直被跳过 → **starvation**。

做 chunked prefill：prompt 被切成 64 个 chunk，每个 step 处理 2K token。长 prompt 和其他请求的 decode 交替执行。

学完这章你能：
- 打开 `scheduler.py:678` 理解 chunked prefill 的三个决策点
- 运行 `implementation/chunked_prefill.py` 看到调度 trace
- 量化 TTFT vs throughput trade-off——长 prompt 涨多少，短请求降多少
- 解释 `scheduler_reserve_full_isl` 为什么是安全阀

---

## 5.1 三个决策点

### Source Trail

打开 `vllm/config/scheduler.py:70-84`，chunked prefill 的全部配置：

```python
enable_chunked_prefill: bool = True       # L84 — 主开关
long_prefill_token_threshold: int = 0     # L80 — 0=disabled; >0=每步上限
max_num_scheduled_tokens: int = 2048      # token budget 总量
scheduler_reserve_full_isl: bool = True   # L140 — 准入检查
```

在 `scheduler.py` 中有三个决策点使用这些参数：

**决策点 1 — Running 请求（`scheduler.py:L413-L415`）：**

```python
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold
```

一个已经在 running 的请求如果还有大量未计算的 prefill token——限制它这次只推进 threshold 个。这保护了 token budget 不被单个请求垄断。

**决策点 2 — Waiting 请求（`scheduler.py:L678-L680`）：**

```python
threshold = long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
```

新请求的 prompt 超过阈值——切成 chunk，只 admission threshold 个 token。

**决策点 3 — Chunked prefill guard（`scheduler.py:L684-L690`）：**

```python
if not enable_chunked_prefill and num_new_tokens > token_budget:
    break  # 不做 chunk → 整个 prompt 放不下 → 拒绝 admission
```

如果 chunked prefill 关了，prompt 超过 token budget → 这个请求永远进不来。如果开了，prompt 会被 cap 到 budget。

这三个决策点分别保护三种资源：token budget、延迟公平性、availability。

---

## 5.2 TTFT vs Throughput

（调度分析对应 `scheduler.py:L413-L415, L678-L692` 的 chunked prefill 逻辑。）

### Theory: 量化

用一个具体例子。1 个 128K token 的 prompt + 8 个 128 token 的 prompt，每个输出 256 token，token budget = 2048。

**不做 chunked prefill：**

```
Step 1-63: 长 prompt prefill (128K/2K = 64 steps)
           8 个短请求在 waiting 中空等
Step 64+:  所有请求 decode
Total: ~320 steps
短请求 TTFT = 63 steps  ← 等了 63 步才开始
```

**做 chunked prefill（threshold=2048）：**

```
Step 1:   长 chunk(2048) + 8 短 prefill(1024) → budget 接近用完
Step 2:   长 chunk(2048) + 8 decode(8) → budget 充足
...
Step 64:  长 chunk(最后一批) + 8 decode(8)
Step 65+: 所有请求 decode
Total: ~320 steps
短请求 TTFT = 1 step  ← 第一步就被 admit！
```

**核心观察：** 总步数几乎一样（~320），但短请求的 TTFT 从 63 步降到 1 步——**降低了 63×**。长请求的 TTFT 从 63 步涨到 64 步——**涨了不到 2%**。

`★ Insight ─────────────────────────────────────`
Chunked prefill 是一个近乎 Pareto-optimal 的优化：它大幅改善短请求延迟而对长请求延迟影响微乎其微。原因：长请求的 prefill 本身是 O(seq²) compute——切成 chunk 后每个 chunk 是 O(chunk²)，总 compute 几乎不变（Σ chunk² ≈ seq² 因为交叉项被消除）。Compute 没增加，只是分布更均匀。
`─────────────────────────────────────────────────`

### 形式化分析

设 $N$ 个请求，prompt 长度 $P_1,...,P_N$，输出长度 $O_1,...,O_N$，token budget 为 $B$。

Static batching 步数下界：

$$
\mathrm{Steps}_{\mathrm{static}} \geq \max_i P_i + \max_i O_i
$$

利用率：

$$
\eta_{\mathrm{static}} \leq \frac{\sum P_i + \sum O_i}{(\max P_i + \max O_i) \cdot B}
$$

当 prompt 长度差异大时，$\eta_{\mathrm{static}} \ll 1$。

以上面例子：$\max P_i = 128000$, $\sum P_i = 128000 + 8\times128 = 129024$。

$$
\eta_{\mathrm{static}} \leq \frac{129024 + 8\times256}{(128000+256) \times 8} \approx 12.6\%
$$

Continuous batching + chunked prefill 步数下界：

$$
\mathrm{Steps}_{\mathrm{CB}} \geq \left\lceil \frac{\sum P_i + \sum O_i}{B} \right\rceil
$$

利用率 $\eta_{\mathrm{CB}} \approx 1$（忽略无法整除的余数）。

**结论：** Continuous batching + chunked prefill 的优势是渐近性的——随 prompt 长度分布方差增大，$\eta_{\mathrm{static}} \to 0$（长 prompt 支配 prefill 阶段），而 $\eta_{\mathrm{CB}} \approx 1$。

---

## 5.3 代码走读 / Code Walkthrough

运行我们的实现：

```bash
python3 implementation/chunked_prefill.py
```

输出：

```
Step 1: 1 reqs, 2048 tokens (long: 2048)
Step 2: 1 reqs, 2048 tokens (long: 2048)
Step 3: 1 reqs, 2048 tokens (long: 2048)
Step 4: 3 reqs, 2048 tokens (long: 1856, short: 192)
Step 5: 4 reqs, 194 tokens (long: 1, short: 193)

After 5 steps: long computed 8001/8000 tokens
Short requests: started decoding while long is still prefill
```

Steps 1-3：长 prompt 独占 budget（2048×3=6144 tokens prefill）。Step 4：长 prompt 只剩 1856 tokens（< budget 2048），budget 余量被短请求填满。这就是 chunked prefill 在 action——长 prompt 分块处理，短请求不用等。

### 核心调度循环

```python
# implementation/chunked_prefill.py:L62-L90
# 对应 vLLM: scheduler.py:L388-L846
def schedule(self):
    budget = self.max_tokens
    scheduled = {}

    # ── Phase 1: Running requests ──
    # REFERENCE: scheduler.py:L413-L415
    for req in self.running[:]:                       # ← [:] 拷贝遍历（避免 preempt 改列表）
        n = req.num_new_tokens
        if self.long_threshold > 0 and req.is_prefilling:
            n = min(n, self.long_threshold)           # ← L413: cap long prefill
        n = min(n, budget)
        if n > 0:
            scheduled[req.request_id] = n
            budget -= n

    # ── Phase 2: Waiting requests ──
    # REFERENCE: scheduler.py:L678-L692
    while self.waiting and budget > 0:
        req = self.waiting[0]
        n = req.num_new_tokens
        if self.long_threshold > 0:
            n = min(n, self.long_threshold)           # ← L678: cap new prompts
        if not self.enable_chunked and n > budget:     # ← L684: guard
            break
        n = min(n, budget)
        if n > 0:
            scheduled[req.request_id] = n
            budget -= n
            self.waiting.pop(0)
            self.running.append(req)
```

**关键细节：** Phase 1 中 `self.running[:]` 创建拷贝来遍历——因为 vLLM 的真实 scheduler 可能在循环内 preempt 请求（修改 `self.running`），拷贝避免迭代器失效。

### 与 vLLM 官方实现的差异

| 我们的实现 | vLLM 源码 | 差异原因 |
|---|---|---|
| `SimRequest` 5 字段 | `Request` 30+ 字段（`vllm/v1/request.py`） | 教学简化 |
| `ChunkedPrefillScheduler` ~80 行 | `Scheduler` 900+ 行（`scheduler.py`） | 省略 KV Cache allocation、preemption、encoder |
| `schedule()` 手动单步 | 每步自动循环 | 让读者看到每步决策 |
| 无 `allocate_slots()` | `scheduler.py:L744` KV Cache 分配 | Ch2/Ch6 独立讲解 |
| 无 `is_prefill_chunk` 标记 | `scheduler.py:L988` | 简化：通过 `num_computed_tokens` 判断 |

---

## 5.4 scheduler_reserve_full_isl 与 discard_request_mask

### Source Trail

`scheduler.py:L753` — admission 前检查整个 prompt 能否放进 KV Cache：

```python
full_sequence_must_fit = (
    scheduler_reserve_full_isl
    and request.num_computed_tokens == 0
)
```

不做这个检查的后果：Scheduler 持续 admit 新请求的 chunk。KV Cache 逐渐填满——每个请求只完成了 prompt 的一部分。当 KV Cache 满了，所有正在进行的 chunked prefill 都无法继续。

`vllm/v1/worker/gpu_model_runner.py:L1928-L1933` — 非最终 chunk 的 token 是 dummy：

```python
self.discard_request_mask[:num_reqs] = (
    optimistic_seq_lens_cpu[:num_reqs] < num_tokens_np
)
```

对于 prompt 还没全部处理完的请求，sampled token 被丢弃——generator offset 在采样后回滚。

---

## 验证

```bash
cd artifacts/05-chunked-prefill && python3 implementation/chunked_prefill.py
```

---

## 总结

- **三个决策点保护三种资源：** token budget（L413）、延迟公平（L678）、availability（L684）
- **Chunked prefill 是近乎免费的长请求优化：** 总 compute 不变，TTFT trade-off 极度倾斜
- **`scheduler_reserve_full_isl` 防止 over-admission：** 在 admit 第一个 chunk 前验证整个 prompt

---

← 第4章 | 第6章 →
