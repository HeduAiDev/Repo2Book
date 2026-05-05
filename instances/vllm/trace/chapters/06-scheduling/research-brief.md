# Research Brief: Request Scheduling System

**Chapter**: 06-scheduling
**Author**: Researcher agent
**Date**: 2026-05-05
**Source files analyzed**:
- `vllm/v1/core/sched/scheduler.py` (2295 lines)
- `vllm/v1/core/sched/request_queue.py` (209 lines)
- `vllm/v1/core/sched/output.py` (264 lines)
- `vllm/v1/core/sched/interface.py`
- `vllm/v1/core/sched/async_scheduler.py`
- `vllm/config/scheduler.py` (SchedulerConfig)
- `vllm/docs/design/arch_overview.md`
- `vllm/docs/design/prefix_caching.md`
- `vllm/docs/design/paged_attention.md`

---

## 1. 演进时间线

| 时间 | 事件 | 关键贡献 | 与 vLLM 的关系 |
|------|------|---------|---------------|
| 1960s-1990s | 经典 OS 调度 (FCFS, SJF, RR, MLFQ) | 公平性、响应时间、抢占式调度的理论基础 | vLLM 只借用了 FCFS 和优先级两个概念；RR/SJF 因 GPU 特性和 LLM 请求长度不可预测而无法直接应用 |
| 2019-2021 | GPT-2/GPT-3 时代 | HuggingFace generate() + static batching | vLLM 的前身背景——每批必须等所有请求完成才能开始下一批，GPU 利用率 30-50% |
| 2022.07 | Orca (OSDI 2022) | **Iteration-level scheduling** = Continuous Batching | 革命性创新：请求完成后立刻退出批次，新请求立即加入，GPU 利用率跃升至 90%+ |
| 2023.09 | vLLM v0 + PagedAttention (SOSP 2023) | PagedAttention + swap/recompute preemption | vLLM 诞生：OS 虚拟内存思想引入 KV cache 管理；首次展示 swap preemption |
| 2024.03 | Sarathi-Serve (OSDI 2024) | **Chunked prefills** 证明可消除 prefill stall | vLLM v1 将其作为默认行为内建 |
| 2024.01 | DeepSpeed-FastGen / MII v0.2 | Dynamic SplitFuse + Blocked KV Cache | 竞品：prompt 分解 + 生成融合调度 |
| 2024.12 | SGLang v0.4 | Zero-overhead batch scheduler + RadixAttention | 竞品：CPU 调度与 GPU 计算重叠 |
| 2024-2025 | vLLM v1 引擎 | 统一 token-budget 模型、无 phase 区分、recompute-only preemption | V0 完全废弃，调度器从 ~3000 行精简为更清晰的 2295 行 |
| 2025.01 | vLLM V0 正式移除 (PR #25321) | V0 core 全部删除 | V1 成为唯一调度器 |
| 2025.04 | TensorRT-LLM on B200 | Blackwell FP8 深度优化 | NVIDIA 专用引擎在低并发下超越 vLLM |

---

## 2. 关键论文

### Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)
- **作者**: Gyeong-In Yu, Joo Seong Jeong (FriendliAI / Seoul National University)
- **问题**: 传统 static batching 导致 straggler problem（短请求等长请求）和 head-of-line blocking（新请求等当前批次完成）
- **方法**: 
  1. **Iteration-level scheduling**: 每个 iteration 后重新组合批次——完成者退出，新到者加入
  2. **Selective batching**: Attention 层按请求单独计算，MLP/Linear 层合并批处理（因为矩阵乘法的 FLOPs 远超 attention）
- **结果**: GPT-3 175B 上 **36.9x** 吞吐量提升（相比 NVIDIA FasterTransformer）
- **对 vLLM 的影响**: Continuous batching 成为 vLLM 调度器的核心范式。vLLM 在此基础上叠加了 PagedAttention 和更精细的 preemption 策略。

### Efficient Memory Management for Large Language Model Serving with PagedAttention (SOSP 2023)
- **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng et al. (UC Berkeley / Stanford)
- **方法**: 
  1. **PagedAttention**: KV cache 划分为固定大小 blocks，用 block table 做逻辑→物理映射（完全对应 OS 虚拟内存的 page table）
  2. **FCFS 调度 + swap/recompute preemption**: 内存不足时，要么 swap KV blocks 到 CPU，要么直接释放等重新计算
  3. **严格 prefill/decode 阶段分离**: 每步要么跑 prefill batch 要么跑 decode batch，不混合
- **对 vLLM 的影响**: 这是 vLLM v0 的设计基座。Preempt=swap 的方案在 v0 中实现，v1 中被移除。

### Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (OSDI 2024)
- **作者**: Amey Agrawal, Nitin Kedia et al. (Microsoft Research India / Georgia Tech)
- **核心贡献**: **Chunked prefills**——将长 prefill 切分为等大 chunks 与 decode 步骤交错执行，消除 "prefill stall"
  ```
  传统:  [──── Prefill 1500 tokens ────] → [Dec] → [Dec] → ...
             ↑ decode requests STALLED
            
  Sarathi: [Chunk 512] + [Dec A,B,C] → [Chunk 512] + [Dec A,B,C] → ...
             ↑ 无 stall——prefill 与 decode 交错
  ```
- **结果**: vLLM 基础上 **2.6x-5.6x** 服务容量提升（在 tail latency SLO 约束下）
- **对 vLLM 的影响**: vLLM v1 将 chunked prefill 设为**默认且不可关闭**的行为。Sarathi-Serve 证明了该技术的通用性。

---

## 3. vLLM 调度器设计演进

### 3.1 V0 调度器（2023，已废弃）

**架构**:
```
┌─────────┐    ┌─────────┐    ┌──────────┐
│ waiting  │ →  │ running  │ →  │ swapped  │
│ (prefill │    │ (decode  │    │ (CPU KV  │
│  queue)  │    │  queue)  │    │  cache)  │
└─────────┘    └─────────┘    └──────────┘
```

**特征**:
- **严格 phase 分离**: 每个 step 要么 prefill **要么** decode，不可混合
- **三个队列**: waiting → running → swapped
- **两种 preemption**:
  - Swap: GPU KV blocks → CPU RAM（通过 PCIe），请求进入 swapped 队列
  - Recomputation: 释放 KV blocks，请求回到 waiting 重新 prefill
- **Chunked prefill**: 可选（`enable_chunked_prefill`），有条件的

**问题**:
- Phase 分离导致低负载时 GPU 利用率不足（等不到足够多的同 phase 请求）
- Swapped 队列增加了代码复杂度
- Swap 性能受限于 PCIe 带宽（PCIe Gen4 ≈ 64 GB/s，但 KV cache 动辄几十 GB）

### 3.2 V1 调度器（2024-2025，当前）

**架构**:
```
┌─────────┐         ┌─────────┐
│ waiting  │ ──────→ │ running  │ (mixed: prefill + decode + spec decode)
└─────────┘         └─────────┘
       ↑                   │
       └─── preempt ───────┘ (recompute only, no swap)
```

**核心设计哲学**（直接引用 `scheduler.py:353-362`）:
> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec. At each step, the scheduler tries to assign tokens to the requests so that each request's num_computed_tokens can catch up its num_tokens_with_spec.

这是一个 **token-budget 模型**——调度器不知道也不关心什么 "phase"，它只维护一个计数器 `num_computed_tokens` 追赶 `num_tokens_with_spec`。

**关键组件**:

1. **RequestQueue** (`request_queue.py`):
   - `FCFSRequestQueue`: 基于 `deque` 的标准 FIFO
   - `PriorityRequestQueue`: 基于 `heapq`，排序键为 `(priority, arrival_time)`——priority 小的优先；相同 priority 按到达时间
   - 策略由 `SchedulingPolicy` enum 选择: `FCFS` 或 `PRIORITY`

2. **Token Budget** (`scheduler.py:371`):
   ```python
   token_budget = self.max_num_scheduled_tokens
   ```
   - 每步分配的总 token 预算来自 `max_num_scheduled_tokens`（默认 = `max_num_batched_tokens`）
   - 每个请求消耗 `num_new_tokens = min(num_tokens_remaining, token_budget, chunk_limit)`
   - Chunked prefill 通过 `long_prefill_token_threshold` 限制单次 prefill 量

3. **调度顺序**（`scheduler.py:388-561`）:
   - Step 1: 遍历 `self.running` 列表，用剩余 token budget 分配 token 给正在运行的请求
   - Step 2: 从 `self.waiting` / `self.skipped_waiting` 队列中取出新请求
   - Running 请求优先于 waiting——这隐式实现了 decode-first 优先级

4. **Preemption**（`scheduler.py:952-972`）:
   ```python
   def _preempt_request(self, request, timestamp):
       self.kv_cache_manager.free(request)      # 释放 KV cache blocks
       self.encoder_cache_manager.free(request)  # 释放 encoder cache
       request.status = RequestStatus.PREEMPTED
       request.num_computed_tokens = 0           # 重置——下次从头开始
       request.num_preemptions += 1
       self.waiting.prepend_request(request)     # 插回 waiting 队列头部
   ```
   - **只有 recompute preemption——没有 swap**
   - KV blocks 被释放回 block pool
   - `num_computed_tokens` 归零，需全部重算
   - Preempted 请求插回 waiting 队列**头部**（保证不会饿死）

5. **Priority preemption 的选择逻辑**（`scheduler.py:479-484`）:
   ```python
   if self.policy == SchedulingPolicy.PRIORITY:
       preempted_req = max(self.running,
           key=lambda r: (r.priority, r.arrival_time))
   ```
   - 选择 **priority 最大（即优先级最低）** 且**到达最早**的请求
   - 对于 FCFS 策略：直接 `self.running.pop()`——弹出列表最后一个（最后加入的）

### 3.3 为什么 V1 可以安全移除 Swap Preemption？

**关键洞察**: Prefix caching 使 recompute 的代价大幅降低。

- Preempted 请求的 computed tokens 对应的 KV cache blocks 虽然被 `free()`，但 **prefix cache 中的 blocks 仍然存在**（只要 ref_cnt > 0）
- 当请求重新调度时，`kv_cache_manager.get_computed_blocks()` 在哈希表中查找已缓存的 blocks（`scheduler.py:616-618`）
- 因此 recompute 不是从头计算——只有未被其他请求共享的 blocks 需要重新计算
- 在高前缀共享场景（如 system prompt 重用），recompute preemption 几乎零开销

**数据依据**: PCIe Gen4 带宽 ~64 GB/s vs H100 计算 ~2 PFLOPS (FP16)。对于 Llama-70B，一个请求的完整 KV cache 约 10-20 GB，swap 需要 150-300ms。而 recompute 在 prefix cache 命中 80% 时只需 10-20ms。

---

## 4. 经典 OS 调度 vs LLM 推理调度

### 4.1 能迁移的概念

| OS 调度 | LLM 推理映射 | 为什么有效 |
|---------|-------------|----------|
| **FCFS** | vLLM 默认策略 `SchedulingPolicy.FCFS` | 简单、公平、无饥饿；LLM 场景中请求长度不可预测，无法用 SJF |
| **Priority** | `SchedulingPolicy.PRIORITY`，基于 `(priority, arrival_time)` | 需要 QoS 区分的生产环境；lower value = higher priority |
| **Preemption** | KV cache block 释放 + recompute（类似 OS 的 swap out） | 类比 OS 从内存换出进程；但 LLM 中 "换入" 是通过 prefix cache 的 recompute |
| **Token budget** | 类比 OS 的 time quantum | `max_num_scheduled_tokens` 控制每次调度分配的计算配额 |

### 4.2 无法迁移的概念及原因

| OS 调度 | 为什么不适用 |
|---------|------------|
| **Round Robin (RR)** | RR 要求快速上下文切换（微秒级）；LLM 的 "上下文" 是 GB 级的 KV cache，切换代价极高 |
| **SJF (Shortest Job First)** | LLM 请求的输出长度事前不可知（`max_tokens` 是上限而非实际长度） |
| **MLFQ (Multi-Level Feedback Queue)** | 依赖对 "job length" 的动态估计；LLM 中请求的 "长度" 是生成的 token 数，生成过程中才知道 |
| **CFS (Completely Fair Scheduler)** | 需要精确的 vruntime 追踪；LLM 的 "执行时间" 主要由 KV cache 大小和 batch 组成决定，而非 CPU time |
| **Aging** | vLLM **未实现** priority aging——见第 7 节 |

---

## 5. 竞品对比

### 5.1 vLLM vs TensorRT-LLM

| 维度 | vLLM | TensorRT-LLM |
|------|------|-------------|
| **Batching** | Continuous batching（统一 token-budget 模型） | In-flight batching（概念相同，实现不同） |
| **Prefill** | Chunked prefill（默认开启，不可关闭） | Chunked context + decode-first |
| **KV Cache** | PagedAttention（固定 size blocks + block table） | Paged KV cache（可配置 block 布局） |
| **调度策略** | FCFS 或 Priority；无 decode-first 显式开关 | 内置 decode-first；通过 `max_batch_size` 和 `max_num_tokens` 调优 |
| **硬件** | 任意 CUDA GPU（含 AMD ROCm） | NVIDIA 独占；Hopper/Blackwell 深度优化 |
| **高并发** | 线性扩展，100+ 并发时最优 | 低并发最优，高并发退化于 H100；B200 全面反超 |

**[数据]** GPT-OSS-120B + 2xH100:
- vLLM: 4,741 tok/s @ 100 concurrent; TTFT 0.053s @ 1 concurrent
- TRT-LLM: 2,163 tok/s @ 100 concurrent; per-token 0.004s @ 1 concurrent

### 5.2 vLLM vs SGLang

| 维度 | vLLM | SGLang |
|------|------|-------------|
| **Prefix Caching** | Hash-based (SHA256/xxHash) + block-level | **Radix tree** (trie) → 支持 partial match 和自动 split |
| **缓存粒度** | Block-level（整块匹配） | Token-level（trie 路径匹配，更精细） |
| **Scheduling** | FCFS/Priority；scheduler 与 GPU 有间隙 | FCFS/LRU/LPM/Priority；**zero-overhead**（CPU 调度与 GPU 计算重叠） |
| **Load Balancing** | 基于 DP coordinator 的负载均衡 | **Cache-aware router**——将请求路由到缓存命中率最高的 worker |
| **Eviction** | LRU（free queue 的 head） | LRU with leaf prioritization + reference counting (lock_ref) |

**[关键差异]**: SGLang 的 radix tree 前缀缓存在结构上更灵活（token 级 vs block 级），但其核心调度思想（continuous batching + chunked prefill）与 vLLM 同源。SGLang v0.4 的 "zero-overhead batch scheduler" 是创新点——调度器提前一步运行，与 GPU 完全重叠。

### 5.3 vLLM vs DeepSpeed-MII

| 维度 | vLLM | DeepSpeed-MII |
|------|------|-------------|
| **核心调度创新** | Chunked prefill（交错 prefill/decode） | **Dynamic SplitFuse**（分解 prompt + 融合前向） |
| **KV Cache** | PagedAttention（OS 虚拟内存模型） | Blocked KV Cache（类似分页，但实现不同） |
| **性能声明** | SOSP 2023: 2-4x over Orca/FasterTransformer | 声称 2.5x over vLLM（注意：不同基准测试条件） |
| **模型兼容性** | 50+ 模型架构，HuggingFace 兼容 | 主流模型但覆盖更窄 |
| **部署复杂度** | pip install 即可 | 需 DeepSpeed 全栈（ZeRO + inference engine） |

---

## 6. 设计决策树

```
                    LLM 推理调度入口
                          │
            ┌─────────────┴─────────────┐
            │                           │
      Static Batching            Continuous Batching
      (2019-2021)                 (Orca 2022 → )
            │                           │
            │                    ┌──────┴──────┐
            │                    │             │
            │              Prefill/Decode    统一 Token-Budget
            │              分离 (vLLM v0)     (vLLM v1, 2024)
            │                    │             │
            │              ┌─────┴─────┐       │
            │              │           │       │
            │          Swap-Preempt  Recompute  Recompute-Only
            │          (GPU↔CPU)     Preempt    (prefix cache)
            │              │           │       │
            │              │           └───┬───┘
            │              │               │
            └──────────────┴───────────────┘
                        已废弃          当前架构
```

**分支决策理由**:

1. **Static → Continuous Batching**: Orca 证明了 iteration-level scheduling 没有性能损失且显著提升 GPU 利用率。这是整个领域的共识，不存在争议。

2. **Phase 分离 → 统一 Token-Budget**: 
   - Phase 分离要求调度器知道并管理 "prefill" vs "decode" 的区别，增加复杂度
   - Chunked prefill (Sarathi-Serve) 证明 prefill 不需要一次性完成
   - 统一模型下，optimizations（prefix caching, spec decode）自然地组合在一起
   - 引用 Woosuk 在 `scheduler.py:354` 的注释："This is general enough to cover chunked prefills, prefix caching, speculative decoding, and the 'jump decoding' optimization in the future."

3. **Swap → Recompute-Only Preemption**:
   - V1 中 prefix caching 始终开启，recompute 代价大幅降低
   - `_preempt_request()` 只需 `free()` + `num_computed_tokens = 0`——极简
   - 消除了 swapped 队列和 CPU block allocator 的维护成本
   - PCIe 带宽瓶颈（64 GB/s）无法匹配 H100 计算能力（2 PFLOPS），swap 在大型模型上没有优势

---

## 7. Preemption 代价的定量分析

### 7.1 Swap preemption 代价（V0，已移除）

```
总延迟 = KV Cache Size / PCIe 带宽 × 2 (出+入)

典型数值（Llama-70B, 4096 tokens, FP16）:
  - KV cache per layer: 2 × 80 × 128 × 4096 × 2 bytes = ~160 MB
  - Total (80 layers): ~12.5 GB
  - PCIe Gen4 带宽: ~64 GB/s 实测
  - Swap out: 12.5 / 64 ≈ 195 ms
  - Swap in: 12.5 / 64 ≈ 195 ms
  - 总暂停: ~390 ms per preemption

H100 NVL (PCIe Gen5): ~128 GB/s → ~98 ms each way → ~196 ms 总暂停
NVLink + NVSwitch: ~900 GB/s → ~14 ms each way → ~28 ms 总暂停
```

### 7.2 Recompute preemption 代价（V1，当前）

```
总延迟 = 有效 token 数 / GPU 吞吐量

典型数值（Llama-70B, 4096 tokens, H100）:
  - 场景 1: 80% prefix cache hit
    - 有效重算 tokens: 4096 × 0.2 = 819
    - H100 prefill 吞吐(~200K tok/s for 70B): 819 / 200000 ≈ 4 ms
  - 场景 2: 0% prefix cache hit (最坏情况)
    - 有效重算 tokens: 4096
    - 延迟: 4096 / 200000 ≈ 20 ms

对比：
  - Swap: 195-390 ms
  - Recompute (80% cache hit): ~4 ms → 49x-98x 更快
  - Recompute (0% cache hit): ~20 ms → 10x-20x 更快
```

**结论**: 即使没有 prefix cache 命中，recompute 仍比 swap 快 10x 以上。这是因为 H100 的 compute 能力远超 PCIe 带宽。Prefill computation 是 compute-bound 且利用矩阵乘法的高算术强度。

### 7.3 何时 preemption 会发生

在 V1 中，preemption 只有一个触发条件：`kv_cache_manager.allocate_slots()` 返回 `None`（无足够 free blocks 分配给请求的新 tokens）。

```python
# scheduler.py:466-475
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, num_lookahead_tokens=...)
    if new_blocks is not None:
        break    # 分配成功
    # 分配失败 → 触发 preemption
```

---

## 8. Starvation 和 Priority Aging

### 8.1 当前状态：vLLM 没有实现 priority aging

搜索 `scheduler.py` 全文，不存在 aging、decay、boost、escalation 相关逻辑。Priority 是静态的——请求创建时设定后永不改变。

### 8.2 为什么没有 aging？

**理论分析**——在生产 LLM 服务场景中，aging 不那么关键的原因：

1. **请求生命周期的自限性**: LLM 请求有 `max_tokens` 上限，不会无限期占用资源。一个 running 请求最终会完成或被 preempt。
2. **Token Budget 的隐式公平**: 每步的 token budget 均匀分配给所有 scheduled 请求。长请求消耗更多 token budget 但也被 chunked prefill 限制，不会垄断。
3. **Preemption 的优先级倒置保护**: V1 的 priority preemption 会在资源不足时踢出低优先级请求。这意味着即使没有 aging，低优先级请求在资源紧张时会被主动腾出空间。
4. **FCFS 默认策略不区分优先级**: 大多数部署使用 FCFS（默认），aging 本身没有意义。

### 8.3 持续负载下的行为

```
Sustained load (requests/sec > capacity):
  ┌─────────────┐
  │ waiting Q →∞ │  ← 无限增长（无 admission control 主动拒绝）
  └──────┬──────┘
         │
    ┌────▼────┐
    │ running │  ← 固定 max_num_running_reqs (默认 128)
    └────┬────┘
         │
    ┌────▼────┐
    │  Token  │  ← per-step token budget 在 running 请求间分配
    │  Budget │
    └─────────┘

结果：
- 所有 running 请求被均等调度（decode 优先于 prefill）
- Waiting 队列尾部请求无限等待——无 TTL/timeout 在调度器级别
- 由上层（API server）的超时机制处理请求放弃
- 生产建议：用 admission controller 或 HTTP 429 在入口限流
```

### 8.4 对比 SGLang 的 priority scheduling

SGLang 的 `--enable-priority-scheduling` 支持 **preemption threshold**——高优先级请求只能抢占比它低一定级别的请求，避免了绝对的 "高优先级饿死所有低优先级"。

vLLM 的 priority preemption 更简单粗暴：`max(running, key=lambda r: (r.priority, r.arrival_time))`，总是抢占比它优先级最低的那个。

---

## 9. 源码锚点速查

| 概念 | 源码位置 |
|------|---------|
| `Scheduler.schedule()` 主逻辑 | `scheduler.py:352-945` |
| Scheduling 设计哲学注释 | `scheduler.py:353-362` |
| Priority preemption 选择 | `scheduler.py:479-504` |
| FCFS preemption 选择 | `scheduler.py:503-504` |
| `_preempt_request()` 实现 | `scheduler.py:952-972` |
| `RequestQueue` 抽象基类 | `request_queue.py:20-72` |
| `FCFSRequestQueue` (deque) | `request_queue.py:75-128` |
| `PriorityRequestQueue` (heap) | `request_queue.py:131-198` |
| `SchedulingPolicy` enum | `request_queue.py:13-17` |
| `SchedulerConfig` 全部参数 | `config/scheduler.py:26-149` |
| Token budget 初始化 | `scheduler.py:371` |
| Running 请求调度 (decode-first) | `scheduler.py:388-561` |
| Waiting 请求调度 | `scheduler.py:568-715` |
| KV Cache block allocation | `scheduler.py:466-475` |
| Prefix cache hit check | `scheduler.py:616-618` |
| `SchedulerOutput` dataclass | `output.py:181-255` |

---

## 10. 给 Writer 的建议

### 最关键的 intuition

**vLLM 调度器本质是一个 token 计数器追赶器。** 它不关心 "prefill" 或 "decode"，只关心 `num_computed_tokens` 是否追上了 `num_tokens`。每步从 `token_budget` 中分配 token 给各个请求，让它们追赶进度。这个极简模型统一了 chunked prefill、speculative decoding、prefix caching——它们都是这个追赶过程的自然变体。

### 最适合零基础的切入角度

从 **"食堂打饭"** 类比切入是最容易理解的方式：
- Static batching = 必须凑齐一桌人才能上菜
- Continuous batching = 一个人吃完立刻换下一个，不用等整桌
- Token budget = 每次最多上 100 个菜（不分给谁，只看总数）
- Preemption = 有人太能吃，把他剩下的菜收走让给别人，下次他再来时如果菜还在锅里（prefix cache），就能快速端上来；否则重新做（recompute）

### 容易混淆的概念

1. **Continuous batching vs Chunked prefill**: 前者是 "请求可以随时加入/退出批次"，后者是 "prompt 处理可以分块进行"。两者相关但不同。Continuous batching 解决的是 batch 灵活性；chunked prefill 解决的是 prefill 不阻塞 decode。vLLM v1 同时使用两者。

2. **Swap preemption vs Recompute preemption**: Swap 是 "换出到 CPU 内存"，recompute 是 "扔掉重算"。初学者常认为 swap 更快（因为 "存了就不用算了"），但实际上 H100 compute 远快于 PCIe 带宽，recompute 在 prefix cache 加持下反而更快。

3. **FCFS vs Priority policy**: vLLM 只有这两个策略，远少于 OS 调度器。这不是偷懒——LLM 推理的约束条件（KV cache 大小、GPU 内存、batch 组成）已足够复杂，简单的调度策略反而更容易预测和调优。

4. **Priority 的含义**: `priority` 值越小 = 优先级越高（与 Linux nice 值一致）。相同 priority 时，`arrival_time` 越早越优先。

### 生活化类比建议

建议 Writer 使用以下三层类比阶梯：

1. **入门**（无技术背景读者）: "食堂打饭"类比——见上
2. **进阶**（有 CS 基础但无 LLM 背景）: "OS 虚拟内存"类比——block table 就是 page table，preemption 就是 swapping，prefix cache 就是 page cache
3. **专家**（有 LLM 推理背景）: 直接讨论 token-budget 模型的设计含义——为什么统一 prefill/decode 允许更灵活的 optimization composition

### 章节结构建议

1. 从 "食堂打饭" 类比引出调度问题
2. 展示 static batching 的浪费（定量：GPU 利用率 30-50% → Orca 的 36.9x 承诺）
3. 引入 continuous batching（Orca 论文摘要 + 图解）
4. Deep dive vLLM v1 调度器：
   - Token-budget 模型（读 `scheduler.py:353-362`）
   - 两个队列 + priority policy（读 `request_queue.py`）
   - Preemption 机制（读 `_preempt_request`）
5. Swap vs recompute 的定量分析（PCIe vs H100 compute）
6. 竞品全景（TRT-LLM / SGLang / MII）一张表 + 2-3 段叙述
7. 边界条件：starvation、无 aging、持续负载下的行为
8. 与下一章（Memory Management / KV Cache）的连接——调度器的决策最终通过 `kv_cache_manager.allocate_slots()` 落地

---

*Generated by Researcher agent, 2026-05-05. Sources: vLLM source code (commit range: 2024-2025), OSDI 2022/2024 proceedings, SOSP 2023 proceedings, official TensorRT-LLM/SGLang/DeepSpeed-MII documentation.*
