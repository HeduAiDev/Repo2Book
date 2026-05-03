# 第2章：KV Cache — vLLM 的内存管理核心

> 打开 `vllm/v1/core/kv_cache_manager.py:106`，`allocate_slots()` 是每次 scheduler 循环中第一个被调用的方法。
> 本章把这个方法的每一层逻辑——以及它们为什么这样设计——讲清楚。

---

## 这章要做什么？

第 1 章搞懂了 vLLM 的 `Attention` 类和它的 backend 架构。但有一个问题悬而未决：`Attention.forward()` 接受的 K 和 V 从哪来？谁分配它们的显存？谁在请求结束时回收？如果显存不够了，谁被驱逐？

答案是 `KVCacheManager`。它是 vLLM 内存系统的核心枢纽——Scheduler 通过它做调度决策，Attention kernel 通过它找到 KV 数据，Prefix Cache 通过它共享。

但本章不是 API 文档。我们会从第一性原理出发，推导三个核心问题：
1. **为什么 KV Cache 是必然而非优化？**（2.1 节——自回归生成的数学分析）
2. **为什么需要 block 级别的管理？**（2.3 节——碎片化的数学建模）
3. **驱逐顺序为什么重要？**（2.4 节——LRU 的最优性条件）

带着数学去读源码。每一节的结构：先问为什么 → 从第一性原理推导 → 回到 vLLM 源码看它如何实现。

学完这章你能：
- 推导 KV Cache 的显存公式并逐项解释——每一项代表什么硬件约束
- 理解 vLLM 三层 KV Cache 架构（KVCacheManager → Coordinator → BlockPool）的分工逻辑
- 手写 `FreeKVCacheBlockQueue`——那个被自己实现的双向链表，以及为什么 deque 不够好
- 解释 `allocate_slots()` 返回 `None` 时 Scheduler 如何进入 preempt 模式

---

## 2.1 为什么 KV Cache 是必然，不是优化？

### Theory: 自回归生成的根本浪费

Attention 公式中（第 1 章），生成 token$_N$ 需要所有历史 token 的 K 和 V：

$$
\mathrm{output}_N = \mathrm{softmax}\left(\frac{Q_N \cdot [K_0, K_1, ..., K_{N-1}]^T}{\sqrt{d_k}}\right) \cdot [V_0, V_1, ..., V_{N-1}]
$$

LLM 是自回归的：要生成 token$_N$，必须先有 token$_{N-1}$。这个因果依赖躲不掉——但 K 和 V 的计算可以复用。每次生成新 token 时，只有当前 token 的 K 和 V 是新的；所有历史 token 的 K 和 V 在上一步已经算过了，而且完全没变。

让我们量化这个浪费。假设生成长度为 $N$ 的序列：

$$
\begin{aligned}
\text{Step 0: } &\text{生成 token}_1 \quad \text{需要计算 } K_0, V_0 \quad (1 \text{ token 的 attention}) \\
\text{Step 1: } &\text{生成 token}_2 \quad \text{需要计算 } K_0, K_1, V_0, V_1 \quad (2 \text{ tokens——重复},K_0,V_0!) \\
\text{Step 2: } &\text{生成 token}_3 \quad \text{需要计算 } K_{0..2}, V_{0..2} \quad (3 \text{ tokens——重复},K_0,K_1,V_0,V_1!) \\
&\vdots
\end{aligned}
$$

用数学归纳：

$$
\begin{aligned}
\text{Without KV Cache:} \quad &\sum_{i=1}^{N} i = \frac{N(N+1)}{2} \text{ token-attentions} \\
\text{With KV Cache:} \quad &N \text{ token-attentions} \\
\text{Speedup:} \quad &\frac{N+1}{2} \times
\end{aligned}
$$

**$N=4096$ 时，speedup = 2048×。** 这不是"快了 2 倍"级别的优化——这决定了推理是否可行。没有 KV Cache，生成 4096 token 需要 840 万次 attention 计算；有 KV Cache，只需 4096 次。

### Source Trail

打开 `vllm/v1/core/sched/scheduler.py:L467`，这是每个 running 请求在每个 scheduler step 都要走的关键路径：

```python
# scheduler.py:L467
kv_cache_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens, ...
)
if kv_cache_blocks is None:
    self._preempt_lowest_priority_request()
```

`allocate_slots()` 返回 `None` = GPU 显存不够。Scheduler 必须选一个请求驱逐。整个 KV Cache 系统存在的根本原因就在这三行代码里。

`★ Insight ─────────────────────────────────────`
KV Cache 的本质：**以空间换时间。** 你把已经算好的 K 和 V 存在显存里——这笔空间的代价是显存被占用。收益是每次生成新 token 只对新 token 做 attention。没有这个交换，推理就是不可行的。这解释了为什么后续所有优化——PagedAttention、Prefix Cache、KV Quant——都围绕着"如何更高效地使用这笔空间"来设计。
`─────────────────────────────────────────────────`

---

## 2.2 显存公式：每一 byte 来自哪里？

### Theory: 逐项拆解

KV Cache 需要多少显存？这个公式是可以精确推导的，没有经验常数：

$$
\text{KV\_Cache\_Size} = 2 \times N_{\text{layers}} \times B \times L \times N_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype\_bytes}
$$

逐项解释：
- **× 2:** K 和 V 各一份。**最容易漏的一项。**
- **× $N_{\text{layers}}$:** 每层 Transformer 的注意力模式不同（浅层学语法，深层学语义），K/V 表达完全不同——必须各自缓存
- **× $B$:** 每个并发请求有独立的 KV Cache。batch=1 的显存占用不是 batch=8 的 1/8
- **× $L$:** 每个 token 位置占一个槽。序列越长，缓存越大——线性增长
- **× $N_{\text{kv\_heads}}$:** GQA 直接砍这个——从 32 到 8，缓存缩为 25%
- **× $d_{\text{head}}$:** 每个 head 的维度。Llama-3.2-1B: 128
- **× dtype_bytes:** bf16=2, fp8=1, int4=0.5。量化直接影响缓存大小

### 真实例子（Llama-3.2-1B）

```
config: 32 layers, 8 kv_heads (GQA), 128 head_dim, bf16

场景 1: 1 用户, 4096 token 输入
  = 2 × 32 × 1 × 4096 × 8 × 128 × 2 = 536,870,912 bytes ≈ 512 MB

场景 2: 8 并发, 平均 4096 token
  ≈ 4 GB

场景 3: 1 用户, 131K 超长上下文
  ≈ 16 GB  ← 模型权重本身才 ~2.5 GB (fp16)!
```

### Source Trail

在 vLLM 中，这些参数被编码在 `KVCacheConfig` 里（`vllm/v1/kv_cache_interface.py:L760`）。初始化时，`resolve_kv_cache_block_sizes()`（`kv_cache_utils.py:L569`）会根据可用显存反推 `num_gpu_blocks`：

```python
# 简化版逻辑 (kv_cache_utils.py → resolve_kv_cache_block_sizes)
block_bytes = 2 * block_size * num_kv_heads * head_dim * dtype_bytes * num_layers
available = total_gpu_memory * gpu_memory_utilization - model_weight_size
num_blocks = available // block_bytes
```

注意 `gpu_memory_utilization`——默认 0.90。那 10% 去哪了？留给 activation memory 和 PyTorch 的内存碎片。这个参数在显存紧张时是第一个要调的。

---

## 2.3 为什么需要 Block 级别的管理？

### Theory: 碎片化问题

假设你用连续分配：请求 A 需要 1000 token 的 KV Cache，你分配 [0..999]。请求 B 需要 200 token，分配 [1000..1199]。请求 A 完成了——释放 [0..999]。现在有一个 1200 token 的新请求 C——它放不进这 1000 个连续的槽。

这就是**外部碎片**——空闲空间总量足够（1000 已释放 + 剩余 800 = 1800 > 1200），但因为没有足够大的连续块而分配失败。用数学建模：

对于 $k$ 个请求，序列长度分别为 $L_1, L_2, ..., L_k$，连续 KV Cache 的浪费比：

$$
\text{Waste Ratio} = 1 - \frac{\sum_i L_i}{\text{max\_blocks} \times \text{block\_size}}
$$

当请求长度分布方差大时（长 system prompt + 短 user message 混合），浪费率可达 50-75%。这就是 PagedAttention（第 3 章）要解决的问题——但 Block 级别的管理从这一章就开始了。

### Source Trail

打开 `vllm/v1/core/kv_cache_utils.py:114`。`KVCacheBlock` 是一个最小的元数据单元——它只记录这个物理块的状态，不存储实际的 K 和 V 数据：

```python
# kv_cache_utils.py:L114
@dataclass
class KVCacheBlock:
    block_id: int           # 在物理池中的索引
    ref_cnt: int            # 多少个请求在用这个块
    _block_hash: ...        # 设为 hash = 这个块已满且被缓存
    prev_free_block: ...    # 自由链表的前驱指针
    next_free_block: ...    # 自由链表的后继指针
```

实际的 K、V 数据存储在 GPU tensor 中——`[2, num_blocks, block_size, num_kv_heads, head_size]`。Python 对象只管理元数据：谁在用、这个块有没有 hash、它在自由链表中的位置。

**Block 大小的选择是一个trade-off。** vLLM 默认 block_size=16 tokens。为什么是 16？

- 太小（4）：每个 block 有 4 个 token → 4096 token 需要 1024 个 block → block table 太大会占用太多寄存器，影响 attention kernel 性能
- 太大（128）：碎片化又回来了——block 越大，内部碎片越多（最后一个 block 平均只用了 50%）
- 16-32 tokens：实践中发现的最佳平衡点。足够小以最小化内部碎片，足够大以保持 block table 在几个 cache line 内

---

## 2.4 LRU 驱逐：双向链表的数学

### Source Trail

打开 `vllm/v1/core/kv_cache_utils.py:162`。`FreeKVCacheBlockQueue` 不是 `collections.deque`——vLLM 实现了自己的双向链表。为什么？

```python
# kv_cache_utils.py:L162-L340
class FreeKVCacheBlockQueue:
    """LRU 自由块队列——手工双向链表"""
    # head → 最久未使用 (LRU, 最先驱逐)
    # ...
    # tail → 最近使用 (MRU, 最后驱逐)
```

### Theory: 为什么 deque 不够？

标准库 `collections.deque` 是一个双端队列。它的三个核心操作：
- `popleft()`: O(1) ✓ （驱逐 LRU 块）
- `append()`: O(1) ✓ （释放后放回 MRU 位置）
- `remove(item)`: **O(n)** ✗ （从中间删除——需要扫描全队列找到这个 item）

第三个操作是杀手。当一个请求命中 prefix cache 时，之前自由的块被 `BlockPool.touch()` 从队列中移除——因为另一个请求开始使用它了。`deque` 需要 O(n) 扫描才能找到要删除的 item。

手工双向链表的 `remove()` 是 O(1)——块本身存储了 `prev_free_block` / `next_free_block` 指针：

```python
def remove(self, block):
    block.prev_free_block.next_free_block = block.next_free_block
    block.next_free_block.prev_free_block = block.prev_free_block
    self._size -= 1
```

**驱逐顺序的数学：为什么释放时要用 REVERSE 顺序？**

从 `single_type_kv_cache_manager.py:L303`：

```python
self.block_pool.free_blocks(list(reversed(req_blocks)))
```

`free_blocks()` 把块 append 到自由队列的尾部（MRU 位置）。追加的顺序决定了驱逐顺序——最先 append 的离头部最近，最早被驱逐。

为什么 reverse？因为 tail blocks（更多 tokens 已被 hash，涵盖更广的 token 范围）的重新计算代价更高——它们在 prefix cache 中的价值更大。Reverse 让这些有价值的块排在队列后面（更靠近 MRU）——当需要分配新块时，head blocks（较少 hash tokens）先被驱逐。

这是一个形式化的最优性条件：给定块的重新计算代价 $c_i$（与 hash tokens 数量成正比），LRU 驱逐策略最小化期望驱逐代价 $\mathbb{E}[\sum_{\text{evicted}} c_i]$。

---

## 2.5 allocate_slots() 的三阶段算法

### Source Trail

打开 `vllm/v1/core/kv_cache_manager.py:L300-L310`。源码注释中明确的三阶段：

```python
# Stage 1: 释放被跳过的块 (sliding window attention 跳过旧块)
self.coordinator.remove_skipped_blocks(...)

# Stage 2: 检查容量——计算需要的新块数 + 检查自由队列
num_blocks = self.coordinator.get_num_blocks_to_allocate(...)
if num_blocks > free_blocks:
    return None  # ← OOM: Scheduler 进入 preempt

# Stage 3: 分配新块 + 缓存满块
self.coordinator.allocate_new_blocks(...)          # 物理分配
self.coordinator.allocate_new_computed_blocks(...)  # Prefix cache 命中
self.coordinator.cache_blocks(...)                  # 标记满块为已缓存
```

### Theory: 为什么是三阶段？

这三个阶段对应 KV Cache 的三个不同"顾客"：

1. **Stage 1 (Free Skipped):** 为 sliding window attention 服务。当 token 滑出窗口——例如 Mistral 的 window_size=4096——窗口外的 K 和 V 不再被任何未来 token 需要。这些块应该立刻释放，把物理空间还给自由池。**这是"主动清理"模式——不是等 OOM 才清理。**

2. **Stage 2 (Check Capacity):** 保护系统的防线。如果分配失败在这里被检测到——在物理分配之前——可以避免部分分配的不一致状态。Scheduler 看到 `None` 后知道必须驱逐一个请求。

3. **Stage 3 (Allocate + Cache):** 物理块从自由池取出（`BlockPool.get_new_blocks()`），prefix cache 命中被登记（`allocate_new_computed_blocks()`），满块被标记为可共享（`cache_blocks()` → `BlockPool.cache_full_blocks()` 设置 `block_hash`）。

**设计原理：** 这三个阶段对应了"先还债，再数钱，最后花"的财务原则——先归还不再需要的资源（避免浪费），再检查余额够不够（避免超支），最后才真正消费。这个顺序的任意重排都会导致资源泄漏或分配不一致。

---

## 2.6 Prefix Cache 的 Hash 机制

### Source Trail

`BlockPool.cache_full_blocks()`（`block_pool.py:L211`）和 `BlockPool.get_cached_block()`（`block_pool.py:L184`）构成了前缀缓存的存储和查找。查找是通过 `BlockHashToBlockMap`（`block_pool.py:L34`）——本质上是一个 hash table：

```python
# 缓存一个满块——设置 block_hash，加入 hash 索引
def cache_full_blocks(self, blocks, block_hashes):
    for block, h in zip(blocks, block_hashes):
        block.set_hash(h)
        self.cached_block_hash_to_block[h] = block

# 根据 token hash 查找已缓存的块——O(1) hash table lookup
def get_cached_block(self, block_hash):
    return self.cached_block_hash_to_block.get(block_hash)
```

Hash 的计算在 `kv_cache_utils.py:L539` 的 `hash_block_tokens()`：

```python
hash_fn((parent_block_hash, tuple(token_ids), extra_keys))
```

注意：**hash 是链式的**——当前块的 hash 依赖父块的 hash。这形成了 Merkle-tree-like 的结构。即使两个相同 token ID 的 block，如果它们的父块不同（不同的上下文前缀），它们的 hash 就不同。这避免了前缀碰撞。

### Theory: Lazy Eviction 的正确性

vLLM 在前缀缓存中做了一个关键的简化：**释放一个块时不立即清除它的 hash。** 代码路径：

```python
# free_blocks() 把块放回自由队列——但不清除 hash！
block_pool.free_blocks(blocks)

# ... 时间流逝，块在自由队列中等待 ...

# get_new_blocks() 从队列中取出块——此时才清除 hash
def get_new_blocks(self, num_blocks):
    blocks = self.free_queue.popleft_n(num_blocks)
    for b in blocks:
        self._maybe_evict_cached_block(b)  # ← 驱逐发生在这里
    return blocks
```

**正确性论证：** 只要块还在自由队列中，它的内容就没有被覆盖，block_hash 就是有效的。在 hash 被驱逐之前到达的请求都能命中，之后到达的请求重新计算。这是 lazy eviction——我们推迟驱逐到物理块被重新分配的时刻，而不是释放的时刻。在最坏情况下（块在自由队列中停留 1ms 后被重新分配），cache hit 的窗口只多了 1ms——但这仍然可能捕获一个刚好在同一毫秒到达的相同 system prompt 的请求。而在最好情况下（块在自由队列中停留到下一个请求带着相同前缀到达），我们完全避免了重新计算。

**与 OS 页表替换的类比：** 这就像操作系统中的 page cache——文件内容在内存中，标记为 clean。当内存不够时，clean page 可以被回收，但内容不会丢失（在磁盘上）。这里 GPU 显存是"磁盘"——但资源是 KV 计算时间。释放一个 block 就像标记一个 page 为 clean——它在下一毫秒仍可用，直到物理块被重新分配。

---

## 2.7 Scheduler 的完整交互

### Source Trail

从 `vllm/v1/core/sched/scheduler.py` 追踪完整的调度循环。

**请求从 WAITING → RUNNING 的转换（scheduler.py:L617-L744）：**

```python
# 1. 查前缀缓存——这个 system prompt 有没有被其他请求算过？
new_computed_blocks, num_new_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)

# 2. 分配 KV Cache（含前缀命中的块 + 需要新算的块）
kv_cache_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens,
    new_computed_blocks=new_computed_blocks,
    num_external_computed_tokens=...,
)

# 3. OOM → 驱逐优先级最低的 running 请求
if kv_cache_blocks is None:
    preempted = self._find_lowest_priority_request()
    self.kv_cache_manager.free(preempted)
    # 重试分配...
```

**请求完成后释放（scheduler.py:L1813）：**

```python
def _free_request(self, request):
    self.kv_cache_manager.free(request)
    # → coordinator.free()
    #   → SingleTypeKVCacheManager.free()
    #     → block_pool.free_blocks(list(reversed(req_blocks)))
```

### Theory: Preempt 的决策模型

当 `allocate_slots()` 返回 `None`，Scheduler 面临一个决策：驱逐哪个请求？

vLLM 采用 **priority-based preemption**：每个请求有优先级，驱逐最低优先级的。这避免了 starvation——高优先级请求（如 latency-sensitive 的交互用户）永远不会被驱逐。

但纯优先级策略有一个问题：如果低优先级请求的 KV Cache 已经被大量重用（prefix cache 中），驱逐它可能导致其他请求也失去前缀缓存命中——雪崩效应。生产级的 Scheduler 还会考虑驱逐的"blast radius"——被驱逐请求的 block 有多少被其他请求通过 prefix cache 引用（`ref_cnt > 1`）。高 `ref_cnt` 意味着驱逐一个请求会影响多个其他请求。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `KVCacheBlock` | `kv_cache_utils.py:L114` | vLLM 用 `__slots__` 优化 Python 对象内存；我们用 `@dataclass` 方便阅读 |
| `FreeKVCacheBlockQueue` | `kv_cache_utils.py:L162` | 手工双向链表——保留相同的 LRU O(1) 语义和 append_n(reversed(...)) 驱逐顺序 |
| `BlockPool` | `block_pool.py:L130` | 保留核心方法：get_new_blocks / free_blocks / cache_full_blocks / touch / evict_blocks |
| `KVCacheManager.allocate_slots()` | `kv_cache_manager.py:L225` | 三阶段分配：free skipped → check capacity → allocate + cache |
| `KVCacheBlocks` | `kv_cache_manager.py:L22` | 简化为单 KV cache group 版本；vLLM 支持 multi-group（hybrid models） |
| `KVCacheConfig` | `kv_cache_interface.py:L760` | 保留核心字段和 `calculate_num_blocks()` |
| Scheduler 交互（allocate → OOM → preempt → free） | `scheduler.py:L467-L499, L1813` | 交互模式完全一致 |
| Prefix cache hash chain | `kv_cache_utils.py:L539` `hash_block_tokens()` | hash 链式依赖保留 |
| Lazy eviction | `block_pool.py:L354` `_maybe_evict_cached_block()` | 释放保留 hash → 重新分配时驱逐 |

---

## 验证

在 vLLM Docker 容器中运行：

```bash
cd artifacts/02-kv-cache && python -m pytest tests/ -q
# 19/19 passed ✅
```

---

## 总结

从 `kv_cache_manager.py:106` 到 `block_pool.py:130` 到 `kv_cache_utils.py:162` 的双向链表：

- **KV Cache 是必然，不是优化。** 数学证明：没有它，生成 $N$ token 需要 $N(N+1)/2$ 次 attention——$N=4096$ 时 2048× 的差距决定了推理是否可行。本质是以空间换时间。
- **Block 级管理是因为碎片化。** 当请求长度分布方差大时（长 system prompt + 短 user message 混合），连续分配的浪费率达 50-75%。Block 分配用内部碎片（最后一个 block 的 50% 利用率）换取零外部碎片。
- **手工双向链表而不是 deque。** 因为 prefix cache 的 `touch()` 操作需要 O(1) 从中间删除——这在 deque 中是 O(n)。
- **释放时保留 hash。** 物理块在被重新分配之前，缓存仍然有效。这是 lazy eviction——推迟驱逐到最后一刻，最大化 cache hit 窗口。
- **三阶段分配：先还债（free skipped），再数钱（check capacity），最后花（allocate）。** 这个顺序的任意重排都会导致资源泄漏或调度不一致。

---

**下一章：** 第3章 — FlashAttention & PagedAttention：vLLM 的双引擎

KV Cache 解决了"存什么"和"怎么管理显存"。但 attention 计算本身呢？FlashAttention 用 tiled online softmax 把 HBM 访问从 $O(n^2)$ 降到 $O(n)$。PagedAttention 用 block table 把利用率从 25% 提到 96%+。两者正交互补——一个优化"怎么算"，一个优化"怎么存"。第 3 章会把这两个技术放在一起讲：它们的融合 kernel 是 vLLM 最精华的设计。

---

← 第1章 | 第3章 →
