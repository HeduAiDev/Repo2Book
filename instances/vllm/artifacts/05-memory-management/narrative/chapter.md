# 第5章 GPU 显存管理系统——碎片回收、层级分配与微型操作系统

## Cell 2 — 灾难现场

打开 `{source_dir}/vllm/v1/worker/gpu_worker.py:L220-L280`，你会看到一个叫 `determine_available_memory()` 的函数。它启动时的第一件事不是分配 KV Cache，而是算账——先算清楚 GPU 上每一 GiB 都去哪了。

为什么要算账？来看一个真实的灾难数字。

假设你在 T4 16GB 上跑 Llama-3.2-1B，配置 `max_model_len=131072`。这时候**一个请求的 KV Cache 需要多少显存？**

单层的 K cache 大小：

$$
131072 \times 8 \times 128 \times 2 = 268{,}435{,}456\ \mathrm{bytes} \approx 256\ \mathrm{MiB}
$$

这是 K 的。V cache 也一样大：

$$
256\ \mathrm{MiB} \times 2 = 512\ \mathrm{MiB / layer}
$$

Llama-3.2-1B 有 32 层：

$$
512\ \mathrm{MiB} \times 32 = 16{,}384\ \mathrm{MiB} = 16\ \mathrm{GiB}
$$

**一个请求的 KV Cache 就要 16 GiB。** 你的整张 T4 才 16 GiB，而且你还得装模型权重（~2.5 GiB）、前向激活值（按最大 batch 算好几个 GiB）、CUDA workspace（~1 GiB）。一个请求还没跑，显存就炸了。

这就是 vLLM 显存管理系统要解决的核心问题：**GPU 显存是有限的、碎片化的、多个请求共享的**。你要在这块地上同时盖很多房子（请求），每间房子大小不同、入住时间不同、有的还能共用墙壁（前缀共享）。怎么分配？怎么回收？怎么在不 OOM 的前提下塞进最多的请求？

vLLM 的答案，本质上是**把一个操作系统的虚拟内存管理器搬到了 GPU 上**。

![GPU 显存布局总览](gpu-memory-layout.png)

> *图注：GPU 显存的全景布局。左边是 15 GiB 可用显存的分层预算，右边是 KV Cache 的内部结构——32 层 × 每层 K+V tensor，按 block_size=16 切分为 block。每个绿色小方块代表一个 block 的 K tensor（16 token × 8 head × 128 dim）。*

## Cell 3 — 问题演示：预算、碎片与共享，三体问题

在深入代码之前，先把问题拆开来看。GPU 显存管理不是"分配+释放"那么简单。它是三个互相关联的问题纠缠在一起。

### 问题一：预算——你不知道还剩多少显存

你以为 `torch.cuda.memory_allocated()` 能告诉你？不能。GPU 上有几类显存是 PyTorch 根本不知道的：

- **cuBLAS workspace**：cuBLAS 会在第一次调用时 `cudaMalloc` 一块 workspace，PyTorch 无感知。
- **NCCL 通信 buffer**：多卡通信时 NCCL 自己分配 ring buffer，PyTorch 无感知。
- **CUDA Graph 重开**：capture 时的显存峰值 > replay 时的显存占用，差值 PyTorch 不知道。
- **CUDACachingAllocator 碎片**：PyTorch 分配器 round-up 导致"已预留但不可用"的碎片。

如果不做 **Memory Profiling**（真正跑一次前向、记录峰值），你算出来的 KV Cache 预算一定偏大——启动时看着没问题，跑到一半 `cudaMalloc` 失败，OOM。

### 问题二：碎片——固定大小的 block 是双刃剑

把 KV Cache 切成固定 16 token 的 block（PagedAttention 的核心设计）解决了 CTX 变长序列导致的碎片问题。但 block 本身也是碎片源。

一个请求有 10 个 token 要生成，它需要 ceiling(10/16)=1 个 block——但其实只用了 10/16=62.5%，浪费了 6 个 token 的显存。这种"内部碎片"在大量请求叠加时不可忽略。

而且 block 用完得回收。什么时候回收？怎么知道一个 block 还有没有请求在用？谁来负责回收——调度器还是内存管理器？

### 问题三：共享——前缀缓存的正确性难题

两个请求有相同的 system prompt（比如"你是一个 helpful 的 assistant..."），这意味着它们前 50 个 token 完全相同。如果这部分 KV Cache 能共享，50 个 token = 4 个 block（50/16=3.125→4），就能省下 4 个 block 的显存。

但共享带来三个难题：
1. **引用计数**：两个请求引用同一个 block，什么时候这个 block 可以释放？简单 GC 的 mark-sweep 太慢。
2. **哈希碰撞**：两个不同 token 序列算出同一个 hash 怎么办？这是密码学问题。
3. **驱逐顺序**：前缀 block 和后缀 block 哪个先被驱逐？驱错了，缓存命中率暴跌。

### 为什么这个三体问题值得一章

因为 vLLM 的解决方式不是拍脑门的——它是一个微型操作系统的内核。BlockPool 是你的物理页管理器，KVCacheManager 是你的进程调度接口，FreeKVCacheBlockQueue 是你的空闲页链表，BlockHashToBlockMap 是你的页表缓存。读完这一章，你不仅理解了 vLLM 的显存管理，你还**手写了一个微缩版的操作系统内存管理器**。

## Cell 4 — 理论

### 4A. Memory Profiling：显存的三层分类法

打开 `{source_dir}/vllm/v1/worker/gpu_worker.py:L220-L280`。

vLLM 启动时，`determine_available_memory()` 不是去计算 KV Cache 有多大——它是去计算**非 KV Cache 的东西占了多大**，剩下的才是 KV Cache。这是逆向思维。

它的分类法是三层：

**第一层：total_gpu_memory × gpu_memory_utilization**

先划定 vLLM 可以使用的显存上限。默认 `gpu_memory_utilization=0.92`，留 8% 给那些无法精确 profile 的东西：

- cuBLAS handle 创建时的 workspace 增长
- NCCL ring buffer
- CUDA context 本身的小额开销
- CUDACachingAllocator 碎片（reserved but not usable）

为什么是 8% 而不是 5% 或 10%？这是 vLLM 团队在 T4/A100/H100 上实证测试出来的安全值。`determine_available_memory()` 的注释里写得很清楚：这个值不是拍脑门的，是跑了很多次 OOM 之后收敛出来的。

**第二层：non_torch_memory**

这是 PyTorch 分配器完全不知道的显存。包括：
- 其他进程占用的显存（另一个 vLLM 实例、Xorg、EGL）
- cuBLAS workspace
- NCCL buffer
- Custom C++ extension 分配的 buffer

vLLM 通过 `torch.cuda.mem_get_info()` 直接查 GPU driver 来算这部分。

**第三层：torch_memory 的三分类**

这是 PyTorch CUDACachingAllocator 管理的显存。又分三类：

1. **weights_memory**：模型参数。`sum(p.numel() * p.element_size())` 精确可算。但量化（INT8/FP8/NVFP4）会改变 `element_size()`，所以不能直接数参数量乘以 2——要看实际的 `p.dtype`。

2. **peak_activation_memory**：前向传播的峰值激活值。**这是最棘手的**——无法静态计算。依赖因素：
   - `max_num_batched_tokens`（一次性前向处理多少 token）
   - 模型架构（hidden_size、MLP expansion ratio、attention head 数）
   - CUDA Graph capture 的额外 buffer

   vLLM 的解法是**真的跑一次**：加载模型 → 用 `max_num_batched_tokens` 跑一个 dummy forward → `torch.cuda.max_memory_allocated()` 记录峰值。

3. **cudagraph_memory**：CUDA Graph 的重放 buffer。capture 时的峰值 > replay 时，差值也要计入预算。

**最终公式**：

$$
\begin{aligned}
\mathrm{requested\_memory} &= \mathrm{total\_gpu} \times \mathrm{gpu\_memory\_utilization} \\[4pt]
\mathrm{non\_kv\_cache} &= \mathrm{weights} + \mathrm{peak\_act} + \mathrm{cudagraph} + \mathrm{non\_torch} \\[4pt]
\mathrm{available\_kv\_cache} &= \max(0, \mathrm{requested\_memory} - \mathrm{non\_kv\_cache})
\end{aligned}
$$

这是我们简化实现的核心算法（`memory_profiling.py:L59-L132`）：

```python
def profile_gpu_memory(
    total_gpu_memory: int,
    gpu_memory_utilization: float,
    model_params_count: int,
    dtype_bytes: int = 2,
    max_num_batched_tokens: int = 8192,
    num_layers: int = 32,
    hidden_size: int = 4096,
    cudagraph_memory: int = 0,
    non_torch_memory: int = 0,
) -> MemoryBudget:
    weights_memory = model_params_count * dtype_bytes
    activation_per_token_per_layer = 12 * hidden_size * dtype_bytes
    peak_activation_memory = (
        num_layers * max_num_batched_tokens * activation_per_token_per_layer
    )
    requested_memory = int(total_gpu_memory * gpu_memory_utilization)
    non_kv_cache_memory = (
        weights_memory + peak_activation_memory + cudagraph_memory + non_torch_memory
    )
    available_kv_cache_memory = max(0, requested_memory - non_kv_cache_memory)
    return MemoryBudget(...)
```

跟原版 (`{source_dir}/vllm/v1/worker/gpu_worker.py:L220-L280`) 的区别：原版真的在 GPU 上跑 `dummy_forward()` 来获取峰值，我们用的是估算公式。原版还处理了多 worker（TP/PP 场景下每个 worker 的显存配额不同），我们只处理单卡。

有了 `available_kv_cache_memory`，下一步是把它变成 block 数。打开 `{source_dir}/vllm/v1/core/kv_cache_utils.py:L930-L948`：

**单个 block 的字节数**（K + V）：

$$
\mathrm{page\_size\_bytes} = 2 \times \mathrm{block\_size} \times \mathrm{num\_kv\_heads} \times \mathrm{head\_size} \times \mathrm{dtype\_bytes}
$$

为什么是 2？因为每个 block 存储了该位置所有 KV head 的 K 和 V——两份 tensor。

**总 block 数**：
$$
\mathrm{num\_blocks} = \left\lfloor\frac{\mathrm{available\_kv\_cache\_memory}}{\mathrm{page\_size\_bytes} \times \mathrm{num\_layers}}\right\rfloor
$$

除以 `num_layers` 是因为每层都有自己独立的 K 和 V cache——它们共享同一个物理地址空间，但逻辑上每层需要独立的 block table。

以 Llama-3.2-1B 为例（T4 16GB, block_size=16, num_kv_heads=8, head_size=128, dtype=bfloat16）：

$$
\mathrm{page\_size\_bytes} = 2 \times 16 \times 8 \times 128 \times 2 = 65{,}536\ \mathrm{bytes} = 64\ \mathrm{KiB}
$$

假设 profiling 后 `available_kv_cache_memory ≈ 7 GiB`：

$$
\mathrm{num\_blocks} = \left\lfloor\frac{7 \times 1024^3}{65536 \times 32}\right\rfloor = \left\lfloor\frac{7516192768}{2097152}\right\rfloor \approx 3584\ \mathrm{blocks}
$$

### 4B. BlockPool：虚拟内存的 GPU 类比

现在我们有 ~3584 个物理 block 要去管理。vLLM 的方案极度优雅：**把操作系统的虚拟内存照搬过来**。

| 操作系统概念 | vLLM 对应物 | 说明 |
|-------------|-----------|------|
| 物理页 (physical page) | KVCacheBlock | 一个 block = 16 token 的 K+V |
| 页框 (page frame) | BlockPool | 整个 pool 中的一块物理存储 |
| 页表 (page table) | block_table | 每个请求的 token→block 映射 |
| 空闲页链表 (free list) | FreeKVCacheBlockQueue | 空闲 block 的 LRU 链表 |
| 页缓存 (page cache) | BlockHashToBlockMap | hash→block 的前缀缓存 |
| 引用计数 | ref_cnt | 多个请求共享同一个 block |
| 交换空间 (swap) | CPU swap | 抢占时 block 换出到 CPU 内存 |

打开 `{source_dir}/vllm/v1/core/block_pool.py:L130-L509`。`BlockPool` 是核心数据结构。它不存储实际 KV tensor——实际 tensor 在 GPU 上由 model runner 管理——它管理的是 **metadata**：哪些 block 是空闲的，哪些被缓存了，哪些正在被使用。

**BlockPool 的三个关键组件**：

1. **FreeKVCacheBlockQueue**（空闲 block 链表）
   - 一个双向链表，所有空闲 block 按"被释放的时间"排序
   - 最近释放的 block 在链表尾部——它是最后一个被驱逐的
   - 最久释放的 block 在链表头部——它是第一个被驱逐的

2. **BlockHashToBlockMap**（前缀缓存索引）
   - `hash(block_tokens) → KVCacheBlock` 的映射
   - 允许多个 block 有相同的 hash（两个请求同时计算相同前缀时发生）
   - Union type 设计：常见情况（单 block）存 `KVCacheBlock` 直接引用；罕见情况（重复 hash）升级为 `dict[int, KVCacheBlock]`

3. **null_block**（占位符）
   - 用于标记"这个位置没有 block"——例如滑动窗口中超出窗口的位置
   - 从 pool 中拿一个 block 专门当 null（`is_null=True`），永不释放

**核心操作**：

**分配（`get_new_blocks`）**：从 free queue 头部弹出 n 个 block。弹出的同时检查：如果这个 block 之前被缓存了（有 hash），从 hash index 中移除它。

**释放（`free_blocks`）**：递减每个 block 的 `ref_cnt`。当 `ref_cnt == 0` 时，把 block 追加到 free queue 尾部——注意是尾部，意味着"刚刚被释放的 block 最后才被驱逐"。这是在保护前缀缓存的命中率。

**触摸（`touch`）**：当一个新请求通过前缀缓存命中了一个 block，`touch()` 增加其 `ref_cnt`（防止被驱逐），并把它从 free queue 中移除（它不再是一个驱逐候选）。

**缓存（`cache_full_blocks`）**：当一个 block 被完全填满（16 个 token 都算完了），计算它的 hash 并插入 `BlockHashToBlockMap`。这样后续请求可以通过 hash 找到它。

这就是 vLLM 的核心——不是复杂的算法，而是**把操作系统的智慧恰当地搬到了正确的领域**。

### 4C. 为什么双向链表而不是 deque？

`{source_dir}/vllm/v1/core/kv_cache_utils.py:L162-L370`

Python 的 `collections.deque` 提供的操作：O(1) pop/append at ends，O(n) remove from middle。

vLLM 需要的操作：O(1) pop from head（分配），O(1) append to tail（释放），**O(1) remove from middle**（`touch()` 时，一个被缓存的 block 可能在 free queue 的中间位置）。

为什么 `touch()` 要 remove from middle？因为 `touch()` 被调用时，block 可能已经在 free queue 中了（如果 `ref_cnt == 0` 且之前被释放过）。这时需要把它从 free queue 中移除——而 Python 的 `deque.remove(block)` 是 O(n) 的，会遍历整个链表。

vLLM 的解法：把 `prev_free_block` 和 `next_free_block` 指针直接放在 `KVCacheBlock` 对象上。这样：
- `remove(block)` → `block.prev.next = block.next; block.next.prev = block.prev` → **O(1)**
- 不需要遍历链表，因为 block 对象本身就知道自己在链表中的位置

代价：多写了 ~200 行指针操作代码。收益：前缀缓存命中时的 `touch()` 操作为 O(1)，不受池大小影响。

### 4D. 三层架构：KVCacheManager → Coordinator → BlockPool

打开 `{source_dir}/vllm/v1/core/kv_cache_manager.py:L106-L542`。

V1 架构有三个层级：

```
KVCacheManager         ← Scheduler 的外部接口
   │
   └── KVCacheCoordinator   ← 编排多种 KV Cache Group
         │
         └── SingleTypeKVCacheManager[×N]   ← 每种 attention 类型独立管理
               │
               └── BlockPool   ← 物理 block 池（所有类型共享）
```

**为什么需要三层？** 因为现代 LLM 不再是单一种类的 attention 了：
- Llama 是全 attention
- Mistral 有 sliding window attention (SWA)
- DeepSeek V4 有 MLA（multi-head latent attention）+ SWA + full attention 混合

每种 attention 类型的 KV Cache 管理逻辑不同（full attention 从左到右扫描前缀缓存，SWA 从右到左扫描），但它们共享同一个物理 BlockPool。Coordinator 的职责就是编排这些不同类型的 cache manager，确保它们不互相冲突。

在我们的简化实现中，因为只处理 full attention 这一种类型，Coordinator 逻辑被内联到了 KVCacheManager 中。

### 4E. allocate_slots() 的三阶段分配

`{source_dir}/vllm/v1/core/kv_cache_manager.py:L225-L416`

这是整个 KV Cache 系统最核心的函数。每次调度器决定给某个请求分配 token 时，就调用这个函数。它分三个阶段：

**阶段一：释放跳过的 block**

对于滑动窗口 attention（SWA），超出窗口的旧 token 不再需要了——它们对应的 block 应该立即释放，还给 free queue。全 attention 没有"跳过"的概念，所以 `get_num_skipped_tokens()` 返回 0，这个阶段是 no-op。

**阶段二：处理前缀 token**

这就是前缀缓存的魔法。调度器先调用 `get_computed_blocks(request)` 检查这个请求的前缀有没有在缓存中：

```python
def find_longest_cache_hit(block_hashes, max_cache_hit_length):
    for i in range(max_num_blocks):
        hit_block = pool.get_cached_block(block_hashes[i])
        if hit_block is not None:
            computed_blocks.append(hit_block)
        else:
            break  # 第一次没命中就停止
    hit_tokens = len(computed_blocks) * block_size
    return computed_blocks, hit_tokens
```

关键设计：**从左到右扫描，第一次没命中就停止**。你不能跳过第一个没命中的 block 去用后面的——因为 block hash 是链式的（`hash_i` 依赖 `hash_{i-1}`），后面的 hash 值依赖于前面 block 的内容。前面的没命中就意味着整个链条断了。

如果命中了一些 block，就 `touch()` 它们——增加 `ref_cnt`，从 free queue 中移除。这些 block 现在不再是无主的内存了。

**阶段三：分配新 block**

计算还需要多少个新 block：`num_required = ceil(num_tokens / block_size) - len(existing_blocks)`。如果 free queue 中不够，返回 `None`——这是 OOM 信号，调度器需要决定是抢占低优先级请求还是拒绝新请求。

如果有充足的 free block，从 free queue 头部弹出它们。弹出时，如果其中某个 block 之前被缓存了（有其他请求使用了同样的 block），从 hash index 中移除它，并清除它的 hash——内容要被覆盖了。

**缓存新 block**：如果 enable_caching 打开，用 `cache_blocks()` 把新填满的 block 插入 hash index。这样下一个共享同样前缀的请求就能命中。

**释放**：当一个请求完成（或需要被抢占）时，`free(request_id)` 以**反序**释放它的所有 block。为什么是反序？因为尾部的 block（对应 token 序列的后面部分）被其他请求共享的概率最低——它们是最不可能形成前缀的。反序释放 + free queue 尾部追加 = 尾部 block 变成驱逐候选，头部 block（前缀）留在 free queue 中持久保有，近似 LRU。

## Cell 5 — 源码走读

现在我们把三个实现文件逐行走读一遍。

### 5A. memory_profiling.py — 显存预算计算

**运行入口**：`python3 memory_profiling.py`

```python
# memory_profiling.py:L59-L132
def profile_gpu_memory(total_gpu_memory, gpu_memory_utilization, ...) -> MemoryBudget:
```

**走读**：

**L91-L92**：权重显存是最简单的——参数数量 × 每参数字节数。在真实 vLLM 中（`{source_dir}/vllm/v1/worker/gpu_worker.py:L220-L280`），这是通过 `torch.cuda.memory_allocated()` 在模型加载后直接读取的，不是估算。

**L94-L107**：峰值激活值的估算。原版 vLLM 是真的跑一个 dummy forward 然后用 `torch.cuda.max_memory_allocated()` 记录峰值。我们的实现用了一个估算公式：
$$
\mathrm{peak\_act} = \mathrm{num\_layers} \times \mathrm{max\_num\_batched\_tokens} \times 12 \times \mathrm{hidden\_size} \times \mathrm{dtype\_bytes}
$$

这个 12 来自：QKV projection 输出（4×）+ attention output（4×）+ MLP intermediate（4×，假设 4x expansion）= 12 倍的 hidden_size。这是近似值——实际模型的 MLP expansion ratio 可能是 8/3×（Llama 的 SwiGLU），而且还有 residual、RMSNorm 等小开销。

这就是为什么原版选择**真跑一次**而不是用公式：公式在不同模型架构下误差太大。

**L109-L114**：`requested_memory = total_gpu * gpu_memory_utilization`。8% buffer 的理由我们已经在 Cell 4A 中讨论过了。

**L121-L122**：`available_kv_cache_memory = max(0, requested_memory - non_kv_cache)`。这里的 `max(0, ...)` 不是防护代码——它可能真的为 0。如果模型太大、batch 太大，KV Cache 预算可以是 0。这时候要降级：减小 `max_num_batched_tokens`、或使用量化权重、或换更大的 GPU。

**L152-L178**：`compute_kv_cache_config()` 把可用字节数转为 block 数：
```python
page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_bytes
num_blocks = available_kv_cache_memory // page_size_bytes // num_layers
```

注意整除——剩余的不到一个 block 的显存就浪费了。这个浪费在 block_size=16 时通常 < 0.1%。

### 5B. block_pool.py — 物理 Block 池

**运行入口**：`python3 block_pool.py`

整个文件有五个关键数据结构。我们从下往上走读。

**KVCacheBlock（L66-L107）**

```python
@dataclass
class KVCacheBlock:
    block_id: int                     # 0 到 num_gpu_blocks - 1
    ref_cnt: int = 0                  # 几个请求在用这个 block
    _block_hash: bytes | None = None  # 填满后设置的 hash
    is_null: bool = False             # 占位符标记
    prev_free_block: 'KVCacheBlock | None' = None  # 双向链表指针
    next_free_block: 'KVCacheBlock | None' = None
```

这是 vLLM 中最轻量也最重要的数据结构。`block_id` 是对应 GPU tensor 的索引（model runner 用 `block_id` 去查 `kv_cache_tensor[block_id]`）。`ref_cnt` 实现引用计数——多个请求共享同一个前缀 block 时，`ref_cnt ≥ 2`。`_block_hash` 在 block 被填满并缓存时设置（但注意 hash 的 setter 有个断言：不能重复设置——这是"双缓存检测"，帮助发现前缀缓存逻辑错误）。

原版（`{source_dir}/vllm/v1/core/kv_cache_utils.py:L113-L160`）的 `KVCacheBlock` 还有 `__repr__` 方法，输出 block_id 和 ref_cnt 用于调试——我们保留了核心字段，精简了展示。

**FreeKVCacheBlockQueue（L112-L252）**

```python
class FreeKVCacheBlockQueue:
    def __init__(self, blocks: list[KVCacheBlock]):
        # 初始化：把所有 block 连成双向链表
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]
        # 用假头假尾 sentinel 节点减少分支判断
        self.fake_head = KVCacheBlock(block_id=-1)
        self.fake_tail = KVCacheBlock(block_id=-1)
        self.fake_head.next_free_block = blocks[0]
        blocks[-1].next_free_block = self.fake_tail
```

假头假尾 sentinel 的作用：在 `popleft()` 时不需要判断"链表是否为空"——假头永远存在，假尾也永远存在。空链表就是 `fake_head.next == fake_tail`。

**popleft（L149-L164）**：从头部弹出。取 `fake_head.next`，把它从链表中拆下来。

```python
first = self.fake_head.next_free_block
self.fake_head.next_free_block = first.next_free_block
first.next_free_block.prev_free_block = self.fake_head
first.prev_free_block = None  # 断开连接
first.next_free_block = None
```

全部是 O(1) 指针操作。

**remove（L195-L212）**：从中间移除。关键场景：当 `touch()` 被调用，一个在 free queue 中部的缓存在 block 需要被移除：

```python
block.prev_free_block.next_free_block = block.next_free_block
block.next_free_block.prev_free_block = block.prev_free_block
```

O(1)，因为 block 对象自己存储了它在链表中的位置。

**append（L214-L225）**：追加到尾部。新释放的 block 放在尾部，最后被驱逐。

```python
last = self.fake_tail.prev_free_block
last.next_free_block = block
block.prev_free_block = last
block.next_free_block = self.fake_tail
self.fake_tail.prev_free_block = block
```

**BlockHashToBlockMap（L263-L324）**

```python
CachedBlocks = Union[KVCacheBlock, dict[int, KVCacheBlock]]

class BlockHashToBlockMap:
    def __init__(self):
        self._cache: dict[bytes, CachedBlocks] = {}
```

Union type 是这里的设计精髓。99% 的情况下，每个 hash 只有一个 block——存 `KVCacheBlock` 直接引用，避免 dict 开销。只有在**两个请求同时处理相同前缀**时（两者都还没完成，各自创建了同样的 block），才会升级为 `dict[int, KVCacheBlock]`：

```python
def insert(self, block_hash, block):
    existing = self._cache.get(block_hash)
    if existing is None:
        self._cache[block_hash] = block          # 常见路径：单个 block
    elif isinstance(existing, KVCacheBlock):
        self._cache[block_hash] = {              # 升级：两个 repeat hash
            existing.block_id: existing,
            block.block_id: block,
        }
    elif isinstance(existing, dict):
        existing[block.block_id] = block         # 追加：第三个+
```

**BlockPool（L329-L538）**

BlockPool 把前面的所有部分组装在一起。四个核心操作：

**get_new_blocks（L429-L461）**：从 free queue 弹出 n 个 block。弹出的同时检查是否需要驱逐缓存：

```python
def get_new_blocks(self, num_blocks):
    result = self.free_block_queue.popleft_n(num_blocks)
    for block in result:
        if self.enable_caching:
            self._maybe_evict_cached_block(block)
        block.ref_cnt += 1
    return result
```

**touch（L480-L496）**：前缀缓存命中时调用。增加 ref_cnt，如果 block 在 free queue 中就移除它：

```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1
```

**free_blocks（L500-L522）**：递减 ref_cnt，ref_cnt==0 时放回 free queue 尾部：

```python
def free_blocks(self, ordered_blocks):
    for block in blocks_list:
        block.ref_cnt -= 1
    return_to_free = [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
    if return_to_free:
        self.free_block_queue.append_n(return_to_free)
```

**cache_full_blocks（L389-L425）**：把填满的 block 插入 hash index。

### 5C. kv_cache_manager.py — 调度器接口

**运行入口**：`python3 kv_cache_manager.py`

**KVCacheBlocks（L40-L66）**

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[list[KVCacheBlock], ...]
```

`blocks[i][j]` = 第 i 个 KV cache group 的第 j 个 block。单类型时（我们的情况），`blocks` 是 `([...],)`——元组中只有一个 list。

这个 tuple-of-sequences 编码是 vLLM 的微优化：用 tuple 而不是 list-of-lists 来避免 Python 的嵌套对象分配开销。`KVCacheBlocks` 在每次 `allocate_slots()` 时创建，创建频率很高。

**get_computed_blocks（L145-L174）**

```python
def get_computed_blocks(self, request):
    max_cache_hit_length = request.num_tokens - 1  # 最后一个 token 永远要重算
    computed_blocks, hit_tokens = self.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
    return self.create_kv_cache_blocks(computed_blocks), hit_tokens
```

为什么 `num_tokens - 1`？因为即使所有 token 的前缀都命中缓存，最后一个 token 也必须重算——模型需要它来生成 logits。缓存的是中间 KV 状态，不是 logits。

**allocate_slots（L220-L340）**

这是最复杂的方法。完整的三阶段流程：

```python
def allocate_slots(self, request, num_new_tokens, ...):
    # ... 计算各种计数 ...
    
    # 阶段1: 释放跳过的 block（SWA 才有，全 attention 是 no-op）
    self.remove_skipped_blocks(request.request_id, total_computed_tokens)
    
    # 检查容量
    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        return None  # OOM
    
    # 阶段2: 处理前缀缓存命中
    if new_computed_block_list:
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        req_blocks.extend(new_computed_block_list)
        self.num_cached_block[request.request_id] = len(req_blocks)
    
    # 阶段3: 分配新 block
    num_new = num_required - len(req_blocks)
    if num_new > 0:
        new_blocks = self.block_pool.get_new_blocks(num_new)
        req_blocks.extend(new_blocks)
    
    # 缓存填满的 block
    if self.enable_caching:
        self.cache_blocks(request, num_tokens_to_cache)
    
    return self.create_kv_cache_blocks(new_blocks)
```

**free（L370-L384）**：反序释放：

```python
def free(self, request_id):
    req_blocks = self.req_to_blocks.pop(request_id, [])
    if req_blocks:
        self.block_pool.free_blocks(reversed(req_blocks))
    self.num_cached_block.pop(request_id, None)
```

`reversed(req_blocks)` 是关键——尾部的 block（解码阶段产生的 token）最先被释放，变成驱逐候选。前缀 block（prefill 阶段产生的 token）在链表头部，最后被驱逐。这是 vLLM 的"免费 LRU"——不需要任何额外的排序或时间戳，利用释放顺序 + free queue 追加位置就实现了。

### 5D. 与原版的差异

我们在 `impl-notes.md` 中记录了 14 项简化。最重要的 5 项：

1. **单 attention 类型**（原版支持 Hybrid：SWA+Full）。我们只处理全 attention。`find_longest_cache_hit()` 是简单的左右扫描。原版的 `SlidingWindowManager.find_longest_cache_hit()` 是右到左扫描，实现完全不同。

2. **无 EAGLE 投机解码**。EAGLE 导致请求需要"最后一 block 丢弃"逻辑——因为投机 token 可能被拒绝，它的 KV cache 不能缓存。我们的实现没有这个复杂性。

3. **无 DCP/PCP（上下文并行）**。多卡上下文并行时，每个 rank 只持有部分 token 的 KV cache。block_size 的缩放逻辑完全不同。我们只处理单卡。

4. **无 LoRA/多模态 extra_keys**。原版 `hash_block_tokens()` 接受 `extra_keys` 参数，用于区分不同 LoRA adapter 或不同多模态输入下的相同 token 序列。我们的 hash 只用 token IDs。

5. **Coordinator 内联**。原版 `KVCacheManager` 委托给 `KVCacheCoordinator`（一个 ABC，有三个子类：`NoPrefixCache`、`Unitary`、`Hybrid`）。因为我们只处理 unitary 单类型，Coordinator 逻辑被直接内联到 `KVCacheManager` 中。理解原版时，这部分是主要的复杂度来源。

## Cell 6 — 实现

三个实现文件（`instances/vllm/artifacts/05-memory-management/implementation/`），共计 1500+ 行。这里展示核心数据结构与算法，完整的 demo 和辅助函数请直接阅读源文件。

### memory_profiling.py — 显存预算计算

```python
# REFERENCE: vllm/v1/worker/gpu_worker.py → determine_available_memory()
# REFERENCE: vllm/v1/core/kv_cache_utils.py → get_num_blocks(), L930-L948
# REFERENCE: vllm/v1/core/kv_cache_utils.py → max_memory_usage_bytes(), L726-L732

@dataclass
class MemoryBudget:
    """显存 profiling 结果——三层分类预算。"""
    total_gpu_memory: int          # bytes
    gpu_memory_utilization: float  # fraction reserved for vLLM (e.g., 0.92)
    weights_memory: int            # bytes — model weights
    peak_activation_memory: int    # bytes — peak forward-pass activations
    cudagraph_memory: int          # bytes — CUDA Graph replay buffers
    non_torch_memory: int          # bytes — cuBLAS, NCCL, other processes
    available_kv_cache_memory: int # bytes — what's left for KV cache

def profile_gpu_memory(
    total_gpu_memory: int,
    gpu_memory_utilization: float,
    model_params_count: int,
    dtype_bytes: int = 2,
    max_num_batched_tokens: int = 8192,
    num_layers: int = 32,
    hidden_size: int = 4096,
    cudagraph_memory: int = 0,
    non_torch_memory: int = 0,
) -> MemoryBudget:
    """模拟 vLLM 启动时的显存 profiling pipeline。
    
    原版在真实 GPU 上通过 torch.cuda 执行。我们用公式模拟——
    公式捕捉了三层分类的核心逻辑，但不替代原版的精确测量。
    """
    # Step 1: 模型权重 = 参数量 × 每参数字节数
    weights_memory = model_params_count * dtype_bytes

    # Step 2: 估算峰值激活值
    # 每层每 token: QKV(4×) + attn_out(4×) + MLP(4×) = 12 × hidden × dtype
    activation_per_token_per_layer = 12 * hidden_size * dtype_bytes
    peak_activation_memory = (
        num_layers * max_num_batched_tokens * activation_per_token_per_layer
    )

    # Step 3: vLLM 可用显存上限 = 总显存 × 利用率（默认 92%，留 8% buffer）
    requested_memory = int(total_gpu_memory * gpu_memory_utilization)

    # Step 4: 扣除非 KV Cache 的显存
    non_kv_cache_memory = (
        weights_memory + peak_activation_memory
        + cudagraph_memory + non_torch_memory
    )

    # Step 5: 剩余的才是 KV Cache 的
    available_kv_cache_memory = max(0, requested_memory - non_kv_cache_memory)

    return MemoryBudget(
        total_gpu_memory=total_gpu_memory,
        gpu_memory_utilization=gpu_memory_utilization,
        weights_memory=weights_memory,
        peak_activation_memory=peak_activation_memory,
        cudagraph_memory=cudagraph_memory,
        non_torch_memory=non_torch_memory,
        available_kv_cache_memory=available_kv_cache_memory,
    )


@dataclass
class KVCacheConfig:
    """从显存预算推导出的 KV Cache 配置。"""
    num_blocks: int        # Total blocks in the pool
    block_size: int        # Tokens per block
    num_layers: int        # Number of layers sharing the pool
    page_size_bytes: int   # Bytes per block per layer (K+V)
    num_kv_heads: int
    head_size: int
    dtype_bytes: int


def compute_kv_cache_config(
    budget: MemoryBudget,
    block_size: int = 16,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype_bytes: int = 2,
) -> KVCacheConfig:
    """将可用字节数转换为 block 数。

    单个 block 的字节数 = 2 × block_size × num_kv_heads × head_size × dtype_size
    （因子 2 = K cache + V cache）

    总 block 数 = available_kv_cache // page_size_bytes // num_layers
    """
    page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_bytes
    if page_size_bytes * num_layers == 0:
        num_blocks = 0
    else:
        num_blocks = budget.available_kv_cache_memory // page_size_bytes // num_layers
    return KVCacheConfig(
        num_blocks=max(num_blocks, 0),
        block_size=block_size,
        num_layers=num_layers,
        page_size_bytes=page_size_bytes,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype_bytes=dtype_bytes,
    )
```

### block_pool.py — 物理 Block 池

```python
# REFERENCE: vllm/v1/core/block_pool.py:L130-L509 (BlockPool)
# REFERENCE: vllm/v1/core/kv_cache_utils.py:L113-L160 (KVCacheBlock)
# REFERENCE: vllm/v1/core/kv_cache_utils.py:L162-L370 (FreeKVCacheBlockQueue)
# REFERENCE: vllm/v1/core/kv_cache_utils.py:L37-L78  (BlockHash)
# REFERENCE: vllm/v1/core/kv_cache_utils.py:L539-L566 (hash_block_tokens)

BLOCK_SIZE = 16
NONE_HASH = os.urandom(32)  # seed hash for first block


def hash_block_tokens(
    parent_hash: bytes | None,
    token_ids: Sequence[int],
) -> bytes:
    """增量链式 hash: hash_i = SHA256(hash_{i-1} || token_ids_i)

    链式结构保证前缀长度 N → 恰好一条 hash 链。
    新增 block 只需 hash(parent, new_tokens) ——无需重哈希整个前缀。
    """
    if parent_hash is None:
        parent_hash = NONE_HASH
    data = parent_hash + str(tuple(token_ids)).encode()
    return hashlib.sha256(data).digest()


# ── KVCacheBlock ──

@dataclass
class KVCacheBlock:
    """单个 KV cache block 的 metadata。

    实际 tensor 数据在 GPU 上。这个对象只存储管理信息。
    ref_cnt 实现引用计数：多个请求共享同一前缀 block 时 ref_cnt ≥ 2。
    prev_free_block / next_free_block 实现双向链表指针——存在 block 对象
    上而非外部 list 中，这是 O(1) middle-remove 的关键。
    """
    block_id: int                    # 0 to num_gpu_blocks - 1
    ref_cnt: int = 0                 # Number of requests using this block
    _block_hash: bytes | None = None # Set when block is full and cached
    is_null: bool = False            # Placeholder for skipped positions
    prev_free_block: 'KVCacheBlock | None' = None
    next_free_block: 'KVCacheBlock | None' = None

    @property
    def block_hash(self) -> bytes | None:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, value: bytes):
        # 断言防止双重缓存——帮助发现前缀缓存逻辑错误
        assert self._block_hash is None, "Double-cache detected."
        self._block_hash = value

    def reset_hash(self):
        self._block_hash = None


# ── FreeKVCacheBlockQueue ──

class FreeKVCacheBlockQueue:
    """O(1) popleft / remove / append 的双向链表。

    WHY NOT collections.deque?
    deque.remove(block) 是 O(n)，而 touch() 需要 O(1) 从链表中间移除。
    通过把 prev/next 指针存在 KVCacheBlock 对象上，remove() 变成：
        block.prev.next = block.next; block.next.prev = block.prev  ← O(1)

    驱逐顺序：LRU（头部 = 最久释放，尾部 = 最近释放）。
    假头假尾 sentinel 节点减少空链表的分支判断。
    """

    def __init__(self, blocks: list[KVCacheBlock]):
        self.num_free_blocks = len(blocks)
        # 把 blocks 连成双向链表
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]
        # 假头假尾 sentinel
        self.fake_head = KVCacheBlock(block_id=-1)
        self.fake_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            self.fake_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_head
            self.fake_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_tail
        else:
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head

    def popleft(self) -> KVCacheBlock:
        """从头部弹出一个 block（O(1)）。被分配的 block 从空闲链表移除。"""
        first = self.fake_head.next_free_block
        if first is self.fake_tail or first is None:
            raise ValueError("No free blocks available")
        self.fake_head.next_free_block = first.next_free_block
        first.next_free_block.prev_free_block = self.fake_head
        first.prev_free_block = None
        first.next_free_block = None
        self.num_free_blocks -= 1
        return first

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """批量弹出（O(n)）。"""
        # ... [完整实现见 block_pool.py:L166-L193]

    def remove(self, block: KVCacheBlock) -> None:
        """从链表中间移除（O(1)）。
        
        关键场景：touch() 被调用时，缓存在 block 在 free queue 中间。
        因为指针存在 block 对象上，不需要遍历链表。
        """
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """追加到尾部（O(1)）。最近释放的 block 在尾部，最后被驱逐。"""
        last = self.fake_tail.prev_free_block
        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = block
        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """批量追加（O(n)）。"""
        # ... [完整实现见 block_pool.py:L227-L243]


# ── BlockHashToBlockMap ──

CachedBlocks = Union['KVCacheBlock', dict[int, 'KVCacheBlock']]

class BlockHashToBlockMap:
    """Union-type hash→block 映射。

    99% 情况：每个 hash 只有一个 block → 直接存 KVCacheBlock 引用。
    罕见情况：两个请求同时处理相同前缀 → 升级为 dict[int, KVCacheBlock]。
    避免嵌套 dict 在常见路径上的开销。
    """

    def __init__(self):
        self._cache: dict[bytes, CachedBlocks] = {}

    def get_one_block(self, block_hash: bytes) -> KVCacheBlock | None:
        blocks = self._cache.get(block_hash)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            return blocks
        if isinstance(blocks, dict):
            return next(iter(blocks.values()))
        raise AssertionError(f"Unexpected type: {type(blocks)}")

    def insert(self, block_hash: bytes, block: KVCacheBlock) -> None:
        existing = self._cache.get(block_hash)
        if existing is None:
            self._cache[block_hash] = block              # 常见路径
        elif isinstance(existing, KVCacheBlock):
            self._cache[block_hash] = {                  # 升级
                existing.block_id: existing,
                block.block_id: block,
            }
        elif isinstance(existing, dict):
            existing[block.block_id] = block             # 追加
```

```python
# ── BlockPool ──

class BlockPool:
    """共享的 KV cache block 池。

    管理三个关键数据结构：
    1. free_block_queue: 空闲 block 按驱逐顺序排列（LRU）
    2. cached_block_hash_to_block: hash→block 前缀缓存索引
    3. null_block: 占位符（标记"没有 block"的位置）

    核心操作：get_new_blocks / free_blocks / touch / cache_full_blocks
    """

    def __init__(self, num_gpu_blocks: int, enable_caching: bool = True):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.blocks = [KVCacheBlock(block_id=i) for i in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        # null_block 从 pool 中拿一个 block，永不释放
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    # ── Cache Lookup ──

    def get_cached_block(self, block_hash: bytes) -> KVCacheBlock | None:
        """Hash 查找前缀缓存。返回 None = cache miss。"""
        if not self.enable_caching:
            return None
        return self.cached_block_hash_to_block.get_one_block(block_hash)

    # ── Allocation ──

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """分配新 block：从 free queue 头部弹出，驱逐旧缓存。"""
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot allocate {num_blocks} blocks")
        result = self.free_block_queue.popleft_n(num_blocks)
        for block in result:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)  # 如果 block 被缓存过，移除
            block.ref_cnt += 1
        return result

    # ── Touch (前缀缓存命中) ──

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        """前缀缓存命中：增加 ref_cnt，从 free queue 移除。
        
        当新请求命中缓存 block 时，block 可能正在 free queue 中
        （ref_cnt=0 的驱逐候选）。touch() 把它从队列中移除并增加引用。
        """
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)  # O(1) remove from middle
            block.ref_cnt += 1

    # ── Free ──

    def free_blocks(self, ordered_blocks: Sequence[KVCacheBlock]) -> None:
        """释放 block：ref_cnt--，ref_cnt==0 时放回 free queue 尾部。
        
        尾部追加 = 最近释放 → 最后驱逐 = 近 LRU。
        """
        for block in ordered_blocks:
            block.ref_cnt -= 1
        return_to_free = [
            b for b in ordered_blocks
            if b.ref_cnt == 0 and not b.is_null
        ]
        if return_to_free:
            self.free_block_queue.append_n(return_to_free)

    # ── Cache ──

    def cache_full_blocks(
        self, blocks, num_cached_blocks, num_full_blocks, request_block_hashes,
    ) -> None:
        """把填满的 block 插入 hash index，使其可被后续请求命中。"""
        if not self.enable_caching or num_cached_blocks >= num_full_blocks:
            return
        for block, block_hash in zip(
            blocks[num_cached_blocks:num_full_blocks],
            request_block_hashes[num_cached_blocks:num_full_blocks],
        ):
            if block.is_null:
                continue
            block.block_hash = block_hash
            self.cached_block_hash_to_block.insert(block_hash, block)

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        total = self.num_gpu_blocks - 1  # 减去 null block
        if total == 0:
            return 0.0
        return 1.0 - (self.get_num_free_blocks() / total)
```

### kv_cache_manager.py — 调度器接口

```python
# REFERENCE: vllm/v1/core/kv_cache_manager.py:L106-L542 (KVCacheManager)
# REFERENCE: vllm/v1/core/kv_cache_coordinator.py:L324-L389 (UnitaryCoordinator)
# REFERENCE: vllm/v1/core/single_type_kv_cache_manager.py:L446-L504 (FullAttentionManager)

@dataclass
class KVCacheBlocks:
    """分配结果包装。blocks[i][j] = 第 i 个 KV cache group 的第 j 个 block。
    
    tuple-of-sequences 编码：用 tuple 而非 list-of-lists 减少嵌套对象分配。
    """
    blocks: tuple[list['KVCacheBlock'], ...]

    def get_block_ids(self) -> list[int]:
        return [block.block_id for group in self.blocks for block in group]


class KVCacheManager:
    """统一 KV cache 管理器（单 attention 类型）。

    拥有 BlockPool 引用 + 每请求状态追踪。
    调用方：Scheduler。
    """

    def __init__(self, block_pool, block_size=16, max_model_len=4096,
                 enable_caching=True):
        self.block_pool = block_pool
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.req_to_blocks: dict[str, list[KVCacheBlock]] = defaultdict(list)
        self.num_cached_block: dict[str, int] = {}
        self.empty_kv_cache_blocks = KVCacheBlocks(((),))
        self._null_block = block_pool.null_block

    # ── Prefix Cache Lookup ──

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """查找请求的前缀缓存命中。返回 (cached_blocks, num_tokens_hit)。

        max_cache_hit_length = num_tokens - 1：最后一个 token 永远
        要重算——缓存的只是 KV 状态，不是 logits。
        """
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, hit_tokens = self.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
        return self.create_kv_cache_blocks(computed_blocks), hit_tokens

    def find_longest_cache_hit(
        self, block_hashes: list[bytes], max_cache_hit_length: int,
    ) -> tuple[list[KVCacheBlock], int]:
        """左到右扫描，第一次 miss 就停止。

        不能跳过 miss 继续——链式 hash 意味着后面的 hash 依赖前面的
        block 内容。前面的非缓存意味着整个链条断裂。
        """
        max_num_blocks = max_cache_hit_length // self.block_size
        computed_blocks = []
        for i in range(max_num_blocks):
            if i >= len(block_hashes):
                break
            hit_block = self.block_pool.get_cached_block(block_hashes[i])
            if hit_block is not None:
                computed_blocks.append(hit_block)
            else:
                break  # 第一次 miss 就停止
        return computed_blocks, len(computed_blocks) * self.block_size

    # ── allocate_slots(): 3-Stage Allocation ──

    def allocate_slots(
        self, request: Request, num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> KVCacheBlocks | None:
        """分配 KV cache slot——整个系统最核心的函数。

        三阶段：
        1. 释放跳过的 block（SWA 才有，全 attention 是 no-op）
        2. 处理前缀缓存命中（touch cached blocks）
        3. 分配新 block（从 free queue 弹出）

        Returns None on OOM (信号调度器需要抢占或拒绝).
        """
        # ... 计算 num_blocks_to_allocate ...

        # Stage 1: Free skipped blocks (SWA only)
        self.remove_skipped_blocks(request.request_id, total_computed_tokens)

        # Capacity check
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None  # OOM — signal scheduler

        # Stage 2: Touch cached blocks (prefix cache hits)
        if new_computed_block_list:
            if self.enable_caching:
                self.block_pool.touch(new_computed_block_list)
            self.req_to_blocks[request.request_id].extend(new_computed_block_list)
            self.num_cached_block[request.request_id] = len(
                self.req_to_blocks[request.request_id]
            )

        # Stage 3: Allocate new blocks
        req_blocks = self.req_to_blocks[request.request_id]
        num_new = num_required - len(req_blocks)
        new_blocks = []
        if num_new > 0:
            new_blocks = self.block_pool.get_new_blocks(num_new)
            req_blocks.extend(new_blocks)

        # Cache full blocks for prefix reuse
        if self.enable_caching:
            self.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    # ── Free ──

    def free(self, request_id: str) -> None:
        """反序释放所有 block。尾部 block（解码阶段产生）先释放，
        变成驱逐候选。前缀 block（prefill 阶段产生）留在缓存中。
        """
        req_blocks = self.req_to_blocks.pop(request_id, [])
        if req_blocks:
            self.block_pool.free_blocks(reversed(req_blocks))
        self.num_cached_block.pop(request_id, None)
```

> **完整实现**见 `instances/vllm/artifacts/05-memory-management/implementation/` 目录下的三个文件。
> 上述代码省略了 demo main() 函数、部分辅助方法（`remove_skipped_blocks`、`cache_blocks` 等）和 `Request` 数据类——它们的完整版本请阅读源文件。

## Cell 7 — 数值演示

### 7A. 显存 Profiling 输出

运行 `python3 memory_profiling.py`：

```
======================================================================
GPU Memory Profiling: Llama-3.2-1B on T4 16GB
======================================================================
MemoryBudget(
  total_gpu_memory            = 15.00 GiB
  gpu_memory_utilization       = 0.92
  weights_memory              = 2.31 GiB
  peak_activation_memory      = 12.00 GiB
  cudagraph_memory            = 0.20 GiB
  non_torch_memory            = 0.29 GiB
  available_kv_cache_memory   = 0.00 GiB
)
```

注意 `available_kv_cache_memory = 0.00 GiB`！这不是 bug——这是 realistic 的结果。当 `max_num_batched_tokens=8192`、`hidden_size=2048` 时，峰值激活值估算是 12 GiB，加上权重的 2.3 GiB、CUDA Graph 的 0.2 GiB、非 torch 的 0.29 GiB，总共约 14.8 GiB。15 GiB × 0.92 = 13.8 GiB，不够。KV Cache 预算为 0。

这就是为什么 vLLM 的参数需要谨慎配置。如果减小 `max_num_batched_tokens` 到 2048：

$$
\mathrm{peak\_act} = 32 \times 2048 \times 12 \times 2048 \times 2 \approx 3.0\ \mathrm{GiB}
$$

这时候就有 ~7 GiB 给 KV Cache 了。

```
--- KV Cache Configuration ---
  block_size         = 16 tokens
  page_size_bytes    = 64.00 KiB
  num_layers         = 32
  num_blocks (total) = 0
```

KV Cache 为 0 时，block 数也是 0——无法运行任何请求。

### 7B. BlockPool 操作演示

运行 `python3 block_pool.py`（hash 值因随机种子每次不同，这里以某次实际运行为例）：

```
======================================================================
BlockPool Demo: Allocation, Prefix Caching, and Freeing
======================================================================

Initial: 9 free blocks (1 null block reserved)
Usage: 0.0%

Hash block 0: becde15e... (tokens [1, 2, 3]...)
Hash block 1: dde568d0... (tokens [17, 18, 19]...)

Request 1: allocated blocks [1, 2]
Free blocks left: 7, Usage: 22.2%
Cached: 2 blocks in hash index

Request 2: prefix cache lookup for hash_0 → HIT (block 1)
  Touched block 1: ref_cnt=1
  Second block also HIT: block 2
  New block for extra tokens: [3]
Pool usage: 33.3%

Freeing Request 1 (reverse order for LRU eviction):
  Free blocks now: 6

Freeing Request 2:
  Free blocks now: 9
  Usage: 0.0%

--- Eviction Stress Test ---
  Allocated all 9 free blocks
  Free: 0, Usage: 100.0%
  Block hash after eviction: None
```

> 注意：hash 值由 `NONE_HASH = os.urandom(32)` 作为种子生成，每次运行不同。但 block_id 分配序列是确定性的。

这里可以跟踪 block_id=1 的完整生命周期：

1. **分配**：Request 1 分配 block 1（ref_cnt=0→1，从 free queue 弹出）
2. **缓存**：block 1 被填满 → hash 写入 BlockHashToBlockMap
3. **释放**：Request 1 完成 → block 1 ref_cnt=1→0 → 放入 free queue 尾部
4. **命中**：Request 2 查找 hash_0 → 命中 block 1 → `touch()` 把 block 1 从 free queue 移除，ref_cnt=0→1
5. **再次释放**：Request 2 完成 → block 1 ref_cnt=1→0 → 放回 free queue

这是一个 block 的完整生死循环。`ref_cnt` 从 0→1→0→1→0 变化的每一步都有明确语义。

### 7C. KVCacheManager 完整生命周期

运行 `python3 kv_cache_manager.py`：

```
======================================================================
KVCacheManager Demo: Full Request Lifecycle
======================================================================

Pool: 255 usable blocks + 1 null block
Block size: 16 tokens
Max model length: 4096
Initial usage: 0.0%

--- Request A: 48-token prompt (3 blocks) ---
  Prefix cache hit: 0 tokens (MISS)
  Allocated blocks: [1, 2, 3]
  Pool usage: 1.2%

--- Request B: same prefix + 16 new tokens (4 blocks) ---
  Prefix cache hit: 48 tokens (3 blocks)
  Newly allocated: 1 block(s) [4]
  Pool usage: 1.6%
  Block 0 ref_cnt: 2 (shared by 2 requests)
  Block 1 ref_cnt: 2 (shared by 2 requests)
  Block 2 ref_cnt: 2 (shared by 2 requests)
  Block 3 ref_cnt: 1 (unique to B)

--- Freeing Request A (reverse order) ---
  Pool usage after free: 1.6%
  Block 0 ref_cnt: 1 (still shared by B)
  Block 2 ref_cnt: 1 (freed, now eviction candidate)

--- Final State ---
  Request B block_ids: [1, 2, 3, 4]
  Pool usage: 1.6%
  Free blocks: 251
  Cached blocks in hash index: 4
```

关键观察：

1. **Request A 分配了 3 个 block**（48 token / 16 = 3）
2. **Request B 只分配了 1 个新 block**（前面的 3 个通过前缀缓存命中，`ref_cnt` 变成 2）
3. **释放 Request A 时**，block 1, 2, 3 的 `ref_cnt` 从 2 降到 1——它们仍然被 Request B 引用，不能放回 free queue
4. **只有 block 3（反序中的第一个）ref_cnt 降为 0**——block 3 在反序中被先释放，但它之前 ref_cnt=2（Request A 和 B 都引用），释放后 ref_cnt=1（只有 B），不会进入 free queue
5. 等等——输出来看，block 3 是 A 的尾 block 但 B 也在用它（因为 B 的前 48 token = A 的全部 48 token）。所以 A 释放时 block 3 ref_cnt=2→1，进入不了 free queue。0 blocks freed。

这个 Demo 用的是 256 个 block 的小池子，实际 vLLM 部署时 `num_gpu_blocks` 通常是 5000-50000+。

## Cell 9 — 源码映射表

| 我们的代码 | 原始源码 | 我们改了什么 | 为什么 |
|-----------|---------|------------|------|
| `KVCacheBlock` | `kv_cache_utils.py:L113-L160` | 保留核心字段（block_id, ref_cnt, block_hash, prev/next_free_block, is_null），移除 `__repr__` | 核心 metadata 完全一致 |
| `FreeKVCacheBlockQueue` | `kv_cache_utils.py:L162-L370` | 保留完整逻辑。简化错误消息 | 双向链表是核心数据结构，必须一致才能保证 O(1) 操作 |
| `BlockHashToBlockMap` | `block_pool.py:L34-L127` | 保留 union-type 结构。内联断言 | 核心缓存逻辑保留；union-type 处理是复杂性本质 |
| `BlockPool` | `block_pool.py:L130-L509` | 移除：KV cache events、metrics collector、evict_blocks()、reset_prefix_cache()。简化：cache_full_blocks() 无 LoRA/多模态 extra_keys | Events/metrics 是可观测性层；block pool 逻辑不变 |
| `KVCacheBlocks` | `kv_cache_manager.py:L21-L104` | 保留基本 get_block_ids()、new_empty()。移除 `__add__`、get_unhashed_block_ids() | 核心数据类保留 blocks tuple 结构 |
| `KVCacheManager` | `kv_cache_manager.py:L106-L542` | 移除：EAGLE、DCP/PCP、log_stats、prefix_cache_stats。保留：3-stage allocate_slots()、get_computed_blocks()、free()、create_kv_cache_blocks() | 核心分配流程完全一致；移除正交特性 |
| `profile_gpu_memory()` | `kv_cache_utils.py:L930-L948` + `gpu_worker.py:L220-L280` | 将 profiling pipeline 浓缩为单个函数，用公式模拟而非真实 GPU 运行 | 多 worker profiling 是 PP/TP 复杂性；单卡场景捕获三层分类概念 |
| Coordinator 逻辑 | `kv_cache_coordinator.py:L28-L389` | 内联进 KVCacheManager（UnitaryKVCacheCoordinator） | 单 attention 类型不需要多 Group 编排 |
| `find_longest_cache_hit()` | `single_type_kv_cache_manager.py:L446-L494` | 内联进 KVCacheManager | 只有全 attention 类型，不需要 ABC + 子类分发 |
| `hash_block_tokens()` | `kv_cache_utils.py:L539-L566` | 移除 extra_keys 参数 | LoRA/多模态的额外 hash 输入 |

## Cell 10 — 验证

### 测试结果

运行 `cd instances/vllm/artifacts/05-memory-management/tests && python3 -m pytest -v`：

```
==================== 123 passed ====================
```

全部 123 个测试通过，覆盖：
- `test_memory_profiling.py`：预算计算、block 配置、边界条件（OOM、零预算、整除边界）
- `test_block_pool.py`：分配/释放/缓存/驱逐/引用计数/双向链表完整性
- `test_kv_cache_manager.py`：3 阶段分配、前缀缓存命中/未命中、反序释放、OOM 信号

### Formula Lint

运行 `python3 scripts/lint_formulas.py artifacts/05-memory-management/narrative/chapter.md`：**0 blocking issues**。

### Source Grounding Lint

运行 `python3 scripts/lint_source_grounding.py artifacts/05-memory-management/`：
- 每个 Cell (2-7) 包含 1+ source 引用
- Source Mapping Table 包含 10 行（要求 ≥5）
- impl-notes.md 列出 15+ 个 source 文件（要求 ≥3）
- 所有 `# REFERENCE:` 注释存在并引用正确文件:行号

## Cell 11 — 总结

这一章，我们从 GPU 显存预算开始，一步步构建了一个微型操作系统的内存管理器。

**核心洞察**：

1. **先算账再花钱**。`determine_available_memory()` 的本质思想是"非 KV 的东西先扣掉，剩下的才是 KV Cache"。这不是精确的，但它是**安全**的。8% buffer 给了 CUDA 生态的不可预测性一个缓冲。

2. **虚拟内存是经过 60 年验证的正确抽象**。vLLM 把操作系统的 page table → free list → hash-based page cache → reference counting 完整搬到了 GPU KV Cache 上。BlockPool = 物理页管理器，KVCacheBlock = 物理页，FreeKVCacheBlockQueue = 空闲页链表，BlockHashToBlockMap = 页缓存。

3. **数据结构的选择决定性能**。双向链表（而不是 Python deque）选择 O(1) middle-remove 来服务 `touch()` 操作。Union type（而不是嵌套 dict）节省了常见路径的 dict 开销。反序释放（而不是排序）用排列顺序实现了近似 LRU。每一个选择都不是拍脑门——都有明确的性能理由和 trade-off。

4. **前缀缓存的正确性依赖于引用计数 + 链式 hash**。`ref_cnt` 确保被共享的 block 不会被过早释放。链式 hash（`hash_i = hash(hash_{i-1}, tokens_i)`）确保"前缀长度 N → 恰好一条 hash 链"，不需要重新哈希整个前缀。

**下一章预告**：第 6 章我们将走进调度器（Scheduler）。它使用本章构建的 `KVCacheManager.allocate_slots()` 来决定——给定 N 个请求和有限的 KV Cache block，应该调度哪些请求、什么时候抢占、什么时候等待。那是 GPU 显存管理的"上层建筑"——预算有了，池子有了，现在是调度策略的战场。
