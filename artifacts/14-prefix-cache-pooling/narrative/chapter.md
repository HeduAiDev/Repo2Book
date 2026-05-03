# 第13章：Prefix Cache Pooling — 全局共享池

> 第 7 章的 Prefix Cache 是单请求视角：一个请求缓存了 system prompt，另一个请求命中。
> 但全局视角下——1000 个并发请求，50 个不同的 system prompt——Prefix Cache 不再只是一个
> hash table。它是一个**全局共享池**——有 admission policy（什么值得进）、
> eviction policy（谁被驱逐）、和 hash chain 的正确性保证（没有假阳性命中）。

---

## 这章要做什么？

打开 `vllm/v1/core/block_pool.py:130`。`BlockPool` 拥有整个 engine session 的**所有** GPU KV cache blocks。它的 `cached_block_hash_to_block`（L171）是一个全局 hash map——**任何**请求的 block 缓存后，**任何其他**请求都能命中。

这不是 per-request cache。这是 per-engine-session global pool。区别在三个维度上：

| | Per-Request Cache | Global Pool (vLLM) |
|---|---|---|
| **所有权** | 请求拥有缓存，请求结束 = 缓存失效 | BlockPool 拥有，跨请求、跨会话共享 |
| **命中范围** | 同一请求的后续 step | 任何请求、任何时间 |
| **驱逐** | 跟着请求一起消亡 | LRU 驱逐——全局池满时才发生 |

学完这章你能：
- 解释为什么 `BlockPool` 不需要锁——单线程 Scheduler 的 sequential 调度保证了线程安全
- 理解 `touch()` 如何让共享 block 不被驱逐——`ref_cnt > 0` 意味着 block 不在 free queue 中
- 理解链式 hash 的正确性证明——为什么 `find_longest_cache_hit` 在第一个 miss 处可以安全停止

---

## 13.1 全局池 vs 请求内缓存

### Source Trail

打开 `vllm/v1/core/block_pool.py:162-171`：

```python
self.blocks: list[KVCacheBlock] = [
    KVCacheBlock(idx) for idx in range(num_gpu_blocks)
]
self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```

`BlockPool` 在 engine 启动时创建——只有**一个实例**。它管理所有 GPU blocks 的整个生命周期：分配、缓存、共享、驱逐。

每个请求通过 `KVCacheManager`（第 2 章）和 `SingleTypeKVCacheManager` 与 pool 交互：

- `alloc_new_blocks(n)` — 从 free queue 取 LRU block，驱逐缓存
- `cache_full_blocks()` — 标记 block 已满→可共享
- `lookup(hash)` — O(1) 全局 hash table 查找

### Theory: 为什么单线程就够了？

`BlockPool` 没有锁——没有 `threading.Lock`、没有 `asyncio.Lock`。vLLM v1 engine 使用**单线程事件循环**。Scheduler 在每一步中 sequential 处理所有请求——先 running、后 waiting。Block 的分配和释放在这个 sequential 上下文中完成。

多线程不安全？不需要——因为没有多线程。这在 vLLM 的设计中是刻意的：**用单线程的简单性换取并发控制的复杂性**。代价是 Scheduler 的计算必须在 GPU forward 之间完成（<1ms）。收益是 Pool 的零锁开销——`ref_cnt` 的生命周期管理完全不需要原子操作。

---

## 13.2 全局 Hash Index：碰撞 vs 去重

### Source Trail

打开 `vllm/v1/core/block_pool.py:34-127`——`BlockHashToBlockMap`。

```python
class BlockHashToBlockMap:
    _cache: dict[BlockHashWithGroupId,
                 KVCacheBlock | dict[int, KVCacheBlock]]
```

**两种状态：**

```python
# 正常：一个 hash → 一个 block
_cache[hash_abc] = block_42

# 碰撞：两个不同物理 block 有相同的 hash → 升级为 dict
_cache[hash_abc] = {42: block_42, 87: block_87}

# get_one_block() 返回任何一个——对于前缀缓存命中来说足够了
```

### Theory: 为什么不 deduplicate？

如果两个 block 有相同的内容（相同 hash），为什么不只保留一个，让后来者指向它？这样能省下一份物理空间。

答案在 `block_pool.py:48-54` 的注释中：**block_table 是 append-only。** 每个请求的 block_table 是一列物理 block ID——这些 ID 不能修改。如果 deduplicate 了 block 87（把它删掉，指向 block 42），需要修改所有引用 block 87 的 block_table——GPU kernel 正在读这些 block_table。这是需要读写同步的——而 vLLM 选择不要。

`★ Insight ─────────────────────────────────────`
vLLM 的 no-deduplication 策略体现了一个更深层的设计原则：**在 GPU 计算的 fast path 和 CPU 管理的 slow path 之间保持接口稳定。** block_table 是 fast path——CUDA kernel 以只读方式访问它。修改 block_table（去重）需要 slow path 和 fast path 之间的同步——要么等 kernel 完成再修改（引入延迟），要么加锁（引入争用）。vLLM 的选择：允许冗余，换取 fast path 的零同步开销。对于一个 hash table 条目来说，冗余的物理 block 代价是 block_size × 2 × head_dim × dtype_bytes = 8 KB（bf16, block_size=16）。对于 80 GB 的 H100——这是可以接受的。
`─────────────────────────────────────────────────`

---

## 13.3 链式 Hash 的正确性

### Source Trail

打开 `vllm/v1/core/kv_cache_utils.py:539`：

```python
def hash_block_tokens(hash_fn, parent_hash, tokens, extra_keys):
    return hash_fn((parent_hash, tuple(tokens), extra_keys))
```

**链式性质：** $H_k$ 依赖 $H_{k-1}$。如果 $H_k$ 命中，则 $H_{k-1}$ 一定在 pool 中。

### Theory: 为什么左到右扫描是安全的？

`FullAttentionManager.find_longest_cache_hit()`（`single_type_kv_cache_manager.py:448`）执行左到右扫描：

```python
for block_hash in block_hashes:
    if cached_block := pool.get_cached_block(block_hash):
        hits.append(cached_block)
    else:
        break  # ← 不需要继续扫描!
```

**正确性证明：** 假设 $H_k$ miss，但 $H_{k+1}$ 命中。根据链式 hash 定义：

$$
H_{k+1} = \mathrm{hash}(H_k, \mathrm{tokens}[k \cdot B : (k+1) \cdot B])
$$

命中意味着存在 block B 满足 `B.block_hash == H_{k+1}`。但 B 的 `parent_block_hash` = $H_k$ —— 而 $H_k$ 不在 cache 中（miss）。这与"cache 中的 block 必须有有效的 parent hash"矛盾——因为 parent hash 也是 block hash 链的一部分，cached 的 block 必然有 valid 的祖先链。

因此 $H_{k+1}$ 不可能命中。■

**实际影响：** 对于 128K token 的长上下文（8000 blocks），前缀查找在 system prompt 的边界（例如前 128 个 block = 2048 tokens）终止。不需要扫描剩余的 7872 个 block。这是 O(1) per block 的连续查找——Bloom filter 无法加速此过程（hash table 已经是 O(1)）。

---

## 13.4 Admission & Eviction：谁能进、谁被赶？

### Admission Policy

一个 block 被 admission 到 global pool 需要四个条件（从 `single_type_kv_cache_manager.py:277` 和 `block_pool.py:211`）：

1. **Block 必须是满的。** `num_full_blocks = num_tokens // block_size`。部分填充的 block 永远不会被缓存——内容可能随更多 token 的到来而变化。
2. **Block 必须未被缓存。** 如果 `block.block_hash is not None`——跳过。
3. **Block 不是 null block。** sliding window 和 Mamba 对齐模式的 null block 不进缓存。
4. **Token 必须是 finalized。** speculative decoding 的 rejected draft tokens 不进缓存。

### Eviction Policy

打开 `vllm/v1/core/kv_cache_utils.py:162`——`FreeKVCacheBlockQueue`。

**LRU 驱逐的精确语义：**

```python
# free_queue: head (LRU) ↔ ... ↔ tail (MRU)
# popleft() → 驱逐最久未用的 block
# append(block) → block 回到 MRU 位置
# free_blocks(list(reversed(req_blocks))) → 尾部 blocks 排在前面
```

**为什么 reversed？** 尾部 blocks（更多 tokens 的 hash chain）的重新计算代价更高——它们涵盖更广的 token 范围。Reverse 让这些值钱的 blocks 排在队列后面（更靠近 MRU、更晚被驱逐）。Head blocks（较少的 chain tokens）先被驱逐——它们的重新计算成本更低。

### touch() — 共享的守护

```python
def touch(block):
    if block.ref_cnt == 0:
        free_queue.remove(block)  # 从驱逐队列中移除
    block.ref_cnt += 1             # 标记"有人用"
```

当一个新请求命中已缓存的 block 时，`touch()` 把它从 free queue 中移除——即使 block 没有活跃的引用计数（`ref_cnt=0`），只要它还在 pool 中，就不应该被驱逐。

---

## 13.5 分布式 Prefix Cache 概述

vLLM 的 KV event 系统（`vllm/distributed/kv_events.py`）允许跨实例共享 prefix cache 信息：

- **BlockStored:** block 被缓存后发出——包含 block_hash + block_id
- **BlockRemoved:** block 被驱逐前发出
- **ZmqEventPublisher:** 通过 ZeroMQ PUB socket 广播事件

外部实例可以通过 KV connector 拉取远程 prefix cache——例如 prefill 实例计算 KV，decode 实例通过 P2P NCCL 或 RDMA 获取。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `GlobalPrefixCachePool` | `block_pool.py:L130` `BlockPool` | 全局 hash index + LRU free list 逻辑一致；简化了 doubly-linked list |
| `compute_chain()` | `kv_cache_utils.py:L539` `hash_block_tokens()` | SHA-256 链式 hash——逻辑一致 |
| `find_longest_prefix()` | `single_type_kv_cache_manager.py:L448` | 左到右扫描、首次 miss 终止——一致 |
| `touch()` | `block_pool.py:L391` | ref_cnt 递增 + cache hit 保护——一致 |

---

## 验证

```bash
cd artifacts/13-prefix-cache-pooling && python -m pytest tests/ -q
# 9/9 passed ✅
```

---

## 总结

- **BlockPool 是全局共享池——不是 per-request cache。** 一个 hash map，跨所有请求共享。
- **单线程 = 零锁。** Sequential 调度消除了并发控制的需要。
- **链式 hash 保证了左到右扫描的正确性。** 第一个 miss 后不需要继续——这是数学证明，不是优化。
- **touch() 是共享机制的核心。** ref_cnt 追踪共享引用，防止被 shared 的 block 被驱逐。
- **不 deduplicate。** 保留 block_table append-only，消除 fast path 的同步开销。

---

**第二部分完结。** 13 章涵盖了 vLLM 从单算子到跨 GPU 分布、从 KV Cache 管理到前缀缓存池化的完整进阶架构。第三部分将进入实战——从 Triton 算子开始构建完整的 Llama-3.2-1B。

---

← 第12章 | 第14章 →
