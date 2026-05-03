# 第7章：Prefix Cache — 共享的 KV Cache 前缀

> 打开 `vllm/v1/core/kv_cache_utils.py:539`。`hash_block_tokens()` 接受三个参数：
> `parent_block_hash`、`curr_block_token_ids`、`extra_keys`。它返回的 hash 不仅取决于当前块的
> token——还取决于**父块的 hash**。这个链式 hash 是整个 APC 系统的数学基石。

---

## 这章要做什么？

你部署了一个 chatbot。每个用户对话都以 2000 token 的 system prompt 开头，内容完全相同。100 个并发用户 = 100 份完全相同的 system prompt K、V 被计算和存储。

这就是 Prefix Cache 要消除的浪费。vLLM 的 Automatic Prefix Caching (APC) 能做到：**第二个请求到达时，system prompt 的 K、V 已经被第一个请求算好了——不需要重新计算，不需要重新分配 block。**

本质上是把第 2 章的 BlockPool 和第 3 章的 PagedAttention 组合起来——同一份 KV block 被多个请求的 block_table 引用。

学完这章你能：
- 打开 `kv_cache_utils.py:539` 理解链式 hash 为什么能保证"没有假阳性命中"
- 理解 `BlockHashToBlockMap`（`block_pool.py:34`）的碰撞处理——为什么不 deduplicate 相同内容的 block
- 解释 `find_longest_cache_hit()`（`single_type_kv_cache_manager.py:448`）为什么遇到第一个 miss 就停止——链式 hash 的数学保证
- 理解 `touch()` 和 `ref_cnt` 如何实现跨请求 block 共享

---

## 7.1 链式 Hash：没有 Radix Tree 的前缀匹配

### Source Trail

打开 `vllm/v1/core/kv_cache_utils.py:539`：

```python
def hash_block_tokens(hash_function, parent_block_hash,
                      curr_block_token_ids, extra_keys=None):
    # H_k = hash( (H_{k-1} or NONE_HASH, tuple(token_ids), extra_keys) )
    return hash_function((
        parent_block_hash if parent_block_hash is not None else NONE_HASH,
        tuple(curr_block_token_ids),
        extra_keys or (),
    ))
```

### Theory: 为什么链式 hash 可以替代 Radix Tree？

Radix Tree（或 Trie）是前缀匹配的经典数据结构——插入和查找都是 O(L)（L 为前缀长度）。但 vLLM 没有用它。vLLM 用了一个简单的 **链式 hash + 线性扫描**。

**链式 hash 的定义：**

```
H_0 = hash(NONE_HASH, tokens[0:16])
H_1 = hash(H_0, tokens[16:32])
H_2 = hash(H_1, tokens[32:48])
...
```

每个 block 的 hash 包含了父 block 的 hash。这形成了一个 hash chain——类似于区块链的链式结构。

**关键性质：** 如果两个请求的 H_k 相同，则它们的前 k×16 个 token 必然完全相同。为什么？因为 H_k 通过父 hash H_{k-1} 间接包含了所有之前的 token——如果任何之前的 token 不同，H_k 一定不同（hash 的碰撞抗性保证了这一点，概率约为 2^{-128}）。

**这如何替代 Radix Tree？** 在前缀匹配中，你只需要找到"直到哪里是相同的"。链式 hash 让你可以用一次 O(1) 的 hash table lookup 检查"第 k 个 block 是否已缓存"。从第一个开始，遇到 miss 就停止——因为链式性质保证了后续不可能命中。

**trade-off：** Radix Tree 的查找是 O(L)，但可以找到任意长度的匹配（不限于 block 边界）。链式 hash 的查找也是 O(L)，但只在 block 边界上检查。vLLM 选择链式 hash 是因为 block 已经是 KV Cache 的天然单元——你不需要非 block 对齐的匹配。

`★ Insight ─────────────────────────────────────`
用链式 hash 替代 Radix Tree 是一个"利用问题结构简化数据结构"的经典例子。Radix Tree 解决的是通用前缀匹配问题——任意字符串，任意边界。但 vLLM 的前缀匹配有一个额外结构：**前缀总是 block 对齐的**，因为 KV Cache 的分配和缓存都以 block 为最小单元。这个约束把"在任意位置找最长公共前缀"简化成了"从位置 0 开始顺序检查每个 block 是否命中"——不需要树结构，一个 hash table 就够了。
`─────────────────────────────────────────────────`

---

## 7.2 BlockHashToBlockMap：碰撞处理

### Source Trail

打开 `vllm/v1/core/block_pool.py:34`。

```python
class BlockHashToBlockMap:
    def __init__(self):
        self._cache: dict[BlockHashWithGroupId,
                          KVCacheBlock | dict[int, KVCacheBlock]] = {}
```

value 的类型是 `KVCacheBlock | dict[int, KVCacheBlock]`——两种状态。这处理了 hash 碰撞：

```python
def insert(self, key, block):              # L75-L91
    if key not in self._cache:
        self._cache[key] = block           # 正常：一个 key → 一个 block
    elif isinstance(self._cache[key], KVCacheBlock):
        # 碰撞！升级为 dict
        existing = self._cache[key]
        self._cache[key] = {existing.block_id: existing, block.block_id: block}
    else:
        # 已经是 dict 了，追加
        self._cache[key][block.block_id] = block
```

### Theory: 为什么不 deduplicate？

如果两个 block 有相同的内容（相同 hash），为什么要保留两个物理副本？为什么不 deduplicate——只保留一个，让后来者指向它？

答案在 `block_pool.py:48-53` 的注释中：**让 block_table 保持 append-only。** 如果 vLLM deduplicate 了——把后来的 block 删掉，指向第一个——那 block_table 就需要修改已经分配好的 block ID。但 block_table 是一个 int32 tensor，被 CUDA kernel 以只读方式使用——在 kernel 运行时修改它需要同步，而且会破坏"block ID 永远不变"的不变性。

所以当两个请求独立产生了内容完全相同的 block，它们各自保留自己的物理块——都在 hash table 中。`get_one_block()`（L62）可以返回**任何一个**——对于前缀缓存命中来说，任何一个都行。随后的 `touch()` 会给被返回的 block 增加 `ref_cnt`，另一个会在 `ref_cnt=0` 时被 LRU 驱逐。

---

## 7.3 find_longest_cache_hit：左到右扫描

### Source Trail

打开 `vllm/v1/core/single_type_kv_cache_manager.py:448`：

```python
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    for block_hash in itertools.islice(block_hashes, max_num_blocks):
        cached_block = self.block_pool.get_cached_block(block_hash, ...)
        if cached_block:
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        else:
            break  # ← 第一个 miss 就停止
```

### Theory: 数学正确性证明

**断言：** 如果 `block_hashes[k]` 在 cache 中 miss，则对所有 j > k，`block_hashes[j]` 也一定 miss。

**证明：** 假设 `block_hashes[j]` 命中——即存在一个 block B 满足 `B.block_hash == block_hashes[j]`。根据链式 hash 的定义：

```
block_hashes[j] = hash(block_hashes[j-1], tokens[j*16:(j+1)*16])
```

如果 B 存在，则 B 的 `parent_block_hash` = `block_hashes[j-1]`。递归向下：`block_hashes[k]` 也必须在 cache 中——因为 `block_hashes[j]` 的祖先链包含了它。

但 `block_hashes[k]` 在 cache 中 miss（题设）。矛盾。因此 j 不可能命中。■

**推论：** 从 k=0 开始左到右扫描，第一个 miss 处终止——之后的都不需要检查。在平均情况下，这使查找变成 O(hit_length) 而不是 O(total_blocks)。

### 长上下文场景分析

对于 128K token 的长 context（8000 blocks）——即使只有前 64 个 block（1024 tokens）的 system prompt 共享——算法在 block 64 处 miss，立即停止。**不需要扫描剩余的 7936 个 block。** 这是纯 hash-table 方案相对于 Radix Tree 的另一个优势：Radix Tree 需要沿树向下走到 miss 点——仍然 O(hit_length)，但常数项更大（树节点的指针追逐 vs hash table 的 O(1) 查找）。

---

## 7.4 跨请求 Block 共享：ref_cnt 机制

### Source Trail

打开 `vllm/v1/core/block_pool.py:391`：

```python
def touch(self, blocks):                    # L391-L406
    for block in blocks:
        if block.ref_cnt == 0:
            self.free_block_queue.remove(block)  # 从自由队列取出
        block.ref_cnt += 1                       # 增加引用计数
```

`touch()` 在 `allocate_new_computed_blocks()` 中被调用（`single_type_kv_cache_manager.py:218`）——当新请求命中已缓存的 block 时，`touch()` 给 block 加一个引用。

### Theory: 引用计数的生命周期

```
请求 A 分配 block 5 → ref_cnt = 1
请求 A 缓存 block 5 → block_hash 被设置
请求 B 前缀命中 block 5 → touch(block 5) → ref_cnt = 2
  现在 block 5 被两个请求共享
请求 A 完成 → free_blocks → ref_cnt = 1 (block 还在)
请求 B 完成 → free_blocks → ref_cnt = 0 → block 回到自由队列
```

**共享期间两个请求的 block_table 都指向物理 block 5。** 因为 `ref_cnt>0`，block 不会被驱逐——即使它在自由队列中（`touch()` 把它从自由队列移除）。这是第 2 章讨论的 lazy eviction 的另一个体现：block 的物理存储保持分配，直到所有引用者都释放。

---

## 7.5 Hash 算法选择与安全

### Source Trail

打开 `vllm/vllm/utils/hashing.py`：

| 算法 | 序列化 | Hash 函数 | 速度 | 跨语言 |
|------|--------|----------|------|--------|
| `sha256` (默认) | Pickle | SHA-256 | 慢 | Python 限定 |
| `sha256_cbor` | CBOR | SHA-256 | 慢 | 是 |
| `xxhash` | Pickle | xxHash 128-bit | 快 | Python 限定 |
| `xxhash_cbor` | CBOR | xxHash 128-bit | 快 | 是 |

默认用 SHA-256 + Pickle。为什么选这么慢的？两个原因：

1. **安全性。** SHA-256 的碰撞概率 $2^{-128}$（实际使用截断的输出长度），对于实际系统来说可以忽略不计——即使处理数十亿请求，碰撞概率仍然 < 2^{-80}。xxHash 更快但碰撞抗性弱得多。
2. **Pickle 是 Python 的默认序列化。** deterministic + 不需要额外依赖。

**如果性能是瓶颈，切到 xxhash_cbor。** hash 计算是每完成一个 block 做一次——对于 128K context 那就是 8000 次 hash。SHA-256 每次 hash 约 1 μs → 8000 × 1μs = 8ms——对于单个请求可以接受。但如果每秒有数百个请求完成 block 并做 hash——xxHash 的 0.1 μs/hash 能把 hash 时间从 8ms 降到 < 1ms。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `ChainedBlockHasher` | `kv_cache_utils.py:L539` `hash_block_tokens()` | SHA-256 + 同样的链式逻辑；未实现 CBOR 序列化 |
| `PrefixCacheIndex` | `block_pool.py:L34` `BlockHashToBlockMap` | 碰撞升级逻辑一致；简化为 `block_id` 而非 `KVCacheBlock` 对象 |
| `PrefixCacheManager.find_longest_cache_hit()` | `single_type_kv_cache_manager.py:L448` | 左到右扫描、首个 miss 终止——逻辑一致 |
| `cache_blocks()` | `single_type_kv_cache_manager.py:L277` | 只缓存新 full blocks 的逻辑一致 |

---

## 验证

```bash
cd artifacts/07-prefix-cache && python -m pytest tests/ -q
# 11/11 passed ✅
```

---

## 总结

- **链式 hash 替代 Radix Tree。** 利用 block 对齐的约束，把"最长前缀匹配"简化为"顺序 block 查找"——O(hit_length) 且常数项极小。
- **左到右扫描的正确性由数学保证。** 链式 hash 的性质：第 k 个 block miss → 之后所有 block 必然 miss。
- **碰撞处理不 deduplicate。** 保留所有物理副本，保持 block_table append-only——这是 GPU kernel 不需要同步的代价。
- **ref_cnt 实现跨请求共享。** touch() 加引用，free_blocks() 减引用——ref_cnt>0 时 block 不会被回收。
- **SHA-256 默认，xxHash 可选。** 安全 vs 速度的 trade-off——生产环境可根据 QPS 选择。

---

**下一章：** 第8章 — Tensor Parallelism 张量并行

Prefix Cache 让多个请求共享 system prompt 的计算。但当模型大到一张 GPU 放不下时——比如 Llama-3.2-70B 的 140 GB 权重，需要跨 4 张 H100——就需要模型并行。第 8 章将从矩阵乘法的切分开始，推导 Column Parallel 和 Row Parallel 的数学等价性，然后追踪 vLLM 从 `ParallelConfig` 到 NCCL AllReduce 的 TP 实现路径。

---

← 第6章 | 第8章 →
