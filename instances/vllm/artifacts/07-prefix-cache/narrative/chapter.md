# 第7章：Prefix Cache 与 APC-Aware Allocation — 没有 Radix Tree 的前缀复用

> 本章涉及的 vLLM 源码：
> - `instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L40-L78`（`BlockHash` / `BlockHashWithGroupId` / group-id 4-byte 大端 pack-unpack）+ `L83-L110`（`init_none_hash` 链头种子）+ `L373-L536`（`extra_keys` 组合）+ `L539-L566`（`hash_block_tokens` 链式 hash 主入口）+ `L635-L686`（`get_request_block_hasher` 请求侧绑定）+ `L162-L370`（`FreeKVCacheBlockQueue` 的 O(1) 中段移除）
> - `instances/vllm/source/vllm/v1/core/block_pool.py:L34-L127`（`BlockHashToBlockMap` — 平面 dict + collision 子 dict）+ `L184-L209`（`get_cached_block` — 多组同时命中）+ `L211-L320`（`cache_full_blocks` insert 路径）+ `L322-L352`（`get_new_blocks` + 懒驱逐）+ `L354-L389`（`_maybe_evict_cached_block`）+ `L391-L422`（`touch` + `free_blocks`）+ `L424-L441`（`evict_blocks`）+ `L48-L52`（"我们不去重" 的 NOTE）
> - `instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L30-L82`（`SingleTypeKVCacheManager`）+ `L277-L318`（`cache_blocks` + 反序 free）+ `L336-L383`（`find_longest_cache_hit` ABC）+ `L446-L494`（`FullAttentionManager.find_longest_cache_hit` 主线）
> - `instances/vllm/source/vllm/v1/core/kv_cache_manager.py:L21-L103`（`KVCacheBlocks` dataclass）+ `L106-L161`（init）+ `L183-L223`（`get_computed_blocks`）+ `L225-L416`（`allocate_slots` — prefix-cache 集成入口）
> - 对照实现 `instances/vllm/artifacts/07-prefix-cache/implementation/`：`block_hash.py`、`prefix_cache_index.py`、`radix_tree.py`、`prefix_cache_manager.py`、`paged_integration.py`、`demo.py`
>
> 本章源码 commit：`98661fe`。第 5 章把 `BlockPool` 和 `KVCacheBlock` 三态生命周期讲完了；第 7 章是这套机器**之上**的 prefix-cache 层——我们不再发明新的物理块管理，而是回答："多个请求共享同一段 system prompt 时，怎么让物理块也共享，让 KV 不重新算？"

---

## 这章要讲什么？

打开 `instances/vllm/source/vllm/v1/core/block_pool.py:L34`，这里有一行：

```python
class BlockHashToBlockMap:
    """A mapping from block hash to block IDs. The block IDs can be a single
    one or a dict ..."""
```

它**是一个 dict**。不是 radix tree、不是 trie、不是 prefix tree。整个 vLLM v1 源码树里搜不到一个 `class.*Radix|Trie|PrefixTree`（commit `98661fe` 实测 0 个匹配）——可外面的文献和讨论里"vLLM 用 radix tree" 这种说法又特别常见。这就是本章要解决的第一个语言陷阱。

vLLM v1 用**链式 hash + 平面 dict**替代 radix tree。math 是简单的：

$$
H_k = \mathrm{sha256}(H_{k-1} \,\Vert\, \mathrm{tokens}[kB:(k+1)B] \,\Vert\, \mathrm{extra\_keys})
$$

如果两个请求的 $H_k$ 相同，那它们前 $(k{+}1) \cdot B$ 个 token **必须**逐字节相同（sha256 抗碰撞）。一个 `dict.get(H_k)` O(1) 就能回答"第 $k$ 个块有没有被其他请求缓存过"——这就是 radix tree 的"tree property"被链式 hash 偷换出来的本质。tree 都没有，何来 tree 操作？

学完这章你能：

- 在白板上推导 `H_k` 链式 hash 公式，并解释为什么链式依赖让 `find_longest_cache_hit` 可以**遇到第一个 miss 就 break**（`single_type_kv_cache_manager.py:L482-L483` 的注释一字不漏）。
- 用 $(N{-}1) \cdot K$ 节省公式回答"50 个请求共享 32 块 system prompt 节省多少 KV"——并把 78% 当成这个公式在某个具体工作负载下的取值，而不是单独的常数。
- 解释为什么 demo §2 微基准里 hash 表比 radix tree 快 **4.65 倍**——这**不是**渐近差异（两者都是 $O(L)$），而是 Python 解释器走 `RadixNode.children.get` 一遍就要付一次属性查询 + 字典查 + 函数 frame 的开销，而 hash 表只需要一次 C 级 `dict.get`。
- 看懂"链断裂"现象：evict 块 0 → 块 1 物理上还在内存里（KV 还能算），但 prefix-cache lookup 永远 miss——直到块 1 被 LRU 单独驱逐。这是 K06 三态生命周期 + K11 chain-break 的组合。
- 区分**4 个语言陷阱**：vLLM 用 radix tree？✗。partial block 也能 cache 命中？✗。命中 = 请求免费？✗。要遍历所有 prefix 才能找最长？✗。

接下来 6 节按 outline 走，但 §7.2 已经从 "Radix Tree 数据结构详解" 重构成 "为什么 vLLM 选了 dict 不选 radix tree"——源码里就没有 radix tree 这个东西，硬讲就是失真。

---

## 7.1 链式 hash：math 基础 + "没有 radix tree"的设计选择

### 7.1.1 打开链式 hash 主入口

源码定位：`instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L539-L566`，`hash_block_tokens()`：

```python
# vllm/v1/core/kv_cache_utils.py:L539-L566 (节选)
def hash_block_tokens(
    hash_function: Callable,
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: tuple[int, ...],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH    # L560-L561 链头种子
    return BlockHash(hash_function(
        (parent_block_hash, curr_block_token_ids, extra_keys)   # L563-L565
    ))
```

三件事情同时进来：**parent hash**、**当前块 token**、**extra_keys**。三者一起 hash 成 `H_k`，然后这个 `H_k` 又会作为下一块的 parent。这是"链式"的字面意思——每一块都背着前面所有块的"压缩签名"。

### 7.1.2 为什么链式 hash 等于 radix tree 的"tree property"

先用大白话讲直觉：你想知道两个请求"前面 64 个 token 是不是相同"。最笨的办法是一对一比 64 个 int——慢。聪明一点是把这 64 个 token **hash 成一个 32 字节摘要**，比摘要——快了 64 倍但还是要把摘要算出来。

链式 hash 再聪明一步：**预先**把每 16 个 token 算一个 hash，然后第 2 块的 hash 把第 1 块的 hash **塞进自己的输入里**。这样的好处分两个方向：

- 正向：如果两个请求的第 2 块 hash 相等，那第 1 块 hash 必须相等（否则 sha256 输入不同则输出不同），第 0 块 hash 也必须相等（同理）——递推下去前 48 个 token 必须逐字节相等。
- 反向：如果第 0 块 hash 不同，第 1 块一定不同（输入里有第 0 块），第 2 块一定不同……一直到链尾。

正式写出来就是 **chain monotonicity 引理**：

$$
\forall k \in \{0, 1, \dots, L/B - 1\}: \quad H_k^{(A)} = H_k^{(B)} \iff \mathrm{tokens}_{0:(k+1)B}^{(A)} = \mathrm{tokens}_{0:(k+1)B}^{(B)}
$$

证明（双向，sha256 抗碰撞下）：

**充分性 (⇐)**：假设前 $(k{+}1) \cdot B$ 个 token 相同。第 0 块输入相同推出第 0 块 hash 相同。归纳：若第 $i$ 块 hash 相同，那么第 $i{+}1$ 块的输入 (parent hash, body, extra_keys) 三项全相同（前两项已同，第三项是请求级常量），sha256 函数性推出第 $i{+}1$ 块 hash 相同。

**必要性 (⇒)**：反证。假设第 $k$ 块 hash 在两请求中相等，但前 $(k{+}1) \cdot B$ 个 token 有至少一处不同。那么第 $k$ 块的 sha256 输入三项里必有一项不同（要么 body，要么 parent hash；parent hash 不同又递推到更下层）。sha256 抗碰撞——不同输入对应不同输出，矛盾。

引理的**实战推论**：**miss at $k$ 蕴含 miss at $k{+}1, k{+}2, \dots$**。`find_longest_cache_hit` 一旦在某个位置 dict lookup 失败，根本没必要再往下查——链上后续都不可能命中。

### 7.1.3 我们的实现：`block_hash.py`

`implementation/block_hash.py:L83-L109`：

```python
# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L539-L566
def hash_block_tokens(
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple = (),
) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH                          # 对应源码 L560-L561
    payload = repr(
        (bytes(parent_block_hash), tuple(curr_block_token_ids), extra_keys)
    ).encode()                                                  # 对应源码 L563-L565
    return BlockHash(hashlib.sha256(payload).digest())
```

我们用 `sha256(repr(...))`，源码用 `sha256_cbor`——序列化方法不同但 **parent hash 注入顺序是核心**，这点完全一致。`repr` 易测试、不引入 cbor2 依赖；CBOR 是源码生产路径。

`chain_block_hashes` (`block_hash.py:L116-L140`) 把 `hash_block_tokens` 串成一根链：

```python
# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L635-L686
def chain_block_hashes(
    token_ids: Sequence[int], block_size: int, extra_keys: tuple = (),
) -> list[BlockHash]:
    hashes: list[BlockHash] = []
    parent: BlockHash | None = None
    for i in range(0, len(token_ids), block_size):
        block = token_ids[i : i + block_size]
        if len(block) < block_size:
            break               # ← 关键：partial block 永不 hash
        h = hash_block_tokens(parent, block, extra_keys)
        hashes.append(h)
        parent = h
    return hashes
```

`if len(block) < block_size: break` 这一行就把后面的语言陷阱 2 钉死了——只有 FULL block 进入 cache。

源码差异：vLLM 在 `Request.update_block_hashes()` 里把这个动作做成**惰性增量**（每次新 token 到来只 hash 新满的那块），我们提取成纯函数方便单测——同一份 token 列表每次输出 hash 链一致。

### 7.1.4 `extra_keys` 把 LoRA / 多模态 / cache_salt 拆开

K09。如果两个请求文本一样但用不同的 LoRA adapter——它们在 KV 空间是**两份不同的 K, V**，prefix cache 必须把它们当作不同的 hash key。源码 `kv_cache_utils.py:L501-L536` 的 `generate_block_hash_extra_keys` 按顺序拼：

$$
\mathrm{extra\_keys} = (\mathrm{lora}, \mathrm{mm}, \mathrm{cache\_salt}_{[\mathrm{block}\,0\,\mathrm{only}]}, \mathrm{prompt\_embeds})
$$

`cache_salt` 只追在块 0 里（L522-L524：`[request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []`）——因为 salt 是请求级的，链式传递保证后续块都受它影响，没必要每块都加。

我们简化版把 `extra_keys=()` 留作默认 tuple——文本/单 LoRA workload 够用；Ch12 多模态时再把 mm tuple 接进来。

### 7.1.5 §7.1 mini 映射表（链式 hash 局部）

| 我们的代码 | vLLM 源码 | 一致 / 简化 |
|-----------|----------|-----|
| `BlockHash`、`BlockHashWithGroupId` (`block_hash.py:L39, L42`) | `kv_cache_utils.py:L40, L45` | 一字不差 NewType |
| `init_none_hash` (`block_hash.py:L48-L54`) | `kv_cache_utils.py:L83-L110` | 去掉 cbor2 警告 |
| `hash_block_tokens` (`block_hash.py:L83-L109`) | `kv_cache_utils.py:L539-L566` | sha256(repr()) 替代 sha256_cbor |
| `chain_block_hashes` (`block_hash.py:L116-L140`) | `Request.update_block_hashes` (`request.py`) | 提取成纯函数 |
| `make_block_hash_with_group_id` (`block_hash.py:L61-L69`) | `kv_cache_utils.py:L53-L62` | 一字不差大端 4 字节 |
| `common_prefix_length` (`block_hash.py:L145-L160`) | — | 新增（demo §1 需要） |

---

## 7.2 BlockHashToBlockMap：append-only 不去重的工程理由

### 7.2.1 为什么没有 radix tree——源码事实先讲

第一句话我们已经说过了：**vLLM v1 没有 radix tree**。本节展开"为什么没有"——这是工程取舍题，不是性能题。

打开 `instances/vllm/source/vllm/v1/core/block_pool.py:L34-L127`：

```python
# vllm/v1/core/block_pool.py:L34-L127 (核心数据结构定义)
class BlockHashToBlockMap:
    """A mapping from block hash to block IDs. The block IDs can be a single
    one or a dict (for collision)."""
    def __init__(self) -> None:
        self.cache: dict[BlockHashWithGroupId, KVCacheBlock |
                         dict[int, KVCacheBlock]] = {}
```

值类型 `KVCacheBlock | dict[int, KVCacheBlock]` 是个 union——**单条**或者**多条**。99.99% 的时间是单条；多条只有在两个不同物理 block 共用一个 hash key 时才出现（sha256 碰撞实际不会，但 NOTE 里说 "we don't deduplicate" 留了这个边界）。

如果换成 radix tree，节点要存子节点指针、edge label tuple、`block_id`、可能的 metadata……还有插入时的 split、删除时的 merge。dict 一行 `self.cache: dict = {}` 就完了。

### 7.2.2 真·radix-tree 反例：`radix_tree.py` 的存在意义

我们在 `implementation/radix_tree.py` 里**手写了一个真 radix tree**——不是为了用它，而是为了让读者**亲眼看到**它有多复杂。`RadixTree.insert` (`radix_tree.py:L138-L187`) 做的事：

```python
# REFERENCE: pedagogical, NOT in vLLM
def insert(self, tokens: list[int], block_id: int) -> None:
    ...
    while idx < len(tokens):
        child = node.children.get(tokens[idx])
        if child is None:
            # 新建一条边，剩下所有 token 一次性塞进 edge label
            ...; return
        edge = child.edge
        i = 0
        while (i < len(edge) and idx + i < len(tokens) and
               edge[i] == tokens[idx + i]):
            i += 1
        if i == len(edge):
            node = child; idx += i; continue            # 全匹配下行
        # 部分匹配 → split edge
        split = RadixNode(edge=edge[:i])
        split.children[edge[i]] = RadixNode(
            edge=edge[i:], children=child.children, block_id=child.block_id,
        )
        ...; return
```

**就这一个 insert 函数 50 行**——还没算 lookup、evict、edge merge。对比 `BlockHashToBlockMap.insert` (`prefix_cache_index.py:L79-L95`) **3 个 if 分支 14 行**。读者可以翻到 `radix_tree.py` 自己读完，再回来——立即明白工程为什么选 dict。

### 7.2.3 真·hash 表 insert 的三个分支

`block_pool.py:L75-L91` → 我们的 `prefix_cache_index.py:L79-L95`：

```python
# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L75-L91
def insert(self, key: BlockHashWithGroupId, block_id: int) -> None:
    entry = self._cache.get(key)
    if entry is None:
        self._cache[key] = block_id                     # 分支 1：空 bucket
    elif isinstance(entry, int):
        self._cache[key] = {entry: entry, block_id: block_id}  # 分支 2：升级 dict
    elif isinstance(entry, dict):
        entry[block_id] = block_id                      # 分支 3：dict 追加
```

三个分支：空、单 int 升级到 dict-of-two、已有 dict 追加。注意 vLLM 的 dict 值类型是 `KVCacheBlock`，我们用 `int`（block_id）替代——块对象本身的属性管理是 Ch05 的领地。

### 7.2.4 K02 的 NOTE：为什么不做 dedup

`block_pool.py:L48-L52` 是**整本 vLLM 源码里最重要的工程注释之一**：

```python
# vllm/v1/core/block_pool.py:L48-L52 (原文)
# NOTE: We currently don't de-duplicate the blocks in the cache, meaning
# that if a block becomes full and is cached, we don't check if there is
# already an identical block in the cache. This is because we want to make
# sure the allocated block IDs won't change so that block tables are
# append-only.
```

翻译："虽然我们能识别出两个新满的 block 内容相同，但故意不合并它们。原因：block_table append-only。"

**深层原因**——PagedAttention 的 GPU kernel 每个 step 都会读 `block_table`（一个 int32 tensor）：

$$
\mathrm{attn}_{i,j} = q_i \cdot k_{\mathrm{block\_table}[j / B], j \bmod B}
$$

如果一个请求 A 的 `block_table[5] = 7`，正在 decode 的某 step 里 GPU kernel 已经把 7 加载进 SRAM；这时 prefix cache 把 block 7 "merge" 到 block 12——下一个 step 里 kernel 读的还是 7，但 cache 索引指向 12。**结果：A 在第 5 块位置读了别的请求的 KV，输出乱码**。

vLLM 的回避方式很硬：**block_id 一旦发给一个 request，永不变**。这是 invariant **I1 — Append-only block_id assignment**。为了它，dict 留一个 "万一两个 hash key 真的相同" 的 collision 子 dict 兜底（实际 sha256 下不会发生），完全放弃了去重收益。

权衡是 work vs correctness。同样道理在分布式系统里到处出现："为了 invariant 不可破，宁可付局部成本"。

### 7.2.5 我们的实现：`prefix_cache_index.py`

完整的 `BlockHashToBlockMap` 在 `prefix_cache_index.py:L44-L125`——insert/pop/get 三个方法各自一个分支三选一。`get_one_block` (`prefix_cache_index.py:L67-L76`) 处理 union 类型：

```python
def get_one_block(self, key: BlockHashWithGroupId) -> int | None:
    entry = self._cache.get(key)
    if entry is None: return None
    if isinstance(entry, int): return entry
    if isinstance(entry, dict): return next(iter(entry.values()))
    raise AssertionError(f"Invalid entry type {type(entry)}")
```

"返回任意一条" 是因为 hash key 相同的两个 block 在功能上等价（都缓存了同一组 token 的 KV）——挑哪一条不影响正确性。

### 7.2.6 多组同时命中：`get_cached_block`

`block_pool.py:L184-L209` → `prefix_cache_index.py:L131-L151`：

```python
# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L184-L209
def get_cached_block(
    cache, block_hash, kv_cache_group_ids: list[int]
) -> list[int] | None:
    cached: list[int] = []
    for group_id in kv_cache_group_ids:
        packed = make_block_hash_with_group_id(block_hash, group_id)
        block_id = cache.get_one_block(packed)
        if block_id is None:
            return None                          # ← 任一组 miss → 整体 miss
        cached.append(block_id)
    return cached
```

K07：**所有组都命中才算命中**。混合注意力模型（full attention + sliding window 同时存在）有多组 KV cache，必须每组都有这个 hash 才能跳过整层 forward——一个组缺，那一层还是要算。单组模型这里 loop 一次就出结果。

---

## 7.3 `find_longest_cache_hit` 走读：scan-and-stop 主线

### 7.3.1 打开主入口

源码定位：`single_type_kv_cache_manager.py:L446-L494`，`FullAttentionManager.find_longest_cache_hit`：

```python
# vllm/v1/core/single_type_kv_cache_manager.py:L446-L494 (节选 L470-L490)
@classmethod
def find_longest_cache_hit(
    cls, block_hashes: list[BlockHash], max_length: int,
    kv_cache_group_ids: list[int], block_pool: "BlockPool",
    kv_cache_spec: KVCacheSpec, use_eagle: bool,
) -> tuple[list[KVCacheBlock], ...]:
    ...
    computed_blocks = tuple([] for _ in kv_cache_group_ids)
    max_num_blocks = max_length // kv_cache_spec.block_size
    for i, block_hash in enumerate(block_hashes[:max_num_blocks]):
        cached_blocks = block_pool.get_cached_block(
            block_hash, kv_cache_group_ids)
        if cached_blocks is None:
            # L482-L483 的注释：
            # "if a block hash is not in cached_block_hash_to_id,
            #  the following block hashes are not computed yet for sure."
            break                                  # ← 链式 hash 让 break 合法
        for j, b in enumerate(cached_blocks):
            computed_blocks[j].append(b)
    return computed_blocks
```

三件事：

1. **`max_length // block_size`** 截断 — 我们最多看链子前这么多块。
2. **每块 → `get_cached_block`** 多组同时命中。
3. **`if cached_blocks is None: break`** — 这就是 §7.1.2 链式 hash 引理在工程里的兑现：第一次 miss 之后**不可能**有 hit，扫到这里直接收手。

### 7.3.2 为什么不用二分查找？

读者可能会想："找最长前缀，二分查找不是更快吗？$O(\log L/B)$ 比 $O(L/B)$ 强啊？"

回答：**二分查找需要"中间能查到"。链式 hash 的"miss at $k$ → miss at $k+1, \dots$" 让 hit set 是个**前缀闭包**——0, 1, 2, ..., k 全 hit，k+1, ..., 全 miss——这种结构上线性扫一次就够，二分还要算两次中间块的 hash + 两次字典查，反而慢。

而且别忘了：每次"看一块"要算一次链式 hash，hash 的输入又依赖 parent hash——不能跳着算。所以哪怕假设二分能省 dict.get 次数，hash 计算本身也得从头串到中点。最干净的实现就是 `for h in chain: dict.get(h); break on miss`，O(hit_length)。

### 7.3.3 4 个变体的 mini-map（不深入，只标记存在）

源码里 `find_longest_cache_hit` 不止一个——`single_type_kv_cache_manager.py` 里 4 个变体，每个对应一种注意力 spec：

| 变体 | 源码位置 | 干什么 | 何时用 | 对本章 |
|----|--------|------|------|------|
| `FullAttentionManager` | `single_type_kv_cache_manager.py:L446-L494` | scan-and-stop 主线 | 标准 self-attention 层 | 本节走读 |
| `SlidingWindowManager` | `single_type_kv_cache_manager.py:L513-L580` | 窗口受限的链式 hit | sliding window attention 层 | Ch12 |
| `MambaManager` | `single_type_kv_cache_manager.py:L1052+` | 状态空间模型；不一定按块切分 | Mamba 等 SSM | Ch20 |
| `CrossAttentionManager` | `single_type_kv_cache_manager.py:L1094+` | encoder-decoder 模式 | 多模态 cross-attn | Ch15 |

§7.3 只走 `FullAttentionManager`——其他三个变体在原书后续章节里铺。读者现在明白：本节扫的是单个变体，`find_longest_cache_hit` 是 ABC（`L336-L383`），子类各自实现 scan policy。

### 7.3.4 我们的实现：`PrefixCacheManager.find_longest_cache_hit`

`prefix_cache_manager.py:L61-L83`：

```python
# REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L446-L494
def find_longest_cache_hit(
    self, block_hashes: list[BlockHash], max_length: int,
) -> list[int]:
    max_num_blocks = max_length // self.block_size
    cached: list[int] = []
    for block_hash in block_hashes[:max_num_blocks]:
        block_id = self.cache.get_one_block(
            make_block_hash_with_group_id(block_hash, self.group_id)
        )
        if block_id is None:
            # 对应源码 L482-L483 的 "scan-and-stop" 注释
            break
        cached.append(block_id)
    return cached
```

我们与源码的差异：

- 单 group：`self.group_id` 替代 `kv_cache_group_ids: list[int]` —— 多组留给 Ch12-13。
- 返回 `list[int]` 而不是 `tuple[list[KVCacheBlock], ...]` —— 块对象 → 块 id 简化（Ch05 边界）。
- 没有 `use_eagle` / `kv_cache_spec` —— 这俩是变体路由，单变体不需要。

逻辑核心是 `if block_id is None: break` —— 一行，对应源码的引理。这一行是本章最重要的 7 个字符。

### 7.3.5 demo §3 trace：4 步看 chain-break

`demo.py:section_3_match_insert_evict()` 用 `block_size=4` 做了一个能整章手算的小工作负载：

```
工作负载：sys_prompt = [101..108] (2 块)，每个请求 = sys_prompt + 8 个独占 token

步骤 1: A = sys + [201..208] → 4 块 (hit 0, fresh 4) ← 第 1 个进来全 fresh
步骤 2: B = sys + [301..308] → 4 块 (hit 2, fresh 2) ← B 拿走 A 的前 2 块
步骤 3: C = [501..508]       → 2 块 (hit 0, fresh 2) ← 不同 prefix，no hit
步骤 4: free A; evict A 块 0 (chain[0])
步骤 5: D = sys + [401..408] → 4 块 (hit 0, fresh 4) ← chain 在块 0 处 broke
```

最关键是步骤 5。直觉会说："块 1 不是没被 evict 吗？D 应该 hit 1 块呀。" 但答案是 **0 块**——即使块 1 还在 dict 里、KV 还在 GPU 显存里。

**为什么？** D 的 chain 是 `[H_0, H_1]`。`find_longest_cache_hit` 先查 `H_0` → miss（A 已 evict 了）→ break。永远到不了 `H_1`。块 1 在 dict 里没有用——访问它的入口（`H_0`）已经断了。

这就是 K11 chain-break test 的核心断言：

```python
# 关键测试：tests/test_prefix_cache_manager.py
mgr.evict_block(blocks[0], chain[0])
hits = mgr.find_longest_cache_hit(chain, max_length=8)
assert hits == []   # NOT [block_id_1] — chain breaks at first miss
```

如果有人以后改坏了 `find_longest_cache_hit`、让它退化成"穷举所有 hash"——这个测试会立刻 fail。链断裂不是一个 bug，**它就是 prefix cache 的核心不变量** I3。

### 7.3.6 demo §2 微基准：4.65× 不是渐近差异

demo §2 跑一个真刀真枪的 microbench——1000 个 prefix、5000 次查询。结果（K12）：

```
data structure        total_time      lookups/sec    rel speed
Trie                   62.276s        80,291/s        1.00x
RadixTree              64.738s        77,232/s        0.96x
Hash table (vLLM)      13.392s       373,348/s        4.65x
```

读者第一眼会想："一定是 hash 表常数时间比 tree 线性快了。" **错。**

正确解读（K12）：两者**都是 $O(L)$ 渐近**——

- Hash 表查长度 L 的 prefix：L/B 次 chained hash 查询，渐近 $O(L/B)$。
- RadixTree 查长度 L：至多 L 次 edge 比较，渐近 $O(L)$。
- 两者渐近差是 1/B（block size 倒数），即 1/16 倍——**不是 4.65 倍**。

剩下的 4.65 倍来自 **Python 解释器开销**：

- Trie/Radix 每节点要做：`node.children`（属性查询）→ `dict.get(token)`（字典查）→ Python frame 创建 → 进入下一次 while 循环。**每个 token 一次**——L 次开销。
- Hash 表每块只做：`self._cache.get(packed_key)` —— 一次 C 级 dict 调用，`__hash__` 已预计算。**每 16 个 token 一次**——L/16 次开销。

在 Python 下这个常数差比渐近差大得多。**写"hash 表渐近更快"是错的；写"hash 表实现常数小得多"才对**。生产编译型语言（Rust、C++）下两者差距会缩到 ~1.5 倍——还是 hash 赢，但绝对没 4.65 倍这么夸张。

这是本章的诚实交代：demo §2 的数字真实，但解读必须谨慎。

---

## 7.4 `cache_blocks` 插入路径 + 三状态生命周期

### 7.4.1 打开插入主入口

源码定位：`single_type_kv_cache_manager.py:L277-L301`，`SingleTypeKVCacheManager.cache_blocks`：

```python
# vllm/v1/core/single_type_kv_cache_manager.py:L277-L301 (节选)
def cache_blocks(self, request: Request, num_tokens: int) -> None:
    num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
    num_full_blocks = num_tokens // self.block_size
    if num_cached_blocks >= num_full_blocks:
        return                                       # L289-L290 idempotent guard
    # L292-L299 委托给 BlockPool.cache_full_blocks
    new_full_blocks = self.req_to_blocks[request.request_id][
        num_cached_blocks:num_full_blocks
    ]
    self.block_pool.cache_full_blocks(
        request, ...
    )
    self.num_cached_block[request.request_id] = num_full_blocks   # L301
```

三步：检查"已经 cache 多少"、跳过已经 cache 的、把新满的塞进 `BlockHashToBlockMap`。其中 `if num_cached_blocks >= num_full_blocks: return` 是**幂等保护**——同一个 (request, num_tokens) 调用两次，第二次 no-op。

### 7.4.2 为什么 partial block 永不进 cache

`num_full_blocks = num_tokens // self.block_size` —— **整数除法**。`block_size=16, num_tokens=17` → `num_full_blocks=1`，第 17 个 token 的 KV 没有进入 cache 管理。

这就是**语言陷阱 2**："cache hit on partial block？" 答案永远是**没有这种事**。partial block 没进 cache 的原因是 **chain hash 需要完整的 block 内容**——hash 算的是 16 个 token 的 tuple，少一个就不是同一个 hash。**部分相同 ≠ 整块相同**。

实战影响：demo §1 hit-rate sweep 故意把这个数字抠出来——

```
shared_frac    requests     cached_tokens    hit_rate
       0.10         100           9,300       9.3%
       0.30         100          29,500       29.5%
       0.50         100          49,500       49.5%
       0.70         100          69,300       69.3%
       0.90         100          88,200       88.2%
```

为什么 0.10 是 **9.3%** 而不是 10%？因为 100 token 共享只能填满 $\lfloor 100/16 \rfloor = 6$ 块（96 token）——剩下 4 个 token 进 partial block 永不缓存。9.3% = 6×16/1024 × 99/100 ≈ 0.0928——和 demo 数字对得上。

类似地 0.50 是 49.5%、0.90 是 88.2%——每一档都比 `shared_frac` 略低，差距就是 partial-block 余数。**不要写 "命中率 = 共享比例"，要写 "命中率 ≈ 共享比例 - partial-block 损失"**。

### 7.4.3 三状态生命周期（K06）

`block_pool.py:L391-L422` 的 `touch` + `free_blocks` 串起 3 个状态。从 `KVCacheBlock` 的角度：

| 状态 | `ref_cnt` | `block_hash` | 在 free queue? | 在 cache index? |
|-----|----------|------------|--------------|---------------|
| **1. cached, in-use** | > 0 | set | 不在 | 在 |
| **2. cached, idle** | == 0 | set | 在 | 在 |
| **3. uncached, idle** | == 0 | None | 在 | 不在 |

转换：

- 1 ← 2：`touch()` 命中 + 把块从 free queue 中段拿出（O(1)，因为 `FreeKVCacheBlockQueue` 是 doubly-linked list — Ch05 K05）。
- 2 ← 1：`free_blocks()`，ref_cnt 减 1 归 0 时回到 free queue 但**保留 hash**——这是 K10 "cached blocks survive free_request" 的本质。
- 3 ← 2：`get_new_blocks()` LRU pop 时调 `_maybe_evict_cached_block()`（懒驱逐）—— hash 变 None，从 cache index pop。

**这三态是怎么让 system prompt 复用工作的**：100 个用户各自打开一段同 prompt 的对话——前 N 个用户 free 后，他们的 system-prompt 块还在 cache index 里、`ref_cnt=0`、躺在 free queue 中段。第 101 个用户来了，`find_longest_cache_hit` 一个 `dict.get` 命中，`touch()` 把块从 free queue 中段抽出来——**KV 不重算**，状态 2 → 1 一气呵成。

### 7.4.4 K05：懒驱逐——eviction 不是后台任务

很多人会以为 prefix cache 有个后台线程定期扫"哪些 cached 块该 evict"——**没有**。`block_pool.py:L322-L352` 的 `get_new_blocks` 才是 eviction 真正发生的地方：

```python
# vllm/v1/core/block_pool.py:L322-L352 (节选)
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if self.free_block_queue.num_free_blocks < num_blocks:
        raise ValueError(...)
    ret = []
    for _ in range(num_blocks):
        block = self.free_block_queue.popleft()      # LRU 头
        if block.block_hash is not None:
            self._maybe_evict_cached_block(block)    # 懒驱逐发生在这里
        block.ref_cnt = 1
        ret.append(block)
    return ret
```

**为什么这样设计**（K05）：

1. **没 GIL 争用**：分配本来就要拿锁，eviction 顺手做完，不引入新的锁竞争。
2. **eviction work 跟 allocation work 等量**：分配 100 个块 → 至多 evict 100 个 cached（每次 `popleft` 至多碰一个 cached）。cache 大小 100K 还是 1M 不影响——如果是后台扫描就要遍历整个 cache。
3. **invariant**：popleft 之后立刻 evict，**cache index 永不指向已经 popleft 的块**。后台扫描就要处理"扫描和 popleft 的中间窗口"——多一个并发 bug 来源。

我们的简化版 `prefix_cache_manager.py:L125-L138` 把 `evict_block` 暴露成直接调用——demo 和测试需要。生产路径是源码的内嵌。

### 7.4.5 §7.4 mini 映射表（cache_blocks 和生命周期）

| 我们的代码 | vLLM 源码 | 一致 / 简化 |
|-----------|----------|-----|
| `PrefixCacheManager.cache_blocks` (`prefix_cache_manager.py:L88-L119`) | `single_type_kv_cache_manager.py:L277-L301` | 幂等 guard 一字不差 |
| `PrefixCacheManager.evict_block` (`prefix_cache_manager.py:L125-L138`) | `block_pool.py:L354-L389` | 一致；暴露成直接调用 |
| `PrefixCacheManager.evict_blocks` (`prefix_cache_manager.py:L141-L143`) | `block_pool.py:L424-L441` | 批量包装 |
| `PrefixCacheManager.touch` (`prefix_cache_manager.py:L151-L154`) | `block_pool.py:L391-L406` | 去掉 metrics_collector |
| `PrefixCacheManager.free_request` (`prefix_cache_manager.py:L159-L166`) | `single_type_kv_cache_manager.py:L303-L318` | reverse-order free 一致 |

---

## 7.5 `prefix_aware_allocate` 端到端 + (N-1)×K 节省公式

### 7.5.1 打开端到端入口

源码定位：`kv_cache_manager.py:L183-L223` (`get_computed_blocks`) + `L225-L416` (`allocate_slots`)。

`get_computed_blocks` 只做 lookup，不分配。`allocate_slots` 做完整路径：lookup → touch → 新分配 → cache_blocks。**Scheduler 先调 get_computed_blocks 决定能不能上、再调 allocate_slots 真分配**——分两步是为了 budget gating（Ch04 §4.3 已经走读过 scheduler 的 budget 决策）。

### 7.5.2 K04：`max_cache_hit_length = num_tokens - 1` 的 `-1` 哪儿来

`kv_cache_manager.py:L208`：

```python
# vllm/v1/core/kv_cache_manager.py:L208 (节选)
max_cache_hit_length = request.num_tokens - 1
```

为什么减 1？想象一个极端：用户提交了一个**和之前完全一样**的 prompt——chain 100% hit。如果不减 1：

- find_longest_cache_hit 返回所有块 → `num_fresh_blocks = 0` → `allocate_new_blocks(0)` → 没分配新块。
- 但 decode 第 1 步要算 `logits = lm_head(hidden[-1])` —— `hidden[-1]` 是**最后一个 token 的位置**的 hidden state。这个 hidden 在 forward pass 里产生，需要 attention 真的算过这个位置。
- cache hit 的块只存了 K, V，**没有 forward pass 需要的 hidden state**。
- 结果：sampler 拿不到 logits → RuntimeError 或者更糟，silent NaN。

**减 1** 的作用：**强制保留至少一个 token 必须重算**，确保 forward pass 走至少一次，logits 有产出。代价：100% cache hit 的请求多算 1 个 token 的 attention（GPU 上 ~ 微秒级）—— FLOPs 和延迟都可忽略。

我们的实现 `paged_integration.py:L89`：

```python
# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_manager.py:L204-L208
max_hit = max(0, len(token_ids) - 1)
cached = mgr.find_longest_cache_hit(chain, max_length=max_hit)
```

`max(0, ...)` 防 `num_tokens == 0` 时返负数——边角但是必要。

### 7.5.3 prefix_aware_allocate 的完整 trace

`prefix_cache_manager.py:L190-L222`：

```python
# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_manager.py:L183-L223
def prefix_aware_allocate(
    mgr, request_id, token_ids, extra_keys=()
) -> tuple[list[int], int]:
    # ① 算链
    chain = chain_block_hashes(token_ids, mgr.block_size, extra_keys)
    # ② match
    cached = mgr.find_longest_cache_hit(chain, max_length=len(token_ids))
    # ③ touch（ref_cnt++，把块从 free queue 抽出）
    mgr.touch(cached)
    # ④ 计算还要补多少
    num_full_blocks = len(token_ids) // mgr.block_size
    num_fresh = max(0, num_full_blocks - len(cached))
    fresh = mgr.fresh_block_ids(num_fresh)
    # ⑤ 注册到 req_to_blocks，ref_cnt 累加
    blocks = cached + fresh
    mgr.register_request(request_id, blocks)
    # ⑥ 把 fresh 部分塞进 cache index 给后续请求用
    mgr.cache_blocks(request_id, len(token_ids), chain)
    return blocks, len(cached)
```

六步对应源码 `allocate_slots` 里的逻辑：lookup → touch → fresh allocate → cache_blocks。源码 (`kv_cache_manager.py:L380-L391`) 把"先 cached 后 fresh" 分成 `allocate_new_computed_blocks` + `allocate_new_blocks` 两个调用，我们合一了——但顺序一致：cached first, fresh after。

### 7.5.4 (N-1)×K 节省公式：78% 是这个公式的工作负载实例

K13。先讲公式，再讲数字。

**N 个请求，每个长度 = sys_prompt 长 + user_tail 长。设 sys_prompt 占 K 块，user_tail 占 U 块（每个请求独占）。**

- **Naive 分配**：每请求各自开 (K+U) 块。总计 `naive = N × (K + U)` 块。
- **Prefix-aware 分配**：第 1 个请求开 K + U 块（fresh），把 K 个 sys 块 cache 进去；第 2..N 个请求 hit 那 K 块、再开 U 块 fresh。总计 `pa = (K + U) + (N-1) × U`。
- **节省**：

$$
\mathrm{saved} = \mathrm{naive} - \mathrm{pa} = N(K + U) - (K + U) - (N-1)U = (N-1)K
$$

$$
\mathrm{saving\_ratio} = \frac{(N-1)K}{N(K + U)}
$$

**这就是 prefix cache 节省的全貌**——和 N、K、U 三个参数都有关。$N$ 大、$K$ 大（sys_prompt 长）、$U$ 小（user 部分短）→ 节省比例大。

**demo §4 的 78% 是怎么来的**——把 $N=50, K=32, U=8$ 代入：

$$
\mathrm{saved} = 49 \times 32 = 1568
$$

也就是 1568 块。

$$
\mathrm{saving\_ratio} = \frac{1568}{2000} = 0.784
$$

demo 输出：

```
allocation          fresh_blocks    kv_bytes_at_64KiB
naive                       2000             125.00 MiB
prefix-aware                 432              27.00 MiB
saved:                      1568              98.00 MiB  (78%)
```

**写法警告**（K13）：**不要把 78% 当成一个常数说出去。** 78% 只是 (50, 32, 8) 这个特定工作负载的取值。如果用户跑 N=2 或 K=4 → 节省比例完全不同。**先公式，后数字。** Ch13 cross-rank pooling 会再次用这个公式分析分布式收益。

测试侧验证（K13）：`test_savings_scale_with_shared_prefix` 跑了 3 个 sweep 点：

| N | K | U | 公式 saved | 实测 saved |
|---|---|---|---------|---------|
| 10 | 4 | 4 | 9 × 4 = 36 | 36 |
| 20 | 8 | 4 | 19 × 8 = 152 | 152 |
| 50 | 32 | 8 | 49 × 32 = 1568 | 1568 |

3 个全部 exact match——**线性 scaling 不是估算，是公式**。

### 7.5.5 语言陷阱 3：cache hit ≠ 请求免费

直觉错觉："命中了 prefix → 这个请求不用 GPU 资源了。"

**错。** cache hit 只省 **prefill 算力**和 **prefix KV 重写带宽**——**不省 decode 块**。

- 一个 prompt = 100 token + max_output = 200 token 的请求，假如 prompt 全 hit（100% hit）：
  - prefill 部分：FLOPs 省了，K, V 不重算（这是 prefix cache 真正的收益）。
  - decode 部分：仍然要分配 $\lceil 200/16 \rceil = 13$ 个 fresh block 装新生成 token 的 K, V。
- 块池吃紧时，**全 hit 的请求依然会因为 decode block 短缺被 admission 阻塞**——prefix cache 解不了这个问题。

这点很重要：scheduler 在 `kv_cache_manager.py:L380-L391` 里，`allocate_new_computed_blocks` 后**还会跑 `allocate_new_blocks`**。两个调用顺序固定但都必要。

**正确说法**：cache hit 节省 **prefill 计算 + KV 写入带宽**，不节省 **block pool 占用 / decode KV bandwidth**。Ch13 (prefix-cache-pooling) 会进一步讨论"cache 命中和 decode 块短缺如何同时发生"。

### 7.5.6 我们的实现 vs 源码差异

| 我们 | 源码 | 简化 |
|-----|-----|------|
| `prefix_aware_allocate` 一次性做 lookup→fresh→cache_blocks | `KVCacheManager.allocate_slots` 拆 4 个内部调用 | 单元更紧凑 |
| `fresh_block_ids` 单调递增 int 列表 | `BlockPool.get_new_blocks` 走 free queue + lazy evict | Ch05 真实路径 |
| `AllocateResult` 3 个字段 (block_ids, hit_blocks, hit_tokens) | `KVCacheBlocks` 9 个字段 | 测试需要的子集 |
| 单 group | 多 group + spec router | Ch12-13 |

---

## 7.6 PagedAttention 协同 + 主映射表 + 跨章衔接

### 7.6.1 三个 invariant 把 prefix cache 与 PagedAttention 锁住

回顾本章已经引出来的三个不变量，凑成 PagedAttention 协同的"三脚架"：

**I1 — Append-only block_id assignment.** 一旦 request A 拿到 `block_id=N`，N 永不变，直到 request 自己 free。**保证**：GPU kernel 读 block_table 永不读到错位的 KV。**来源**：`block_pool.py:L48-L52` NOTE。

**I2 — `ref_cnt` 反映活跃共享。** 块在 free queue 当且仅当 `ref_cnt == 0`；prefix-cache hit 时 `touch()` bump `ref_cnt` 并把块**从 free queue 中段**移除（O(1) 双向链表）；request 完成时 `ref_cnt--`，归 0 时块**回到 free queue 但保留 hash**。**保证**：N 个 request 共享同一块时它的 `ref_cnt = N`；驱逐 LRU 头永远不会驱逐"还在用的块"。**来源**：`block_pool.py:L391-L422`。

**I3 — chained hash 让 match length 单调。** `find_longest_cache_hit` 在第一个 miss 处 break。**保证**：扫描成本 = $O(\mathrm{hit\_length})$，不是 $O(L/B)$。**来源**：`single_type_kv_cache_manager.py:L473-L483`。

`paged_integration.py:verify_invariants` 在 demo §5 检查这三个：

```
I1_append_only_ids                PASS
I2_ref_cnt_consistent             PASS
I3_chain_monotone                 PASS
cache_size: 39 entries
requests:   10 alive
```

PASS/PASS/PASS 不是装饰——它意味着我们简化版的 manager 满足和源码同样的三个并行性 / 正确性约束。Ch13 cross-rank prefix cache 把这三个 invariant 推广到**多 rank 共同维护**——届时增加 I4 (cross-rank consistency)，三脚架变四脚架。

### 7.6.2 跨章衔接：Ch03 / Ch05 / Ch13 / Ch23

Ch07 不是孤立的——它是底层 PagedAttention（Ch03）+ BlockPool（Ch05）之上的一层，又是上层 prefix-cache pooling（Ch13）+ PD prefix cache（Ch23）的基础。

| 关系 | 章节 | 引用方向 |
|-----|-----|---------|
| **back-pointer** | Ch03 PagedAttention | block_table append-only invariant 的源头 |
| **back-pointer** | Ch05 BlockPool / FreeKVCacheBlockQueue | KVCacheBlock 三态、O(1) 中段移除 |
| **forward-pointer** | Ch12 Hybrid attention multi-group | `kv_cache_group_ids`、SlidingWindow 变体 |
| **forward-pointer** | Ch13 Prefix-cache pooling | (N-1)×K 在 cross-rank pooling 下的 envelope |
| **forward-pointer** | Ch23 PD prefix cache | prefill / decode 分离场景 |

**Ch13 forward-pointer**：今天我们用 `(N-1) \cdot K$ 公式描述单实例节省。Ch13 把 N 推广到"跨 N 个数据并行 rank 的请求池"——多 rank 共享 prefix 时怎么协调？答案是 chain hash 的 collision resistance 让"跨 rank 不需要协调" 直接成立。具体在那章展开。

**Ch23 forward-pointer**：PD 架构下 prefill 和 decode 在不同节点。Ch23 解释"prefix cache 跨 prefill/decode 节点搬运"的成本与设计。本章讲单节点 prefix cache，是 Ch23 的子集。

### 7.6.3 主映射表：Our code → vLLM source 1:1 (≥20 行)

| 我们的代码 | vLLM 源码 | 我们改了什么 | 为什么 |
|-----------|----------|-------------|--------|
| `BlockHash`, `BlockHashWithGroupId` (`block_hash.py:L39, L42`) | `kv_cache_utils.py:L40, L45` | 一字不差 NewType | typing |
| `init_none_hash` (`block_hash.py:L48-L54`) | `kv_cache_utils.py:L83-L110` | 去 cbor2 警告 | 简化依赖 |
| `make_block_hash_with_group_id` (`block_hash.py:L61-L69`) | `kv_cache_utils.py:L53-L62` | 一字不差大端 4 字节 | — |
| `get_block_hash` / `get_group_id` (`block_hash.py:L73-L79`) | `kv_cache_utils.py:L65-L72` | 一字不差 unpack | — |
| `hash_block_tokens` (`block_hash.py:L83-L109`) | `kv_cache_utils.py:L539-L566` | sha256(repr()) 替 sha256_cbor | 去依赖 |
| `chain_block_hashes` (`block_hash.py:L116-L140`) | `Request.update_block_hashes` (`request.py`) | 提纯函数 | 可测 |
| `common_prefix_length` (`block_hash.py:L145-L160`) | — | 新增 | demo §1 用 |
| `BlockHashToBlockMap` (`prefix_cache_index.py:L44-L125`) | `block_pool.py:L34-L127` | dict 值 int 替 KVCacheBlock | Ch05 边界 |
| `BlockHashToBlockMap.get_one_block` (`prefix_cache_index.py:L67-L76`) | `block_pool.py:L62-L73` | 一致 union 处理 | — |
| `BlockHashToBlockMap.insert` (`prefix_cache_index.py:L79-L95`) | `block_pool.py:L75-L91` | 一致 3 分支 | — |
| `BlockHashToBlockMap.pop` (`prefix_cache_index.py:L98-L118`) | `block_pool.py:L93-L121` | 一致 mismatch 兜底 | — |
| `get_cached_block` (`prefix_cache_index.py:L131-L151`) | `block_pool.py:L184-L209` | 一致多组 all-must-hit | — |
| `Trie` (`radix_tree.py:L58-L95`) | NOT in vLLM | 新增（教学对照） | bench |
| `RadixTree` (`radix_tree.py:L115-L231`) | NOT in vLLM | 新增（教学对照） | bench |
| `PrefixCacheManager` (`prefix_cache_manager.py:L37-L185`) | `SingleTypeKVCacheManager` (`single_type_kv_cache_manager.py:L30-L82`) | 单 group 折叠 | Ch12 多组 |
| `PrefixCacheManager.find_longest_cache_hit` (`prefix_cache_manager.py:L61-L83`) | `single_type_kv_cache_manager.py:L446-L494` | 单 group | Ch12 多组 |
| `PrefixCacheManager.cache_blocks` (`prefix_cache_manager.py:L88-L119`) | `single_type_kv_cache_manager.py:L277-L301` | 一致 idempotent guard | — |
| `PrefixCacheManager.evict_block` (`prefix_cache_manager.py:L125-L138`) | `block_pool.py:L354-L389` | 暴露成直接调用 | demo |
| `PrefixCacheManager.evict_blocks` (`prefix_cache_manager.py:L141-L143`) | `block_pool.py:L424-L441` | 一致批量 | — |
| `PrefixCacheManager.touch` (`prefix_cache_manager.py:L151-L154`) | `block_pool.py:L391-L406` | 去 metrics_collector | 可选 |
| `PrefixCacheManager.free_request` (`prefix_cache_manager.py:L159-L166`) | `single_type_kv_cache_manager.py:L303-L318` | 一致 reverse-order free | — |
| `PrefixCacheManager.fresh_block_ids` (`prefix_cache_manager.py:L178-L185`) | `BlockPool.get_new_blocks` (`block_pool.py:L322-L352`) | 单调 int 列表 | Ch05 真路径 |
| `prefix_aware_allocate` (`prefix_cache_manager.py:L190-L222`) | `KVCacheManager.allocate_slots` (`kv_cache_manager.py:L225-L416`) | 6 步合一 | 走读紧凑 |
| `AllocateResult` (`paged_integration.py:L52-L66`) | `KVCacheBlocks` (`kv_cache_manager.py:L21-L103`) | 3 字段子集 | 测试需要 |
| `get_computed_blocks` (`paged_integration.py:L70-L95`) | `kv_cache_manager.py:L183-L223` | 一致 max_hit=N-1 | K04 |
| `allocate_with_prefix_cache` (`paged_integration.py:L98-L122`) | `kv_cache_manager.py:L225-L416` | 完整 match→touch→fresh→cache | — |
| `verify_invariants` (`paged_integration.py:L130-L180`) | 多源合成（无单一源） | 新增 I1/I2/I3 三件套 | 诊断 |

**故意砍掉的内容**（每项指向后续章节）：

- 多 group `find_longest_cache_hit` → Ch12 hybrid attention。
- `cache_full_blocks` 的 KV-event 发射（distributed prefix cache 日志）→ Ch11+。
- `MambaManager.cache_blocks` / `CrossAttentionManager` → Ch15 / Ch20。
- `extra_keys` 完整组合（mm + lora + cache_salt + prompt_embeds）→ Ch08 多模态 / Ch09 LoRA。
- 真实 throughput benchmark 数字 → Ch13 (cross-rank pooling) 用真实硬件 sweep。
- `verify_invariants` 是乐观诊断（结构性检查）；生产审计要回放 request log。

### 7.6.4 4 个语言陷阱回顾（Ch06 §6.3.2 风格）

读到这里，重新过一遍开篇的 4 个语言陷阱——现在每个都有源码 + math 兜底：

**陷阱 1：vLLM 用 radix tree。** 错。`block_pool.py:L34-L127` 是 `dict`。"tree property" 由链式 hash (`kv_cache_utils.py:L539-L566`) 隐式给出。SGLang 的 RadixAttention 论文用真 radix tree，vLLM **没有**。`grep 'class.*Radix\|class.*Trie\|class.*PrefixTree' instances/vllm/source/` 在 commit `98661fe` 上**零匹配**。

**陷阱 2：Cache hit on partial block.** 错。`cache_blocks` (`single_type_kv_cache_manager.py:L286-L290`) 在 `num_full_blocks = num_tokens // block_size` 处用整数除法——partial block 永不进 cache。demo §1 hit_rate 永远比 shared_frac 略低就是这个原因（partial-block 余数）。

**陷阱 3：Cache hit makes the request free.** 错。cache hit 只省 prefill 算力 + KV 写带宽；decode 块照样要分配（`kv_cache_manager.py:L380-L391` 里 `allocate_new_blocks` 在 `allocate_new_computed_blocks` 之后**仍然跑**）。block pool 吃紧时，全 hit 的请求会因为 decode block 短缺被阻塞。

**陷阱 4：Need exhaustive search to find longest cached prefix.** 错。链式 hash 让 head-first 扫描 + break-on-first-miss **是**找最长前缀的正确算法。`single_type_kv_cache_manager.py:L473-L483` 一字不漏地写出来了。

---

## 验证

### 跑测试

```bash
cd instances/vllm/artifacts/07-prefix-cache
python3 -m pytest tests/ --ignore=tests/_legacy -q
```

预期输出：

```
83 passed in 0.49s
```

83 个测试覆盖 6 个模块：

| 模块 | 测试数 | 状态 |
|---|---|---|
| `test_block_hash.py` | 19 | PASS |
| `test_prefix_cache_index.py` | 14 | PASS |
| `test_radix_tree.py` | 12 | PASS |
| `test_prefix_cache_manager.py` | 18 | PASS |
| `test_paged_integration.py` | 11 | PASS |
| `test_integration.py` | 9 | PASS |

### 跑 lint

```bash
python3 scripts/lint_formulas.py instances/vllm/artifacts/07-prefix-cache/narrative/chapter.md
python3 scripts/lint_source_grounding.py instances/vllm/artifacts/07-prefix-cache/
```

两个 linter 都应当 0 阻塞。

### 跑 demo

```bash
python3 -m instances.vllm.artifacts.07-prefix-cache.implementation.demo
```

5 段输出对应本章 5 个核心数字：

- §1 hit_rate sweep：0.10 → 9.3%、0.50 → 49.5%、0.90 → 88.2%（partial-block 损失把每档拉低 1-2 个百分点）
- §2 微基准：Hash 373,348/s vs Trie 80,291/s vs RadixTree 77,232/s（4.65× 是 Python 解释器开销，不是渐近差）
- §3 chain-break trace：A 全 fresh、B 重用 2 块、evict A 块 0、D **hit 0** 全 fresh（链断裂 = THE invariant）
- §4 节省：naive 2000 块、prefix-aware 432 块、saved 1568 块、ratio 78.4%（公式 (N-1)×K = 49×32 = 1568 一字不差）
- §5 invariants：I1 PASS、I2 PASS、I3 PASS

---

## 总结

第 7 章把 vLLM v1 的 prefix cache 拆开了。**最重要的一句话**：vLLM 没有 radix tree——它用**链式 hash + 平面 dict + scan-and-stop** 实现了 radix tree 的"tree property"，而且代码量 / 复杂度比真 radix tree 少一个数量级。

四件值得记住的事：

1. **链式 hash 引理**：$H_k$ 由 $H_{k-1}$ 递推 → "miss at $k$ → miss at $k+1, \dots$"。这一条 lemma 让 `find_longest_cache_hit` 是个 7 行的扫描函数，不是 70 行的 tree walk。

2. **三态生命周期 (K06)**：cached-in-use / cached-idle / uncached-idle。`touch` / `free_blocks` / `_maybe_evict_cached_block` 三个 API 转换三态。FreeKVCacheBlockQueue 是 doubly-linked list（Ch05 K05）让 `touch` O(1) 中段移除——没这个数据结构 prefix cache 命中路径就退化成 O(n)。

3. **(N-1) × K 节省公式 (K13)**：N 个请求共享 K 块 sys_prompt → 节省 $(N-1) \cdot K$ 块。78% 是 (50, 32, 8) 这个工作负载下的取值——**先公式，后数字**。生产部署时把自己的 N、K、U 代进去。

4. **三个 invariant**：I1 append-only block_id（GPU kernel 不会读错位）；I2 ref_cnt 反映共享（free queue 永不含 in-use 块）；I3 chain monotone（扫描线性）。这三个共同把 prefix cache 跟 PagedAttention 锁在一起——少一个就出现"输出 silently corrupted"或"cache hit 退化"。

**4 个语言陷阱也别再犯**：vLLM 没有 radix tree、partial block 永不命中、cache hit ≠ 请求免费、扫描不是穷举。

### 下章预告

第 8 章 `Tensor Parallelism` 把单 GPU 上的所有机制——KV cache、batching、scheduling、prefix cache——往多 GPU 上推。本章的 chain-hash collision resistance 在那里再次出现：跨 GPU rank 的 prefix-cache 一致性**根本不需要协调**——同样的 token + extra_keys 在每个 rank 上 hash 出同样的 $H_k$，dict.get 各自命中，无锁。这是 chain hash 真正的设计杀招——不是性能，是**不需要分布式同步**。

更远处：第 13 章 `Prefix Cache Pooling` 把 (N-1) × K 公式推广到跨 rank 的请求池，定量分析多机情况下的节省 envelope；第 23 章 `PD Prefix Cache` 处理 prefill / decode 分离架构下"prefix 怎么搬"的问题。Ch07 是它们的基础——每一个细节本章已经埋好。

---

← 第 6 章：请求调度策略 | 第 8 章：Tensor Parallelism →
