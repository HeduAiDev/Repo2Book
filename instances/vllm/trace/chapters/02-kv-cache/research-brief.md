# Research Brief: KV Cache 内存模型与实现

## 演进时间线

| 时间 | 事件 | 关键贡献 |
|------|------|---------|
| 2017.06 | Transformer 论文 (Attention Is All You Need) | 提出 self-attention，但训练时 teacher forcing，无需 KV Cache |
| 2018-2019 | GPT-2 / 自回归推理普及 | 实践中发现：每次 decode 重算所有 K/V 导致计算量 O(n²) → KV Cache 概念自然出现 |
| 2019.09 | Multi-Query Attention (MQA, Shazeer) | 多个 Q head 共享 1 个 KV head → KV Cache 体积压缩为 1/h |
| 2023.05 | Grouped-Query Attention (GQA, Ainslie et al.) | MHA 与 MQA 的折中：Q heads 分组，每组共享 KV → 精度-内存 tradeoff 可控 |
| 2023.06 | PagedAttention 论文 (vLLM, Kwon et al., SOSP 2023) | KV Cache 分页管理，block_size=16，消除 0-1.6% 内部碎片；block table 类比 OS 页表 |
| 2024.03 | vLLM V0 进入维护期 | Block Manager v2 成为默认（#8704），随后 V0 完全移除（#25321） |
| 2024.10 | vLLM V1 Alpha (#9289, Woosuk Kwon) | KVCacheManager 从零重写：108 行，极简设计，预分配策略 |
| 2024.12 | Hybrid Memory Allocator (#17996) | 支持不同 block_size 的 KV cache group（Mamba + Attention 混合） |
| 2025.01 | Prefix Cache block hashing 升级 (#20511) | SHA-256 + CBOR 可复现哈希，替代旧 int hash |
| 2025.02 | xxHash 高性能哈希 (#29163) | 前缀缓存哈希加速选项 |
| 2025.03 | Hybrid Allocator 多 block_size 支持 (#29143) | 同一模型不同 layer 使用不同 block_size 的 KV cache group |
| 2025.04 | KV Connector + HMA (#30166) | 混合 allocator + connector 架构，支持 P/D 分离、KV 传输 |
| 2025.04 | Scheduler full ISL 调度 (#37307) | 基于完整 ISL 的准入控制，避免 chunked prefill 过度接纳 |

## 关键论文

- **Attention Is All You Need** (Vaswani et al., 2017) — 提出 self-attention 机制。自回归解码中，每个新 token 的 Q 需要 attend 所有历史的 K/V。朴素做法是每步重算，复杂度 O(n²d)。KV Cache 的核心直觉：**K/V 对已生成的 token 是不变的，不需要重算**。

- **Fast Transformer Decoding: One Write-Head Is All You Need** (Shazeer, 2019) — 提出 MQA：多个 Q head 共享同一组 K/V head。在 KV Cache 上下文中，这意味着 KV 体积从 `num_heads × head_dim × seq_len × 2` 变为 `1 × head_dim × seq_len × 2`，压缩比 = num_heads。代价是注意力精度下降，但对 decode 影响较小。

- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023) — GQA 将 Q heads 分成 G 组，每组共享 K/V。Llama 2 70B 使用 8 组，KV Cache 压缩比 = num_heads / 8。这是 Llama 系列广泛采用的方案。

- **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., SOSP 2023) — vLLM 的核心论文。三大贡献：
  1. **Block-based 分配**：将 KV Cache 分成固定大小 block（默认 16 token），类比 OS 虚拟内存分页
  2. **Block Table**：逻辑位置→物理 block_id 的映射，允许非连续存储
  3. **Prefix Sharing**：相同 prompt prefix 的多个请求可共享 KV block（通过 ref_cnt）
  实验表明：相比 contiguous 方案，block-based 内存浪费从 ~73% 降至 <4%。

- **SGLang: Efficient Execution of Structured Language Model Programs** (Zheng et al., 2024) — 提出 RadixAttention：用 Radix Tree（压缩前缀树）管理 prefix cache，LRU 淘汰。与 vLLM 的 block hash 方式相比，RadixAttention 天然支持前缀共享的层级结构。

## vLLM 源码考古：KV Cache 模块的演变

### Phase 1: V0 Block Manager (2023.06 - 2024.10)

V0 的 block manager 经历了两个版本：

```
vllm/core/block_manager.py    → BlockManagerV1 (简单列表，无 prefix caching)
vllm/core/block_manager_v2.py → BlockManagerV2 (hash-based prefix caching)
```

- V1: 每个 request 维护自己的 `block_ids` 列表，通过 `ref_cnt` 跟踪共享
- V2: 引入 `BlockHash` 和 prefix cache lookup，用 Python dict 做 `{block_hash: block}`
- 问题：V0 架构是单线程，block manager 与 scheduler 紧耦合，扩展性受限

### Phase 2: V1 KVCacheManager 初版 (2024.10, PR #9289)

Woosuk Kwon 的 V1 重构，初始 108 行：

```python
class KVCacheManager:
    def __init__(self, block_size, num_gpu_blocks, sliding_window=None,
                 enable_caching=True, num_preallocate_tokens=64):
        self.free_block_ids = list(range(num_gpu_blocks))       # ★ 简单列表
        self.req_to_block_ids: Dict[str, List[int]] = {}        # ★ 请求→block映射
        self.ref_cnts = np.zeros(num_gpu_blocks, dtype=np.int32) # ★ 引用计数
```

核心设计在当时已经出现：
- **预分配策略**：`num_preallocate_tokens=64` → 提前分配 block，减少每步的分配开销
- **引用计数共享**：多个请求共享同一 prefix block 时 `ref_cnt > 1`
- **简单 free block 管理**：`list.pop()` 分配，`list.append()` 释放

但缺失了关键功能：hash-based prefix caching、LRU 淘汰、多 KV cache group。

### Phase 3: 当前 V1 架构 (2024.12 - 至今)

当前 KVCacheManager 已经从 108 行增长到 ~1000+ 行，关键演变：

| PR | 变更 | 影响 |
|-----|------|------|
| #17996 | Hybrid Memory Allocator | 支持不同 block_size 的 KV cache group |
| #20511 | SHA-256 + CBOR block hash | 确定性的可复现哈希，支持 KV cache 跨节点重建 |
| #24964 | 预构造空 KVCacheBlocks | 避免 Python GC 开销 |
| #29143 | 多 block_size group 支持 | Hybrid 模型（Attention + SSM）的不同 block_size |
| #30166 | KV Connector + HMA | P/D 分离，外部 KV 连接器支持 |
| #37307 | Full ISL 调度 | 基于完整序列长度的准入控制 |

当前架构层次：

```
Scheduler
  └→ KVCacheManager (接口层)
       ├→ KVCacheBlocks (分配结果封装)
       ├→ KVCacheCoordinator (多 group 协调)
       │    └→ SingleTypeKVCacheManager (单 group 管理)
       │         ├→ BlockPool (物理 block 池)
       │         │    ├→ BlockHashToBlockMap (hash→block 映射，支持去重)
       │         │    └→ FreeKVCacheBlockQueue (双向链表，O(1) 中间删除)
       │         ├→ KVCacheBlock (元数据：block_id, ref_cnt, block_hash, prev/next)
       │         └→ Prefix Caching (find_longest_cache_hit 算法)
       └→ KVCacheMetricsCollector (指标收集)
```

### KVCacheBlock 数据结构（源码 kv_cache_utils.py:114）

```python
class KVCacheBlock:
    block_id: int                  # 物理 block 编号 0 ~ num_gpu_blocks-1
    ref_cnt: int = 0              # 引用计数（被几个 request 共享）
    _block_hash: BlockHashWithGroupId | None = None  # 前缀缓存哈希（满 block 时设置）
    prev_free_block / next_free_block  # 双向链表节点（FreeKVCacheBlockQueue 专用）
    is_null: bool = False         # padding block，永不缓存
```

### FreeKVCacheBlockQueue（源码 kv_cache_utils.py:162）

不是 Python `deque`，而是**自实现双向链表**，直接在 KVCacheBlock 上操作 `prev_free_block`/`next_free_block` 指针。原因：
- `deque` 不支持 O(1) 删除中间元素
- LRU 淘汰需要在队列任意位置删除被淘汰的 block
- 避免 Python 对象分配开销（操作现有属性，不 new 对象）

队列顺序：按 block_id 初始排序 → 释放时按 LRU 顺序追加回队尾。

### Prefix Caching 流程（源码 kv_cache_manager.py:183-223）

```
get_computed_blocks(request):
  1. 检查 enable_caching 和 skip_reading_prefix_cache
  2. coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
     → 在 BlockHashToBlockMap 中查找最长前缀匹配
  3. 返回 (computed_blocks, num_new_computed_tokens)
  4. 记录 prefix cache stats (hit rate, preempted)
```

## 设计决策树

```
KV Cache 内存管理
├── Contiguous Allocation (TensorRT-LLM)
│   ├── 原理：每个请求分配一段连续显存，seq_len 增长时整体搬迁或预分配 max_seq_len
│   ├── 优点：实现简单，GPU kernel 直接连续访问，无 block table 查询开销
│   ├── 缺点：外部碎片严重（73%+ 浪费），preemption 代价高（整段换出），prefix sharing 只能整段共享
│   └── 适用场景：单 batch、固定 seq_len 的高吞吐场景
│
├── Radix Tree Prefix Cache (SGLang)
│   ├── 原理：所有请求的 token 序列组织成一棵 Radix Tree（压缩前缀树），每个节点对应一段共享 token
│   ├── 优点：天然支持层级式 prefix sharing，LRU 淘汰粒度细（tree node 级别）
│   ├── 缺点：tree 操作复杂度 O(log n)，需要维护 tree 结构，多 GPU 共享复杂
│   └── 参考：SGLang RadixAttention，每个 tree node 包含完整 KV block 列表
│
├── Block-based Allocation (vLLM)
│   ├── 原理：KV Cache 分成固定大小 block (默认 16 token)，通过 block table 做逻辑→物理映射
│   ├── 与 OS 虚拟内存的类比：
│   │   ├── block = page (固定大小分配单元)
│   │   ├── block_table = page table (逻辑→物理映射)
│   │   ├── ref_cnt = reference count (共享检测)
│   │   └── preemption = page eviction (剔除到 CPU/NVMe)
│   ├── 优点：
│   │   ├── 零外部碎片（每个 block 独立分配，无连续空间要求）
│   │   ├── 内部碎片 ≤ block_size-1 token（block_size=16 → ≤15 token ≤0.1% for 16K seq）
│   │   ├── 自然支持 prefix sharing（同一 hash 的 block 被多个 request ref）
│   │   └── Preemption 高效（逐 block 换出，而非整段序列）
│   ├── 缺点：
│   │   ├── Block table 查询开销（GPU kernel 需要查表）
│   │   ├── 非连续显存访问（影响 memory coalescing）
│   │   └── 元数据管理开销
│   └── vLLM 选择：block-based — 因为推理服务本质上需要处理可变长度、动态到达的序列
│
└── LMDeploy (TurboMind)
    ├── 原理：使用 contiguous KV cache + paged attention 混合策略
    ├── 特点：更激进的量化（w4a16 KV Cache）、更小的 page size
    └── 取舍：牺牲部分灵活性换取更低的显存占用
```

### block_size=16 的来历

`CacheConfig.DEFAULT_BLOCK_SIZE: ClassVar[int] = 16`（config/cache.py:45）

这个数字来自 PagedAttention 论文的实验分析：
1. **太小**（如 8 token）：block 数量翻倍 → block_table 变大 → GPU 查表开销增大 → FreeKVCacheBlockQueue 链表操作增加
2. **太大**（如 64 token）：内部碎片增加 → 每个请求平均浪费 block_size/2 token → 小请求（如 100 token）浪费显著
3. **16 的权衡**：
   - 对于 2048 token 的典型 decode，内部碎片率 ≤ 15/2048 = 0.73%
   - block_table 大小合理（2048/16 = 128 个 block per request）
   - GPU cache line 对齐友好（16 × 4096 bytes = 64KB for BF16/h=64/d=128 → 4 个 128B cache line）
   - 实际上 16 是一个巧合+工程经验的结果，vLLM 现在也支持自定义 block_size

后续演进：V1 引入了 `hash_block_size` 作为前缀缓存的哈希粒度（可以比物理 block_size 更细），以及 hybrid allocator 支持同一个模型不同 layer 使用不同 block_size。

## 给 Writer 的建议

### 最重要的 intuition
KV Cache 的本质是**时间换空间的经典 tradeoff 之反用**——我们用**空间换时间**。不是 tradeoff 的 tradeoff：自回归推理中，K/V 一经算出就不会改变（因果掩码确保当前 token 看不到未来），所以缓存它们不是"可能有用"，而是"必然有用"。这个必然性让 KV Cache 成为推理优化中少有的"无脑正确"的优化。

### 最适合零基础读者的切入角度
从**单 token decode 的计算量爆炸**入手：
1. 不放任何缓存，第 n 个 token decode 要算 n 次 K/V → 总 O(n²)
2. 如果把前 n-1 个 token 的 K/V 存起来，第 n 个 token 只算新的 Q、K、V → 总 O(n)
3. 存 K/V 需要显存，显存是有限的 → 这就引出分配管理问题
这个梯子比直接讲 block table 好得多。

### 容易混淆的概念
- **KV Cache ≠ Prefix Cache**：KV Cache 是机制（存 K/V 避免重算），Prefix Cache 是策略（多个请求共享相同 prefix 的 KV Cache）。Ch02 应该聚焦 KV Cache 机制本身，Prefix Cache 是 Ch03 或更后面的内容。
- **block_size 指的是 token 数，不是 byte 数**：一个 block 存储 `block_size × num_heads × head_dim × num_layers × 2 (K+V) × dtype_size` 字节。
- **Block Table 不是 Attention 的"mask"**：Block Table 是物理地址映射表，和传统的 attention mask（causal/padding）是不同的概念。

### 建议的类比/生活例子
- **Block-based 分配 ≈ 图书馆书架**：书（token）按固定数量放入书盒（block），每个书盒有一个编号（block_id）。读者的借阅单（block_table）只记录书盒编号，不关心书盒在图书馆的物理位置。当一本书的前半部分被多个读者共享时（prefix sharing），只需要一个书盒 + 引用计数，而不是每个读者复印一份。
- **Contiguous vs Block-based ≈ 停车场的连续车位 vs 散位**：连续车位要求所有车首尾相接，来一辆大车就得整体挪动；散位（按 block 分页）允许车停在任何空位，通过"车位号纸条"（block table）记录每辆车的各个部分分别停在哪里。

### vLLM 源码阅读路线（给 Writer 参考，按理解深度递进）
1. `config/cache.py:L45` — `DEFAULT_BLOCK_SIZE = 16`，一切的起点
2. `v1/core/kv_cache_utils.py:L114-159` — `KVCacheBlock` 数据结构，理解元数据
3. `v1/core/kv_cache_utils.py:L162-250` — `FreeKVCacheBlockQueue`，理解分配/释放
4. `v1/core/kv_cache_manager.py:L106-170` — `KVCacheManager.__init__`，理解整体架构
5. `v1/core/kv_cache_manager.py:L225-320` — `allocate_slots()`，理解分配流程（最重要）
6. `v1/worker/block_table.py:L18-80` — `BlockTable`，理解 GPU 侧的 block table 实现
7. `v1/core/block_pool.py:L34-55` — `BlockHashToBlockMap`，理解 prefix caching（可选，属于进阶内容）
