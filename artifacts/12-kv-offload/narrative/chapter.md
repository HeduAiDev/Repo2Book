# 第12章：KV Cache Offload — 层级存储

> GPU HBM 只有 80 GB。当 KV 缓存需要 120 GB——超长上下文 + 大批量——怎么办？
> 买更多 GPU？不是每个团队都负担得起。vLLM 的答案：**把 KV 缓存搬一部分到 CPU 内存。**
> 不是"不用了就扔"，而是"暂时不用就存 CPU，需要时再拉回来"。

---

## 这章要做什么？

前 11 章的所有优化——PagedAttention、Prefix Cache、Chunked Prefill、DCP——都在 GPU 显存内打转。它们提高了显存利用率，但没有改变 80 GB 的总量上限。

KV Cache Offload 突破了这个上限。它把 CPU DRAM（512 GB+）当作 GPU HBM 的**第二级存储**——不是替换，是扩展。GPU 上保持热点 KV blocks，CPU 上存储冷 blocks，PCIe 作为两者之间的"总线"。

这不是免费的——PCIe Gen5 x16 的 64 GB/s 带宽比 GPU HBM 的 2 TB/s 慢了 30 倍。但 offload 的设计使这个瓶颈几乎不可见。

学完这章你能：
- 追踪一次 offload 的完整生命周期：prepare_store → DMA transfer → complete_store → lookup → prepare_load → complete_load
- 理解为什么 LRU 是 offload 的默认驱逐策略——不是 ARC，不是 LFU
- 量化 PCIe 带宽的开销——为什么 per-step offload 时间可以隐藏在 GPU compute 下

---

## 12.1 层级存储模型

### Theory: 三级存储的延迟阶梯

```
GPU HBM (80 GB)          CPU DRAM (512 GB)        NVMe SSD (2 TB)
  BW: 2 TB/s               BW: 50 GB/s              BW: 7 GB/s
  Latency: ~100 ns          Latency: ~100 ns          Latency: ~10 µs
       ↕ PCIe Gen5 x16 (~64 GB/s)    ↕ (NOT IMPLEMENTED)
       └── vLLM offload ──┘
```

vLLM 的 offload 只跨越 GPU↔CPU。没有 NVMe 第三级——`/dev/shm`（tmpfs，RAM 磁盘）是最接近的"持久化"层。

### Source Trail

打开 `vllm/config/cache.py:167`：

```python
kv_offloading_size: Optional[int] = None  # 单位：GiB
```

当这个值 >0，`VllmConfig._post_init_kv_transfer_config()` 自动注入 `kv_connector = "OffloadingConnector"`。

打开 `vllm/v1/kv_offload/cpu/spec.py:57-83`——`CPUOffloadingSpec.get_manager()`：

```python
num_cpu_blocks = cpu_bytes_to_use // kv_bytes_per_offloaded_block
manager = CPUOffloadingManager(num_blocks=num_cpu_blocks)
```

CPU 上的 block 数量由 `kv_offloading_size` 和单块大小计算。单块大小可以与 GPU block_size 不同——通过 `block_size_factor` 控制（CPU block 可以是 GPU block 的倍数，减少管理开销）。

---

## 12.2 LRU vs ARC：驱逐策略的选择

### Source Trail

打开 `vllm/v1/kv_offload/cpu/policies/lru.py:1-47`——整个 LRU 策略不到 50 行：

```python
class LRUCachePolicy(CachePolicy):
    _blocks: OrderedDict[OffloadKey, BlockStatus]

    def evict(self, n: int, protected: Set[OffloadKey]):
        """从 OrderedDict 头部开始驱逐——最早插入的最先驱逐"""
        for key in list(self._blocks.keys()):
            if len(evicted) >= n: break
            if key in protected: continue
            if self._blocks[key].ref_cnt == 0:
                evicted.append(key)
        return evicted
```

ARC 策略（`arc.py`）长得多——157 行，维护四个列表（t1/t2/b1/b2），自适应调节 recency 与 frequency 的权重。

### Theory: 为什么默认是 LRU 而不是 ARC？

ARC 在理论上有更好的命中率——它根据访问模式自适应调节 recency/frequency 权重。但 KV Cache offload 的访问模式有一个关键特征：**system prompt prefix 是访问的主要模式。**

一个 system prompt 的 prefix block 被每个新会话命中一次——然后不再访问（因为 prefix cache 在 GPU 侧也命中了）。流式对话的中间 blocks 被访问一次——然后随会话结束而失效。真正的"热"blocks——被反复访问的——极少。

在这种模式下，LRU 和 ARC 的命中率差异很小——LRU 已经自然地把"只用一次"的 blocks（流式 token）推到驱逐队列前端。ARC 的额外复杂度（4 个队列，自适应的 t1/t2 比例调节）不值得它的代码维护成本。

`★ Insight ─────────────────────────────────────`
LRU 被广泛低估了。在具有"幂律访问分布"的系统中（多数 block 访问 1-2 次，极少数访问 N 次），LRU 的最优性差距在 10% 以内。KV Cache offload 的访问模式恰好满足这个条件——system prompt prefix 被多个请求命中（高频率），中间 token 只被生成它的请求使用一次（低频率）。LRU 自然保持高频率的 prefix blocks 在队列尾部（MRU 端），驱逐低频率的中间 blocks。ARC 的多队列自调节在这个场景下几乎没有额外收益。
`─────────────────────────────────────────────────`

---

## 12.3 Offload 生命周期

### Source Trail

打开 `vllm/v1/kv_offload/cpu/manager.py`——`CPUOffloadingManager`。

**阶段 1 — prepare_store（标记"要存"）：**

```python
def prepare_store(self, keys):
    new_keys = [k for k in keys if k not in cache]     # 去重
    num_to_evict = len(new_keys) - num_free_blocks
    if num_to_evict > 0:
        evicted = self.policy.evict(num_to_evict)        # LRU 驱逐
    for key in new_keys[:num_free]:
        block = allocate_block()
        block.ref_cnt = -1  # ← 标记为 "未就绪"
        self.policy.insert(key, block)
```

`ref_cnt = -1` 的含义：这个 block 已经在 CPU 内存中分配了，但**数据还没从 GPU 传过来**。任何 lookup 看到 `ref_cnt = -1` 都返回 miss——即使 key 已经在 cache 中。

**阶段 2 — DMA Transfer（异步传输）：**

Worker 侧（`gpu_worker.py`）使用**独立的低优先级 CUDA stream** 做 `cudaMemcpyAsync`：

```python
# 在专用 stream 上——不阻塞 compute stream
with torch.cuda.stream(self.offload_stream):
    swap_blocks_batch(src_gpu_ptrs, dst_cpu_ptrs, sizes)
```

CUDA 12.8+ 使用 `cuMemcpyBatchAsync`——单次 driver call 批量提交所有 memcpy。

**阶段 3 — complete_store（标记"就绪"）：**

```python
def complete_store(self, keys):
    for key in keys:
        block = self.policy.get(key)
        block.ref_cnt = 0  # ← 现在可读了
```

**阶段 4 — lookup（检查 CPU cache 命中）：**

```python
def lookup(self, key):
    block = self.policy.get(key)      # 检查存在
    return block is not None and block.is_ready  # 且已就绪
```

### Theory: 为什么 store 是 deferred？

Store 操作在**当前 step 的 model forward 完成后**、**下一个 step 的 scheduling 开始前**执行。这避免了 store DMA 和当前 token 的 generation 竞争 PCIe 带宽。这个设计把 store 延迟隐藏在"model forward 和下一个 scheduling step 之间的间隙"中——这个间隙通常在 0.5-2 ms，对于典型 offload 传输 (<0.1 ms) 完全足够。

---

## 12.4 PCIe 带宽与 Overlap

### Theory: Per-Step 传输量

以 DeepSeek V3（1 KV head MLA，head_dim=128, 64 layers, bf16）为例：

**Per-token 存储（1 个 decode token）：**
```
kv_bytes = 2 × 1 × 128 × 2 × 64 = 32,768 bytes ≈ 32 KB
Store time @ PCIe Gen5 x16 (64 GB/s): 32 KB / 64 GB/s = 0.5 µs
```

**Per-block 预取（16 token block）：**
```
block_bytes = 2 × 16 × 128 × 2 × 64 = 524,288 bytes ≈ 512 KB
Load time: 512 KB / 64 GB/s = 8 µs
```

**结论：** Store 是 <1 µs，load 是 <10 µs per block。一个典型的 decode step 的 GPU compute 是 2-5 ms。**Offload DMA 开销是 GPU compute 的 <1%——完全隐藏在异步 stream 之下。**

### Source Trail

`reuse_manager.py` 的 `store_threshold=2`：一个 block 必须被 lookup 至少 2 次才允许 offload。这防止了流式 token（只用一次）污染 CPU offload 池——只有 system prompt 的 prefix blocks（被多个会话重用）才值得占用珍贵的 CPU DRAM 空间。

---

## 我们实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `LRUPolicy` | `cpu/policies/lru.py` | OrderedDict + evict/touch/insert 一致 |
| `CPUOffloadingManager` | `cpu/manager.py` | prepare_store→complete_store→lookup 生命周期一致；简化了 multi-worker fence |
| PCIe 分析 | 原创——基于 vLLM 的 DMA 架构 | 量化 per-step 开销 |
| `OffloadBlock` | `cpu/policies/base.py:L10` `BlockStatus` | 简化版——无 C struct 优化 |

---

## 验证

```bash
cd artifacts/12-kv-offload && python -m pytest tests/ -q
# 12/12 passed ✅
```

---

## 总结

- **KV Offload = GPU HBM + CPU DRAM 两级存储。** PCIe 作为 bus——store 到 CPU、load 回 GPU。
- **LRU 驱逐是默认——不是因为没有更好的，而是足够好。** 幂律访问分布下 LRU 与 ARC 的差距 <10%。
- **Store 是 deferred——在 step 间隙执行。** 避免与 token 生成竞争 PCIe 带宽。
- **Per-step DMA <1% GPU compute 时间。** 异步独立 CUDA stream → 完全重叠。

---

**下一章：** 第13章 — Prefix Cache 跨请求共享与池化

Offload 把冷 KV blocks 存到 CPU。但如果有 1000 个并发请求全用相同的 system prompt——CPU 上存 1000 份副本？Prefix Cache Pooling 的答案：全局共享池——一份 system prompt，所有请求共享。第 13 章将分析分布式 prefix cache 的 hash ring 和跨节点 cache coherence。

---

← 第11章 | 第13章 →
