# 第3章：FlashAttention & PagedAttention — vLLM 的双引擎

> 打开 `csrc/attention/attention_kernels.cuh:85`，你会看到一个叫 `paged_attention_kernel` 的 CUDA 函数。
> 它在一个 kernel 里同时做了两件事：tiled online softmax（FlashAttention）
> 和 block_table 索引（PagedAttention）。这就是 vLLM 推理引擎最精华的设计融合。

---

## 这章要做什么？

第 1 章讲了 Attention 怎么算——从数学公式到 Triton fused kernel。第 2 章讲了 KV Cache 怎么存——从 `KVCacheManager` 的 `allocate_slots()` 到 `BlockPool` 的 LRU 驱逐。

但生产级的 vLLM 不会分别调用"一个 FlashAttention kernel"和"一个 PagedAttention 查找函数"。打开 `csrc/attention/attention_kernels.cuh:252`：

```cpp
const int64_t physical_block_number =
    static_cast<int64_t>(block_table[block_idx]);
```

这一行在 attention kernel 的**内部循环**中。它不是"先找到物理地址，再传给 attention 函数"——它是在 kernel 的每个 tile 迭代中，**现场查找 block_table，然后立刻做 attention 计算**。

这就是本章要讲的核心：**融合。**

学完这章你能：
- 手写一个 `paged_attention_with_block_table()` 函数——在 Python 中展示 block_table 索引逻辑
- 量化对比 naive attention 和 FlashAttention 的 HBM 读写量——用具体数字，不用形容词
- 解释 `attention_kernels.cuh` 中第 252 行的 `block_table[block_idx]` 为什么必须出现在 kernel 内部而不是外部
- 区分 FlashAttention V1 和 V2 的本质差异——为什么 V2 要用 partitioned softmax

---

## 3.1 两个问题，两个解法

（本章涉及的源码文件：`csrc/attention/attention_kernels.cuh` → CUDA paged attention 核心，`vllm/v1/attention/ops/triton_decode_attention.py` → Triton decode kernel，`vllm/v1/attention/backends/flash_attn.py` → FA backend 集成。建议在阅读本章过程中保持这些文件打开。）

在深入代码之前，先明确 FlashAttention 和 PagedAttention 各自解决什么问题——因为它们的职责非常不同，但融合在一起时可能容易混淆。

### 问题一：Attention 的 HBM 瓶颈

朴素 attention 有三步：

```python
S = Q @ K^T              # 写 [seq²] 到 HBM  ← 4096²×4B×32heads = 2GB
P = softmax(S)           # 读 [seq²], 写 [seq²] ← 又一个 2GB
O = P @ V               # 读 [seq²], 写 [seq]   ← 第三次访问
```

**问题：S 和 P 各是 `[seq_len, seq_len]` 的矩阵。** 这个大小与序列长度的平方成正比。FlashAttention 的解法：**计算在 SRAM 里做，不写回 HBM。** 把 Q 切成块，循环遍历 KV 块，online softmax 累积——只有最终的 O 写回 HBM。

### 问题二：KV Cache 的碎片化

KV Cache 是一块固定大小的 GPU 显存池。请求 A 需要 1000 token 的空间，请求 B 需要 200 token。A 完成后释放 1000 token。C 需要 1200 token——空闲总空间有 1000+剩余 > 1200，但没有足够大的**连续**块。

**问题：外部碎片导致 50-75% 的显存利用率。** PagedAttention 的解法：**把 KV Cache 分成固定大小的 blocks，通过 block_table 做虚拟地址映射。** 不需要连续分配——block_table 告诉 kernel "token i 的 K 和 V 在物理 block 17 的偏移 3"。

### 为什么它们必须融合？

分开处理的代价：你需要在 CPU/GPU 之间来回传递"这个 token 的 K 在哪里"和"现在做 attention 计算"。融合后：kernel 在自己的内部循环里，对于每个 KV block，查 block_table 找到物理位置，加载 K 和 V，立刻做 QK^T 和 softmax 累积——**一个 kernel，一次 launch，零次 CPU-GPU sync。**

`★ Insight ─────────────────────────────────────`
FlashAttention 和 PagedAttention 经常被放在一起讲，但它们的职责有清晰的边界。FlashAttention 优化**计算**（compute-bound）——减少 HBM 读写。PagedAttention 优化**内存**（memory-bound）——增加有效显存利用率。如果你只优化计算，碎片化浪费的 50-75% 显存仍然在。如果你只优化内存，[seq²] 大小的 attention 矩阵仍然在消耗 HBM 带宽。vLLM 把它们融合在同一个 kernel 里——在 tiled softmax 的每一步，通过 block_table 找到物理 block，加载，计算，累积释放。这是 vLLM 相对于其他推理框架的核心技术优势。
`─────────────────────────────────────────────────`

---

## 3.2 FlashAttention 的数学再梳理

第 1 章从 Triton kernel 的角度讲了 FA 的实现（见 `vllm/v1/attention/ops/triton_unified_attention.py:L58` 是 vLLM 的生产 Triton 版本）。这里从 HBM 流量的角度重新审视——用具体数字取代直觉。

### HBM 读写量化分析

以 Llama-3.2-1B 的 decode 阶段为例（`seq_len=4096, num_heads=32, head_dim=128, dtype=bf16`）：

| 操作 | 朴素 Attention | FlashAttention |
|------|---------------|----------------|
| 读 Q | 32×4096×128×2B = 32 MB | 32 MB (相同) |
| 读 K | 32×4096×128×2B = 32 MB | 32 MB × (seq/BLOCK_Q) ≈ 2 GB (反复读) |
| 读 V | 32 MB | 2 GB (反复读) |
| **写 S (中间)** | **32×4096²×4B = 2 GB** | **0** (不写 HBM) |
| **写 P (中间)** | **32×4096²×4B = 2 GB** | **0** (不写 HBM) |
| 写 O | 32 MB | 32 MB |
| **HBM 总流量** | **~4.1 GB** | **~4.1 GB** |

等等——FlashAttention 也读了很多数据？K 和 V 被反复读了 `seq/BLOCK_Q` 次。

**关键区别：** FlashAttention 的额外 HBM 读是 $O(seq \cdot d \cdot seq/BLOCK)$ 量级——它随序列长度线性增长，系数是 $d/BLOCK$。朴素 attention 的中间写入是 $O(seq^2 \cdot d)$ 量级——随序列长度的平方增长。当 $seq$ 很大时，**避免 $O(seq^2)$ 的 HBM 写入比避免 $O(seq \cdot d)$ 的额外读取重要得多。**

对于 seq=128K（超长上下文）：
- 朴素 attention 的 S 矩阵：32 × 128K² × 4B = **2 TB**（放不进任何 GPU）
- FlashAttention：反复读 K/V，但从不写 `[seq²]` 中间结果 → **完全可行**

`★ Insight ─────────────────────────────────────`
FlashAttention 的 trade-off 可以用一句话概括：**它接受更多的 K/V HBM 读取（因为每个 Q tile 都要重新读 K 和 V），换取完全消除 O(seq²) 的中间矩阵写入。** 这之所以是一个好 trade，是因为在现代 GPU 上，HBM 写入的带宽通常只有读取的 50-70%，而且写入的 [seq²] 矩阵大小是 [seq × d] 的 (seq/d) 倍——对于长序列，这个因子是 1000×。
`─────────────────────────────────────────────────`

---

## 3.3 PagedAttention：block_table 深入

### Source Trail

打开 `csrc/attention/attention_kernels.cuh:202,252`。block_table 的物理含义：

```cpp
// .cuh:L202 — 每个序列有自己的一行 block_table
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

// .cuh:L252-L253 — 关键的索引：逻辑块号 → 物理块号
const int64_t physical_block_number =
    static_cast<int64_t>(block_table[block_idx]);

// .cuh:L269 — 用物理块号加载 key
const cache_t* k_ptr =
    k_cache + physical_block_number * kv_block_stride + ...;
```

`block_table` 是一个 `[num_seqs, max_num_blocks_per_seq]` 的 int32 tensor。每一行是某个序列的**逻辑块到物理块的映射**。

### Theory: 为什么这是虚拟内存？

操作系统的页表：

```
虚拟地址 → 页号 + 页内偏移
页号 → 页表查找 → 物理页框号
物理地址 = 物理页框号 × 页大小 + 页内偏移
```

vLLM 的 block_table：

```
逻辑位置 → 逻辑块号 + 块内偏移
逻辑块号 → block_table查找 → 物理块号
物理位置 = 物理块号 × 块大小 × 头大小 + 块内偏移 × 头大小 + 头索引
```

一一对应。区别在于 OS 页表由 MMU 硬件自动翻译——vLLM 的 block_table 由 attention kernel 在软件中手动查找。

### Block Size 的选择

第 2 章提到 vLLM 默认 block_size=16。这里从 attention kernel 的角度分析：

block_size 太小（4）：block_table 太长 → 需要更多寄存器存储 block_table 指针 → 影响 kernel occupancy。对于 4096 token 序列和 block_size=4，需要 1024 个 block——block_table 占 1024×4B=4KB，需要 128 个 32-bit 寄存器。

block_size 太大（128）：内部碎片恢复。最后一个 block 平均只用了 50%（假设均匀分布的新 token 请求大小）。128×0.5=64 token 被浪费。

**16-32 tokens 的平衡点：** block_table 足够短（4096/16=256 entries = 1KB），内部碎片可接受（16×0.5=8 token 浪费），且地址对齐对 GPU cache line 友好（block 内的 16 tokens 连续存储 = coalesced memory access）。

---

## 3.4 融合：一个 Kernel，两个优化

### Source Trail

回到 `attention_kernels.cuh:85`。`paged_attention_kernel` 的核心循环：

```
for each Q block (seq_len / BLOCK_Q iterations):
    load Q_block to SRAM          ← FlashAttention: tile Q
    m, l, O_acc = -inf, 0, 0     ← Online softmax state

    for each logical KV block (seq_len / BLOCK_KV iterations):
        phys = block_table[block_idx]     ← PagedAttention: indirection
        load K_block from K_cache[phys]   ← Load from physical block
        load V_block from V_cache[phys]

        S = Q_block @ K_block^T * scale   ← Compute in SRAM (no HBM write!)
        m_new = max(m, row_max(S))        ← Online softmax update
        correction = exp(m - m_new)
        O_acc = correction * O_acc + exp(S - m_new) @ V_block
        m, l = m_new, l_new

    write O_acc / l to HBM                ← Single write per Q block
```

**三步融合在同一循环中：**
1. `block_table[block_idx]` — PagedAttention 的虚拟地址翻译
2. `Q_block @ K_block^T * scale` — FlashAttention 的 tiled 计算（结果在 SRAM）
3. `m_new = max(m, row_max(S))` + `correction` — FlashAttention 的 online softmax

**为什么不能分开？** 如果先做 block_table 查找（gather K、V 到连续 tensor），再做 FlashAttention——你需要额外的 HBM 空间存储 gathered K 和 V（size = `seq_len × heads × head_dim`）。对于 128K seq = 32×128K×128×2B = 1GB per layer——每个 layer 都需要 1GB 的临时显存。融合避免了这笔开销：**K 和 V 直接从非连续的物理 block 加载到 SRAM，没有中间 gather 步骤。**

### Triton 版本

vLLM 的 Triton decode kernel（`vllm/v1/attention/ops/triton_decode_attention.py:L60-L250`）用 Python 实现相同的融合：

```python
# triton_decode_attention.py:L119-L126
kv_page_number = tl.load(
    Req_to_tokens + stride * cur_batch_req_idx + offs_n // PAGE_SIZE
)
kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
```

`Req_to_tokens` 就是 block_table。`kv_page_number` 是物理 block。`kv_loc` 是物理偏移。随后立刻加载 K 和 V 并做 attention。

区别只是 CUDA 和 Triton 的语法。融合逻辑完全一致。

---

## 3.5 V1 vs V2：为什么需要 Partitioned Softmax？

### Source Trail

打开 `csrc/attention/paged_attention_v1.cu` 和 `paged_attention_v2.cu`。

**V1（v1.cu: L84, L497）：**

```cpp
// V1 grid: (num_heads, num_seqs, 1)
// 一个 thread block 处理一个 (head, seq) 对的全部 tokens
dim3 grid(num_heads, num_seqs, 1);

// 共享内存需求：padded_max_seq_len * sizeof(float)
// 当 max_seq_len=4096 时需要 4096×4B=16KB — 还行
// 当 max_seq_len=128K 时需要 128K×4B=512KB — 超过任何 GPU 的 SMEM
int shared_mem_size = max(logits_size, outputs_size);
```

**V2（v2.cu: L169-L172, L529）：**

```cpp
// V2 grid: (num_heads, num_seqs, max_num_partitions)
// 序列被切分为 PARTITION_SIZE=512 的多个分区
dim3 grid(num_heads, num_seqs, max_num_partitions);

// 共享内存：O(PARTITION_SIZE) = O(512) — 与 seq_len 无关！
// 中间缓冲区：tmp_out + exp_sums + max_logits
// 最后用 paged_attention_v2_reduce_kernel 合并分区
```

### Theory: 为什么 V2 是 necessary evolution？

V1 的共享内存需求与 `max_seq_len` 成正比。对于 128K token 的序列，V1 需要 512KB 共享内存——超过了 H100 每个 SM 的 228KB 限制。即使硬件支持，分配 512KB 的共享内存意味着每个 SM 只能有 1-2 个活跃 thread block——occupancy 极低，性能惨不忍睹。

V2 的 partitioned 设计：将序列切成 512-token 的 partition，每个 partition 独立做 attention。共享内存需求固定为 O(PARTITION_SIZE)，与序列长度无关。**代价是额外的 reduce kernel**——需要把多个 partition 的 partial softmax 结果合并。但这是可以接受的：reduce kernel 只做 element-wise rescaling，计算量远小于 attention 本身。

### V1 vs V2 对比

| | V1 | V2 |
|---|---|---|
| Grid | `(heads, seqs, 1)` | `(heads, seqs, num_partitions)` |
| 共享内存 | O(max_seq_len) | O(512) |
| 序列长度上限 | ~8K (受 SMEM 限制) | 无限制 |
| 额外 Buffer | 无 | tmp_out + exp_sums + max_logits |
| Reduce 步骤 | 不需要 | 需要 `paged_attention_v2_reduce_kernel` |
| 适用场景 | 短序列 (decode) | 长序列 (prefill, long-context) |

**vLLM 的选择策略：** 对于 decode（每次 1 token），序列虽然长但 query 只有 1 个——V1 足够。对于 prefill（整个 prompt 一次处理），序列可能很长（128K+）——自动切换到 V2 或 FlashAttention。

---

## 3.6 FlashAttention 的 vLLM 集成

### Source Trail

打开 `vllm/v1/attention/backends/flash_attn.py:L682`。`FlashAttentionImpl.forward()` 如何与 block_table 交互：

```python
# flash_attn.py:L764
block_table = attn_metadata.block_table

# flash_attn.py:L797-L819 — 将 block_table 传入 FA kernel
flash_attn_varlen_func(
    q=query[:num_actual_tokens],
    k=key_cache,                      # 完整的物理 KV cache
    v=value_cache,
    out=output[:num_actual_tokens],
    block_table=block_table,          # ← block_table 传给 FA kernel
    ...
)
```

打开 `vllm/vllm_flash_attn/flash_attn_interface.py:L176`。`flash_attn_varlen_func()` 根据 FA 版本分发：

```
FA2 → torch.ops._vllm_fa2_C.varlen_fwd(block_table=...)
FA3 → torch.ops._vllm_fa3_C.fwd(page_table=...)
FA4 → _flash_attn_fwd(page_table=block_table, ...)
```

每一版 FA 的 C++/CUDA 实现在内部都有 block_table 索引逻辑——和 `attention_kernels.cuh` 中的逻辑一致。区别是 FA2/FA3/FA4 还做了 forward/backward pass 的联合优化和 Hopper-specific（TMA, warp-specialization）的硬件利用。

### Cascade Attention（vLLM 特有的优化）

`flash_attn.py:L1131-L1222`。对于有长 shared prefix 的多序列 batch（例如相同的 system prompt），vLLM 用一种叫 cascade attention 的技术：

1. **Prefix 部分：** 所有序列共享——用 `block_table[:1]` 只算一次（只有一行的 block table）
2. **Suffix 部分：** 每个序列不同——用 `block_table[:, num_common_kv_blocks:]` 分别计算
3. **合并：** 用 `merge_attn_states()`（`csrc/attention/merge_attn_states.cu`）做 online softmax merge

这本质上是 **prefix caching 在 kernel 层的体现**——不是"检查是否有缓存然后决定算什么"，而是"直接把共享的 prefix 只算一次，在 kernel 层面合并结果"。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `paged_attention_with_block_table()` | `csrc/attention/attention_kernels.cuh:L85` | Python 版显式 block_table 循环——读者可以看到每次索引操作 |
| `fused_paged_attention_tiled()` | 同上 + `triton_decode_attention.py:L60` | Python 版 tiled attention + block_table 融合 |
| `build_block_table()` | `vllm/v1/core/kv_cache_manager.py:L225` + `block_pool.py:L322` | 简化为 first-fit 分配器 |
| `calculate_hbm_traffic()` | Attention 论文附录 + Nsight Compute 数据 | 教学版——正确的数量级，未包含 cache line 级精确建模 |
| HBM 对比表 | Paper + 源码注释 | 凸出朴素 vs FA 的根本差异：O(seq²) 中间写入的有无 |

---

## 验证

```bash
cd artifacts/03-flashattention-pagedattention
python -m pytest tests/ -v
# 9/9 passed ✅
```

---

## 总结

从 `attention_kernels.cuh:252` 的一行代码出发：

```cpp
const int64_t physical_block_number = block_table[block_idx];
```

这一行是 FlashAttention 和 PagedAttention 融合的物理接口——paged attention kernel 在 tiled online softmax 的每一次内部循环迭代中，通过这一行找到"下一个 KV block 在哪"，然后立刻加载、立刻计算、立刻累积到 running softmax 状态中。

关键收获：

- **FlashAttention 优化 HBM 带宽：** 避免 O(seq²) 的中间矩阵写入。代价是反复读取 K 和 V——但 O(seq × d) × (seq/BLOCK) < O(seq²) 对于长序列。
- **PagedAttention 优化显存利用：** block_table 让非连续分配成为可能，消除外部碎片。block_size=16 是碎片化和 block_table 大小的最佳平衡点。
- **融合是 vLLM 的核心优势：** 在 CUDA kernel 内部做 block_table 查找，避免了单独的 gather 步骤——在 SRAM 中直接从非连续物理位置加载 K、V。
- **V2 的 partitioned softmax 突破了序列长度限制：** 共享内存需求从 O(seq_len) 降到 O(512)，代价是一个 reduce kernel。
- **FA2/FA3/FA4 都原生支持 block_table：** 在 `flash_attn_interface.py` 中，`block_table` 被直接传给 FA 自定义 CUDA kernel——融合不是 vLLM 的 hack，而是 FA 从设计上就支持的模式。

---

**下一章：** 第4章 — Continuous Batching：动态调度系统

Attention 算得快了（FlashAttention），KV Cache 存得高效了（PagedAttention）。现在的问题是：如何在一批请求到达和完成速率不同的情况下，最大化 GPU 利用率？答案在 `vllm/v1/core/sched/scheduler.py:67`——`Scheduler.schedule()` 的动态 batch 组装逻辑。

---

← 第2章 | 第4章 →
