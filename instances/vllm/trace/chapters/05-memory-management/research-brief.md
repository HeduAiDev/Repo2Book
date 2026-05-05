# Research Brief: GPU 显存管理系统 — 从碎片回收到层级分配

**Researcher**: researcher
**Date**: 2026-05-05
**For**: Writer — Ch05 叙事参考
**本轮调研重点**：GPU 显存管理演进史、vLLM 分配策略（V0→V1 架构跃迁）、竞品对比、关键设计决策

---

## 演进时间线

| 时间 | 事件 | 论文/PR | 关键贡献 |
|------|------|---------|---------|
| 2014 | cuDNN 首版发布 | NVIDIA | GPU 内存池雏形——workspace 预分配避免运行时 `cudaMalloc` |
| 2017.03 | PyTorch CUDACachingAllocator | PyTorch 0.1 | 将 `cudaMalloc` 包装为缓存分配器，round-up + split + coalesce 策略 |
| 2019 | HuggingFace Transformers | HF | 模型加载即占满显存——权重=显存上限，KV Cache 只能"挤" |
| 2022.05 | FlashAttention | Dao et al. | Attention 显存从 O(N²) 降至 O(N)，激活值不再是瓶颈 |
| 2022.08 | vLLM PagedAttention 论文 | Kwon et al. (arxiv 2309.06180) | 虚拟内存分页思想引入 KV Cache：block table 映射，<4% 浪费 |
| 2023.06 | vLLM 开源 | vLLM 0.1 | CacheEngine + BlockSpaceManager 架构，`gpu_memory_utilization=0.9` |
| 2023.09 | Prefix Caching | PR #d10f8e1d | Block hash → 复用，命中率依赖 system prompt 重复 |
| 2023.10 | Swap/Recompute Preemption | PR #2af23ab0 | 抢占策略：swap 到 CPU vs recompute，CPU offload 雏形 |
| 2024.03 | KV Cache offload framework | vLLM 0.4 | 结构化 CPU offload：pin memory + async copy |
| 2025.01 | CuMemAllocator (Sleep Mode) | PR #22724 | PyTorch pluggable allocator → sleep/wake 释放 GPU 内存 |
| 2025.06 | **V1 架构：KVCacheManager 重构** | PR #31916 系列 | 单一 BlockPool + Coordinator + 多类型 KV Cache Group |
| 2025.09 | Hybrid KV Cache Manager | PR #29143 | 不同 block_size 的注意力层共存（full attn + sliding window） |
| 2025.10 | CUDAGraph Memory Profiling | PR #38284 | 启动时精确估算 CUDA Graph 占用，避免 OOM |
| 2025.12 | KV Cache per-token-head Quant | PR #38378 | INT8/FP8 per-token-head 动态量化 → 2-4x KV Cache 压缩 |
| 2026.01 | TurboQuant 2-bit KV Cache | PR #38479 | 2-bit KV Cache，4x 容量 |
| 2026.04 | DeepSeek V4 多层 KV Cache | PR #40860 | MLA + SWA + Full Attn 共享内存池的复杂分配 |
| 2026.04 | Cumem expandable_segments 冲突 | PR #40812 | PyTorch expandable_segments 与 cumem memory pool 冲突，vLLM 自动禁用 |
| 2026.05 | HMA KV Offload 完成 | PR #41445 | 异构内存架构（Grace-Hopper）KV offload 的 HMA 支持，统一虚拟地址访问 |

---

## 最近关键事件深度解读

### CumemAllocator × PyTorch expandable_segments 战争 (2026.04)

```
问题: PyTorch 2.x 默认启用 expandable_segments
  → PyTorch allocator 在 OOM 前会尝试扩展已分配的 segment
  → cumem 通过 PyTorch Pluggable Allocator 管理内存池
  → expandable_segments 扩展行为与 cumem 的内存池边界冲突
  → 表现: 间歇性 OOM，内存碎片不可预测增长

vLLM 修复 (PR #40812):
  → 在 cumem memory pool 启用时自动禁用 expandable_segments
  → 不影响非 cumem 场景（正常的 CUDACachingAllocator 仍可用 expandable_segments）
  → 关键认知: Pluggable Allocator 与 expandable_segments 是互斥的
```

### HMA (Heterogeneous Memory Architecture) KV Offload (2026.05)

```
背景: Grace-Hopper (GH200) 和类似的统一内存架构
  → CPU 和 GPU 共享物理地址空间
  → cudaMallocManaged / cuMemCreate with CU_MEM_ACCESS_FLAGS_PROT_READWRITE
  → 传统 offload: D2H memcpy → PCIe 传输（显式拷贝）
  → HMA offload: GPU 直接通过 NVLink-C2C 访问 CPU 内存（零拷贝语义）

PR #41445 (13/N 系列终章):
  → Unified memory layout for offloading workers
  → KV cache tensor 使用 cuMem Map/Unmap 映射到统一地址空间
  → 去除了显式的 D2H/H2D memcpy
  → 延迟: 从 PCIe 32GB/s → NVLink-C2C 450GB/s（GH200），约 14x 加速
  → 局限: 仅支持 HMA 硬件平台（GH200、Grace-Blackwell 等）
```

---

## vLLM 源码考古（Git Log 深度追踪）

### V0 时代 (2023-2025): 一次性分配的简单时代

```
vllm/core/block_manager.py  (已删除)
vllm/core/block_space_manager.py
vllm/worker/cache_engine.py  ← 核心：concat KV cache tensor → 直接分配

架构特征:
  CacheEngine.allocate_gpu_cache() → 一次性 torch.empty() 分配整块 KV Cache
  BlockSpaceManager → 管理 block 分配/释放
  NaiveBlockAllocator → 简单链表 free list
  PrefixCachingBlockAllocator → hash → block 复用（但 eviction 简单）
```

**V0 的问题**：
1. **单一大 tensor**：一次性分配 `[num_blocks, block_size, num_kv_heads, head_size]` tensor，不支持动态扩容
2. **碎片回收粗放**：只有 block 用完才 free，没有 watermark 回收
3. **混合模型不支持**：所有层共享同一个 block_size，无法处理 SWA + Full Attention 混合
4. **Prefix Cache 孤岛**：每个请求独立 hash 查找，无全局 cache pool 概念

### V1 时代 (2025.06-现在): 层级化+多类型+回收感知

```
vllm/v1/core/
  kv_cache_manager.py     ← KVCacheManager：对外接口
  kv_cache_coordinator.py ← Coordinator：编排多种类型 KV Cache Group
  kv_cache_utils.py       ← KVCacheBlock, BlockHash, FreeKVCacheBlockQueue, get_kv_cache_configs
  block_pool.py           ← BlockPool：全局 block 池 + prefix cache 管理
  single_type_kv_cache_manager.py ← 单类型 KV Cache 的实际分配逻辑
  kv_cache_metrics.py     ← 可观测性：residency metrics

vllm/v1/kv_cache_interface.py  ← 接口定义：KVCacheSpec 家族（9 种子类！）
vllm/v1/worker/gpu_worker.py   ← determine_available_memory → memory_profiling → initialize_from_config
```

**V1 的核心改进**：

1. **BlockPool 全局池化**（block_pool.py）
   - 单一 shared pool 管理所有 KV Cache block
   - FreeKVCacheBlockQueue：高效 O(1) 弹出/推入
   - 多个 KV Cache Group（full attn / SWA / MLA / Mamba）共享同一物理池

2. **Coordinator 分层编排**（kv_cache_coordinator.py）
   - `find_longest_cache_hit()`: prefix cache 查找
   - `allocate_new_blocks()`: 新 block 分配
   - `remove_skipped_blocks()`: 滑动窗口超出的 block 立即回收
   - `get_num_blocks_to_allocate()`: 预计算所需 block 数（含 admission cap）

3. **Memory Profiling 三层分类**（mem_utils.py + gpu_worker.py）
   ```
   GPU 显存 = non_torch_memory (NCCL, cuBLAS workspace, 其他进程)
            + torch_memory (PyTorch 管理)
                ├── weights_memory (模型权重，含量化参数)
                ├── peak_activation_memory (前向峰值激活值)
                ├── cudagraph_memory (CUDA Graph 重放开销)
                └── available_kv_cache_memory = requested_memory - non_kv_cache_memory
   ```

4. **多类型支持**（kv_cache_interface.py）
   - `AttentionSpec` → `FullAttentionSpec` / `SlidingWindowSpec` / `MLAAttentionSpec` / `ChunkedLocalAttentionSpec` / `CrossAttentionSpec`
   - `UniformTypeKVCacheSpecs` → 多 layer 同类型合并
   - 每种类型独立 block_table，但共享底层 BlockPool
   - `max_admission_blocks_per_request()` 每种类型定义自己的 admission cap

5. **Sleep Mode (CuMemAllocator)**（device_allocator/cumem.py）
   - PyTorch Pluggable Allocator → 应用层控制 GPU 内存分配/释放
   - `sleep(offload_tags)` → D2H memcpy 备份 → 释放 GPU 内存
   - `wake_up(tags)` → 重新 `cudaMalloc` → H2D memcpy 恢复
   - 用途：vLLM 实例间 GPU 内存切换（多租户）、滚动更新

---

## vLLM 显存全景图

### 物理布局（以 Llama-3.2-1B 在 T4 16GB 为例）

```
┌─────────────────────────────────────────────────────────────────┐
│ GPU 显存 (15.0 GiB 可用)                                        │
├─────────────────────────────────────────────────────────────────┤
│ 非 vLLM 占用 (~0.5 GiB)                                         │
│   - 其他进程 / Xorg / EGL                                       │
│   - NCCL buffers (非 PyTorch 管理)                               │
├─────────────────────────────────────────────────────────────────┤
│ vLLM 占用 — non_torch (~1.0 GiB)                                │
│   - cuBLAS workspace (cuBLAS handle 预分配)                     │
│   - cuDNN convolution workspace                                 │
│   - Custom C++ extension buffers                                │
│   - NCCL communication buffers                                  │
├─────────────────────────────────────────────────────────────────┤
│ vLLM 占用 — torch (CUDACachingAllocator 管理)                    │
│   ├── Model Weights (~2.5 GiB)                                  │
│   │   - Transformer layers: QKV projection + MLP + RMSNorm      │
│   │   - dtype=float16 (小模型) 或 bfloat16 (大模型)             │
│   │   - 量化时可降至 int8/int4/fp8                              │
│   ├── Peak Activations (~3.0 GiB)                               │
│   │   - 前向峰值: attention output + MLP intermediate           │
│   │   - max_num_batched_tokens 决定峰值（一次性调度多少 token） │
│   │   - Chunked Prefill 通过 tiling 降低此值                     │
│   ├── CUDA Graph Replay (~1.0 GiB, 可选)                        │
│   │   - Captured graph 的输入/输出 buffer                       │
│   │   - 每个 batch_size 一组 graph（或 padded 复用）            │
│   └── KV Cache (剩余 ~7.0 GiB ← gpu_memory_utilization=0.92)    │
│       ├── Layer 0: K cache + V cache (2 × block_size × H × d × dtype)│
│       ├── Layer 1: K cache + V cache                             │
│       ├── ... 32 layers × 2 tensors                              │
│       └── 分 block: block_size=16 tokens, 总 block 数 ≈ 6000    │
└─────────────────────────────────────────────────────────────────┘
```

### 关键公式

**单个 block 的字节数**（FullAttentionSpec）：
```
page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size
```

**KV Cache 总 block 数**：
```
num_blocks = floor(available_kv_cache_memory / page_size_bytes / num_layers)
available_kv_cache_memory = total_gpu_memory × gpu_memory_utilization - non_kv_cache_memory
```

**单请求最大 block 数**：
```
max_blocks_per_request = ceil(max_model_len / block_size)
```

**最大并发请求数**（理论）：
```
max_concurrency = num_blocks / max_blocks_per_request
```

---

## 关键设计决策

### 决策 1: 为什么 V1 从"单一大 tensor"改为"BlockPool + Coordinator + Group"三层架构？

```
V0 架构:
  CacheEngine.allocate_gpu_cache() → torch.empty([num_blocks, ...]) 
  → 一次性分配，所有层共享 block 数
  → 问题: 混合模型（SWA + Full Attn）block_size 不同时无法处理
  
V1 架构:
  KVCacheManager (对外接口: allocate_slots / free / get_computed_blocks)
    └── Coordinator (编排: find_longest_cache_hit / allocate_new_blocks / remove_skipped_blocks)
          └── SingleTypeKVCacheManager[×N] (每类型独立管理)
                └── BlockPool (共享物理池 + prefix cache hash map)
```

**驱动因素**：
1. **混合模型**（SWA + Full Attention + MLA）成为常态：不同层需要不同 block_size
2. **Prefix Cache** 需要全局视角（跨请求共享），不是单个请求的属性
3. **回收感知**：滑动窗口超出 → 立即释放 block → 需要细粒度控制
4. **Admission Gate**：`max_admission_blocks_per_request()` 每种类型不同

### 决策 2: `gpu_memory_utilization=0.92` 为什么不是 1.0？

```
问题: 为什么不100%利用显存？
├── 方案 A: utilization=1.0 → malloc 所有可用的 free 显存
│   └── 风险: CUDA context 本身需要 runtime memory（cuBLAS handle、streams）
│       PyTorch CUDACachingAllocator 的 metadata、CUDA Graph 重开
│       → 1.0 会导致 cudaMalloc 失败 / OOM
├── vLLM 选择: utilization=0.92 (默认)
│   └── 8% buffer 留给:
│       1. 非 PyTorch 管理的显存（cuBLAS workspace 增长）
│       2. CUDA Graph capture 峰值（capture 时的显存 > replay 时）
│       3. NCCL 通信临时 buffer
│       4. PyTorch allocator 碎片导致的"预留但不可用"显存
└── 用户可调: kv_cache_memory_bytes 精确覆盖此值（用于手动控制）
```

### 决策 3: 为什么需要 Memory Profiling 而不是静态计算 KV Cache 大小？

```
静态计算的失败:
  weights_size = sum(p.numel() × p.element_size())  ← 这是准确的
  peak_activations = ???                              ← 无法静态计算！
    - 不同 batch_size → 不同峰值
    - 不同模型架构 → 不同中间张量大小
    - CUDA Graph → 额外的 buffer 占用
    - custom kernel → cuBLAS workspace 动态分配
  non_torch_memory = ???                              ← NCCL, cuBLAS 等，完全无法静态计算

vLLM 的方案:
  1. 加载模型权重（不计 KV Cache）
  2. 用 max_num_batched_tokens 做一次 dummy forward
  3. 记录 torch peak memory + non_torch increase
  4. available_kv_cache = requested - non_kv_cache_memory
  → 这是唯一准确的方法
```

### 决策 4: 为什么 block_size 默认是 16？

```
block_size 选择:
├── 太小 (e.g. 4): 
│   ├── 优点: 碎片少，接近精确分配
│   └── 缺点: block table 条目多，kernel launch 次数多，metadata 开销大
├── 太大 (e.g. 128):
│   ├── 优点: block table 小，连续读写效率高
│   └── 缺点: 内部碎片大（请求只要 10 token 也分配 128 token block）
└── vLLM 选择 16:
    └── 经验值: 绝大多数请求 seq_len 在 256-2048 区间
        → 16-128 个 block，block table 大小合理（~1KB）
        → 碎片率 = 平均 (block_size-1)/2 / block_size ≈ 47/16 ≈ 3% 浪费
        → token 级 kernel 配合 block 级分配，IO 效率高
```

### 决策 5: Swap vs Recompute — 抢占策略的权衡

```
当需要腾出显存给更高优先级请求时:
├── Swap (CPU offload):
│   ├── 优点: 恢复快（D2H + 后续 H2D）
│   │   延迟 = 2 × KV_size / PCIe_BW
│   │   e.g. 1GB KV → PCIe 32GB/s → 31ms swap out + 31ms restore
│   ├── 缺点: 占用 CPU 内存（需预留 cpu_swap_space）
│   └── 适合: KV cache 小、PCIe 带宽充足
├── Recompute:
│   ├── 优点: 不占用 CPU 内存，不依赖 PCIe
│   │   延迟 = 1 次完整 prefill forward pass
│   │   e.g. 2048 token prefill → ~50ms (A100)
│   ├── 缺点: 计算开销大，尤其是长 prefill
│   └── 适合: prefill 短、GPU 算力充足
└── vLLM 选择: Swap 为主（默认），Recompute 可选
    └── 原因: 现代 GPU 显存大但 PCIe 足够快（NVLink 更快）
        + CPU 内存通常充足（256GB+，远大于 GPU 显存的 KV Cache）
```

---

## 竞品对比

### TensorRT-LLM 显存管理

| 维度 | vLLM | TensorRT-LLM |
|------|------|-------------|
| **显存分配** | PyTorch CUDACachingAllocator + 手动 block pool | TRT Runtime 统一管理（编译时确定） |
| **KV Cache 分页** | PagedAttention：16-token block，block table | 类似 paged KV cache（GptManager + KvCacheConfig） |
| **显存预算** | 启动时 profiling 确定 non-KV memory | 编译时 graph optimization 已知峰值 |
| **量化 KV Cache** | INT8/FP8/FP8-per-token-head/NVFP4/TurboQuant(2-bit) | FP8/INT8 KV Cache（量化为 TRT plugin） |
| **Prefix Cache** | Hash-based，block pool 内全局共享 | 由 TRT GptManager 的 `maxAttentionWindow` 管理 |
| **CPU Offload** | CumemAllocator sleep/wake + native offloading | Managed via Unified Virtual Memory (UVM) |
| **混合模型** | Full SWA + SWA + MLA → 多 KV Cache Group | 编译时 fuse，无运行时不同类型切换 |
| **CUDA Graph** | 启动后 profile graph 额外显存 | 编译时已知 graph 显存占用 |
| **灵活性** | 高：运行时可调 block_size, utilization, quantization | 低：编译后固定，需重新 build engine |
| **性能** | 接近 peak（hand-tuned allocator + kernel） | 接近 peak（TRT 编译优化 + memory planning） |

**关键差异**: TRT-LLM 用编译时 memory planning（类似 TVM/MLIR 的静态内存规划），vLLM 用运行时 profiling + 动态 block 分配。前者显存效率更高（无运行时开销），后者灵活（支持任意模型和配置组合）。

### SGLang 显存管理

| 维度 | vLLM | SGLang |
|------|------|-------------|
| **KV Cache 分页** | PagedAttention（虚拟内存类比） | RadixAttention（前缀树共享） |
| **内存池** | BlockPool 全局池化 | RadixCache 树结构池化 |
| **Prefix Cache** | Hash-based block matching（需 compute hash） | Radix Tree 自然匹配（结构即索引） |
| **回收策略** | Reference counting + watermark 回收 | Radix Tree eviction（LRU leaf first） |
| **显存预算** | gpu_memory_utilization + profiling | 类似机制（mem_fraction_static） |
| **CPU Offload** | CumemAllocator + native | 不支持（当前版本） |
| **Swap** | GPU↔CPU swap via torch tensor | 类似（via cpu offload 实验性支持） |

**关键差异**: SGLang 的 RadixAttention 天然支持前缀共享——树节点自动对应公共前缀 token，不需要额外 hash。对于 system prompt 场景，SGLang 更自然高效。vLLM 的 hash-based 更通用（不依赖 token 序列结构），但对 tree-structured 共享场景效率略低。

### LMDeploy 显存管理

| 维度 | vLLM | LMDeploy |
|------|------|-------------|
| **KV Cache 管理** | PagedAttention（Python 层管理 block） | TurboMind C++ Runtime（更底层） |
| **量化 KV Cache** | 多种量化方案 | W4A16 KV Cache（int4） |
| **显存效率** | ~4% 浪费（block_size=16 碎片） | ~2% 浪费（更小的管理粒度） |
| **持久化 KV Cache** | 不原生支持 | 支持 KV Cache 磁盘持久化 |
| **CPU Offload** | sleep mode + native offloading | 结构化 offloading pipeline |
| **多实例** | CumemAllocator sleep/wake | 类似的内存池化管理 |

**关键差异**: LMDeploy 的 TurboMind 用 C++ 管理 KV Cache，效率更高但灵活性降低。vLLM 的 Python 层管理更易扩展但有一定 GC 开销。

### vLLM 相比竞品的独特取舍

1. **运行时灵活性 > 极致效率**: Python block pool 管理比 C++/编译时方案慢，但支持热更新配置、多模型、自定义分配策略
2. **层级抽象 > 单一实现**: V1 的 Coordinator + SingleTypeKVCacheManager 架构为未来扩展（如不同硬件定制分配策略）留出空间
3. **Memory Profiling 作为准确性的基石**: 不做静态估算，每次启动都真跑一次 dummy forward
4. **Block Pool 是通用抽象**: 同一个 pool 同时服务 Full Attention / SWA / MLA / Mamba，而 TensorRT-LLM 和 LMDeploy 倾向于每种类型独立管理

---

## 本章范围界定

### 应该深度覆盖的内容
1. GPU 显存布局全景图：权重/KV Cache/激活值/workspace 各自占比与依赖关系
2. CUDACachingAllocator 原理：round-up、split、coalesce、`cudaMalloc` overhead
3. V1 架构：KVCacheManager → Coordinator → BlockPool 三层设计
4. Memory Profiling 的完整流程：`determine_available_memory()` 的三分类法
5. Block Pool 的分配/回收/prefix cache 命中：FreeKVCacheBlockQueue + BlockHashToBlockMap
6. Watermark-based 回收：滑动窗口超出 → `remove_skipped_blocks()` 立即释放
7. Swap/Recompute trade-off：延迟量化分析（PCIe BW vs GPU FLOPS）

### 应该点到为止的内容（留给后续章节）
- Prefix Cache 的 Radix Tree 结构（Ch07 主题）
- KV Cache Offload 的层级存储策略（Ch12 主题）
- TP 下的 KV Cache 分片（Ch08 覆盖）
- CuMemAllocator Sleep Mode 的完整流程（Ch12 覆盖）
- ModelRunner 中 CUDA Graph 的 capture 流程（Ch20 覆盖）

### 不应涉及的内容
- Attention kernel 内部实现（Ch01/03/18 覆盖）
- Scheduler 调度策略（Ch06 主题）
- PD 分离架构的显存管理（Ch22-25 覆盖）
- 量化方案的实现细节（属于量化专题）
- 跨节点的 KV Cache 分布式共享（Ch13/23 覆盖）

---

## 给 Writer 的建议

### 最重要的 intuition
GPU 显存管理不是"分配+释放"那么简单——它是**预算、仲裁、回收**的三体问题。vLLM 的核心贡献不是发明了新的内存分配算法，而是把操作系统虚拟内存的理念（page table、demand paging、watermark 回收）恰当地映射到 GPU KV Cache 的场景。读者应理解：**vLLM 的 KV Cache Manager 本质上是一个微型操作系统内存管理器**。

### 最适合零基础读者的切入角度
从**灾难现场**讲起：
1. 开启 Llama-3.2-1B + max_model_len=131072 → 单请求 KV Cache = 2 × 131072 × 32 × 8 × 64 × 2B = **8.6 GiB**（仅 KV Cache！）
2. T4 只有 16 GiB → 一个请求就炸了
3. 问题自然展开：如何压缩？如何分页？如何回收？如何共享？

### 容易混淆的概念
1. **`page_size` vs `block_size`**: 在 vLLM 代码中，`block_size` 是一个 block 包含的 token 数；`page_size_bytes` = 该 block 实际占用的字节数。两者是正交概念。
2. **`num_gpu_blocks` vs `num_cpu_blocks`**: 前者是 GPU 上的 KV Cache block 数，后者是 CPU swap space 的 block 数。CPU blocks 只在 swap 抢占时使用。
3. **`gpu_memory_utilization` vs `kv_cache_memory_bytes`**: 前者是百分比（"给我 92% 的可用显存"），后者是绝对值（"给我 10 GiB"）。后者覆盖前者。
4. **`torch_memory` vs `non_torch_memory`**: torch_memory 是 PyTorch 通过 CUDACachingAllocator 分配的内存（model + activations + KV cache），non_torch_memory 是绕过 PyTorch 分配的内存（NCCL、cuBLAS、custom C++ 扩展）。
5. **V1 `KVCacheManager` vs V0 `BlockSpaceManager`**: V1 是全局共享 BlockPool + 多类型 Group，V0 是单一大 tensor + 简单 free list。架构完全不同。

### 建议的类比/生活例子
- **BlockPool = 酒店的房卡管理系统**：每个 block 是一个房间，block_id 是房号，ref_cnt 是入住人数，free_block_queue 是空房列表，prefix cache 是"总统套房常备"。
- **Memory Profiling = 搬新家前的空间规划**：先把大家具（模型权重）搬进去，再估算日常活动空间（激活值峰值），剩下的（可用显存）才是储物空间（KV Cache）。
- **Watermark 回收 = 滑动窗口的垃圾车**：每推进一步，超出窗口的 token 对应的 KV block 就被回收，像垃圾车定时收走不再需要的垃圾。
