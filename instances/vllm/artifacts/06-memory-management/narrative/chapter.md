# 第5章：GPU 显存管理系统

> 打开 `vllm/v1/worker/gpu_worker.py:352`。`determine_available_memory()` 是 vLLM 启动时最关键的
> 函数之一——它在真正运行任何推理之前，精确计算出 GPU 上还剩多少显存可以给 KV Cache。
> 算错一个参数，要么 OOM 崩溃，要么浪费几 GB 显存。

---

## 这章要做什么？

前四章讲了 vLLM 做什么（Attention、KV Cache、FA+PA、Scheduler），但所有这些组件共享有限的 GPU 显存。谁来决定一个 H100 的 80GB 怎么分？多少给模型权重？多少给 KV Cache blocks？多少留给 activation peaks？

答案是一套**启动时的 profiling 流程**。vLLM 不是从配置文件读取这些数字——它在启动时实际加载模型、跑一个 dummy forward pass，**测量真实的显存占用**，然后反推还剩多少给 KV Cache。

理解这套流程的最佳方式：把它看作一个**显存会计系统**。每一项显存占用都有来源、有测量方法、有责任人。本章不会罗列所有来源——而是教你 vLLM 的会计逻辑。

学完这章你能：
- 解释 `gpu_memory_utilization=0.92` 中那 8% 的 margin 去哪了
- 打开 `gpu_worker.py:352` 跟踪 `determine_available_memory()` 的完整计算链
- 理解 vLLM 为什么不实现自己的 GPU allocator——而是用 PyTorch 的 `torch.zeros()` + reshape

---

## 5.1 显存的三级划分

### Source Trail

打开 `vllm/utils/mem_utils.py:190`。`memory_profiling()` 是 vLLM 的显存会计引擎。它在启动时运行一次，产生一个精确的内存预算。

### Theory: 四个会计科目

vLLM 把 GPU 显存分成四个科目。每一笔支出都有记录：

```
┌─────────────────────────────────────────────┐
│              GPU 总显存 (80 GB)              │
├─────────────────────────────────────────────┤
│ 1. 模型权重 (~5 GB)                          │
│    → DeviceMemoryProfiler 实测               │
│    → torch.load + model.to(device) 后的增量  │
├─────────────────────────────────────────────┤
│ 2. 峰值 Activation (~2-5 GB)                │
│    → memory_profiling() dummy forward pass   │
│    → 在 max_num_batched_tokens 条件下实测    │
├─────────────────────────────────────────────┤
│ 3. CUDA Graph + NCCL (~0.5-1 GB)            │
│    → profile_cudagraph_memory() 实测         │
│    → NCCL ring buffers + graph replays       │
├─────────────────────────────────────────────┤
│ 4. KV Cache (剩余的全部)                     │
│    → available = requested - (1+2+3)        │
│    → 切成 num_blocks 个固定大小的 block      │
├─────────────────────────────────────────────┤
│ 5. 未使用 Margin (~6.4 GB)                   │
│    → (1 - gpu_memory_utilization) × total   │
│    → PyTorch 碎片、CUDA 上下文、安全边界     │
└─────────────────────────────────────────────┘
```

**为什么不是静态配置？** 因为 activation 峰值取决于 batch size、模型架构、attention 实现（FA vs naive），**不是固定值**。在 H100 上跑 Llama-3.2-1B 的 peak activation 和在 A100 上跑 Llama-3.2-70B 的 peak activation 完全不同——静态配置要么浪费，要么 OOM。

### 测量流（从 gpu_worker.py:352）

```
determine_available_memory()
  │
  ├─ 1. 如果 kv_cache_memory_bytes 已配置 → 跳过 profiling（用户知道自己在干什么）
  │
  ├─ 2. 否则，运行 memory_profiling():
  │     ├─ init_snapshot = MemorySnapshot.measure()   # 基线
  │     ├─ weights_memory = model 加载后的增量         # DeviceMemoryProfiler
  │     ├─ profile_run() → dummy forward pass           # 峰值 activation
  │     ├─ profile_cudagraph_memory()                   # CUDA graph 开销
  │     └─ 计算:
  │         torch_peak_increase = 分析后的峰值 - 基线峰值
  │         non_torch_increase = 非 PyTorch 内存增量
  │         non_kv_cache = non_torch + torch_peak + weights
  │         available_kv_cache = requested - non_kv_cache - cudagraph
  │
  └─ 3. 返回 available_kv_cache_bytes
```

`★ Insight ─────────────────────────────────────`
vLLM 的 profiling 设计体现了"measure, don't guess"的哲学。但有一个容易忽略的点：`memory_profiling()` 中 dummy forward 的 batch size 是 `max_num_batched_tokens`——这是用户配置的上限。如果实际运行时的 batch 比这个上限小，activation peak 就比 profiled 值低——换句话说，部分 KV Cache 空间被"预留过多"了。反过来，如果用户把 `max_num_batched_tokens` 设得过高，profiled 的 activation peak 会过大——挤占 KV Cache 空间。这就是为什么这个参数不能随便调高——它有内存代价。
`─────────────────────────────────────────────────`

---

## 5.2 gpu_memory_utilization 的 8% margin

### Source Trail

打开 `vllm/config/cache.py:66`：

```python
gpu_memory_utilization: float = 0.92
```

打开 `vllm/v1/worker/utils.py:403`：

```python
def request_memory(total_memory, gpu_memory_utilization):
    requested = ceil(total_memory * gpu_memory_utilization)
    if free < requested:
        raise ValueError(...)
    return requested
```

### Theory: 8% 去哪了？

默认 `0.92` 意味着 80 GB H100 只用了 73.6 GB。6.4 GB 去哪了？

| 去向 | 估计量 | 为什么 profiling 测不到 |
|------|--------|------------------------|
| PyTorch CachingAllocator 碎片 | ~1-3 GB | 碎片取决于分配/释放模式，运行时变化 |
| CUDA context + driver overhead | ~0.5-1 GB | 在第一个 torch 调用之前就分配了 |
| Profile run 和实际运行的 activation 差异 | ~0.5-1 GB | 不同的输入大小 → 不同的中间张量 |
| torch.compile 的 JIT 缓存 | ~0.5 GB | 编译后的 kernel 存在额外的显存中 |
| 其他进程（监控、MIG 分区） | 0-2 GB | 取决于部署环境 |

8% 是一个经验值——在大多数配置下足够安全，又不会过度浪费。如果你知道你的工作负载非常稳定，可以调到 0.95。如果你在 MIG 分区上跑，可能需要调到 0.85。

### 动态调整

vLLM 在 `compile_or_warm_up_model()` 中（`gpu_worker.py:628-683`）会输出 `--kv-cache-memory` 的建议值。用户在初次运行后可以用这个值**替代** `gpu_memory_utilization`，避免重复 profiling：

```
# 第一次运行（profiling 模式）
# 输出: "Suggest setting --kv-cache-memory 65000000000"

# 后续运行（跳过 profiling）
vllm serve model --kv-cache-memory 65000000000
```

---

## 5.3 为什么不自己写 GPU Allocator？

### Source Trail

打开 `vllm/v1/worker/gpu/attn_utils.py:128`：

```python
def _allocate_kv_cache(tensor_size, device):
    return torch.zeros(tensor_size, dtype=torch.int8, device=device)
```

三行代码。整个 KV Cache 就是一个巨大的 `torch.zeros(tensor)`。vLLM **不实现自己的 GPU 内存分配器**。

### Theory: 买 vs 造

PyTorch 自带一个 `CachingAllocator`，它：
- 管理 CUDA memory pool
- 处理碎片化
- 提供 `torch.cuda.empty_cache()` 手动干预
- 经过 NVIDIA + Meta 多年的生产验证

自己写一个 GPU allocator 能获得什么？可能：更快的小块分配、KV Cache 专用的 page 对齐、与 block pool 的紧密集成。但获得这些的代价是：复杂的内存管理代码、CUDA driver API（`cuMemAlloc`, `cuMemFree`）、与 PyTorch 的 allocator 共存时的冲突。

vLLM 选择不买这个代价。它用 `torch.zeros()` 一次分配整个 KV Cache tensor，然后**通过 reshape 来"划分" blocks**——不需要 allocator，因为空间是静态预分配的。Block 的分配/释放完全通过第 2 章讲的 `BlockPool` 的元数据管理来实现，不涉及实际的 `cudaFree`/`cudaMalloc`。

**唯一的例外**：`vllm/device_allocator/cumem.py` 中的 `CuMemAllocator`。但它不是给 KV Cache 用的——它是**sleep mode**（多实例 GPU 共享）用的。它用 `cuMemCreate`/`cuMemMap` 做物理内存管理，在 sleep 时卸载内存、在 wake 时恢复。这是电源管理功能，不是内存管理功能。

---

## 5.4 KV Cache 的 Tensor 布局

### Source Trail

打开 `vllm/v1/worker/gpu/attn_utils.py:145`：

```python
def _reshape_kv_cache(kv_cache_tensor, num_blocks, num_heads, head_size, block_size):
    # 从 flat int8 张量 reshape 为 backend 期望的布局
    # 注意：不同 backend 的布局不同！
```

不同 backend 期望不同的 KV Cache 形状：

| Backend | 布局 |
|---------|------|
| FlashAttention | `[2, num_blocks, block_size, num_kv_heads, head_size]` |
| PagedAttention V1/V2 | `[num_blocks, num_kv_heads, head_size/x, block_size, x]` (x = 16/dtype_size) |
| Triton decode | `(num_blocks, PAGE_SIZE, num_heads, head_dim)` |
| MLA (DeepSeek) | `[num_blocks, block_size, compressed_dim]` (不同于标准布局) |

### Theory: 为什么同一份数据要不同形状？

KV Cache 的内容是固定的——K 和 V 的数值。但不同的 kernel 访问这些数据的方式不同。

- FlashAttention 从 `block_table` 找到物理 block，然后一次性加载 `block_size` 个 token 的 K 和 V——block_size 维在靠前位置能 coalesced 访问。
- PagedAttention V1 把 `head_size/x` 和 `x` 拆开——这是为了满足 16-byte 对齐的 coalesced 内存事务。
- MLA 完全不存储 K 和 V——它存储压缩的 latent 表示——形状完全不同。

**Reshape 是零成本的操作。** `torch.Tensor.view()` 只修改 stride 元数据，不拷贝数据。同一个 40 GB 的 KV Cache tensor 可以被多个 backend 以不同形状查看——内存只有一个副本，但访问模式各不相同。

---

## 5.5 从 available memory 到 num_blocks

### Source Trail

打开 `vllm/v1/core/kv_cache_utils.py:931`：

```python
def get_num_blocks(available_memory, page_size, num_layers):
    return available_memory // (page_size * num_layers)
```

`page_size` = 一个 block 在一个 layer 中的字节数 = `2 × block_size × num_kv_heads × head_dim × dtype_bytes`。

### Theory: 整数除法导致的浪费

`num_blocks` 的计算用的是整数除法（`//`）。因为 `available_memory` 不是 `page_size * num_layers` 的整数倍，会有余数。这部分显存被浪费了——它不够分配一个新的 block，又不够给其他用途。

对于典型的 Llama-3.2-1B + H100：
- `page_size = 2 × 16 × 8 × 128 × 2 = 65536 B = 64 KB per layer`
- `num_layers = 32`
- `total per block = 64 KB × 32 = 2 MB`
- `available = ~60 GB`
- `num_blocks = 60 GB // 2 MB = 30720 blocks`
- `wasted = 60 GB - 30720 × 2 MB = ~0.5 MB` (可以忽略)

但对于小的模型 + 大 block size，浪费可能显著。这就是为什么 `block_size` 越小，浪费越少——但 block_table 越长（第 2、3 章讨论的 trade-off）。

---

## 5.6 运行时 OOM 预防

### Source Trail

vLLM **没有 catch OOM 的 try-except**。它的策略是**主动预防，被动响应**。

**主动预防（启动时）：**

1. `check_enough_kv_cache_memory()`（`kv_cache_utils.py:789`）——验证至少一个请求的完整 `max_model_len` 可以放进 KV Cache。如果不，抛出 `ValueError`。
2. `auto_fit_max_model_len()`（`kv_cache_utils.py:1816`）——当 `max_model_len=-1` 时，二分搜索找到能放入的最大上下文长度。

**被动响应（运行时）：**

1. `KVCacheManager.allocate_slots()` 返回 `None` → Scheduler preempt 请求（第 4 章）
2. `scheduler_reserve_full_isl=True` → 在 admission 时检查完整 prompt 长度，不够就拒绝

没有 try-except `torch.cuda.OutOfMemoryError`。为什么？因为当 PyTorch 抛出 OOM 时，CUDA context 可能已经处于不一致状态——catch 了也恢复不了。vLLM 选择在 OOM 发生之前阻止它。

### Theory: 为什么预先计算而不是动态适应？

动态适应（满了就驱逐、不够就拒绝）在请求长度均匀分布的假设下效果很好。但 LLM 推理的请求长度分布**极度不均匀**——一个 128K token 的长 prompt 和八个 128 token 的短 prompt 同时到达，系统的瞬时内存压力相差 1000 倍。

预先分配 + 主动预防的策略在"最坏情况"和"平均情况"之间提供了一个平衡：worst-case 是 total_blocks 个 block 同时被占用——但 profiling 保证即使这种情况发生了也不会 OOM。代价是可能浪费一些显存——但显存相对于模型权重和 KV Cache 来说通常不是瓶颈资源。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `MemorySnapshot` | `vllm/utils/mem_utils.py:L71` | 教学版 stats 建模；vLLM 版实时读取 GPU API |
| `MemoryProfiler.profile()` | `gpu_worker.py:L352` `determine_available_memory()` | 分析性计算 vs 实测 profiling |
| `KVCacheSpec.page_size_bytes()` | `kv_cache_interface.py:L138` | 同样的公式，不同的 spec 层级结构 |
| `LlamaModelMemory.weight_size()` | vLLM 实测（`DeviceMemoryProfiler`） | 分析性近似 vs 加载后测量 |
| `total_block_bytes_per_block` | `kv_cache_utils.py:L931` `get_num_blocks()` | 整数除法逻辑一致 |

---

## 验证

```bash
cd artifacts/05-memory-management && python -m pytest tests/ -q
# 10/10 passed ✅
```

---

## 总结

vLLM 的显存哲学：**measure, don't guess。** profiling 一次性测量所有非 KV Cache 的内存开销，剩余的全部给 KV Cache。

- **四级会计科目：** 模型权重 → 峰值 activation → CUDA Graph + NCCL → KV Cache。每笔支出都有测量方法。
- **8% margin 不是浪费——是安全边界。** CachingAllocator 碎片、CUDA context、JIT 缓存——这些 profiling 测不到但实际存在。
- **No custom allocator.** `torch.zeros()` + reshape 提供了免费的形状转换，不需要自己管理 `cudaMalloc`。
- **OOM 是设计选择，不是意外。** vLLM 选择在 OOM 发生之前主动阻止——不 catch `OutOfMemoryError`。

---

**下一章：** 第6章 — 请求调度系统

有了内存管理的基础，下一章深入 Scheduler 的详细决策——FCFS vs Priority vs Preemption 三种策略的比较，以及它们在多目标优化（延迟 vs 吞吐 vs 公平性）下的 Pareto frontier。

---

← 第4章 | 第6章 →
