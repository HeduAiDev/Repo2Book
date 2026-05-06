# 第14章：Triton 算子编写基础

> 打开 `vllm/v1/worker/utils.py:41`。`_zero_kv_blocks_kernel` 是 vLLM 中最简单的 Triton kernel。
> 18 行代码——加载一个 block ID，计算 GPU 地址，写一堆零。但它包含了 Triton 编程的全部核心概念。
> 从这 18 行开始，到本章末尾，你会写出一个完整的 tiled matrix multiplication kernel。

---

## 这章要做什么？

Part 3 的目标：用 Triton 手写每一个算子，逐层构建 Llama-3.2-1B 的完整推理。但在写 RMSNorm、RoPE、Attention、MLP 之前，需要先理解 Triton 的编程模型。

这章用三个递进的例子建立这"个模型：Vector Add（grid 概念）→ Tiled MatMul（tile 循环）→ Block Zero（vLLM 的真实 kernel）。

学完这章你能：
- 解释 Triton 和 CUDA 的核心区别：**写单 block 的程序，Triton 在 grid 上自动并行**
- 手写 tiled matrix multiplication——理解为什么 `B_ptr += BLOCK_K * N`（不是 `BLOCK_K`）
- 理解 `tl.constexpr` 为什么是 Triton 编译时多路径选择的关键——vLLM 的 attention kernel 用它在**同一个 kernel 源码**中支持 8 种硬件配置

---

## 14.1 Triton vs CUDA：写一次，自动并行

### Theory: 两种编程模型的对比

CUDA 编程需要处理三个层级：
- **Thread** — 单个执行单元
- **Warp** — 32 threads 的 SIMD 组
- **Block** — 多个 warps，共享 shared memory

Triton 只需要处理一个层级：**Block。** 你写一个处理一个 tile 的程序，Triton 在 grid 上自动并行化：

```
CUDA 你需要写的:
    threadIdx.x, blockIdx.x, blockDim.x
    __syncthreads()
    shared memory management
    Warp-level reductions

Triton 你需要写的:
    pid = tl.program_id(axis)  ← "我是哪个 block？"
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.load(ptr + offsets, mask=...)
    tl.store(ptr + offsets, values, mask=...)
```

这不是"少写几行"——是**抽象层级的提升**。Triton 编译器负责把 block 级程序编译为高效的 CUDA 代码（register allocation、shared memory banking、warp scheduling）。vLLM 广泛使用 Triton 来实现 attention backend——`triton_unified_attention.py` 的 749 行用 Triton 实现了 CUDA 需要数千行才能完成的功能。

`★ Insight ─────────────────────────────────────`
Triton 的一个关键权衡：你失去了对 shared memory 和 warp 的精确控制，但编译器通常比你更擅长分配 register 和 schedule warps。vLLM 的 attention kernel 选择 Triton 而不是手写 CUDA，不是因为 Triton 更快——CUDA FlashAttention 仍然更快——而是因为 Triton 让**可维护性**和**硬件可移植性**成为可能。同一个 `kernel_unified_attention` 源码，在 A100 和 H100 上编译为不同的 PTX，支持 FP8 和 bf16 的 constexpr 分支，不需要 `#ifdef`。
`─────────────────────────────────────────────────`

---

## 14.2 核心概念

### Program ID 和 Grid

```python
@triton.jit
def my_kernel(ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                          # "我是第几个 block？"
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 我负责哪些元素？
    mask = offsets < N                              # 边界保护
    data = tl.load(ptr + offsets, mask=mask)        # 从 HBM 读到 SRAM
    result = data * 2
    tl.store(ptr + offsets, result, mask=mask)       # 从 SRAM 写回 HBM

# Launch: grid = (5,) → 5 个 block 实例并行运行
grid = (5,)
my_kernel[grid](tensor, N=5000, BLOCK_SIZE=1024)
```

关键理解：**你写的代码在每个 block 实例上运行一次。** `grid` 定义了有多少个 block。`tl.program_id` 告诉每个实例"你是第几个"。

Mask 是 Triton 的边界保护机制——当 N 不是 BLOCK_SIZE 的整数倍时，最后一个 block 的 `offsets` 会超出数组边界。`mask=mask` 告诉 `tl.load` 和 `tl.store` 忽略这些越界访问。

### Constexpr：编译时多路径

```python
# vLLM pattern (triton_unified_attention.py:L94-L143)
@triton.jit
def kernel(..., IS_3D: tl.constexpr, USE_ALIBI_SLOPES: tl.constexpr, ...):
    if IS_3D:
        # 3D softmax path → compiled into kernel variant A
        ...
    else:
        # 2D path → compiled into kernel variant B
        ...
```

`tl.constexpr` 不是运行时 if——它在**编译时**求值。Triton 编译器对不同 constexpr 组合生成不同的 PTX——死代码被完全消除。vLLM 用这来实现"一个源码，8 种硬件路径"。

---

## 14.3 Tiled Matrix Multiplication

这是 Triton 的 "Hello World"——从第一个 kernel 开始，逐步理解 tiling。

### Why Tiling?

朴素 GEMM：每个 thread 计算一个输出元素 → 需要读整行 A + 整列 B → 每个 thread 读 `M+K+N` 个元素 → 带宽受限。

Tiled GEMM：每个 block 计算一个 `[BLOCK_M × BLOCK_N]` 的 tile → 只要读 `BLOCK_M×BLOCK_K + BLOCK_K×BLOCK_N` 个元素 → 这些元素在 SRAM 中复用 `min(BLOCK_M, BLOCK_N)/BLOCK_K` 次 → 计算受限。

### 指针算术的陷阱

```python
# A: [M, K] — advancing in K moves 1 element right (1 column)
a_ptrs += BLOCK_K           # +BLOCK_K columns = +BLOCK_K elements ✓

# B: [K, N] — advancing in K moves 1 ROW down (N elements!)
b_ptrs += BLOCK_K * N       # +BLOCK_K rows × N elements per row ✓
```

**这是 Triton 新手最常犯的错误。** 在 PyTorch 中你写 `a @ b`，PyTorch 处理 strides。在 Triton 中你手动管理指针——B 的 stride 是 `N`（列数），因为每个 row 有 `N` 个元素。忘记乘 `N` → 读取错误的 K tile → 输出偏差几十到几百。

### Tile 大小的约束

```python
# 典型配置 (from vLLM triton_unified_attention.py)
BLOCK_M = 16   # Query tile (rows)
BLOCK_N = 32   # KV tile (columns)  
BLOCK_K = 32   # Inner dimension tile
HEAD_DIM = 128

# SRAM 用量
A_tile = BLOCK_M × BLOCK_K × 2B  = 1 KB   (bf16)
B_tile = BLOCK_K × BLOCK_N × 2B  = 2 KB   (bf16)
C_acc  = BLOCK_M × BLOCK_N × 4B  = 2 KB   (fp32 accumulator)
# Total: ~5 KB — fits easily in 228 KB L1

# vLLM 的 attention tile 更复杂 (Q+K+V+S+P+O_acc):
# ~112 KB — 也放得下，但不能再大了
```

---

## 14.4 Block Zero — vLLM 的真实 Kernel

### Source Trail

打开 `vllm/v1/worker/utils.py:41`。这是 vLLM 代码库中最简单的 kernel：

```python
@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr, block_ids_ptr, n_blocks,
    N_SEGS: tl.constexpr, PAGE_SIZE_EL: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    chunks = PAGE_SIZE_EL // BLOCK_SIZE
    work_per_block = N_SEGS * chunks
    block_index = pid // work_per_block    # 哪个 block 要清零？
    seg_index = (pid % work_per_block) // chunks  # 哪个 segment？
    chunk_index = (pid % work_per_block) % chunks # block 内哪个 chunk？

    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    addr = seg_addr + block_id * (PAGE_SIZE_EL * element_size)
           + chunk_index * BLOCK_SIZE * element_size

    tl.store(addr + tl.arange(0, BLOCK_SIZE), tl.zeros([BLOCK_SIZE], dtype=tl.int8))
```

**为什么需要这个？** 每次一个 KV cache block 被重新分配，它的物理 GPU 内存包含上一个请求的数据——必须清零。`KVBlockZeroer` 用这个 kernel 在单次 launch 中清零多个 block 的多个 segment。

**Grid 的灵活性：** 注意 `pid` 被分解为三个维度——block_index、seg_index、chunk_index——全部在 1D grid 中。这避免了启动 3D grid 的 kernel launch 开销——Triton 的 1D grid 通常更快。

---

## 14.5 vLLM 为什么不用 `@triton.autotune`？

### Source Trail

打开 `vllm/v1/attention/ops/prefix_prefill.py:23-25`（已注释的 autotune）：

```python
# # FIXME: triton 3.2 first call to autotune causes long pause
# @triton.autotune(configs=[...], key=[...])
```

**vLLM 的 attention kernel 全部手选 tile 大小。** 不是不会用 autotune——Mamba 和 FLA kernel 广泛使用了它。但在 attention 路径上，**首次调用的延迟**不可接受——用户的第一个请求要等 Triton 搜索配置空间（几秒到几十秒）。

替代方案：**启发式选择。** 从 `triton_unified_attention.py:578-609`：

```python
BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(...)
# Prefill: BLOCK=32. Decode: BLOCK=16 (or 32 for FP8).
```

这种硬编码的启发式在大多数情况下足够好——和 autotune 找到的最优配置差距在 5% 以内。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `_vector_add_kernel` | — | 教学用最简单 kernel |
| `_tiled_matmul_kernel` | `triton_unified_attention.py:L58` 的 tile 循环模式 | 同样的 tile 循环结构（load→dot→accumulate→store） |
| `_block_zero_kernel` | `worker/utils.py:L41` `_zero_kv_blocks_kernel` | 同样的 pid 分解 + 地址计算模式 |
| Triton vs Torch benchmark | — | 原创——展示 tiling 什么时候比 torch 快 |

---

## 验证

```bash
cd artifacts/14-triton-primer && python -m pytest tests/ -v
# 6/6 passed ✅ (including GPU Triton tests)
```

---

## 总结

- **Triton = 写 block 级程序，编译器自动并行化。** 不需要管理 thread、warp、shared memory。
- **Tiled matmul 是理解后续所有 attention kernel 的基础。** 指针算术中 B 的 stride 是 N——忘了乘 N 是最常见的 bug。
- **`tl.constexpr` 是 Triton 的编译时多路径。** vLLM 用它在同一 kernel 中支持 8 种硬件配置。
- **vLLM 不在 attention path 上用 autotune。** 首次调用延迟不可接受——启发式手选 tile 大小。

---

**下一章：** 第15章 — Llama-3.2-1B 模型架构全景

有了 Triton 基础，下一章将拆解 Llama-3.2-1B 的完整架构——从 `LlamaConfig` 的每个参数到 32 层 decoder layer 的内部结构。然后从第 16 章开始逐算子手写：RMSNorm → RoPE → Attention → MLP。

---

← 第13章 | 第15章 →
