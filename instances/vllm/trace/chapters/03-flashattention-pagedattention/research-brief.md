# Research Brief: FlashAttention & PagedAttention — vLLM 的双引擎注意力优化

**Researcher**: researcher
**Date**: 2026-05-04
**For**: Writer-2 — Ch03 v5 叙事参考
**Sections**: subsections [73-79] of book-outline.json

---

## 演进时间线

| 时间 | 事件 | 关键贡献 | 与 vLLM 的关系 |
|------|------|---------|---------------|
| 2017.06 | Transformer (Vaswani et al.) | Self-Attention 朴素实现, O(N²) HBM | Ch01 已覆盖，此章起点 |
| 2020-2021 | Linformer, Performer, Big Bird | 线性注意力（O(n) 近似 O(n²)） | vLLM 未采用——精度损失不兼容推理 |
| 2022.05 | **FlashAttention** (Dao et al., Stanford) | IO-aware tiling, online softmax, HBM traffic O(n) | vLLM 默认 CUDA backend 基石 |
| 2022.08 | PagedAttention 概念萌芽 | KV cache 碎片 → OS 虚拟内存类比 | vLLM 核心创新 |
| 2023.03 | **vLLM paper** (Kwon et al., SOSP'23) | PagedAttention + block_table 正式发表 | vLLM 命名来源，SOSP 最佳论文 |
| 2023.07 | **FlashAttention-2** (Dao) | 序列维并行化, 2x 加速, ALiBi 支持 | FA2 成为 vLLM 主力 CUDA backend |
| 2023.10 | FlashDecoding (Dao et al.) | Decode-phase 特殊优化, split-KV | 启发 triton_decode_attention.py |
| 2024.03 | **FlashAttention-3** (Shah et al.) | Hopper wgmma 异步 GEMM, FP8, 3x decode | vLLM FA3 backend (仅 SM90+) |
| 2024.06 | SGLang RadixAttention | Prefix-aware LRU 缓存, 树形存储 | vLLM prefix caching 竞争对手 |
| 2024.09 | vLLM V1 architecture | Unified attention, 3D kernel, batch invariance | triton_unified_attention.py |
| 2025-Q1 | #40631 Unify 2D/3D kernels | 单 kernel 同时支持 prefill/decode | 当前代码形态 |
| 2025-Q2 | FA4 (Blackwell SM100) + TurboQuant | head_size 512+, FP4 quant | flash_attn.py 最新 |

---

## 一、FlashAttention 演进史 —— "为什么不那样做"

### 1.1 朴素 Attention 的根本问题：FLOPS 不是瓶颈

```
朴素 attention:
  S = Q @ K^T        → 写入 HBM       ← write N² elements
  P = softmax(S)     → 读 S, 写 P      ← read N² + write N²
  O = P @ V          → 读 P,V, 写 O    ← read N² + Nd + write Nd

总 HBM traffic: ~4N² + 2Nd 次读写 (≈ 3N² 主导)

量化 (N=2048, d=128, fp16, 单头):
  3 × 4,194,304 × 2 bytes ≈ 25 MB per head
  40 layers × 32 heads → 32 GB per forward pass → 仅 attention！

H100: 989 TFLOPS (fp16 peak) vs 3.35 TB/s HBM bandwidth
  算术强度 ≈ d = 128 FLOPS/byte
  需要的算术强度 ≈ 989 TFLOPS / 3.35 TB/s ≈ 295 FLOPS/byte
  → 差 2.3x → memory-bound! 大量 Tensor Core 空闲
```

### 1.2 FlashAttention (Dao et al. 2022) — IO-Aware 的核心

**核心洞察**：不把完整 S/P 矩阵写入 HBM。将 Q,K,V 分块（tile），每次加载一个 Q tile 和一个 KV tile 到 SRAM，直接在 SRAM 内完成 attention，只把最终 O 写回 HBM。

**关键算法 — Online Softmax（精确，非近似！）**：
```
递推公式（每遇到新 tile）:
  m_new   = max(m_old, max_row(S_tile))
  l_new   = l_old * exp(m_old - m_new) + sum(exp(S_tile - m_new))
  acc_new = acc_old * exp(m_old - m_new) + softmax(S_tile) @ V_tile

最终: output = acc / l  ← 与朴素 softmax 数学完全等价
```

证明依赖于恒等式：`exp(x_i - c) / Σ exp(x_j - c)` 对于任意常数 c 不变。每次用新的局部 max 替换，修正因子 `exp(m_old - m_new)` 精确补偿。

**HBM traffic**：O(N²d) → O(Nd × N/B_r)，其中 B_r 是 SRAM tile size。对于 N=2048, B_r=128，减少 ~16x。

### 1.3 FA1 → FA2 → FA3 → FA4：每代的取舍

| 版本 | 发布 | 核心创新 | 限制 | vLLM 应用 |
|------|------|---------|------|-----------|
| FA1 | 2022.05 | IO-aware tiling, online softmax | Single-thread over blocks, 无 ALiBi | vLLM 不再使用 |
| FA2 | 2023.07 | 序列维并行化, 减少 non-matmul FLOPs, 2x 加速 | 需要 SM80+ | vLLM **主力** backend |
| FA3 | 2024.03 | H100 wgmma 异步GEMM, FP8, 3x decode 加速 | **仅 SM90 (H100)** | vLLM H100 backend |
| FA4 | 2025.Q1 | Blackwell SM100, head_size 512+, Diff-KV | 仅 SM100, batch invariance broken | vLLM Blackwell backend |

**为什么不所有人都用 FA3？**
FA3 依赖 H100 的 wgmma (warp group matrix multiply accumulate) 指令和 TMA (tensor memory accelerator)——Hopper 架构独有的硬件特性。A100 (SM80) 和 AMD MI300X 不支持。
```python
# fa_utils.py:L78-82
if device_capability.major == 9:          # Hopper → FA3
elif device_capability.major == 10:       # Blackwell → FA4
else:                                     # everything else → FA2
```
用户可通过 `attention_config.flash_attn_version` 手动 override。

**FA2→FA3 的实际提升**：
- Decode (batch=256, seq=1, kv=4096): 3x throughput
- Prefill (batch=1, seq=4096): 1.5-2x（prefill 已是 compute-bound）
- 最大收益：GQA decode——wgmma 异步特性适合"很多小 matmul"的 decode 场景

### 1.4 FlashDecoding — Decode-phase 特化

vLLM 的 `triton_decode_attention.py` 受此启发：
- Insight: prefill 时 Q 维度大、K 维度大；decode 时 Q 维度=1、K 维度巨大
- 方案：将 KV 维度 split 成多个 groups，每个 group 用不同 thread block 处理
- vLLM 实现：`NUM_KV_SPLITS` 参数。Stage1 kernel 做 per-split partial softmax，Stage2 做 reduce

---

## 二、PagedAttention 起源 —— vLLM 的核心发明

### 2.1 PagedAttention 是 vLLM 的原创吗？

**是的。** PagedAttention 由 Kwon et al. 在 2023 年 SOSP 论文 "Efficient Memory Management for Large Language Model Serving with PagedAttention" 中首次提出。这篇论文获得了 SOSP'23 最佳论文奖。

**之前的方案**：
- HuggingFace TGI：每个请求预分配 max_seq_len 的连续显存。max_seq_len=128K 时一个请求就吃 ~GB。
- FasterTransformer (NVIDIA)：类似预分配策略，预留 max_seq_len 空间。

**PagedAttention 的类比本质**：
- 操作系统虚拟内存 → 进程看到连续虚拟地址，页表将虚拟页映射到物理帧
- PagedAttention → 模型看到连续逻辑序列位置，block_table 将逻辑位置映射到物理 KV block
- 关键简化：OS 页表是 2D（4-level radix tree），PagedAttention 的 block_table 是 1D（直接 offset 索引）

### 2.2 KV Cache 碎片问题的根源

```
假设 GPU 有 100 个 KV block，3 个请求:
  A: prefill 2000 tokens → 分配 blocks [0,127]
  B: prefill 500 tokens  → 分配 blocks [128,159]
  C: prefill 300 tokens  → 分配 blocks [160,178]

A 解码 100 步后结束 → 释放 [0,127]
B 继续解码，需要新 block → 从 [0,..] 分配
C 继续解码 → 从 [0,127] 中继续分配

结果: B 的 block 在 [128,159] 混杂 [0,..], C 的也在 [160,178] 混杂 [0,..]
→ 如果没有 block table → 无法连续访问 → 必须一次性预留最大长度（浪费显存）
→ 有 block table → 通过映射表在非连续物理 block 上模拟连续逻辑序列
```

### 2.3 为什么 block_table indirection 必须融合在 kernel 里

**vLLM 的做法**（triton_unified_attention.py:L237）：
```python
# 在 tiled attention 循环内
physical_block_idx = tl.load(block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE)
k_offset = physical_block_idx[None,:] * stride_k_cache_0 + kv_head_idx * stride_k_cache_2 + ...
K_load = tl.load(key_cache_ptr + k_offset, mask=..., other=0.0)
# 然后继续做 tl.dot(Q, K_load) → softmax → acc += V
```

**为什么不 "先 gather 到连续 tensor，再做 attention"？**

1. **显存浪费**：gather 需要额外 O(seq_len × num_heads × head_dim) 临时缓冲。seq=128K, d=128 → 32 MB/头
2. **额外 kernel launch**：gather + attention = 2 个 kernel。在 decode（许多小 batch）场景，launch overhead 显著
3. **延迟隐藏**：block_table lookup 延迟 ~40 cycles (L1 cache)，matmul ~200 cycles → 查找在等待 matmul 完成前就做好了
4. **对齐**：FlashAttention 的 tile 循环 (TILE_SIZE=16/32) 和 KV Cache 的 block 粒度 (BLOCK_SIZE=16/32) 天然对齐——每个 tile 恰好是一个 block 或 block 的一部分

### 2.4 竞争方案对比

| 方案 | 来源 | 核心思想 | vLLM 为什么不采用 |
|------|------|---------|-----------------|
| Contiguous pre-allocation | HF TGI | 每个请求预分配 max_seq_len 连续 KV | 显存浪费太大。128K×d×layers×dtype=GB级 |
| vAttention | ASPLOS'24 | CUDA virtual memory API 做 page table | 依赖 cuMemMap, GPU 页表大小受限 |
| RadixAttention | SGLang | Radix tree 做 prefix-aware 缓存 | SGLang 也用 fused block_table，但用 radix tree 做前缀匹配（vLLM 用 hash-based） |
| TensorRT-LLM | NVIDIA | 类 block_table，预编译 CUDA kernel | 闭源，不可定制 |
| LightLLM | 学术/社区 | DeepSeek 早期的轻量框架 | vLLM 的 triton_decode_attention.py 和 triton_prefill_attention.py 大量借鉴了 LightLLM |

**vLLM 为什么选 hash-based prefix caching 而非 radix tree？**
- Block table 是简单的 2D int tensor，融合到 kernel 只需一次 `tl.load(table + offset)`
- Radix tree 的指针追踪不适合 GPU kernel
- Hash-based 设计比 radix tree 更简单，更容易维护

### 2.5 vLLM 的 Block Table 数据结构

```python
# kv_cache_utils.py:L114
class KVCacheBlock:
    block_id: int             # 物理 block ID (0..num_gpu_blocks-1)
    ref_cnt: int = 0          # 引用计数（多个请求共享 prefix 时 >1）
    _block_hash: ...          # prefix caching 的 hash key
    prev_free_block / next_free_block  # 双向链表（用于 free list）

# block_pool.py:L130
class BlockPool:
    # 核心操作:
    #   allocate_new_blocks(request) → list[KVCacheBlock]
    #   free_blocks(ordered_blocks)   → 回到 free list
    #   get_cached_block(block_hash)  → 查找已计算的 block
```

---

## 三、FA + PA 的融合 —— vLLM 的独特创新

### 3.1 融合架构全景

```
triton_unified_attention.py::kernel_unified_attention —— FA+PA 融合 kernel

输入: Q, K_cache, V_cache, block_table, seq_lens
内核循环:
  for each tile in KV range:
    1. physical_block_idx = block_table[seq][tile_idx]     ← PA: 逻辑→物理映射
    2. K_tile = load(K_cache, offset=physical_block_idx)   ← PA: 非连续地址 (5D stride)
    3. V_tile = load(V_cache, offset=physical_block_idx)   ← PA: 非连续地址 (4D stride)
    4. S = scale * tl.dot(Q, K_tile)                       ← FA: tiled matmul
    5. M, L, P = softmax_step(S, M, L)                     ← FA: online softmax
    6. acc = acc * alpha + tl.dot(P, V_tile)               ← FA: 累加
  end
  output = acc / L

融合 = 在同一个循环体中同时做 block_table 查找 + tiled attention。
```

**关键 insight**: FlashAttention 的 tiled 循环和 PagedAttention 的 block 粒度天然对齐。
- FlashAttention tile 大小: TILE_SIZE (16/32)
- KV Cache block 大小: BLOCK_SIZE (16/32)
- 每个 tile 恰好是一个 KV block → block_table lookup 完美嵌入 tiling 循环

### 3.2 融合 vs 分离的量化

```
分离方案 (gather then compute):
  gather_kv(block_table) → contiguous_KV    # 额外 HBM write: Nd bytes
  flash_attention(Q, contiguous_KV)          # 额外 HBM read: Nd bytes
  总额外 HBM: 2Nd

融合方案 (current vLLM):
  flash_attention_with_block_table(Q, K_cache, V_cache, block_table)
  额外 HBM: 0 (block_table 在 L1 cache 中, lookup 被 matmul 延迟完全隐藏)

量化: N=4096, d=128, fp16 → 2Nd = 2MB/头
  40层 × 32头 × 2MB = 2.56 GB per forward pass 节省
```

### 3.3 V1 vs V2 的 kernel 形态演进

**V2 (legacy)**：separate prefill/decode kernels
- `triton_prefill_attention.py` (context_attention_fwd) — 源自 SGLang
  - 用于 prefill：Q 序列长，KV 完整
  - 不使用 block_table（连续 K,V tensor）
- `triton_decode_attention.py` (decode_attention_fwd) — 源自 SGLang/LightLLM
  - 用于 decode：Q=1 token, KV 长
  - 支持 block_table, split-kv 并行

**V1 (当前)**：unified 2D/3D kernel
- `triton_unified_attention.py` (kernel_unified_attention)
  - **2D mode** (IS_3D=False): 完整处理每个 (Q_block, kv_head)。用于 prefill 或小 batch。
  - **3D mode** (IS_3D=True): KV 范围分割成 segments，各自独立计算，最后 `reduce_segments` 合并。用于 decode 大 batch。
  - 两种模式共享完全相同的 kernel 代码，通过 tl.constexpr 分支
  - 关键 PR: #40631 "Unify 2D/3D kernels"

**为什么统一？**
- 减少维护成本——两份 kernel 的对齐逻辑（mask、sliding window 等）容易出错
- Prefill-decode 混合 batch：2D mode 一次处理完成
- 为什么 3D 不总是更好？3D 需要额外 segment 中间缓冲。prefill (seq=4096, batch=1) 时 SM 利用率已足够，2D 更快

### 3.4 5D/4D Strided 寻址

vLLM 的 KV cache 用多维 stride 支持非连续 block 的高效访问：
```
K_cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]  ← 5D
V_cache: [num_blocks, num_kv_heads, head_size, block_size]          ← 4D

其中 x = 16 / element_size (for vectorized load, x=8 for fp16)
```

**为什么这么复杂？**
- `head_size // x, x` 的最后一维允许 vectorized load（一次 load 8 个 fp16 = 128 bits）
- 相邻 token 的 K 值在内存中连续 → coalesced access
- 物理 block 间不连续，但通过 `physical_block_idx * stride_k_cache_0` 跳转

---

## 四、Backend 调度架构 —— vLLM 的 20+ Backend

### 4.1 三层调度

**Layer 1**: Platform 检测 — `vllm/platforms/__init__.py`
- 自动检测 CUDA/ROCm/XPU/CPU/TPU
- `current_platform` 全局单例

**Layer 2**: Backend 优先级 — `vllm/platforms/cuda.py`
```
非 MLA, SM100:    [FLASHINFER, FLASH_ATTN, TRITON_ATTN, ...]
非 MLA, 非 SM100: [FLASH_ATTN, FLASHINFER, TRITON_ATTN, ...]
MLA, SM100:       [FLASHINFER_MLA, CUTLASS_MLA, ...]
MLA, 非 SM100:    [FLASH_ATTN_MLA, FLASHMLA, ...]
```

**Layer 3**: 逐层验证 — `vllm/v1/attention/selector.py`
- `get_attn_backend(head_size, dtype, ...)` 创建 `AttentionSelectorConfig`
- 遍历优先级列表，`backend.validate_configuration()` 验证
- 选第一个通过所有验证的 backend

### 4.2 为什么是 20+ Backend 而不是一个？

1. **硬件多样性**：CUDA (Ampere/Hopper/Blackwell)、AMD ROCm、Intel XPU、CPU。一个 kernel 无法优化所有架构
2. **计算模式多样性**：
   - Decode (memory-bound, 单 token) → FlashInfer 专精
   - Prefill (compute-bound, 全prompt) → FA 专精
   - MLA (低秩 KV) → 完全不同的内存布局
   - Sparse → 部分 KV 索引
3. **数值精度与特性**：FP8 → FA4/FlashInfer；ALiBi → FA2/FA3；Sinks → FA3/FA4
4. **渐进式部署**：新 backend 先在特定配置启用，成熟后扩大范围

**架构哲学**："不是选最好的 kernel 优化它，而是让每种场景都有最合适的 kernel。"

### 4.3 vLLM vs SGLang backend 策略

| 维度 | vLLM | SGLang |
|------|------|--------|
| Backend 数量 | 20+ backend, auto-selection | 少数 backend, 倾向 FlashInfer |
| 策略 | "让每个场景都有最合适的" | "少而精，极致优化少数组合" |
| 平台支持 | CUDA+ROCm+XPU+CPU+TPU | 聚焦 CUDA |
| 可扩展性 | 插件化注册, 第三方可添加 | 相对封闭 |
| 适用场景 | 社区驱动的多硬件生态 | 极致优化特定硬件 |

---

## 五、关键设计决策

### 5.1 为什么 scale 是外部参数而非 kernel 内计算？

同 Ch01——scale 是模型属性（`head_dim**-0.5`），不是 attention kernel 的责任。某些模型（Gemma）有 softcap，某些量化场景需要 scale 融合（`scale * k_token_head_scales`）。传给 kernel 而不是让 kernel 计算 = 单一职责。

### 5.2 为什么 causal mask 在 kernel 内用算术？

已在 Ch01 §4.3 详述。本章强调：在 FA+PA 融合的上下文中，O(1) 空间 mask 的好处更明显——block_table lookup 已经是不规则的，如果 mask tensor 也是 O(N²)，损失更大。

### 5.3 为什么 softcap 融合在 kernel 里？

```python
# triton_unified_attention.py:L308-309 (constexpr 分支, 编译时优化掉)
if USE_SOFTCAP:
    S = apply_softcap(S, softcap)  # x * tanh(S/x)
```
Softcap (Gemini 2024 引入) 防止 attention score 过大。融合在 kernel 避免额外 HBM round-trip。

---

## 六、源码考古 —— 关键文件演进

| 文件 | 首次引入 | 关键 PR | 当前状态 |
|------|---------|--------|---------| 
| `v1/attention/backends/flash_attn.py` | v1 重构 (#31916) | FA3/FA4/FP8/DCP support | 活跃, ~1222 行 |
| `v1/attention/backends/fa_utils.py` | v1 → 独立抽取 | FA version selection, reshape_and_cache | 活跃 |
| `v1/attention/ops/triton_unified_attention.py` | #31916 后 | **#40631 unify 2D/3D**, #38378 FP8 quant | 活跃, 核心 kernel ~750 行 |
| `v1/attention/ops/triton_decode_attention.py` | 早期 | 源自 SGLang/LightLLM, PAGE_SIZE≥1 | 稳定, normal/grouped 双路径 |
| `v1/attention/ops/triton_prefill_attention.py` | 早期 | 源自 SGLang, SWA 支持 | legacy, 被 unified 取代趋势 |
| `v1/attention/ops/triton_attention_helpers.py` | #31916 后 | 共享 helper 抽取 | 稳定 |
| `v1/attention/ops/chunked_prefill_paged_decode.py` | 较晚 | 2D kernel, IBM 作者贡献 | 活跃 |
| `v1/core/block_pool.py` | v0→v1 重写 | BlockPool + prefix cache hash map | 稳定, ~500 行 |
| `v1/core/kv_cache_utils.py` | v0→v1 重写 | KVCacheBlock, BlockHash | 稳定 |
| `v1/kv_cache_interface.py` | v1 架构 | KVCacheSpec, KVQuantMode | 稳定 |

---

## 七、给 Writer 的建议

### 最重要的 intuition

**FA 和 PA 解决两个正交问题，融合到一个 kernel：**
- FlashAttention → "怎么算得快"：减少 HBM 往返。核心 = tiled matmul + online softmax
- PagedAttention → "怎么存得省"：消除 KV cache 碎片。核心 = block table 映射
- 融合依赖于巧合：FA 的 tile 循环和 KV block 的 block 粒度天然对齐

### 本章叙事弧线

```
朴素 attention (O(N²) HBM, 连续KV分配)
  ├─ 路线A: FlashAttention → tiling → online softmax → FA2 → FA3
  ├─ 路线B: PagedAttention → block table → block pool → prefix cache
  └─ 汇合: FA tiling循环 + block_table lookup = vLLM unified attention
```

### 易混淆概念

1. **FA 的 "O(n²)" 问题 ≠ 计算瓶颈**：FA 不能减少 compute FLOPs (仍然是 O(n²d))。FA 减少的是 HBM traffic。
2. **Online softmax 是精确的，不是近似**：依赖恒等式 `exp(x_i - c) / Σ exp(x_j - c)` 对任意 c 不变
3. **Block Table ≠ Mask**：Mask 决定"哪些 KV 可见"；Block Table 决定"可见的 KV 在物理内存哪里"
4. **Block size ≠ Tile size**：Block size (e.g. 16) = KV cache 分配粒度；Tile size (e.g. 16/32) = kernel 循环粒度。通常相等，非必须。

### 源码与理论映射表

| 数学概念 | vLLM 源码位置 | 说明 |
|---------|-------------|------|
| Online softmax 递推 | `triton_attention_helpers.py:softmax_step()` → `triton_unified_attention.py:L325` | (M, L, acc) 一次迭代 |
| Tiled matmul | `triton_unified_attention.py:L232-340` | tile 循环 + `tl.dot(Q, K)` |
| Block table 查找 | `triton_unified_attention.py:L237` | `physical_block_idx = tl.load(block_tables_ptr + ...)` |
| 非连续 KV load | `triton_unified_attention.py:L240-265` | 5D/4D strided addressing |
| FA 版本选择 | `fa_utils.py:L77-140` | SM90→FA3, SM100→FA4, else→FA2 |
| KV block 分配 | `block_pool.py:130 BlockPool` | free list → allocate → block table update |
| Causal + SW mask | `triton_attention_helpers.py:L261 compute_kv_seq_mask()` | kernel 内算术, O(1) 空间 |
| Softcap | `triton_unified_attention.py:L308-309` | `x * tanh(S/x)` fused |
| ALiBi | `triton_attention_helpers.py:apply_alibi_to_score()` → L315-318 | per-head slope 偏置 |

### 深度覆盖 vs 点到为止 vs 不应涉及

**深度覆盖**:
1. Tiled attention + online softmax 精确性证明
2. PA 的 block table 逻辑→物理映射原理
3. FA+PA 融合的 kernel 核心循环 walkthrough
4. HBM traffic 量化对比（具体数字的表格）
5. vLLM FA version selection + backend dispatch

**点到为止**:
- Prefix caching (hash-based) — 留给 Ch05 Memory Management
- KV cache 分配/回收 — 留给 Ch05
- Split-kv 并行 — 可提及概念

**不应涉及**:
- DCP (Decode Context Parallelism) — Ch08-09
- Cascade Attention — Ch25+
- MLA — Part 4 专家模型专题
- Marlin/TurboQuant 量化 — Part 5
