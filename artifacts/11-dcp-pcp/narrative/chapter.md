# 第11章：DCP/PCP — 上下文并行

> TP 切分 attention head 维度。但当 `num_kv_heads < tp_size` 时——例如 DeepSeek-R1 的 1 个
> KV 头在 8 张 GPU 上——KV 缓存会被**复制** 8 次。DCP 的解法：不切头，切序列。把 KV 缓存
> 沿时间轴分片——每张 GPU 只存 1/8 的 token 的 K 和 V。

---

## 这章要做什么？

第 8 章讲的 TP 有一个隐藏假设：`num_kv_heads >= tp_size`。当这个条件不满足时——例如 MLA 架构的 1 个 KV 头在 8 张 H100 上——TP 只能复制 KV 头。8 张 GPU，每张都存着完全相同的 8 份 KV 缓存副本。

DCP（Decode Context Parallelism）解决的就是这个浪费。它沿序列维度切 KV 缓存：token 0-511 在 GPU 0，token 512-1023 在 GPU 1……每张 GPU 只存 1/dcp_size 的 KV 数据。Attention 计算时，query 被 all-gather 收集，每张 GPU 对自己那部分 KV 做 attention，然后 LSE-merge 合并部分结果。

PCP（Prefill Context Parallelism）是 DCP 的 prefill 对应物。它把 prefill 的 Q 分片——每张 GPU 处理 1/pcp_size 的 query，通过 ring attention 或 KV all-gather 获取其他 GPU 的 KV。

学完这章你能：
- 量化 KV 缓存复制的浪费——用 `dcp_size = tp_size / num_kv_heads` 消除
- 追踪 DCP attention 的 6 步流程：query all-gather → 本地 KV attention → LSE merge → scatter
- 区分 AG+RS（2 次 NCCL）和 A2A（1 次 NCCL）两种通信模式

---

## 11.1 KV 缓存复制问题

### Theory: 什么时候 TP 不够用？

TP 沿 head 维度切权重和 KV 缓存。每个 GPU 拥有 `num_kv_heads / tp_size` 个 KV 头。当 `num_kv_heads >= tp_size` 时——例如 Llama-3.2-70B 的 8 个 KV 头在 8 张 GPU 上——每张 GPU 恰好有 1 个 KV 头，没有浪费。

但当 `num_kv_heads < tp_size` 时——例如 DeepSeek-R1 的 1 个 MLA KV 头在 8 张 GPU 上——`num_kv_heads / tp_size <= 0`，每张 GPU 只能有 1 个（不能有 0.125 个 KV 头）。TP 的处理：**复制。** 8 张 GPU 每张都存储完整的 1 个 KV 头。

浪费比例 = $max(1, tp\_size / num\_kv\_heads)$。

| 模型 | KV 头数 | TP Size | 复制因子 | 浪费 |
|------|---------|---------|---------|------|
| Llama 70B (GQA) | 8 | 8 | 1× | 0% |
| DeepSeek V3 (MLA) | 1 | 8 | 8× | 87.5% |
| Qwen 235B (GQA) | 4 | 8 | 2× | 50% |

8 倍复制意味着 87.5% 的 KV 缓存显存是浪费的——同样的数据存了 8 份。这在 batch 较小时无所谓（单份 KV 缓存也不大），但 batch 稍大——KV 缓存可能占 60%+ 的显存——就很致命。

### Source Trail

打开 `vllm/v1/attention/backends/flash_attn.py:595`：

```python
class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True
    dcp_world_size = get_dcp_group().world_size   # 自动解析
    dcp_rank = get_dcp_group().rank_in_group
```

DCP 的信息在 `AttentionImplBase.__new__()` 中自动注入到每个 attention 实现——`dcp_world_size > 1` 触发 DCP 路径。

---

## 11.2 DCP 的 Attention 计算

### Source Trail

打开 `vllm/v1/attention/backends/flash_attn.py:884`——`_forward_with_dcp()`。

**Step 1 — Query AllGather（L906）：**

```python
query_across_dcp = get_dcp_group().all_gather(query, dim=1)
# query: [num_tokens, H_local, head_size]
# → [num_tokens, H_local × dcp_size, head_size]
```

Query 需要在所有 DCP rank 间共享——因为每个 rank 只有部分 KV 缓存。AllGather 把各 rank 的 Q 拼接在一起，这样每个 rank 都看到完整的 Q。

**Step 2 — 本地 KV Attention（L917-L930）：**

```python
context_output, context_lse = flash_attn_varlen_func(
    q=query_across_dcp,   # Full Q
    k=key_cache,           # LOCAL key cache (partial KV!)
    v=value_cache,         # LOCAL value cache
    causal=False,          # 不连续位置 → 不能假设因果性
    seqused_k=local_seq_lens,  # 本地序列长度
    return_softmax_lse=True,   # 需要 LSE 做 merge
)
```

`causal=False` 是因为 KV 缓存中的 token 位置不连续（被 DCP 交错到不同 rank）。`return_softmax_lse=True` 是关键——DCP 必须启用这个 flag。

**Step 3 — LSE Merge（L941-L946）：**

每个 rank 的 attention 输出是部分的——因为 KV 是部分的。需要把各 rank 的部分输出合并。`merge_attn_states()` 使用 LSE 做加权融合：

```
LSE_global = max(LSE_0, LSE_1, ..., LSE_{P-1})
weight_i = exp(LSE_i - LSE_global) / sum_j exp(LSE_j - LSE_global)
output = sum_i weight_i × partial_output_i
```

这和 Chapter 3 的 partitioned softmax merge 是同一个算法——只是这里跨 GPU 而不是跨 KV tile。

### Theory: 通信 vs 精度的 Trade-off

AG+RS 模式做两次 NCCL 调用：AllGather LSE + ReduceScatter output。为什么不能一次搞定？因为 LSE merge 需要所有部分 LSE 才能计算全局 LSE——这是一个 all-to-all 依赖。必须先收集 LSE，再计算 merge weights，再 scatter 结果。这是信息论上的必然——不是工程问题。

A2A 模式把 LSE 和 output 打包到一次 All-to-All 中——减少到 1 次 NCCL 调用。代价是打包/解包的 Triton kernel 计算。对于通信量大的模型（MLA），省 1 次 NCCL 调用的收益 > 打包/解包的开销。

---

## 11.3 KV 缓存交错

### Source Trail

打开 `vllm/v1/worker/gpu/cp_utils.py:36-61`——`prepare_dcp_local_seq_lens` Triton kernel。

```python
# Round-robin distribution of tokens across DCP ranks
rounds = seq_lens // (dcp_size * cp_interleave)
remainder = seq_lens % (dcp_size * cp_interleave)
remainder = max(remainder - dcp_rank * cp_interleave, 0)
remainder = min(remainder, cp_interleave)
local = rounds * cp_interleave + remainder
```

`cp_kv_cache_interleave_size` 控制 token 分配粒度：
- **1（token-level）:** token_0 → rank 0, token_1 → rank 1, ..., token_7 → rank 7, token_8 → rank 0, ...
- **16（block-level，=block_size）:** token_0..15 → rank 0, token_16..31 → rank 1, ...

Token-level 最均匀但 block-table 的管理更复杂（每个 rank 的 block 都只有部分 token）。Block-level 和 PagedAttention 的 block 大小对齐——简化 block 分配但可能有轻微的不平衡（最后一个 block 可能不满）。

---

## 11.4 PCP：Prefill 的并行化

### Source Trail

PCP 仍在活跃开发中。当前支持两种模式（`vllm/v1/attention/backends/flashinfer.py:212`——`BatchDCPPrefillWrapper`）：

**模式 1——部分 Q，完整 KV：** 各 GPU all-gather KV 缓存，然后每个 GPU 对自己那部分 Q 做 attention。适合短/中等长度的 prompt——KV all-gather 的通信量与 seq_len 成正比，在 seq_len 不大时可接受。

**模式 2——部分 Q，部分 KV（ring attention）：** 各 GPU 持有 Q 和 KV 的部分分片。Q 分片在 ring 中传递——每个 GPU 拿到邻居的 Q，对自己那部分 KV 做 attention，传下去。适合长 prompt——通信量固定（不随 seq_len 增长）。

`★ Insight ─────────────────────────────────────`
PCP vs DCP 的命名反映了它们优化不同的"上下文"阶段。DCP 优化 decode——KV 缓存分片减少 decode 阶段的显存占用，增加 batch size。PCP 优化 prefill——Q 分片减少 prefill 的 TTFT。两者在设备 mesh 中正交——`total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`——一个 GPU 可以同时属于一个 DCP 组和一个 PCP 组。
`─────────────────────────────────────────────────`

---

## 11.5 配置决策

### Theory: 什么时候用 DCP？

决策树：
```
tp_size > num_kv_heads?
  ├── NO  → DCP 不需要 （KV 缓存已经被 TP 均匀分片）
  └── YES → dcp_size = tp_size / num_kv_heads
            → 消除 KV 复制
            → trade-off: 增加通信（2 次 NCCL / layer）
```

通信开销：每层 attention 2 次 NCCL（AG+RS）或 1 次（A2A）。对于 H100 NVLink（~900 GB/s 双向），AllGather + ReduceScatter 的延迟通常在 <0.5 ms——相对于 decode 的 attention 计算时间很小（~2-5 ms per layer）。通信占比 <10%——与 TP 的 AllReduce 开销可比。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `analyze_kv_cache_replication()` | 原创量化分析 | 推导 DCP 必要性 |
| `simulate_dcp_attention()` | `cp_utils.py:L36-L61` | 本地序列长度计算逻辑一致；通信量为估计 |
| `lse_weighted_merge()` | `merge_attn_states.py` | 相同的 LSE-merge 算法 |
| `recommend_cp_config()` | 原创决策辅助 | 基于复制因子的 DCP size 推荐 |

---

## 验证

```bash
cd artifacts/11-dcp-pcp && python -m pytest tests/ -q
# 10/10 passed ✅
```

---

## 总结

- **DCP 解决 `tp_size > num_kv_heads` 时的 KV 缓存复制。** 沿序列维度分片 KV——每 GPU 存 1/dcp_size 的 token 数据。
- **DCP attention 的 6 步：** query all-gather → 本地 KV attention(causal=False) → LSE 收集 → LSE merge → scatter 输出。
- **AG+RS（默认）vs A2A（低延迟）。** A2A 把 2 次 NCCL 减到 1 次——对于 MLA 模型值得。
- **PCP 是 DCP 的 prefill 对应物。** Q 分片减少 TTFT——仍在活跃开发。

---

**下一章：** 第12章 — KV Cache Offload 与层级存储

DCP 优化了 KV 缓存的多 GPU 分布。但单 GPU 显存仍然是有限的——如果 KV 缓存太大（超长上下文 + 大批量），能把它卸到 CPU 内存或 NVMe SSD 吗？第 12 章将分析 vLLM 的 KV offload 机制。

---

← 第10章 | 第12章 →
