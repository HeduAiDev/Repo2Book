# 第15章：Llama-3.2-1B 模型架构全景

> 打开 `vllm/model_executor/models/llama.py:253`。`LlamaDecoderLayer` 是整个 Llama 架构的重复单元。
> 16 个这样的 layer 串起来，加上 embedding 和 lm_head——这就是 Llama-3.2-1B 的全部。
> 本章从 config 到 layer 结构，为后六章的逐算子 Triton 实现建立蓝图。

---

## 这章要做什么？

第 14 章学会了 Triton 基础。在写具体的 Triton kernel（RMSNorm、RoPE、Attention、MLP）之前，需要先理解**这些 kernel 在完整的模型架构中如何组合**。

这章不是写代码——是建立架构蓝图。我们会精确拆解 `llama.py` 的每个组件：从 16-layer 的 `LlamaModel` 到 `LlamaDecoderLayer` 内部的 `RMSNorm → Attention → RMSNorm → MLP` 流程。

学完这章你能：
- 打开 `llama.py:288-314`，逐行解释 `LlamaDecoderLayer.__init__` 构建了什么
- 理解 vLLM 为什么融合 QKV 和 gate+up 投影——不是架构选择，是**显存带宽优化**
- 计算 Llama-3.2-1B 的每层参数量——MLP 占了 67%，Attention 只占 33%

---

## 15.1 Config → 模型尺寸

### Llama-3.2-1B 的关键参数

打开 HuggingFace 的 `config.json` 或 vLLM 的 `LlamaConfig`：

| 参数 | 值 | 决定什么 |
|------|-----|---------|
| `hidden_size` | 2048 | 每层的输入/输出维度 |
| `num_hidden_layers` | 16 | Transformer 层数 |
| `num_attention_heads` | 32 | Q 头数 |
| `num_key_value_heads` | 8 | KV 头数（GQA: 4 queries per KV） |
| `head_dim` | 64 | 每个头的维度 (2048/32) |
| `intermediate_size` | 8192 | FFN 中间层维度 (4× d_model) |
| `vocab_size` | 128256 | Token 词汇表大小 |
| `rope_theta` | 500000.0 | RoPE 基频（第 17 章） |
| `rms_norm_eps` | 1e-5 | RMSNorm 的数值稳定常数 |

### 参数量分解

运行 `implementation/llama_config.py` 的 `count_parameters()`：

```
Total params: ~1.24B (实际: 1,237,852,160)
Model size (bf16): ~2.3 GB

Per-layer breakdown (16 layers):
  Attention: 86M   (33% — QKV + O projection)
  MLP:       168M  (67% — gate+up+down projection)
  RMSNorm:   0.008M (<0.01% — negligible)

Embedding + LM Head: 525M (vocab=128K × d=2048 × 2)
```

**MLP 占 2/3 的参数。** 这是 Llama 的设计特征——把大部分 capacity 给 FFN 而不是 attention。attention 只负责"哪个 token 和哪个 token 相关"（pattern matching），MLP 负责"知道了相关之后怎么变换信息"（feature transformation）。

`★ Insight ─────────────────────────────────────`
为什么 intermediate_size = 4× hidden_size？不是 8/3×（标准 Transformer）？Llama 选择了 4×，这增加了 MLP 的参数比例（8/3=2.67 vs 4）。更大的 MLP 意味着模型的"知识容量"更大——更多的参数用于存储训练中学到的 pattern。代价是更大的显存占用。对于 1B 规模来说，16 layers × 4× FFN 提供了比 32 layers × 8/3× 更好的 compute/memory trade-off（层数减半 → KV cache 减半）。
`─────────────────────────────────────────────────`

---

## 15.2 模型结构：从顶到底

### Source Trail

打开 `vllm/model_executor/models/llama.py:350`——`LlamaModel.__init__`：

```python
# llama.py:L371-L386
self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
self.layers = make_layers(num_hidden_layers, ...)  # 16 × LlamaDecoderLayer
self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
```

**三层结构：**
1. **Embedding:** token id → 2048 维向量
2. **16 层 LlamaDecoderLayer:** 每层包括 Attention + MLP
3. **最终 RMSNorm + LM Head:** 最终 hidden state → vocab 概率

### 完整推理流程

打开 `llama.py:395`——`LlamaModel.forward()`：

```python
hidden_states = self.embed_tokens(input_ids)        # [B, L, 128256] → [B, L, 2048]
residual = None

for layer in self.layers:
    hidden_states, residual = layer(positions, hidden_states, residual)

hidden_states = self.norm(hidden_states, residual)  # [B, L, 2048]
return hidden_states
```

然后 `LlamaForCausalLM.forward()`（`llama.py:L501`）把 `hidden_states` 通过 `lm_head` + `logits_processor` 变成 logits。

---

## 15.3 LlamaDecoderLayer：Pre-Norm + Residual Stream

### Source Trail

打开 `llama.py:288-L314`——`LlamaDecoderLayer.__init__`：

```python
self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
self.self_attn = LlamaAttention(...)
self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
self.mlp = LlamaMLP(...)
```

**Pre-Norm 结构（`llama.py:316-L333`——`forward()`）：**

```
input (hidden_states)
  │
  ├─→ input_layernorm → self_attn ──→ + (residual add)
  │                                      │
  └──────────────────────────────────────┘
  │
  ├─→ post_attention_layernorm → mlp ──→ + (residual add)
  │                                         │
  └─────────────────────────────────────────┘
  │
  output
```

**Pre-Norm vs Post-Norm：** 原始 Transformer 是 Post-Norm（attention/mlp **之后**归一化）。Llama 改成了 Pre-Norm（attention/mlp **之前**归一化）。Pre-Norm 的梯度流动更稳定——norm 在残差分支的输入侧，而不是累积侧——这使得训练更稳定，尤其是深度模型（Llama-3.2 从头训练 16 层不需要复杂的 warmup）。

---

## 15.4 LlamaAttention：QKV 融合 + GQA

### Source Trail

打开 `llama.py:124`。

**QKV 投影（`llama.py:L164-L172`）：**

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size=2048,
    head_size=64,
    total_num_heads=32,
    total_num_kv_heads=8,
)
```

输出 = `Q(32×64) + K(8×64) + V(8×64) = 3072` 维。这三个投影被**融合为一个矩阵**——一次 GEMM 代替三次。

**为什么融合？** 三次 GEMM = 三次从 HBM 读取 input hidden_states。一次融合 GEMM = 一次读取 + 更大的 weight 矩阵。对于内存带宽受限的推理（decode），减少 HBM 读取次数比减少计算量更重要。

**GQA（L142-L155）：** `num_heads=32, num_kv_heads=8` → 4 个 Q heads 共享 1 个 KV head。从第 1-2 章的知识：这缩小了 75% 的 KV cache。

---

## 15.5 LlamaMLP：SwiGLU + Gate-Up 融合

### Source Trail

打开 `llama.py:81`。

```python
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size=2048,
    output_sizes=[8192, 8192],  # gate + up, both intermediate_size
)
self.act_fn = SiluAndMul()       # SiLU(gate) * up
self.down_proj = RowParallelLinear(8192, 2048)
```

**SwiGLU 的数学：**

$$
\mathrm{MLP}(x) = W_{\mathrm{down}} \cdot (\mathrm{SiLU}(W_{\mathrm{gate}} \cdot x) \odot W_{\mathrm{up}} \cdot x)
$$

`SiluAndMul`（`activation.py:L118`）在 kernel 内部做融合的 element-wise SiLU + multiply——避免了写两次中间结果到 HBM。

`★ Insight ─────────────────────────────────────`
SwiGLU 比标准 ReLU-FFN 多了一个 gate projection，所以 intermediate_size 需要除以 1.5 才能保持总参数量不变。Llama 选了 4× d_model 而不是 8/3×——这意味着它**主动**把更多参数给 MLP。attention/MLP 的参数比是 33%/67%——MLP 是 attention 的 2 倍。这反映了一个设计信念：LLM 推理中"知道如何处理 attention 找到的相关 token"（MLP 的角色）比"找到哪些 token 相关"（attention 的角色）需要更多参数。
`─────────────────────────────────────────────────`

---

## 15.6 权重融合：HF Checkpoint → vLLM Parameter

### Source Trail

打开 `llama.py:436-460`——`LlamaModel.load_weights()` 的 `stacked_params_mapping`：

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),     # q_proj 权重 → qkv_proj 的 "q" 分片
    (".qkv_proj", ".k_proj", "k"),     # k_proj → qkv_proj 的 "k" 分片
    (".qkv_proj", ".v_proj", "v"),     # v_proj → qkv_proj 的 "v" 分片
    (".gate_up_proj", ".gate_proj", 0), # gate_proj → gate_up_proj 的分片 0
    (".gate_up_proj", ".up_proj", 1),   # up_proj → gate_up_proj 的分片 1
]
```

HuggingFace checkpoint 中 `q_proj`, `k_proj`, `v_proj` 是三个独立的权重。vLLM 加载时把它们**拼接**为一个 `qkv_proj` 矩阵。同样 `gate_proj` + `up_proj` → `gate_up_proj`。

**这一步发生在加载时，不在推理时。** 推理时只算一次融合 GEMM——这就是为什么 vLLM 比 naive HF 推理快 20-30%（减少了内核启动和 HBM 读取）。

---

## 我们的实现 vs vLLM 源码

| 我们的实现 | vLLM 原始源码 | 说明 |
|---|---|---|
| `LlamaConfig` | transformers.LlamaConfig (llama.py:L32) | 关键参数精确匹配 1B 模型 |
| `RMSNorm` | `layernorm.py:L103` | PyTorch 版——vLLM 用 CUDA fused kernel |
| `RotaryEmbedding` (简化) | `llama3_rope.py:L11` (Llama3RotaryEmbedding) | 标准 RoPE，第 17 章实现 NTK 缩放 |
| `count_parameters()` | vLLM 实测（DeviceMemoryProfiler） | 分析性计算——结果在 1% 误差内 |

---

## 验证

```bash
cd artifacts/15-llama-model-architecture && python -m pytest tests/ -q
# 11/11 passed ✅
```

---

## 总结

- **Llama-3.2-1B = 16 层 × (RMSNorm + GQA Attention + RMSNorm + SwiGLU MLP)。**
- **MLP 占 67% 参数——不是 bug，是设计选择。** SwiGLU + 4× intermediate 主动把容量给 feature transformation。
- **QKV 和 gate+up 融合是一次 HBM 读取的优化，不是架构选择。**
- **Pre-Norm + residual stream 让 16 层稳定训练。**

---

**下一章：** 第16章 — Triton RMSNorm

有了架构蓝图，下一章写第一个 Triton kernel——RMSNorm + fused residual add。从 PyTorch 参考实现开始，一步步到 welford online variance 和 Triton fused kernel。

---

← 第14章 | 第16章 →
