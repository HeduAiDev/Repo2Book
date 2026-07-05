# ch31 implementation notes —— MLA 原理精读：论文忠实的小型参考实现

本章是 **primer 原理章**（豁免"只做减法"）：`implementation/` 不是任何真实代码仓的精简版，
而是把 DeepSeek-V2 论文（arXiv:2405.04434 §2.1）的公式**逐条**变成可跑的 NumPy 代码，供
读者用小维度亲手验证"压缩率""权重吸收恒等式""解耦 RoPE 为何必须存在"这几件事。落地时
回指 [第 20 章 MLA on NPU](../ch20-mla-on-npu/narrative/chapter.md) 的真实代码
`vllm_ascend/attention/mla_v1.py`。

## 文件划分

| 文件 | 覆盖论文小节 | 内容 |
|---|---|---|
| `numerics.py` | — | softmax / 因果掩码小工具（非论文公式，公共依赖） |
| `mha_baseline.py` | §2.1.1 Eq.1-8 | 标准 MHA 基线：KV cache = 2·n_h·d_h·l 的由来 |
| `low_rank_mla.py` | §2.1.2 Eq.9-13 + 吸收恒等式 | 低秩 KV 联合压缩、q 侧低秩、q 侧/o 侧权重吸收 |
| `decoupled_rope.py` | §2.1.3 Eq.14-19 | 解耦 RoPE + "为什么直接对 k^C 加 RoPE 不可吸收"的数值证明 |
| `mla_reference.py` | Eq.9-19 装配 | 端到端前向：prefill 物化路径 vs decode 吸收路径，逐 token 验证等价 |
| `kv_cache_table.py` | §2.1.4 Table 1 | KV cache 元素数对比公式 + DeepSeek-V2 真实数字代入 |

## Paper Map（公式 ↔ 函数 ↔ 落地代码锚点）

| 论文公式 | 参考实现函数 | 对应 vllm_ascend/attention/mla_v1.py |
|---|---|---|
| Eq.1-3（Q/K/V 投影） | `mha_baseline.project_qkv` | 对照组：真实代码不再有满维 W^K/W^V 分支 |
| Eq.7-8（标准注意力+输出投影） | `mha_baseline.scaled_dot_product_attention` / `output_projection` | 对照组：MHA cache 基线，真实代码走的是下面 Eq.9 起的压缩路径 |
| Eq.9-11（低秩 KV 联合压缩） | `low_rank_mla.kv_joint_compression` | `exec_kv_decode`/`exec_kv_prefill`（L1312-L1375）：`npu_kv_rmsnorm_rope_cache` 产出 c^{KV} |
| Eq.11 之后的吸收恒等式（q 侧） | `low_rank_mla.precompute_absorbed_query_weights` / `score_absorbed_nope` | `process_weights_after_loading`（L924-L957）算出 `W_UK_T`；`_q_proj_and_k_up_proj`（L910-L922）用它把 q_nope 吸收进潜空间 |
| Eq.11 之后的吸收恒等式（o 侧） | `low_rank_mla.attention_in_latent_space` / `latent_to_value` | `_forward_decode` 对缓存的潜向量直接注意力（L1389-L1593）+ `_v_up_proj`（L900-L907）还原到 value 空间 |
| Eq.12-13（q 侧低秩） | `low_rank_mla.q_joint_compression` | `_mla_preprocess`（L1640-L1691）里 `fused_qkv_a_proj` 拆出的 `q_c` |
| Eq.14-15（解耦 q^R/k^R） | `decoupled_rope.decoupled_query_rope` / `decoupled_key_rope` | `mla_preprocess_decode`（L1620-L1638）里对 `q_pe`/`k_pe` 单独 `rope_single`，`exec_kv_decode` 写 k_pe cache |
| Eq.16-17（拼接 nope+rope） | `decoupled_rope.concat_nope_rope_query` / `concat_nope_rope_key` | 代码里 `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` 的隐式拼接 |
| Eq.18（解耦注意力打分） | `decoupled_rope.decoupled_attention_scores` | `_forward_decode` 调 `npu_fused_infer_attention_score_v2(q_nope, k_nope, k_nope)`（K=V=潜向量的 MQA 式注意力） |
| §2.1.3 文字（不可吸收论证） | `decoupled_rope.rope_on_compressed_key_score` / `effective_middle_matrix` | 反证：真实代码里没有这条路径——正因为它不可吸收，vllm_ascend 才必须走解耦分支 |
| §2.1.4 Table 1（KV cache 对比） | `kv_cache_table.compare_kv_cache` / `deepseek_v2_numbers` | dossier `theory` 引用的 DeepSeek-V2 §3.1.2 超参 |
| Eq.9-19 端到端装配 | `mla_reference.MLAReference.forward_full` / `.decode_step` | `forward`（L1718-L1804）按 decode/prefill 分派 `mla_preprocess_decode`+`_forward_decode` 或 `mla_preprocess_prefill`+`_forward_prefill` |

## 关键设计取舍

1. **行向量约定**：所有权重矩阵按 `nn.Linear` 惯例存成 `(out_dim, in_dim)`，序列按 `h_seq @ W.T`
   批量计算，逐 token 等价于论文的列向量记号 `W @ h_t`——与 `vllm_ascend` 的 `nn.Linear` 风格一致。
2. **RoPE 用显式旋转矩阵而非 rotate-half 技巧**：`decoupled_rope.rope_rotation_matrix` 直接构造
   分块对角矩阵，牺牲一点性能换取"矩阵乘不交换"这条核心论证可以被逐元素验证
   （`effective_middle_matrix`/`verify_relative_position_property`）。小维度下完全够用。
3. **`mla_reference.decode_step` 的缓存只有两个张量**：`c_kv_history (t,d_c)` 与 `k_r_history (t,d_h_r)`——
   刻意不出现任何 `(t, n_h*d_h)` 形状的物化 key/value，逼真复现"decode 期不重算历史 key"这件事。
4. **权重吸收矩阵在 `__init__` 时算一次**（`precompute_absorbed_query_weights`/`precompute_uv_head_slices`），
   `decode_step` 从不重新计算——对应真实代码 `process_weights_after_loading` 只在加载后跑一次。

## 测试

`tests/`（25 例，host `python3 -m pytest`）覆盖：
- MHA 基线的形状/因果性/手算校验（`test_mha_baseline.py`）；
- 低秩压缩的维度收缩 + q 侧/o 侧权重吸收恒等式逐元素校验（`test_low_rank_mla.py`）；
- RoPE 旋转矩阵正交性、相对位置性质、"中间矩阵 M(delta) 随 delta 变化不可吸收"的数值反证、
  解耦管线的形状与因果 softmax（`test_decoupled_rope.py`）；
- **旗舰测试**：decode 逐 token 增量计算与 prefill 一次性计算逐位置完全一致，且吸收矩阵全程不重算
  （`test_mla_reference.py`）；
- Table 1 公式代数关系 + DeepSeek-V2 真实数字（32768、2.25 等效 GQA 组）（`test_kv_cache_table.py`）。
全部通过。
