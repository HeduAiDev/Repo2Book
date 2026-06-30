# ch20 implementation notes —— MLA 在 NPU 上（subtract-only）

精简版 = `implementation/mla_v1.py`，与真仓 `vllm_ascend/attention/mla_v1.py`（~1804 行）同名同结构同控制流，
**只做减法**。本章实际引用/对照的规范源码落点共三处：
- `vllm_ascend/attention/mla_v1.py`（主角：AscendMLAImpl / AscendMLAMetadataBuilder 全部主线）
- `vllm/model_executor/layers/attention/mla_attention.py`（对照基座：MLACommonMetadataBuilder / MLACommonImpl 基类——昇腾把通用 MLA 的 prefill/decode 拆分与权重吸收换成 torch_npu 融合算子）
- `vllm_ascend/utils.py`（ACL_FORMAT_FRACTAL_ND / FRACTAL NZ=29 等格式常量来源，absorb 的 npu_format_cast 用它）
host 无 CANN/torch_npu：纯 Python·形状级控制流可在 host 复现（absorb 的 split/permute/bmm 形状代数 /
三段 metadata 装配 / chunked-context 的 LSE 合并循环 / forward 按 decode-prefill 派发）；真实
`npu_kv_rmsnorm_rope_cache` / `npu_fused_infer_attention_score_v2` / `npu_format_cast` 等融合算子不真跑（昇腾才有内核），
测试由「记录调用」的 torch_npu 替身承接（`tests/conftest.py`）。

## 1:1 Source Map（精简版 ↔ 真源码 ↔ 改动 ↔ 原因）

| 精简版符号 | vllm_ascend/attention/mla_v1.py | 改动 | 原因 |
|---|---|---|---|
| `AscendMLABackend.get_impl_cls/get_builder_cls` | L87-L111 | 保留 enable_cp() f-收口 + 主线 return | ch18 路由落点；CP 分支延迟 import（旁支回指 ch08） |
| `process_weights_after_loading` | L924-L992 | 删 enable_mlapo/fa_quant 量化融合 + else 的 copy_ 回写 + layer_sharding post_process | 只留 bf16·非量化主线：kv_b_proj→FRACTAL_ND(2).T→split W_UK/W_UV→W_UK_T=permute(1,2,0)→maybe_trans_nz(NZ=29) |
| `_q_proj_and_k_up_proj` | L909-L922 | 逐字保留 | 运行期吸收核心：`torch.bmm(q_nope, W_UK_T)` 把 q 投到 latent |
| `_v_up_proj` | L900-L907 | 逐字保留 | latent 输出经 W_UV 投回 V（`npu_transpose_batchmatmul`） |
| `AscendMLAMetadataBuilder.build` | L427-L487 | 逐字保留双路装配骨架 | `split_decodes_and_prefills` 切 decode/prefill → build_prefill/decode_metadata |
| `build_chunked_metadata` | L489-L531 | 逐字保留 | `max_context_chunk=workspace//prefills→round_down(block_size)`，`num_chunks=cdiv(...)` |
| `build_prefill_metadata` | L545-L580 | 逐字保留 | prefill 段从 decode 段之后切；`actual_seq_lengths_q=cumsum(query_lens)`（TND 右边界） |
| `build_decode_metadata` | L582-L662 | 删 graph_pad_size>num_reqs 的 MTP/fullgraph batch padding 整段 + cp_seq_len | 非投机非 fullgraph 主路不触发该分支 |
| `_compute_prefill_context` | L1136-L1241 | 删 `_reorg_kvcache`(CP 恒等)/fa_quant 反量化/head_padding cat | 逐 chunk 读 cache→`kv_b_proj` 解压→FIA→`npu_attention_update` 在线合并 LSE |
| `_forward_prefill` | L1243-L1310 | 删 dtype!=bf16 转换 + head_padding cat | `npu_fused_infer_attention_score`(TND) 算新 token + `_compute_prefill_context` 合历史 |
| `exec_kv_decode` / `exec_kv_prefill` | L1312-L1375 | 删 A5+fa_quant 的 c_kv_scale | `npu_kv_rmsnorm_rope_cache` 一把 RMSNorm+RoPE+写cache；prefill `is_output_kv=True` 取后两输出 |
| `_forward_decode` | L1389-L1593 | 删 fa_quant/enable_kv_nz/SpecDecoding 多 layout + capturing 图捕获 | 主线 BNSD_NBSD + `npu_fused_infer_attention_score_v2(q,k_nope,k_nope)`（K=V=隐向量 MQA）→ `_v_up_proj` |
| `mla_preprocess_decode` | L1620-L1638 | 删 fa_quant+A5 npu_dynamic_quant | 吸收路：`_q_proj_and_k_up_proj`→`rope_single`→`exec_kv_decode` |
| `mla_preprocess_prefill` | L1598-L1618 | 逐字保留 | MHA 路：q_proj 满维 q + `exec_kv_prefill` + `kv_b_proj` 显式解压 k_nope/value |
| `_mla_preprocess` | L1640-L1691 | 删 weight_prefetch/Flash-Comm allgather/connector/layer_sharding/reshape_cache_event | `fused_qkv_a_proj` 拆 q_c/kv_no_split → has_decode/has_prefill 双路 |
| `forward` | L1718-L1804 | 删 mla_preprocess_only_decode 快路 + o_proj 预取 + connector save | 真实分流处：写 o_proj_input 切片 → o_proj(is_prefill=...) |

## 被删模块（subtraction_plan.delete 批准项）
- 量化融合权重：`_process_weights_for_fused_mlapo[_a5]` / `_process_weights_for_fused_fa_quant`（L994-L1115）。
- Context-Parallel：enable_cp 分支体内的 CP impl/builder、`_reorg_kvcache`、cp_seq_len/dcp_mtp_attn_mask。
- 图捕获/replay：`update_graph_params`（L785-L898）、`_forward_decode` 的 `_EXTRA_CTX.capturing` 录制段、build_for_graph_capture 内不变。
- MTP/spec-decode padding：`reorder_batch`、`pad_actual_seq_len_q_mtp_*`、build_decode 的 graph_pad_size 整段。
- 分布式/连接器/预取：weight_prefetch、Flash-Comm allgather、layer_sharding、connector 钩子、mla_preprocess_only_decode。
- `forward_mha`/`forward_mqa`（仅 raise NotImplementedError；真实分流在 forward 内——纠偏 2）。
- `_forward_prefill` 的 bf16 兼容转换。

## 关键纠偏（来自 dossier source_reading_notes）
1. **format 不是 29**：`kv_b_proj.weight` cast 用 `ACL_FORMAT_FRACTAL_ND`（=2，ND 排布，便于 .T/view/split）；
   只有 `W_UK_T` 末尾经 `maybe_trans_nz` 转 `FRACTAL_NZ`（=29，喂昇腾 cube）。两者不能混淆。
2. **forward 不调 forward_mqa/forward_mha**：本版二者仅 `raise NotImplementedError`；真正的 decode/prefill 分流在
   `forward → _mla_preprocess` 内（has_decode→吸收路 + `_forward_decode`，has_prefill→MHA 路 + `_forward_prefill`）。

## 测试
`tests/test_mla_on_npu.py`（23 例，host `python3 -m pytest`）覆盖：后端契约/f-收口、权重吸收形状代数、
三段 metadata 装配、build_chunked_metadata 分块数学、_mla_preprocess/forward 派发、exec_kv 的 is_output_kv 差异、
_compute_prefill_context 的 LSE 合并循环。全部通过。
