# ch19 实现说明 — 标准 MHA 的 NPU 内核与状态机（subtract-only 精简版）

精简版四文件，与真仓同名同结构同控制流，只删不增。host 无 NPU/CANN：纯 Python 控制流可跑
（五态机分流 / 拆批 / build 装配 / forward_impl 选路），真实 torch_npu 算子由测试的记录替身承接、不真算。

## 1:1 Source Map

| 精简版符号 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `attention_v1.AscendAttentionBackend`（get_name/get_impl_cls/get_builder_cls/get_kv_cache_shape/get_supported_kernel_block_sizes） | `vllm_ascend/attention/attention_v1.py:L73-L140` | 删 swap_blocks/copy_blocks | 块搬运是 KV 管理辅助（preemption/CoW），非注意力前向主线（delete#6） |
| `attention_v1.AscendAttentionState`（五态） | `vllm_ascend/attention/attention_v1.py:L143-L148` | 原样 | 全章立意，must_keep |
| `attention_v1.AscendMetadata` | `vllm_ascend/attention/attention_v1.py:L151-L210` | 删 num_actual_tokens_pcp_padded / prefill(pcp) / decode_meta(dcp) / reshape_cache_event | PCP/DCP 字段（delete#5）+ PD KV 传输事件（delete#6） |
| `attention_v1.AscendAttentionMetadataBuilder.build` | `vllm_ascend/attention/attention_v1.py:L272-L332` | 删 CrossAttentionSpec / parallel_drafting 覆盖分支 | encoder-decoder / 并行草稿特例，非标准 MHA 主线（delete#6/#5） |
| `attention_v1.AscendAttentionMetadataBuilder.build_for_graph_capture` | `vllm_ascend/attention/attention_v1.py:L334-L354` | 原样 | 章纲要求讲图捕获分支，must_keep |
| `attention_v1.AscendAttentionBackendImpl.forward` | `vllm_ascend/attention/attention_v1.py:L1279-L1343` | 删 hamming layerIndex / pooling 分支 | hamming 默认关（delete#3）+ 非自回归 pooling（delete#6） |
| `attention_v1.AscendAttentionBackendImpl.forward_impl` | `vllm_ascend/attention/attention_v1.py:L1258-L1277` | 原样 | 状态分流核心，must_keep |
| `attention_v1.AscendAttentionBackendImpl.reshape_and_cache` | `vllm_ascend/attention/attention_v1.py:L1229-L1256` | 删 is_kv_producer 的 reshape_cache_event | disaggregated PD 的 KV 传输同步（delete#6） |
| `attention_v1.AscendAttentionBackendImpl._get_fia_params` | `vllm_ascend/attention/attention_v1.py:L985-L1043` | 删 key_cache 懒初始化的 isinstance 形状判断 | 与 forward 顶部同套路，主线直接取 [0]/[1] |
| `attention_v1.AscendAttentionBackendImpl.forward_fused_infer_attention` | `vllm_ascend/attention/attention_v1.py:L1045-L1164` | 删 capturing→full_graph_fia / hamming kvcomp / self.sinks 分支 | 图捕获（delete#1）+ hamming（delete#3）+ attention-sink（delete#4），保留非 sinks 三选一 sparse_mode 0/4/3 |
| `attention_v1.AscendAttentionBackendImpl.forward_paged_attention` | `vllm_ascend/attention/attention_v1.py:L1166-L1185` | 原样（保留 capturing→full_graph_pa） | decode/paged 算子路径，must_keep |
| `attention_v1.AscendAttentionBackendImpl.full_graph_pa` | `vllm_ascend/attention/attention_v1.py:L922-L945` | 删 ExternalEvent/graph_task_group 录制（L947-L983） | 只取 workspace 预取节拍范例，图捕获机制不展开（delete#1） |
| `utils.split_decodes_and_prefills` | `vllm_ascend/attention/utils.py:L273-L315` | 删 PCP 分支（query_lens_pcp_full） | 仅 prefill_context_parallel 时非 None，CP 回指 ch08（delete#5） |
| `utils.using_paged_attention` | `vllm_ascend/attention/utils.py:L44-L55` | 原样 | paged 路径多门槛判据，must_keep |
| `utils.enable_cp` | `vllm_ascend/attention/utils.py:L58-L61` | 原样 | f7 回收开关，must_keep |
| `utils.AscendCommonAttentionMetadata` | `vllm_ascend/attention/utils.py:L147-L191` | 删父类 CommonAttentionMetadata 与 ~30 旁路字段 / unpadded() | 仅保留 build/split 实际读到的字段作可跑输入容器 |
| `device_op.BaseDeviceAdaptor.reshape_and_cache` / `DeviceOperator` | `vllm_ascend/device/device_op.py:L42-L47, L1663-L1670` | 删 A5DeviceAdaptor 与数十个非注意力设备算子 | A5 特化非主线、MoE/通信算子非本章 |
| `attention_mask.AttentionMaskBuilder`（get_attention_mask/get_splitfuse_attn_mask/get_attn_mask） | `vllm_ascend/attention/attention_mask.py:L34-L79` | 删 MLA 专用 mask（get_mla_mask/get_pcp_mla_mask/get_final_mla_mask） | MLA 是 ch20 内容，非本章 MHA 主线 |
| `AscendC8AttentionBackendImpl` | `vllm_ascend/attention/attention_v1.py:L1346-L1783` | 整子类删除，仅 SUBTRACTED 注释点名 | INT8 KV 量化减法候选/选讲，不走标准 MHA 主路径（delete#2） |

## must_keep 核对
linter `lint_fidelity` 校验全部 must_keep 符号在场：五态成员、AscendMetadata、get_impl/builder_cls、
get_kv_cache_shape、enable_cp、build/build_for_graph_capture、split_decodes_and_prefills、slot_mapping、
block_tables、forward/forward_impl/reshape_and_cache、forward_paged/fused、_get_fia_params、
using_paged_attention、DeviceOperator、AscendC8AttentionBackendImpl —— 均保留。

## 验证
- `tests/`：25 项纯 Python 控制流测试全过（host `python3 -m pytest`）。
- `lint_fidelity`：无 BLOCKING。
