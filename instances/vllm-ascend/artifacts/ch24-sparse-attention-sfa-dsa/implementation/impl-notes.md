# ch21 实现笔记 —— 稀疏注意力 SFA / DSA（subtract-only 精简版）

本章是**增量扩展**型：vLLM 主干无对位后端（pairs vllm_paths 为空），不做基座算子级对照。
精简版只验「可读控制流·形状级」：SFA/DSA 如何复用 MLA 再叠稀疏选择、Lightning Indexer 选
top-512/2048 的元数据装配与前向选择、DeviceOperator 门面按设备代际派发。host 无 NPU/CANN，
真实私有算子（npu_quant_lightning_indexer_metadata / npu_quant_lightning_indexer /
npu_lightning_indexer / npu_sparse_flash_attention / npu_sparse_attn_sharedkv / compressor）
由测试「记录调用」替身承接，**不真跑**（昇腾才有内核）。

体量极大（sfa_v1.py 1376 行 + dsa_v1.py 2897 行 + device_op.py 1670 行），按 dossier
`subtraction_plan` **重度减法**：聚焦机制、不逐行。

## 文件
- `abstract.py` — DSAAttentionImpl 抽象基类（DSA 自起一套的根，几乎逐字）。
- `device_op.py` — DeviceOperator 门面（Base/A5 多态 + 稀疏注意力链路触达的方法）。
- `sfa_v1.py` — SFA：建在 vLLM MLA 之上 + 两段式稀疏选择。
- `dsa_v1.py` — DSA：自起一套抽象 + Lightning Indexer（top-512）。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `device_op.BaseDeviceAdaptor.indexer_select_post_process` | `vllm_ascend/device/device_op.py:L371` | 删 use_sparse_c8_indexer / use_torch_npu_lightning_indexer 两旁支，保 default | C8 量化索引/代际分叉与「选 top-k」语义无关；三条都 sparse_count=2048,sparse_mode=3 |
| `device_op.BaseDeviceAdaptor.execute_sparse_flash_attention_process` | `vllm_ascend/device/device_op.py:L437` | 逐字保留 | SFA「只对 top-k 算」的落点（sparse_indices=topk_indices） |
| `device_op.get_device_adaptor` / `DeviceOperator` | `device_op.py:L1663,L1670` | 逐字保留 | 门面按 AscendDeviceType 多态选 Base/A5（立意(4)） |
| `device_op.A5DeviceAdaptor.reshape_and_cache` | `device_op.py:L787` | 保此 override，删其余几十个 | 以一个 override 示意 A5 代际分叉 |
| `sfa_v1.AscendSFAMetadataBuilder` | `vllm_ascend/attention/sfa_v1.py:L170` | `__init__` 删 dsa_cp/c8/spec 分支与 build() 本体，保 super() 委托 | 元数据装配复用 ch20 MLA 基类；只需可见「继承 MLACommonMetadataBuilder」 |
| `sfa_v1.AscendSFAImpl.__init__` | `sfa_v1.py:L403` | 删 enable_mlapo/enable_dsa_cp*/use_sparse_c8_indexer/o_proj_tp 初始化，保 indexer 必备断言 | 服务已减的 MLAPO/CP/C8；`assert indexer is not None` 是稀疏选择必备件 |
| `sfa_v1.AscendSFAImpl.indexer_select_pre/post_process` | `sfa_v1.py:L961,L1002` | 删 HAS_TRITON / MLAPO-C8 / hadamard 量化旁支，保 npu_rotary_mul + wq_b 主路 | 阶段一 query/key 侧投影 + RoPE → 门面 npu_lightning_indexer |
| `sfa_v1.AscendSFAImpl.forward` | `sfa_v1.py:L1107` | 删 MLAPO/CP/SP/c8 cache 写/IndexCache 分支，保 native 主脊 | 主脊：MLA 低秩 prolog → 选 top-k(2048) → 稀疏 flash → _v_up_proj → o_proj |
| `dsa_v1.AscendDSAMetadataBuilder.build_prefill_metadata` | `vllm_ascend/attention/dsa_v1.py:L604` | 删 ratio 缓存字典 ping-pong / 压缩位置闭包 / compress_ratio<=1·128 sas 旁支，保 c4 sas + qli | qli_metadata 是 sparse_count=index_topk=512,sparse_mode=3 的承载点（章节核心） |
| `dsa_v1.AscendDSAMetadataBuilder.build_decode_metadata` | `dsa_v1.py:L862` | 同上，保 qli | decode 路 indexer 元数据与 prefill 对称 |
| `dsa_v1.AscendDSAImpl._forward_prefill` / `_forward_decode` | `dsa_v1.py:L1866,L2186` | 删 multistream/W8A8/compress_ratio<=1·128 旁支，保 c4 主稀疏路 | 内联 MLA 式低秩 prolog → indexer_select_qli 选 top-512 → npu_sparse_attn_sharedkv(cmp_sparse_indices) |
| `dsa_v1.AscendDSAImpl.indexer_select_qli` → `_indexer_qkv_prepare`/`_indexer_qli_finish`/`_indexer_qli` | `dsa_v1.py:L2706,L2509,L2610,L2660` | 删 W8A8 量化分支，保 native 串行链 | Lightning Indexer 真身：压缩 KV + 量化写 indexer cache → npu_quant_lightning_indexer(sparse_count=index_topk=512) |
| `dsa_v1.AscendDSAImpl.forward` | `dsa_v1.py:L1574` | 删 need_prefill_gather/A5 o_proj/olora_tp，保拆 [decode|prefill] + wo_a/wo_b 主路 | 主脊：hidden_states 拆两段各走稀疏路，合并过 o_proj |
| `dsa_v1.hadamard_transform_ref` / `rotate_activation` | `dsa_v1.py:L61,L78` | 逐字保留 | indexer 的哈达玛旋转参考实现（被 _indexer_qkv_prepare 调用） |

## must_keep 覆盖
dossier `subtraction_plan.must_keep` 全部 22 个符号均保留（`lint_fidelity` 通过）：
AscendSFABackend / AscendSFAMetadataBuilder / AscendSFAImpl / indexer_select_pre_process /
indexer_select_post_process / _execute_sparse_flash_attention_process / AscendDSABackend /
AscendDSAMetadataBuilder / build_prefill_metadata / build_decode_metadata /
npu_quant_lightning_indexer_metadata / AscendDSAImpl / indexer_select_qli / _indexer_qli /
index_topk / _forward_prefill / _forward_decode / DeviceOperator / get_dsa_sparse_attn_op /
execute_sparse_flash_attention_process / reshape_and_cache / get_device_adaptor。

## 验证
- `python3 -m pytest tests/ -q` → 12 passed（后端契约/继承、门面多态、SFA 两段式 2048、
  DSA Lightning Indexer 元数据与选择 512）。
- `python3 scripts/lint_fidelity.py <chapter_dir>` → 无 BLOCKING。
