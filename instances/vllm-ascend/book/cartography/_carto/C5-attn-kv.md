# C5 — 注意力与 KV（NPU 特化）测绘 digest

> 对象：vllm-ascend v0.21.0rc1，`vllm_ascend/attention/`（~12.8k）+ `vllm_ascend/core/`（~3k）+ `kv_offload/` + `simple_kv_offload/`。
> 基座对照：vLLM v0.21.0 `vllm/v1/attention/`、`vllm/v1/core/`。这是 vLLM 书 ch13–16/24 的昇腾对位篇。

## 主线一句话

vllm-ascend 不改 vLLM 引擎主循环，而是**通过 vLLM 的 OOT 平台插件契约把 CUDA 路径整体顶替成 NPU 路径**：① `AscendPlatform.get_attn_backend_cls()`（`platform.py:739`）按 `(use_mla, use_sparse, use_compress)` 三元 key 路由到 4 个昇腾后端类，外加 `@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")` 把自家后端注册进 vLLM 的 backend registry；② 每个后端实现 vLLM 抽象基类（`AttentionBackend`/`AttentionMetadataBuilder`/`AttentionImpl`、MLA 走 `MLACommonMetadataBuilder`/`MLAAttentionImpl`），但 forward 内核全换成 `torch_npu.*`（`npu_fused_infer_attention_score`、`_npu_paged_attention`、`npu_kv_rmsnorm_rope_cache` 等）；③ KV 侧 `single_type_kv_cache_manager` 复用 vLLM `FullAttentionManager`/`BlockPool` 只加一个 `CompressAttentionManager` 处理压缩 spec，scheduler 全是 vLLM `Scheduler`/`AsyncScheduler` 子类。**减法点：CUDA flash_attn/flashinfer 内核、ROCm 路径、GDN/Mamba 后端不在本子系统范围。**

## 关键 vLLM 对照点

- backend 注册/选择：`vllm/v1/attention/backends/registry.py` + `flash_attn.py` ↔ `platform.py:739` 三元 key 路由 + `attention_v1.py:73` `register_backend`。
- KV cache shape 约定：vLLM 各 backend 的 `get_kv_cache_shape` ↔ Ascend `(2, num_blocks, block_size, num_kv_heads, head_size)`（`attention_v1.py:101`）。
- MLA absorb：`vllm/v1/attention/backends/mla/common.py` 的 `MLACommonImpl` ↔ `mla_v1.py:700` `AscendMLAImpl`（`_q_proj_and_k_up_proj`/`npu_kv_rmsnorm_rope_cache`/`npu_transpose_batchmatmul` 权重吸收）。
- KV cache 管理：`vllm/v1/core/single_type_kv_cache_manager.py` `FullAttentionManager`/`block_pool.py` ↔ Ascend 子类 + `CompressAttentionManager`（`single_type_kv_cache_manager.py:28`）+ `get_manager_for_kv_cache_spec` 重定向。
- 调度器：`vllm/v1/core/sched/scheduler.py`/`async_scheduler.py` ↔ `SchedulerDynamicBatch`/`RecomputeScheduler`/`ProfilingChunkScheduler`（均为子类，覆写 `schedule()`）。
- 后端有意 `get_name()` 返回 "FLASH_ATTN" 绕过 vLLM model-runner 的 name 断言（`attention_v1.py:78`，HACK 注释）——OOT 插件如何"伪装"成内置后端的典型样本。

## 建议章节（6 章，中等深度）

### 1. 昇腾如何接进 vLLM 的注意力后端选择
- focus：OOT 插件契约全貌——`get_attn_backend_cls` 三元 key 路由表、`register_backend(CUSTOM,"ASCEND")`、`get_name()` 伪装断言、`get_kv_cache_shape`/`swap_blocks`/`copy_blocks` 静态契约、`get_impl_cls`/`get_builder_cls` 在 `enable_cp()` 下的运行期分流。
- key_source_paths：`vllm_ascend/platform.py:739-774`、`vllm_ascend/attention/attention_v1.py:73-141`、`vllm_ascend/attention/abstract.py`。
- pairs_with：`vllm/v1/attention/backends/registry.py`、`backends/utils.py`、`backends/flash_attn.py`（get_kv_cache_shape）；vLLM 书 ch24《Attention Backends and Metadata》。
- teach_value：高——全书"昇腾顶替 CUDA"主线的总入口，一章讲清插件如何无侵入挂载。
- est_size：中（~14 页）。deps：无（应作为本 Part 开篇）。

### 2. AscendAttentionBackendImpl：标准 MHA 的 NPU 内核与状态机
- focus：`AscendMetadataBuilder.build`（split_decodes_and_prefills、slot_mapping、block_table）、`AscendAttentionState` 五态机、`forward_impl` 按状态分流到 `forward_paged_attention`（`_npu_paged_attention`+workspace 预取）vs `forward_fused_infer_attention`（`npu_fused_infer_attention_score(_v2)`）、`reshape_and_cache` 经 `DeviceOperator`。C8（INT8 KV 量化）子类作为减法候选/选讲。
- key_source_paths：`attention_v1.py:213-345`（builder）、`357-1345`（impl）、`attention_mask.py`、`utils.py`。
- pairs_with：`vllm/v1/attention/backends/flash_attn.py`（FlashAttentionImpl + builder）；vLLM 书 ch24。
- teach_value：高——非 MLA 模型的主路径，最能直观对照 FlashAttention→torch_npu 算子替换。
- est_size：大（~18 页）。deps：ch1。

### 3. MLA 在 NPU 上的 prefill/decode 拆分与权重吸收
- focus：`AscendMLAImpl` 的 absorb（`_q_proj_and_k_up_proj`、`npu_format_cast` 权重格式转换 29/FRACTAL）、`_forward_prefill`（含 `_compute_prefill_context` chunked context + `npu_attention_update` LSE 合并）、`_forward_decode`（MQA 吸收路径 + `npu_kv_rmsnorm_rope_cache` 融合 RoPE）、`AscendMLAMetadataBuilder` 的 chunked/prefill/decode 三段 metadata。
- key_source_paths：`mla_v1.py:77-690`（backend+metadata+builder）、`700-1750`（impl 全 forward）。
- pairs_with：`vllm/v1/attention/backends/mla/common.py`（MLACommonImpl/MLACommonMetadataBuilder）；vLLM 书 ch24（MLA 篇）/ch16。
- teach_value：最高——MLA 是 DeepSeek 类模型核心，昇腾改造最深、torch_npu 融合算子最密集，是本子系统旗舰章。
- est_size：大（~20 页）。deps：ch1,ch2。

### 4. 稀疏注意力：SFA 与 DSA（Lightning Indexer）
- focus：`AscendSFAImpl`（基于 MLA + `use_sparse_c8_indexer`、Hadamard 变换、`execute_sparse_flash_attention_process`）；`AscendDSAImpl`（DeepSeek sparse，`npu_quant_lightning_indexer_metadata`、`index_topk=512`、`sparse_mode=3`、prefill/decode 各自的 indexer metadata 构建）。两者如何复用 MLA 基类再叠稀疏选择。
- key_source_paths：`sfa_v1.py:71-1300`、`dsa_v1.py:180-1300`、`device/device_op.py`（算子分派）。
- pairs_with：vLLM 主干无对位（vLLM 无昇腾 lightning indexer）——讲"昇腾自有扩展"；松挂 vLLM 书 ch24。
- teach_value：中高——展示插件不止"顶替"还能"增量扩展"，但体量大、算子私有，宜聚焦机制不逐行。
- est_size：大（~16 页，需重度减法）。deps：ch3。

### 5. KV cache 管理与协调器的 NPU 特化
- focus：`single_type_kv_cache_manager` 复用 vLLM `FullAttentionManager`/`BlockPool`，新增 `CompressAttentionManager`（处理 `MLAAttentionSpec`/压缩 spec 的 block 分配/前缀命中）、`get_manager_for_kv_cache_spec` 重映射；spec 类型 → manager 映射如何接入 vLLM coordinator。
- key_source_paths：`core/single_type_kv_cache_manager.py` 全文。
- pairs_with：`vllm/v1/core/single_type_kv_cache_manager.py`、`block_pool.py`、`kv_cache_coordinator.py`、`kv_cache_manager.py`；vLLM 书 ch15/ch16。
- teach_value：中——昇腾对 KV 管理改动小，正好讲"哪些能原样复用、哪些必须特化"。
- est_size：中（~12 页）。deps：ch1。

### 6. NPU 调度器特化与 KV offload
- focus：三个 `Scheduler` 子类——`SchedulerDynamicBatch`（BudgetRefiner 按 decode token 数查表调预算）、`Recompute Scheduler`（重算/异步 + remote KV）、`ProfilingChunkScheduler`（profiling 驱动动态 chunk）；KV offload 双实现：`kv_offload/cpu_npu.py` `CpuNpuOffloadingHandler`（torch.npu.Event 异步搬运）+ `simple_kv_offload/`（`NPUDmaCopyBackend` DMA 拷贝线程、`register_kv_caches` 重建 block view）。
- key_source_paths：`core/scheduler_dynamic_batch.py`、`recompute_scheduler.py`、`scheduler_profiling_chunk.py`、`profiling_chunk_predictor.py`；`kv_offload/cpu_npu.py`、`kv_offload/npu.py`、`simple_kv_offload/{worker,copy_backend,npu_mem_ops}.py`。
- pairs_with：`vllm/v1/core/sched/scheduler.py`/`async_scheduler.py`、`vllm/v1/kv_offload/`；vLLM 书 ch13/ch14。
- teach_value：中——调度器/offload 是子类化典范，但偏工程、可整合为一章避免碎片。
- est_size：大（~16 页，需取舍）。deps：ch5。

## 范围/减法提示
- `context_parallel/`（attention_cp/mla_cp/dsa_cp/sfa_cp，~4k）属上下文并行（PCP/DCP），与 C? 分布式/并行子系统可能重叠——本子系统仅在 ch1 提 `enable_cp()` 运行期分流，正文不深入，避免与并行篇打架。
- `fa3_v1.py`/`_310p`/`kvcomp_attn`：边角后端，作减法或选讲。
- `profiling_chunk_predictor.py` 依赖 pandas 查表——选讲。
