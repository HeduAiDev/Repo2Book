# ch25《MLA 的两种展开》impl-notes —— 只做减法精简版

对应真实源码 pin **vLLM v0.27.1（6e448d0ea）**，行号全部现核（2026-09-12，
`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。

运行：`cd instances/vllm/artifacts-v3/ch25-mla-two-expansions && python -m pytest tests/ -q`
→ **38 passed**（~4s）。纯 host 单元测试：真 torch/numpy（真 torch.library 注册
的 unified_mla_* 双算子、真 bmm 吸收、真 LSE merge 恒等式数值验证）；无 vllm
包安装、无 CUDA 上下文——CUDA kernel 面（FlashMLA / merge_attn_states /
concat_and_cache_mla / gather_cache）以 HOST SEAM 镜像承载**精确数学**（文件头
伪码的逐式实现，等价性测试直接数值对照）；后端选择按 ch21/ch23 域惯例走真实
注入位（`MLAAttention.__init__` 的 `attn_backend` 参数 + `register_backend(
CUSTOM)` 覆盖表 + prefill 家族 `MLAPrefillBackendEnum.CUSTOM` 注册槽）以参考
后端注入。

**验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本包（HOST SEAM
例外见 §Seam 清单——每个 seam 行内标注真实源锚）。SUBTRACTED 标记逐条挂
dossier.subtraction_plan.delete[0..8] 批准项编号（章界外域段另以「→ chN」注记
——ch18 起的切面惯例）。

**lint**：`python scripts/lint_fidelity.py <本章目录>` → **全部通过（无
BLOCKING）**；must_keep 44 符号经 linter `over_subtraction` 项全数核在。

**锚点双置惯例**（ch22/ch23 同款）：每个 def/class 的 `# SOURCE:` 锚点既在
声明上方（供读者），也在 def/class 体内复置一行——后者是
`lint_fidelity._spans_missing_source` 的跨度判据。两处锚点同源（体内行带
「锚点双置」尾注）。

## 本章四幕 ↔ 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `vllm/model_executor/layers/mla.py` | 同名 | **主文件 1**（站 3/9）：MLAModules（L14-L31 逐字）+ MultiHeadLatentAttentionWrapper（@PluggableLayer.register 位 + __init__ L55-L148 / forward L150-L226 减法：delete[0] dcp_q_replicate 三处、delete[3] indexer 调用位与 topk 布线、elide llama_4_scaling） |
| `vllm/model_executor/layers/attention/mla_attention.py` | 同名 | **主文件 2**（站 3/4/5/8/10/11/12/13）：文件头数学文档 **L3-L188 全文保留**（两种展开的伪码真相源）+ MLAAttention（__init__ 后端断言/自注册/双后端装配 + forward 直调/算子双路径 + forward_impl 混批切刀 + process_weights_after_loading 吸收重排 + get_kv_cache_spec + _v_up_proj）+ unified_mla_* 双算子（真注册）+ QueryLenSupport + MLACommonBackend + MLACommon{Prefill,Decode,}Metadata + MLADims/get_mla_dims + build_mla_chunked_context_metadata + MLACommonMetadataBuilder（workspace 定容/build 分流）+ MLACommonBaseImpl（_concat_k_nope_k_pe/_compute_prefill_context/forward_mha）+ MLACommonImpl（forward_mqa 抽象） |
| `vllm/model_executor/models/deepseek_v2.py` | 同名 | **主文件 3**（站 1/2）：DeepSeekV2FusedQkvAProjLinear（L912-L956 减法：min-latency GEMM 位删）+ DeepseekV2MLAAttention（L959-L1194 减法：投影装配双支逐字 + qrep/indexer 删）+ DeepseekV2DecoderLayer（选型三岔 L1226-L1238 逐字 + self_attn 装配；MoE/MLP/forward → ch28）+ Deepseek(Attention/V2Attention) 名字位标记类（章界外 body 收窄） |
| `vllm/v1/attention/backends/utils.py` | 同名 | **主文件 5**（站 7/8）：split_decodes_and_prefills（L564-L635 逐字）+ reorder_batch_to_split_decodes_and_prefills（L665-L742 逐字——互斥四区+换位）+ get_num_attention_heads_from_layers（L169-L197 逐字） |
| `vllm/v1/kv_cache_interface.py` | 同名 | **主文件 6**（站 5/m09/m12/m13）：KVQuantMode/get_kv_quant_mode + KVCacheSpec/AttentionSpec/FullAttentionSpec（merge 全链）+ _apply_alignment_padding + MLAAttentionSpec（L388-L468 逐字——storage÷compress、584B/656B 特账、merge 四字段断言）+ SlidingWindowSpec/SlidingWindowMLASpec + UniformTypeKVCacheSpecs（DSV4 layer-tuple 账）+ KVCacheGroupSpec + spec 注册表 base-spec 映射（HOST SEAM 子集，对照 single_type_kv_cache_manager.py:L1881-L1938） |
| `vllm/v1/core/kv_cache_utils.py` | 同名 | **主文件 7**（站 6/m11）：get_kv_cache_groups（L1781-L1852 逐字——四级分流）及其直接依赖族（create_kv_cache_group_specs/is_kv_cache_spec_uniform/uniform_spec/uniform_type/页归一族/_promote_local/fallback/unify_hybrid/group_and_unify/_approximate_gcd/uniform_groups——DSV4 案全链逐字） |
| `vllm/v1/worker/gpu_model_runner.py` | 同名 | **主文件 8**（站 6/7）：三薄层——_may_reorder_batch（L1115-L1138 逐字）/ calculate_reorder_batch_threshold（L7220-L7238 逐字）/ get_kv_cache_spec（L7800-L7837 减法）挂在 GPUModelRunnerSlice 承载类上 |
| `vllm/v1/attention/backends/mla/flashmla.py` | 同名 | **主文件 4**（站 7/13）：FlashMLABackend + FlashMLAMetadataBuilder（**reorder_batch_threshold=128** 'process small prefills with decode pathway' 逐字）+ FlashMLAImpl.forward_mqa（L266-L346 减法：**flash_mla_with_kvcache 调用面逐字**；fp8 双 kernel 分派按 delete[6] 删、spec-decode reshape 按 delete[4] 删为等价 view、VLLM_BATCH_INVARIANT 手工 tile_scheduler 按 delete[5] 删） |
| `vllm/v1/attention/backends/mla/sparse_swa.py` | 同名 | m15：DeepseekV4SWACache（L56-L107 逐字——SWA 子缓存以独立 prefix 注册 + 自报 SlidingWindowMLASpec）；DeepseekSparseSWABackend 家族 → ch26 |
| `vllm/v1/attention/backends/mla/prefill/{base,registry,selector}.py` | 同名 | m10：MLAPrefillBackend ABC（L37-L183 逐字减 supports_quant_output）+ MLAPrefillBackendEnum（五成员+CUSTOM 逐字）+ get_mla_prefill_backend（显式支+优先级表逐字）——第二套 prefill 家族 |
| `vllm/models/deepseek_v4/attention.py` | 同名 | **主文件 9**（m12/m13/m14/m15）：_resolve_dsv4_kv_cache_dtype（逐字）+ DeepseekV4Attention ABC（dims/逐层 compress_ratio L207-L213 逐字/fused_wqa_wkv/wo_a(bmm)/wo_b/SWA 子缓存/compressor 装配/get_kv_cache_spec L655-L674 逐字）+ DeepseekV4IndexerCache（L677-L715 逐字）；forward 的融合 CUDA 链 → ch26/ch28 |
| `vllm/models/deepseek_v4/compressor.py` | 同名 | m12：CompressorStateCache（block_size 布局注释+spec 逐字）+ DeepseekCompressor 装配面（compress→norm→RoPE→store 融合核 → ch26） |
| `vllm/v1/attention/ops/merge_attn_states.py` | 同名 | m08：LSE 精确合并调度器全文逐字（CUDA/Triton 双支真实分派；host 走 Triton 支 seam） |
| `vllm/v1/attention/ops/flashmla.py` | 同名 | 可用性探测逐字 + 接口门面（bf16 kernel；fp8 dense 元数据族按 delete[6] 删） |
| `vllm/model_executor/layers/attention/attention.py` | 同名 | MLA 共用面：should_load_quant_weights/set_default_quant_scales/_init_kv_cache_quant（减法）+ get_attention_context（L732-L772 逐字）；Attention 类本体 → ch21/ch23 |
| `vllm/model_executor/layers/attention_layer_base.py` | 同名 | 全文逐字 |
| `vllm/v1/attention/backend.py` | 同名 | 协议层消费面：AttentionType/MultipleOf/AttentionCGSupport + AttentionBackend（四身份证+shape/head_size/is_mla+stride_order 探针）+ CommonAttentionMetadata（消费字段）+ AttentionMetadataBuilder（阈值声明位+_init 非 spec 分支）+ AttentionLayer + AttentionImplBase/MLAAttentionImpl（forward_mha/forward_mqa 双抽象 + **do_kv_cache_update 写腿逐字**） |
| `vllm/v1/attention/backends/registry.py` | 同名 | decode 家族注册表子集（本章引用成员+CUSTOM+register_backend 逐字——测试经真实第三方注册流注入参考后端） |
| `vllm/v1/attention/selector.py` | 同名 | get_attn_backend 显式后端支（backend=None 的自动优先级面 → ch21——host 平台无可用 MLA 后端与真实 CPU 平台同型失败面） |
| `vllm/model_executor/{custom_op,utils}.py`、`layers/{layernorm,linear,rotary_embedding/}`、`models/utils.py`、`quantization/utils/quant_utils.py` | 同名 | 装配积木消费面：PluggableLayer/replace_parameter（逐字）/RMSNorm（forward_native 数学逐字）/TP 线性族（HOST SEAM TP=1 退化，[out,in] 朝向断言于测试）/get_rope 工厂+GPT-J 旋转数学（ch23 同款切面）/extract_layer_index/get_and_maybe_dequant_weights |
| `vllm/{config,platforms,distributed,forward_context,envs,logger,compilation,utils/_seams}.py` 等 | 各自真实路径 | HOST SEAM 载体（见 §Seam 清单） |

## 1:1 Source Map（核心行；改动=减法或 seam，原因=批准条/章节边界）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `DeepseekV2DecoderLayer` 选型三岔 | deepseek_v2.py:L1226-L1238 | **逐字** | must_keep（use_mla 分流）；站 1 |
| `DeepseekV2MLAAttention.__init__` 投影装配 | deepseek_v2.py:L1006-L1066 | 逐字（qrep L1026-L1033 按 delete[0] 删、indexer 段 L1091-L1141 按 delete[3] 删） | must_keep（fused_qkv_a_proj/kv_a_proj_with_mqa/q_a_layernorm/q_b_proj/kv_a_layernorm/kv_b_proj 全在此） |
| `MultiHeadLatentAttentionWrapper.forward` | mla.py:L150-L226 | 逐字减 delete[0]/[3] 三处 + llama_4_scaling elide | must_keep；站 9 外层低秩链+解耦 RoPE |
| `MLAAttention.__init__` 后端断言+自注册 | mla_attention.py:L423-L436、L512-L518 | 逐字 | must_keep（is_mla 身份证/static_forward_context） |
| `MLAAttention.forward_impl` 分流决策 | mla_attention.py:L766-L772 | 逐字（must_keep：num_mqa_tokens=num_decode_tokens / num_mha_tokens=其余——token 维切刀） | 站 11；等价性测试直接验证两腿 |
| MQA 吸收腿全程 | mla_attention.py:L831-L949 | 减法（delete[0] DCP 段/delete[1] aiter/delete[2] fp8 拼接；**torch.bmm 吸收 L888 与 _v_up_proj L949 逐字**） | must_keep（forward_mqa/W_UK_T/W_UV） |
| `process_weights_after_loading` | mla_attention.py:L994-L1100 | 减法（delete[1] L1037-L1091）；拆分断言 L1017-L1035 与 replace_parameter L1092-L1096 逐字 | must_keep；站 4 权重吸收重排 |
| `forward_mha` | mla_attention.py:L2581-L2666 | 减法（delete[0] DCP 支）；上投影+merge 主干逐字 | must_keep；站 12 |
| `_compute_prefill_context` | mla_attention.py:L2301-L2422 | 逐字（gather 三路分派保留调用面；host bf16 走 gather seam） | must_keep；m08 分块循环现场 |
| `determine_chunked_prefill_workspace_size` | mla_attention.py:L1802-L1831 | **逐字**（144MB vs 3GB 注释原文） | must_keep |
| `MLACommonMetadataBuilder.build` | mla_attention.py:L2005-L2159 | 减法（delete[8] non-causal 段 L2028-L2054；split 调用 L2056-L2063 逐字） | must_keep；站 8 |
| `split_decodes_and_prefills` | backends/utils.py:L564-L635 | **逐字** | must_keep |
| `reorder_batch_to_split_decodes_and_prefills` | backends/utils.py:L665-L742 | **逐字** | must_keep；站 7 四区重排 |
| `MLAAttention.get_kv_cache_spec` | mla_attention.py:L1140-L1152 | **逐字** | must_keep；站 5 |
| `MLAAttentionSpec` | kv_cache_interface.py:L388-L468 | **逐字** | must_keep（584B/656B 特账+merge 四字段断言） |
| `get_kv_cache_groups` | kv_cache_utils.py:L1781-L1852 | **逐字**（四级分流） | must_keep；站 6 |
| `_may_reorder_batch` / `calculate_reorder_batch_threshold` | gpu_model_runner.py:L1115-L1138 / L7220-L7238 | **逐字** | 站 7（threshold 取各组最小） |
| `FlashMLAMetadataBuilder.reorder_batch_threshold=128` | flashmla.py:L118-L122 | **逐字**（含 'process small prefills with decode pathway' 注释） | must_keep；混批测试以 >128 chunk 走 MHA 腿实证该阈值 |
| `FlashMLAImpl.forward_mqa` 的 `flash_mla_with_kvcache` 调用 | flashmla.py:L332-L342 | **逐字** | must_keep（MQA kernel 真实调用面） |
| `DeepseekV4Attention` compress_ratio 逐层解析 | deepseek_v4/attention.py:L207-L213 | **逐字** | must_keep（MTP 恒 1 护栏）；m12 |
| `DeepseekV4Attention.get_kv_cache_spec` | deepseek_v4/attention.py:L655-L674 | **逐字**（compress_ratio≤1 → None） | must_keep |
| `DeepseekV4SWACache.get_kv_cache_spec` | sparse_swa.py:L87-L102 | **逐字** | must_keep；一层多子缓存实证 |
| `merge_attn_states` | ops/merge_attn_states.py:L9-L111 | **逐字** | must_keep；LSE 恒等式数值验证 |
| `MLAPrefillBackendEnum` | prefill/registry.py:L34-L96 | **逐字** | must_keep；第二套 prefill 家族 |

## §Seam 清单（真实代码之外唯一允许的承载面；每处行内另有源锚）

- **B1 CUDA kernel 镜像（精确数学）**：`vllm/third_party/flashmla/flash_mla_interface.py`（FlashMLA MQA kernel 的参考实现——文件头 Data-Movement 伪码逐式；tile_scheduler 元数据为契约容器）、`vllm/v1/attention/ops/triton_merge_attn_states.py`（LSE 重标定公式 + mask_empty_context）、`vllm/_custom_ops.py`（concat_and_cache_mla 的 cat+散写、gather_and_maybe_dequant_cache 的分页 gather；fp8 两个 gather 变体如实抛错——bf16/auto 路径不触达）。等价性测试（MQA 腿 == 上投影 MHA、chunked+merge == 全量、混批两腿）对这些 seam 做数值闭环。
- **B2 平台面**：`platforms/__init__.py`（is_cuda=False→FlashMLA 探测恒不可用、opaque_attention_op=False→直调路径、get_device_capability→None/测试注入 SM90、get_attn_backend_cls 只实现显式后端支——自动优先级归 ch21）+ `platforms/interface.py`（DeviceCapability 容器）+ `utils/platform_utils.py`（num_compute_units 名义位）。
- **C 配置面**：`config/__init__.py`（六 namespace 的消费字段子集；**use_mla 属性为真实 L1791-L1792 逐字**；get_layers_from_vllm_config L2454-L2475 逐字）+ `config/cache.py`（CacheDType Literal 面）。
- **D 装配积木**：`model_executor/layers/linear.py`（TP=1 退化：[out,in] 朝向 + F.linear forward；分片 weight_loader 算术归 ch23）+ `rotary_embedding/`（cos_sin_cache 构造与 GPT-J 旋转数学逐字；kernel 派发位删——ch23 同款）+ `quantization/utils/quant_utils.py`（get_and_maybe_dequant_weights 非量化支）。
- **E 顶层设施**：`envs.py`（真实默认值）、`logger.py`（no-op）、`distributed/__init__.py`（get_dcp_group 同型抛 AssertionError——builder 的 "DCP might not be initialized in testing" 捕获路径）、`forward_context.py`（ch23 同款骨架）、`compilation/breakable_cudagraph.py`（直通装饰器）、`utils/{math_utils,import_utils,torch_utils,flashinfer}.py`（cdiv/round_up/round_down、resolve_obj_by_qualname、is_quantized_kv_cache+get_dtype_size+direct_register_custom_op 真注册、has_flashinfer 恒 False）。
- **F spec 注册表**：`kv_cache_interface.py` 尾部 `_register_spec_base` 族——`KVCacheSpecRegistry` 的 uniform_type_base_spec MRO 走查语义逐字，manager 注册面归 ch14。

## SUBTRACTED 台账（挂 dossier.subtraction_plan.delete 编号）

- **delete[0] DCP/PCP**：mla.py dcp_q_replicate 三处；mla_attention.py 的 PCP+DCP raise（L749-752）、dcp_world_size 惰性初始化（L743-744，改 __init__ 直定 1）、W_UK_T_dcp_qrep、all-gather/LSE 归并两段（L902-914、L922-946）、_context_parallel_compute_prefill_context（L2424-2579）、reorg_kvcache（L2162-2232）、ChunkedContextMetadata 的 DCP 附字段、builder 的 DCP 扩容与 decode 换表段、maybe_gather_mla_latent_cache_inputs/finalize_mla_pcp_decode/cp_lse_*/dcp_a2a 导入与调用、_annotate_eagle_groups_deepseek_v4 的 spec 面。
- **delete[1] ROCm aiter**：forward_impl 的 fp4/fp8 bmm 两支（L854-873）、process_weights 的 mxfp4/fp8 量化+1024 档预编译循环（L1037-1091）、_v_up_proj 两支（L1158-1173）、is_aiter 两旗标、rocm_aiter_ops 导入、dynamic_per_batched_tensor_quant。
- **delete[2] 融合输出量化**：_detect_output_quant_key（L295-328）、quant_key 探测与临时缓冲交换（L705-721）、mha_use_quant_output 门（L797-810）、输出打包尾段（L951-988）、_DecodeConcatQuantFP8/QuantFP8 构造与类、supports_quant_query_input 及其分支、backend_supports_prefill_query_quantization、MLAPrefillBackend.supports_quant_output、forward_impl 签名的 quant 参数族。
- **delete[3] 稀疏 MLA**：_canonicalize_sparse_mla_kv_cache_dtype、is_sparse 的 use_mha/use_masked_mha 覆写（L774-795）、_DSV32_MASKED_MHA_THRESHOLDS/_use_masked_mha（L1533-1563）、topk 字段族、mla.py indexer 调用位与布线、deepseek_v2.py 的 V3.2 indexer 装配段（L1091-1141）、DSV4 indexer 装配与 sparse 断言支、DeepseekV4Indexer/DeepseekSparseSWABackend/CompressorBackend 类本体（→ ch26，get_attn_backend 如实抛 NotImplementedError 带章界注记）。
- **delete[4] spec-decode**：backend.py 的 _init_reorder_batch_threshold 抬阈值段（L666-683）、build_for_cudagraph_capture、flashmla 的 reshape_query_for_spec_decode/reshape_attn_output_for_spec_decode（等价 view/flatten 替代并注记）。
- **delete[5] batch-invariant**：prefix-caching 禁用告警（L466-479）、FlashMLA 手工 tile_scheduler 段（L287-315）。
- **delete[6] 双 kernel 分派**：flash_mla_with_kvcache_fp8 支与 get_mla_metadata_dense_fp8/CG 缓冲拷贝段（保留 bf16 单格式）。
- **delete[7] DSV4 aux/scratch**：aux_stream_list/ln_events 参数与三路并行族、eager_scratch_pool 参数与消费、PREFILL_CHUNK_SIZE、compressor 的 scratch 位。
- **delete[8] kv_transfer/non-causal**：maybe_transfer_kv_layer 装饰器、build 的 non_causal_decode 段（L2028-L2054）。
- **章界收窄（→ chN 注记）**：DecoderLayer 的 MoE/MLP/forward（→ ch28）、yarn mscale/变体 rope（→ 章界外）、Deepseek(Attention/V2Attention) body（名字位标记类）、runner 其余全量（→ ch17/19/21/23）、selector 自动优先级（→ ch21）、Attention 通用插座（→ ch21/23）、kv_sharing 分派（→ ch23）、MambaSpec 支与 spec 长尾族（→ ch13/14/27）、nvfp4/turboquant 分支（→ ch27）。

## 测试地图（38 例 ↔ 机制）

- 装配幕：三岔选型 ×3 / 投影形状账（[out,in] 契约）/ 无 q-lora 分支 / Wrapper+MLAModules / 吸收重排布局（W_UK_T 逐元素核对 + bmm 形状账）/ is_mla 家族断言。
- spec/组化幕：spec 自报（576/单头/无 K/V 拆维）/ 字节账（bf16/584B/656B/storage÷compress/alignment padding/merge 断言）/ get_mla_dims 双记法 / 组化四级（uniform/DSV4 grouped/页归一/attention-free）。
- 前置幕：split 边界（含 uniform 语义与两早退）/ 四区重排换位 / 阈值 128 / runner 取最小 / _may_reorder_batch / runner 收 spec（None 跳过）。
- 前向幕（数值等价——本章招牌）：**MQA 吸收腿 == 上投影 MHA**（decode）/ **chunked 分块 + LSE merge == 全量**（workspace=144<context=150 → 2 块）/ **混批两腿同一前向**（decode 1 + prefill 140）/ token 维切刀探针 / merge 恒等式（两块 LSE 合并 == 整块 softmax）+ ptwc 门 / FlashMLA seam kernel 逐请求对照 / RoPE 只打 rope 段（nope 逐位相等 + rope 段旋转）/ kv_a_layernorm 只打 kv_c / workspace 公式三档 / chunked metadata 账本 / builder 混批计数。
- DSV4 幕：compress_ratio 逐层+MTP+max(1,·) 护栏 / spec 双布局（None/uint8+576B 对齐/584B 特账）/ SWA 子缓存自报+独立 prefix 注册 / wo_a bmm 组化。
- 注册表面：MLAPrefillBackendEnum 成员+CUSTOM 注册流+clone / decode 家族覆盖表。
