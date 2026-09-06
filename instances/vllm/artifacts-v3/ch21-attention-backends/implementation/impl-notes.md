# ch21《注意力后端》impl-notes —— 只做减法精简版

对应真实源码 pin **vLLM v0.27.1（6e448d0ea）**，行号全部现核（2026-09-06，
`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。
运行：`cd instances/vllm/artifacts-v3/ch21-attention-backends && python -m pytest tests/ -q`
→ **70 passed**（~2s）。纯 host 单元测试：真 torch/numpy/pydantic（真
torch.library 注册的统一算子、真 @cache/@lru_cache、真 register_backend 覆盖
表）；无 vllm 包、无 CUDA 上下文——vllm_flash_attn CUDA op / pynvml NVML /
容器平台面以 HOST SEAM 承载（见 §Seam 清单）。

**验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本包（HOST SEAM
例外见 §Seam 清单——每个 seam 行内标注真实源锚）。SUBTRACTED 标记逐条挂
dossier.subtraction_plan.delete[0..9] 批准项编号（章界外域段另以「→ chN」
注记——ch18/ch19 立下的切面惯例）。

**lint**：`python scripts/lint_fidelity.py <本章目录>` → **全部通过（无
BLOCKING）**；must_keep 94 符号经 linter `over_subtraction` 项全数核在。

## 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `v1/attention/backend.py` | `vllm/v1/attention/backend.py` | **协议层全文**：AttentionType/MultipleOf + AttentionBackend ABC（四身份证 + get_kv_cache_block_dim 哨兵探测 + get_kv_cache_stride_order 物理序声明 + 全能力探针 + validate_configuration 聚合器 L319-L393）+ CommonAttentionMetadata 核心 10 字段（delete[1] 删长尾）+ AttentionCGSupport 四档 + AttentionMetadataBuilder（build 唯一中心入口/update_block_table/build_for_cudagraph_capture）+ AttentionImplBase/AttentionImpl（delete[2] 删 DCP/PCP/LSE 族与 __new__/fused_* 钩子） |
| `v1/attention/selector.py` | `vllm/v1/attention/selector.py` | **选后端入口全文**：AttentionSelectorConfig NamedTuple（L24-L58）+ get_attn_spec_kind（L61-L98）+ get_attn_backend（L101-L174——维度打包 + backend_per_kind 覆写）+ _cached_get_attn_backend（L177-L208——resolve_obj_by_qualname + set_kv_cache_layout 副作用）；delete[3] 删 mamba 双函数 |
| `platforms/cuda.py` | `vllm/platforms/cuda.py` | **选后端三件**：_get_backend_priorities（L82-L163 逐字——use_mla×算力代×fp8 KV×num_heads 分档）+ _backend_cls_path/_get_attn_backend_class/_BackendCandidate + CudaPlatform.{get_valid_backends（L358-L394——ImportError 容忍）, get_attn_backend_cls（L396-L492——显式只校验/自动 min(priority) 胜者 + --block-size warning + 胜者日志）}；get_device_capability 以算力代装配位承载（NVML 域不进 host） |
| `v1/attention/backends/registry.py` | `vllm/v1/attention/backends/registry.py` | 注册表全文切面：_AttentionBackendEnumMeta（错误信息列全成员）+ AttentionBackendEnum（优先级表引用的成员一个不删；delete[4] 删 ROCM/XPU/DSV4/MiniMax/HPC/CPU_ATTN 15 同构条目 + MambaAttentionBackendEnum 整类）+ _ATTN_OVERRIDES + register_backend（装饰器/直注两形态；is_mamba 分支连带删） |
| `v1/attention/backends/flash_attn.py` | `vllm/v1/attention/backends/flash_attn.py` | **代表性后端四件套**：FlashAttentionBackend 全探针面（含 get_kv_cache_shape (B,H,N,2D) 与 stride_order NHD/HND 分叉 L133-L168）+ FlashAttentionMetadata 全 dataclass（L242-L299 逐字）+ Builder（__init__ 按 delete[6] 删 DCP/rswa 段、build 的 DCP/cascade 分支与尾段按 delete[5] 删、schedule 闭包/update_block_table 逐字）+ Impl（__init__ 删 DCP 尾段；forward 删 encoder 早退/descale/mm/R-SWA/dynamic_causal/cascade 支；do_kv_cache_update L1098-L1132 逐字） |
| `v1/attention/backends/fa_utils.py` | `vllm/v1/attention/backends/fa_utils.py` | 版本探针族真身（is_fa_version_supported/flash_attn_supports_kv_cache_dtype/…_quant_query_input/…_sinks——host 走真实 ImportError 回退）+ 平台 import 面（CUDA 支名字面）+ 两个 CUDA op 的 HOST SEAM 镜像（flash_attn_varlen_func 精确数学 / get_scheduler_metadata 恒 None——真身同型的 ROCm stub） |
| `v1/attention/backends/utils.py` | `vllm/v1/attention/backends/utils.py` | KV layout 覆盖机制：KVCacheLayoutType/_KV_CACHE_LAYOUT_OVERRIDE + get_kv_cache_layout（L82-L109）/set_kv_cache_layout（L112-L115）+ PAD_SLOT_ID/NULL_BLOCK_ID 界碑；其余（mm/fast-prefill/local-attn/split 族）章界收窄 |
| `v1/kv_cache_interface.py` | `vllm/v1/kv_cache_interface.py` | 消费面：KVQuantMode/get_kv_quant_mode/is_quantized_kv_cache + KVCacheSpecKind 十 kind + KVCacheSpec/AttentionSpec/FullAttentionSpec（merge 全链）/SlidingWindowSpec/EncoderOnlyAttentionSpec/MambaSpec（mamba_type 字段随 delete[4] 连带删）/UniformTypeKVCacheSpecs + get_kv_cache_spec_kind + KVCacheTensor/KVCacheGroupSpec/KVCacheConfig |
| `config/attention.py` | `vllm/config/attention.py` | AttentionConfig：backend + backend_per_kind（docstring 自带混布例）+ use_non_causal/flash_attn_max_num_splits_for_cuda_graph 消费位 + 两 field_validator（"auto"→None、kind 校验）；其余字段章界收窄 |
| `model_executor/layers/attention/attention.py` | `.../layers/attention/attention.py` | **模型层插座**：Attention.__init__（选后端段 L350-L363 + 装配段 L420-L446 + 自注册 L444-L446；delete[9] 删 alibi/互斥/chunk/FLEX 段与 query_quant）+ forward（双算子分发 L488-L582；L512-L513 保留）+ get_kv_cache_spec（L621-L694；turboquant 支删）+ 统一算子三件（get_attention_context/unified_kv_cache_update/unified_attention_with_output + fake + **真实 torch.library 注册**）+ validate_kv_sharing_target/_largest_kernel_block_within/set_default_quant_scales/_init_kv_cache_quant |
| `model_executor/layers/attention_layer_base.py` | `.../attention_layer_base.py` | 全文逐字（get_attn_backend/get_kv_cache_spec 抽象 + bind_kv_cache 默认实现） |
| `model_executor/models/utils.py` | `vllm/model_executor/models/utils.py` | extract_layer_index（L917-L940 逐字——bind_kv_cache 排序与 kv_sharing 校验消费） |
| `v1/worker/utils.py` | `vllm/v1/worker/utils.py` | AttentionGroup（L216-L227）+ create_metadata_builders（L229-L259——spec 覆写块/kernel_block_size）+ get_metadata_builder + select_common_block_size（L266-L332 逐字含 Case 1/Case 2 证明注释）+ prepare_kernel_block_sizes（L335-L376 逐字）+ bind_kv_cache（L466-L525 逐字） |
| `v1/worker/gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | 站 6-12 切面：initialize_attn_backend（L7020-L7125——AttentionGroupKey/get_attn_backends_for_group/create_attn_groups；delete[8] 删 FastPrefill/CP 检查）+ _check_and_update_cudagraph_mode（L7161-L7202 最弱链）+ initialize_metadata_builders + initialize_kv_cache（L7624-L7681 按 delete[8] 裁）+ initialize_kv_cache_tensors/_allocate/_reshape_kv_cache_tensors（L7312-L7479 逐字）+ _reshape_attention_kv_cache（gpu/attn_utils.py:L211-L264 逐字）+ _has_mixed_attention_kv_layout/_update_hybrid_attention_mamba_layout + _get_slot_mappings（L4082-L4154 逐字）+ _build_attention_metadata（L2284-L2618 按 delete[7] 裁）+ execute_model 切面（L4307-L4443 主段：has_separate_kv_update/三元选择/set_forward_context）+ _model_forward（ENGINE SEAM） |
| `_custom_ops.py` | `vllm/_custom_ops.py` | reshape_and_cache_flash thin wrapper（L2614-L2633 逐字转发面）+ kernel 本体 host 镜像（csrc/libtorch_stable/cache_kernels.cu:L315-L344） |
| `forward_context.py` | `vllm/forward_context.py` | BatchDescriptor + ForwardContext（attn_metadata/slot_mapping 两通道）+ get/create/override/set_forward_context（L259-L344 主干——站 10 落点；观测/DP/MoE 面收窄） |
| `utils/import_utils.py` | `vllm/utils/import_utils.py` | resolve_obj_by_qualname（L104-L109 逐字——懒加载原语） |
| `utils/torch_utils.py` | `vllm/utils/torch_utils.py` | is_quantized_kv_cache/canonicalize_singleton_dim_strides/kv_cache_dtype_str_to_dtype/get_dtype_size/nvfp4_kv_cache_full_dim + LayerName opaque 族（L832-L889）+ direct_register_custom_op（L901-L939——真实注册）+ vllm_lib |
| `utils/math_utils.py` | `vllm/utils/math_utils.py` | cdiv/round_up |
| `_host_seams.py` | （跨域缝合，见 §Seam 清单） | HOST SEAM 登记处 |

## 1:1 Source Map（核心行；改动=减法或 seam，原因=批准条/章节边界）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `AttentionBackend.validate_configuration` | backend.py:L319-L393 | **逐字**（十五连探针 + supports_combination 收尾） | must_keep；站 4 判定核心 |
| `AttentionBackend.get_kv_cache_block_dim` | backend.py:L99-L117 | 逐字（_S=1234567 哨兵） | must_keep；混布归一探测 |
| `_get_backend_priorities` | cuda.py:L82-L163 | **逐字**（含 TOKENSPEED_MLA 的 bs≈8 注释原文） | must_keep；站 3 优先级表 |
| `get_valid_backends` | cuda.py:L358-L394 | 逐字（ImportError→["ImportError"] 原因条） | must_keep；站 4 回退循环体 |
| `get_attn_backend_cls` | cuda.py:L396-L492 | 逐字（显式只校验 L407-L423 / min(priority) L453-L456 / --block-size warning L463-L480 / 胜者日志 L482-L490） | must_keep；站 5 裁决 |
| `get_attn_backend` | selector.py:L101-L174 | 逐字（CacheDType Literal 面等值承载） | must_keep；站 2 |
| `_cached_get_attn_backend` | selector.py:L177-L208 | 逐字（set_kv_cache_layout 副作用 + info_once） | must_keep；站 5 副作用 |
| `AttentionBackendEnum.get_path/get_class` | registry.py:L130-L159 | 逐字（覆盖表感知 + 懒加载） | must_keep ×2 |
| `register_backend` | registry.py:L241-L293 | 逐字（is_mamba 分支 delete[4] 连带删） | must_keep |
| `Attention.__init__` 选段 | attention.py:L350-L363/L420-L446 | 逐字（L364-L418、L467-L486 按 delete[9] 删，L460-L466 保留） | must_keep（Attention/static_forward_context/use_direct_call/kv_sharing_target_layer_name） |
| `Attention.forward` | attention.py:L488-L582 | 逐字（L508-L511/L514-L524 按 delete[9] 删；L512-L513 保留——output_dtype guard） | must_keep（unified_* 双算子 + kv_cache_dummy_dep） |
| `get_attention_context` 三态解包 | attention.py:L732-L772 | 逐字 | must_keep |
| `unified_kv_cache_update`（+fake+注册） | attention.py:L775-L814 | 逐字 + 真实 torch.library 注册 | must_keep；站 12 写腿入口 |
| `unified_attention_with_output`（+fake+注册） | attention.py:L817-L867 | 逐字（两装饰器为 ch19/ch16 域 seam 恒等） | must_keep；读+算算子 |
| `initialize_attn_backend` | gpu_model_runner.py:L7020-L7125 | 逐字（FastPrefill 支 L7064-L7068 与 CP 检查 L7121-L7122 按 delete[8] 删） | must_keep（AttentionGroupKey/full_cls_name/get_attn_backends_for_group） |
| `_check_and_update_cudagraph_mode` | gpu_model_runner.py:L7161-L7202 | 逐字（drafter 尾 L7204-L7218 删；降级链 seam 记录 min_cg_support） | must_keep ×4（最弱链） |
| `prepare_kernel_block_sizes`/`select_common_block_size` | worker/utils.py:L335-L376 / L266-L332 | **逐字**（Case 1/Case 2 注释证明原文） | must_keep ×2；「256→4×64」协商点 |
| `_reshape_kv_cache_tensors` | gpu_model_runner.py:L7364-L7479 | **逐字**（shape/stride 消费 L7433-L7453 是站 7 本体） | must_keep |
| `_reshape_attention_kv_cache` | gpu/attn_utils.py:L211-L264 | **逐字**（permute 置换/as_strided/contiguous 三支） | 站 7 as_strided 定形实现 |
| `bind_kv_cache` | worker/utils.py:L466-L525 | **逐字**（index2name 排序 + 逐层 bind） | must_keep |
| `_build_attention_metadata` | gpu_model_runner.py:L2284-L2618 | cm_base L2430-L2449（被删字段 kwargs 连带删，防 TypeError）+ _get_block_table L2325-L2341（NULL_BLOCK_ID 尾行填）+ build 分派 L2526-L2546 + layer_name 铺设 L2555-L2556 + 逐组换表 L2561-L2574 逐字；delete[7] 各支就地标注 | must_keep ×5 |
| `FlashAttentionBackend.get_kv_cache_shape/stride_order` | flash_attn.py:L133-L168 | **逐字**（(B,H,N,2D) + NHD (0,2,1,3)/HND (0,1,2,3) + 层维变体） | must_keep ×2；布局声明 |
| `FlashAttentionMetadataBuilder.build` | flash_attn.py:L458-L726 | 解构 L468-L476 逐字；schedule 闭包 L515-L541 逐字；主路径 else 支 L642-L650 去缩进无条件（DCP/cascade 分支 delete[5] 删）；装配 L672-L696 逐字 | must_keep（build/schedule/get_scheduler_metadata） |
| `FlashAttentionMetadataBuilder.update_block_table` | flash_attn.py:L728-L737 | **逐字**（copy.copy 浅拷只换表） | must_keep；混合组换表复用 |
| `FlashAttentionImpl.forward` | flash_attn.py:L838-L1096 | K/V 视图 L904-L905 + 解包 L929-L935 + varlen 调用 L1041-L1066 逐字（descale/s_aux/mask_mod kwargs 与 mm/R-SWA/dynamic_causal/cascade 支 delete[5] 删）；NOTE(woosuk) eager 接缝警告原文保留 | must_keep；站 12 读腿 |
| `FlashAttentionImpl.do_kv_cache_update` | flash_attn.py:L1098-L1132 | **逐字**（woosuk NOTE：slot_mapping 形状定 token 数） | must_keep；站 12 写腿 |
| `reshape_and_cache_flash` | _custom_ops.py:L2614-L2633 + cache_kernels.cu:L315-L344 | 转发面逐字 + kernel 本体 host 镜像（slot<0 跳过 + slot//bs、slot%bs 逆分解） | must_keep；写腿算子 |
| `set_forward_context` | forward_context.py:L259-L344 | 签名 + create/override/yield 主干逐字（观测/DP/平台注入面收窄） | must_keep；站 10 |
| `AttentionConfig.validate_backend_per_kind_before` | config/attention.py:L153-L177 | **逐字**（kind 值域校验 + 枚举解析） | must_keep（backend_per_kind 用户面） |

## 删除账本（dossier.subtraction_plan.delete[0..9] 落点）

| delete | 内容 | 落点（# SUBTRACTED 标注） |
|---|---|---|
| [0] | MLAAttentionImpl 整类 + subclass_* 两工厂 | backend.py 尾（L1009-L1122 → ch24/25） |
| [1] | CommonAttentionMetadata 长尾字段与方法 | backend.py（L442-L600）+ gpu_model_runner.py cm_base kwargs 行 |
| [2] | AttentionImplBase DCP/PCP/LSE 族 + __new__；AttentionImpl fused_*；AttentionLayer Protocol | backend.py（L775-L793/L804-L889/L936-L1006） |
| [3] | get_mamba_attn_backend 双函数 | selector.py（L211-L230） |
| [4] | registry 15 同构条目 + MambaAttentionBackendEnum + is_mamba 分支；保留优先级表全部成员与 TORCH_SDPA | registry.py；AttentionConfig.__post_init__ MSA 别名块连带删 |
| [5] | cascade/DCP/encoder 前向 + mm/R-SWA/dynamic_causal 段 + descale/sinks + Builder/Impl 的 DCP/rswa 字段 + build 尾段 | flash_attn.py（各支就地标注；数学归 ch20、扩展态归 ch24+/多模态域） |
| [6] | Builder.__init__ DCP/rswa/持久缓冲段（保留 dcp_world_size try/except 与 aot_sliding_window=None——schedule 闭包 L528/L537 无条件读）+ aot_sliding_window 多窗口回退 | flash_attn.py（L401-L456/L484-L498） |
| [7] | _build_attention_metadata 的 mm/rswa/replayssm/dcp/is_prefilling/ubatch/spec-decode 分支 + routed_experts 快照（L2315-L2321 max_seq_len 与 L2343-L2345 保留） | gpu_model_runner.py |
| [8] | initialize_attn_backend FastPrefill/CP 检查 + initialize_kv_cache mamba/connector/kv_sharing 尾部（L7640 保留——站 6 入口） | gpu_model_runner.py |
| [9] | Attention.__init__ alibi/互斥/chunk/FLEX 段 + quant/query_quant 初始化（L460-L466 保留）+ forward 量化段（L512-L513 保留）+ get_kv_cache_spec turboquant 支 | model_executor attention.py |

## Seam 清单（HOST/ENGINE SEAM——真实代码之外唯一允许的承载，行内标注）

1. **`flash_attn_varlen_func`**（fa_utils.py 尾）：读腿 CUDA op 的精确数学
   镜像——每请求穿 block_table 逐逻辑块 gather K/V（间接寻址）、GQA 广播、
   causal（bool 或 int32 per-seq 张量——FA4 dynamic_causal 的消费面）、fp64
   softmax。只承载本章用面（(-1,-1) 无窗口/无 alibi）——滑窗/级联数学 → ch20。
2. **`get_scheduler_metadata`**（fa_utils.py 尾）：真身同型 stub（ROCm 无 FA3
   平台返 None——fa_utils.py:L63-L65 原文形态）；host 恒 FA2 → 无 AOT 调度。
3. **`get_flash_attn_version`**（_host_seams.py）：真身按算力代+配置解析 3/4/2；
   host 无库 → 装配恒 2（FA2 语义可镜像承载，ch22 同款）。
4. **`reshape_and_cache_flash`**（_custom_ops.py）：thin wrapper 转发面逐字，
   转发目标为 cache_kernels.cu:L315-L344 kernel 本体的逐 token 镜像。
5. **`current_platform`**（_host_seams.py）：get_attn_backend_cls /
   get_device_capability / is_device_capability_family 惰性转出 platforms/
   cuda.py 的 CudaPlatform 切面（真实 detect_platform 按 host 探测——host 无
   CUDA 会选 CPU 平台，其 get_attn_backend_cls 是 interface.py:L373 空实现，
   与本章「CUDA 优先级表」主线不符故转出）；`CudaPlatform.get_device_capability`
   以算力代类属性装配位承载（真实 NVML L733-L742 不进 host）；`opaque_attention_op`
   False（host 走 direct-call 分支；torch.ops 分支同控制流且真实注册）。
6. **`DeviceCapability`**（_host_seams.py）：interface.py:L89-L119 逐字语义的
   类面（NamedTuple 全序比较展开——@cache 键与 >= 探针都吃它）。
7. **组探测/logger/envs/deprecated 观测**：get_dcp_group/get_pcp_group 与真实
   「测试环境未初始化组」同型抛 AssertionError（ImplBase 组探测的源码原生
   退化路径）；init_logger 包 once 族转发到真 logging（caplog 可见）；
   record_function_or_nullcontext nullcontext；eager_break_during_capture/
   maybe_transfer_kv_layer 恒等装饰器（ch19/ch16 域）。
8. **配置面**（VllmConfigSeam）：各 config namespace 的本章消费字段子集
   （被删支开关取默认关的真实值，守卫位永不触发）；@config 装饰器机制逐字
   （pydantic dataclass，extra=forbid）；set_current_vllm_config/get_current_
   vllm_config 模块级单值 + 保存/恢复（vllm/config/vllm.py 机制）；
   `resolve_cudagraph_mode_and_sizes` 与 `initialize_cudagraph_keys` 为 ch19
   域观测位（记录 min_cg_support 供测试断言能力传染）；CacheDType 以等值
   Literal 承载类型面。
9. **`GPUModelRunner` 切面 __init__ + `_model_forward`**（ENGINE SEAM，
   ch17/ch22 同款切面构造）：真实 __init__（L456-L760）的选后端/metadata/
   前向上下文消费字段直供；持久缓冲面（query_start_loc/seq_lens/optimistic_
   seq_lens_cpu 的 CpuGpu 切片面）由测试按拍覆写；`_model_forward` 以逐
   Attention 层直调承载（每层 forward 内部先写腿后读腿——attention.py 真身
   的两算子调用序），观测位记录前向内按 layer_name 取到的两通道。
10. **`InputBatchSeam`/`_BlockTableSeam`**（ch18/ch22 域协议载体）：块表线
    字段面（block_tables[gid] 的 get_device_tensor/slot_mapping 切片面）；
    `get_block_table_width` 真身逐字收录（block_table.py:L20-L40，ch22 域）。
11. **execute_model 切面的观测/直供位**：seam_num_reqs/seam_num_tokens/
    seam_max_query_len/seam_cudagraph_mode/seam_batch_desc 承载 ch18/ch19 域
    产出的消费面（真实 _update_states/_prepare_inputs/_determine_batch_
    execution_and_padding 归彼域，ch22 同款）。
12. **量化域类型面**：UnquantizedLinearMethod/QuantizeMethodBase/
    QuantizationConfig/BaseKVCacheMethod 占位（ch27 域；本章 quant_config=None
    分支不可达，调用位保留）。Q/K/V_SCALE_CONSTANT 以等值 1.0 承载。

## 已知偏差（非 seam 的显式记录）

- `AttentionImplBase` 删 DCP/PCP/LSE 族与 `__new__`（delete[2] 批准区间
  L804-L889），但**保留**「Required attributes」注解块（num_heads/head_size/
  scale，L812-L815）——批注描述限定「DCP/PCP/LSE 族字段与 __new__」，注解
  块不属该族；FlashAttentionImpl.__init__ 的组探测退化（dcp_world_size=1）
  由 Builder/Impl 内的 try/except 真实路径承载。
- `register_backend` 的 `is_mamba` 参数与分支连带删（delete[4] 删 Mamba 枚举
  与 _MAMBA_ATTN_OVERRIDES 后无消费面）；签名其余逐字。
- `flash_attn.py` 的 `forward` 保留 `is_dynamic_causal` 死变量（L972 不在
  批准删除区间；其消费段 L974-L1039 已删）。
- `_custom_ops.py` 只含 reshape_and_cache_flash 一个 op——其余 ~120 op 属
  各 kernel 章（ch13/ch20/ch27/ch33），章界收窄。
- 测试经 `register_backend`（真实第三方注册机制）把包内 FlashAttentionBackend
  挂到 FLASH_ATTN 名下、测试桩挂到 TRITON_ATTN 名下——这本身就是该机制的
  真实用法；未注册成员（FLASHINFER 等）走真实 vllm.* 路径 → host ImportError
  → get_valid_backends 的 ImportError 容忍分支被真实触发。
- 测试构造 Attention 层时以 `torch.set_default_dtype(torch.float16)` 包裹
  ——真实路径是 load 期把模型 dtype 设为默认 dtype（FA 只支持 fp16/bf16）。

## 2026-09-06 复核（dossier 13:08 修订版对账）

对 13:08 修订版 dossier 逐项复核后确认：10 条 delete 批准项的**全部保留区
括注**（[5] varlen kwargs 连带删、[6] L392-L400/L441-L443 保留、[7]
L2315-L2321/L2343-L2345 保留+L2576-L2598 连带收尾+调用点 L4378 解包改
单值、[8] L7640 保留+drafter 连带删、[9] L460-L466/L512-L513/L531 保留）
在精简版中全部落实，must_keep 由 87 扩到 94 全数核在。同轮锚注现核修正
（全部对 v0.27.1 现核，修正后 363 个 `# SOURCE:` 锚 0 漂移）：

- `kv_cache_interface.py` 尾段 ~20 处行号锚修正（SlidingWindowSpec/MambaSpec/
  尾部三 dataclass 段的区间偏移）；`MambaSpec.max_memory_usage_bytes` 实为
  **全三分支逐字**（L729-L738），原文误标「none 档逐字」；
  `FullAttentionSpec.merge` 尾断言消息回归逐字（原文多出
  "in a same kv cache group"——违只删不增）；`fields`/`prod` 改回真实
  import 面（原 `fields_of`/`prod_of` 手搭替身属发明，已删）。
- 补账此前无 SUBTRACTED 注记的章界收窄段（现全部显式标注）：
  `FullAttentionSpec.real_page_size_bytes`（L335-L361）/`TQFullAttentionSpec`
  （L362-L387）→ ch27；`HiddenStateCacheSpec`/`RSWASpec`/`ChunkedLocalAttentionSpec`
  与 `SlidingWindowMLASpec`（L630-L708）→ ch24/25/enc-dec（delete[0] 同域）；
  registry `CPU_ATTN`（L124）。
- `_host_seams.py` 4 处 seam 锚修正（init_logger L204、opaque_attention_op
  L1116+cuda.py:L569、set_additional_forward_context L1265、DeviceCapability
  __repr__ 改为 NamedTuple 机制注记）；cuda.py 尾段 SUBTRACTED 更名
  NonNvmlCudaPlatform（L717-L1016）；worker/utils 清零族更名
  `_zero_kv_blocks_kernel`/`KVBlockZeroer`（L44-L213）；models/utils
  `extract_layer_index` 区间补全 L917-L946；import_utils
  `resolve_obj_by_qualname` 区间补全 L104-L110；
  `attention_layer_base.py` 恢复 `vllm_config: VllmConfig` 注解（逐字）。
