# ch19 精简版 impl-notes — 编译与捕获（Part V：GPU 不等 Python）

- **Pin**：vLLM v0.27.1（`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`）。全部 `# SOURCE:` 行号
  对当前 pin 树现核（`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。
- **产物**：`implementation/`（包布局镜像源码树：config / compilation / forward_context /
  model_executor / utils / v1）+ `tests/test_compile_capture.py`（101 测）。
- **跑法**：`cd instances/vllm/artifacts-v3/ch19-compile-capture && python -m pytest tests/ -q`
  → **101 passed**（~6s；CUDA-gated 2 测在本机有 CUDA 时实跑、无 CUDA 自动 skip）。
  host 可跑：真 torch（真 Dynamo/FX/split_module、真 torch.library 注册的统一算子）、
  真 numpy；无 vllm 包、无 CUDA graph（捕获/回放真路径的 CUDA 段以 `@pytest.mark.cuda`
  门控——与上游 tests/v1/cudagraph 的门控同款）。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支（+ HOST SEAM 面替换，见
  §Seam 清单）应当 ≈ 得到本包。`# SUBTRACTED:` 标记逐条挂 dossier.subtraction_plan.
  delete[0..9] 批准项编号或「章界外域收窄」（impl-notes §范围裁剪）；lint_fidelity 的
  `over_subtraction` 项核 must_keep 52 符号全数在精简版。
- **lint**：`python scripts/lint_fidelity.py <本章目录>` → 无 BLOCKING（见文末）。

## 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `config/vllm.py` | `vllm/config/vllm.py` | OptimizationLevel -O0..-O3 + 四张预设表 + O2 谓词函数 + VllmConfig 的 ch19 切面（optimization_level 字段、post_init 的 has_blocked_weights→+quant_fp8（F10 苗）/mode 落账/ir_enable_torch_wrap/custom_ops 基础档/档位预设应用/相容闸）+ config-context 三件套；VllmConfig 其余 20+ 子配置与 post_init 主体是 ch03 域 SUBTRACTED |
| `config/compilation.py` | `vllm/config/compilation.py` | CUDAGraphMode 全枚举、PassConfig、DynamicShapesConfig、CompilationConfig 主字段集 + `_attention_ops` 13 算子 + `set_splitting_ops_for_v1` 主支（delete[9] 四条扩展分支删）+ splitting_ops 谓词族 + `resolve_cudagraph_mode_and_sizes` 最弱链降级链 + `adjust_cudagraph_sizes_for_spec_decode` + `get_compile_ranges` |
| `config/utils.py` | `vllm/config/utils.py` | Range（compile_ranges 区间载体）；hash/config 装饰器族 ch03 域删 |
| `forward_context.py` | `vllm/forward_context.py` | BatchDescriptor frozen 五字段 + DPMetadata（SP 扩展 delete[8] 删）+ ForwardContext + get/create/override/set_forward_context（batchsize 统计 delete[8] 删） |
| `model_executor/custom_op.py` | `vllm/model_executor/custom_op.py` | CustomOp 全主链（构造期 dispatch_forward/enabled/default_on/maybe_compile/register）；PluggableLayer+OOT 族 delete[1] 删 |
| `model_executor/layers/layernorm.py` | `vllm/model_executor/layers/layernorm.py` | RMSNorm 代表实例（register + forward_native/cuda/xpu 三实现）；其余 norm 层模型域删 |
| `model_executor/layers/attention/attention.py` | `.../attention/attention.py` | Attention 构造自注册 + forward out-variant + 统一算子三件（get_attention_context / unified_kv_cache_update / unified_attention_with_output + fake + direct_register_custom_op 注册）；kv_scales/kv_sharing/get_kv_cache_spec delete[10] 删、后端选择 ch21 域收窄（impl/attn_backend 注入） |
| `utils/torch_utils.py` | `vllm/utils/torch_utils.py` | current_stream/weak_ref_tensor(s) + is_torch_equal_or_newer + LayerName opaque 族（_encode/_resolve/_USE_LAYERNAME）+ direct_register_custom_op + vllm_lib |
| `utils/import_utils.py` | `vllm/utils/import_utils.py` | resolve_obj_by_qualname 逐字 |
| `utils/gc_utils.py` | `vllm/utils/gc_utils.py` | freeze_gc_heap（m18 尾四连之二）+ maybe_attach_gc_debug_callback no-op 面 |
| `utils/gpu_sync_debug.py` | `vllm/utils/gpu_sync_debug.py` | sync-check 闸门 global + enable_gpu_sync_check（m18 尾四连之三）；装饰器/patch 族观测域删 |
| `compilation/backends.py` | `vllm/compilation/backends.py` | make_compiler/CompilerManager（缓存面恒禁用）/split_graph 切图算法/wrap_with_cudagraph_if_needed/PiecewiseCompileInterpreter.call_module/VllmBackend.__call__ 主链（缓存/落盘/序列化巨块 delete[3] 删、两个边界优化 pass delete[4] 删） |
| `compilation/partition_rules.py` | `vllm/compilation/partition_rules.py` | should_split + inductor_partition_rule_context（回归消费点） |
| `compilation/piecewise_backend.py` | `vllm/compilation/piecewise_backend.py` | PiecewiseBackend 编译管理（create_concrete_args/get_fake_args_from_graph/compile_all_ranges/_find_range_for_shape/__call__）；序列化热启动缓存域删 |
| `compilation/wrapper.py` | `vllm/compilation/wrapper.py` | TorchCompileWithNoGuardsWrapper（丢 guard + fullgraph=True + dynamic=False 一次编译）；nvtx/bytecode_hook/AOT 实验态删 |
| `compilation/cuda_graph.py` | `vllm/compilation/cuda_graph.py` | CUDAGraphStat + CUDAGraphEntry/Options + CUDAGraphWrapper 捕获/回放全链（CUDAGraphLogging 聚合、offloader 同步、_runnable_str delete[5] 删；DEBUG data_ptr 断言保留） |
| `compilation/monitor.py` | `vllm/compilation/monitor.py` | 捕获窗口 tripwire（set/validate_cudagraph_capturing_enabled）+ torch_compile_start_time |
| `compilation/counter.py` | `vllm/compilation/counter.py` | 全文逐字（观测计数器） |
| `compilation/compiler_interface.py` | `vllm/compilation/compiler_interface.py` | CompilerInterface/EagerAdaptor 逐字 + InductorAdaptor 壳（compile 体 torch-internal 域）+ trigger_inductor_lazy_init |
| `v1/attention/backend.py` | `vllm/v1/attention/backend.py` | ch19 消费的契约面：AttentionType / AttentionBackend 类头（forward_includes_kv_cache_update）/ AttentionMetadata marker / AttentionCGSupport / AttentionMetadataBuilder.get_cudagraph_support；后端实现族 ch20/ch21 域删 |
| `v1/cudagraph_dispatcher.py` | `vllm/v1/cudagraph_dispatcher.py` | CudagraphDispatcher 全类（delete[2] LoRA 专化路径删净）；上游 tests/v1/cudagraph/test_cudagraph_dispatch.py 的 dispatch/capture-descs 用例在本书 tests 镜像复证 |
| `v1/worker/worker_base.py` | `vllm/v1/worker/worker_base.py` | CompilationTimes NamedTuple；WorkerBase 控制面 ch17 域删 |
| `v1/worker/gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | 执行形态 spans：padding 四件套（四个载体方法）、一拍裁决（_determine_batch_execution_and_padding 全文 + execute_model 两个 pinned 段 + _model_forward + _is_uniform_decode + _pad_for_sequence_parallelism）、捕获编排（capture_model/_warmup_and_capture/_capture_cudagraphs/_freeze_gc）、load_model 尾段 FULL wrapper 挂载、_check_and_update_cudagraph_mode 最弱链、_get_slot_mappings 全文 |
| `v1/worker/gpu_worker.py` | `vllm/v1/worker/gpu_worker.py` | Worker.compile_or_warm_up_model 启动编排全文（delete[7] 的 startup_plan/KV 建议段与 V2 分支删、@instrument 观测删）；Worker 其余 ch17/ch34 域删 |
| `_host_seams.py` | （跨域缝合，见 §Seam 清单） | HOST SEAM 登记处 |

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因；核心行）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `CUDAGraphMode` 全枚举 | config/compilation.py:L53-L103 | **逐字** | must_keep（m11）；组合档 tuple 值/decode_mode/mixed_mode/has_mode 全保留 |
| `OptimizationLevel` + 四张 -O 预设表 | config/vllm.py:L104-L327 | 逐字（O0-O3 表 + 谓词函数 + OPTIMIZATION_LEVEL_TO_CONFIG） | must_keep（m11 站 1 读者入口） |
| `VllmConfig.__post_init__` ch19 段 | config/vllm.py:L1253-L1321 | 逐字（has_blocked_weights→+quant_fp8 / mode 落账 / ir_enable_torch_wrap / custom_ops 基础档 / 预设应用 / 相容闸） | must_keep；前后大段 ch03 域删（impl 正文引用基线） |
| `CompilationConfig.set_splitting_ops_for_v1` 主支 | config/compilation.py:L1133-L1184 | 逐字 minus delete[9] 四扩展支（fuse_attn_quant 支/空 splitting_ops 降级/SP 全图要求/DeepEP 禁图） | must_keep（m07）；默认 O2 无一触发 |
| `CustomOp.dispatch_forward` | model_executor/custom_op.py:L174-L207 | 逐字（enforce_enable 的 NOTE 块按 dossier elide 授权收窄为一行注记） | must_keep（m01 构造期冻结） |
| `CustomOp.enabled/default_on/maybe_compile` | custom_op.py:L209-L311 | 逐字（maybe_compile 的 dynamic_arg_dims 包装段按 elide 收窄至 mark_dynamic 点名） | must_keep（m02/m03） |
| `RMSNorm` 三实现 | layernorm.py:L36-L122 | 逐字（forward_native 经 vllm.ir 引用面；VLLM_BATCH_INVARIANT 实验分支保留原文） | must_keep（m01 代表实例） |
| `Attention.__init__` 注册面 | attention.py:L326-L447 | 逐字 minus delete[10]（kv scales/kv_sharing/mm_prefix clamp）与 ch21 后端选择段（impl/attn_backend 置 None 由注入面补） | must_keep（m04/m05）；query_quant 分支 F10 苗原文保留 |
| `Attention.forward` | attention.py:L488-L582 | 逐字（kv_sharing 两处守卫随 delete[10] 删） | must_keep（m05 out-variant） |
| `unified_kv_cache_update`(+fake+注册) | attention.py:L775-L814 | **逐字** | must_keep（m05 dummy 依赖保序） |
| `unified_attention_with_output`(+fake+注册) | attention.py:L817-L867 | **逐字** | must_keep（m05 统一算子） |
| `get_attention_context` | attention.py:L732-L772 | **逐字**（含 DBO list 支——spec decode 回指 ch33） | must_keep（m04 消费口） |
| `BatchDescriptor` / `ForwardContext` / `set_forward_context` | forward_context.py:L29-L58 / L131-L193 / L259-L344 | 逐字 minus delete[8]（batchsize 统计/SP 扩展/MoE 计数器族） | must_keep ×5 |
| `should_split` | compilation/partition_rules.py:L14-L38 | **逐字** | must_keep（m08 切点判定） |
| `split_graph` | compilation/backends.py:L553-L627 | 逐字 minus delete[4]（_decompose_size_nodes/_merge_empty_only_subgraphs 及调用行） | must_keep（m08）；tests 保留多切点样例证切图仍正确 |
| `wrap_with_cudagraph_if_needed` | backends.py:L633-L684 | **逐字** | must_keep（m09 每片包 PIECEWISE wrapper） |
| `PiecewiseCompileInterpreter.call_module` | backends.py:L730-L776 | **逐字**（@instrument run 包装随观测域删） | must_keep（m09 片间拼跑） |
| `VllmBackend.__call__` 主链 | backends.py:L1019-L1339 | 逐字骨架 minus delete[3]（缓存/落盘/instrument/序列化尾段——返回 split_gm 本体，生成式 execution code 逐个调用同一批 submod callables、同序） | must_keep（m08/m09 主链） |
| `TorchCompileWithNoGuardsWrapper` | compilation/wrapper.py:L47-L201 | 逐字 minus nvtx/bytecode_hook/AOT 实验态 | must_keep（m10 丢 guard） |
| `CudagraphDispatcher` 全类 | v1/cudagraph_dispatcher.py:L15-L350 | 逐字 minus delete[2]（LoRA 专化：_get_lora_cases/specialize_lora_count/bisect 分支/两处 product 的 lora_cases 维折平） | must_keep ×6；非 LoRA 部署逐字节等价 |
| `CUDAGraphWrapper` 捕获/回放 | compilation/cuda_graph.py:L145-L361 | 逐字 minus delete[5]（CUDAGraphLogging 聚合、offloader 三处同步、_runnable_str；CUDAGraphStat 保留——站 10 直接消费） | must_keep ×6（m14） |
| `_determine_batch_execution_and_padding` | gpu_model_runner.py:L3932-L4044 | **逐字**（dispatch_cudagraph 闭包/cascade·encoder 禁 FULL/DP re-dispatch/CUDAGraphStat） | must_keep（m12 消费侧/站 10） |
| padding 四件套 | gpu_model_runner.py:L2073-L2078 / L2325-L2341 / L4082-L4154 / L3662-L3664 | 各自在其载体方法内逐字（_prepare_inputs/_build_attention_metadata 的 _get_block_table 闭包/_get_slot_mappings 全文/_preprocess） | must_keep（m13）；前三载体方法的其余体 ch18/ch20 域删 |
| `capture_model` + `_warmup_and_capture` + `_capture_cudagraphs` | gpu_model_runner.py:L6814-L6918 / L6920-L6966 / L6968-L7018 | 逐字 minus delete[6]（profiler 装配/encoder cudagraph/lock_workspace；tqdm 进度条与 DBO 阈值支随观测/扩展态删） | must_keep ×3（m15 大到小+窗口开关+先热身后捕获） |
| `load_model` 尾段 | gpu_model_runner.py:L5435-L5479 | 逐字 minus breakable 分支（m17 脚注）与 UBatchWrapper 分支（DBO 扩展态） | m14 尾：FULL wrapper 挂模型外 |
| `_check_and_update_cudagraph_mode` | gpu_model_runner.py:L7161-L7202 | **逐字**（drafter 尾段 spec decode 域删） | must_keep（m16 最弱链→keys 初始化） |
| `Worker.compile_or_warm_up_model` | gpu_worker.py:L679-L853 | 逐字 minus delete[7]（图池 estimate 对比日志+KV 内存建议/startup_plan 落盘段 L719-L791、use_v2 分支、@instrument） | must_keep（m15/m18 启动编排） |
| `AttentionCGSupport` / `AttentionMetadataBuilder.get_cudagraph_support` | v1/attention/backend.py:L606-L620 / L650-L657 | **逐字** | min_cg_support 接口（ch21 站 5）；builder 其余接口 ch20/ch21 域 |
| `freeze_gc_heap` / `enable_gpu_sync_check` | utils/gc_utils.py:L96-L108 / utils/gpu_sync_debug.py:L26-L34 | 逐字（GCDebugger 回调装配与 sync suppressor patch 族观测域删） | m18 尾四连之二/三 |
| `trigger_inductor_lazy_init` | compilation/compiler_interface.py:L768-L794 | **逐字**（try 体原文） | m18 尾四连之一（best-effort try/except） |

## 对 delete[N] 的执行说明

- **delete[1]（custom_op OOT/PluggableLayer）**：整类 + op_registry_oot + `__new__` 替换分支
  + register_oot 删净；op_registry 主链完整（空 OOT 表时 `__new__` 本就是对
  super 的透传，删后走 nn.Module 默认构造，行为等价）。
- **delete[2]（dispatcher LoRA 专化）**：见 Source Map。折平后的两处循环以
  `for bs in ...: _create_padded_batch_descriptor(bs, False/True, False, 0)` 单层遍历，
  `num_active_loras` 恒 0 —— `lora_config=None` 时 `_get_lora_cases` 恒 `[0]` 的逐字节
  等价。`itertools.product` 导入随最后用点删除。
- **delete[3]（backends 缓存/落盘）**：`VllmBackend.__call__` 的 hash/cache_dir/元数据
  json/计时 instrument/split_gm 副本/depyf 钩子/生成式序列化尾段全删；
  `CompilerManager` 恒 `disable_cache=True`（伴读版每次进程内重编译，行为与冷启动一致）；
  **返回值改为 split_gm 本体**——真源的生成式 execution code 逐个调用
  `getattr(split_gm, submod)` 得到的同一批 callables、同序（getattr 先查 `__dict__` 的
  PiecewiseBackend/wrapper、再落回 `_modules` 的原 FX 子图），拼跑语义等价。
- **delete[4]（两个边界优化 pass）**：`_decompose_size_nodes`/`_merge_empty_only_
  subgraphs` 及调用行删——删的是优化不是功能（split_module 语义不变，可能多出平凡
  子图）；tests 的三片切图样例（TestSplitGraph）验证主算法仍正确。
- **delete[5]（cuda_graph 观测/offloader）**：CUDAGraphLogging 聚合类与 offloader 三处
  同步删；`_runnable_str` 删后 `__getattr__` 剩裸 `raise AttributeError`；
  **CUDAGraphStat（L32-L37）保留**——站 10 的 `_determine_batch_execution_and_padding`
  直接构造它（must_keep 方法不能残缺），delete[5] 的实质（表格聚合+offloader）已删。
- **delete[6]（capture_model 观测/扩展段）**：profiler 装配、encoder cudagraph 两段、
  lock_workspace 删；『大到小捕获+窗口开关』主链不变。
- **delete[7]（worker 启动建议段）**：图池 estimate 对比、KV 内存建议长消息、
  maybe_save_startup_plan（ch17 启动计划域）、use_v2_model_runner 分支删；编排顺序
  warmup→kernel_warmup→capture→sampler 预热→lazy init→JIT monitor→freeze GC→sync
  check 完整（tests 断言此序）。
- **delete[8]（forward_context 观测/SP）**：batchsize 统计头部/finally 尾段与模块级
  全局量、DPMetadata 的 SP 两方法删；单 DP 路径行为不变（tests 覆盖 DP=1 注入）。
- **delete[9]（set_splitting_ops_for_v1 扩展分支）**：四条降级支（fuse_attn_quant→
  attn_fusion 切点、空 splitting_ops 降级、SP/fuse_gemm_comms 全图要求、DeepEP 禁图）
  连同调用点守卫删净、无悬空引用；主支『splitting_ops=_attention_ops+kv update 两算子』
  原文保留（tests 断言 13+2 与 ClassVar 不被改写）。
- **delete[10]（attention kv 域）**：maybe_calc_kv_scales 算子+注册+forward 调用、
  kv_sharing 校验/存储/两处守卫、mm_prefix_clamp、get_kv_cache_spec（ch13-16 域）删；
  query_quant 分支（F10 苗）**保留原文**。

## 范围裁剪（章界外域，SUBTRACTED+归属注记）

- **ch03**：VllmConfig 20+ 子配置与 post_init 主体（测试构造面经 `_for_tests` HOST
  SEAM 七键收窄）；config/utils hash 族。
- **ch17**：runner 的模型加载主体/execute_model 入口族/采样路径；Worker 其余；
  `_dummy_run` 体（dummy 批次装配——签名+docstring 逐字保留，体 SUBTRACTED，直接
  调用即越界 raise NotImplementedError）。
- **ch18**：_prepare_inputs/_preprocess 的输入收集主体（padding 段保留）；块表/
  CpuGpuBuffer 载体（测试 double 承载）。
- **ch20/ch21**：_build_attention_metadata 装配主体（block_table 闭包段保留）；后端
  选择/impl 构造（attn.impl/attn.attn_backend 由调用方/测试注入——ch21 接口）。
- **ch13-16**：get_kv_cache_spec/KV cache 初始化/connector。
- **ch22**：slot_mapping 的 Triton 数学（本章只留尾部 -1 pad 段）。
- **ch33**：spec decode/drafter 尾段、DBO（ubatch_slices 构造消费点）。
- **ch34**：DP 对齐全貌（coordinate_batch_across_dp HOST SEAM 结构洞——单 DP 部署
  不可达）。
- **m17 路线图脚注**：BreakableCUDAGraph/use_inductor_graph_partition 只留 env 门与
  分支删除注记（正文脚注素材，不进正典链）。

## Seam 清单（HOST SEAM，逐个登记）

`_host_seams.py` 承载章界外名字（同一可观察接口子集、逐个带 `# SOURCE:` 锚）：

- **logger**：`init_logger` + `*_once` 包装（vllm/logger.py）。
- **envs**：本章消费的 VLLM_* 标志类面（默认值=pin 的默认路径）。
- **current_platform**：平台谓词子集（is_cuda/is_cpu/.../opaque_attention_op/
  get_compile_backend/get_static_graph_wrapper_cls/graph_pool 族/
  set_additional_forward_context/apply_config_platform_defaults）。
  **dispatch_key 恒 "CPU"**——伴读张量全在 host CPU 上（direct_register_custom_op 的
  CPU 注册面）；"CUDA" 面是 vllm 容器域。
- **distributed**：graph_capture（nullcontext=非 CUDA no-op 支）/get_world_group/
  is_global_first_rank/get_pp_group；`coordinate_batch_across_dp` 结构洞
  （DP>1 不可达即 raise——ch34 域）。
- **UBatchSlices**：DBO 注记占位类型。
- **kernel_warmup**：no-op（kernel 调优域，调用位保留在编排序里）。
- **ir**：vllm.ir 的 IrOp 包装面——RMSNorm forward_native 的引用目标；**native 数学
  逐字来自 vllm/ir/ops/layernorm.py**（L10-L21/L44-L62），IrOp 注册/优先级机制域外。
- **weak_ref_tensor**：无 C++ 扩展时以 detach() 近似（同 storage/data_ptr 可观察；
  差异只在图池显存回收）。
- **set_graph_pool_id**：共享图池注入 no-op（分配器管道 ch14 域）。
- **breakable/kv-transfer 装饰器**：`is_breakable_cudagraph_enabled`（env 门）/
  `eager_break_during_capture`/`maybe_transfer_kv_layer` identity（默认关/无 connector
  路径即恒等装饰——本章 pin 的分支）。
- **启动尾面**：`activate_jit_monitor`（mode=None 即 no-op）/`set_random_seed`/
  `_apply_constrain_to_fx_strides_patch`/`maybe_disable_graph_partition`。
- **aiter/flashinfer 谓词**：O2 预设 lambda 引用面（非 ROCm/非 flashinfer 恒 False
  =pin 的非融合路径）。
- **小数学/类型面**：round_up（逐字）/rms_norm_batch_invariant（实验路径 raise）/
  EncoderOnlyAttentionSpec marker/GroupShape/QuantFP8 构造面（ch27 域，F10 苗分支文本
  保留）/NULL_BLOCK_ID（逐字常量 0）/record_function_or_nullcontext（默认关 profiling
  env 路径=nullcontext）。
- **vllm.* 模块别名**：`install_vllm_module_aliases` 把
  `vllm.compilation.cuda_graph` 代理到本包——`resolve_obj_by_qualname` 经规范 qualname
  解析 `current_platform.get_static_graph_wrapper_cls()` 时落回伴读本尊。

## Host 决策

1. **pydantic→dataclass**：真源 CompilationConfig/PassConfig 用 pydantic @config +
   field_validator；伴读版以纯 dataclass 承载同一字段集，三条 validator（str→enum ×2、
   dict→PassConfig）折进 `__post_init__` 头部，行为面等价（ch18 同款决策）。
2. **VllmConfig 切面**：真源 VllmConfig 需全量子配置（ch03 域）；伴读版只留 ch19
   消费的载体字段（model/parallel/quant/kernel/speculative/observability/
   compilation/optimization_level），`_for_tests` 是测试构造面（七键收窄，构造后
   __post_init__ 自跑）。测试的 `make_seam_vllm_config` 另以 SimpleNamespace 承载
   scheduler/parallel 两个子面，并 **mimic 真源 post_init 的档位落账**（mode 默认→
   VLLM_COMPILE、custom_ops 基础档、cudagraph_mode 默认→O2 预设）——镜像上游
   tests/v1/cudagraph/test_cudagraph_dispatch.py 的 `_create_vllm_config` 同款 mimic。
3. **scoped excerpt 方法**：`_prepare_inputs`/`_build_attention_metadata`/`_preprocess`
   /`execute_model`/`load_model`/`__init__` 是『段锚载体』——pinned 段逐字、
   章界外体 SUBTRACTED，方法不可独立运行（tests 对四件套三段手工驱动、对
   _get_slot_mappings 全文真调）；可运行单元（dispatcher/wrapper/算子三件/
   _determine_batch_execution_and_padding/capture_model/_warmup_and_capture/
   compile_or_warm_up_model）全真调。
4. **CUDAGraphWrapper 捕获/回放的 CUDA 段**：真 CUDA 路径以 `@pytest.mark.cuda` 门控
   （上游同款门控）——本机有 CUDA 时实跑『真捕获→真回放→data_ptr 断言』两测，
   无 CUDA 自动 skip；CPU 侧覆盖 __call__ 头（直通/建表项/DEBUG 断言）与 wrapper
   挂载/清空。
5. **CustomOp/Attention 构造契约**：真源要求在 `set_current_vllm_config` 上下文内
   构造（custom_op 的 dispatch 查询面）；tests 的 `_cfg_ctx` helper 承载该契约。

## 测试与真源行为对表（要点）

- **dispatcher 镜像上游**：`test_dispatcher_key_init_and_lookup` 五参数组 +
  `FULL_AND_PIECEWISE+CompilationMode.NONE` 构造即 assert（上游 L106-L112 的早退守卫
  同款）；`get_capture_descs` 降序/分组断言对上游 L208-L218。
- **enum 语义**：组合档 `.value` 是裸值元组 `(2, 0)`（python Enum 类体成员引用烘焙
  为原始值——plain Enum 非 IntEnum 成员），`decode_mode()` 经 `CUDAGraphMode(value[0])`
  还原；tests 按此断言。
- **worker 编排序**：capture 后紧跟 V1 sampler 预热（先 `_dummy_run(num_tokens=
  max_num_seqs, mode=NONE)` 再 `_dummy_sampler_run`）——NOTE『刻意在 capture 后』
  的原文保位；`enforce_eager` 是 capture 的闸（tests 断言 eager 跳过）。
- **compile_ranges 补尾**：区间**含端点**（Range inclusive）——`[9,32]` 被 compile
  size 20 覆盖则不补 32（tests 的负例断言）。

## lint 收口

```
$ python scripts/lint_fidelity.py instances/vllm/artifacts-v3/ch19-compile-capture
→ 无 BLOCKING（must_keep 52 符号全数核在；missing_source/invention 零）
```
