# ch18 精简版 impl-notes — 持久批次与固定地址（Part V：GPU 不等 Python）

- **Pin**：vLLM v0.27.1（`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`）。全部 `# SOURCE:` 行号
  对当前 pin 现核（`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。收工锚点
  全量清扫：**523 个** `# SOURCE: vllm/...:Lxxx` 标签逐个对 pin 树核过（文件存在 + 行区间在
  界内 + 区间内确含所指符号）；核心方法另做 ast 归一化**有序子序列核验**（把精简版方法体
  的每条语句与真实区间比对——只删不增，无杜撰语句；见 §验收判据）。
- **产物**：`implementation/`（包布局镜像源码树）+ `tests/test_persistent_batch.py`（44 测）。
- **跑法**：`cd instances/vllm/artifacts-v3/ch18-persistent-batch-fixed-addresses && python -m pytest
  tests/ -q` → **44 passed**（~1.7s）。host 可跑：真 torch / 真 numpy / 真 triton（kernel 定义
  逐字保留、host 上不 launch）；无 vllm 包、无 CUDA 上下文——worker 侧 CUDA 面以 HOST SEAM
  承载（HostEvent/HostCopyStream/脚本化前向，见 §Seam 清单）。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本包（HOST SEAM 例外见
  §Seam 清单——每个 seam 行内标注并在此登记）。98 个 `# SUBTRACTED:` 标记逐条挂
  dossier.subtraction_plan.delete[0..8] 批准项编号。
- **lint**：`python scripts/lint_fidelity.py <本章目录>` → **全部通过（无 BLOCKING）**；
  must_keep 65 符号经 linter `over_subtraction` 项全数核在。

## 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `output.py` | `vllm/v1/core/sched/output.py` | 差量协议三类载体全文：NewRequestData（新请求全量）/ CachedRequestData（老请求 diff + resumed 语义注释 L118-L121）/ SchedulerOutput（协议二分头 L193-L205，含 v0.27 新增的 new_block_ids_to_zero/kv_cache_block_copies/partial_tail_offloads 字段全保留）+ ScheduledEncoderInputStats。删除仅 from_request（scheduler 侧构造面，ch10/ch12 域）与 GrammarOutput（ch30 域） |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | 本章舞台类 GPUModelRunner 全切面（下表逐方法）；ExecuteModelState 十字段 NamedTuple 逐字 |
| `gpu_input_batch.py` | `vllm/v1/worker/gpu_input_batch.py` | 持久批次容器全文：CachedRequestState + InputBatch（token_ids_cpu R×L 布局、列式 CPU 镜像、采样参数列、add/remove/condense/swap_states/refresh_metadata、_make_sampling_metadata、异步三件 update_async_*、property 族） |
| `block_table.py` | `vllm/v1/worker/block_table.py` | BlockTable/MultiGroupBlockTable 全方法 + Triton kernel `_compute_slot_mapping_kernel` **逐字**（delete[8] 明示 kernel 本体不动、hybrid 细分与 CP 局部量原样保留）；仅加 CPU host 镜像 seam |
| `utils.py` | `vllm/v1/utils.py` | CpuGpuBuffer 全文逐字（m05 本章标题之一）+ copy_slice + record_function_or_nullcontext |
| `outputs.py` | `vllm/v1/outputs.py` | SamplerOutput / ModelRunnerOutput（sample_tokens 组装与 EMPTY 单例）；Async 包裹协议归 ch12 |
| `metadata.py` | `vllm/v1/sample/metadata.py` | SamplingMetadata 纯 dataclass 逐字（refresh_metadata 产出型） |
| `logits_processor/interface.py` | `vllm/v1/sample/logits_processor/interface.py` | MoveDirectionality / BatchUpdate / LogitsProcessor ABC |
| `logits_processor/state.py` | `vllm/v1/sample/logits_processor/state.py` | BatchUpdateBuilder 全文逐字（removed 恒降序/pop_removed/peek_removed）+ LogitsProcessors |
| `logits_processor/__init__.py` | `vllm/v1/sample/logits_processor/__init__.py` | re-export + build_logitsprocs（pooling 短路支逐字；BUILTIN 链归 ch30，空集承载同调用面） |
| `ngram_proposer_gpu.py` | `vllm/v1/spec_decode/ngram_proposer_gpu.py` | update_scheduler_for_invalid_drafts **全文逐字**（must_keep；可变性裁决的就地裁剪实现）；NgramProposerGPU drafter 类归 ch33 |
| `math_utils.py` / `torch_utils.py` | `vllm/utils/math_utils.py` / `.../torch_utils.py` | cdiv / PIN_MEMORY（消费面） |
| `_host_seams.py` | （跨域缝合，见 §Seam 清单） | HOST SEAM 登记处 |

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因；核心行）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `SchedulerOutput` 全类 | vllm/v1/core/sched/output.py:L193-L283 | 逐字（协议字段全保留，含 v0.27 新增字段） | must_keep（m01）；delete 未列任何本类删除项 |
| `CachedRequestData` 全类 | output.py:L116-L181 | 逐字（resumed 语义注释 L118-L121 原文） | must_keep（m01/m11） |
| `NewRequestData` | output.py:L35-L112 | 逐字 minus from_request（L50-L69） | must_keep；删除项=ch10/ch12 域 scheduler 侧构造面 |
| `execute_model` | gpu_model_runner.py:L4166-L4535 | 逐字编排骨架（双向断言/空拍早退/num_scheduled 装配/_prepare_inputs 调用位/打包暂存）minus ngram replace 段保留、前向段→ENGINE SEAM | must_keep（m10 入口）；前向深水 ch17 边界 |
| `_update_states` | gpu_model_runner.py:L1192-L1566 | 逐字主干（finished 出缓存出批次/unscheduled 出批次留缓存/新请求建 CachedRequestState/老请求 append·resumed 替换/reqs_to_add 落位/condense/_may_reorder_batch/refresh_metadata）minus delete[0][2][4][5][6][7] 各特性支 | must_keep（m02 主方法） |
| `InputBatch.add_request` | gpu_input_batch.py:L350-L501 | 逐字（token 写行/三列记账/block_table.add_row/采样参数列装填）minus embeds 分支（delete[1]）、pooling 支（delete[2]）、LoRA（delete[3]） | must_keep（m04） |
| `InputBatch._register_add_request` | gpu_input_batch.py:L324-L348 | **逐字** | must_keep（m03 slot 复用机制核心） |
| `InputBatch.remove_request` | gpu_input_batch.py:L530-L584 | 逐字（打洞+解绑+clear_row，must-be-followed-by-condense docstring）minus LoRA/pooling 清理 | must_keep（m03） |
| `InputBatch.condense` | gpu_input_batch.py:L708-L838 | 逐字（降序 removed 双指针、尾部滑入、只拷活跃前缀） | must_keep（m03）；非删任何逻辑 |
| `InputBatch.swap_states` | gpu_input_batch.py:L586-L701 | 逐字（活跃前缀交换 + moved 登记） | must_keep（m12） |
| `BatchUpdateBuilder` 全类 | vllm/v1/sample/logits_processor/state.py:L18-L195 | **全文逐字**（无删除项） | must_keep ×3（m03 索引真相源） |
| `_prepare_inputs` | gpu_model_runner.py:L1960-L2282 | 逐字（commit_block_table 先行/np.repeat/_get_cumsum_and_arange/positions/token_indices/index_select/query_start_loc pad/optimistic_seq_lens/discard mask/GPU positions·seq_lens/compute_slot_mapping/_prepare_input_ids）minus mrope/xdrope（delete[0]）、embeds 填充（delete[1]）、spec 门控（delete[4][6]）、spec else 支（delete[6]④）、LoRA 热切换（delete[3]） | must_keep（m06/m07/m08） |
| `_prepare_input_ids` | gpu_model_runner.py:L1784-L1913 | 逐字（固定地址前缀上载/common-case 单 slice 直拷/scatter 兜底/异步 prev_sampled_token_ids 消费）minus embeds 守卫支（delete[1]） | must_keep（m09 异步写回消费点） |
| `_get_cumsum_and_arange` | gpu_model_runner.py:L1743-L1767 | **逐字**（源码注释算例 [2,5,3]→[2,7,10]/[0,1,0,1,2,3,4,0,1,2] 保留） | must_keep（m06） |
| `CpuGpuBuffer` 全类 | vllm/v1/utils.py:L110-L149 | **全文逐字**（cpu pinned+gpu+np 三视图、copy_to_gpu(n) 活跃前缀、bfloat16 拒 numpy） | must_keep（m05 本章标题之一） |
| `GPUModelRunner.__init__` 持久缓冲块 | gpu_model_runner.py:L763-L810 | 逐字（'Persistent buffers for CUDA graphs' 首行注释 + input_ids/positions/query_start_loc/seq_lens/optimistic_seq_lens_cpu/num_computed_tokens/req_indices/prev_positions/num_scheduled_tokens/is_token_ids/discard_request_mask 全量分配）minus mrope/xdrope 特例（delete[0]）、inputs_embeds 缓冲（delete[1]）、dcp 守卫（delete[8]） | must_keep（m05/m14） |
| `_bookkeeping_sync` | gpu_model_runner.py:L3723-L3862 | 逐字（同步 D2H/写回 token_ids_cpu 行/num_tokens_no_spec 前移/output_token_ids 增长 + 异步分支 prev_sampled_token_ids 缓存/-1 占位/prev_req_id_to_index 快照） | must_keep（m09 写回闭环） |
| `sample_tokens` | gpu_model_runner.py:L4552-L4840 | 逐字（@torch.inference_mode/解包暂存态/_sample/_bookkeeping_sync/ModelRunnerOutput 组装/同步直返）minus 空槽早退、draft、connector（ch16/33 域）与 Async 包裹（ch12） | must_keep；两段式契约后半 |
| `synchronize_input_prep` | gpu_model_runner.py:L3864-L3877 | **逐字**（pinned buffer 防踩：等上拍 prepare_inputs_event 再 record） | must_keep（m13） |
| `_may_reorder_batch` | gpu_model_runner.py:L1115-L1138 | 逐字（attention backend 重排钩子；reorder 本体是 HOST SEAM 空返） | must_keep（m12） |
| `BlockTable.append_row/add_row/clear_row/move_row/swap_row` | block_table.py:L138-L180 | 逐字（差量追加记账/整行重写/清行/搬行/换行） | must_keep ×4 |
| `BlockTable.commit_block_table` | block_table.py:L213-L214 | **逐字**（活跃行前缀拷贝——m08 先行拷贝） | must_keep |
| `BlockTable.compute_slot_mapping` + kernel | block_table.py:L182-L211 / L379-L442 | 派发逐字 + kernel 本体**逐字**；CPU host 走 seam 镜像 | must_keep（slot 数学 → ch22，入口保留） |
| `update_scheduler_for_invalid_drafts` | ngram_proposer_gpu.py:L475-L515 | **全文逐字** | must_keep（m10 worker 改写调度器输出的罕见点） |
| `LateInteractionRunner` 5 处调用位 | gpu_model_runner.py:L967/L976/L1207-L1209/L1310/L1645 | 调用位整组保留；打分本体 seam no-op | delete[6]【不删（防悬空）】明示 |

## 删除账本（dossier.subtraction_plan.delete[0..8] 落点）

| delete | 内容 | 落点（# SUBTRACTED 标记） |
|---|---|---|
| [0] | M-RoPE/XD-RoPE 四方法+全部守卫调用点 | gpu_model_runner.py：缓冲分配、_get_positions 两 if、新请求初始化、streaming 内、_prepare_inputs 内、GPU 拷贝漂移修正、_preprocess/padding 辅助、multimodal-pruning 守卫 |
| [1] | prompt_embeds 路径（保留 req_prompt_embeds 空 dict、is_token_ids 缓冲与 L387-L388 核心写回） | gpu_model_runner.py 缓冲+三守卫支+收集填充；gpu_input_batch.py L383-L386 分支 |
| [2] | pooling 早退/装填（保留两 dict 与 getter 族——读端在保留区） | gpu_input_batch.py L120/L473-L481/L560-L563/L660-L663/L792-L795/L843-L847；gpu_model_runner.py L1286-L1293 |
| [3] | LoRA 整链（含 make_lora_inputs 方法与其唯一调用方 runner 热切换块） | gpu_input_batch.py L259-L262/L488-L499/L550-L558/L655-L658/L788-L790/L1005-L1028；gpu_model_runner.py L2269-L2277 |
| [4] | async spec 乐观纠偏链（保留 optimistic_seq_lens_cpu/discard_request_mask 无条件装配、else 默认填充/拷贝去缩进为无条件） | gpu_model_runner.py prev_num_draft_len 记账/恢复段、deferred 闭包、num_accepted 事件门控、update_num_computed_tokens_for_batch_change |
| [5] | ngram-GPU drafter 镜像（保留可变性裁决协议：replace 段、guard+调用、is_ngram_gpu/ngram_gpu_new_reqs 局部量） | gpu_model_runner.py L610-L626 elif 支、L1522-L1532 增量调用 |
| [6] | 可选特性守卫块（新块置零/CoW/encoder 释放/mamba-GDN/spec else 支/kv_sharing_fast_prefill；**不删** thinking budget holder/replayssm/logits_processing_needs_token_ids/late_interaction_runner——删写端必留读端悬空） | gpu_model_runner.py L1219-L1231、L2143-L2150、L2242-L2267、L852-L858、L2466-L2470、L2926 起 |
| [7] | PP 非末 rank 回填与广播（token_ids_cpu 增量、is_last_rank 支与 elif 对齐臂） | gpu_model_runner.py L1408-L1439、L1478-L1499 |
| [8] | DCP 上下文并行链（gpu_model_runner 侧两处成对删；block_table 侧 CP 局部量与 hybrid 细分**原样保留**） | gpu_model_runner.py L791-L795、L2451-L2462 |

## Seam 清单（HOST/ENGINE SEAM——真实代码之外唯一允许的承载，每个行内标注）

1. **`_host_seams.HostEvent` / `HostCopyStream`**（CUDA 契约位）：CPU host 无
   torch.cuda.Event(blocking=True)/Stream。契约语义逐条保留：未 record 的事件
   synchronize() 立即返回；record()=入队未完成；synchronize()=阻塞至完成——host 侧无真
   DMA、等待即刻满足（完成时刻是同步的）；record/synchronize 的**调用顺序**（m13 防踩
   协议）以计数器观测（test_synchronize_input_prep 断言「先等后录」）。
2. **前向 ENGINE SEAM**（ch17 边界、ch12 同款）：真实 L4255-L4514 的
   cascade/DBO/cudagraph/ubatch/attention-metadata/前向段归 ch19/ch21/ch22/ch34；精简版以
   `enqueue_logits()` 测试钩子 + `_seam_model_forward()` 脚本化 logits 行承载
   （每请求一行 [vocab] 张量），`hidden_states/sample_hidden_states/aux_hidden_states/
   ec_connector_output/cudagraph_stats/slot_mappings/kv_connector_output` 以 None 绑定
   ——ch17 立下的契约行为不变法。两段式协议本身（入口断言/打包暂存/return None/解包即清）
   全真。
3. **`Sampler` greedy 支**：`logits.argmax(dim=-1).view(-1)`（sampler.py:L241 逐字）；
   penalties/temperature/topk-topp 支归 ch08（本章测试全 greedy）。
4. **`reorder_batch_to_split_decodes_and_prefills` 空返**（ch21/22 域）：_may_reorder_batch
   调用位逐字保留；四区重排本体以 `return False`（=未重排）承载——单注意力组配置下真实
   亦不重排。
5. **`BlockTable._compute_slot_mapping_host`**（CPU 镜像）：kernel L397-L442 的逐行 numpy
   镜像（同一恒等式 slot = 块号×block_size+块内偏移、同一 PAD 尾、同一 CP 变量名与单卡退化）；
   CUDA 设备分支 kernel 派发逐字保留（容器内真跑）。ch13 差分电池同款手法。
6. **`LateInteractionRunner` / `KVBlockZeroer` / `ThinkingBudgetStateHolder` / `build_logitsprocs`**：
   调用面/构造面逐字保留、域外本体 no-op（ch29/ch14/ch30 域）——delete[2][6] 防悬空条款的
   承载侧。
7. **配置/环境占位**：`PIN_MEMORY=False`（host 无 pinned；行为分支无涉，只影响拷贝速度）、
   envs 三剖面开关默认 False、distributed group 面（单机单卡 world_size=1 / is_last_rank=True，
   即精简配置的真实取值）、CacheConfig.DEFAULT_BLOCK_SIZE=16、SamplingParams/PoolingParams
   属性面（全量归 ch08）。
8. **注解面**：章界外类型名（SpecDecodeMetadata/LogprobsLists/LogprobsTensors/
   IntermediateTensors/AsyncModelRunnerOutput/CommonAttentionMetadata/ECConnectorOutput/
   CUDAGraphStat/GrammarOutput）以真实名出现在注解位（`from __future__ import annotations`
   下永不求值）；outputs.py 里被 ch16/ch19/ch08 域收走的类型以 `= object` 占位（字段恒 None，
   行为无涉）。

## 已知偏差（非 seam 的显式记录）

- `_update_states_after_model_execute` 保留签名、方法体以 `return None` 承载（mamba 对齐
  GPU 后处理与 num_accepted 记账归 ch14/ch33；精简配置 spec=None 下真实首行守卫即返回，
  行为等价）。
- `sample_tokens` 异步支（use_async_scheduling=True）返回同步形态 ModelRunnerOutput——
  AsyncGPUModelRunnerOutput 包裹协议 ch12 已全文立；本章保留 _bookkeeping_sync 的 async
  分支本体（真 token 留 GPU/-1 占位/prev_req_id_to_index 快照都在那里发生），D2H 重叠包裹
  不在面内。测试以 async_scheduling=True 配置断言占位语义（第 884/912 行两测）。
- 跨拍地址断言（m14）：`test_fixed_addresses_across_steps` 以 `data_ptr()` 断言
  input_ids(cpu/gpu)/positions/query_start_loc/seq_lens/token_ids_cpu 共 6 个缓冲跨拍不变
  ——CPU 张量地址恒定与 CUDA graph 的 data_ptr 断言（vllm/compilation/cuda_graph.py:
  L346-L355）同一不变量；GPU 端容器验证归 ch19。
