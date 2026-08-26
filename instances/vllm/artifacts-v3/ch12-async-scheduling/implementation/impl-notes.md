# ch12 异步调度 — impl-notes（只做减法精简版）

对应真实源码 pin **vLLM v0.27.1 (6e448d0ea)**，行号全部本日现核（2026-08-27，
`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。
运行：`cd instances/vllm/artifacts-v3/ch12-async-scheduling && python -m pytest tests/ -q`
（39 passed；纯 host 单元/契约测试，不 import vllm、不触 CUDA——worker 侧 CUDA 面
以 HOST SEAM 承载，见下）。

## 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `engine.py` | `vllm/v1/engine/__init__.py` | FinishReason / EngineCoreOutput(s) 四字段版 |
| `core.py` | `vllm/v1/engine/core.py` | EngineCore：批队列装配/step_fn 绑定/同步 step 对照/post_step/step_with_batch_queue 两态循环/has_work |
| `core_client.py` | `vllm/v1/engine/core_client.py` | InprocClient（离线门面直调 step_fn） |
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | 基类：early-stop 剪枝/追赶公式占位项/_update_after_schedule 乐观推进/_preempt_request async 账单/update_from_output 热循环+spec 拒绝回扣/get_grammar_bitmask |
| `async_scheduler.py` | `vllm/v1/core/sched/async_scheduler.py` | AsyncScheduler 两个覆写（占位 +1 / 占位 -1+块转正）——全文件 70 行只剩 V2 分支删除 |
| `output.py` | `vllm/v1/core/sched/output.py` | SchedulerOutput 的 async 标志对 + num_spec_tokens_to_schedule + GrammarOutput + New/CachedRequestData |
| `request.py` | `vllm/v1/request.py` | 四个异步计数器 + use_structured_output + spec_token_ids |
| `scheduler_config.py` | `vllm/config/scheduler.py` | async_scheduling 配置位 + get_scheduler_cls 换型 |
| `vllm_config.py` | `vllm/config/vllm.py` | max_concurrent_batches 深度仲裁 + None→True 默认仲裁（含显式 True 硬失败） |
| `executor_factory.py` | `vllm/v1/executor/executor_factory.py` | 后端→executor 映射（仲裁链输入） |
| `uniproc_executor.py` | `vllm/v1/executor/uniproc_executor.py` | AsyncOutputFuture 逐字 + collective_rpc non_block 路径 + supports_async_scheduling |
| `gpu_worker.py` | `vllm/v1/worker/gpu_worker.py` | @with_gpu_sync_check 包裹位（tripwire 落点） |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | AsyncGPUModelRunnerOutput / _prepare_input_ids / _compute_prev_positions / 乐观纠错群 / _bookkeeping_sync / sample_tokens 切面 |
| `gpu_input_batch.py` | `vllm/v1/worker/gpu_input_batch.py` | 影子字段 prev_sampled_token_ids / prev_req_id_to_index + 写回三件 |
| `gpu_sync_debug.py` | `vllm/utils/gpu_sync_debug.py` | tripwire 门 + 非 CUDA no-op 分支（即 host 语义） |
| `outputs.py` | `vllm/v1/outputs.py` | ModelRunnerOutput / AsyncModelRunnerOutput / EMPTY |
| `kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` | 消费面：allocate_slots/free/cache_blocks（块池内景归 ch13/ch15） |
| `request_queue.py` | `vllm/v1/core/sched/request_queue.py` | FCFS 队列（PRIORITY 归 ch10/ch11） |
| `utils.py` | `vllm/v1/core/sched/utils.py` | check_stop 四判（第五判重复检测归 ch11）+ remove_all |
| `interface.py` | `vllm/v1/core/sched/interface.py` | SchedulerInterface 方法面 |
| `logger.py` | `vllm/logger.py` | HOST SEAM：init_logger + once 去重 |

## 1:1 Source Map（关键段；改动=减法或 seam，原因=批准条/章节边界）

| 精简版 | 真实源 | 改动 | 原因 |
|---|---|---|---|
| `core.py step_with_batch_queue` | `vllm/v1/engine/core.py:L625-L739` | 两态循环逐字；删 is_ec_consumer 分支（L658-L659）、is_pooling_model 条件（L661）、观测上下文（L654/L697-L699/L714）、deferred 内层 spec 块压缩为存证注释（L722-L730）、throttle_prefills 实参 | dossier.delete 第 1/2/3/8/5 条 |
| `core.py __init__` 批队列段 | `vllm/v1/engine/core.py:L206-L234` | 逐字（batch_queue_size=max_concurrent_batches、>1 建 deque、step_fn 静态绑定、async_scheduling 落字段） | m2/m3 主线 |
| `core.py step`（同步对照） | `vllm/v1/engine/core.py:L584-L614` | 串行脊柱保留；删观测上下文与 throttle 实参 | 第 3/5 条 |
| `core.py post_step` | `vllm/v1/engine/core.py:L616-L623` | 逐字（三条件短路） | m17 |
| `core.py has_work` | `vllm/v1/engine/core.py:L1365-L1371` | 逐字（bool(batch_queue) 保活） | m20 |
| `async_scheduler.py _update_after_schedule` | `vllm/v1/core/sched/async_scheduler.py:L19-L49` | 逐字；删 V2 的 next_decode_eligible_step（L46-L49）与 pp_size 字段（L17） | 第 4 条 |
| `async_scheduler.py _update_request_with_output` | `vllm/v1/core/sched/async_scheduler.py:L51-L70` | 逐字（stale 不扣 + cache_blocks(computed−ph)） | m7 |
| `scheduler.py` RUNNING 循环 early-stop | `vllm/v1/core/sched/scheduler.py:L488-L502` | 逐字恢复（ch11 曾删给本章） | m9 本章主线 |
| `scheduler.py` 追赶公式 | `vllm/v1/core/sched/scheduler.py:L516-L532` | 逐字（占位项 + 双钳制） | m8 |
| `scheduler.py _update_after_schedule` | `vllm/v1/core/sched/scheduler.py:L1317-L1365` | 乐观推进三件套 + has_structured_output_requests 置位逐字；删 defer_block_free/routed 快照 | 第 4 条 + 观测 |
| `scheduler.py _preempt_request` async 账单 | `vllm/v1/core/sched/scheduler.py:L1274-L1315` | L1297-L1308 逐字（stale=assign、占位清零）；encoder free 删 | 第 2 条（ch11 删项同源） |
| `scheduler.py update_from_output` 热循环 | `vllm/v1/core/sched/scheduler.py:L1670-L2055` | 扣在途+stale 锁步 drain（L1736-L1743）与 spec 拒绝回扣（L1766-L1791）逐字恢复；删 logprobs/routed/connector/perf/encoder | 观测/子系统面 |
| `scheduler.py get_grammar_bitmask` | `vllm/v1/core/sched/scheduler.py:L1646-L1668` | 逐字（manager 为 ch30 seam：全 1 位掩码） | m14 |
| `vllm_config.py max_concurrent_batches` | `vllm/config/vllm.py:L539-L550` | 逐字（v0.27.1 唯一出处；v0.21 在 executor 侧——迁移陷阱） | m2 |
| `vllm_config.py check_and_set_default_async_scheduling` | `vllm/config/vllm.py:L1057-L1143` | 显式 True 硬失败（L1064-L1094）+ None→True 五类降级（L1095-L1143）逐字；ROCm 判定 HOST SEAM 恒 False | m1 |
| `scheduler_config.py get_scheduler_cls` | `vllm/config/scheduler.py:L170-L178` | 逐字（scheduler_cls 自定义分支删） | 装配面 |
| `uniproc_executor.py AsyncOutputFuture` | `vllm/v1/executor/uniproc_executor.py:L26-L42` | 逐字 | m13 |
| `gpu_model_runner.py AsyncGPUModelRunnerOutput` | `vllm/v1/worker/gpu_model_runner.py:L259-L350` | 构造即拷贝+record / get_output 逐字；logprobs/routed/ep-fault/spec 解析删 | 第 6/7 条 |
| `gpu_model_runner.py _prepare_input_ids` | `vllm/v1/worker/gpu_model_runner.py:L1784-L1913` | 正常拍/单 slice/scatter 三路逐字；删 prompt_embeds 补拷与 spec 二段 scatter | 第 6/7 条 |
| `gpu_model_runner.py _bookkeeping_sync` | `vllm/v1/worker/gpu_model_runner.py:L3723-L3862` | async 分支（L3797-L3813）+ 写回循环（L3815-L3846）逐字；sync 对照分支留 _to_list seam；spec RejectionSampler 删 | 第 6 条 |
| `gpu_model_runner.py` 乐观纠错群 | `vllm/v1/worker/gpu_model_runner.py:L2081-L2105` | 逐字（ENGINE SEAM：从 _prepare_inputs 内联块抽出两方法以便单测，控制流不动） | m18 |
| `gpu_input_batch.py` 影子字段 | `vllm/v1/worker/gpu_input_batch.py:L309-L316` | 逐字；set_async_sampled_token_ids（L1030-L1045）留调用面 | m10 |
| `gpu_worker.py` tripwire 位 | `vllm/v1/worker/gpu_worker.py:L846-L848/L1010-L1021` | @with_gpu_sync_check 包裹逐字；warmup 翻闸以注释存证 | m19 |

## HOST SEAM / ENGINE SEAM 登记（跨章边界，非减法）

- **HostEvent / HostCopyStream**（gpu_model_runner.py）：CPU host 无 CUDA
  stream/event；threading.Event 站 `torch.cuda.Event(blocking=True)` 的契约位。
  `record()`=入队未完成、`synchronize()`=阻塞、测试 `release_async_copies()`=
  模拟 D2H DMA 完成——e2e 测试由此显式驱动『拷贝完成时刻』，证明重叠窗口存在。
- **脚本化前向**（`enqueue_logits`/`_seam_logits`，ch17 边界）：每步一个
  {req_id: logits 行} 字典；greedy argmax（sampler.py:L239-L241 逐字）采出
  可预测 token。不在环内伪造 forward。
- **StructuredOutputManager**（ch30 边界）：grammar_bitmask 恒全 1（=无约束），
  数值不变；掩码算法归 ch30。
- **CpuGpuBuffer / InputBatch**（ch18 边界）：.cpu/.gpu/copy_to_gpu 消费面；
  容器内景（condense/block table）归 ch18。
- **KVCacheManager**（ch13/ch15 边界）：块池朴素分配；前缀命中恒 0；
  `cache_blocks_calls` 观测账供 m7 测试对账。
- **run_method / logger / gpu_sync_debug 非 CUDA 分支**：单进程反射直调 /
  标准日志+once / 真实文件的 `else` no-op 分支（gpu_sync_debug.py:L158-L165）
  即 host 语义——CUDA 分支在容器内由真 vllm 执行。
- **VllmConfig / configs**（ch03 边界）：裸字段镜像仲裁输入面；仲裁逻辑逐字。
- **乐观纠错两方法**（`_compute_optimistic_seq_lens`/`_compute_discard_request_mask`）
  从 `_prepare_inputs` 内联块抽出（控制流逐字）以便单测——抽出而非改写。

## 账本口径备注（writer 可用）

- e2e（prompt=2、max_tokens=2、无 spec）四拍后终态：`computed=3`、`ph=0`、
  `computed−ph=3` = 有确认 KV 的位置数（t0,t1 两 prompt 位 + t7 decode 位）；
  t9 是位置 2 的采样输出，early-stop 后位置 3 不再前向。
- `update_from_output` 的 finished_req_ids 清账拍会让 `has_requests()` 多真一拍
  （基类 `has_unfinished or has_finished`）——忙循环靠 `bool(batch_queue)` 与
  该拍自然收敛，测试 `test_has_work_kept_alive_by_inflight_batch` 复现。
- spec 拒绝回扣测试的占位语义：`ph = 1 bonus + 3 spec = 4`，回扣 3 + delivery
  扣 1 → 0（对齐 `_update_after_schedule` 的 `num_sampled + spec` 灌值）。

## 减法批准对照（dossier.subtraction_plan.delete 八条 → 落点）

1. is_ec_consumer（core.py:L658-L659）→ `core.py` 上半段 SUBTRACTED 注释。
2. is_pooling_model 快路（L661 条件）→ 保留 `not model_executed` 支。
3. 观测上下文（capture_iteration_details/log_error_detail/_attach）→ step 与
   step_with_batch_queue 的 with 包装拆除、代码体内联。
4. V2 runner 分支 → async_scheduler L46-L49 删；Scheduler.use_v2_model_runner
   恒 False；VllmConfig.use_v2_model_runner 压平为字段（真实为 env property）。
5. DP/PP → _should_throttle_prefills 参数与覆写删（schedule() 无参）；PP 广播/
   回传、_pp_* 、use_pp 分支、_pp_send_work 均删（engine/gpu_model_runner/
   gpu_worker 各就地注释）。
6. spec 深层 → 二段 scatter/update_num_computed_tokens_for_batch_change/
   propose_draft_token_ids 闭包/valid_sampled_token_count_gpu/
   _copy_draft_token_ids_to_cpu/_update_states_after_model_execute/Rejection
   Sampler 删（调用点与 -1 占位语义保留；cur_num_spec_tokens=0 不影响主线）。
7. worker 正交分支 → routed_experts/mm/encoder/LoRA/mamba/cascade/
   enable_prompt_embeds 补拷删（_bookkeeping_sync/_prepare_input_ids 就地注释）。
8. deferred 内层块（L722-L730）→ 压缩为存证注释，方法名
   take_draft_token_ids / update_draft_token_ids_in_output 保留在注释中。

## 与 must_keep 的对账

dossier.subtraction_plan.must_keep 全部 42 个符号均在精简版中出现
（lint_fidelity 的 over_subtraction 检查绿）。其中 `optimistic_seq_lens_cpu`/
`discard_request_mask`/`prev_req_id_to_index` 等 worker 侧符号随 m10/m11/m18
主线保留；`take_draft_token_ids`（executor 方法 + post_step 调用位 + worker
恒 None 存根）与 `check_for_draft_tokens`（EngineCore 字段 + 双调用位）齐备。
