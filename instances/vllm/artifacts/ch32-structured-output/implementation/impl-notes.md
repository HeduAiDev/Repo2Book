# ch32 精简版实现说明（只做减法 / pin ad7125a4 / v0.21.0）

精简版忠实镜像 vLLM v1 结构化输出子系统在**调度侧装配掩码 → worker 侧把掩码落到
logits** 这一整条链路：`get_grammar_bitmask` 交棒点（ch31 已建）→ 调度侧门控与行序
不变式 → `StructuredOutputManager.grammar_bitmask`（并行/串行两条填充路径 +
投机位置的推进/回滚）→ 两条并存的 worker 落地路径（默认的 xgrammar 库函数 vs.
opt-in 的自写 Triton kernel）→ 投机解码耦合（装配前过滤 + 延后采样链）。与真实
vLLM **同名、同结构、同控制流**，只删不增。所有删除点都带 `# SUBTRACTED:` 注释
并标注原 `vllm/...:Lxxx`。

## 路线前提（写在最前，因为它决定了哪条 worker 路径是主线）

`VLLM_USE_V2_MODEL_RUNNER` 默认 `False`（`vllm/envs.py:251`，`:1711-1713` 取
env `"0"`），`gpu_worker.py:316-330` 据此二选一构造 `GPUModelRunnerV1` /
`GPUModelRunnerV2`。**默认部署**走 `structured_output/utils.py:apply_grammar_bitmask`
（本章 `utils.py`）→ `xgr.apply_token_bitmask_inplace`；**只有显式打开开关**才走
`worker/gpu/structured_outputs.py:StructuredOutputsWorker`（本章
`structured_outputs.py`）→ 自写的 `_apply_grammar_bitmask_kernel`。两条路径在本
精简版里都保留、都可运行——`test_legacy_path.py` 覆盖默认路径，`test_worker_gpu.py`
覆盖 opt-in 路径的真实 GPU + Triton 执行。

## 可运行性

本轮环境**有真实 CUDA GPU + Triton**（与本书其它 vLLM 章节假定的"host 无
CUDA"不同，已用 `nvidia-smi` / `torch.cuda.is_available()` 现场核实），因此
`test_worker_gpu.py` / `test_draft_writeback.py` 里的 `@triton.jit` kernel 与
`torch.cuda.Stream/Event` 路径**在 host 上被真实执行并通过**，不是仅语法检查。
`xgrammar` 库本身未安装（`ModuleNotFoundError`），`backend_xgrammar.py` /
`utils.py` 用 `try: import xgrammar as xgr / except ImportError: xgr = None`
顶层导入，测试里用轻量 Fake 对象替身 monkeypatch 模块级 `xgr` 名字——被替身的只是
外部库函数，vLLM 自身的重排/映射/门控逻辑一字不改地被真实执行。

## 验收判据

把真实 vLLM 删掉所有 `# SUBTRACTED:` 标注的分支，应 ≈ 得到本精简版。删除项严格
限于 `dossier.subtraction_plan.delete` 批准范围。`must_keep` 的全部符号原样保留
（用 `python3 scripts/lint_fidelity.py` 校验）。

## 对 dossier 建议删除项的一处自我订正

`subtraction_plan.delete` 第 7 条建议 `step_with_batch_queue` 只删 "EC consumer /
队列长度调度等无关分支"。实现中发现**队列长度早退分支**
（`if model_executed and len(batch_queue) < self.batch_queue_size and not
batch_queue[-1][0].done(): return None, True`，`core.py` 原函数内）不能删——它
正是让 `deferred_scheduler_output` 真正跨调用延迟一轮的机制本身：没有它，本轮
刚 append 的 future 会在同一次调用里被立刻 `pop()` 掉，队列深度永远是 0，
"延后到下一轮" 就无从谈起（`test_engine_core.py` 的
`test_deferred_chain_waits_for_draft_tokens_before_computing_bitmask` 两轮测试
如果删掉这段会在第二轮 `pop from an empty deque`）。故实现里**保留了**这段早退
分支，只删了 EC consumer / pooling model 分支与可观测性外壳。已按 implementer
契约"不确定就保留，不擅自扩大删除范围"处理；如需改动请 Lead 复核。

## 文件与真实源码对应

| 精简版文件 | 真实源码 | 说明 |
|---|---|---|
| `backend_types.py` | `vllm/v1/structured_output/backend_types.py` | 六方法契约 ABC；只留签名，实现属 ch31 |
| `backend_xgrammar.py` | `vllm/v1/structured_output/backend_xgrammar.py` | 只留 `allocate_token_bitmask` + `compile_grammar` 尾段（`max_rollback_tokens` 钉死为 `num_speculative_tokens`），五个编译分支已删（ch31 范围） |
| `sampling_params.py` | `vllm/sampling_params.py` | `StructuredOutputsParams` 字段容器 + 精简版 `SamplingParams`（六选一互斥校验/后端选择阶梯属 ch31，已删） |
| `so_request.py` | `vllm/v1/structured_output/request.py` | `StructuredOutputRequest`，本章相对 ch31 版本**额外保留** `reasoning_ended`/`reasoner`/`reasoning_parser_kwargs`（m16 需要） |
| `request.py` | `vllm/v1/request.py` | `Request` 精简到 `num_computed_tokens`/`num_tokens`/`num_output_placeholders`/`spec_token_ids`/`is_prefill_chunk`/`all_token_ids`/`prompt_token_ids`/`is_finished`（恒 False，简化） |
| `output.py` | `vllm/v1/core/sched/output.py`, `vllm/v1/outputs.py` | `SchedulerOutput`（精简字段）、`GrammarOutput`、`DraftTokenIds` |
| `structured_output_manager.py` | `vllm/v1/structured_output/__init__.py` | `StructuredOutputManager`：`grammar_bitmask`（并行+串行分支全保留）、`_fill_bitmasks`、`_async_submit_fill_bitmask`、`should_fill_bitmask`、`should_advance`、`_get_reasoner`；`__init__` 签名从 `vllm_config` 简化为直接接受 `max_num_seqs`/`max_num_spec_tokens`（同一简化已见于 `XgrammarBackend`） |
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | `Scheduler`：`_update_after_schedule`（门控二行）、`get_grammar_bitmask`（交棒点）、`update_draft_token_ids_in_output`（装配前投机过滤+`-1`补齐）、`make_spec_decoding_stats`（`num_invalid_spec_tokens` 消费点）；`SpecDecodingStats` 精简为两个累计字段 |
| `async_scheduler.py` | `vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler._update_after_schedule`：置位 `pending_structured_output_tokens` |
| `utils.py` | `vllm/v1/structured_output/utils.py` | **默认路径**：`apply_grammar_bitmask`（GPU 主路径，CPU 兜底分支已删） |
| `input_batch.py` | `vllm/v1/worker/gpu/input_batch.py` | `InputBatch` 精简到 `req_ids`/`cu_num_logits_np`/`logits_indices`/`has_structured_output_reqs` |
| `buffer_utils.py` | `vllm/v1/worker/gpu/buffer_utils.py` | 只留 `async_copy_to_gpu`（`UvaBuffer` 与掩码搬运无关，已删） |
| `structured_outputs.py` | `vllm/v1/worker/gpu/structured_outputs.py` | **opt-in 路径**：`StructuredOutputsWorker`/`apply_grammar_bitmask`/`_apply_grammar_bitmask_kernel`（真实 `@triton.jit`，本章高潮） |
| `async_utils.py` | `vllm/v1/worker/gpu/async_utils.py` | 只留 `async_copy_to_np` |
| `spec_decode_utils.py` | `vllm/v1/worker/gpu/spec_decode/utils.py` | `DraftTokensHandler`：`set_draft_tokens`/`get_draft_tokens`（async scheduling 关闭时的 `[-1]` 占位分支已删，批准项7） |
| `model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.sample()`：只留掩码落地的调用点，采样器分派（ch30/ch34）已删 |
| `engine_core.py` | `vllm/v1/engine/core.py` | `EngineCore.step()`（m07 重叠）+ `step_with_batch_queue()`（m15/9b 延后采样链，骨架完整保留，见上一节自我订正） |

## 关键符号 Source Map（节选，完整对应见各文件内 `# SOURCE:` 注释）

| 精简版符号 | 真实源码 | 备注 |
|---|---|---|
| `Scheduler._update_after_schedule` | `scheduler.py:L932-951` | `has_structured_output_requests` 门控的置位点 |
| `Scheduler.get_grammar_bitmask` | `scheduler.py:L1224-1246` | 上下篇交棒点；行序 = `num_scheduled_tokens` 迭代顺序 |
| `StructuredOutputManager.grammar_bitmask` | `__init__.py:L203-299` | 并行/串行分支、投机位置推进+`rollback` |
| `StructuredOutputManager._fill_bitmasks` | `__init__.py:L185-196` | `fill_bitmask` vs `_full_mask` 二选一 |
| `Scheduler.update_draft_token_ids_in_output` | `scheduler.py:L1623-1657` | `-1` 补齐 + `num_invalid_spec_tokens` 记录 |
| `Scheduler.make_spec_decoding_stats` | `scheduler.py:L1901-1917` | `num_invalid_spec_tokens` 的消费点 |
| `XgrammarBackend.compile_grammar_tail`（改名） | `backend_xgrammar.py:L115-122` | 只留最后一段：`max_rollback_tokens = num_speculative_tokens` |
| `apply_grammar_bitmask`（`utils.py`） | `structured_output/utils.py:L44-105` | 默认路径：`sorted_bitmask` 重排 + `xgr.apply_token_bitmask_inplace` |
| `StructuredOutputsWorker.apply_grammar_bitmask` | `worker/gpu/structured_outputs.py:L23-80` | opt-in 路径：H2D 搬运 + 行映射 + kernel 启动 |
| `_apply_grammar_bitmask_kernel` | `worker/gpu/structured_outputs.py:L86-115` | 真实 `@triton.jit`，位解包 + `-inf` 写回 |
| `EngineCore.step_with_batch_queue` | `engine/core.py:L447-561` | 延后采样链骨架（含上述自我订正） |

## 测试

`tests/`（38 个）：

- `test_schedule_gate.py`（5）：m01 门控 + 行序不变式
- `test_manager_fill.py`（5）：m03 形状预算、m04 并行阈值结构性死代码、m05 串行填充+
  投机推进/回滚、m06 `-1` 复位
- `test_reasoning_gate.py`（5）：m16 推理段门控
- `test_spec_prefilter.py`（5）：m12 装配前过滤 + `num_invalid_spec_tokens` 消费
- `test_backend_xgrammar_rollback_bound.py`（2）：`max_rollback_tokens`/`allocate_token_bitmask`
- `test_legacy_path.py`（3）：m11 默认路径（`sorted_bitmask` 重排、投机偏移）
- `test_engine_core.py`（4）：m07 重叠顺序、m15/9b 延后采样链两轮因果
- `test_worker_gpu.py`（4，**GPU 实测**）：m09 行映射、m10 kernel 位解包与 `-inf` 写回
- `test_draft_writeback.py`（2，**GPU 实测**）：m14 `has_structured_output_reqs` 门控
- `test_model_runner_sample.py`（3）：m18 落地时机

GPU 测试用 `pytest.mark.skipif(not torch.cuda.is_available())` 门控（同 ch18/ch19
惯例）；本轮环境有真实 GPU，全部 38 个测试（含 GPU 组）已在 host 上跑通，非仅
语法检查。
