# ch07 精简版 impl-notes — 上行：从 token 到文字（Part II：API 进程上行泳道）

- **Pin**：vLLM v0.27.1（`6e448d0ea`）。全部 `# SOURCE:` 行号对当前 pin 现核（收工前又对 18 个
  verbatim 段做了机械相似度审计：逐字段逐一 diff，非 1.0 的差异全部精确等于批准的删除项/类型
  收窄——见 §审计；未照抄 v2 资产的 v0.21.0 旧行号）。
- **产物**：`implementation/uplink.py`（单模块精简版 ~2370 行，221 个 `# SOURCE` 锚 + 71 个
  `# SUBTRACTED` 标记）+ `implementation/_msgspec_seam.py`（msgspec 宿主替身，ch05/ch06 同款
  seam 原样复制——本章 EngineCoreOutputs 过线走真 msgpack 字节）。host 可跑：**真 tokenizers
  0.22.2 Rust `DecodeStream`**（Fast 路径原生增量解码/UTF-8 边界是真实行为，不是模拟）、真
  msgpack、真 asyncio；无 vllm 包、无 msgspec 包、host transformers 4.57 无 `TokenizersBackend`
  （pin 需新版——seam 回退类只替名字与 `._tokenizer` 触达面）。
- **跑法**：`cd instances/vllm/artifacts-v3/ch07-uplink-token-to-text && python -m pytest tests/ -q`
  → **72 passed**（~3.5s，纯 host 单元/异步测试，无平台分支 → 无需进容器）。`python
  scripts/lint_fidelity.py <本章目录>` → **全部通过（无 BLOCKING、无警告）**；must_keep 60 个
  符号经 linter `over_subtraction` 项全数核在。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本模块（HOST SEAM 例外见 §Seam
  清单——每个 seam 行内标注并在此登记）。

## 本章主题 = 全真部分（与 ch04/ch05/ch06/ch08 的分工）

- **单槽邮箱**（m16）：`RequestOutputCollector` 全类——单槽 + `asyncio.Event` + put 就地合并
  （`RequestOutput.add` 按 CompletionOutput.index 配对逐字，outputs.py:L152-181）、Exception
  无条件抢槽、`get` 阻塞 / `get_nowait` 快路径——刻意不是 asyncio.Queue（WC1）。
- **唯一单循环**（m3）：`OutputProcessor.process_outputs` 全函数（含 NOTE FOR DEVELOPERS 注释
  逐字）——demux 按内部 id 查 RequestState、查不到=已 abort 幂等跳过 → detokenize+stop 判定 →
  造输出 → `queue.put` / `list.append` 分叉 → 完成清理 + `reqs_to_abort`。
- **三道闸**（m13/m14/m15）：`make_request_output`——FINAL_ONLY 未完零构造、stream_interval
  节流（DELTA 从 `sent_tokens_offset` 切、per-request clamp 取 max）、n>1 走
  `ParentRequest.get_outputs` 父聚合（流式逐子转发 / FINAL_ONLY 攒齐 n 个）。
- **增量去 token 化**（m4-m12）：detokenizer 全层级——空壳 + 三路工厂（`TokenizersBackend`
  判据是 v0.27.1 的，非 v0.21.0 的 PreTrainedTokenizerFast）、`update` 主流程（跳 stop token/
  min_tokens 推进 stop_check_offset/PR #22014）、`get_next_output_text` delta 切片 +
  `stop_buffer_length = max(len(s))-1` holdback、`check_stop_strings` 窗口查找 + **v0.27.x
  新语义**（多条命中取『完成最早』者——非 v0.21.0 的列表序首个）、Fast（DecodeStream native
  prefill + `_protected_step` 两类异常恢复：Overflow/TypeError 吞掉、'Invalid prefix
  encountered' 重建流）、Slow（prefix/read 双 offset 窗口 + `convert_prompt_ids_to_tokens`
  初始窗口 = 尾部 7 token、prefix 退 5 + `�` 尾判定吐空串冻结 read_offset）。
- **回程到港与常驻分发**（m1/m2/m17）：`AsyncMPClient._ensure_output_queue_task`（recv →
  `validate_alive` 逐字 → msgpack 解码 → `outputs_queue.put_nowait`；异常/死讯也走队列）+
  `get_output_async` 逐字；`AsyncLLM._run_output_handler` 拉批分块（`VLLM_V1_OUTPUT_PROC_
  CHUNK_SIZE=128` 切片 + 片间 `await asyncio.sleep(0)`）+ stop-string 反向 abort + 异常
  `propagate_error` 广播；eager/lazy 双启动（L173-179 与 L390-393 两段逐字）。
- **消费与断连**（m18/m19/m22）：`generate()` 消费循环（`q.get_nowait() or await q.get()`、
  两条注释逐字）+ 错误层级（CancelledError/GeneratorExit → `abort(internal=True)` 两跳：
  本进程移状态 + ABORT 终态解阻塞 → 跨进程停算；EngineDeadError 不 abort；意外 Exception →
  abort + `EngineGenerateError`）；abort 双轨（外部 id 展开 / 内部 id 单删 / 父 id 联动子）；
  `add_request` 上行登记（collector 诞生于过线之前 + n>1 扇出 `idx_` 前缀子 id、末子复用原
  对象、seed 逐子克隆/无 seed 缓存复用——`_get_child_sampling_params` 逐字）。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因；核心行）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `RequestOutputCollector`（put/get/get_nowait） | vllm/v1/engine/output_processor.py:L45-L96 | 逐字 minus `_input_stream_task`/`close`/`__del__`（L60、L98-106）与 Pooling 替换分支（L73-76）；类型联合收窄去 Pooling | must_keep×6；delete 项 2/3 |
| `RequestOutput.add` | vllm/outputs.py:L152-L181 | **逐字**（相似度 0.96，差异仅为行内 SOURCE 注释） | must_keep；单槽合并实现 |
| `RequestOutput`/`CompletionOutput` | vllm/outputs.py:L21-L197 | RequestOutput 逐字；CompletionOutput 字段全保留，`cumulative_logprob`/`logprobs` 加默认 None（机械，见 §机械表） | 承载对象；kv/ec/metrics 字段保留默认 |
| `RequestOutputKind` | vllm/sampling_params.py:L182-L188 | **逐字** | must_keep；三态契约本体 |
| `FinishReason`/`FINISH_REASON_STRINGS` | vllm/v1/engine/__init__.py:L29-L65 | **逐字**（IntEnum+__str__） | finish_reason 字符串是外部 API |
| `EngineCoreRequest`/`EngineCoreOutput`/`EngineCoreOutputs`/`EngineCoreRequestType` | vllm/v1/engine/__init__.py:L97-L154、L184-L215、L230-L258、L261-L274 | 字段/成员全保留（线格式 schema 契约；logprobs/pooling/trace 等他章域字段类型放宽为 Any——注解面非运行时） | must_keep；线载体 + m1/m15 |
| `IncrementalDetokenizer`（空壳+工厂） | vllm/v1/engine/detokenizer.py:L31-L66 | **逐字** | must_keep×2；m5 三路分派 |
| `USE_FAST_DETOKENIZER`/`INVALID_PREFIX_ERR_MSG` | vllm/v1/engine/detokenizer.py:L23-L28 | **逐字** | must_keep×2；版本闸/恢复判别串 |
| `BaseIncrementalDetokenizer`（__init__/update/decode_next/get_next_output_text） | vllm/v1/engine/detokenizer.py:L69-L165 | **逐字**（update 相似度 1.000） | must_keep×5；m4/m9/m11/m12 |
| `FastIncrementalDetokenizer` | vllm/v1/engine/detokenizer.py:L168-L248 | native prefill + `_protected_step` 逐字；空格抑制段删（L189-209/L214-220） | must_keep×3；m6（delete 项 7） |
| `SlowIncrementalDetokenizer` | vllm/v1/engine/detokenizer.py:L251-L307 | 逐字 minus prompt_embeds 兜底（L272-276 else + L278 `or [0]*prompt_len`）；覆写 output_token_ids/num_output_tokens 减 prompt_len 保留 | must_keep；m7/m8（delete 项 9） |
| `check_stop_strings` | vllm/v1/engine/detokenizer.py:L310-L362 | **逐字**（含 v0.27.x『完成最早』docstring） | must_keep；m10 |
| `_replace_none_with_empty`/`INITIAL_..OFFSET`/`convert_prompt_ids_to_tokens`/`detokenize_incrementally` | vllm/tokenizers/detokenizer_utils.py:L11-14、L57-59、L119-140、L176-268 | 前三逐字；detokenize_incrementally 主线逐字、else 分支（L246-258）退化为统一 `convert_tokens_to_string` | must_keep×4；m7/m8（delete 项 8） |
| `ParentRequest`（含 `_get_child_sampling_params`/`get_child_info`/`get_outputs`） | vllm/v1/engine/parallel_sampling.py:L13-L126 | **逐字** minus observe_*（L128-150） | must_keep×5；m15（delete 项 5） |
| `RequestState`（__init__/from_new_request/make_request_output/_new_*） | vllm/v1/engine/output_processor.py:L129-L423 | 三道闸/工厂逐字；删 lora/stats/routed_experts/streaming 字段与 logprobs/pooling/kv-ec 分支（各锚点行内标） | must_keep×10；delete 项 1/2/3/4/5/6 |
| `OutputProcessor`（__init__/abort_requests/add_request/process_outputs/_finish_request/_update_stats_*） | vllm/v1/engine/output_processor.py:L429-L836 | demux/双轨展开/终态解阻塞/父联动/三表注销逐字；stats 内部删（调用点保留、恒 None 早返回）；`get_num_unfinished_requests`/`has_unfinished_requests` 保留 | must_keep×7；m3/m18/m22 |
| `AsyncMPClient`（_ensure_output_queue_task/get_output_async/_format_exception/_send_input/add_request_async/abort_requests_async） | vllm/v1/engine/core_client.py:L974-L1014、L1016-L1091、L1093-L1102、L695-L699、L1104-L1114、L1145-L1152 | 队列任务/取队/异常格式化/ADD-ABORT 面逐字（删项 11 的 utility/EEP/FT 分支与 weakref 机制）；`_send_input` 为记录面 seam | must_keep×2 + m1；站 4 |
| `AsyncLLM`（add_request/_add_request/generate/_run_output_handler/abort/is_running/errored） | vllm/v1/engine/async_llm.py:L72-L418、L420-L435、L544-L655、L657-L727、L729-L738、L1085-L1100 | 上行五件套逐字（删项 3/12 的流式输入与日志行、ch06 域渲染分流、ch38 域面板）；L390-393/L593-599/L608-610 三段注释原话全保留 | must_keep×6；站 1-2/13-14 |
| `InputProcessor.assign_request_id` | vllm/v1/engine/input_processor.py:L231-L249 | **逐字**（ch06 产品域 seam，上行登记的依赖面） | 站 1 前置；m22 双轨 id |
| `random_uuid`/`length_from_prompt_token_ids_or_embeds`/`as_list` | vllm/utils/__init__.py:L11-36、utils/collection_utils.py:L49-51 | **逐字** | 保留代码触到的真实小件 |

## 删除台账 — dossier subtraction_plan 14 项 delete（全部执行）

1. **LogprobsProcessor** ✓ — `update_from_output` 调用行（L663-665）、`_new_completion_output`
   logprobs 准备段（L404-407/L419/L420）、`_new_request_output` prompt_logprobs 分支（L366-371
   →恒 None）、`from_new_request` 的 LogprobsProcessor 构造（L230-233→None）、两处 assert；类
   本体不实现。`RequestState.logprobs_processor` 字段名保留（恒 None，ch8 回填）。
2. **pooling 全链** ✓ — `process_outputs` 的 pooling 读取与分支守卫（L632/L652）、
   `make_request_output` 的 pooling_output 形参与分支（L317-322）、`_new_pooling_output`
   （L425-426）、collector 的 Pooling 替换分支（L73-76）、abort 的 EMPTY_CPU_TENSOR 三元
   （L504-508）、常量本体（L41-42）、`from_new_request` pooling else（L242-250）、`is_pooling`。
3. **流式输入（resumable）** ✓ — StreamingUpdate/apply_streaming_update/_update_streaming_
   request_state/input_chunk_queue/streaming_input 全部（L115-127、L186-209、L533-537、
   L556-587、process_outputs 两处分支 L676-677/L688-693）、collector `_input_stream_task`/
   close/__del__（L60、L98-106）、generate 的 STREAM_FINISHED 判别（L605）与 InputStreamError
   分支（L631-636）与 finally q.close()（L653-655）、`_add_streaming_input_request`/
   `_validate_streaming_input_sampling_params`（L437-538）、InputStreamError 类（L60-69）、
   STREAM_FINISHED 哨兵（outputs.py L200-208）。
4. **LoRA** ✓ — RequestState lora_request/lora_name（L130 形参/L157-158）、OutputProcessor
   lora_states（L446）、abort 的 request_finished 调用（L498）、_new_request_output 的
   lora_request 实参（L375）。
5. **stats/tracing** ✓ — do_tracing 全方法（L730-790）、_update_stats_from_output/_finished
   内部（L802-811/L822-836——调用点保留、恒 None 早返回）、update_scheduler_stats（L727-728）、
   RequestState.stats（L177）、output_handler 的 IterationStats 构造与 update_scheduler_stats
   与第 4 步 Logging（L683-685/L711-722）、ParentRequest.observe_*（L128-150）、
   `metrics=self.stats` 实参。
6. **routed_experts 与 kv/ec 透传** ✓ — RequestState.routed_experts_chunks（L179-180）、
   process_outputs 累积（L637-640）、_new_completion_output 拼接（L409-412）、kv/ec 形参+读取+
   实参全链（make_request_output L282-283/L338-339、process_outputs L635-636、_new_request_
   output L347-348/L381-382、abort L511-512）。CompletionOutput/RequestOutput 的字段本身保留
   （核心字段完整，默认 None）。
7. **Fast 空格抑制** ✓ — `__init__` 的 added_token_ids 预计算（L189-209）与 decode_next 抑制段
   （L214-220）——decode_next 退化为 `token or ""`。
8. **detokenizer_utils 辅助** ✓ — `_convert_tokens_to_string_with_added_encoders`（L17-54）、
   detokenize_incrementally 的 else 分支（L246-258 退化统一）、marker 三件套 +
   convert_ids_list_to_tokens（L62-170）。
9. **Slow embeds 兜底** ✓ — L272-276 else 与 L278 `or [0]*prompt_len` 占位——只走
   prompt_token_ids 主线。
10. **弹性 EP logger 机制** ✓ — `_logger_ref`/logger_ref（L668-672/L716-722）与 renderer 引用
    （seam 装配不引入）。
11. **core_client 旁支** ✓ — process_outputs_socket 的 utility/EEP/FT 分支（L1043-1080）、
    output_handler 回调参数与 `_self_ref` weakref（L1024-1035）；异常入队语义保留。
12. **generate 日志** ✓ — 意外 Exception 的文案构造压缩（L639-652 → `raise EngineGenerateError()
    from e`）与全部 log_requests 日志行（L434-435/L614-615/L620-621/L627/L634-635/L642-651/
    L740-741）。
13. **serving/SSE 层** ✓ — with_cancellation/listen_for_disconnect/protocol.to_sampling_params/
    serving generator 不入（ch38 域；F5 第一跳叙事引用——`generate()` 的 CancelledError 路径由
    测试直接 cancel 任务驱动，不需要 HTTP 层）。
14. **LLMEngine/离线同步面** ✓ — step 四步/_run_engine 不入（叙事对照讲；分叉判据
    `queue is None` 与 list.append 分支保留在 process_outputs，测试直驱）。

### 机械删除/替换（不在 delete 单——为可跑性与章边界所必需，**请 reviewer 逐条过目**）

| 位置 | 内容 | 理由 |
|---|---|---|
| outputs.py L44-45 | `CompletionOutput.cumulative_logprob/logprobs` 加默认 `None`（真码无默认） | logprobs 准备段删除后调用点不再传这两参；dataclass 后续字段本有默认，加默认不改变既有构造语义；ch8 回填时恢复传参即可 |
| output_processor.py 全文件 | PoolingRequestOutput/PoolingOutput 从类型联合与 isinstance 中移除 | delete 项 2『PoolingRequestOutput 分支』的类型面机械后果 |
| output_processor.py L631-636 | `pooling_output`/`kv`/`ec` 局部变量读取行删除 | 所喂分支已删、变量无消费者（项 2/6 机械后果） |
| async_llm.py L352-381 | dict/raw prompt 渲染分流不引入（else 结构洞） | ch06 产品域；非 EngineCoreRequest 的 prompt 会 NameError——与 ch06『n>1 返回 None』同款结构洞，测试不触达 |
| from_new_request L242-250 | sampling_params 为 None 的 pooling else 结构洞 | 同上（pooling 删除的结构后果，生成式主线恒有 sampling_params） |
| async_llm.py 面板 | encode/pause/check_health/profile/weight-*/__del__/shutdown 等不入 | 上行主线不触达（ch38/ch34/ch39 域；ch06 同款『章边界不入』先例） |
| core_client.py 面板 | SyncMPClient/InprocClient/DP 面/shutdown 不入 | 同上；本章只保留 AsyncMPClient 上行面 |
| 装配线 | AsyncLLM.__init__/AsyncMPClient.__init__ 为 seam 构造（注入 engine_core/tokenizer/stream_interval） | ch03 装配产物；L141-146/L173-179/L997/L1006-1014 逐字保留 |
| Struct 注解 | EngineCoreRequest/Output 的他章域字段类型放宽为 Any（字段名/顺序/默认全保） | 注解面非运行时行为；避免引入 ch8/ch36/metrics 域类型 |
| utils | `logging`（NullHandler+*_once）/envs/exceptions 为 stdlib seam | ch04/ch05 同款 |

## Seam 清单（HOST SEAM，全在行内标注）

| Seam | 位置 | 是什么 | 为什么仍忠实 |
|---|---|---|---|
| `_msgspec_seam` | 独立文件（ch05 同款） | msgspec API 子集的 msgpack-backed 实现 | 线上字节是真 msgpack；array_like 全字段编码已容器实测三版对齐 |
| `TokenizersBackend` 回退类 | 模块头 try/except | host transformers 4.57 无此类；seam 只替名字 + `._tokenizer`（Rust tokenizers.Tokenizer） | 被保真代码触达面就是 isinstance + `._tokenizer`（detokenizer.py:L61/L178）；容器内真 import 优先生效 |
| `SeamOutputSocket`/`SeamFrame` | §8 | ch05 PULL socket 替身：`recv_multipart` 面 + 测试注入已编码帧 | 帧内容 = 真 msgpack 编码的 EngineCoreOutputs（引擎侧等价物）；socket 物理层归 ch5 |
| `MsgpackDecoder`（单面） | §8 | ch05 全量 decoder 的单面子：`decode(frames)` 解 bufs[0] | 镜像 serial_utils.py:L340-348 形状；本章线载体无张量/aux 帧 |
| `BackgroundResources` | §8 | 资源面子集：output_socket/output_queue_task/engine_dead | `validate_alive` L490-493 逐字；其余为 ch05 域 |
| `AsyncMPClient._send_input` | §8 | 记录 (帧标签, 请求) 不触 ZMQ | ch06 同款章边界；ADD/ABORT 行为可断言 |
| `InputProcessor` | §9 | ch06 产品域 seam（只含 assign_request_id 逐字） | 双轨 id 出发侧 ch6 已讲；本章只用其产物 |
| `SamplingParams`/`PoolingParams` | §2 | 字段 seam + 真实默认值（n/stop/min_tokens/include/skip/spaces/detokenize/output_kind/stream_interval/seed/max_tokens/top_p/temperature） | 参数域归 ch06（其 must_keep 三方法已在那边逐字）；本章触达字段全在 |
| `envs`/logger/exceptions | §0 | stdlib seam | ch04 同款；`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 默认 128 与 envs.py:L160/L1371-1372 同源（含 env 覆读） |

## 已知偏差（reviewer 重点）

1. **Fast 路径的测试 tokenizer 是手工 byte-level WordLevel + ByteLevel decoder**：非 HF 预训练
   tokenizer，但 DecodeStream 步进/Native prefill/UTF-8 缓冲是**真 Rust 行为**（E4→None、
   B8→None、AD→'中'；stranded 续字节→'�A' 实测与 docstring 语义一致）。vocab 构造按
   GPT-2 `bytes_to_unicode`（¡..¬ 161-172 **含** 172——首版漏 172 曾致 '�ń'，已修）。
2. **'Invalid prefix encountered' 恢复分支以 stub 流注入触发**：真 DecodeStream 上难以从良构
   vocab 自然触发非单调无效前缀（issue #17448 场景）；stub 抛 pin 同款错误串，验证的是**真码**
   `_protected_step` 的捕获-重建-重放逻辑（重建后流为 fresh、无 prefill 上下文——真实语义）。
3. **Slow 路径 TokenizerLike 是测试侧 seam**（byte 级 convert_ids_to_tokens/convert_tokens_
   to_string 用 latin-1→utf-8 replace 造出 � 语义）：与 Fast 路径同一 token 流对照断言
   output_text/delta 一致（test_fast_and_slow_agree_on_the_same_byte_stream）。真实 HF
   tokenizer 的 cleanup 算法差异不进本章（那是 tokenizer 域）。
4. **`q.get_nowait() or await q.get()` 的真值地雷**（WC1 cost）未在测试断言——RequestOutput 恒
   真是默认对象语义，测试 fast-path 用例已覆盖行为面。
5. **结构洞**：非 EngineCoreRequest 的 prompt / pooling 请求 / n>1 之外的 params=None 路径会
   NameError/落空——与『删掉批准分支后的真码』严格一致（ch06 先例），测试不触达。
6. **异步测试用 `asyncio.run` 包装**（host 无 pytest-asyncio，ch06 同款）；每测试独立事件循环，
  Harness.close() 在 finally 里取消 output_handler 与 queue task。

## 收工审计（2026-08-18）

- **verbatim 段机械相似度审计**（ignoring comment-only lines）：`BaseIncrementalDetokenizer.update`
  1.000、`get_next_output_text` 1.000、`_protected_step` 1.000、`Slow.decode_next` 1.000、
  `ParentRequest.get_outputs`/`_get_child_sampling_params` 1.000、`_finish_request` 1.000、
  `propagate_error` 1.000、`get_output_async` 1.000、`AsyncLLM.abort` 1.000、`RequestOutput.add`
  0.962（差异=行内锚注释）、`check_stop_strings` 0.975、`RequestOutputCollector.get/get_nowait`
  0.889（差异=类型联合收窄）、`put` 0.727（差异=恰为批准删除的 Pooling elif）、
  `detokenize_incrementally` 0.797（差异=恰为批准删除的 else 分支 + Any 类型面）、
  `assign_request_id` 0.947。**全部非 1.0 差异逐条核对 = 批准删除项/注解收窄，无未批准改动。**
- `python -m pytest tests/ -q` → **72 passed**；`python scripts/lint_fidelity.py 本章目录` →
  **全部通过**（含 must_keep 60 符号 over_subtraction 空账）。
- 测试面按机制覆盖 m1-m22 全部 22 个 mechanism（见 tests/test_uplink.py 头部清单）。

## 测试面（72 passed；断言的是 pin 可观察行为，非自洽）

- 单槽邮箱：DELTA 合并/CUMULATIVE 替换/异常抢槽/新 index append/get 阻塞/get_nowait 空槽、
  `get_nowait() or await get()` 快路径、aggregate 三态映射
- 去token：Fast（真 DecodeStream）delta/cumulative、native prefill 不含 prompt、多字节
  E4/B8/AD 步进 None→'中'、stranded 续字节 '�A'、Invalid prefix 重建恢复、Overflow/TypeError
  吞掉；Slow 双 offset 初始窗口（7 token/prefix 退 5）、� 冻结 read_offset、空 token 零增量、
  OOV 空串、Fast/Slow 同流对照一致；工厂三路分派 + detokenize=False 空壳
- stop：命中截断/返回串、include 截尾与 -1、holdback max(len)-1（含 include=0）、min_tokens
  窗口吞 stop 与越界检出（PR #22014）、stop token 排除（id 记账/文本不记账）、check_stop_strings
  完成最早仲裁/窗口下界/空窗
- 三道闸：FINAL_ONLY 零构造（槽永不沾）与终值、DELTA/CUMULATIVE 快照、stream_interval 首
  token/攒批/完成三触发 + DELTA 无损拼接 + per-request clamp 取 max、外部 id 写回、demux 双请求
  同批 + 幽灵 id 幂等跳过、prefill_stats 记 num_cached_tokens、stop-string 终态改写 finish_reason
  + reqs_to_abort、queue=None 同步面返列表、完成三表清空
- n>1：扇出 idx_ 前缀/共享 collector/子 params n=1、seed 逐子克隆 vs 无 seed 缓存复用、流式逐子
  转发、FINAL_ONLY 攒齐 n 个、重复完成子不重发
- abort：内部 id（ABORT 终态解阻塞+三表清）、外部 id 展开、父 id 联动
- e2e：EngineCoreOutputQueueTask 任务名、generate 逐 token yield 外部 id、FINAL_ONLY 单值、
  chunk=128→2 切片 + 片间 sleep(0) 让出（watcher 落在两片之间）、eager/lazy 双启动、死讯哨兵
  →EngineDeadError 且不 abort、socket 异常→EngineGenerateError(cause) 广播到全部消费者、
  queue task 取消→EngineDeadError 入队、stop-string 反向 abort 帧、断连 cancel→两跳 abort、
  msgpack round-trip、双轨 id 8-hex/预设 external_req_id 拒绝、三态枚举值、chunk 默认 128
