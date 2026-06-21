# Implementation Notes — ch08 Stage 3 Output Processing (subtract-only)

源码 pin `f3fef123`。本精简版为 `vllm/v1/engine/output_processor.py` 及其子系统
（detokenizer / logprobs / parallel_sampling / outputs.add / async_llm 的
output_handler+generate）的**只做减法**忠实子集。不引入 vLLM 没有的抽象；为脱离
`import vllm` 运行，把 tokenizer 库/torch 张量/metrics 容器替换为最小占位（标 SUBTRACTED），
其余控制流与 vLLM 1:1。

## 验收判据
把真实 vLLM 删掉所有 `# SUBTRACTED:` 标注的分支/字段，应 ≈ 得到本精简版。

## 1:1 Source Map

| 精简版符号 | 真实 vllm 源 | 改动 | 原因 |
| --- | --- | --- | --- |
| `output_processor.RequestOutputCollector`（put/get/get_nowait/ready/aggregate） | `vllm/v1/engine/output_processor.py:L45-L96` | 删 `_input_stream_task`/close/__del__、PoolingRequestOutput 合并分支 | f1 队列本体保真；流式输入清理与池化产品线正交（subtraction_plan） |
| `output_processor.OutputProcessorOutput` | `output_processor.py:L109-L112` | 无 | 返回值原样 |
| `output_processor.RequestState`（__init__/from_new_request/make_request_output/_new_request_output/_new_completion_output） | `output_processor.py:L129-L410` | 删 LoRA/metrics(stats)/tracing 输入字段(top_p/n/temp/max_tokens)/prompt_embeds/streaming_input/pooling 分支/routed_experts | per-request 累积状态 + 三道节流闸门保真；其余皆正交特性（subtraction_plan） |
| `output_processor.make_request_output` | `output_processor.py:L269-L331` | 删 pooling 分支 | FINAL_ONLY/stream_interval/DELTA/父聚合四闸门保真 |
| `output_processor._new_completion_output` | `output_processor.py:L376-L407` | 删 routed_experts 字段 | delta vs 全量 text/token/logprob 语义保真 |
| `output_processor.OutputProcessor.process_outputs` | `output_processor.py:L572-L687` | 删 pooling 分支/streaming_input 分支/do_tracing 分支；stats 调用点保留为占位（iteration_stats=None 早返回） | **本章主轴单循环**完整：取 state(None 跳过)→stats→prefill 翻转→去 token+停止串→logprobs→造输出/入队或入列表→完成清理+reqs_to_abort |
| `OutputProcessor._finish_request` | `output_processor.py:L689-L701` | 无 | 三张表注销保真 |
| `OutputProcessor.add_request` | `output_processor.py:L508-L537` | 删 streaming-input 更新分支与 LoRA wiring | 注册 RequestState + 三表映射 |
| `OutputProcessor.{has_unfinished_requests,get_num_unfinished_requests,propagate_error}` | `output_processor.py:L433-L444` | 无 | 生命周期/错误广播查询 |
| `detokenizer.IncrementalDetokenizer`（基类 + from_new_request） | `vllm/v1/engine/detokenizer.py:L30-L65` | 删 Fast 选择分支 | 空检测/工厂保真；Fast=tokenizers DecodeStream 属库内部 |
| `detokenizer.BaseIncrementalDetokenizer.update` | `detokenizer.py:L95-L142` | 无（算法逐行保留） | 增量去 token + min_tokens 保护 + 停止串截断 |
| `detokenizer.get_next_output_text` | `detokenizer.py:L148-L164` | 无 | delta/全量 + stop_buffer_length 回退 |
| `detokenizer.SlowIncrementalDetokenizer.decode_next` | `detokenizer.py:L245-L301` | 注入 `decode(id)->str` 替代 prefix/read_offset+detokenize_incrementally+byte-fallback | tokenizer 内部 + byte-fallback 容错正交（subtraction_plan） |
| `detokenizer.check_stop_strings` | `detokenizer.py:L304-L339` | 无 | 停止串窗口匹配 + 截断规则原样 |
| `logprobs.LogprobsProcessor`（from_new_request/_update_sample_logprobs/_update_prompt_logprobs/pop_prompt_logprobs/update_from_output） | `vllm/v1/engine/logprobs.py:L29-L353` | 容器换 list、删 byte-fallback 纠正(_verify_tokens/_correct_decoded_token/_get_sampled_context_ids)、payload 预 pythonize | cumulative_logprob 累积 + DELTA pop 语义保真；纠正逻辑正交（subtraction_plan） |
| `parallel_sampling.ParentRequest`（__init__/get_outputs/child_requests/output_aggregator） | `vllm/v1/engine/parallel_sampling.py:L13-L126` | 删 _get_child_sampling_params/get_child_info/n/observe_* | n>1 父聚合(get_outputs)保真；请求拆分侧属 input/ch04 |
| `outputs.RequestOutput` + `outputs.add` | `vllm/outputs.py:L88-L173` | 删 metrics/lora/encoder/num_cached_tokens/kwargs | DELTA 归并(add)真正实现逐行保留 |
| `outputs.STREAM_FINISHED` | `vllm/outputs.py:L192` | 无 | 哨兵 |
| `output_processor.output_handler` | `vllm/v1/engine/async_llm.py:L656-L689` | 删 IterationStats/logger 指标/update_scheduler_stats/propagate_error 包裹 | f3 生产者核心循环：拉批→分块→process_outputs→断言不回填→reqs_to_abort 反向 abort |
| `output_processor.generate` | `vllm/v1/engine/async_llm.py:L576-L586` | 删外层 add_request/取消与错误处理（ch04 已讲）；以 return list 收集替代 yield 便于单测 | f3 消费者循环：get_nowait() or await get() → 非 STREAM_FINISHED 收集 → finished 终止 |
| `_types.{RequestOutputKind,FinishReason,EngineCoreOutput,CompletionOutput,PrefillStats}` | `sampling_params.py:L151`/`v1/engine/__init__.py:L30,L42`/`outputs.py:L22` | 字段精简到 Stage 3 所读 | 支撑类型忠实子集，脱离 import vllm |

## 测试（`tests/`，纯单元，不 import vllm）
- `test_collector.py` — f1 邮箱：put/Event、DELTA 归并、CUMULATIVE 替换、不同 index 不互覆盖、异常优先、get 阻塞。
- `test_detokenizer.py` — 增量累积、停止串截断/包含、stop-terminated 末 token 排除但留 id、min_tokens 延迟、delta/全量 + stop_buffer 回退、check_stop_strings 窗口。
- `test_parallel_sampling.py` — n>1 流式逐子转发/已返回不重发/全完判定、FINAL_ONLY 攒齐 n 个一次返回。
- `test_process_outputs.py` — 跳过已 abort、prefill 翻转+num_cached、Async 入队 vs LLMEngine 返回列表、停止串反向 abort、_finish_request 注销三表、FINAL_ONLY 节流、DELTA prompt logprobs 一次性 pop。
- `test_stream_interval.py` — 首 token 必发、攒够 interval 才发、完成强发、DELTA sent_tokens_offset 无重叠无丢失。
- `test_producer_consumer.py` — f3 单生产者扇出多消费者、chunk 分块、停止串反向 abort EngineCore。

35 tests，host `python3 -m pytest tests/` 全通过。
