# ch31 精简版实现笔记（只做减法）

离线 `LLM` 门面 + 同步 `LLMEngine.step()` 驱动 + `EngineCoreClient` 三分支工厂的忠实子集。
精简版可独立运行（`python3 -m pytest tests/`，不 `import vllm`），用 in-process 替身顶替
**独立后台进程 EngineCore + ZMQ**（ch05/ch07 主题）与 **去 token 化/RequestOutput 装配**
（ch08-ch10 主题），其余同步主干与真实 vLLM 同名/同结构/同控制流。

## 模块

| 精简版文件 | 角色 |
| --- | --- |
| `messages.py` | EngineCoreRequest/Output(s)、RequestOutput、Pooling/EmbeddingRequestOutput、SamplingParams、RequestOutputKind 的最小字段子集 |
| `envs.py` | `VLLM_ENABLE_V1_MULTIPROCESSING`（默认 True）—— 关键澄清的源码锚点 |
| `core_client.py` | `EngineCoreClient.make_client` 三分支 + `SyncMPClient`（后台线程+阻塞队列）+ `InprocClient`（进程内回退） |
| `llm_engine.py` | `LLMEngine.from_engine_args/__init__/add_request/step/has_unfinished_requests` |
| `llm.py` | `LLM` 门面：generate/chat/encode/embed + `_run_completion/_run_chat/_render_and_run_requests/_render_and_add_requests/_add_request/_run_engine` |
| `engine_core_stub.py` | `_StubEngineCore`（schedule+execute 替身, ch07）、`StubEngineCoreProc`（ZMQ output_socket 替身） |
| `processors_stub.py` | `InputProcessor`/`OutputProcessor`/`ParentRequest` 薄替身（ch05/ch06/ch08-ch10） |

## 1:1 Source Map（精简版 ↔ 真实 vLLM ↔ 改动 ↔ 原因）

| 精简版符号 | 真实 vLLM (`f3fef123`) | 改动 | 原因 |
| --- | --- | --- | --- |
| `LLMEngine.from_engine_args` | `vllm/v1/engine/llm_engine.py:L151-L177` | 删 `create_engine_config`/`Executor.get_class`，保留 `if envs.VLLM_ENABLE_V1_MULTIPROCESSING: enable_multiprocessing = True` | 配置归一/执行器选型是 ch03；本章只钉住 env 强翻 `multiprocess_mode=True` 的关键澄清锚点 |
| `EngineCoreClient.make_client` | `vllm/v1/engine/core_client.py:L80-L103` | 保留 3 分支判定，async+mp 分支改 raise（指 ch04） | 离线默认命中 `(mp=True, async=False)→SyncMPClient`；async 侧是 ch04 主题 |
| `SyncMPClient.__init__/get_output/add_request` | `vllm/v1/engine/core_client.py:L716-L830` | 后台 `process_outputs_socket` daemon 线程 + `outputs_queue`（queue.Queue）+ `get_output` 阻塞 `outputs_queue.get()` 原样保留；ZMQ socket/decoder/_send_input 用 in-process `StubEngineCoreProc.output_socket`(queue) 顶替 | 体现"后台线程喂队列 + 主线程阻塞取"的同步本质；ZMQ 字节协议 ch05/ch06 已讲 |
| `InprocClient.__init__/get_output/add_request` | `vllm/v1/engine/core_client.py:L274-L301` | 保留 `get_output=step_fn()+post_step`，`EngineCore` 用 `_StubEngineCore` 顶替 | 回退对照：真·进程内、无 ZMQ；仅 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 使用 |
| `LLMEngine.step` | `vllm/v1/engine/llm_engine.py:L287-L325` | 删 dummy-batch(DP)/profiler `with` 包裹/`record_stats`，保留 4 步：get_output→process_outputs→abort→return | profiler/stats 是可观测性旁路；DP 是 ch04/DP 侧 |
| `LLMEngine.add_request` | `vllm/v1/engine/llm_engine.py:L209-L285` | 删 EngineCoreRequest 弃用分支/extract_prompt_components，保留双注册 + `n>1` `ParentRequest` 扇出 | 弃用分支 v0.18 移除；ParentRequest 并行采样 ch04/ch06 已讲，此处只点复用 |
| `LLM._run_engine` | `vllm/entrypoints/llm.py:L1839-L1892` | 保留 `while has_unfinished_requests(): step()` + `sorted(by request_id)`；删 toks/s 估算细节 | 同步驱动主干 + 乱序完成按提交序还原是本章灵魂 |
| `LLM._add_request` | `vllm/entrypoints/llm.py:L1818-L1837` | 原样保留 `output_kind=FINAL_ONLY` + 自增 `request_id` | FINAL_ONLY 是与 ch04 DELTA 流式的关键对照点 |
| `LLM._render_and_run_requests` | `vllm/entrypoints/llm.py:L1760-L1787` | 原样保留 `isinstance(prompts,(list,tuple))→warning_once` | chat 路径专属物化 warning；completion 不走此函数 |
| `LLM._add_completion_requests` | `vllm/entrypoints/llm.py:L1592-L1626` | 保留逐 prompt 生成器渲染→`_render_and_add_requests`（不打 warning） | 与 chat 路径区分的对照锚点 |
| `LLM.generate/chat/encode/embed` | `vllm/entrypoints/llm.py:L446/L981/L1075/L1223` | 保留 runner 守卫 + 各自汇流入口；删 docstring/边角参数 | 四入口汇流同一条同步脊 |

## 已删除（SUBTRACTED 摘要，均 dossier `delete` 批准项）

- `LLM.__init__` 参数归一化 (L259-L380)、beam_search/classify/reward/score、`_resolve_mm_lora`、`_adjust_params_for_parsing`(Gemma4)。
- `LLMEngine` 的 sleep/wake_up/profile/lora/reset_*cache 等运维转发、DP 分支(dp_group/dummy_batch)、step 的 profiler/stats。
- `SyncMPClient/InprocClient` 的运维转发方法、`_send_input` 的 ZMQ 多帧/pending message 细节。
- 真实 EngineCore(schedule+execute)/EngineCoreProc(独立进程+busy loop)、InputProcessor 渲染、OutputProcessor 去 token 化——以 in-process 替身顶替（ch05/ch06/ch07/ch08-ch10 主题）。

## 验证

```
python3 -m pytest tests/test_offline_llm.py -q      # 17 passed
python3 scripts/lint_fidelity.py instances/vllm/artifacts/ch31-entrypoints   # ✓ 全部通过
```

测试钉住的真实 vLLM 可观察行为：默认→SyncMPClient（非 InprocClient）、env=0→InprocClient、
from_engine_args 强翻 multiprocess_mode、SyncMPClient 后台 daemon 线程喂阻塞队列、
`_add_request` 强设 FINAL_ONLY、`_run_engine` 按 request_id 排序（乱序完成也还原）、
completion 不 warning / chat 物化 list 才 warning、embed=encode 薄封装、n>1 ParentRequest 扇出。
