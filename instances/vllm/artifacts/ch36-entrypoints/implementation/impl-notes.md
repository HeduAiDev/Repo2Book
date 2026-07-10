# ch32 精简版实现笔记 —— OpenAI 兼容服务器（只做减法）

源码 pin `f3fef123`。精简版与真实 vLLM **同名、同结构、同控制流**，仅删 dossier
`subtraction_plan.delete` 批准项；`must_keep` 全部保留。把所有 `# SUBTRACTED:` 分支删回去
≈ 真实 vLLM 在 **generate 任务 / 单进程 TCP 无 SSL / 普通采样（n==1、无 harmony/mistral/beam/logprobs）**
下的主干。

## 模块组织（按真实文件归位）

| 精简版模块 | 对应真实 vLLM 文件 | 内容 |
|---|---|---|
| `api_server.py` | `vllm/entrypoints/openai/api_server.py` | build_async_engine_client(_from_engine_args) / run_server_worker / setup_server / build_and_serve / build_app / init_app_state |
| `api_router.py` | `vllm/entrypoints/openai/chat_completion/api_router.py` + `vllm/entrypoints/utils.py` | POST /v1/chat/completions handler + with_cancellation / listen_for_disconnect |
| `chat_serving.py` | `vllm/entrypoints/openai/chat_completion/serving.py` | OpenAIServingChat：create_chat_completion / render_chat_request / stream_generator / full_generator |
| `render_serving.py` | `vllm/entrypoints/serve/render/serving.py` | OpenAIServingRender：render_chat / preprocess_chat |
| `engine_serving.py` | `vllm/entrypoints/openai/engine/serving.py` | OpenAIServing 基类：_check_model / _base_request_id / create_error_response / _raise_if_error / _maybe_get_adapters |
| `launcher.py` | `vllm/entrypoints/launcher.py` | serve_http / watchdog_loop / terminate_if_errored |
| `server_utils.py` | `vllm/entrypoints/openai/server_utils.py` | lifespan / AuthenticationMiddleware / XRequestIdMiddleware |
| `messages.py` | `vllm/engine/protocol.py` + `.../chat_completion/protocol.py` + `vllm/outputs.py` | EngineClient 协议 + ChatCompletion(Stream)Response / DeltaMessage / UsageInfo / ErrorResponse / RequestOutput / SamplingParams |
| `envs.py` | `vllm/envs.py` | 本章读到的 4 个环境开关 |
| `_framework.py` | fastapi / uvicorn / starlette（第三方） | 极小替身：FastAPI/APIRouter/Request/JSONResponse/StreamingResponse/CORS。第三方框架，正交于 vLLM 主线。 |

## 1:1 Source Map（精简版 ↔ 真实 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实 vllm/...:Lxxx | 改动 | 原因 |
|---|---|---|---|
| `build_async_engine_client` | `openai/api_server.py:L77` | 删 forkserver 预导入分支 | dossier delete 批准（部署期可选，正交于主线） |
| `build_async_engine_client_from_engine_args` | `openai/api_server.py:L108` | 原样保留 try/finally shutdown；AsyncLLM 起引擎内部留给 ch04 | EngineClient 生命周期边界，承接 ch04 三段式 |
| `run_server_worker` | `openai/api_server.py:L681` | 删 tool/reasoning parser plugin 导入 | dossier delete 批准；保留 async-with→build_and_serve→await shutdown_task 编排 |
| `setup_server` | `openai/api_server.py:L533` | 删 uds/ipv6/ssl 分支、日志、set_ulimit | dossier delete 批准；保留『先绑 socket + SIGTERM』主线 |
| `build_and_serve` | `openai/api_server.py:L578` | 删 ssl_*/h11_* kwargs 透传、log_config | dossier delete 批准；保留 get_supported_tasks→build_app→init_app_state→serve_http |
| `build_app` | `openai/api_server.py:L157` | 删 chat 外全部条件路由 + 多数中间件 + 5 个 exception_handler（留 1 示例） | dossier delete 批准；保留 FastAPI(lifespan)+chat 路由+CORS/Auth/XRequestId |
| `init_app_state` | `openai/api_server.py:L317` | 删 Tokenization/pooling/transcription/realtime 服务对象 | dossier delete 批准；保留 Models/Render/Chat 挂 app.state |
| `lifespan` | `openai/server_utils.py:L446` | 原样保留 _force_log + freeze_gc_heap（freeze_gc_heap 内部占位 gc.collect） | must_keep；点名本章 |
| `AuthenticationMiddleware` | `openai/server_utils.py:L38` | scope 读法用 dict 替身（无真 starlette URL/Headers） | must_keep；sha256+compare_digest 防时序侧信道原样保留 |
| `XRequestIdMiddleware` | `openai/server_utils.py:L89` | 同上 scope 替身 | dossier key_classes 点名 |
| `OpenAIServing` | `openai/engine/serving.py:L135` | 删 beam_search、logprobs 辅助、trace/log 细节 | dossier delete 批准；保留校验/request_id/错误工厂 |
| `_check_model` | `openai/engine/serving.py:L417` | 删运行时 LoRA 加载分支 | dossier 标注非主线；保留 404 NotFoundError |
| `_base_request_id` | `openai/engine/serving.py:L592` | 原样 | must_keep；X-Request-Id 优先 |
| `create_error_response` / `create_streaming_error_response` / `_raise_if_error` | `openai/engine/serving.py:L372/L381/L398` | 原样 | must_keep；统一错误工厂 + GenerationError 桥 |
| `OpenAIServingChat.create_chat_completion` | `chat_completion/serving.py:L229` | 删 beam_search/reasoning_ended 预判/mistral grammar | dossier delete 批准；保留 render→request_id→SamplingParams→generate→分流 |
| `chat_completion_stream_generator` | `chat_completion/serving.py:L408` | 删 tool/reasoning/logprobs/harmony 增量构建（delta_text=output.text） | dossier delete 批准；保留首块 role 空 delta / 逐 output delta / finish_reason / include_usage 末块 / [DONE] |
| `chat_completion_full_generator` | `chat_completion/serving.py:L1148` | 删 logprobs/tool_choice 组装长链 | dossier delete 批准；保留 async-for 聚合 final_res → ChatCompletionResponse + UsageInfo |
| `OpenAIServingRender.render_chat` | `serve/render/serving.py:L184` | 删 mistral 重序列化 / harmony 分支 | dossier delete 批准；保留 tool_choice 校验 + validate_chat_template + preprocess_chat |
| `preprocess_chat` | `serve/render/serving.py:L523` | 删 params with_defaults 多模态合并 + parser adjust_request 后处理 | dossier 标注点到即止；保留 renderer.render_chat_async 渲染点 |
| `serve_http` | `launcher.py:L26` | 删路由打印 / SSLCertRefresher / 端口占用诊断 | dossier delete 批准；保留 watchdog_task+server_task+handle_shutdown，handle_shutdown 内嵌（engine.shutdown via run_in_executor → server.should_exit） |
| `watchdog_loop` / `terminate_if_errored` | `launcher.py:L144/L156` | 原样 | must_keep；引擎死亡兜底关停 |
| `with_cancellation` | `entrypoints/utils.py:L56` | listen_for_disconnect 用 is_disconnected 替身（无真 ASGI receive 通道） | must_keep；双任务竞速语义原样 |
| `EngineClient` | `engine/protocol.py:L44` | 仅留本章用到的抽象（generate/errored/dead_error/shutdown/...） | must_keep；AsyncLLM 是其 v1 实现（ch04） |

## 替身（stub）一览（均为第三方框架或 ch04 接缝，非 vLLM 杜撰）

- `_framework.py`：FastAPI / APIRouter / Request / JSONResponse / StreamingResponse / CORSMiddleware —— fastapi/uvicorn/starlette 的极小替身，只保留被 vLLM 直接读写的接口语义。
- `_StubRenderer.render_chat_async`：HF/Mistral chat template + tokenizer 渲染的替身（自成体系，本章正交）；给确定 token_ids 以跑通控制流。
- `AsyncLLM.from_vllm_config`：ch04 三段式起引擎的接缝；本章是消费侧，内部留给 ch04。
- `_UvicornServer/_UvicornConfig`：uvicorn.Server/Config 替身，保留 serve/should_exit/shutdown。
- `freeze_gc_heap`：占位 gc.collect()，不引入 GC 调优副作用，语义保留『启动期调用一次』。

## 验证

`python3 -m pytest tests/ -q` → 22 passed（纯单元，不 import vllm）。
测试钉住 dossier 记录的真实可观察行为：request_id 头优先、404、渲染前 errored 抛 dead_error、
SSE 首块 role / [DONE] 哨兵 / include_usage 末块、FINAL_ONLY 聚合、流内异常转 error 帧、
router 的 StreamingResponse vs JSONResponse 分流、Auth 中间件、watchdog terminate、
build_async_engine_client finally shutdown、with_cancellation 返回 handler 结果。
