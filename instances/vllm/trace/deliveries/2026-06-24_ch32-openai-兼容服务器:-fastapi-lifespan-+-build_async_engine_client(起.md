# OpenAI 兼容服务器: FastAPI lifespan + build_async_engine_client(起 AsyncLLM/AsyncMPClient, 复用 ch04/ch31 异步路径)、OpenAIServing 基类、chat/completion handler、Renderer、SSE 流式(DELTA) vs 非流式 JSON(FINAL_ONLY)、launcher(uvicorn serve + 优雅关停)

- **Type**: delivery
- **Chapter**: 32
- **Date**: 2026-06-24
- **Timestamp**: 2026-06-24T14:33:59Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: ch32, entrypoints, openai-server, fastapi, sse, launcher, part-viii, APPROVED

## What happened

ch32 reviewer 判定 APPROVED 交付。Part VIII entrypoints 服务侧。主线: run_server→build_async_engine_client(@asynccontextmanager, AsyncLLM.from_vllm_config 起异步引擎对接 ch04 三段式/AsyncMPClient)→setup_server(先绑 socket 再起引擎避端口竞争)→build_app(lifespan+中间件+6 exception_handler)→init_app_state(OpenAIServingChat/Render/...)→serve_http(uvicorn+watchdog+handle_shutdown)。核心概念=§32.5『同一个 generate 生成器、两种姿态』: 流式 chat_completion_stream_generator(SSE DELTA 逐 token, 对接 ch04 output_handler) vs 非流式 chat_completion_full_generator(FINAL_ONLY 聚合 JSON), 二者经 to_sampling_params 按 request.stream 设 output_kind(DELTA/FINAL_ONLY)区分语义。22 单元测试通过(host 纯单元, 桩 fake, 不 import vllm); 4/4 linter 全过(保真度/结构/公式/源码根基)。reviewer verdict=APPROVED, 9 条 issue 全 non-blocking+negotiable。

## Why it matters

entrypoints 服务侧收尾, 承接 ch31 离线对照 + ch04 异步三段式/output_handler + ch10 logprobs 装配。bible 登记 5 个精简版接口。foreshadow due 为空(本章非任何伏笔 plant/payoff; 对 ch04/ch31/ch10 的引用均非 bible 强制登记项, 以'此前已细讲、此处点位置'口吻呼应)。

## What to remember

ch32 reviewer 判定 APPROVED 交付。Part VIII entrypoints 服务侧。主线: run_server→build_async_engine_client(@asynccontextmanager, AsyncLLM.from_vllm_config 起异步引擎对接 ch04 三段式/AsyncMPClient)→setup_server(先绑 socket 再...
