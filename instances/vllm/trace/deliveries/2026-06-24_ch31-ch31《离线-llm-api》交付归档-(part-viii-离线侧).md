# ch31《离线 LLM API》交付归档 (Part VIII 离线侧)

- **Type**: delivery
- **Chapter**: ch31
- **Date**: 2026-06-24
- **Timestamp**: 2026-06-24T13:49:57Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch31, entrypoints, offline, LLM, SyncMPClient, part8, delivery, approved

## What happened

ch31 reviewer APPROVED 并归档。主线: LLM 离线门面 generate/chat/embed/encode→EngineArgs→LLMEngine。关键澄清: 默认离线路径用 SyncMPClient(后台进程 EngineCore+ZMQ, 因 VLLM_ENABLE_V1_MULTIPROCESSING 默认 True, llm_engine.py:L165-167 强制 enable_multiprocessing=True); InprocClient(真进程内/无 ZMQ) 只是 VLLM_ENABLE_V1_MULTIPROCESSING=0 的回退, 非默认。与 ch04 AsyncLLM 的真正对比 = 同步阻塞 while step() 驱动(_run_engine)+FINAL_ONLY 批量收集 vs 异步事件循环+背景 output_handler+DELTA 流式——该对比对 SyncMPClient 也成立, 不依赖 in-process。chat 走 _render_and_run_requests 打包+warning; completion 走 _render_and_add+单独 run 不 warning。§31.5 终止性证明(N 单调不增→0)+ sorted 按 request_id 排序还原乱序完成。3 张图(01/02/03 含 request-lifecycle)。reviewer 5 条 issue 全 non-blocking+negotiable: embed 块'形态等价于'示意改直贴真源码 L1223-1266、长句轻拆 §31.2/§31.6、§31.5.1'严格减1(至少不增)'自相矛盾改单调不增、建议补 3-请求逐拍 while step() 数值追踪表、图03 中英混排标签微调。

## Why it matters

Part VIII 离线侧关键章, 与 ch04 AsyncLLM 形成同步/异步对照。纠正常见误解: 离线默认不是进程内 InprocClient 而是 SyncMPClient(后台进程+ZMQ), 同步 vs 异步对比不依赖 in-process。

## What to remember

ch31 reviewer APPROVED 并归档。主线: LLM 离线门面 generate/chat/embed/encode→EngineArgs→LLMEngine。关键澄清: 默认离线路径用 SyncMPClient(后台进程 EngineCore+ZMQ, 因 VLLM_ENABLE_V1_MULTIPROCESSING 默认 True, llm_engine.py:L165-167...
