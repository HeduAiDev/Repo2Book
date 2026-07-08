# 实例：vLLM（vLLM v1 源码解读：从真实源码读懂推理引擎）

> 本文件 = 当前实例的「源码版本 + 当前状态 + 实例专属规则」。**通用方法论/工厂运转见仓库根 `CLAUDE.md`**；本实例配置见 `instances/vllm/repo2book.json`。
> 中文、高级读者，解读 vLLM v1 推理引擎；异步三段式解耦为旗舰。

> **2026-06-21 重大转向**：旧的"从零简化重写 + 理论推导"产出太抽象、脱离代码，**全部废弃**。新书 = **直接解读真实 vLLM v1 源码**，按真实模块组织。旧 agent 提示词/经验仅作批判参考。

## 源码版本（行号基线）
- 书锚定 vLLM **v0.21.0**（发布提交 `ad7125a4`，`instances/vllm/source/` 工作树即此版）。
- 全书 ~3000 处精确行号引用已由 `scripts/remap_lines_v021.py` 确定性重映射到 v0.21.0（difflib 行级对齐：平移类自动改号、内容真改处定点重抽 v0.21.0 片段）；v0.21.0 引入的面向读者新特性另以「v0.21.0 更新」注织入。
- `f3fef123`（v0.20.1 线、v0.21.0 前 245 提交）= 升级前基线，仅作历史 diff（remapper 据此工作，可复用于未来升级）。少数被 v0.21.0 删除的代码示例（如 RMSNorm `forward_static`）显式标注为 `(基线 f3fef123)`。

## 实例专属硬规则
- **vLLM 相关代码调试一律进 Docker 容器**（host 无 CUDA/vLLM）：`scripts/vllm_docker.sh ...`，镜像 `vllm/vllm-openai:latest`。容器内 vLLM 版本可能与 v0.21.0 有行号差，仅用于观察行为。
- 正文规范路径前缀 `vllm/…`（**绝不** `instances/vllm/source/…`）。
- 架构地图 + 大纲在 `instances/vllm/book/cartography/`（`ARCHITECTURE.md` 全量、`outline-final.json` 8 Part/33 章、`map.json` 结构化）。章节用 `ch`-前缀 slug，置于 `instances/vllm/artifacts/`。

## 当前状态（2026-06-28）：📕 全书 33 章完成 + 已全量重基到 v0.21.0
- **ch01–ch36 全部完成**（31 源码解读章 + ch01/ch02/ch28 三个 meta 概览章，meta 走 `skip_impl` 轻流程无精简版）。全 APPROVED、已推 `vllm-book-v2-rebuild` 远程。规模 ≈24.5k 行正文 + 142 图。
- **v0.21.0 升级（2026-06-28）**：① 16 章自然织入「v0.21.0 更新」内容（ch03/08/09/17/18/19/23/24/25/26/27/28/29/31/32 + ch29 XPU 分发图）；② `source/` 工作树 checkout 到 v0.21.0，~3000 处行号引用确定性重映射（1458 identity + 1531 shift + 41 content-resync）；③ bible glossary+interfaces 登记 v0.21.0 增量。整理稿见 `instances/vllm/book/_v021-update/`。
- **连贯性干净**：26/26 伏笔全回收；glossary/interfaces 登记 30 章；全书 0 断裂章内锚点 / 0 半角标点 / 0 图几何问题。
- 体系经实战加固：archive 注入完整 reviewV+崩溃重试、防假通过 escape hatch、dossier-verify 对抗自核（实战拦下 ch34 SyncMPClient / ch01 CompilationMode 等事实错误）、off-spine 分层 roadmap、git push 须前台。
- **润色已大体完成**：断锚/半角/图标签重叠/术语对齐 glossary、算法维度增补 24 章均已做并推远程。剩余仅最低价值声线微调 + lint_formulas 内联密度软噪声。

## 实例专属坑
1. 别写脱离代码的抽象——正文以真实 vllm 源码为主线、自包含内嵌。
2. implementer 别过度删减/误删——只删 `delete` 批准项，`must_keep` 必保留。
3. 标记完成前跑全部 linter。
4. vLLM 相关运行进容器；行号以 v0.21.0（`ad7125a4`）为准，升级前基线 `f3fef123` 仅作历史 diff。
5. 别赌自己的上下文——决策/状态写进 trace、Bible、本文件。

## 源码事实备忘（原 knowledge/ 归并，2026-07-04）
ch04（async-engine，`vllm/v1/engine/`）：
- `AsyncLLM.__init__`（L70,L132-153）一次性构好三段：InputProcessor/OutputProcessor 进程内（stage1/3），EngineCore 经 `EngineCoreClient.make_async_mp_client` 走独立进程（stage2）；源码注释原话区分 "this process" vs "separate process"。
- 三段扇出点 `AsyncLLM._add_request`（L400-415）：`output_processor.add_request(...)` 进程内登记（L409）+ `await engine_core.add_request_async(request)` 发去独立 EngineCore 进程（L412）。
- `RequestOutputCollector`（`output_processor.py:L45-106`）**不是** `asyncio.Queue`：单槽 `self.output`(单条) + `self.ready`(Event)；`put()` 置位 Event，消费者跟不上时用 `self.output.add(output, aggregate=...)` 合并而非排队；`get()` 等 `ready.wait()`，`get_nowait()` 是非阻塞快路径。
- `generate()` 消费循环核心（`async_llm.py:L524-635`）：`out = q.get_nowait() or await q.get()`——先非阻塞取，空了才 await（注释：避免高负载下任务切换）；`out.finished` 收尾；`CancelledError`/`GeneratorExit` → `self.abort`（L591-593）。
- `_run_output_handler`（L637-707）是生产者侧：单个长驻后台 `asyncio.Task`；循环 `await engine_core.get_output_async()`（L660）→ 按 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 分块 → `output_processor.process_outputs()`（L675）→ 块间 `await asyncio.sleep(0)`（L683）；`engine_core`/`output_processor` 捕成局部变量（L643-645）避免闭包反向引用 `self` 挡 GC。
- `output_handler` 懒/急两种启动（L170-176,L373,L640-641）：`__init__` 先尝试 `asyncio.get_running_loop()`（急启动），拿不到就吞掉 `RuntimeError`；首次 `add_request`（L373）懒启动；`_run_output_handler`（L640-641）已启动则提前返回（幂等）——为了让 `__init__` 能在事件循环存在之前跑（OpenAI server 启动场景）。
- 输出多路分发（demux）= `OutputProcessor.process_outputs`（`output_processor.py:L572-660`）：遍历 `EngineCoreOutputs`，按 `req_id`（L603）查 `RequestState`（L604，查不到=已 abort 则跳过），`if req_state.queue is not None: req_state.queue.put(request_output)`（L655-657）分发回该请求自己的队列；else 分支（L658-660）走同步 `LLMEngine` 路径。一批 EngineCore 输出可扇出到 N 个请求队列。
- IPC 接缝（`core_client.py:L990-999,L1058-1061`）恰好两个 `AsyncMPClient` 方法：`add_request_async`（编码+经 ZMQ input_socket `_send_input`）与 `get_output_async`（await `self.outputs_queue`，由后台 `process_outputs_socket` 任务喂）；`AsyncLLM` 只看得到这两个 await，ZMQ/msgpack/进程管理全部藏在后面。
- 跨进程消息是 `msgspec.Struct`（`vllm/v1/engine/__init__.py:L80-131,L161-191`）：`EngineCoreRequest`（进：tokenized prompt_token_ids + sampling_params）、`EngineCoreOutput`（出：request_id + new_token_ids + finish_reason；`.finished` 属性 L189 驱动 `generate` 停止）；`array_like`/`omit_defaults`/`gc=False` 压缩序列化。
- `STREAM_FINISHED`（`vllm/outputs.py:L192`）是仅用于**流式输入**场景的哨兵 `RequestOutput(finished=True)`，解除 `generate` 循环阻塞；`generate` L585 跳过 yield 它。


## 原理篇交错与 gap 治理(2026-07-08 收官)
- 36 章新序:三章原理篇归位(ch24 FlashAttention 原理/ch26 量化数学/ch30 EAGLE),映射存档 book/cartography/renumber-2026-07-06.json;补章走 RUNBOOK「补章发车 SOP」。
- 跨章链接三规生效(../../ 两层+文字号=目录号);全书节号=目录号;concepts.json 140 条建账。
- 诊断→治理闭环:2026-07-06 全书审计 cliffs=1(FA 黑盒)+71 bumps → FA 原理章+交错+接缝+inline 引用批次(Orca/Sarathi/PagedAttention/DeepSeek/XGrammar)→ 2026-07-08 定向终验受影响 9 章 cliffs=0。bump 级 advisory 留审计报告待日后 triage。
