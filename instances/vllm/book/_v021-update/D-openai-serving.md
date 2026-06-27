# v0.21.0 更新摘要 — OpenAI 兼容服务（File Group D，目标章节 ch32）

基线 `f3fef1235` → 标签 `v0.21.0`。范围：`vllm/entrypoints/openai/{api_server.py, engine/serving.py, chat_completion/serving.py}` + `launcher.py`，并连带审视同子系统的 `completion/serving.py`、`*/protocol.py`、`utils.py`。

`launcher.py` 在该区间内**无任何提交**——ch32 的 §32.7（优雅关停 / watchdog / `shutdown_task`）无需改动。
`git diff --stat` 确认 `vllm/entrypoints/openai/` 下**无新增 endpoint 文件、无新路由**（name-status 全 `M`）。所以本次更新都是既有 serving 路径上的行为/签名变化，而非新页面。

---

## 1. KV connector 拒绝清理：`_with_kv_transfer_rejection_cleanup` 包裹 create_* 协程

- class: **BEHAVIOR-CHANGE**
- v0.21.0 anchor: `vllm/entrypoints/openai/engine/serving.py:L623` `OpenAIServing._with_kv_transfer_rejection_cleanup`；构造期 `self.has_kv_connector`（`engine/serving.py:L162`）；调用点 `vllm/entrypoints/openai/chat_completion/serving.py:L237`（`create_chat_completion` 现在只是把 `_create_chat_completion(...)` 套进这个包裹器；`completion/serving.py` 同构，新增 `_create_completion`）。
- target: ch32（§32.5.3「错误处理的真相源：基类 `OpenAIServing`」最贴切，可作该节的一个补丁段）。
- 集成（书声）：在分离架构下，`do_remote_prefill` 的请求会让 P 节点预占远端 KV 块；若该请求在抵达引擎前就被拒（抛错或返回 `ErrorResponse`），这些块会成为孤儿。v0.21.0 把每个 `create_*` 入口拆成「公开方法 = `_with_kv_transfer_rejection_cleanup` 包裹器 + `_create_*` 真身」两层：包裹器仅在 `has_kv_connector` 且请求带 `do_remote_prefill` 时生效，用 `try/finally` 在请求未触达引擎时回调 `engine_client.notify_kv_transfer_request_rejected(...)` 通知连接器释放被钉住的远端块。这把「错误即资源泄漏」从隐患变成了显式补偿路径，正好呼应 §32.5.3 里「基类统一兜错」的论点。
- diagram impact：可选。若给 §32.5.3 配「请求一生」泳道图，可加一条「拒绝 → 通知 KV connector 释放」的补偿回边；非必须。

## 2. 流式 tool_choice 大重构：required/named 收敛进 DelegatingParser

- class: **BEHAVIOR-CHANGE**（实现合并，reader-facing 输出语义不变，但 ch32 现引用的代码块已不存在）
- v0.21.0 anchor: `vllm/entrypoints/openai/chat_completion/serving.py` `chat_completion_stream_generator`。删除了 `OpenAIServingChat.extract_tool_call_required_streaming` 方法、`tool_choice_uses_parser` 分支、`function_name_returned` 数组，以及约 130 行「named / required」手写流式解析分支；改为统一走 `parser.parse_delta(...)`（解析器内部 `_stream_state` 携带 `tool_call_id_type` / `history_tool_call_cnt`）。`engine/serving.py` 顶部不再 import `extract_named_tool_call_streaming` / `extract_required_tool_call_streaming`。
- target: ch32。
- 集成（书声）：v0.21.0 把「required / named tool_choice」的流式增量解析从 `OpenAIServingChat` 内的专用分支整体下沉进 `DelegatingParser`，serving 层不再自己 `extract_*`，而是无差别调用 `parser.parse_delta`。ch32 §32.5.1 若引用过这些已删分支需核对：现行控制流是「门控条件 = auto / function_name / `required` / reasoning」时初始化 per-choice parser，随后统一委派。这是**移动 + 合并**，对 SSE 输出无可观察行为差异——若章节正文未逐行讲 required-streaming，可仅在脚注标注「v0.21.0 起统一走 parser.parse_delta」；标 **SKIP** 级别的正文改动，但行号引用需复核。
- diagram impact：无。

## 3. 新请求字段 `return_prompt_text`：回显 chat 模板渲染后的 prompt

- class: **NEW-FEATURE**
- v0.21.0 anchor: 请求 `vllm/entrypoints/openai/chat_completion/protocol.py` `ChatCompletionRequest.return_prompt_text`；响应 `ChatCompletionResponse.prompt_text` / `ChatCompletionStreamResponse.prompt_text`；serving 填充点 `chat_completion/serving.py:L512`（流式，仅首 chunk `res.prompt`）与 `:L1379`（非流式 `final_res.prompt`）。
- target: ch32（§32.5「同一个生成器，两种姿态」——正好对照流式只在首 chunk 给、非流式末尾给）。
- 集成（书声）：v0.21.0 新增请求开关 `return_prompt_text`，置真时响应携带 `prompt_text`——即 chat template 渲染后、真正喂给模型的那串 prompt 文本，便于调试「我到底发进去了什么」。它与 §32.5 的两种姿态天然对偶：流式只在**首个 SSE chunk** 写 `prompt_text`（之后的增量 chunk 故意留空），非流式则在聚合后从 `final_res.prompt` 一次性写入。可作为 §32.5 的一个小专栏，强化「同一生成器、两种装配节奏」的主题。
- diagram impact：可选，§32.5 对照表里加一行「`prompt_text`：流式=首 chunk / 非流式=末尾」即可。

## 4. `truncate_prompt_tokens` 现纳入 `max_tokens` 计算

- class: **BEHAVIOR-CHANGE**
- v0.21.0 anchor: `vllm/entrypoints/utils.py` `get_max_tokens(..., truncate_prompt_tokens=None)`；调用点 `chat_completion/serving.py:L297`、`completion/serving.py:L163`、`responses/serving.py` 均新传 `truncate_prompt_tokens=request.truncate_prompt_tokens`。
- target: ch32（§32.4「从 HTTP 到 token」已引 `vllm/entrypoints/utils.py:L56-L98` 讲 `get_max_tokens`，正中靶心）。
- 集成（书声）：旧版 `get_max_tokens` 用**未截断**的输入长度反推可生成的 `max_tokens`，当请求带 `truncate_prompt_tokens` 时会低估剩余预算。v0.21.0 给 `get_max_tokens` 增参 `truncate_prompt_tokens`：先把 `input_length` 夹到截断上限（`-1` 表示用 `max_model_len`），再做 `max_model_len < input_length` 越界检查与余量计算。ch32 §32.4 若展开了 `get_max_tokens` 的余量公式，需补这一步夹取。
- diagram impact：无。

## 5. forced tool_choice 容忍空 content（不再 assert 崩）

- class: **BEHAVIOR-CHANGE**
- v0.21.0 anchor: `vllm/entrypoints/openai/engine/serving.py` `OpenAIServing._parse_tool_calls_from_content`：两处 `assert content is not None` 改为 `if content is None: return [], None`。
- target: ch32（§32.5.3 错误处理 / 鲁棒性语境下一句带过即可）。
- 集成（书声）：强制函数调用（Responses API / named tool_choice）此前在模型吐空 content 时会因 `assert content is not None` 直接抛 AssertionError；v0.21.0 改为优雅返回空 tool_calls，避免一个边界输出打穿请求。属鲁棒性微调，ch32 可不单列，必要时在 §32.5.3 末尾一句脚注。
- diagram impact：无。

## 6. MoE routed_experts 回显字段（routed_experts / prompt_routed_experts）

- class: **NEW-FEATURE**（与 ch32 serving 主线弱相关）
- v0.21.0 anchor: `chat_completion/protocol.py` `ChatCompletionResponseChoice.routed_experts`、`ChatCompletionResponse.prompt_routed_experts`；`completion/protocol.py` 同名字段；填充于 `chat_completion/serving.py:L1101/L1327/L1371` 与 `completion/serving.py`。
- target: ch32（仅提及级别，可 **SKIP** 进正文）。
- 集成（书声）：新增可选回显字段 `routed_experts`（生成段，`[gen_len, num_layers, top_k]`）与 `prompt_routed_experts`（prompt 段），用于把每 token 命中的 MoE 专家路由暴露给调用方做追踪。这属 MoE 可观测性而非 OpenAI 服务主线，ch32 无需展开；若 MoE 专章涉及路由可在那里引用。标 **SKIP**（对 ch32 而言）。
- diagram impact：无。

---

## 不影响 ch32 / 标 SKIP 的其余改动

- `vllm/entrypoints/openai/api_server.py:L321+` `init_app_state` 新增一段：从 `vllm_config.structured_outputs_config.enable_in_reasoning` 调 `set_enable_structured_outputs_in_reasoning(...)`，把该 flag 跨进程传到 API-server（engine core 在独立进程，contextvar 在此栈为 None）。属结构化输出 / 工具调用 reasoning 的跨进程配置传播（commit 844df5426，xgrammar 0.2.0 structural tags），与 ch32 讲的「HTTP→AsyncLLM 桥」主线无关。**SKIP**（如有结构化输出/工具调用专章则归彼处）。
- `responses/protocol.py`、`responses/serving.py`、`speech_to_text.py` 的改动不在本组 ch32 范围（Responses / 音频转写各属其页面）。**SKIP**。
