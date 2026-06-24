# 精简版（只做减法）—— Responses API 多轮有状态会话
#
# 对应真实源码：
#   vllm/entrypoints/openai/responses/serving.py  (OpenAIServingResponses)
#   vllm/entrypoints/openai/responses/utils.py     (construct_input_messages)
#   vllm/entrypoints/openai/responses/context.py    (HarmonyContext)
# 源码 pin：f3fef123
#
# 本文件抽取"跨轮历史拼接 + 本轮 output 自动留存"主线，与真实源码同名、
# 同结构、同控制流，只删不增。background/streaming/event_store、token-usage
# 统计、tool_server 探测、harmony render 细节等正交特性按 dossier
# subtraction_plan 删除，删除处以 `# SUBTRACTED:` 标注。

from copy import copy


# ============================================================================
# utils.py —— 非 harmony 多轮拼接
# ============================================================================

# SUBTRACTED: 真实 utils.py 顶部 import 了 openai/harmony 的 ResponseOutputItem /
#   ResponseOutputMessage / ChatCompletionMessageParam / ResponseInputOutputItem
#   等类型与 construct_chat_messages_with_tool_call。本章用结构等价的轻量占位
#   保留控制流（isinstance(ResponseOutputMessage)、content.text、role 字段），
#   不杜撰逻辑。原 vllm/entrypoints/openai/responses/utils.py 顶部 import。


class ResponseOutputMessage:
    """SUBTRACTED 占位：真实是 openai-types 的 ResponseOutputMessage（assistant
    输出消息，带 .content 列表，每项有 .text）。原 utils.py:L105。"""

    # SOURCE: openai.types.responses (ResponseOutputMessage) — 经 utils.py:L105 isinstance 判定
    def __init__(self, content):
        self.content = content


class _Content:
    # SOURCE: openai.types.responses (content item with .text) — utils.py:L106
    def __init__(self, text):
        self.text = text


# SOURCE: vllm/entrypoints/openai/responses/utils.py:L79-L121
def construct_input_messages(
    *,
    request_instructions=None,
    request_input,
    prev_msg=None,
    prev_response_output=None,
):
    messages = []
    if request_instructions:
        messages.append(
            {
                "role": "system",
                "content": request_instructions,
            }
        )

    # Prepend the conversation history.
    if prev_msg is not None:
        # Filter out system messages from previous conversation -- per the
        # OpenAI spec, instructions should NOT carry over across responses.
        # The current request's instructions (if any) were already added above.
        messages.extend(m for m in prev_msg if m.get("role") != "system")
    if prev_response_output is not None:
        # Add the previous output.
        for output_item in prev_response_output:
            # NOTE: We skip the reasoning output.
            if isinstance(output_item, ResponseOutputMessage):
                for content in output_item.content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.text,
                        }
                    )

    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request_input, str):
        messages.append({"role": "user", "content": request_input})
    else:
        # SUBTRACTED: construct_chat_messages_with_tool_call(request_input) 把
        #   结构化 input 项转 chat 消息（工具调用展开）。本章多轮主线只需
        #   "list input → extend"控制流，故委派给等价的逐项透传。
        #   原 utils.py:L119。
        input_messages = list(request_input)
        messages.extend(input_messages)
    return messages


# ============================================================================
# context.py —— harmony 会话上下文（_messages 与 msg_store 共享同一 list）
# ============================================================================

# SUBTRACTED: 真实 HarmonyContext 继承 ConversationContext(ABC)，并依赖
#   get_streamable_parser_for_assistant / openai_harmony 的 Message / Author /
#   Role / TextContent / render_for_completion 等（vllm/.../context.py 顶部 +
#   L522 父类）。本章主线只需 _messages 共享 + append_output extend 语义，
#   故父类抽象与 streaming 子类省略，parser 用注入替身驱动。


class HarmonyContext:
    # SOURCE: vllm/entrypoints/openai/responses/context.py:L523-L548 (__init__, 主线字段)
    def __init__(self, messages, available_tools):
        self._messages = messages
        self.finish_reason = None
        self.available_tools = available_tools
        # SUBTRACTED: _tool_sessions / called_tools / parser 初始化与
        #   num_prompt_tokens / num_output_tokens / num_cached_tokens /
        #   num_reasoning_tokens / num_tool_output_tokens / TurnMetrics /
        #   all_turn_metrics / is_first_turn / first_tok_of_message /
        #   kv_transfer_params 等 token 计数旁路字段（context.py:L531-L548）。
        #   本章只读 _messages 共享语义，故省略；下面 append_output 保留 parser
        #   解消息 + extend 控制流，parser 由注入替身提供 .process / .messages。
        self.parser = None
        self.num_init_messages = len(messages)

    # SOURCE: vllm/entrypoints/openai/responses/context.py:L559-L579
    def append_output(self, output) -> None:
        output_token_ids = output.outputs[0].token_ids
        # SUBTRACTED: self.parser = get_streamable_parser_for_assistant() 新建
        #   harmony 流式 parser（context.py:L561）。本章由外部注入 self.parser
        #   替身（提供 .process(token_id) 与 .messages），保留逐 token process
        #   →取 parser.messages→extend 控制流。
        for token_id in output_token_ids:
            self.parser.process(token_id)
            # SUBTRACTED: self._update_num_reasoning_tokens() 统计推理 token
            #   （旁路计数，dossier 批准删）。原 context.py:L565。
        # SUBTRACTED: self._update_prefill_token_usage(output) /
        #   self._update_decode_token_usage(output) token usage 统计旁路
        #   （dossier 批准删）。原 context.py:L566-567。
        if output.kv_transfer_params is not None:
            self.kv_transfer_params = output.kv_transfer_params
        # SUBTRACTED: all_turn_metrics.append(...) / current_turn_metrics.reset()
        #   TurnMetrics 旁路（dossier 批准删）。原 context.py:L571-572。
        # append_output is called only once before tool calling
        # in non-streaming case
        # so we can append all the parser messages to _messages
        output_msgs = self.parser.messages
        # The responses finish reason is set in the last message
        self.finish_reason = output.outputs[0].finish_reason
        self._messages.extend(output_msgs)

    # SOURCE: vllm/entrypoints/openai/responses/context.py:L581-L583
    def append_tool_output(self, output) -> None:
        output_msgs = output
        self._messages.extend(output_msgs)

    # SOURCE: vllm/entrypoints/openai/responses/context.py:L677 (messages property)
    @property
    def messages(self):
        # SOURCE: vllm/entrypoints/openai/responses/context.py:L677 (messages property)
        return self._messages

    # SOURCE: vllm/entrypoints/openai/responses/context.py:L712-L713
    def render_for_completion(self) -> list:
        # SUBTRACTED: 真实委派 harmony 的 render_for_completion(self.messages) 把
        #   消息序列渲染回 token ids 喂引擎（context.py:L713）。本章由注入的
        #   _render 替身完成，保留"消息序列→token ids"调用语义。
        return self._render(self.messages)


# ============================================================================
# serving.py —— OpenAIServingResponses 多轮 handler
# ============================================================================

# SUBTRACTED: 真实 OpenAIServingResponses 继承 OpenAIServing，__init__ 接 engine_client /
#   model_config / tokenizer / tool_server / parser / renderer 等大量依赖
#   （vllm/.../serving.py:L153-L246）。本章只读多轮主线用到的 response_store /
#   msg_store / use_harmony / enable_store 与三个方法，故构造精简为这些字段，
#   由测试注入生成器替身。background_tasks / event_store / response_store_lock
#   的并发包装在本同步精简版中折叠（dossier 批准删 background/streaming）。


# SOURCE: vllm/entrypoints/openai/responses/serving.py:L356 (_make_not_found_error -> ErrorResponse 404)
class _NotFoundError:
    """SUBTRACTED 占位：真实 _make_not_found_error 返回 ErrorResponse(404)。
    原 serving.py:L356。"""

    def __init__(self, response_id):
        # SOURCE: vllm/entrypoints/openai/responses/serving.py:L356
        self.response_id = response_id


class OpenAIServingResponses:
    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L153-L246 (__init__, 多轮主线字段)
    def __init__(self, use_harmony=False, enable_store=True):
        # SOURCE: vllm/entrypoints/openai/responses/serving.py:L215
        self.use_harmony = use_harmony
        self.enable_store = enable_store
        # SOURCE: vllm/entrypoints/openai/responses/serving.py:L235 (response_store)
        self.response_store: dict = {}
        # SOURCE: vllm/entrypoints/openai/responses/serving.py:L241 (msg_store)
        self.msg_store: dict = {}
        # SUBTRACTED: response_store_lock(asyncio.Lock) / event_store /
        #   background_tasks 等（serving.py:L236-L246）。本同步精简版无并发/
        #   background，dossier 批准删。

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L318-L575 (create_responses, 同步多轮主线)
    def create_responses(self, request):
        # SUBTRACTED: _check_model / _validate_create_responses_input /
        #   engine_client.errored 死检（serving.py:L327-339）。
        if request.store and not self.enable_store:
            # Disable the store option.
            # NOTE(woosuk): ... implicitly disable store ...
            request.store = False

        # Handle the previous response ID.
        prev_response_id = request.previous_response_id
        if prev_response_id is not None:
            prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return self._make_not_found_error(prev_response_id)
        else:
            prev_response = None

        # SUBTRACTED: _maybe_get_adapters / models.model_name（lora 适配）
        #   （serving.py:L360-361）。

        if self.use_harmony:
            messages, engine_inputs = self._make_request_with_harmony(
                request, prev_response
            )
        else:
            messages, engine_inputs = self._make_request(request, prev_response)

        # SUBTRACTED: tool_server.has_tool(browser/python/container) 探测填
        #   builtin_tool_list（serving.py:L384-405，dossier 批准删）；available_tools 置空。
        available_tools: list = []

        # SUBTRACTED: per-engine_input 的 sampling_params 构造 / trace_headers /
        #   reasoning_parser 装配 / _generate_with_builtin_tools 调度
        #   （serving.py:L408-492）。本章多轮主线由注入的 _make_generator 替身
        #   产出 (result_generator, context)，保留"messages→context→生成器"控制流。
        result_generator, context = self._make_generator(
            request, messages, available_tools
        )

        # Store the input messages.
        # SOURCE: vllm/entrypoints/openai/responses/serving.py:L497-L499
        if request.store:
            self.msg_store[request.request_id] = messages

        # SUBTRACTED: request.background 分支（_run_background_request[_stream] +
        #   response_store 占位 + background_tasks 注册，serving.py:L501-554，
        #   dossier 批准删）与 request.stream 分支（responses_stream_generator，
        #   serving.py:L556-565）。本章只走非流式同步路径。

        return self.responses_full_generator(
            request, result_generator, context
        )

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L577-L601 (_make_request, 非 harmony)
    def _make_request(self, request, prev_response):
        # SUBTRACTED: construct_tool_dicts(request.tools, request.tool_choice)
        #   （serving.py:L582）。
        # Construct the input messages.
        messages = construct_input_messages(
            request_instructions=request.instructions,
            request_input=request.input,
            prev_msg=self.msg_store.get(prev_response.id) if prev_response else None,
            prev_response_output=prev_response.output if prev_response else None,
        )

        # SUBTRACTED: openai_serving_render.preprocess_chat(...) 把 messages
        #   渲染成 engine_inputs token（serving.py:L591-600）。本章多轮主线只验
        #   "messages 拼接"，engine_inputs 由注入替身或占位返回。
        engine_inputs = self._preprocess_chat(messages)
        return messages, engine_inputs

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L1133-L1210 (_construct_input_messages_with_harmony)
    def _construct_input_messages_with_harmony(self, request, prev_response):
        messages = []
        if prev_response is None:
            # New conversation.
            # SUBTRACTED: extract_tool_types / has_custom_tools /
            #   _construct_harmony_system_input_message / get_developer_message /
            #   construct_harmony_previous_input_messages 建 system/developer 消息
            #   （serving.py:L1141-1153）。本章续轮主线只需"新会话造起始消息"占位。
            messages += self._construct_harmony_new_conversation_messages(request)
        else:
            # Continue the previous conversation.
            # FIXME(woosuk): Currently, request params like reasoning and
            # instructions are ignored.
            prev_msgs = self.msg_store[prev_response.id]

            # FIXME(woosuk): The slice-delete-reappend cycle below is
            # currently a no-op --- it removes messages then puts them all
            # back unfiltered. It may be intentionally deferred (see FIXME
            # above) or redundant if the Harmony encoder already strips
            # analysis messages at render time. If analysis messages need
            # to be dropped here, add a channel != "analysis" filter when
            # re-appending, similar to auto_drop_analysis_messages in
            # harmony_utils.py.
            if len(prev_msgs) > 0:
                last_msg = prev_msgs[-1]
                if last_msg.channel == "final":
                    prev_final_msg_idx = -1
                    for i in range(len(prev_msgs) - 2, -1, -1):
                        prev_msg_i = prev_msgs[i]
                        if prev_msg_i.channel == "final":
                            prev_final_msg_idx = i
                            break
                    recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                    del prev_msgs[prev_final_msg_idx + 1 :]
                    for msg in recent_turn_msgs:
                        prev_msgs.append(msg)
            messages.extend(prev_msgs)
        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            # Skip empty string input when previous_input_messages supplies
            # the full conversation history.
            if request.input or not request.previous_input_messages:
                messages.append(self._get_user_message(request.input))
        else:
            if prev_response is not None:
                prev_outputs = copy(prev_response.output)
            else:
                prev_outputs = []
            for response_msg in request.input:
                # SUBTRACTED: response_input_to_harmony(response_msg, prev_outputs)
                #   把结构化 input 转 harmony 消息、跳过 system、并把工具调用请求
                #   加进 prev_outputs（serving.py:L1200-1209）。本章续轮主线由注入
                #   替身完成，保留"逐项转换 + append"控制流。
                new_msg = self._response_input_to_harmony(response_msg, prev_outputs)
                if new_msg is not None and new_msg.role != "system":
                    messages.append(new_msg)
        return messages

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L744-L893 (responses_full_generator, 落库主线)
    def responses_full_generator(self, request, result_generator, context):
        # SUBTRACTED: AsyncExitStack / _initialize_tool_sessions / async for 驱动
        #   生成（serving.py:L758-764）。本同步精简版直接驱动注入的 result_generator
        #   把 output 喂给 context.append_output（这正是"本轮 output extend 进
        #   共享 _messages"发生处）。
        for output in result_generator:
            context.append_output(output)

        # SUBTRACTED: status 判定 / use_harmony·ParsableContext·SimpleContext 三分支
        #   make_response_output_items / ResponseUsage token 统计
        #   （serving.py:L766-873，dossier 批准删 token-usage）。
        # SOURCE: vllm/entrypoints/openai/responses/serving.py:L874-L885 (from_request)
        response = self._make_response_from_request(request, context)

        # SOURCE: vllm/entrypoints/openai/responses/serving.py:L887-L892
        if request.store:
            stored_response = self.response_store.get(response.id)
            # If the response is already cancelled, don't update it.
            if stored_response is None or stored_response.status != "cancelled":
                self.response_store[response.id] = response
        return response

    # ---- 以下为被 SUBTRACTED 的真实依赖在精简版里的注入点（非 vLLM 抽象，仅替身钩子）----

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L709 (_make_request_with_harmony)
    def _make_request_with_harmony(self, request, prev_response):
        # SUBTRACTED: 真实 _make_request_with_harmony 调
        #   _construct_input_messages_with_harmony 再 render_for_completion
        #   （serving.py:L709，harmony 路径）。本章保留对前者的调用，engine_inputs
        #   由注入 render 替身产出。
        messages = self._construct_input_messages_with_harmony(request, prev_response)
        engine_inputs = self._render_harmony(messages)
        return messages, engine_inputs

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L356 (_make_not_found_error)
    def _make_not_found_error(self, response_id):
        return _NotFoundError(response_id)

    # 下列方法各对应真实源码中一处被 SUBTRACTED 的渲染/转换/调度调用；本精简版
    # 留作注入点（测试覆写），默认实现为最小占位，绝不杜撰 vLLM 逻辑。每个标注
    # 其在真实 serving.py 中被替代的调用点。
    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L480 (_generate_with_builtin_tools 调度) — 注入点
    def _make_generator(self, request, messages, available_tools):
        raise NotImplementedError("inject in test")

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L591 (openai_serving_render.preprocess_chat) — 注入点
    def _preprocess_chat(self, messages):
        return None

    # SOURCE: vllm/entrypoints/openai/responses/serving.py (render_for_completion, harmony 路径) — 注入点
    def _render_harmony(self, messages):
        return None

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L1144-L1153 (_construct_harmony_system_input_message/get_developer_message/construct_harmony_previous_input_messages) — 注入点
    def _construct_harmony_new_conversation_messages(self, request):
        return []

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L1193 (get_user_message) — 注入点
    def _get_user_message(self, text):
        raise NotImplementedError("inject in test")

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L1200 (response_input_to_harmony) — 注入点
    def _response_input_to_harmony(self, response_msg, prev_outputs):
        raise NotImplementedError("inject in test")

    # SOURCE: vllm/entrypoints/openai/responses/serving.py:L874-L885 (ResponsesResponse.from_request) — 注入点
    def _make_response_from_request(self, request, context):
        raise NotImplementedError("inject in test")
