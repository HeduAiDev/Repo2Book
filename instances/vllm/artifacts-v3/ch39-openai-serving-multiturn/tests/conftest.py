# ch38《服务面：OpenAI 协议与多轮》测试配置。
# 行为基准 = 真实 vLLM v0.27.1（6e448d0ea，instances/vllm/source 现核行号）：
#   - vllm/entrypoints/serve/utils/api_utils.py:L37-L206（断连竞速/四方取 min）
#   - vllm/entrypoints/openai/chat_completion/protocol.py:L646-L734（to_sampling_params）
#   - vllm/entrypoints/openai/chat_completion/serving.py:L422-L1138（SSE/非流式）
#   - vllm/parser/abstract_parser.py:L801-L940（两阶段状态机）
#   - vllm/parser/harmony.py:L63-L370（三通道分拣）
#   - vllm/entrypoints/openai/parser/harmony_utils.py:L206-L461（多轮编码）
#   - vllm/v1/engine/async_llm.py:L596-L741 + output_processor.py:L476-L523（abort）
import asyncio
import pathlib
import sys

IMPL_DIR = pathlib.Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))

import pytest  # noqa: E402

from vllm.config import ModelConfig, VllmConfig  # noqa: E402
from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402
from vllm.renderers.base import BaseRenderer  # noqa: E402
from vllm.sampling_params import RequestOutputKind, SamplingParams  # noqa: E402
from vllm.v1.engine.input_processor import InputProcessor  # noqa: E402


# ---------------------------------------------------------------------------
# ch6 边界假件：render_chat_async 是 ch6 渲染四步流水的边界（黑盒回指）。
# 真实行为契约（vllm/renderers/base.py:L1071）：输入 messages 列表 + 参数，
# 输出 (conversation, engine_input) 列表——测试按 tokens 直通承载。
# ---------------------------------------------------------------------------
class FakeRenderer(BaseRenderer):
    def __init__(self, token_ids=None):
        self._token_ids = token_ids if token_ids is not None else [1, 2, 3]
        self.warmed_up = []
        self.tokenizer = None

    def get_tokenizer(self):
        return self.tokenizer

    def warmup(self, chat_params):
        self.warmed_up.append(chat_params)

    async def render_chat_async(
        self,
        messages_list,
        chat_params,
        tok_params,
        *,
        prompt_extras=None,
        skip_mm_cache=False,
    ):
        results = []
        for messages in messages_list:
            from vllm.inputs import tokens_input

            conv = [m for m in messages if isinstance(m, dict)]
            results.append((conv, tokens_input(list(self._token_ids))))
        return results


# ---------------------------------------------------------------------------
# ch4 边界假件：EngineClient 抽象面的测试替身（AsyncLLM 是全仓唯一实现）。
# ---------------------------------------------------------------------------
class FakeEngineClient:
    """EngineClient 测试替身：记录 generate 调用，回放预设 RequestOutput。"""

    def __init__(self, model_config=None, renderer=None, outputs=None, vllm_config=None):
        self.model_config = model_config or ModelConfig(
            model="test-model", max_model_len=1024
        )
        self.vllm_config = vllm_config or VllmConfig(model_config=self.model_config)
        self.renderer = renderer or FakeRenderer()
        self.input_processor = InputProcessor(
            vllm_config=self.vllm_config,
            model_config=self.model_config,
            renderer=self.renderer,
        )
        self.generate_calls = []
        self.outputs = outputs or []

    @property
    def is_running(self):
        return True

    @property
    def is_stopped(self):
        return False

    @property
    def errored(self):
        return False

    @property
    def dead_error(self):
        from vllm.v1.engine.exceptions import EngineDeadError

        return EngineDeadError()

    async def is_tracing_enabled(self):
        return False

    async def do_log_stats(self):
        return None

    async def check_health(self):
        return None

    async def get_supported_tasks(self):
        return ("generate",)

    def shutdown(self, timeout=None):
        return None

    def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.generate_calls.append(
            {"prompt": prompt, "params": sampling_params, "request_id": request_id,
             **kwargs}
        )
        outputs = self.outputs

        async def _gen():
            for out in outputs:
                yield out

        return _gen()


def make_serving_chat(engine_client=None, model_type="llama", reasoning_parser="",
                      tool_parser=None, enable_auto_tools=False,
                      enable_prompt_tokens_details=False,
                      enable_force_include_usage=False):
    """OpenAIServingChat 测试装配（构造签名 = serving.py:L111-L133）。"""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.renderers.online_renderer import OnlineRenderer

    engine = engine_client or FakeEngineClient(
        model_config=ModelConfig(model="test-model", model_type=model_type,
                                 max_model_len=1024)
    )
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name="test-model", model_path="test-model")],
    )
    renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=enable_auto_tools,
        tool_parser=tool_parser,
        reasoning_parser=reasoning_parser,
    )
    return OpenAIServingChat(
        engine,
        models,
        "assistant",
        online_renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        reasoning_parser=reasoning_parser,
        enable_auto_tools=enable_auto_tools,
        tool_parser=tool_parser,
        enable_prompt_tokens_details=enable_prompt_tokens_details,
        enable_force_include_usage=enable_force_include_usage,
    )


def make_request(**overrides):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    base = dict(messages=[{"role": "user", "content": "hi"}], model="test-model")
    base.update(overrides)
    return ChatCompletionRequest(**base)


def make_output(text="", token_ids=None, finish_reason=None, index=0,
                stop_reason=None):
    return CompletionOutput(
        index=index,
        text=text,
        token_ids=token_ids if token_ids is not None else [],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


def make_res(prompt_token_ids=(1, 2, 3), outputs=None, finished=False,
             num_cached_tokens=None, num_cache_creation_tokens=None):
    return RequestOutput(
        request_id="req-1",
        prompt=None,
        prompt_token_ids=list(prompt_token_ids),
        prompt_logprobs=None,
        outputs=outputs if outputs is not None else [],
        finished=finished,
        num_cached_tokens=num_cached_tokens,
        num_cache_creation_tokens=num_cache_creation_tokens,
    )


# ---------------------------------------------------------------------------
# harmony token 工具：用真实 openai_harmony 编码产 token 流（与
# harmony_utils.get_encoding 同源——编码即真相）。
# ---------------------------------------------------------------------------
def harmony_ids(user_msg="hi", assistant_messages=None):
    """渲染 [user 消息 + assistant 各通道消息] 的 completion token ids。

    assistant_messages: list[(channel, recipient_or_None, content, content_type)]
    """
    from openai_harmony import Conversation, Message, RenderConversationConfig, Role
    from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

    msgs = [Message.from_role_and_content(Role.USER, user_msg)]
    for channel, recipient, content, content_type in assistant_messages or []:
        m = Message.from_role_and_content(Role.ASSISTANT, content)
        m = m.with_channel(channel)
        if recipient:
            m = m.with_recipient(recipient)
        if content_type:
            m = m.with_content_type(content_type)
        msgs.append(m)
    enc = get_encoding()
    conv = Conversation.from_messages(msgs)
    return enc.render_conversation_for_completion(
        conv, Role.ASSISTANT,
        config=RenderConversationConfig(auto_drop_analysis=False),
    )


def assistant_token_span(total_ids, n_prompt_ids=None):
    """从 completion ids 里剥出 assistant 生成段（<|start|>assistant 之后）。"""
    from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

    enc = get_encoding()
    start_tok = enc.encode("<|start|>assistant")
    # 最后一处 <|start|>assistant 之后即生成段（preamble/user 之后）
    start = max(
        i for i in range(len(total_ids) - len(start_tok) + 1)
        if total_ids[i:i + len(start_tok)] == start_tok
    ) + len(start_tok)
    return total_ids[start:]


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.get_event_loop_policy()
