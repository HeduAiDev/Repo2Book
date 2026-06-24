"""测试用最小 fake：一个产出固定增量的 EngineClient + 一个 ChatCompletionRequest 替身。

这些不是被测对象，只是给真实控制流喂数据的夹具，行为对齐 dossier 记录的 vLLM 可观察语义。
"""

from __future__ import annotations

from messages import CompletionOutput, EngineClient, RequestOutput, SamplingParams, RequestOutputKind


class FakeModelConfig:
    max_model_len = 4096


class FakeRenderer:
    tokenizer = object()

    async def render_chat_async(self, messages_list, chat_params, tok_params,
                                prompt_extras=None, skip_mm_cache=False):
        from render_serving import EngineInput
        (messages,) = messages_list
        text = " ".join(m.get("content", "") for m in messages)
        token_ids = list(range(max(len(text), 1)))
        return ([list(messages)], [EngineInput(token_ids)])


class FakeEngine(EngineClient):
    """产出两个增量块（'Hel' → 'Hello'）的引擎；增量语义下末块累积全文。"""

    def __init__(self, deltas=("Hel", "lo"), errored=False, dead=None):
        self.model_config = FakeModelConfig()
        self.renderer = FakeRenderer()
        self.input_processor = None
        self.vllm_config = type("VC", (), {"shutdown_timeout": 1.0})()
        self._errored = errored
        self._dead = dead or RuntimeError("engine dead")
        self._deltas = deltas
        self.shutdown_calls = 0

    @property
    def errored(self) -> bool:
        return self._errored

    @property
    def dead_error(self) -> BaseException:
        return self._dead

    def generate(self, prompt, sampling_params, request_id, **kwargs):
        deltas = self._deltas
        finish_each = sampling_params.output_kind == RequestOutputKind.FINAL_ONLY

        async def gen():
            acc = ""
            n = len(deltas)
            for idx, d in enumerate(deltas):
                acc += d
                last = idx == n - 1
                if finish_each and not last:
                    # FINAL_ONLY：中间块不发，只在末尾给最终（这里直接跳过非末块）。
                    continue
                out = CompletionOutput(
                    index=0,
                    text=acc if finish_each else d,
                    token_ids=list(range(len(d))),
                    finish_reason="stop" if last else None,
                )
                yield RequestOutput(
                    request_id=request_id,
                    prompt_token_ids=getattr(prompt, "token_ids", [0]),
                    outputs=[out],
                    finished=last,
                )
        return gen()

    async def reset_mm_cache(self) -> None:
        pass

    def shutdown(self, timeout=None) -> None:
        self.shutdown_calls += 1

    async def get_supported_tasks(self):
        return ("generate",)


class FakeRequest:
    """ChatCompletionRequest 的最小替身（被 chat handler 读到的字段）。"""

    def __init__(self, model="companion-model", stream=False, tools=None,
                 tool_choice=None, include_usage=False, request_id=None):
        self.model = model
        self.stream = stream
        self.messages = [{"role": "user", "content": "hi"}]
        self.tools = tools
        self.tool_choice = tool_choice
        self.chat_template = None
        self.chat_template_kwargs = None
        self.max_completion_tokens = None
        self.max_tokens = 16
        self.priority = 0
        self.request_id = request_id
        self.stream_options = {"include_usage": True} if include_usage else None

    def to_sampling_params(self, max_tokens, default_params):
        kind = RequestOutputKind.CUMULATIVE if self.stream else RequestOutputKind.FINAL_ONLY
        return SamplingParams(max_tokens=max_tokens, output_kind=kind, n=1)
