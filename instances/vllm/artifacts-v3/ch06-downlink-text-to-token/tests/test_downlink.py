"""TDD tests for the v3 ch06 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).

These assert the *observable vLLM behavior* the chapter teaches — the API
process downlink swimlane: a user's text becomes token ids, is assembled into
an EngineCoreRequest, gets its dual-track request id, and leaves the process
(through the same handoff points the real AsyncLLM uses). Runs on a plain CPU
host without the vllm package; the only stand-ins are the documented HOST
SEAMs in downlink.py (tokenizer / config / msgspec / mm-processor / engine
core client), each anchored to the real vLLM interface it mirrors:

- m1  render pipeline: render_messages -> tokenize -> extras ->
      process_for_engine (chat & completion isomorphic; async = gather);
- m2  thread-pool offload: raw-prompt preprocessing (tokenize / mm) runs on
      the renderer's ThreadPoolExecutor, never on the event loop;
      AsyncLLM.add_request dispatches rendered-EngineInput (sync fast path,
      zero tokenizer calls) vs raw prompt (await process_inputs_async);
- m3  dual pools: _executor = renderer_num_workers workers (tokenize),
      _mm_executor = exactly 1 worker (mm, P0/P1 order per #38418);
- m4  born-tokenized: EngineCoreRequest has NO text field (#11963); raw
      prompt fallback still works but logs the deprecation warning;
- m5  dual-track request id: external_req_id <- user id, request_id <- user
      id + "-" + 8 random hex chars (PR #27987);
- m6  params clone & completion: max_tokens defaults to max_model_len -
      seq_len; update_from_generation_config injects eos; bad words via
      update_from_tokenizer; caller's params object never mutated;
- m7  validation chain: params task routing / LoRA / data_parallel_rank /
      empty prompt / over-length / equal-length / vocab bound;
- m8  mm flatten: dict-of-list mm inputs -> list[MultiModalFeatureSpec]
      sorted by placeholder offset; cache hit -> data=None (skip IPC);
      identifier gains LoRA prefix with tower-connector LoRA;
- m9  PlaceholderRange offset/length (+is_embed mask, get_num_embeds) and
      the encoder-cache budget pre-check;
- m10 EngineInput family: TokensInput / EmbedsInput (must .cpu(), 3D batch
      squeeze) / MultiModalInput / enc_dec split via split_enc_dec_input;
- m11 departure: _add_request double registration (local OutputProcessor
      BEFORE cross-process engine core) -> add_request_async stamps
      client_index and sends EngineCoreRequestType.ADD (b"\\x00").

Run:  cd instances/vllm/artifacts-v3/ch06-downlink-text-to-token
      python -m pytest tests/ -q
"""

import asyncio
import hashlib
import importlib
import re
import sys
import threading
import time
from pathlib import Path

import pytest

_IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(_IMPL_DIR))

downlink = importlib.import_module("downlink")


# ---------------------------------------------------------------------------
# Test-side seams (the parts vLLM itself never implements: a toy word
# tokenizer standing in for the HF tokenizer, and a chat-template renderer
# standing in for HfRenderer.render_messages — the template engine is the
# black box this chapter deliberately does not open).
# ---------------------------------------------------------------------------


class SeamWordTokenizer:
    """Deterministic word tokenizer with the TokenizerLike surface the kept
    code touches: __call__ -> {"input_ids": [...]}, encode/decode, special
    token ids, max_token_id, max_chars_per_token, truncation_side."""

    is_fast = False
    supports_grammar = False
    truncation_side = "right"
    max_chars_per_token = 4
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    VOCAB_SIZE = 100

    MARKER_IDS = {"<image>": 31, "<audio>": 32}

    def __init__(self, delay: float = 0.0, order: list | None = None):
        self.calls: list[tuple[int, str]] = []
        self._delay = delay
        self._order = order if order is not None else []
        self._ids = {"<bos>": 1, "<eos>": 2, "<pad>": 0, **self.MARKER_IDS}
        self._next_id = 3

    def _word_id(self, word: str) -> int:
        if word not in self._ids:
            self._ids[word] = self._next_id
            self._next_id += 1
        return self._ids[word]

    def __call__(self, text: str, add_special_tokens: bool = True,
                 truncation=None, max_length=None, **kwargs):
        self.calls.append((threading.get_ident(), text))
        self._order.append("tokenize")
        if self._delay:
            time.sleep(self._delay)
        ids = [self._word_id(w) for w in text.split()]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if truncation and max_length is not None:
            # right-side truncation (the seam tokenizer's default side)
            ids = ids[:max_length]
        return {"input_ids": ids}

    def encode(self, text: str, add_special_tokens: bool = False):
        self._order.append("encode")
        return [self._word_id(w) for w in text.split()]

    def decode(self, token_ids) -> str:
        rev = {v: k for k, v in self._ids.items()}
        return " ".join(rev.get(int(t), "?") for t in token_ids)

    @property
    def max_token_id(self) -> int:
        return self.VOCAB_SIZE - 1


class ChatRendererSeam(downlink.BaseRenderer):
    """Concrete renderer for tests: render_messages applies a toy chat
    template (join message texts; image/audio content blocks become
    <image>/<audio> markers + multi_modal_data) — the same contract
    HfRenderer.render_messages fulfills for the HF template engine."""

    def render_messages(self, messages, params):
        conversation = []
        parts = []
        mm_data: dict[str, list] = {}
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                text_bits = []
                for block in content:
                    if block["type"] == "text":
                        text_bits.append(block["text"])
                    elif block["type"] == "image_url":
                        mm_data.setdefault("image", []).append(block["image_url"])
                        text_bits.append("<image>")
                    elif block["type"] == "input_audio":
                        mm_data.setdefault("audio", []).append(block["input_audio"])
                        text_bits.append("<audio>")
                    else:
                        raise ValueError(f"unknown content block {block['type']}")
                parts.append(f"{role}: {' '.join(text_bits)}")
            else:
                parts.append(f"{role}: {content}")
            conversation.append({"role": role, "content": content})

        prompt = downlink.TextPrompt(prompt="\n".join(parts))
        if mm_data:
            prompt["multi_modal_data"] = mm_data
        return conversation, prompt


class TracingRenderer(ChatRendererSeam):
    """Records the pipeline steps the base class drives, in order."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.trace: list[str] = []
        tokenizer._order = self.trace

    def render_messages(self, messages, params):
        self.trace.append("render_messages")
        return super().render_messages(messages, params)

    def _process_multimodal(self, *args, **kwargs):
        self.trace.append("mm_preprocess")
        return super()._process_multimodal(*args, **kwargs)


class FakeChatRequest:
    """ChatCompletionRequest stand-in carrying the fields OnlineRenderer
    reads, with the real build_chat_params/build_tok_params bodies
    (chat_completion/protocol.py:L555-L610)."""

    def __init__(self, messages, *, tools=None, tool_choice=None,
                 chat_template=None, chat_template_kwargs=None,
                 max_tokens=None, max_completion_tokens=None,
                 truncate_prompt_tokens=None, truncation_side=None,
                 add_special_tokens=False, echo=False, return_token_ids=False,
                 return_token_offsets=False, add_generation_prompt=True,
                 continue_final_message=False, documents=None,
                 reasoning_effort=None, media_io_kwargs=None,
                 return_assistant_tokens_mask=False, response_format=None,
                 mm_processor_kwargs=None, cache_salt=None,
                 suffix=None, prompt=None, prompt_embeds=None,
                 prompt_logprobs=None):
        self.messages = messages
        self.tools = tools
        self.tool_choice = tool_choice
        self.chat_template = chat_template
        self.chat_template_kwargs = chat_template_kwargs
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.truncate_prompt_tokens = truncate_prompt_tokens
        self.truncation_side = truncation_side
        self.add_special_tokens = add_special_tokens
        self.echo = echo
        self.return_token_ids = return_token_ids
        self.return_token_offsets = return_token_offsets
        self.add_generation_prompt = add_generation_prompt
        self.continue_final_message = continue_final_message
        self.documents = documents
        self.reasoning_effort = reasoning_effort
        self.media_io_kwargs = media_io_kwargs
        self.return_assistant_tokens_mask = return_assistant_tokens_mask
        self.response_format = response_format
        self.mm_processor_kwargs = mm_processor_kwargs
        self.cache_salt = cache_salt
        # completion-face fields (render_completion reads these)
        self.suffix = suffix
        self.prompt = prompt
        self.prompt_embeds = prompt_embeds
        self.prompt_logprobs = prompt_logprobs

    def build_chat_params(self, default_template, default_template_content_format):
        extra_kwargs = dict(
            add_generation_prompt=self.add_generation_prompt,
            continue_final_message=self.continue_final_message,
            documents=self.documents,
            reasoning_effort=self.reasoning_effort,
        )
        user_kwargs = self.chat_template_kwargs or {}
        return downlink.ChatParams(
            chat_template=self.chat_template or default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=downlink.merge_kwargs(
                self.chat_template_kwargs, extra_kwargs,
            ),
            media_io_kwargs=self.media_io_kwargs,
            return_assistant_tokens_mask=bool(self.return_assistant_tokens_mask),
            tool_choice=self.tool_choice if self.tools else None,
            response_format=self.response_format,
        )

    def build_tok_params(self, model_config):
        if self.max_completion_tokens is not None:
            max_output_tokens = self.max_completion_tokens
        else:
            max_output_tokens = self.max_tokens
        return downlink.TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=max_output_tokens or 0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            truncation_side=self.truncation_side,
            add_special_tokens=self.add_special_tokens,
            needs_detokenization=bool(self.echo and not self.return_token_ids),
            return_token_offsets=bool(self.return_token_offsets),
        )


def mk_config(**over):
    """Build the seam VllmConfig (the EngineArgs assembly line is ch03)."""
    model_over = dict(over.pop("model_config", {}) or {})
    parallel_over = dict(over.pop("parallel_config", {}) or {})
    cache_over = dict(over.pop("cache_config", {}) or {})
    lora_over = over.pop("lora_config", None)
    mm_over = over.pop("multimodal_config", None)
    if mm_over is None:
        mm_over = model_over.pop("multimodal_config", None)
    hf_over = model_over.pop("hf_config", None)
    model = downlink.ModelConfig(**model_over)
    if mm_over is not None:
        model.multimodal_config = downlink.MultimodalConfig(**mm_over)
    if hf_over is not None:
        model.hf_config = downlink.HFConfig(**hf_over)
    cfg = downlink.VllmConfig(
        model_config=model,
        parallel_config=downlink.ParallelConfig(**parallel_over),
        cache_config=downlink.CacheConfig(**cache_over),
        lora_config=None if lora_over is None else downlink.LoRAConfig(**lora_over),
    )
    assert not over, f"unconsumed config keys: {over}"
    return cfg


def mk_renderer(tokenizer, config=None, *, trace=False):
    cfg = config or mk_config()
    cls = TracingRenderer if trace else ChatRendererSeam
    return cls(cfg, tokenizer)


def mk_engine(tokenizer=None, *, config=None, renderer=None, client_index=0,
              supported_tasks=("generate",)):
    config = config or mk_config()
    tokenizer = tokenizer or SeamWordTokenizer()
    renderer = renderer or mk_renderer(tokenizer, config)
    llm = downlink.AsyncLLM(
        config, renderer, client_index=client_index,
        supported_tasks=supported_tasks,
    )
    return llm, renderer, tokenizer


def engine_input_from(engine_input):
    return engine_input  # readability alias in tests


# ---------------------------------------------------------------------------
# m1 — the four-step render pipeline (chat & completion isomorphic)
# ---------------------------------------------------------------------------


def test_render_chat_four_step_pipeline_order():
    """render_chat_async drives render_messages -> tokenize -> (extras) ->
    process_for_engine in order, and the product is a typed EngineInput."""
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer, trace=True)
    request = FakeChatRequest(messages=[{"role": "user", "content": "hello world"}])

    online = downlink.OnlineRenderer(
        mk_config().model_config, renderer,
        request_logger=None, chat_template=None,
        chat_template_content_format="string",
    )
    conversation, engine_inputs = asyncio.run(online.render_chat(request))

    assert renderer.trace[:1] == ["render_messages"]
    assert "tokenize" in renderer.trace
    # a pure text request never enters the mm step
    assert "mm_preprocess" not in renderer.trace
    assert renderer.trace.index("render_messages") < renderer.trace.index("tokenize")

    (engine_input,) = engine_inputs
    assert engine_input["type"] == "token"
    ids = engine_input["prompt_token_ids"]
    # add_special_tokens=False for chat (build_tok_params chat default)
    assert ids == tokenizer.encode("user: hello world")
    # arrival_time is stamped at render entry (theory: TTFT clock starts at
    # "text arrives", not "engine admits")
    assert "arrival_time" in engine_input
    assert len(conversation) == 1


def test_render_cmpl_isomorphic_sync_face():
    """The completion face runs the same four steps synchronously."""
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    engine_inputs = renderer.render_cmpl(
        [downlink.TextPrompt(prompt="one two three")],
        downlink.TokenizeParams(max_total_tokens=100, add_special_tokens=True),
    )
    (engine_input,) = engine_inputs
    assert engine_input["type"] == "token"
    # add_special_tokens=True -> bos ... eos wrapped (HF convention)
    ids = engine_input["prompt_token_ids"]
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id
    assert ids[1:-1] == tokenizer.encode("one two three")


def test_render_chat_async_batch_parallel_tokenization():
    """Multiple conversations tokenize concurrently on the renderer pool."""
    tokenizer = SeamWordTokenizer(delay=0.2)
    config = mk_config(model_config={"renderer_num_workers": 2})
    renderer = mk_renderer(tokenizer, config)

    start = time.perf_counter()
    _, engine_inputs = asyncio.run(renderer.render_chat_async(
        [
            [{"role": "user", "content": f"message number {i}"}]
            for i in range(2)
        ],
        downlink.ChatParams(),
        downlink.TokenizeParams(max_total_tokens=100, add_special_tokens=False),
    ))
    elapsed = time.perf_counter() - start
    # two 0.2s tokenizations overlapping on a 2-worker pool finish well
    # under 2 x 0.2s (serial would be >= 0.4)
    assert elapsed < 0.38, f"tokenization did not run concurrently ({elapsed:.3f}s)"
    assert len(engine_inputs) == 2


# ---------------------------------------------------------------------------
# m2 / WC2 — tokenization never blocks the event loop
# ---------------------------------------------------------------------------


def test_raw_prompt_tokenizes_on_renderer_pool_not_event_loop():
    """AsyncLLM.add_request with a raw text prompt awaits
    process_inputs_async -> tokenizer runs on a pool thread, and the event
    loop stays responsive while it blocks."""
    tokenizer = SeamWordTokenizer(delay=0.25)
    llm, _, _ = mk_engine(tokenizer)
    heartbeats = []

    async def main():
        async def heartbeat():
            for _ in range(5):
                heartbeats.append(time.monotonic())
                await asyncio.sleep(0.03)

        hb = asyncio.create_task(heartbeat())
        await llm.add_request(
            "req-1", "a slow prompt to tokenize", downlink.SamplingParams()
        )
        await hb

    asyncio.run(main())

    assert tokenizer.calls, "tokenizer was never invoked"
    pool_thread_ids = {tid for tid, _ in tokenizer.calls}
    assert threading.get_ident() not in pool_thread_ids, (
        "tokenization ran on the event-loop thread (would block all requests)"
    )
    # the heartbeat task kept ticking while tokenize slept -> loop not blocked
    assert len(heartbeats) >= 4


def test_rendered_engine_input_takes_sync_fast_path():
    """A rendered EngineInput (dict carrying 'type') hits process_inputs
    synchronously — the tokenizer is never called again."""
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    engine_input = renderer.render_cmpl(
        [downlink.TextPrompt(prompt="already rendered")],
        downlink.TokenizeParams(max_total_tokens=100),
    )[0]
    calls_before = len(tokenizer.calls)

    llm, _, _ = mk_engine(tokenizer, renderer=renderer)
    asyncio.run(llm.add_request("req-2", engine_input, downlink.SamplingParams()))

    assert len(tokenizer.calls) == calls_before, (
        "rendered EngineInput re-tokenized: the sync fast path was missed"
    )
    sent = [e for e in llm.events if e[0] == "send_input"]
    assert len(sent) == 1


def test_make_async_runs_on_executor_thread():
    """make_async wraps a blocking callable into loop.run_in_executor on the
    given executor (vllm/utils/async_utils.py:L28-L45)."""
    seen_threads = []

    def blocking(x):
        seen_threads.append(threading.get_ident())
        return x * 2

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        async_wrapped = downlink.make_async(blocking, executor=pool)

        async def main():
            return await async_wrapped(21)

        assert asyncio.run(main()) == 42
    finally:
        pool.shutdown(wait=False)
    assert seen_threads == [seen_threads[0]] and seen_threads[0] != threading.get_ident()


def test_process_inputs_async_bound_to_renderer_pool():
    """InputProcessor.__init__ wires process_inputs_async = make_async(
    process_inputs, executor=renderer._executor) (PR #49608): the raw-prompt
    preprocessing runs on one of the renderer pool's worker threads."""
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    config = mk_config()
    ip = downlink.InputProcessor(config, renderer)

    async def run_on_pool():
        return await ip.process_inputs_async(
            "raw-1", "raw text down the pool", downlink.SamplingParams(),
            supported_tasks=("generate",),
        )

    request = asyncio.run(run_on_pool())
    # raw fallback -> cmpl default params: bos + words + eos
    assert request.prompt_token_ids == (
        [tokenizer.bos_token_id] + tokenizer.encode("raw text down the pool")
        + [tokenizer.eos_token_id]
    )
    tokenize_thread = tokenizer.calls[-1][0]
    pool_thread_ids = {
        getattr(t, "thread", t).ident
        for t in renderer._executor._threads
        if getattr(t, "thread", t) is not None
    }
    assert tokenize_thread in pool_thread_ids, (
        "process_inputs_async did not run on the renderer's thread pool"
    )


# ---------------------------------------------------------------------------
# m3 — dual pool layout
# ---------------------------------------------------------------------------


def test_dual_pool_layout():
    """_executor has renderer_num_workers workers; _mm_executor is exactly
    single-worker (#38418 P0/P1 order)."""
    tokenizer = SeamWordTokenizer()
    config = mk_config(model_config={"renderer_num_workers": 4})
    renderer = mk_renderer(tokenizer, config)
    assert renderer._executor._max_workers == 4
    assert renderer._mm_executor._max_workers == 1

    default_renderer = mk_renderer(SeamWordTokenizer())
    assert default_renderer._executor._max_workers == 1  # config default


# ---------------------------------------------------------------------------
# m4 / WC1 — EngineCoreRequest is born tokenized (no text field)
# ---------------------------------------------------------------------------


def test_engine_core_request_has_no_text_field():
    """#11963: the wire struct carries token ids / embeds / mm features —
    and no prompt string. Accessing a 'prompt' attribute must fail."""
    llm, _, tokenizer = mk_engine()
    request = asyncio.run(llm.add_request(
        "req-3", "some user text", downlink.SamplingParams(),
    ))
    sent = [e for e in llm.events if e[0] == "send_input"][0][2]

    assert "prompt" not in type(sent).__struct_fields__
    with pytest.raises(AttributeError):
        _ = sent.prompt  # noqa: B018
    # the token ids DID make it across (raw fallback -> cmpl params: bos+eos)
    assert sent.prompt_token_ids == (
        [tokenizer.bos_token_id] + tokenizer.encode("some user text")
        + [tokenizer.eos_token_id]
    )


def test_engine_core_request_wire_bytes_carry_tokens_not_text():
    """Encoding the EngineCoreRequest (the msgpack wire the ADD frame would
    carry) contains the token-id array and not the user's text."""
    from _msgspec_seam import seam_msgspec

    llm, _, _ = mk_engine()
    asyncio.run(llm.add_request("req-4", "secret user prose", downlink.SamplingParams()))
    sent = [e for e in llm.events if e[0] == "send_input"][0][2]
    wire = seam_msgspec.msgpack.encode(sent)
    assert b"secret user prose" not in wire
    # array_like: the positional array carries every field
    decoded = seam_msgspec.msgpack.decode(wire, type=type(sent))
    assert decoded == sent


def test_raw_prompt_fallback_deprecated_but_working(caplog):
    """Passing a raw prompt to InputProcessor still tokenizes (the deprecated
    InputPreprocessor fallback) and warns about it."""
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    ip = downlink.InputProcessor(mk_config(), renderer)

    downlink._ONCE_SEEN.clear()
    with caplog.at_level("WARNING"):
        request = ip.process_inputs(
            "fallback text", "plain raw prompt",
            downlink.SamplingParams(), ("generate",),
        )
    # the deprecated fallback tokenizes with the CMPL default params
    # (add_special_tokens=True): bos + words + eos — real vLLM behavior
    assert request.prompt_token_ids == (
        [tokenizer.bos_token_id] + tokenizer.encode("plain raw prompt")
        + [tokenizer.eos_token_id]
    )
    assert any("deprecated" in r.message for r in caplog.records)


def test_engine_core_request_passthrough_deprecated(caplog):
    """Handing a finished EngineCoreRequest straight to add_request is
    accepted (request reused verbatim) but logs the deprecation warning."""
    llm, _, _ = mk_engine()
    prebuilt = downlink.EngineCoreRequest(
        request_id="prebuilt-1", prompt_token_ids=[3, 4], mm_features=None,
        sampling_params=downlink.SamplingParams(), pooling_params=None,
        arrival_time=1.0, lora_request=None, cache_salt=None,
        data_parallel_rank=None,
    )
    with caplog.at_level("WARNING"):
        queue = asyncio.run(llm.add_request(
            "prebuilt-1", prebuilt, downlink.SamplingParams(),
        ))
    assert any("deprecated" in r.message for r in caplog.records)
    assert queue is not None
    sent = [e for e in llm.events if e[0] == "send_input"][0][2]
    assert sent.prompt_token_ids == [3, 4]


# ---------------------------------------------------------------------------
# m5 / WC3 — dual-track request id
# ---------------------------------------------------------------------------


def test_dual_track_request_id():
    """assign_request_id keeps the user id in external_req_id and derives the
    internal id as f"{external}-{8 random hex chars}"."""
    llm, _, _ = mk_engine()
    asyncio.run(llm.add_request("user-given-id", "hi", downlink.SamplingParams()))
    sent = [e for e in llm.events if e[0] == "send_input"][0][2]

    assert sent.external_req_id == "user-given-id"
    assert re.fullmatch(r"user-given-id-[0-9a-f]{8}", sent.request_id)
    # the OutputProcessor-side collector is keyed by the INTERNAL id
    queue_added = [e for e in llm.events if e[0] == "output_processor.add_request"]
    assert len(queue_added) == 1


def test_duplicate_external_ids_get_distinct_internal_ids():
    """Retrying with the same external id must not collide in the demux table."""
    llm, _, _ = mk_engine()
    asyncio.run(llm.add_request("same-id", "first try", downlink.SamplingParams()))
    asyncio.run(llm.add_request("same-id", "retry", downlink.SamplingParams()))
    sent = [e[2] for e in llm.events if e[0] == "send_input"]
    assert sent[0].external_req_id == sent[1].external_req_id == "same-id"
    assert sent[0].request_id != sent[1].request_id


def test_disable_randomization_keeps_ids_equal(monkeypatch, caplog):
    """VLLM_DISABLE_REQUEST_ID_RANDOMIZATION warns and keeps the two ids
    identical (the documented footgun escape hatch)."""
    monkeypatch.setattr(
        downlink.envs, "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", True
    )
    llm, _, _ = mk_engine()
    downlink._ONCE_SEEN.clear()
    with caplog.at_level("WARNING"):
        asyncio.run(llm.add_request("stable-id", "hi", downlink.SamplingParams()))
    sent = [e for e in llm.events if e[0] == "send_input"][0][2]
    assert sent.request_id == sent.external_req_id == "stable-id"
    assert any("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION" in r.message
               for r in caplog.records)


def test_assign_request_id_rejects_preset_external_field():
    """Passing an EngineCoreRequest that already set external_req_id is a
    user error — rejected with the source's own message."""
    llm, _, _ = mk_engine()
    request = downlink.EngineCoreRequest(
        request_id="r", prompt_token_ids=[1], mm_features=None,
        sampling_params=None, pooling_params=None, arrival_time=0.0,
        lora_request=None, cache_salt=None, data_parallel_rank=None,
        external_req_id="already-set",
    )
    with pytest.raises(ValueError, match="external_req_id field should not be set"):
        downlink.InputProcessor.assign_request_id(request)


# ---------------------------------------------------------------------------
# m6 — params clone & completion
# ---------------------------------------------------------------------------


def test_max_tokens_defaults_to_model_len_minus_seq_len():
    """Unset max_tokens -> generate up to max_model_len - seq_len."""
    tokenizer = SeamWordTokenizer()
    config = mk_config(model_config={"max_model_len": 64})
    llm, _, _ = mk_engine(tokenizer, config=config)
    asyncio.run(llm.add_request(
        "req", "one two three four", downlink.SamplingParams(),
    ))
    sent = [e for e in llm.events if e[0] == "send_input"][0][2]
    assert sent.sampling_params.max_tokens == 64 - len(sent.prompt_token_ids)


def test_params_clone_isolates_caller_object():
    """process_inputs works on params.clone(); the caller's params object is
    never mutated (max_tokens stays None for the caller)."""
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    ip = downlink.InputProcessor(mk_config(model_config={"max_model_len": 32}), renderer)
    params = downlink.SamplingParams(max_tokens=None)
    request = ip.process_inputs(
        "req",
        {"type": "token", "prompt_token_ids": tokenizer.encode("x y"),
         "arrival_time": 1.0},
        params, ("generate",),
    )
    assert request.sampling_params.max_tokens == 32 - 2
    assert params.max_tokens is None
    assert request.sampling_params is not params


def test_update_from_generation_config_injects_eos():
    """generation_config eos ids land in the cloned params' stop set."""
    tokenizer = SeamWordTokenizer()
    config = mk_config(model_config={
        "max_model_len": 64,
        "generation_config_fields": {"eos_token_id": [2, 77]},
    })
    llm, _, _ = mk_engine(tokenizer, config=config)
    asyncio.run(llm.add_request("req", "hello", downlink.SamplingParams()))
    sent = [e for e in llm.events if e[0] == "send_input"][0][2]
    stops = sent.sampling_params.all_stop_token_ids
    assert tokenizer.eos_token_id in stops
    assert 77 in stops


def test_update_from_tokenizer_expands_bad_words():
    """bad_words are tokenized into bad_words_token_ids (both with and
    without a prefix space)."""
    tokenizer = SeamWordTokenizer()
    llm, _, _ = mk_engine(tokenizer)
    asyncio.run(llm.add_request(
        "req", "hello", downlink.SamplingParams(bad_words=["forbidden"]),
    ))
    sent = [e for e in llm.events if e[0] == "send_input"][0][2]
    bad = sent.sampling_params.bad_words_token_ids
    assert bad and all(isinstance(seq, list) for seq in bad)
    flat = [t for seq in bad for t in seq]
    assert flat == tokenizer.encode("forbidden")


def test_pooling_params_clone_path():
    """Pooling requests clone their params and cross the boundary with
    pooling_params set (sampling_params None)."""
    llm, _, _ = mk_engine(supported_tasks=("generate", "embed"))
    asyncio.run(llm.add_request(
        "req", "embed me", downlink.PoolingParams(task="embed"),
    ))
    sent = [e for e in llm.events if e[0] == "send_input"][0][2]
    assert sent.pooling_params is not None
    assert sent.sampling_params is None
    assert sent.pooling_params.task == "embed"


# ---------------------------------------------------------------------------
# m7 — validation chain
# ---------------------------------------------------------------------------


def _sync_process(config=None, **prompt_over):
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    ip = downlink.InputProcessor(config or mk_config(), renderer)
    return ip, tokenizer


def test_empty_decoder_prompt_rejected():
    ip, _ = _sync_process()
    with pytest.raises(downlink.VLLMValidationError, match="cannot be empty"):
        ip.process_inputs(
            "r", {"type": "token", "prompt_token_ids": []},
            downlink.SamplingParams(), ("generate",),
        )


def test_over_length_prompt_rejected():
    ip, _ = _sync_process(mk_config(model_config={"max_model_len": 8}))
    with pytest.raises(downlink.VLLMValidationError, match="longer than the maximum"):
        ip.process_inputs(
            "r", {"type": "token", "prompt_token_ids": list(range(9))},
            downlink.SamplingParams(), ("generate",),
        )


def test_full_length_prompt_rejected_for_generate():
    """A prompt exactly max_model_len long cannot generate even one more
    token — rejected up front, in the front-end process."""
    ip, _ = _sync_process(mk_config(model_config={"max_model_len": 8}))
    with pytest.raises(downlink.VLLMValidationError, match="at least 1"):
        ip.process_inputs(
            "r", {"type": "token", "prompt_token_ids": list(range(8))},
            downlink.SamplingParams(), ("generate",),
        )


def test_vocab_out_of_range_rejected():
    """Token ids beyond max(tokenizer.max_token_id, vocab-1) are OOV and are
    caught before crossing the process boundary."""
    ip, tokenizer = _sync_process()  # tokenizer.max_token_id == 99
    with pytest.raises(downlink.VLLMValidationError, match="out of vocabulary"):
        ip.process_inputs(
            "r", {"type": "token", "prompt_token_ids": [3, 999]},
            downlink.SamplingParams(), ("generate",),
        )


def test_vocab_bound_takes_max_of_both_sides():
    """A token legal for the model vocab but beyond the tokenizer's own max
    id (or vice versa) must NOT be rejected (Qwen3-style mismatch)."""
    ip, _ = _sync_process(mk_config(model_config={"vocab_size": 500}))
    request = ip.process_inputs(
        "r", {"type": "token", "prompt_token_ids": [3, 200]},
        downlink.SamplingParams(), ("generate",),
    )
    assert request.prompt_token_ids == [3, 200]


def test_data_parallel_rank_bounds():
    ip, _ = _sync_process(mk_config(parallel_config={"data_parallel_size": 2}))
    ok = ip.process_inputs(
        "r", {"type": "token", "prompt_token_ids": [3]},
        downlink.SamplingParams(), ("generate",), data_parallel_rank=1,
    )
    assert ok.data_parallel_rank == 1
    with pytest.raises(downlink.VLLMValidationError, match="out of range"):
        ip.process_inputs(
            "r", {"type": "token", "prompt_token_ids": [3]},
            downlink.SamplingParams(), ("generate",), data_parallel_rank=2,
        )


def test_pooling_on_generation_only_model_rejected():
    ip, _ = _sync_process()
    with pytest.raises(downlink.VLLMValidationError, match="does not support pooling"):
        ip.process_inputs(
            "r", {"type": "token", "prompt_token_ids": [3]},
            downlink.PoolingParams(), ("generate",),
        )


def test_generation_on_pooling_only_model_rejected():
    ip, _ = _sync_process()
    with pytest.raises(downlink.VLLMValidationError, match="does not support generation"):
        ip.process_inputs(
            "r", {"type": "token", "prompt_token_ids": [3]},
            downlink.SamplingParams(), ("embed",),
        )


def test_wrong_params_type_rejected():
    ip, _ = _sync_process()
    with pytest.raises(TypeError, match="SamplingParams or PoolingParams"):
        ip.process_inputs(
            "r", {"type": "token", "prompt_token_ids": [3]},
            {"not": "params"}, ("generate",),
        )


def test_lora_request_without_lora_config_rejected():
    ip, _ = _sync_process()
    with pytest.raises(downlink.VLLMValidationError, match="LoRA is not enabled"):
        ip.process_inputs(
            "r", {"type": "token", "prompt_token_ids": [3]},
            downlink.SamplingParams(), ("generate",),
            lora_request=downlink.LoRARequest("lora-1"),
        )


# ---------------------------------------------------------------------------
# m8 — multimodal flatten, ordering, cache-hit skip, LoRA-prefixed identifier
# ---------------------------------------------------------------------------


MM_CONFIG = dict(
    multimodal_config={"seam_tokens_per_item": {"image": 2, "audio": 3}},
)


def mm_messages(*blocks):
    return [{"role": "user", "content": list(blocks)}]


def test_mm_flatten_sorted_by_offset():
    """Interleaved image+audio items flatten to one list ordered by each
    item's position in prompt_token_ids (argsort_mm_positions)."""
    tokenizer = SeamWordTokenizer()
    config = mk_config(**MM_CONFIG)
    renderer = mk_renderer(tokenizer, config)
    request = FakeChatRequest(messages=mm_messages(
        {"type": "text", "text": "look at"},
        {"type": "image_url", "image_url": "IMGDATA-A"},
        {"type": "text", "text": "and hear"},
        {"type": "input_audio", "input_audio": "AUDDATA-B"},
        {"type": "text", "text": "please"},
    ))
    online = downlink.OnlineRenderer(
        config.model_config, renderer, request_logger=None,
        chat_template=None, chat_template_content_format="string",
    )
    _, engine_inputs = asyncio.run(online.render_chat(request))
    (engine_input,) = engine_inputs
    assert engine_input["type"] == "multimodal"

    ip = downlink.InputProcessor(config, renderer)
    core_request = ip.process_inputs(
        "mm-1", engine_input, downlink.SamplingParams(), ("generate",),
    )
    feats = core_request.mm_features
    assert [f.modality for f in feats] == ["image", "audio"]
    offsets = [f.mm_position.offset for f in feats]
    assert offsets == sorted(offsets)
    img = feats[0]
    assert img.mm_position.length == 2  # seam: 2 image placeholder tokens
    assert img.data is not None
    assert img.identifier == img.mm_hash  # no LoRA -> identifier == hash
    # placeholders actually expanded in the token ids
    ids = core_request.prompt_token_ids
    assert len(ids) > 0


def test_mm_hash_values_are_strings_and_duplicate_detected():
    """mm_hashes must be all-strings; the flatten respects per-modality
    lists and repeated items of the same modality keep order."""
    tokenizer = SeamWordTokenizer()
    config = mk_config(**MM_CONFIG)
    renderer = mk_renderer(tokenizer, config)
    request = FakeChatRequest(messages=mm_messages(
        {"type": "image_url", "image_url": "IMG-1"},
        {"type": "text", "text": "then"},
        {"type": "image_url", "image_url": "IMG-2"},
    ))
    online = downlink.OnlineRenderer(
        config.model_config, renderer, request_logger=None,
        chat_template=None, chat_template_content_format="string",
    )
    _, (engine_input,) = asyncio.run(online.render_chat(request))
    ip = downlink.InputProcessor(config, renderer)
    feats = ip.process_inputs(
        "mm-2", engine_input, downlink.SamplingParams(), ("generate",),
    ).mm_features
    assert [f.modality for f in feats] == ["image", "image"]
    assert feats[0].mm_hash != feats[1].mm_hash
    assert feats[0].mm_position.offset < feats[1].mm_position.offset


def test_mm_cache_hit_second_time_data_none():
    """The same image twice (two requests): first crossing carries the
    payload; the second is a processor-cache hit -> data=None (skip the
    cross-process tensor copy)."""
    tokenizer = SeamWordTokenizer()
    config = mk_config(**MM_CONFIG)
    renderer = mk_renderer(tokenizer, config)
    llm = downlink.AsyncLLM(config, renderer)

    messages = mm_messages({"type": "image_url", "image_url": "SAME-IMAGE"})
    online = downlink.OnlineRenderer(
        config.model_config, renderer, request_logger=None,
        chat_template=None, chat_template_content_format="string",
    )
    for i in range(2):
        _, (engine_input,) = asyncio.run(online.render_chat(FakeChatRequest(
            messages=messages,
        )))
        asyncio.run(llm.add_request(f"mm-cache-{i}", engine_input, downlink.SamplingParams()))

    sent = [e[2] for e in llm.events if e[0] == "send_input"]
    assert sent[0].mm_features[0].data is not None
    assert sent[1].mm_features[0].data is None, (
        "cache hit should skip the IPC payload (data=None per docstring)"
    )
    # the hash still crosses — the engine side can look it up
    assert sent[1].mm_features[0].mm_hash == sent[0].mm_features[0].mm_hash


def test_mm_identifier_gets_lora_prefix_with_tower_connector():
    """With enable_tower_connector_lora the cache identifier is prefixed with
    the LoRA name to prevent cross-LoRA cache hits."""
    tokenizer = SeamWordTokenizer()
    config = mk_config(
        lora_config={"enable_tower_connector_lora": True}, **MM_CONFIG,
    )
    renderer = mk_renderer(tokenizer, config)
    request = FakeChatRequest(messages=mm_messages(
        {"type": "image_url", "image_url": "IMG-LORA"},
    ))
    online = downlink.OnlineRenderer(
        config.model_config, renderer, request_logger=None,
        chat_template=None, chat_template_content_format="string",
    )
    _, (engine_input,) = asyncio.run(online.render_chat(request))
    ip = downlink.InputProcessor(config, renderer)
    lora = downlink.LoRARequest("my-lora")
    feats = ip.process_inputs(
        "mm-3", engine_input, downlink.SamplingParams(), ("generate",),
        lora_request=lora,
    ).mm_features
    assert feats[0].identifier == f"my-lora:{feats[0].mm_hash}"
    # with plain (non-tower-connector) LoRA the identifier is the bare hash
    # (LoRA still enabled, so _validate_lora accepts the request)
    plain_config = mk_config(
        lora_config={"enable_tower_connector_lora": False}, **MM_CONFIG,
    )
    plain_renderer = mk_renderer(SeamWordTokenizer(), plain_config)
    plain_online = downlink.OnlineRenderer(
        plain_config.model_config, plain_renderer, request_logger=None,
        chat_template=None, chat_template_content_format="string",
    )
    _, (plain_input,) = asyncio.run(plain_online.render_chat(FakeChatRequest(
        messages=mm_messages({"type": "image_url", "image_url": "IMG-LORA"}),
    )))
    plain_feats = downlink.InputProcessor(plain_config, plain_renderer).process_inputs(
        "mm-4", plain_input, downlink.SamplingParams(), ("generate",),
        lora_request=downlink.LoRARequest("my-lora"),
    ).mm_features
    assert plain_feats[0].identifier == plain_feats[0].mm_hash


# ---------------------------------------------------------------------------
# m9 — PlaceholderRange & encoder-cache budget
# ---------------------------------------------------------------------------


def test_placeholder_range_aaaa_bbbb_example():
    """The docstring's own teaching example: prompt `AAAA BBBB ...` -> A at
    (0,4), B at (5,4)."""
    a = downlink.PlaceholderRange(offset=0, length=4)
    b = downlink.PlaceholderRange(offset=5, length=4)
    assert (a.offset, a.length) == (0, 4)
    assert (b.offset, b.length) == (5, 4)
    assert a.get_num_embeds() == 4
    assert b.get_num_embeds() == 4


def test_placeholder_range_embed_mask():
    import torch
    mask = torch.tensor([False, True, False, True, True])
    pr = downlink.PlaceholderRange(offset=2, length=5, is_embed=mask)
    assert pr.get_num_embeds() == 3
    assert downlink.PlaceholderRange(offset=0, length=3).get_num_embeds() == 3


def test_encoder_cache_budget_precheck():
    """An mm item whose embed count exceeds the encoder cache budget is
    rejected in the front-end (before crossing)."""
    tokenizer = SeamWordTokenizer()
    config = mk_config(
        model_config={"max_model_len": 4096},
        multimodal_config={
            "seam_tokens_per_item": {"image": 8},
            "encoder_cache_size": 4,
        },
    )
    renderer = mk_renderer(tokenizer, config)
    request = FakeChatRequest(messages=mm_messages(
        {"type": "image_url", "image_url": "BIG-IMG"},
    ))
    online = downlink.OnlineRenderer(
        config.model_config, renderer, request_logger=None,
        chat_template=None, chat_template_content_format="string",
    )
    _, (engine_input,) = asyncio.run(online.render_chat(request))
    ip = downlink.InputProcessor(config, renderer)
    with pytest.raises(downlink.VLLMValidationError, match="encoder cache"):
        ip.process_inputs(
            "mm-5", engine_input, downlink.SamplingParams(), ("generate",),
        )


def test_argsort_mm_positions_flat_order():
    """argsort_mm_positions flattens dict-of-list placeholders and sorts by
    offset, returning (modality, idx) keys."""
    positions = {
        "audio": [downlink.PlaceholderRange(offset=9, length=3)],
        "image": [
            downlink.PlaceholderRange(offset=2, length=2),
            downlink.PlaceholderRange(offset=14, length=2),
        ],
    }
    assert downlink.argsort_mm_positions(positions) == [
        ("image", 0), ("audio", 0), ("image", 1),
    ]


# ---------------------------------------------------------------------------
# m10 — EngineInput family: embeds path & enc/dec split
# ---------------------------------------------------------------------------


def test_prompt_embeds_cpu_and_batch_squeeze():
    """prompt_embeds must land on CPU for cross-process serialization and a
    (1, seq, hidden) batch is squeezed to (seq, hidden)."""
    import torch
    tokenizer = SeamWordTokenizer()
    config = mk_config(model_config={"enable_prompt_embeds": True,
                                     "max_model_len": 16})
    renderer = mk_renderer(tokenizer, config)
    embeds = torch.randn(1, 4, 8)  # batch-of-one shape

    engine_input = renderer.render_cmpl(
        [downlink.EmbedsPrompt(prompt_embeds=embeds)],
        downlink.TokenizeParams(max_total_tokens=16),
    )[0]
    assert engine_input["type"] == "embeds"
    assert engine_input["prompt_embeds"].shape == (4, 8)

    ip = downlink.InputProcessor(config, renderer)
    request = ip.process_inputs(
        "emb-1", engine_input, downlink.SamplingParams(), ("generate",),
    )
    assert request.prompt_embeds is not None
    assert request.prompt_embeds.shape == (4, 8)
    assert request.prompt_token_ids is None
    # embeds length drives max_tokens default
    assert request.sampling_params.max_tokens == 16 - 4


def test_prompt_embeds_requires_flag():
    import torch
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)  # enable_prompt_embeds defaults False
    with pytest.raises(ValueError, match="enable-prompt-embeds"):
        renderer.render_cmpl(
            [downlink.EmbedsPrompt(prompt_embeds=torch.randn(2, 8))],
            downlink.TokenizeParams(max_total_tokens=16),
        )


def test_length_from_prompt_token_ids_or_embeds():
    import torch
    f = downlink.length_from_prompt_token_ids_or_embeds
    assert f([1, 2, 3], None) == 3
    assert f(None, torch.zeros(5, 8)) == 5
    assert f([1, 2], torch.zeros(2, 8)) == 2
    with pytest.raises(ValueError):
        f(None, None)
    with pytest.raises(ValueError):
        f([1, 2, 3], torch.zeros(2, 8))


def test_enc_dec_split_and_decoder_start_token():
    """Encoder-decoder raw prompts route through InputPreprocessor ->
    build_enc_dec_input (decoder start token prepended), and process_inputs
    validates BOTH sides via split_enc_dec_input."""
    tokenizer = SeamWordTokenizer()
    # real vLLM implements even text-only enc-dec models as mm-registered
    # (inputs/engine.py MultiModalEncDecInput docstring) — mirror that here so
    # the encoder-side length check runs against a real encoder cache budget
    config = mk_config(
        model_config={
            "is_encoder_decoder": True,
            "hf_config": {"decoder_start_token_id": 7},
        },
        multimodal_config={},
    )
    renderer = mk_renderer(tokenizer, config)
    ip = downlink.InputProcessor(config, renderer)
    request = ip.process_inputs(
        "encdec-1",
        {"encoder_prompt": "translate this", "decoder_prompt": None},
        downlink.SamplingParams(), ("generate",),
    )
    # decoder start token prepended to the (cmpl-defaults) encoder tokens:
    # [7, bos] + encode("translate this") + [eos]
    expected = (
        [7, tokenizer.bos_token_id]
        + tokenizer.encode("translate this")
        + [tokenizer.eos_token_id]
    )
    assert request.prompt_token_ids == expected
# -------------------------------------------------------------------
# m11 — departure: double registration & client_index
# ---------------------------------------------------------------------------


def test_double_registration_local_before_cross_process():
    """_add_request registers with the OutputProcessor (this process) BEFORE
    handing the request to EngineCore (separate process)."""
    llm, _, _ = mk_engine()
    asyncio.run(llm.add_request("req-x", "hello there", downlink.SamplingParams()))
    seq = [e[0] for e in llm.events]
    assert seq.index("output_processor.add_request") < seq.index("send_input")


def test_client_index_stamped_on_crossing():
    """add_request_async stamps request.client_index before the ADD frame
    (core_client.py:L1145-L1148); the frame tag is b'\\x00'."""
    llm, _, _ = mk_engine(client_index=3)
    asyncio.run(llm.add_request("req-y", "hi", downlink.SamplingParams()))
    tag, sent = [(e[1], e[2]) for e in llm.events if e[0] == "send_input"][0]
    assert tag == downlink.EngineCoreRequestType.ADD
    assert tag.value == b"\x00"
    assert sent.client_index == 3


def test_request_output_collector_carries_internal_id():
    """The queue handed back to the caller is keyed by the INTERNAL
    (randomized) request id."""
    llm, _, _ = mk_engine()
    queue = asyncio.run(llm.add_request("ext-id", "hi", downlink.SamplingParams()))
    sent = [e[2] for e in llm.events if e[0] == "send_input"][0]
    assert queue.request_id == sent.request_id
    assert queue.request_id != "ext-id"


def test_engine_dead_raises_engine_dead_error():
    llm, _, _ = mk_engine()
    llm.engine_core.resources.engine_dead = True
    with pytest.raises(downlink.EngineDeadError):
        asyncio.run(llm.add_request("req-z", "hi", downlink.SamplingParams()))


# ---------------------------------------------------------------------------
# arrival_time — stamped at render entry, not at engine admission
# ---------------------------------------------------------------------------


def test_arrival_time_taken_at_render_not_process_inputs(monkeypatch):
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    t0 = 1_000.0
    monkeypatch.setattr(downlink.time, "time", lambda: t0)
    engine_input = renderer.render_cmpl(
        [downlink.TextPrompt(prompt="early bird")],
        downlink.TokenizeParams(max_total_tokens=100),
    )[0]
    assert engine_input["arrival_time"] == t0

    # a much later clock at process_inputs must NOT overwrite the render stamp
    monkeypatch.setattr(downlink.time, "time", lambda: t0 + 999.0)
    ip = downlink.InputProcessor(mk_config(), renderer)
    request = ip.process_inputs(
        "arr-1", engine_input, downlink.SamplingParams(), ("generate",),
    )
    assert request.arrival_time == t0

    # ... but a raw prompt with no render stamp gets stamped at process time
    raw_request = ip.process_inputs(
        "arr-2", "late raw text", downlink.SamplingParams(), ("generate",),
    )
    assert raw_request.arrival_time == t0 + 999.0


# ---------------------------------------------------------------------------
# wire struct — array_like encoding of EngineCoreRequest
# ---------------------------------------------------------------------------


def test_struct_array_like_encodes_all_fields():
    from _msgspec_seam import seam_msgspec

    fields = downlink.EngineCoreRequest.__struct_fields__
    request = downlink.EngineCoreRequest(
        request_id="r", prompt_token_ids=[1, 2], mm_features=None,
        sampling_params=None, pooling_params=None, arrival_time=1.5,
        lora_request=None, cache_salt=None, data_parallel_rank=None,
    )
    wire = seam_msgspec.msgpack.encode(request)
    decoded = seam_msgspec.msgpack.decode(wire, type=downlink.EngineCoreRequest)
    assert decoded == request
    # decode the raw array: array_like -> every field rides the array
    raw = seam_msgspec.msgpack.decode(wire)
    assert isinstance(raw, list) and len(raw) == len(fields)


def test_params_property_returns_sampling_or_pooling():
    r = downlink.EngineCoreRequest(
        request_id="r", prompt_token_ids=None, mm_features=None,
        sampling_params=downlink.SamplingParams(), pooling_params=None,
        arrival_time=0.0, lora_request=None, cache_salt=None,
        data_parallel_rank=None,
    )
    assert isinstance(r.params, downlink.SamplingParams)


# ---------------------------------------------------------------------------
# OnlineRenderer gatekeeping (station 1 -> 2)
# ---------------------------------------------------------------------------


def _online(config=None, renderer=None):
    config = config or mk_config()
    renderer = renderer or mk_renderer(SeamWordTokenizer(), config)
    return downlink.OnlineRenderer(
        config.model_config, renderer, request_logger=None,
        chat_template=None, chat_template_content_format="string",
    )


def test_tool_choice_auto_without_parser_rejected():
    online = _online()  # no tool parser, auto tools off
    resp = asyncio.run(online.render_chat(FakeChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        tool_choice="auto",
    )))
    assert not isinstance(resp, tuple), "should be an ErrorResponse, not a success"
    assert "enable-auto-tool-choice" in resp.message


def test_tool_choice_required_without_parser_rejected():
    online = _online()
    resp = asyncio.run(online.render_chat(FakeChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        tool_choice="required",
    )))
    assert not isinstance(resp, tuple)
    assert "--tool-call-parser" in resp.message


def test_untrusted_request_chat_template_refused():
    online = _online()  # trust_request_chat_template defaults False
    resp = asyncio.run(online.render_chat(FakeChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        chat_template="{% for m in messages %}{{ m.content }}{% endfor %}",
    )))
    assert not isinstance(resp, tuple)
    assert "untrusted chat template" in resp.message


def test_online_render_chat_happy_path_produces_engine_input():
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    online = _online(renderer=renderer)
    conversation, engine_inputs = asyncio.run(online.render_chat(FakeChatRequest(
        messages=[{"role": "user", "content": "good morning"}],
    )))
    assert len(conversation) == 1
    (engine_input,) = engine_inputs
    assert engine_input["type"] == "token"
    assert engine_input["prompt_token_ids"] == tokenizer.encode(
        "user: good morning"
    )


def test_online_render_completion_face():
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    online = _online(renderer=renderer)
    request = FakeChatRequest(
        messages=[{"role": "user", "content": "x"}],
        prompt="a b c", add_special_tokens=True,
    )
    engine_inputs = asyncio.run(online.render_completion(request))
    (engine_input,) = engine_inputs
    assert engine_input["type"] == "token"
    assert engine_input["prompt_token_ids"][1:-1] == tokenizer.encode("a b c")


# ---------------------------------------------------------------------------
# process_for_engine keeps text/cache_salt side-channel fields
# ---------------------------------------------------------------------------


def test_process_tokens_carries_prompt_text_and_cache_salt():
    tokenizer = SeamWordTokenizer()
    renderer = mk_renderer(tokenizer)
    engine_input = renderer.process_for_engine(
        downlink.TokensPrompt(prompt_token_ids=tokenizer.encode("keep me"),
                              prompt="keep me", cache_salt="tenant-42"),
        arrival_time=1.0,
    )
    assert engine_input["prompt"] == "keep me"
    assert engine_input["cache_salt"] == "tenant-42"
    assert engine_input["type"] == "token"


def test_random_uuid_shape():
    v = downlink.random_uuid()
    assert re.fullmatch(r"[0-9a-f]{16}", v)

