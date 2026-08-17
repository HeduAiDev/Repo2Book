"""Driver for m1 (render four-step pipeline) — host run against the ch06
subtract-only companion (implementation/downlink.py, pin vLLM v0.27.1).

Records the observable facts the m1 figure needs:
- the four steps run in order render_messages -> tokenize -> extras ->
  process_for_engine (a pure-text request never enters the mm step);
- the payload type at each handoff (DictPrompt -> TokensPrompt ->
  EngineInput with 'type');
- chat face and completion face are isomorphic (same four steps);
- a batch render stamps arrival_time ONCE at entry and every EngineInput in
  the asyncio.gather fan-out carries the same stamp (theory: TTFT clock
  starts at "text arrives", base.py:L993/L1044/L1080).

Test-side seams reused from tests/test_downlink.py (SeamWordTokenizer /
ChatRendererSeam / FakeChatRequest / mk_config): the HF tokenizer and the
Jinja2 chat-template engine are this chapter's documented black boxes.
"""
import asyncio
import importlib
import json
import sys
import threading
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_downlink")
downlink = td.downlink


class StepTracer(td.ChatRendererSeam):
    """Records each pipeline step the BaseRenderer drives, in order.

    ChatRendererSeam fulfills the HfRenderer.render_messages contract (the
    template engine itself is the chapter's black box); everything below
    step 1 is the real BaseRenderer code under test.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.trace: list[str] = []
        self.tokenizer._order = self.trace  # "tokenize"/"encode" markers

    def render_messages(self, messages, params):
        self.trace.append("step1.render_messages(chat 模板)")
        return super().render_messages(messages, params)

    async def _render_prompt_async(self, prompt):
        self.trace.append("step1.render_prompt(completion 直通)")
        return await super()._render_prompt_async(prompt)

    def _apply_prompt_extras(self, prompts, prompt_extras):
        self.trace.append("step3.extras")
        return super()._apply_prompt_extras(prompts, prompt_extras)

    async def process_for_engine_async(self, prompt, arrival_time, **kwargs):
        self.trace.append("step4.process_for_engine")
        return await super().process_for_engine_async(prompt, arrival_time, **kwargs)

    def _process_multimodal(self, *args, **kwargs):
        self.trace.append("step4.mm 预处理")
        return super()._process_multimodal(*args, **kwargs)


def online_of(renderer, config):
    return downlink.OnlineRenderer(
        config.model_config, renderer,
        request_logger=None, chat_template=None,
        chat_template_content_format="string",
    )


async def main():
    out = {
        "driver": "run_m1_pipeline.py",
        "mechanism": "m1 渲染四步流水 render_prompt/render_messages -> tokenize -> extras -> process_for_engine",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "loop_thread_id": threading.get_ident(),
    }

    # -- A. pure-text chat request: step order + payload handoffs ------------
    tok = td.SeamWordTokenizer()
    renderer = StepTracer(td.mk_config(), tok)
    online = online_of(renderer, td.mk_config())
    request = td.FakeChatRequest(messages=[{"role": "user", "content": "hello world"}])
    conversation, engine_inputs = await online.render_chat(request)
    (ei,) = engine_inputs
    out["A_pure_text_chat"] = {
        "trace_steps": list(renderer.trace),
        "mm_step_entered": "mm" in " ".join(renderer.trace),
        "engine_input_type": ei["type"],
        "engine_input_keys": sorted(ei.keys()),
        "prompt_token_ids": ei["prompt_token_ids"],
        "prompt_token_ids_len": len(ei["prompt_token_ids"]),
        "arrival_time_present": "arrival_time" in ei,
        "conversation_len": len(conversation),
        "add_special_tokens_chat_default": False,
    }

    # -- B. batch of three conversations (one with an image): gather fan-out,
    #       single arrival_time stamp carried by every EngineInput ------------
    cfg_mm = td.mk_config(multimodal_config={})  # mm-capable model config
    tok2 = td.SeamWordTokenizer()
    renderer2 = StepTracer(cfg_mm, tok2)
    conversations = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": [
            {"type": "text", "text": "see"},
            {"type": "image_url", "image_url": "IMG-1"},
            {"type": "text", "text": "here"},
        ]}],
        [{"role": "user", "content": "three words now"}],
    ]
    chat_params = request.build_chat_params(None, "string")
    tok_params = request.build_tok_params(cfg_mm.model_config)
    out_conversations, eng_prompts = await renderer2.render_chat_async(
        conversations, chat_params, tok_params
    )
    stamps = [p.get("arrival_time") for p in eng_prompts]
    out["B_batch_gather"] = {
        "conversations": 3,
        "engine_input_types": [p["type"] for p in eng_prompts],
        "mm_request_index": [p["type"] for p in eng_prompts].index("multimodal"),
        "mm_step_entered_for_mm_request": "mm 预处理" in " ".join(renderer2.trace),
        "trace_steps": list(renderer2.trace),
        "tokenize_marker_count": renderer2.trace.count("tokenize"),
        "mm_marker_count": sum(1 for s in renderer2.trace if "mm 预处理" in s),
        "arrival_time_stamps": stamps,
        "single_arrival_time_stamp": len(set(stamps)) == 1,
        "prompt_token_ids_lens": [len(p["prompt_token_ids"]) for p in eng_prompts],
        "conversations_returned": len(out_conversations),
    }

    # -- C. completion face is isomorphic (same four steps) ------------------
    tok3 = td.SeamWordTokenizer()
    renderer3 = StepTracer(td.mk_config(), tok3)
    online3 = online_of(renderer3, td.mk_config())
    creq = td.FakeChatRequest(messages=None, prompt="hello there friend")
    engine_inputs_c = await online3.render_completion(creq)
    out["C_completion_face"] = {
        "trace_steps": list(renderer3.trace),
        "engine_input_type": engine_inputs_c[0]["type"],
        "prompt_token_ids": engine_inputs_c[0]["prompt_token_ids"],
        "prompt_token_ids_len": len(engine_inputs_c[0]["prompt_token_ids"]),
        "mm_step_entered": "mm" in " ".join(renderer3.trace),
    }

    dest = Path(__file__).resolve().parent / "m1_pipeline.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
