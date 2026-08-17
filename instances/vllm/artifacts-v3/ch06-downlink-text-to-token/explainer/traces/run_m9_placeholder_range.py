"""Driver for m9 (PlaceholderRange + encoder-cache budget pre-check) — host
run against the ch06 subtract-only companion (pin vLLM v0.27.1).

Records:
- round 1: the source's own teaching example (docstring, inputs.py:L126-L135):
  prompt "AAAA BBBB What is in these images?" -> A: (offset=0, length=4),
  B: (offset=5, length=4); get_num_embeds() == length when is_embed is None;
- round 2: the non-degenerate is_embed branch — a 6-position placeholder with
  mask [1,1,0,0,1,0] admits only 3 embeddings (cumsum [1,2,2,2,3,3]);
  get_embeds_indices_in_range(0,4) = (0,2); extract_embeds_range = two
  regions, shifted by the placeholder offset;
- round 3: the encoder-cache budget pre-check (input_processor.py:L463-L476):
  an image item expanding to 6 placeholder tokens is REJECTED before crossing
  when mm encoder_cache_size=4 ("which exceeds the pre-allocated encoder
  cache size 4") and ADMITTED when the budget is 8.

Seam note: the 6-token expansion is the seam's configurable placeholder
length (real processors expand to encoder feature sizes, e.g. 576/image) —
the budget arithmetic compared against get_num_embeds() is the real code.
"""
import asyncio
import importlib
import json
import sys
from pathlib import Path

import torch

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_downlink")
downlink = td.downlink

MM_MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "look"},
    {"type": "image_url", "image_url": "IMG-1"},
    {"type": "text", "text": "here"},
]}]


async def main():
    out = {
        "driver": "run_m9_placeholder_range.py",
        "mechanism": "m9 PlaceholderRange(offset,length[,is_embed]) + 编码器缓存预算前置校验",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
    }

    # -- round 1: the docstring's AAAA BBBB teaching example ------------------
    a = downlink.PlaceholderRange(offset=0, length=4)
    b = downlink.PlaceholderRange(offset=5, length=4)
    out["round1_aaaa_bbbb_docstring_example"] = {
        "action": "源码 docstring 教学例：prompt = 'AAAA BBBB What is in these images?'",
        "example_prompt": "AAAA BBBB What is in these images?",
        "A": {"offset": a.offset, "length": a.length},
        "B": {"offset": b.offset, "length": b.length},
        "A_get_num_embeds": a.get_num_embeds(),
        "B_get_num_embeds": b.get_num_embeds(),
        "is_embed_none_means_full_length": (
            a.get_num_embeds() == a.length and b.get_num_embeds() == b.length
        ),
        "gap_between_placeholders": b.offset - (a.offset + a.length),
    }

    # -- round 2: is_embed mask (mixed-embedding special case) ----------------
    mask = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
    pr = downlink.PlaceholderRange(offset=2, length=6, is_embed=mask)
    out["round2_is_embed_mask"] = {
        "action": "PlaceholderRange(offset=2, length=6, is_embed=[1,1,0,0,1,0])",
        "mask": mask.tolist(),
        "embeds_cumsum": pr.embeds_cumsum,
        "get_num_embeds": pr.get_num_embeds(),
        "embeds_fewer_than_positions": pr.get_num_embeds() < pr.length,
        "get_embeds_indices_in_range_0_4": list(
            pr.get_embeds_indices_in_range(0, 4)
        ),
        "extract_embeds_range": [list(t) for t in pr.extract_embeds_range()],
        "note": "嵌入数按掩码计数；range 查询把占位符内位置映射到编码器输出下标",
    }

    # -- round 3: encoder-cache budget pre-check (before crossing) ------------
    async def attempt(encoder_cache_size: int):
        cfg = td.mk_config(multimodal_config={
            "encoder_cache_size": encoder_cache_size,
            "seam_tokens_per_item": {"image": 6},
        })
        llm, renderer, _ = td.mk_engine(td.SeamWordTokenizer(), config=cfg)
        online = downlink.OnlineRenderer(
            cfg.model_config, renderer,
            request_logger=None, chat_template=None,
            chat_template_content_format="string",
        )
        req = td.FakeChatRequest(messages=MM_MESSAGES)
        _, engine_inputs = await online.render_chat(req)
        try:
            await llm.add_request(
                "mm-budget", engine_inputs[0], downlink.SamplingParams(max_tokens=4)
            )
            frames = [e for e in llm.events if e[0] == "send_input"]
            return {"rejected": False, "add_frames_crossed": len(frames)}
        except downlink.VLLMValidationError as e:
            frames = [e2 for e2 in llm.events if e2[0] == "send_input"]
            return {
                "rejected": True,
                "error": str(e),
                "add_frames_crossed": len(frames),
            }

    reject4 = await attempt(4)
    admit8 = await attempt(8)
    out["round3_encoder_cache_budget"] = {
        "action": "image item 展开为 6 个占位 token，编码器缓存预算 4 vs 8",
        "budget_4": {
            "encoder_cache_size": 4,
            "num_embeds": 6,
            **reject4,
        },
        "budget_8": {
            "encoder_cache_size": 8,
            "num_embeds": 6,
            **admit8,
        },
        "rejection_happens_before_crossing": reject4["add_frames_crossed"] == 0,
        "note": "预算校验在 process_inputs 内、构造 EngineCoreRequest 之前——不过线就拦下",
    }

    dest = Path(__file__).resolve().parent / "m9_placeholder_range.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
