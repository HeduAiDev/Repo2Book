"""Driver for m8 (mm feature flatten) — host run against the ch06
subtract-only companion (pin vLLM v0.27.1).

The worked example is the non-degenerate case: modalities INTERLEAVED in the
prompt (image, audio, image) while the processor hands InputProcessor a
dict-of-list grouped BY MODALITY — flatten must restore prompt order via
argsort_mm_positions (multimodal/utils.py:L145-L165). Records:
- round 1: dict-of-list mm_placeholders vs the flattened
  list[MultiModalFeatureSpec] in prompt-token order (offsets ascending);
- round 2: same items resent -> processor cache hit -> data=None on every
  spec while mm_hash/identifier still cross ("skip IPC between API server
  and engine core", inputs.py:L331-L337);
- round 3: tower-connector LoRA prefixes the identifier cache key
  (input_processor.py:L174-L190); a plain LoRA keeps the bare hash.

Seam note (impl-notes 已知偏差 1): the mm processor seam expands each item to
a FIXED placeholder length (image 2 / audio 3 tokens, marker ids 31/32,
placeholder ids 40/45) and hashes sha1(repr)[:16] — the real processor uses
encoder feature sizes (e.g. 576/image) and MultiModalHasher. Order/flatten/
cache semantics are the real code paths.
"""
import asyncio
import importlib
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_downlink")
downlink = td.downlink

MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "look at"},
    {"type": "image_url", "image_url": "IMG-B"},
    {"type": "text", "text": "then hear"},
    {"type": "input_audio", "input_audio": "AUD-A"},
    {"type": "text", "text": "finally"},
    {"type": "image_url", "image_url": "IMG-A"},
]}]


async def render_once(online, renderer, request):
    conv, engine_inputs = await online.render_chat(request)
    return engine_inputs[0]


def spec_view(req):
    return [
        {
            "modality": f.modality,
            "identifier": f.identifier,
            "offset": f.mm_position.offset,
            "length": f.mm_position.length,
            "data": None if f.data is None else str(f.data),
            "mm_hash": f.mm_hash,
        }
        for f in (req.mm_features or [])
    ]


def ph_view(engine_input):
    """dict-of-list view BEFORE the flatten (what the processor handed over)."""
    return {
        modality: [
            {"offset": pr.offset, "length": pr.length}
            for pr in items
        ]
        for modality, items in engine_input["mm_placeholders"].items()
    }


async def main():
    out = {
        "driver": "run_m8_mm_flatten.py",
        "mechanism": "m8 mm 特征展平 argsort_mm_positions → list[MultiModalFeatureSpec]",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "seam_placeholder_shape": {"image": 2, "audio": 3},
        "seam_marker_ids": {"image": 31, "audio": 32},
        "seam_placeholder_ids": {"image": 40, "audio": 45},
    }

    cfg = td.mk_config(multimodal_config={})
    llm, renderer, tok = td.mk_engine(td.SeamWordTokenizer(), config=cfg)
    online = downlink.OnlineRenderer(
        cfg.model_config, renderer,
        request_logger=None, chat_template=None, chat_template_content_format="string",
    )
    req_obj = td.FakeChatRequest(messages=MESSAGES)
    params = downlink.SamplingParams(max_tokens=4)

    # -- round 1: first send — dict-of-list vs flattened, prompt order --------
    ei1 = await render_once(online, renderer, req_obj)
    await llm.add_request("mm-1", ei1, params)
    req1 = [e[2] for e in llm.events if e[0] == "send_input"][-1]
    specs1 = spec_view(req1)
    out["round1_first_send"] = {
        "action": "交错多模态 prompt（image→audio→image）首次过泳道",
        "prompt_token_ids": ei1["prompt_token_ids"],
        "prompt_token_ids_len": len(ei1["prompt_token_ids"]),
        "mm_placeholders_dict_of_list_before_flatten": ph_view(ei1),
        "dict_order_is_not_prompt_order": True,
        "mm_features_after_flatten": specs1,
        "flattened_count": len(specs1),
        "offsets_ascending": [s["offset"] for s in specs1],
        "is_sorted_by_offset": (
            [s["offset"] for s in specs1]
            == sorted(s["offset"] for s in specs1)
        ),
        "order_matches_prompt": [s["modality"] for s in specs1]
        == ["image", "audio", "image"],
        "data_present_first_time": [s["data"] is not None for s in specs1],
        "processor_cache_hits_so_far": renderer.mm_processor.cache.hits,
    }

    # -- round 2: resend same items -> cache hit -> data=None, hash still crosses
    ei2 = await render_once(online, renderer, req_obj)
    await llm.add_request("mm-2", ei2, params)
    req2 = [e[2] for e in llm.events if e[0] == "send_input"][-1]
    specs2 = spec_view(req2)
    out["round2_cache_hit"] = {
        "action": "同一批 item 重发（processor cache 命中）",
        "engine_input_mm_kwargs_payloads": [
            None if item is None else str(item)
            for items in ei2["mm_kwargs"].values() for item in items
        ],
        "mm_features_after_flatten": specs2,
        "data_none_on_hit": [s["data"] is None for s in specs2],
        "hashes_still_cross": [s["mm_hash"] for s in specs2],
        "hashes_equal_to_round1": (
            [s["mm_hash"] for s in specs2] == [s["mm_hash"] for s in specs1]
        ),
        "processor_cache_hits_so_far": renderer.mm_processor.cache.hits,
        "cache_hits_delta": renderer.mm_processor.cache.hits
        - out["round1_first_send"]["processor_cache_hits_so_far"],
        "identifier_uses_bare_hash": specs2[0]["identifier"] == specs2[0]["mm_hash"],
    }

    # -- round 3: tower-connector LoRA prefixes the identifier ----------------
    cfg_lora = td.mk_config(
        multimodal_config={},
        lora_config={"enable_tower_connector_lora": True},
    )
    llm_l, renderer_l, _ = td.mk_engine(td.SeamWordTokenizer(), config=cfg_lora)
    online_l = downlink.OnlineRenderer(
        cfg_lora.model_config, renderer_l,
        request_logger=None, chat_template=None, chat_template_content_format="string",
    )
    ei3 = await render_once(online_l, renderer_l, req_obj)
    await llm_l.add_request(
        "mm-3", ei3, params, lora_request=downlink.LoRARequest(lora_name="style")
    )
    req3 = [e[2] for e in llm_l.events if e[0] == "send_input"][-1]
    specs3 = spec_view(req3)
    plain_cfg = td.mk_config(multimodal_config={}, lora_config={"enable_tower_connector_lora": False})
    llm_p, renderer_p, _ = td.mk_engine(td.SeamWordTokenizer(), config=plain_cfg)
    online_p = downlink.OnlineRenderer(
        plain_cfg.model_config, renderer_p,
        request_logger=None, chat_template=None, chat_template_content_format="string",
    )
    ei4 = await render_once(online_p, renderer_p, req_obj)
    await llm_p.add_request(
        "mm-4", ei4, params, lora_request=downlink.LoRARequest(lora_name="style")
    )
    req4 = [e[2] for e in llm_p.events if e[0] == "send_input"][-1]
    specs4 = spec_view(req4)
    out["round3_lora_identifier_prefix"] = {
        "action": "enable_tower_connector_lora=True/False 各过一次（lora_name=\"style\"）",
        "tower_connector_identifier": specs3[0]["identifier"],
        "tower_connector_hash": specs3[0]["mm_hash"],
        "tower_identifier_is_prefixed": specs3[0]["identifier"]
        == f"style:{specs3[0]['mm_hash']}",
        "plain_lora_identifier": specs4[0]["identifier"],
        "plain_identifier_is_bare_hash": specs4[0]["identifier"]
        == specs4[0]["mm_hash"],
    }

    dest = Path(__file__).resolve().parent / "m8_mm_flatten.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
