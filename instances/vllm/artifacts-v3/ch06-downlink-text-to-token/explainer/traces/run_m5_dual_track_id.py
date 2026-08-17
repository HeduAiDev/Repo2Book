"""Driver for m5 (dual-track request id) — host run against the ch06
subtract-only companion (pin vLLM v0.27.1).

The worked example is the non-degenerate case the mechanism exists for: a
client RETRIES with the SAME external request id — two engine-side requests
must stay distinguishable (v0's single-track id collided here). Records:
- round 1: first add_request("chatcmpl-7f3a") -> external_req_id keeps the
  user id, request_id gains "-<8 hex>" suffix (PR #27987);
- round 2: retry with the same external id -> a DIFFERENT 8-hex suffix; the
  OutputProcessor external->internal map now holds 2 internal ids under 1
  external id; both requests crossed (2 ADD frames, client_index stamped 0);
- round 3: VLLM_DISABLE_REQUEST_ID_RANDOMIZATION -> internal == external and
  the correctness warning fires;
- round 4: an EngineCoreRequest that arrives with external_req_id PRE-SET is
  rejected (the field is vLLM-internal, input_processor.py:L236-L239);
- random_uuid() shape: 16 hex chars, f"{...:.8}" truncates to 8 (32 bits).
"""
import asyncio
import importlib
import json
import logging
import re
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_downlink")
downlink = td.downlink


class _Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(record.getMessage())


def add_frames(llm):
    return [e for e in llm.events if e[0] == "send_input"]


async def main():
    out = {
        "driver": "run_m5_dual_track_id.py",
        "mechanism": "m5 双轨 request_id（assign_request_id, PR #27987）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "external_id": "chatcmpl-7f3a",
    }
    cap = _Cap()
    downlink.logger.addHandler(cap)
    downlink._ONCE_SEEN.clear()

    llm, renderer, tok = td.mk_engine()
    params = lambda: downlink.SamplingParams(max_tokens=4)  # noqa: E731

    # -- round 1: first send --------------------------------------------------
    q1 = await llm.add_request(
        "chatcmpl-7f3a", downlink.tokens_input([3, 4, 5]), params()
    )
    r1 = add_frames(llm)[-1][2]
    sfx1 = r1.request_id.removeprefix("chatcmpl-7f3a-")
    out["round1_first_send"] = {
        "action": 'add_request(request_id="chatcmpl-7f3a")',
        "external_req_id": r1.external_req_id,
        "internal_request_id": r1.request_id,
        "suffix": sfx1,
        "suffix_len": len(sfx1),
        "suffix_is_8_hex": bool(re.fullmatch(r"[0-9a-f]{8}", sfx1)),
        "collector_key": q1.request_id,
        "collector_key_is_internal_id": q1.request_id == r1.request_id,
        "client_index_stamped": r1.client_index,
        "add_frames_total": len(add_frames(llm)),
    }

    # -- round 2: retry with the SAME external id -----------------------------
    q2 = await llm.add_request(
        "chatcmpl-7f3a", downlink.tokens_input([3, 4, 6]), params()
    )
    r2 = add_frames(llm)[-1][2]
    sfx2 = r2.request_id.removeprefix("chatcmpl-7f3a-")
    ext_map = llm.output_processor.external_req_ids
    out["round2_retry_same_external"] = {
        "action": 'add_request(request_id="chatcmpl-7f3a")  # 重试/复用同一外部 id',
        "external_req_id": r2.external_req_id,
        "internal_request_id": r2.request_id,
        "suffix": sfx2,
        "suffix_differs_from_round1": sfx1 != sfx2,
        "demux_map_external_keys": len(ext_map),
        "demux_map_internal_ids_under_external": len(ext_map["chatcmpl-7f3a"]),
        "request_states_registered": len(llm.output_processor.request_states),
        "add_frames_total": len(add_frames(llm)),
        "engine_sees_two_distinct_ids": r1.request_id != r2.request_id,
    }

    # -- round 3: randomization disabled (escape hatch) -----------------------
    downlink.envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION = True
    downlink._ONCE_SEEN.clear()
    q3 = await llm.add_request(
        "chatcmpl-noRand", downlink.tokens_input([3]), params()
    )
    r3 = add_frames(llm)[-1][2]
    warnings_r3 = [m for m in cap.records if "RANDOMIZATION" in m]
    out["round3_randomization_disabled"] = {
        "action": "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 后 add_request(\"chatcmpl-noRand\")",
        "external_req_id": r3.external_req_id,
        "internal_request_id": r3.request_id,
        "internal_equals_external": r3.request_id == r3.external_req_id,
        "warning_fired": len(warnings_r3) == 1,
        "warning_snippet": warnings_r3[0][:60] + "..." if warnings_r3 else None,
    }
    downlink.envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION = False

    # -- round 4: preset external_req_id is rejected ---------------------------
    req4 = llm.input_processor.process_inputs(
        "chatcmpl-preset",
        downlink.tokens_input([3, 9]),
        params(),
        supported_tasks=("generate",),
    )
    req4.external_req_id = "preset-by-caller"
    error = None
    try:
        downlink.InputProcessor.assign_request_id(req4)
    except ValueError as e:
        error = str(e)
    out["round4_preset_external_rejected"] = {
        "action": "EngineCore.external_req_id 预先填值后调 assign_request_id",
        "raised": error is not None,
        "error_type": "ValueError",
        "error_snippet": error[:70] + "..." if error else None,
    }

    # -- random_uuid shape ------------------------------------------------------
    sample = downlink.random_uuid()
    truncated = f"{sample:.8}"
    out["random_uuid_shape"] = {
        "random_uuid_sample": sample,
        "random_uuid_len": len(sample),
        "is_16_hex": bool(re.fullmatch(r"[0-9a-f]{16}", sample)),
        "fstring_precision_truncated": truncated,
        "truncated_len": len(truncated),
        "truncated_is_prefix": sample.startswith(truncated),
        "bits_of_entropy": 32,
    }

    dest = Path(__file__).resolve().parent / "m5_dual_track_id.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
