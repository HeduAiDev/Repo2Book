"""m12 worked-example driver — prompt-logprobs first-token None placeholder +
DELTA pop-once semantics (logprobs.py:L162-L167 create_prompt_logprobs,
v1/engine/logprobs.py:L189-L206 pop_prompt_logprobs).

The first prompt token has no conditional probability (nothing precedes it),
so the container is BORN with a None at index 0. pop_prompt_logprobs returns
everything once and forgets (`if plp:` guard: after reset the empty list is
falsy so the reset is skipped but the empty list is still returned); a
disabled request returns None forever.
"""
import json
import sys
from pathlib import Path

import torch

IMPL = Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

import logprobs_lane as lane  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def R(x, nd=4):
    return round(float(x), nd)


def make_request(rid, params):
    return lane.EngineCoreRequest(
        request_id=rid,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=0,
    )


def main():
    # nested face
    nested = lane.create_prompt_logprobs(flat_logprobs=False)
    # flat face
    flat = lane.create_prompt_logprobs(flat_logprobs=True)

    # feed one prompt tensor batch (2 scored positions, k=1 -> 2 columns)
    params = lane.SamplingParams(logprobs=1, prompt_logprobs=1)
    proc = lane.LogprobsProcessor.from_new_request(
        tokenizer=None, request=make_request("r0", params)
    )
    pt = lane.LogprobsTensors(
        torch.tensor([[256, 257], [257, 256]], dtype=torch.int32),
        torch.tensor([[-0.1, -0.3], [-0.2, -0.4]]),
        torch.tensor([1, 1], dtype=torch.int32),
    )
    proc.update_from_output(
        lane.EngineCoreOutput(
            request_id="r0", new_token_ids=[1], new_prompt_logprobs_tensors=pt
        )
    )

    # disabled request face
    params_off = lane.SamplingParams(logprobs=1)
    proc_off = lane.LogprobsProcessor.from_new_request(
        tokenizer=None, request=make_request("r1", params_off)
    )

    # capture the container state BEFORE popping (pop forgets everything)
    after_len = len(proc.prompt_logprobs)
    pos0_none = proc.prompt_logprobs[0] is None

    first = proc.pop_prompt_logprobs()
    second = proc.pop_prompt_logprobs()
    off = proc_off.pop_prompt_logprobs()

    trace = {
        "mechanism": "m12",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "birth": {
            "nested_first_container": [None],
            "nested_len": len(nested),
            "flat_len": len(flat),
            "flat_position0_empty": flat[0] == {},
            "flat_start_indices": [int(v) for v in flat.start_indices],
            "flat_end_indices": [int(v) for v in flat.end_indices],
            "why": "first prompt token has no preceding context -> no conditional probability to report",
        },
        "after_one_prefill": {
            "container_len": after_len,
            "position0_is_none": pos0_none,
            "scored_positions": 2,
        },
        "pop_once": {
            "first_pop_len": len(first),
            "first_pop_layout": [
                "None" if pos is None else sorted(int(k) for k in pos)
                for pos in first
            ],
            "first_pop_logprobs": [
                None
                if pos is None
                else [R(v.logprob) for v in pos.values()]
                for pos in first
            ],
            "container_len_after_first_pop": len(proc.prompt_logprobs),
            "second_pop_len": len(second),
            "second_pop_is_empty_list_not_none": second == [],
            "disabled_request_pop": off,
            "note": "taken == forgotten: DELTA gets everything exactly once at end of prefill",
        },
    }

    p = Path(__file__).resolve().parent / "m12_first_none_pop.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
