"""m14 worked-example driver — exit loading (output_processor.py:L366-L386 +
L388-L423): DELTA tail-slices logprobs[-len(token_ids):] into each
CompletionOutput while cumulative_logprob rides along whole; prompt logprobs
pop exactly once in DELTA (取走即清空) vs direct-read in FINAL/CUMULATIVE.

Three RequestStates over the same 3-step token stream [256, 257, 65]:
  DELTA       stream=True face  -> every output carries ONE logprobs entry;
  CUMULATIVE  library face      -> every output carries the WHOLE list;
  FINAL_ONLY  stream=False face -> (impl walks the straight line; the three
               gates are ch7's domain — the middle outputs still construct).
The detokenizer is a HOST face maintaining output_token_ids exactly like the
real IncrementalDetokenizer does across steps (ch7 product).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

IMPL = Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

import logprobs_lane as lane  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def R(x, nd=4):
    return round(float(x), nd)


class DetokFace:
    """HOST face of ch7's IncrementalDetokenizer output side."""

    def __init__(self):
        self.output_token_ids = []

    def get_next_output_text(self, finished, delta):
        return ""


def make_request(rid, output_kind):
    params = lane.SamplingParams(
        logprobs=1, prompt_logprobs=1, output_kind=output_kind
    )
    req = lane.EngineCoreRequest(
        request_id=rid,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=0,
    )
    req.external_req_id = f"ext-{rid}"
    return req


def drive(output_kind, label, prompt_tensors_round1=None):
    op = lane.OutputProcessor(tokenizer=None, log_stats=False)
    rid = label
    op.add_request(make_request(rid, output_kind), prompt="hi", parent_req=None,
                   request_index=0, queue=None, detokenizer=DetokFace())
    rs = op.request_states[rid]
    detok = rs.detokenizer

    rounds = []
    for i, (tid, lp_val, top_val) in enumerate(
        [(256, -0.25, -0.5), (257, -1.5, -2.0), (65, -0.05, -0.1)]
    ):
        lp = lane.LogprobsLists(
            logprob_token_ids=np.array([[tid, 256]], dtype=np.int32),
            logprobs=np.array([[lp_val, top_val]]),
            sampled_token_ranks=np.array([1], dtype=np.int32),
        )
        kwargs = {}
        if i == 0 and prompt_tensors_round1 is not None:
            kwargs["new_prompt_logprobs_tensors"] = prompt_tensors_round1
        eco = lane.EngineCoreOutput(
            request_id=rid, new_token_ids=[tid], new_logprobs=lp, **kwargs
        )
        detok.output_token_ids.append(tid)  # ch7 face: ids accumulate
        res = op.process_outputs([eco])
        ro = res.request_outputs[0]
        co = ro.outputs[0]
        rounds.append(
            {
                "round": i + 1,
                "new_token": tid,
                "token_ids_in_completion": [int(t) for t in co.token_ids],
                "logprobs_entries": len(co.logprobs),
                "logprobs_first_decoded_or_none": (
                    None
                    if co.logprobs is None
                    else (
                        next(iter(co.logprobs[-1].values())).decoded_token
                        if hasattr(co.logprobs[-1], "values")
                        else None
                    )
                ),
                "cumulative_logprob": R(co.cumulative_logprob),
                "prompt_logprobs_len": (
                    None
                    if ro.prompt_logprobs is None
                    else len(ro.prompt_logprobs)
                ),
            }
        )
    return rounds


def main():
    pt = lane.LogprobsTensors.empty_cpu(2, 2)  # 2 scored prompt positions
    pt.logprob_token_ids.copy_(
        torch.tensor([[256, 257], [257, 256]], dtype=torch.int32)
    )
    pt.logprobs.copy_(
        torch.tensor([[-0.1, -0.3], [-0.2, -0.4]], dtype=torch.float32)
    )
    pt.selected_token_ranks.copy_(torch.tensor([1, 1], dtype=torch.int32))

    trace = {
        "mechanism": "m14",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "token_stream": [256, 257, 65],
            "k": 1,
            "sampled_logprobs": [-0.25, -1.5, -0.05],
            "prompt_tensors_round1": "2 scored positions (k=1)",
            "detokenizer": "HOST face maintaining output_token_ids like the real ch7 detokenizer",
        },
        "DELTA_stream_true": drive(
            lane.RequestOutputKind.DELTA, "d0", prompt_tensors_round1=pt
        ),
        "CUMULATIVE_library": drive(
            lane.RequestOutputKind.CUMULATIVE, "c0", prompt_tensors_round1=pt
        ),
        "FINAL_ONLY_stream_false": drive(
            lane.RequestOutputKind.FINAL_ONLY, "f0", prompt_tensors_round1=pt
        ),
        "notes": {
            "DELTA_prompt_pop": "round 1 carries the prompt logprobs (None placeholder + 2 positions = 3), later rounds get the emptied container (0)",
            "three_gates": "impl make_request_output walks the straight line — FINAL_ONLY's no-middle-output gate is ch7's domain (impl-notes known deviation 4)",
        },
    }

    p = Path(__file__).resolve().parent / "m14_exit_loading.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
