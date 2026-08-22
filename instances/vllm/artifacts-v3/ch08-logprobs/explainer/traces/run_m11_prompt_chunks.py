"""m11 worked-example driver — prompt logprobs chunked recompute
(gpu_model_runner.py:L5620-L5727): prefill re-runs compute_logits on the
prompt hidden states, target = prompt[i+1], chunks accumulate into
in_progress_prompt_logprobs_cpu and ONLY the last chunk is delivered.

Prompt = 5 tokens, chunk schedule [2, 3] (chunked prefill):
  chunk 1: start_idx=0, num_logits=2 (positions 0-1) -> not delivered,
           tensors stay parked on the request;
  chunk 2: start_idx=2, num_logits=2 (positions 2-3, targets prompt[3..4])
           -> num_tokens 3 > remaining 2 -> last chunk -> delivered once.
Fake model: compute_logits is the identity (hidden states ARE logits rows) —
deterministic and hand-checkable. num_computed_tokens advanced between chunks
by the driver (the real scheduler's accounting).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

IMPL = Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

import logprobs_lane as lane  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def R(x, nd=4):
    return round(float(x), nd)


class IdentityModel:
    """The WC2 two-method contract face: compute_logits(hidden)=hidden."""

    def compute_logits(self, hidden_states):
        return hidden_states


def main():
    PROMPT = [1, 2, 0, 2, 1]  # 5 tokens
    NPLP = 1  # num_prompt_logprobs -> k+1 = 2 columns per position
    # hidden row per prompt position (V=6), peaks walk so top1 differs:
    hidden = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],  # pos0 top1=0
            [4.0, 5.0, 3.0, 2.0, 1.0, 0.0],  # pos1 top1=1
            [3.0, 4.0, 5.0, 2.0, 1.0, 0.0],  # pos2 top1=2
            [2.0, 3.0, 4.0, 5.0, 1.0, 0.0],  # pos3 top1=3
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.0],  # pos4 top1=4
        ]
    )

    req = lane.CachedRequestState(
        req_id="p0",
        prompt_token_ids=list(PROMPT),
        sampling_params=None,
        num_computed_tokens=0,
    )
    runner = lane.GPUModelRunner(
        requests={"p0": req},
        num_prompt_logprobs={"p0": NPLP},
        model=IdentityModel(),
        model_config=SimpleNamespace(logprobs_mode="raw_logprobs"),
        sampler=lane.Sampler(),
        device="cpu",
        query_start_loc=SimpleNamespace(np=np.array([0, 5])),
        input_batch=SimpleNamespace(req_id_to_index={"p0": 0}),
    )

    trace = {
        "mechanism": "m11",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "prompt_token_ids": PROMPT,
            "num_prompt_tokens": len(PROMPT),
            "num_prompt_logprobs": NPLP,
            "chunk_schedule": [2, 3],
            "model": "identity compute_logits (hidden rows are the logits)",
            "hidden_rows": hidden.tolist(),
        },
        "chunks": [],
    }

    for step, num_tokens in enumerate([2, 3]):
        scheduled = {"p0": num_tokens}
        hidden_step = hidden[runner.requests["p0"].num_computed_tokens :
                             runner.requests["p0"].num_computed_tokens + num_tokens]
        before_computed = runner.requests["p0"].num_computed_tokens
        out = runner._get_prompt_logprobs_dict(hidden_step, scheduled)
        req_after = runner.requests["p0"]
        chunk = {
            "chunk": step + 1,
            "num_scheduled_tokens": num_tokens,
            "num_computed_tokens_before": before_computed,
            "start_idx": before_computed,
            "start_tok_target_start": before_computed + 1,
            "num_logits_this_chunk": num_tokens
            if num_tokens <= len(PROMPT) - (before_computed + 1)
            else len(PROMPT) - (before_computed + 1),
            "is_last_chunk": "p0" in out,
            "delivered_dict_keys": list(out.keys()),
            "in_progress_parked_after": req_after.in_progress_prompt_logprobs_cpu
            is not None,
            "num_prompt_logprobs_dict_after": dict(runner.num_prompt_logprobs),
        }
        if "p0" in out:
            lt = out["p0"]
            chunk["delivered_tensors"] = {
                "shape_ids": list(lt.logprob_token_ids.shape),
                "shape_logprobs": list(lt.logprobs.shape),
                "shape_ranks": list(lt.selected_token_ranks.shape),
                "column_ids": [
                    [int(v) for v in row] for row in lt.logprob_token_ids.tolist()
                ],
                "column_logprobs": [
                    [R(v) for v in row] for row in lt.logprobs.tolist()
                ],
                "ranks": [int(v) for v in lt.selected_token_ranks.tolist()],
                "note": "column 0 of each row = the TARGET token prompt[i+1]",
            }
        trace["chunks"].append(chunk)
        # the real scheduler advances the accounting between steps:
        runner.requests["p0"].num_computed_tokens += num_tokens

    trace["accounting"] = {
        "scored_positions_total": len(PROMPT) - 1,
        "empty_cpu_shape_at_first_chunk": [len(PROMPT) - 1, NPLP + 1],
        "first_chunk_park_then_last_chunk_deliver": (
            not trace["chunks"][0]["is_last_chunk"]
            and trace["chunks"][1]["is_last_chunk"]
        ),
        "defer_note": (
            "num_tokens == num_remaining would also take the chunk branch "
            "(defer to a step with new generated tokens, comment L5665-L5670) "
            "— not triggered here: its natural continuation reaches a "
            "subtracted defensive branch (num_logits<=0)"
        ),
    }

    p = Path(__file__).resolve().parent / "m11_prompt_chunks.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
