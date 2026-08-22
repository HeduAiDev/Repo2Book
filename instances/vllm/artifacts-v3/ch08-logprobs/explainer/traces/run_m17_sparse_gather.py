"""m17 worked-example driver — the logprob_token_ids sparse path
(sampler.py:L111-L149 + L151-L225, v0.27 new, generative-scoring face):
gather logprobs for an EXPLICIT token list per request instead of top-k —
padded to the batch max, column 0 always the sampled token, padded cells
-inf-masked; ranks still counted for the sampled token. When both knobs are
set the sparse result PREEMPTS the dense gather (the `prefer` branch).
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


def metadata(max_num_logprobs=None, logprob_token_ids=None):
    return lane.SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        max_num_logprobs=max_num_logprobs,
        logprob_token_ids=logprob_token_ids,
    )


def main():
    V = 8
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 3.5, 2.5, 1.5, 0.5],  # req0's row
            [4.5, 3.5, 2.5, 1.5, 0.5, 0.0, 0.0, 0.0],  # req1's row
        ]
    )
    # req_index -> requested ids (heterogeneous: 2 vs 1 -> pad to 2)
    sparse = {0: [5, 7], 1: [2]}
    sampled = torch.tensor([3, 1])  # greedy argmax of each row

    lp = lane.Sampler.compute_logprobs(logits)
    sampler = lane.Sampler()

    lt = sampler.gather_specific_token_logprobs(lp, sparse, sampled)

    # the preemption face: forward with BOTH max_num_logprobs and the sparse
    # dict — the sparse result wins over the dense top-k gather.
    md_both = metadata(max_num_logprobs=1, logprob_token_ids=sparse)
    out = sampler.forward(logits, md_both)
    lt_fwd = out.logprobs_tensors

    trace = {
        "mechanism": "m17",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "vocab_size": V,
            "logprob_token_ids_req_index_map": {"0": [5, 7], "1": [2]},
            "sampled": [3, 1],
            "note": "req keys are req_INDEX (InputBatch._make_sampling_metadata converts req_id keys)",
        },
        "sparse_gather_output": {
            "padded_columns": 3,
            "token_ids_matrix": [
                [int(v) for v in row] for row in lt.logprob_token_ids.tolist()
            ],
            "logprobs_matrix": [
                [R(v) for v in row] for row in lt.logprobs.tolist()
            ],
            "ranks": [int(v) for v in lt.selected_token_ranks.tolist()],
            "valid_mask": [
                [True, True, True],
                [True, True, False],
            ],
            "padded_cell_value": "-inf",
            "padded_cell_logprob_row1_col2": R(float(lt.logprobs[1][2].item()), 1),
            "column0_is_sampled": [int(row[0]) for row in lt.logprob_token_ids.tolist()] == [3, 1],
            "hand_lookup_row0": {
                "lp_sampled_3": R(lp[0, 3].item()),
                "lp_requested_5": R(lp[0, 5].item()),
                "lp_requested_7": R(lp[0, 7].item()),
            },
            "hand_lookup_row1": {
                "lp_sampled_1": R(lp[1, 1].item()),
                "lp_requested_2": R(lp[1, 2].item()),
            },
        },
        "preemption_in_forward": {
            "max_num_logprobs_also_set": 1,
            "dense_would_have_columns": 2,
            "forward_returned_columns": int(lt_fwd.logprob_token_ids.shape[1]),
            "sparse_won": int(lt_fwd.logprob_token_ids.shape[1]) == 3,
            "forward_sampled": [int(v) for v in out.sampled_token_ids.flatten().tolist()],
        },
        "scoring_note": (
            "generative-scoring face: compare probabilities of fixed label "
            "tokens without paying the full-vocab (logprobs=-1) bill"
        ),
    }

    p = Path(__file__).resolve().parent / "m17_sparse_gather.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
