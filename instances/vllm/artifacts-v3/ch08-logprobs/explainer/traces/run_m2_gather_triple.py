"""m2 worked-example driver — the gather triple: topk values+ids, the sampled
token's logprob, and the COUNT-based vocab rank; cat into [num_tok, k+1] with
the sampled token ALWAYS in column 0 (sampler.py:L308-L356, ops/logprobs.py).

Row design (V=6, k=2 -> 3 columns):
  row0 sampled=0  : the top-1 itself (rank 1) — the common case.
  row1 sampled=3  : rank 4, NOT in topk — proves the k+1-th column exists for
                    the off-podium sampled token.
  row2 sampled=2  : tied logprob at value 2.5 with id 1 — count rank 3 (upper
                    bound shared by ties) while topk orders id1 first.
Also drives batched_count_greater_than directly on the logprob row to show
(x >= values).sum(-1) counting without sorting.
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


def main():
    V = 6
    K = 2
    logits_rows = [
        [3.0, 2.5, 2.5, 1.0, 0.5, 0.0],  # row0: id0 top1; ids 1,2 tied at 2.5
    ]
    sampled_ids = [0, 3, 2]  # one "sampled" per demonstration row

    trace = {
        "mechanism": "m2",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "vocab_size": V,
            "k": K,
            "logits_row": logits_rows[0],
            "sampled_per_row": sampled_ids,
            "columns_k_plus_1": K + 1,
        },
    }

    logprobs = lane.Sampler.compute_logprobs(torch.tensor(logits_rows))  # [1, V]
    trace["log_softmax_row"] = [R(v) for v in logprobs[0].tolist()]

    # Direct kernel demo: count-based rank, no sorting.
    # values: the logprob of each of the three sampled candidates.
    cand_lps = torch.tensor(
        [[logprobs[0, t].item()] for t in sampled_ids], dtype=torch.float32
    )
    x3 = logprobs.expand(3, V)
    counts = lane.batched_count_greater_than(x3, cand_lps)
    trace["batched_count_direct"] = {
        "sampled_ids": sampled_ids,
        "sampled_logprobs": [R(logprobs[0, t].item()) for t in sampled_ids],
        "count_ranks_(x>=v).sum": [int(c) for c in counts.tolist()],
        "note": "count includes the token itself -> 1-based rank; ties share the upper bound",
    }

    # The real gather: one call with three "rows" (same logits tiled).
    lt = lane.Sampler.gather_logprobs(
        logprobs.repeat(3, 1), K, token_ids=torch.tensor(sampled_ids, dtype=torch.int64)
    )
    trace["gather_logprobs_output"] = {
        "indices_num_tok_by_k_plus_1": [
            [int(v) for v in row] for row in lt.logprob_token_ids.tolist()
        ],
        "logprobs_num_tok_by_k_plus_1": [
            [R(v) for v in row] for row in lt.logprobs.tolist()
        ],
        "selected_token_ranks": [int(v) for v in lt.selected_token_ranks.tolist()],
        "shape_indices": list(lt.logprob_token_ids.shape),
        "shape_logprobs": list(lt.logprobs.shape),
        "shape_ranks": list(lt.selected_token_ranks.shape),
        "column_0_is_sampled": [
            int(row[0]) for row in lt.logprob_token_ids.tolist()
        ] == sampled_ids,
        "topk_indices_k": [
            int(v) for v in torch.topk(logprobs[0], K).indices.tolist()
        ],
        "topk_values_k": [
            R(v) for v in torch.topk(logprobs[0], K).values.tolist()
        ],
        "tie_note": "torch.topk breaks the 2.5/2.5 tie by lower index: id1 precedes id2 in topk",
    }

    # Per-row reading of the outcome.
    rows = []
    per_row_expect = [
        ("sampled=top1 (rank 1)", 0, 1),
        ("sampled off-podium (rank 4, not in topk)", 3, 4),
        ("sampled tied at 2.5 (count rank 3, topk position 2)", 2, 3),
    ]
    idx = lt.logprob_token_ids.tolist()
    lps = lt.logprobs.tolist()
    for i, (label, sid, rank) in enumerate(per_row_expect):
        rows.append(
            {
                "row": i,
                "case": label,
                "sampled_token": sid,
                "columns_token_ids": [int(v) for v in idx[i]],
                "columns_logprobs": [R(v) for v in lps[i]],
                "column0_logprob_is_sampled": R(lps[i][0])
                == R(logprobs[0, sid].item()),
                "rank_reported": int(lt.selected_token_ranks[i].item()),
                "rank_expected": rank,
            }
        )
    trace["per_row"] = rows

    p = Path(__file__).resolve().parent / "m2_gather_triple.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
