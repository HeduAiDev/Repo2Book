"""m9 worked-example driver — the rank chain chain((rank,), range(1, k+1))
and the 'position 0 = sampled token' invariant (logprobs.py:L175-L206 +
logprobs.py:L107-L109 of v1/engine). Cases:

  1 sampled NOT in topk: keys keep [sampled, top1, top2] order, sampled
    carries its own vocab rank (3), topk carry positional 1..k.
  2 sampled == top1: dict key inserted twice — key order keeps sampled
    first, values identical (rank 1) — 'inserting duplicated data twice
    is the same as doing it once' (source comment).
  3 TIE case through the REAL gather + append pipeline: the sampled token
    is tied at logprob 2.5 -> count rank 3 (upper bound) but sits at topk
    position 2; the later topk insert OVERWRITES the value, so the stored
    rank is the positional 2 while insertion ORDER still starts with the
    sampled key. Observable in upstream too (same verbatim code).
  4 k=-1: rank chain expands to all columns (num_logprobs=len(logprobs)).
  5 flat container: FlatLogprobs keeps BOTH duplicate columns (no dict
    dedup at write time); dedup happens per-read in __getitem__.
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


def nested_case(name, token_ids, lps, decoded, rank, k):
    container = lane.create_sample_logprobs(flat_logprobs=False)
    lane.append_logprobs_for_next_position(
        container, token_ids, lps, decoded, rank, k
    )
    entry = container[0]
    return {
        "case": name,
        "token_ids_in": token_ids,
        "logprobs_in": lps,
        "count_rank_in": rank,
        "k": k,
        "key_order": [int(k2) for k2 in entry],
        "stored_ranks": [int(entry[k2].rank) for k2 in entry],
        "stored_logprobs": [R(entry[k2].logprob) for k2 in entry],
        "stored_decoded": [entry[k2].decoded_token for k2 in entry],
    }


def main():
    trace = {
        "mechanism": "m9",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "cases": [],
    }

    # case 1: sampled off podium
    trace["cases"].append(
        nested_case(
            "1: sampled not in topk — own rank leads",
            [12, 4, 7], [-0.4, -1.1, -2.2], ["x", "y", "z"], 3, 2,
        )
    )
    # case 2: sampled == top1
    trace["cases"].append(
        nested_case(
            "2: sampled == top1 — duplicate key, values identical",
            [12, 12, 7], [-0.4, -0.4, -2.2], ["x", "x", "z"], 1, 2,
        )
    )

    # case 3: tie through the real gather pipeline.
    logits = torch.tensor([[3.0, 2.5, 2.5, 1.0]])
    lp = lane.Sampler.compute_logprobs(logits)
    lt = lane.Sampler.gather_logprobs(
        lp, 2, token_ids=torch.tensor([1], dtype=torch.int64)
    )
    indices = [int(v) for v in lt.logprob_token_ids[0].tolist()]
    lps = [R(v) for v in lt.logprobs[0].tolist()]
    count_rank = int(lt.selected_token_ranks[0].item())
    topk_position_of_sampled = indices.index(1, 1)  # first occurrence after col 0
    container = lane.create_sample_logprobs(flat_logprobs=False)
    lane.append_logprobs_for_next_position(
        container, indices, lps, ["a", "b", "c"], count_rank, 2
    )
    entry = container[0]
    trace["cases"].append(
        {
            "case": "3: tie — count rank (upper bound) vs stored rank (positional overwrite)",
            "logits_row": [3.0, 2.5, 2.5, 1.0],
            "sampled_token": 1,
            "gather_columns": indices,
            "gather_logprobs": lps,
            "count_rank_from_sampler": count_rank,
            "topk_position_of_sampled": topk_position_of_sampled,
            "rank_chain_written": [count_rank, 1, 2],
            "key_order": [int(k2) for k2 in entry],
            "stored_ranks": [int(entry[k2].rank) for k2 in entry],
            "note": (
                "dict comprehension: later duplicate key overwrites the VALUE "
                "with the topk positional rank; insertion ORDER keeps the "
                "sampled key first"
            ),
        }
    )

    # case 4: k=-1 expands the chain to every column.
    trace["cases"].append(
        nested_case(
            "4: k=-1 — chain spans all columns",
            [12, 4], [-0.4, -1.1], ["x", "y"], 3, -1,
        )
    )

    # case 5: flat container keeps duplicates at write time.
    flat = lane.create_sample_logprobs(flat_logprobs=True)
    lane.append_logprobs_for_next_position(
        flat, [12, 12, 7], [-0.4, -0.4, -2.2], ["x", "x", "z"], 1, 2
    )
    trace["cases"].append(
        {
            "case": "5: FlatLogprobs — no dedup at write, dedup per read",
            "flat_token_ids": [int(v) for v in flat.token_ids],
            "flat_ranks": [int(r) for r in flat.ranks],
            "flat_logprobs": [R(v) for v in flat.logprobs],
            "flat_start_indices": [int(v) for v in flat.start_indices],
            "flat_end_indices": [int(v) for v in flat.end_indices],
            "positions": len(flat),
            "columns_written": 3,
            "getitem_keys": [int(k2) for k2 in flat[0]],
            "getitem_unique_entries": len(flat[0]),
        }
    )

    p = Path(__file__).resolve().parent / "m9_rank_chain.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
