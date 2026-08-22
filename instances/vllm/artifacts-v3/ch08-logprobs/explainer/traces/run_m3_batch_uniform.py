"""m3 worked-example driver — batch-uniform max_num_logprobs vs per-request
slicing/truncation (gpu_input_batch.py:L435-L440 + L1149-L1151,
scheduler.py:L1909-L1915, outputs.py:L41-L50, logprobs.py:L175-L206).

Batch: r_a wants k=1, r_b wants k=3, r_c silent -> max=3 -> the GPU gathers
k+1=4 columns for EVERY row. On the way back each request's LogprobsProcessor
truncates the rank chain to its own k: r_a keeps 2 entries (sampled+top1),
r_b keeps 4 columns but dedups to 3 (sampled==top1). Also registers
logprobs=-1 -> vocab_size and shows max collapsing as requests finish.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

IMPL = Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

import logprobs_lane as lane  # noqa: E402
import tokenizers  # noqa: E402
import tokenizers.decoders  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def R(x, nd=4):
    return round(float(x), nd)


class ByteFallbackFace:
    def __init__(self, tk):
        self._tk = tk
        self.backend_tokenizer = tk

    def decode(self, ids):
        return self._tk.decode([ids] if isinstance(ids, int) else list(ids))


def byte_fallback_tokenizer():
    vocab = {f"<0x{i:02X}>": i for i in range(256)}
    vocab["hello"] = 256
    vocab[" world"] = 257
    tk = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token=None)
    )
    tk.decoder = tokenizers.decoders.ByteFallback()
    return ByteFallbackFace(tk)


def make_request(rid, prompt_ids, params):
    return lane.EngineCoreRequest(
        request_id=rid,
        prompt_token_ids=list(prompt_ids),
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
    tk = byte_fallback_tokenizer()
    V = 6

    # ---- registration (station 2) ----
    ib = lane.InputBatch(vocab_size=8)
    p_a = lane.SamplingParams(logprobs=1)
    p_b = lane.SamplingParams(logprobs=3)
    p_c = lane.SamplingParams()
    ib.add_request("r_a", 0, p_a)
    ib.add_request("r_b", 1, p_b)
    ib.add_request("r_c", 2, p_c)
    md = ib._make_sampling_metadata()

    reg = {
        "num_logprobs_dict": dict(ib.num_logprobs),
        "max_num_logprobs": ib.max_num_logprobs,
        "metadata_max_num_logprobs": md.max_num_logprobs,
        "silent_request_registered": "r_c" not in ib.num_logprobs,
    }

    # logprobs=-1 -> full vocab registration probe.
    p_d = lane.SamplingParams(logprobs=-1)
    ib.add_request("r_d", 3, p_d)
    reg["minus_one_probe"] = {
        "registered_value": ib.num_logprobs["r_d"],
        "vocab_size": ib.vocab_size,
        "max_after_d": ib.max_num_logprobs,
    }
    ib.remove_request("r_d")

    # ---- the GPU gather at the batch-uniform k=3 (station 3/4) ----
    K_BATCH = 3  # = max_num_logprobs
    logits_rows = [
        [0.0, 1.0, 2.0, 3.0, 1.5, 0.5],  # r_a's row: top3 = ids 3,2,4
        [4.0, 2.0, 1.0, 0.5, 0.0, -1.0],  # r_b's row: top3 = ids 0,1,2
    ]
    sampled = [5, 0]  # r_a samples id5 (rank 5, off podium); r_b id0 (top1)
    lp = lane.Sampler.compute_logprobs(torch.tensor(logits_rows))
    lt = lane.Sampler.gather_logprobs(
        lp, K_BATCH, token_ids=torch.tensor(sampled, dtype=torch.int64)
    )
    lists = lt.tolists()  # -> numpy LogprobsLists (the D2H face, m4)

    # ---- scheduler per-request row slicing (station 6) ----
    sl_a = lists.slice_request(0, 1)
    sl_b = lists.slice_request(1, 1)

    # ---- arrival assembly truncates to each request's own k (station 9/11) ----
    proc_a = lane.LogprobsProcessor.from_new_request(
        tokenizer=tk, request=make_request("r_a", [1], p_a)
    )
    proc_b = lane.LogprobsProcessor.from_new_request(
        tokenizer=tk, request=make_request("r_b", [1], p_b)
    )
    proc_a.update_from_output(
        lane.EngineCoreOutput(
            request_id="r_a", new_token_ids=[5], new_logprobs=sl_a
        )
    )
    proc_b.update_from_output(
        lane.EngineCoreOutput(
            request_id="r_b", new_token_ids=[0], new_logprobs=sl_b
        )
    )

    entry_a = proc_a.logprobs[0]
    entry_b = proc_b.logprobs[0]

    # ---- max collapses as requests finish ----
    ib.remove_request("r_b")
    max_after_b = ib.max_num_logprobs
    ib.remove_request("r_a")
    max_after_a = ib.max_num_logprobs

    trace = {
        "mechanism": "m3",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "batch": "r_a k=1 / r_b k=3 / r_c silent / r_d logprobs=-1 (probe)",
            "vocab_size": V,
            "input_batch_vocab_size": 8,
            "logits_rows": logits_rows,
            "sampled": sampled,
        },
        "registration": reg,
        "gpu_gather_batch_uniform": {
            "k_used_by_sampler": K_BATCH,
            "columns_k_plus_1": K_BATCH + 1,
            "indices": [
                [int(v) for v in row] for row in lists.logprob_token_ids.tolist()
            ],
            "logprobs": [
                [R(v) for v in row] for row in lists.logprobs.tolist()
            ],
            "ranks": [int(v) for v in lists.sampled_token_ranks.tolist()],
        },
        "slice_request": {
            "r_a_rows": sl_a.logprobs.shape[0],
            "r_b_rows": sl_b.logprobs.shape[0],
            "r_a_columns_over_the_wire": sl_a.logprobs.shape[1],
            "r_b_columns_over_the_wire": sl_b.logprobs.shape[1],
        },
        "assembly_truncation": {
            "r_a_own_k": 1,
            "r_a_kept_entries": len(entry_a),
            "r_a_keys": [int(k) for k in entry_a],
            "r_a_decoded": [entry_a[k].decoded_token for k in entry_a],
            "r_a_ranks_stored": [int(entry_a[k].rank) for k in entry_a],
            "r_b_own_k": 3,
            "r_b_columns_received": 4,
            "r_b_kept_entries": len(entry_b),
            "r_b_keys": [int(k) for k in entry_b],
            "r_b_decoded": [entry_b[k].decoded_token for k in entry_b],
            "r_b_note": "sampled id0 == top1 -> dict key dedup merges the duplicate column",
        },
        "max_collapse": {
            "max_initial": 3,
            "max_after_r_b_finishes": max_after_b,
            "max_after_r_a_finishes": max_after_a,
        },
    }

    p = Path(__file__).resolve().parent / "m3_batch_uniform.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
