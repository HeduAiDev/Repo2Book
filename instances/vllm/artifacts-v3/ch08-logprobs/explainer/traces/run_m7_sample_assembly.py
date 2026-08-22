"""m7 worked-example driver — _update_sample_logprobs main assembly
(logprobs.py:L69-L119): per-column .tolist() -> convert_ids_list_to_tokens
(non-incremental per-token decode) -> cumulative_logprob += logprobs[0] ->
append into the container. REAL Rust byte-fallback tokenizer
(tokens 256="hello" / 257=" world" / 65="A").
"""
import json
import sys
from pathlib import Path

import numpy as np

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


def make_request(rid, params):
    return lane.EngineCoreRequest(
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


def main():
    tk = byte_fallback_tokenizer()
    params = lane.SamplingParams(logprobs=1)  # k=1 -> 2 columns per position
    proc = lane.LogprobsProcessor.from_new_request(
        tokenizer=tk, request=make_request("r0", params)
    )

    rounds = [
        # (token_ids row [sampled, top1], logprobs row, rank)
        ([256, 257], [-0.25, -0.5], 1),
        ([257, 256], [-1.5, -2.0], 1),
        ([65, 256], [-0.05, -0.1], 1),
    ]

    trace = {
        "mechanism": "m7",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "k": 1,
            "tokenizer": "real tokenizers 0.22.2 Rust WordLevel + ByteFallback decoder",
            "vocab_notes": "256=hello 257=' world' 65='A'",
        },
        "initial_state": {
            "cumulative_logprob": proc.cumulative_logprob,
            "logprobs_len": len(proc.logprobs),
        },
        "rounds": [],
    }

    for i, (ids, lps, rank) in enumerate(rounds):
        cum_before = proc.cumulative_logprob
        lp = lane.LogprobsLists(
            logprob_token_ids=np.array([ids], dtype=np.int32),
            logprobs=np.array([lps]),
            sampled_token_ranks=np.array([rank], dtype=np.int32),
        )
        # The same non-incremental decode the assembly runs (for the trace).
        raw_decoded = lane.convert_ids_list_to_tokens(tk, ids)
        proc.update_from_output(
            lane.EngineCoreOutput(
                request_id="r0", new_token_ids=[ids[0]], new_logprobs=lp
            )
        )
        entry = proc.logprobs[i]
        first_id = next(iter(entry))
        first = entry[first_id]
        trace["rounds"].append(
            {
                "round": i + 1,
                "columns_token_ids": ids,
                "columns_logprobs": [R(v) for v in lps],
                "sampled_token_ranks": [rank],
                "non_incremental_decoded": raw_decoded,
                "cumulative_before": R(cum_before),
                "cumulative_added_logprobs_0": R(lps[0]),
                "cumulative_after": R(proc.cumulative_logprob),
                "container_len": len(proc.logprobs),
                "first_key_is_sampled": int(first_id) == ids[0],
                "first_decoded_token": first.decoded_token,
                "first_rank": int(first.rank),
                "entry_keys": [int(k) for k in entry],
                "entry_decoded": [entry[k].decoded_token for k in entry],
            }
        )

    trace["final"] = {
        "cumulative_logprob": R(proc.cumulative_logprob),
        "positions": len(proc.logprobs),
        "decoded_sequence": [
            next(iter(pos.values())).decoded_token for pos in proc.logprobs
        ],
        "hand_check": "-0.25 + -1.5 + -0.05 = -1.8",
    }

    p = Path(__file__).resolve().parent / "m7_sample_assembly.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
