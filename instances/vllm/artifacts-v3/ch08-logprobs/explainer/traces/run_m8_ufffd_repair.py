"""m8 worked-example driver — U+FFFD context byte repair
(logprobs.py:L312-L346 _verify_tokens + L249-L310 _correct_decoded_token +
L208-L247 _get_sampled_context_ids). REAL Rust byte-fallback tokenizer:
the Chinese char 中 = UTF-8 bytes E4 B8 AD splits into three byte tokens
228/184/173; each single-token decode of a fragment yields the replacement
char U+FFFD — the exact real-HF-tokenizer behavior the repair exists for.

Sequence sampled across 4 positions: [256 "hello", 228, 184, 173], k=1 so
each position also carries a top-1 alternative (the horizontal axis):
  pos4 candidates = [sampled 173, top1 228] — each repaired INDEPENDENTLY
  against the same vertical context [228, 184].
The tokenizer.decode call sequence is recorded live (recorder wrapper).
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
        self.calls = []

    def decode(self, ids):
        out = self._tk.decode([ids] if isinstance(ids, int) else list(ids))
        self.calls.append({"ids": list(ids), "out": out})
        return out


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
    params = lane.SamplingParams(logprobs=1)
    proc = lane.LogprobsProcessor.from_new_request(
        tokenizer=tk, request=make_request("r0", params)
    )

    rounds = [
        ([256, 257], [-0.25, -0.5], 1),   # pos1 clean "hello"
        ([228, 256], [-0.1, -0.3], 1),    # pos2 lead byte  -> U+FFFD -> ""
        ([184, 256], [-0.15, -0.35], 1),  # pos3 middle byte -> U+FFFD -> ""
        ([173, 228], [-0.05, -0.25], 1),  # pos4 completing byte -> "中"; alt 228 -> ""
    ]

    trace = {
        "mechanism": "m8",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "k": 1,
            "char": "中",
            "utf8_bytes_E4_B8_AD": [228, 184, 173],
            "vocab_notes": "256=hello 257=' world'; 228=<0xE4> 184=<0xB8> 173=<0xAD>",
        },
        "rounds": [],
    }

    for i, (ids, lps, rank) in enumerate(rounds):
        tk.calls = []  # record only this position's decode calls
        lp = lane.LogprobsLists(
            logprob_token_ids=np.array([ids], dtype=np.int32),
            logprobs=np.array([lps]),
            sampled_token_ranks=np.array([rank], dtype=np.int32),
        )
        raw_decoded = lane.convert_ids_list_to_tokens(tk, ids)
        context_before = lane.LogprobsProcessor._get_sampled_context_ids(
            proc.logprobs
        )
        proc.update_from_output(
            lane.EngineCoreOutput(
                request_id="r0", new_token_ids=[ids[0]], new_logprobs=lp
            )
        )
        entry = proc.logprobs[i]
        trace["rounds"].append(
            {
                "position": i + 1,
                "candidates_token_ids": ids,
                "candidates_logprobs": [R(v) for v in lps],
                "raw_decoded_before_repair": raw_decoded,
                "vertical_context_ids": [int(c) for c in context_before],
                "vertical_context_len": len(context_before),
                "decode_calls_this_position": list(tk.calls),
                "final_decoded": [entry[k].decoded_token for k in entry],
                "endswith_replacement": [s.endswith("�") for s in raw_decoded],
            }
        )

    trace["final"] = {
        "decoded_sequence_sampled_axis": [
            next(iter(pos.values())).decoded_token for pos in proc.logprobs
        ],
        "cumulative_logprob": R(proc.cumulative_logprob),
        "note": (
            "pos2/pos3 byte fragments decode to '' (incomplete, text "
            "attributed to the completing token); pos4 sampled 173 takes "
            "the whole char '中' while alternative 228 independently -> ''"
        ),
    }

    p = Path(__file__).resolve().parent / "m8_ufffd_repair.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
