"""m13 worked-example driver — _update_prompt_logprobs assembly
(logprobs.py:L121-L187): recover the [num_tok, k] shape from the torch
tensors, ONE-SHOT flat detokenization (token_ids.flatten() decoded once),
per-position slicing (offset = pos * num_logprobs), UTF-8 correction with
context accumulated from the container, NO cumulative accumulation.

Prompt = [256 "hello", 228, 184, 173] (the same 中 E4 B8 AD split), k=2 ->
3 columns per scored position: [target, top1, top2].
"""
import json
import sys
from pathlib import Path

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
        prompt_token_ids=[256, 228, 184, 173],
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
    params = lane.SamplingParams(logprobs=1, prompt_logprobs=2)
    proc = lane.LogprobsProcessor.from_new_request(
        tokenizer=tk, request=make_request("r0", params)
    )

    # [num_prompt_tokens-1, k+1] = [3, 3]: [target, top1, top2] per position
    pt = lane.LogprobsTensors(
        torch.tensor(
            [[228, 256, 257], [184, 256, 257], [173, 256, 257]], dtype=torch.int32
        ),
        torch.tensor(
            [[-0.1, -0.9, -1.5], [-0.2, -0.8, -1.4], [-0.05, -0.7, -1.3]]
        ),
        torch.tensor([3, 3, 1], dtype=torch.int32),
    )

    tk.calls = []
    proc.update_from_output(
        lane.EngineCoreOutput(
            request_id="r0", new_token_ids=[173], new_prompt_logprobs_tensors=pt
        )
    )

    positions = []
    for pos, entry in enumerate(proc.prompt_logprobs):
        if entry is None:
            positions.append({"pos": pos, "entry": None})
        else:
            positions.append(
                {
                    "pos": pos,
                    "keys": [int(k) for k in entry],
                    "decoded": [entry[k].decoded_token for k in entry],
                    "logprobs": [R(entry[k].logprob) for k in entry],
                    "ranks": [int(entry[k].rank) for k in entry],
                }
            )

    flat_ids = pt.logprob_token_ids.flatten().tolist()
    trace = {
        "mechanism": "m13",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "prompt": [256, 228, 184, 173],
            "num_prompt_logprobs": 2,
            "columns_k_plus_1": 3,
        },
        "shape_recovery": {
            "logprobs_shape": list(pt.logprobs.shape),
            "num_prompt_tokens_recovered": pt.logprobs.shape[0],
            "num_logprobs_recovered": pt.logprobs.shape[1],
        },
        "flat_one_shot": {
            "flattened_ids": [int(v) for v in flat_ids],
            "flattened_len": len(flat_ids),
            "decode_calls": list(tk.calls),
            "note": "ONE convert_ids_list_to_tokens over the flattened 9 ids, then per-position slices",
        },
        "per_position_slices": {
            "pos0_offset": 0,
            "pos0_offset_end": 3,
            "pos1_offset": 3,
            "pos1_offset_end": 6,
            "pos2_offset": 6,
            "pos2_offset_end": 9,
        },
        "assembled_positions": positions,
        "no_cumulative": {
            "cumulative_logprob": proc.cumulative_logprob,
            "note": "prompt tokens are not model-generated; the sample path alone accumulates",
        },
    }

    p = Path(__file__).resolve().parent / "m13_prompt_assembly.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
