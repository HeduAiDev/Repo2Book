"""m1 worked-example driver — raw logprobs snapshot BEFORE any transformation.

Drives the impl's Sampler.forward (verbatim flow, sampler.py:L72-L149):
  fwd_raw       forward(model_logits, md)  default raw_logprobs  -> the v1
                default lens: logprobs from the ORIGINAL logits (NOTE(woosuk),
                sampler.py:L80-L83). Greedy samples the input's argmax.
  fwd_v0        forward(PENALIZED_logits, md) default mode -> V0 semantics
                emulation: v0 computed the top-k logprobs from the logits
                actually used for sampling (after penalties/temperature), and
                penalties are applied IN PLACE to that tensor (utils.py:L88
                `logits -= presence_penalties.unsqueeze(dim=1) * output_mask`).
                Feeding the already-penalized tensor reproduces exactly what
                v0's logprobs consumer saw.
  fwd_processed a Sampler(logprobs_mode="processed_logprobs") instance run on
                PENALIZED_logits -> the v0.27 four-state knob: the greedy path
                materializes compute_logprobs(logits-after-processors) and
                OVERWRITES the None snapshot (sampler.py:L103-L104) — numbers
                equal fwd_v0. (The mode is set at Sampler construction from
                the engine config; forward's logprobs_mode_override param is
                the spec-decode bonus face.)
  fwd_raw_logits Sampler(logprobs_mode="raw_logits") on model_logits ->
                snapshot WITHOUT log_softmax (the four-state surface).

HOST label: the penalized tensor is built by this DRIVER per the real penalty
formula (vllm/model_executor/layers/utils.py:L88) because the impl's processor
body is the Part VII placeholder (dossier delete item 1) — the impl faithfully
computes log_softmax of whatever logits arrive, so feeding penalized logits in
IS the V0 lens, no simulation of the logprobs math itself.
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
    V = 5
    K = 2  # top-k logprobs requested -> k+1 = 3 columns
    # The model's honest opinion this step (one request, vocab 0..4):
    model_logits = [2.0, 1.9, 0.5, 0.0, -1.0]  # token 0 is the model's top-1
    # Sampling intervention: token 0 was already generated in an earlier
    # step; presence_penalty=2.0 pushes its logit down by the real formula
    # logits -= presence_penalties * output_mask (utils.py:L88).
    presence_penalty = 2.0
    present_token = 0
    penalized = [
        v - presence_penalty if i == present_token else v
        for i, v in enumerate(model_logits)
    ]

    md = metadata(max_num_logprobs=K)

    trace = {
        "mechanism": "m1",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "vocab_size": V,
            "k": K,
            "model_logits": model_logits,
            "presence_penalty": presence_penalty,
            "present_token": present_token,
            "penalized_logits": [R(v) for v in penalized],
            "driver_note": (
                "penalized tensor built per utils.py:L88 formula in the "
                "driver (impl processor body = Part VII placeholder); "
                "impl computes log_softmax of whatever logits arrive"
            ),
        },
    }

    # ---- lens 1: v1 default (raw_logprobs) on the ORIGINAL logits ----
    sampler = lane.Sampler()  # default logprobs_mode="raw_logprobs"
    out_raw = sampler.forward(torch.tensor([model_logits]), md)
    lt = out_raw.logprobs_tensors
    raw_full = lane.Sampler.compute_logprobs(torch.tensor([model_logits]))[0]
    trace["fwd_raw_on_original"] = {
        "mode": "raw_logprobs (v1 default)",
        "sampled_greedy_argmax_of_input": int(out_raw.sampled_token_ids[0, 0]),
        "columns_k_plus_1": list(lt.logprob_token_ids[0].tolist()),
        "column_logprobs": [R(v) for v in lt.logprobs[0].tolist()],
        "sampled_token_ranks": [int(v) for v in lt.selected_token_ranks.tolist()],
        "full_raw_log_softmax_row": [R(v) for v in raw_full.tolist()],
        "note": (
            "greedy path samples the input tensor's argmax; in the real "
            "engine this step's penalties would be applied in place AFTER "
            "this snapshot, so the engine would sample token 1 while the "
            "reported logprobs stay this row (raw opinion)"
        ),
        "raw_logprob_of_penalized_argmax_token_1": R(raw_full[1].item()),
        "raw_logprob_of_model_top1_token_0": R(raw_full[0].item()),
    }

    # ---- lens 2: V0 semantics (logprobs from the logits used for sampling) ----
    out_v0 = sampler.forward(torch.tensor([penalized]), md)
    lt0 = out_v0.logprobs_tensors
    pen_full = lane.Sampler.compute_logprobs(torch.tensor([penalized]))[0]
    trace["fwd_v0_on_penalized"] = {
        "mode": "raw_logprobs mode, PENALIZED input (= V0 lens)",
        "sampled_greedy_argmax_of_input": int(out_v0.sampled_token_ids[0, 0]),
        "columns_k_plus_1": [int(v) for v in lt0.logprob_token_ids[0].tolist()],
        "column_logprobs": [R(v) for v in lt0.logprobs[0].tolist()],
        "sampled_token_ranks": [int(v) for v in lt0.selected_token_ranks.tolist()],
        "full_penalized_log_softmax_row": [R(v) for v in pen_full.tolist()],
        "penalized_top1_is_token_1": True,
        "v0_reported_logprob_of_sampled_token_1": R(pen_full[1].item()),
    }

    # ---- lens 3: v0.27 processed_logprobs knob reproduces V0 explicitly ----
    sampler_proc = lane.Sampler(logprobs_mode="processed_logprobs")
    out_proc = sampler_proc.forward(torch.tensor([penalized]), md)
    ltp = out_proc.logprobs_tensors
    trace["fwd_processed_override"] = {
        "mode": "processed_logprobs (Sampler ctor; greedy path materializes and overwrites, sampler.py:L103-L104)",
        "sampled_greedy_argmax_of_input": int(out_proc.sampled_token_ids[0, 0]),
        "columns_k_plus_1": [int(v) for v in ltp.logprob_token_ids[0].tolist()],
        "column_logprobs": [R(v) for v in ltp.logprobs[0].tolist()],
        "sampled_token_ranks": [int(v) for v in ltp.selected_token_ranks.tolist()],
        "equals_v0_lens": (
            [R(v) for v in ltp.logprobs[0].tolist()]
            == [R(v) for v in lt0.logprobs[0].tolist()]
        ),
    }

    # ---- lens 4: raw_logits mode (snapshot without log_softmax) ----
    sampler_logits = lane.Sampler(logprobs_mode="raw_logits")
    out_rl = sampler_logits.forward(torch.tensor([model_logits]), md)
    ltl = out_rl.logprobs_tensors
    trace["fwd_raw_logits_mode"] = {
        "mode": "raw_logits (no log_softmax; snapshot IS the fp32 logits)",
        "sampled_greedy_argmax_of_input": int(out_rl.sampled_token_ids[0, 0]),
        "columns_k_plus_1": [int(v) for v in ltl.logprob_token_ids[0].tolist()],
        "column_values_are_raw_logits": [R(v) for v in ltl.logprobs[0].tolist()],
        "sampled_token_ranks": [int(v) for v in ltl.selected_token_ranks.tolist()],
        "note": "four-state surface (m16): raw_logits/raw_logprobs/processed_logits/processed_logprobs",
    }

    # ---- the decision delta the user sees for the SAME sampled token 1 ----
    trace["same_token_two_numbers"] = {
        "sampled_token_after_penalty": 1,
        "v1_raw_reported": R(raw_full[1].item()),
        "v0_processed_reported": R(pen_full[1].item()),
        "delta": R(pen_full[1].item() - raw_full[1].item()),
        "model_top1_flips": {
            "raw_lens_top1": 0,
            "processed_lens_top1": 1,
        },
    }

    p = Path(__file__).resolve().parent / "m1_raw_snapshot.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
