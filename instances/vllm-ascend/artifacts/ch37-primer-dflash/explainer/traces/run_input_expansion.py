"""Driver: input-expansion kernel — pure-Python re-enactment of the index math
in copy_and_expand_dflash_inputs_kernel_single_grid
(vllm_ascend/ops/triton/spec_decode/utils.py:L95-L136). For each request it
lays out the context tokens, then num_query_per_req = 1 + num_speculative_tokens
query tokens: q_idx==0 gets the bonus token, the rest get the mask token
(parallel_drafting_token_id) and are registered into token_indices for
sampling. This mirrors the kernel's control flow exactly (no triton on host).
Dumps every number used in explainer.json.
"""
from __future__ import annotations
import json

# Config
num_speculative_tokens = 3
num_query_per_req = 1 + num_speculative_tokens      # 4
page_block_size = 4                                  # KV page size
PARALLEL_DRAFTING_TOKEN_ID = 151669                  # mask placeholder (config)

# One request: context tokens at positions 0,1,2 ; next_token (bonus) = 42 ;
# block_table maps logical block 0 -> physical block 5, block 1 -> block 9.
req_idx = 0
context_positions = [0, 1, 2]
last_pos = context_positions[-1]                     # 2
bonus_token = 42
block_table = {0: 5, 1: 9}

def slot_of(pos: int) -> int:
    block_num = pos // page_block_size
    block_id = block_table[block_num]
    return block_id * page_block_size + (pos % page_block_size)

# context slots
context_rows = []
for j, pos in enumerate(context_positions):
    context_rows.append({"j": j, "position": pos, "slot": slot_of(pos)})

# query expansion
query_rows = []
for q_idx in range(num_query_per_req):
    query_pos = last_pos + 1 + q_idx
    query_out_idx = req_idx * num_query_per_req + q_idx
    if q_idx == 0:
        input_id = bonus_token
        is_mask = 0
        sample_out_idx = None
    else:
        input_id = PARALLEL_DRAFTING_TOKEN_ID
        is_mask = 1
        sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
    query_rows.append({
        "q_idx": q_idx, "query_pos": query_pos, "query_out_idx": query_out_idx,
        "input_id": input_id, "slot": slot_of(query_pos), "is_mask": is_mask,
        "sample_out_idx": sample_out_idx,
    })

out = {
    "params": {"num_speculative_tokens": num_speculative_tokens,
               "num_query_per_req": num_query_per_req,
               "page_block_size": page_block_size,
               "parallel_drafting_token_id": PARALLEL_DRAFTING_TOKEN_ID,
               "bonus_token": bonus_token, "last_context_pos": last_pos},
    "context_rows": context_rows,
    "query_rows": query_rows,
    "num_mask_positions": num_speculative_tokens,
}
print(json.dumps(out, indent=2))
with open("input_expansion.json", "w") as f:
    json.dump(out, f, indent=2)
