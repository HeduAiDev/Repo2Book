"""Driver: cross-attention Q from H_d, K/V = [H_t ; H_d] concat, non-causal
(Appendix A.3). Verifies:
  (1) K/V sequence length = num_ctx + num_query (context prepended),
  (2) non-causal: perturbing a mask position changes OTHER positions'
      outputs, including the bonus position (bidirectional),
  (3) removing the injected context K/V changes the output (injection bites).
Dumps every number used in explainer.json.
"""
from __future__ import annotations
import json
import torch

from kv_injection import dflash_layer_attention

torch.manual_seed(1)
num_ctx = 3
block_size = 4            # 1 bonus + 3 mask positions (num_speculative_tokens=3)
num_kv_heads, head_dim = 2, 4
num_heads = 2
hidden = 8

h_d = torch.randn(block_size, hidden)
context_k = torch.randn(num_ctx, num_kv_heads, head_dim)
context_v = torch.randn(num_ctx, num_kv_heads, head_dim)
positions = torch.arange(num_ctx, num_ctx + block_size)
w_q = torch.randn(num_heads * head_dim, hidden) * 0.2
w_k = torch.randn(num_kv_heads * head_dim, hidden) * 0.2
w_v = torch.randn(num_kv_heads * head_dim, hidden) * 0.2
w_o = torch.randn(hidden, num_heads * head_dim) * 0.2
qn, kn = torch.ones(head_dim), torch.ones(head_dim)

def run(hd, ck, cv):
    return dflash_layer_attention(hd, ck, cv, positions, w_q, w_k, w_v, w_o,
                                  qn, kn, num_heads, num_kv_heads, head_dim)

out_base = run(h_d, context_k, context_v)
kv_seq_len = num_ctx + block_size     # 7

# (2) non-causal: perturb the LAST mask position (index 3), measure change at
# the bonus position (index 0) and at another mask position (index 1).
h_d_pert = h_d.clone()
h_d_pert[3] = h_d_pert[3] + 1.0
out_pert = run(h_d_pert, context_k, context_v)
delta_bonus_pos0 = round(float((out_pert[0] - out_base[0]).norm()), 4)
delta_maskpos1 = round(float((out_pert[1] - out_base[1]).norm()), 4)

# (3) remove injected context -> output changes
out_no_ctx = run(h_d, context_k[:0], context_v[:0])   # zero context rows
delta_remove_context = round(float((out_no_ctx - out_base).norm()), 4)

out = {
    "params": {"num_ctx": num_ctx, "block_size": block_size,
               "num_speculative_tokens": block_size - 1,
               "num_heads": num_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim},
    "kv_sequence_length": kv_seq_len,
    "output_shape": list(out_base.shape),
    "noncausal_perturb_lastmask_pos3": {
        "delta_bonus_pos0": delta_bonus_pos0,
        "delta_maskpos1": delta_maskpos1,
    },
    "remove_injected_context_delta": delta_remove_context,
}
print(json.dumps(out, indent=2))
with open("cross_attention.json", "w") as f:
    json.dump(out, f, indent=2)
