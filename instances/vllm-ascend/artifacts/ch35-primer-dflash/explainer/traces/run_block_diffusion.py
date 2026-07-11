"""Driver: block-diffusion parallel drafting — one forward pass produces the
whole block regardless of block size (Eq.3), vs autoregressive gamma passes
(Eq.2). Runs the reference impl and dumps every number used in explainer.json.

Run:  PYTHONPATH=<chapter>/implementation python3 run_block_diffusion.py
"""
from __future__ import annotations
import json
import torch

from latency_model import autoregressive_draft_cost, diffusion_draft_cost
from dflash_draft_model import (
    TinyDflashDraftModel,
    count_forward_calls_autoregressive,
    count_forward_calls_diffusion,
)

# Illustrative per-pass costs (toy units): one small draft forward = t_step;
# one 5-layer block-diffusion forward over the whole block = t_parallel.
t_step = 0.2
t_parallel = 0.5

rows = []
for gamma in (4, 8, 16):
    ar_calls = count_forward_calls_autoregressive(gamma)
    diff_calls = count_forward_calls_diffusion(gamma)
    ar_cost = round(autoregressive_draft_cost(gamma, t_step), 4)
    diff_cost = round(diffusion_draft_cost(t_parallel, gamma), 4)
    rows.append({
        "gamma": gamma,
        "ar_forward_calls": ar_calls,
        "diff_forward_calls": diff_calls,
        "ar_T_draft": ar_cost,
        "diff_T_draft": diff_cost,
    })

# Structurally verify "single forward returns the whole block" by actually
# running TinyDflashDraftModel once for block_size=4 and reading output shape.
torch.manual_seed(0)
hidden, target_hidden, vocab = 8, 8, 32
num_layers, num_heads, num_kv_heads, head_dim = 3, 2, 2, 4
num_selected = 5
block_size = 4
num_ctx = 3

model = TinyDflashDraftModel(
    hidden_size=hidden, target_hidden_size=target_hidden, vocab_size=vocab,
    num_layers=num_layers, num_heads=num_heads, num_kv_heads=num_kv_heads,
    head_dim=head_dim, num_selected_target_layers=num_selected,
)
sel = [torch.randn(num_ctx, target_hidden) for _ in range(num_selected)]
ctx_pos = torch.arange(num_ctx)
block_ids = torch.tensor([7, 0, 0, 0])  # bonus token=7, rest masked placeholders
q_pos = torch.arange(num_ctx, num_ctx + block_size)

logits = model(sel, ctx_pos, block_ids, q_pos)

out = {
    "params": {"t_step": t_step, "t_parallel": t_parallel,
               "num_draft_layers": num_layers, "block_size_ran": block_size,
               "num_ctx": num_ctx, "vocab": vocab},
    "rows": rows,
    "single_forward_check": {
        "block_size": block_size,
        "logits_shape": list(logits.shape),   # [block_size, vocab] from ONE call
        "logits_rows": logits.shape[0],
        "diff_model_calls": 1,
    },
}
print(json.dumps(out, indent=2))
with open("block_diffusion.json", "w") as f:
    json.dump(out, f, indent=2)
