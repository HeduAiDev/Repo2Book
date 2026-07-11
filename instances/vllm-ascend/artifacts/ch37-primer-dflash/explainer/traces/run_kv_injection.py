"""Driver: KV injection (Appendix A.3). Verifies numerically:
  (1) fused one-GEMM-for-all-layers == per-layer looped projection,
  (2) H_t genuinely depends on all 5 selected target layers,
  (3) swapping target features changes the draft output (conditioning works).
Dumps every number used in explainer.json.
"""
from __future__ import annotations
import json
import torch

from kv_injection import (
    build_fused_kv_weight,
    fuse_target_context_features,
    precompute_layer_kv_fused,
    precompute_layer_kv_looped,
    dflash_layer_attention,
)

torch.manual_seed(0)
L = 2               # draft layers
num_ctx = 3
num_kv_heads, head_dim = 2, 4
kv_size = num_kv_heads * head_dim   # 8
hidden = 8
target_hidden = 8
num_selected = 5

h_t = torch.randn(num_ctx, hidden)
positions = torch.arange(num_ctx)
k_weights = [torch.randn(kv_size, hidden) * 0.1 for _ in range(L)]
v_weights = [torch.randn(kv_size, hidden) * 0.1 for _ in range(L)]
k_norm_weights = [torch.ones(head_dim) for _ in range(L)]

# (1) fused vs looped
fused_w = build_fused_kv_weight(k_weights, v_weights)
k_loop, v_loop = precompute_layer_kv_looped(
    h_t, k_weights, v_weights, k_norm_weights, positions, num_kv_heads, head_dim)
k_fus, v_fus = precompute_layer_kv_fused(
    h_t, fused_w, k_norm_weights, positions, L, num_kv_heads, head_dim)
max_diff_k = round(float((k_fus - k_loop).abs().max()), 6)
max_diff_v = round(float((v_fus - v_loop).abs().max()), 6)
fused_rows = list(fused_w.shape)   # [L*2*kv_size, hidden] = [32, 8]

# (2) H_t depends on every one of the 5 selected layers
w_c = torch.randn(hidden, num_selected * target_hidden) * 0.1
norm_w = torch.ones(hidden)
base_layers = [torch.randn(num_ctx, target_hidden) for _ in range(num_selected)]
h_t_base = fuse_target_context_features(base_layers, w_c, norm_w)
perturb_norms = []
for j in range(num_selected):
    layers = [t.clone() for t in base_layers]
    layers[j] = layers[j] + 1.0          # perturb only layer j
    h_t_j = fuse_target_context_features(layers, w_c, norm_w)
    perturb_norms.append(round(float((h_t_j - h_t_base).norm()), 4))

# (3) conditioning: swap target features -> draft-layer output changes
h_d = torch.randn(4, hidden)
qpos = torch.arange(4)
w_q = torch.randn(num_kv_heads * head_dim, hidden) * 0.1
w_o = torch.randn(hidden, num_kv_heads * head_dim) * 0.1
qn, kn = torch.ones(head_dim), torch.ones(head_dim)
out_base = dflash_layer_attention(
    h_d, k_fus[0], v_fus[0], qpos, w_q, k_weights[0], v_weights[0], w_o, qn, kn,
    num_kv_heads, num_kv_heads, head_dim)
# recompute context K/V from a DIFFERENT target feature
h_t2 = torch.randn(num_ctx, hidden)
k2, v2 = precompute_layer_kv_fused(h_t2, fused_w, k_norm_weights, positions, L, num_kv_heads, head_dim)
out_swapped = dflash_layer_attention(
    h_d, k2[0], v2[0], qpos, w_q, k_weights[0], v_weights[0], w_o, qn, kn,
    num_kv_heads, num_kv_heads, head_dim)
cond_delta = round(float((out_swapped - out_base).norm()), 4)

out = {
    "params": {"L_draft_layers": L, "num_ctx": num_ctx, "num_kv_heads": num_kv_heads,
               "head_dim": head_dim, "kv_size": kv_size, "num_selected_layers": num_selected},
    "fused_vs_looped": {"max_abs_diff_K": max_diff_k, "max_abs_diff_V": max_diff_v,
                        "fused_kv_weight_shape": fused_rows,
                        "layer_major_shape_2_L_ctx_nkv_hd": [2, L, num_ctx, num_kv_heads, head_dim]},
    "h_t_depends_on_each_selected_layer_norm": perturb_norms,
    "conditioning_output_delta_norm": cond_delta,
    "table_reference_ints": [1, 2, 3, 4, 5],  # step/selected-layer labels used in the teaching table
}
print(json.dumps(out, indent=2))
with open("kv_injection.json", "w") as f:
    json.dump(out, f, indent=2)
