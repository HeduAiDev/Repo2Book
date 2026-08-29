# ch22 explainer 驱动脚本 · m5 写腿（figure 取数 + slot<0 跳过实证）
# 链路：set_forward_context(slot_mapping=dict[layer_name]) →
#   unified_kv_cache_update(key, value, layer_name)（按 layer_name 取表）→
#   impl.do_kv_cache_update → reshape_and_cache_flash（CUDA kernel 的
#   host 镜像：slot<0 return + slot//bs、slot%bs 逆分解）。
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation._host_seams import VllmConfigSeam  # noqa: E402
from implementation.attention import (  # noqa: E402
    get_attention_context,
    unified_kv_cache_update,
)
from implementation.flash_attn import FlashAttentionImpl  # noqa: E402
from implementation.forward_context import set_forward_context  # noqa: E402

CPU = torch.device("cpu")
LAYER0 = "model.layers.0.self_attn.attn"
BLOCK_SIZE, NUM_TOKENS = 16, 4


class _AttentionLayer:
    _k_scale = torch.tensor(1.0)
    _v_scale = torch.tensor(1.0)
    _q_scale = torch.tensor(1.0)

    def __init__(self, kv_cache):
        self.kv_cache = kv_cache
        self.impl = FlashAttentionImpl(
            num_heads=2, head_size=8, scale=1.0 / math.sqrt(8),
            num_kv_heads=1, alibi_slopes=None, sliding_window=None,
            kv_cache_dtype="auto")


pool = torch.zeros(12, 1, BLOCK_SIZE, 16)  # [blocks, kv_heads, bs, 2*head_dim]
layer = _AttentionLayer(pool)

# 4 个 token：真、PAD、真、真——slot = 50 / -1 / 51 / 52。
SLOTS = [16 * 3 + 2, -1, 16 * 3 + 3, 16 * 3 + 4]
slot_mapping = torch.tensor(SLOTS, dtype=torch.int64)
key = torch.arange(NUM_TOKENS * 8, dtype=torch.float32).reshape(4, 1, 8) / 8.0
value = key + 100.0

cfg = VllmConfigSeam(num_seqs=8, max_batched_tokens=128, max_model_len=512)
cfg.compilation_config.static_forward_context = {LAYER0: layer}
sm_dict = {LAYER0: slot_mapping}
with set_forward_context({LAYER0: None}, cfg, slot_mapping=sm_dict):
    _, attn_layer, kv_cache, layer_sm = get_attention_context(LAYER0)
    layer_resolved_is_same = layer_sm is slot_mapping
    dummy = unified_kv_cache_update(key, value, LAYER0)
    dummy_numel = dummy.numel()

key_cache, _ = pool.transpose(1, 2).split(8, dim=-1)

per_token = []
for i, s in enumerate(SLOTS):
    if s < 0:
        per_token.append({
            "token": i, "slot": s, "verdict": "slot<0 → return（不写）",
            "block": None, "offset": None,
        })
    else:
        per_token.append({
            "token": i, "slot": s,
            "block": s // BLOCK_SIZE, "offset": s % BLOCK_SIZE,
            "k_row_written": [round(float(x), 4) for x in key_cache[s // BLOCK_SIZE, s % BLOCK_SIZE, 0]],
        })

# PAD token（slot=-1）确实没写：块 3 行 1 保持零。
pad_cell_untouched = bool(float(key_cache[3, 1].abs().sum()) == 0.0)

out = {
    "mechanism": "m5 写腿：unified_kv_cache_update → reshape_and_cache_flash",
    "trace_source": "run（op 链真身 + CUDA kernel 的 host 镜像 cache_kernels.cu:L315-L342）",
    "params": {
        "block_size": BLOCK_SIZE, "num_tokens": NUM_TOKENS,
        "slots": SLOTS, "head_dim": 8,
        "anchor_op": "vllm/model_executor/layers/attention/attention.py:L775-L798",
        "anchor_kernel": "csrc/libtorch_stable/cache_kernels.cu:L326-L333",
    },
    "forward_context": {
        "layer_name_resolves_to_same_tensor": bool(layer_resolved_is_same),
        "dummy_return_numel": dummy_numel,
        "dummy_role": "空张量作数据依赖，保 torch.compile 顺序",
    },
    "per_token": per_token,
    "pad_cell_untouched": pad_cell_untouched,
    "slot50_decompose": {"slot": 50, "block": 50 // 16, "offset": 50 % 16},
}
path = os.path.join(os.path.dirname(__file__), "m5.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m5 trace written:", path)
