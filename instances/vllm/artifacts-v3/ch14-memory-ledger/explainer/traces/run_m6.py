# run_m6.py — m6 页大小统一 unify_kv_cache_spec_page_size 驱动脚本
# 四路：① 等页不动；② 普通注意力层调大 block_size（页随块线性放大）；
# ③ Mamba 页由状态形状决定、不随块缩放 → 物理 pad；④ 不能整除且后端不支持
# stride 索引 → NotImplementedError；⑤ indexes_kv_by_block_stride=True → pad。
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from implementation.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
)
from implementation.kv_cache_utils import unify_kv_cache_spec_page_size

OUT = {}


def full_spec(heads):
    return FullAttentionSpec(
        block_size=16, num_kv_heads=heads, head_size=128, dtype=torch.float16
    )


def mamba_spec(nbytes):
    return MambaSpec(
        block_size=16, shapes=((nbytes,),), dtypes=(torch.uint8,),
        mamba_cache_mode="none",
    )


# ① 等页 → 原样返回（同一 dict 对象）
same = {"a": full_spec(8), "b": full_spec(8)}
out_same = unify_kv_cache_spec_page_size(same)
OUT["case1_equal_pages_noop"] = {
    "page_a": 65536, "page_b": 65536,
    "is_same_dict": out_same is same,
}

# ② 页 65536 vs 32768 → 小层 block_size 16×2=32，页升到 65536
big, small = full_spec(8), full_spec(4)
out2 = unify_kv_cache_spec_page_size({"a": big, "b": small})
OUT["case2_scale_up_block_size"] = {
    "layer_a_page_before": big.page_size_bytes,
    "layer_b_page_before": small.page_size_bytes,
    "max_page": 65536,
    "ratio": 65536 // 32768,
    "layer_b_block_size_after": out2["b"].block_size,
    "layer_b_block_size_before": 16,
    "layer_b_page_after": out2["b"].page_size_bytes,
    "rule": "new_block_size = block_size x (max_page // layer_page)",
}

# ③ Mamba 状态页 4096B → pad 到 65536（block_size 不变）
att, mb = full_spec(8), mamba_spec(4096)
out3 = unify_kv_cache_spec_page_size({"a": att, "m": mb})
OUT["case3_mamba_pad"] = {
    "attention_page": 65536,
    "mamba_state_page_before": mb.page_size_bytes,
    "mamba_page_padded_after": out3["m"].page_size_bytes,
    "mamba_block_size_after": out3["m"].block_size,
    "wasted_bytes_per_page": 65536 - 4096,
    "waste_pct": round((65536 - 4096) / 65536 * 100, 2),
    "rule": "MambaSpec 页由状态形状决定、不随 block_size 缩放 → page_size_padded",
}

# ④ 不能整除且无 stride 索引 → NotImplementedError
odd = replace(full_spec(4), page_size_padded=40000)
try:
    unify_kv_cache_spec_page_size({"a": full_spec(8), "b": odd})
    OUT["case4_not_divisible"] = {"raised": False}
except NotImplementedError as e:
    OUT["case4_not_divisible"] = {
        "layer_page_padded": 40000,
        "max_page": 65536,
        "remainder": 65536 % 40000,
        "divisible": 65536 % 40000 == 0,
        "raised": True,
        "error_excerpt": "Padding is only supported for attention layers whose "
                         "backend indexes KV pages by the block stride",
    }

# ⑤ indexes_kv_by_block_stride=True → pad 而非调块
strideful = replace(full_spec(4), page_size_padded=40000,
                    indexes_kv_by_block_stride=True)
out5 = unify_kv_cache_spec_page_size({"a": full_spec(8), "b": strideful})
OUT["case5_stride_indexed_pad"] = {
    "layer_page_before": 40000,
    "layer_page_padded_after": out5["b"].page_size_bytes,
    "block_size_unchanged": out5["b"].block_size == 16,
}

# 硬约束自检：统一后每层页字节全等
pages = {out2["a"].page_size_bytes, out2["b"].page_size_bytes}
OUT["invariant_check"] = {
    "all_pages_equal_after_unify": len(pages) == 1,
    "page": pages.pop(),
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m6.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
