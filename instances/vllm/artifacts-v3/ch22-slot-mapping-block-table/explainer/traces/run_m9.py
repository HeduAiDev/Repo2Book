# ch22 explainer 驱动脚本 · m9 kernel 块细分（hybrid blocks）
# 分配块（kv_cache_block_size=32）≠ kernel 块（block_size=16）：
#   map_to_kernel_blocks 拆块算术（docstring 例 [0,1,2]→[0..5]）→
#   BlockTable 表宽乘 blocks_per_kv_block=2 → append_row([7]) 落行 [14,15] →
#   slot 恒等式不变（BLOCKS_PER_KV_BLOCK 只乘进 block_indices）。
# 另取 get_block_table_width 的 128-token 对齐 + select_common_block_size 的
# 后端公共块尺寸选取（FA=[MultipleOf(16)] vs int 候选 [32,64]）。
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.backend import MultipleOf  # noqa: E402
from implementation.block_table import (  # noqa: E402
    BlockTable,
    get_block_table_width,
)
from implementation.worker_utils import (  # noqa: E402
    select_common_block_size,
)

CPU = torch.device("cpu")

# 1) 拆块算术：docstring 例逐字复跑。
split_docstring = {
    "kv_manager_block_ids": [0, 1, 2],
    "blocks_per_kv_block": 2,
    "kernel_block_ids": BlockTable.map_to_kernel_blocks(
        np.array([0, 1, 2]), 2, np.arange(0, 2).reshape(1, -1)
    ).tolist(),
    "pass_through_when_k_is_1": True,
}

# 2) hybrid BlockTable：32-token 内存块 × 16-token kernel 块。
bt = BlockTable(
    block_size=32, max_num_reqs=4, max_num_blocks_per_req=8,
    max_num_batched_tokens=64, pin_memory=False, device=CPU,
    kernel_block_size=16, cp_kv_cache_interleave_size=1,
)
bt.add_row([7], row_idx=0)   # kv_manager 块 7 → kernel 块 [14, 15]
bt.commit_block_table(1)
positions = torch.arange(32, dtype=torch.int64)
bt.compute_slot_mapping(1, torch.tensor([0, 32], dtype=torch.int32), positions)
slots = [int(x) for x in bt.slot_mapping.gpu]

per_alloc_block = []
for b, kv_ids in ((0, [0, 1]), (1, [2, 3]), (2, [4, 5]), (7, [14, 15])):
    per_alloc_block.append({
        "alloc_block": b, "kernel_blocks": kv_ids,
        "row_entries_when_appended": (
            BlockTable.map_to_kernel_blocks(
                np.array([b]), 2, np.arange(0, 2).reshape(1, -1)).tolist()),
    })

slot_checkpoints = {
    "pos_0": slots[0], "pos_15": slots[15],
    "pos_16": slots[16], "pos_31": slots[31],
    "first_half_is_block14": all(
        slots[p] == 14 * 16 + p for p in range(16)),
    "second_half_is_block15": all(
        slots[p] == 15 * 16 + (p - 16) for p in range(16, 32)),
}

# 3) 表宽：128-token 对齐 + 细分放大。
width_cases = {
    "align_7_blocks_bs16": get_block_table_width(7, 16),
    "align_5_blocks_bs32_k16": get_block_table_width(5, 32, 16),
    "align_4_blocks_bs32_k16": get_block_table_width(4, 32, 16),
    "no_align_5_blocks_bs32_k16": get_block_table_width(5, 32, 16,
                                                        token_alignment=None),
}


class _FALikeBackend:
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]


class _IntSizesBackend:
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [32, 64]


# 4) 后端公共块尺寸：Case 1 管理块尺寸直用 / Case 2 候选降序取公共因子。
common_cases = {
    "case1_16_fa_only": select_common_block_size(16, [_FALikeBackend]),
    "case1_64_fa_and_int": select_common_block_size(
        64, [_FALikeBackend, _IntSizesBackend]),
    "case2_96_falls_to_32": select_common_block_size(
        96, [_FALikeBackend, _IntSizesBackend]),
}

out = {
    "mechanism": "m9 kernel 块细分：分配块≠kernel 块",
    "trace_source": "run（精简版逐字函数）",
    "params": {
        "kv_cache_block_size": 32, "kernel_block_size": 16,
        "blocks_per_kv_block": 2,
        "use_hybrid_blocks": bool(bt.use_hybrid_blocks),
        "table_width_allocated": 8, "table_width_after_split": int(bt.max_num_blocks_per_req),
        "fa_supported": "MultipleOf(16)",
        "int_backend_supported": [32, 64],
        "anchor_map": "vllm/v1/worker/block_table.py:L220-L248",
        "anchor_select": "vllm/v1/worker/utils.py:L266-L332",
    },
    "split_docstring_example": split_docstring,
    "row_after_append_7": [int(x) for x in bt.block_table.np[0, :2]],
    "per_alloc_block": per_alloc_block,
    "slot_checkpoints": slot_checkpoints,
    "width_cases": width_cases,
    "common_block_size_cases": common_cases,
}
path = os.path.join(os.path.dirname(__file__), "m9.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m9 trace written:", path)
