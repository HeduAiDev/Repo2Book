# run_m14.py — m14 kernel 块细分与多组块表 驱动脚本
# map_to_kernel_blocks 纯 numpy 算术（[0,1,2]→[0..5]）+ BlockTable 32→2x16
# 细分（append_row 展开）+ MultiGroupBlockTable 每组一表 + 后端协商
# prepare_kernel_block_sizes。
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from implementation.block_table import BlockTable, MultiGroupBlockTable, SlotMappingMode
from implementation.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from implementation.worker_utils import prepare_kernel_block_sizes

OUT = {}

# ------------------------------------------------ map_to_kernel_blocks 算术
ids = np.array([0, 1, 2])
arange = np.arange(0, 2).reshape(1, -1)
OUT["map_to_kernel_blocks"] = {
    "kv_manager_block_ids": ids.tolist(),
    "kv_manager_block_size_tokens": 32,
    "kernel_block_size_tokens": 16,
    "blocks_per_kv_block": 32 // 16,
    "mapped": BlockTable.map_to_kernel_blocks(ids, 2, arange).tolist(),
    "identity_no_split": BlockTable.map_to_kernel_blocks(
        np.array([5, 9]), 1, None).tolist(),
    "formula": "kernel_id = kv_id x blocks_per_kv_block + j, j in [0, k)",
}

# ------------------------------------------------ BlockTable 32 → 2x16
bt = BlockTable(
    block_size=32, max_num_reqs=4, max_num_blocks_per_req=8,
    max_num_batched_tokens=64, pin_memory=False, device=torch.device("cpu"),
    kernel_block_size=16, cp_kv_cache_interleave_size=1,
    slot_mapping_mode=SlotMappingMode.TOKEN_TO_KV_SLOT,
)
bt.append_row([0, 1], row_idx=0)
OUT["block_table_hybrid_split"] = {
    "use_hybrid_blocks": bt.use_hybrid_blocks,
    "blocks_per_kv_block": bt.blocks_per_kv_block,
    "kernel_view_block_size": bt.block_size,
    "max_num_blocks_per_req_before": 8,
    "max_num_blocks_per_req_scaled": bt.max_num_blocks_per_req,
    "row_appended_block_ids": [0, 1],
    "row_first4_kernel_ids": bt.block_table.np[0, :4].tolist(),
    "row_full": bt.block_table.np[0].tolist(),
}

bt_std = BlockTable(
    block_size=16, max_num_reqs=2, max_num_blocks_per_req=4,
    max_num_batched_tokens=32, pin_memory=False, device=torch.device("cpu"),
    kernel_block_size=16, cp_kv_cache_interleave_size=1,
)
OUT["block_table_standard"] = {
    "use_hybrid_blocks": bt_std.use_hybrid_blocks,
    "blocks_per_kv_block": bt_std.blocks_per_kv_block,
}

# ------------------------------------------------ MultiGroupBlockTable 每组一表
mgt = MultiGroupBlockTable(
    max_num_reqs=2, max_num_batched_tokens=32, pin_memory=False,
    device=torch.device("cpu"),
    block_sizes=[32, 16], kernel_block_sizes=[16, 16],
    max_num_blocks=[4, 4],
)
mgt.append_row(([0, 1], [2]), row_idx=0)
OUT["multi_group"] = {
    "num_tables": len(mgt.block_tables),
    "group0_block_size": 32, "group0_kernel": 16,
    "group1_block_size": 16, "group1_kernel": 16,
    "appended_group_ids": [[0, 1], [2]],
    "group0_row": mgt[0].block_table.np[0].tolist(),
    "group1_row": mgt[1].block_table.np[0].tolist(),
    "note": "组 0 细分 [0,1]→[0,1,2,3]；组 1 无细分原样 [2]（余位补 0 占位）",
}

# ------------------------------------------------ 后端协商（纯算术半边）


def full_spec(bs):
    return FullAttentionSpec(
        block_size=bs, num_kv_heads=8, head_size=128, dtype=torch.float16
    )


class Backend16:
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16]


class Backend32Or16:
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [32, 16]


class Group:
    def __init__(self, backend):
        self.backend = backend


config = KVCacheConfig(
    num_blocks=10, kv_cache_tensors=[],
    kv_cache_groups=[
        KVCacheGroupSpec(["a"], full_spec(32)),
        KVCacheGroupSpec(["b"], full_spec(16)),
    ],
)
OUT["backend_negotiation"] = {
    "group0_kv_block_size": 32, "group0_backend_supports": [16],
    "group1_kv_block_size": 16, "group1_backend_supports": [32, 16],
    "kernel_block_sizes": prepare_kernel_block_sizes(
        config, [[Group(Backend16())], [Group(Backend32Or16())]]),
    "rule": "全体后端都支持的最大公因子块；协商内景 → ch21",
    "source_comment": "gpu_model_runner.py 注释例：256-token 内存块拆 4x64",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m14.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
