# ch22 explainer 驱动脚本 · m4 block_table 双镜像与 commit 先行（figure 取数）
# 展示三件事：
#   1. CPU 侧行长账增量写（append_row 接着写，num_blocks_per_row 记账）；
#   2. commit_block_table(num_reqs) 只拷活跃行——拷贝字节数 = 行数×表宽×4B；
#   3. 非活跃行 CPU 后写、GPU 镜像不跟（copy_to_gpu 只传前缀的直接证据）。
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.block_table import BlockTable  # noqa: E402

CPU = torch.device("cpu")
MAX_NUM_REQS, MAX_BLOCKS_PER_REQ = 8, 8

bt = BlockTable(
    block_size=16, max_num_reqs=MAX_NUM_REQS,
    max_num_blocks_per_req=MAX_BLOCKS_PER_REQ,
    max_num_batched_tokens=128, pin_memory=False, device=CPU,
    kernel_block_size=16, cp_kv_cache_interleave_size=1,
)
bt.append_row([3, 8, 2], row_idx=0)   # r0 首段 3 块
bt.append_row([7, 1], row_idx=0)      # r0 下一拍差量追加 2 块（增量写）
bt.add_row([9, 4, 6, 8], row_idx=1)   # r1 新增请求整行写

num_active_rows = 2
bt.commit_block_table(num_active_rows)
gpu_after_commit = [[int(x) for x in bt.block_table.gpu[r]] for r in range(3)]

# 非活跃行 CPU 侧后写：GPU 镜像不跟。
bt.block_table.np[2, :2] = [88, 88]
gpu_row2_after_stale_write = [int(x) for x in bt.block_table.gpu[2]]

bytes_copied = num_active_rows * MAX_BLOCKS_PER_REQ * 4  # int32 = 4B

out = {
    "mechanism": "m4 block_table 双镜像与 commit 先行（figure 取数）",
    "trace_source": "run（精简版 host 镜像——CpuGpuBuffer 逐字 vllm/v1/utils.py:L110-L149）",
    "params": {
        "max_num_reqs": MAX_NUM_REQS,
        "max_num_blocks_per_req": MAX_BLOCKS_PER_REQ,
        "dtype_bytes": 4, "dtype": "int32",
        "block_size": 16,
        "anchor_commit": "vllm/v1/worker/block_table.py:L213-L214",
        "anchor_prepare_first_line": "vllm/v1/worker/gpu_model_runner.py:L1977-L1979",
    },
    "row_ledger": {
        "row0_after_two_appends": [int(x) for x in bt.block_table.np[0, :5]],
        "row0_len": int(bt.num_blocks_per_row[0]),
        "row1_after_add": [int(x) for x in bt.block_table.np[1, :4]],
        "row1_len": int(bt.num_blocks_per_row[1]),
    },
    "commit": {
        "num_active_rows": num_active_rows,
        "full_table_rows": MAX_NUM_REQS,
        "bytes_copied": bytes_copied,
        "gpu_rows_0_2_after_commit": gpu_after_commit,
        "gpu_row2_after_stale_cpu_write": gpu_row2_after_stale_write,
        "stale_value_seen_on_gpu": bool(88 in gpu_row2_after_stale_write),
    },
}
path = os.path.join(os.path.dirname(__file__), "m4.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m4 trace written:", path)
