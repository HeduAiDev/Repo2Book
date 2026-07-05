"""驱动脚本：跑真实 _build_block_views（极简路径 block 视图重建）取数值轨迹。

host 无 NPU，但 _build_block_views 是纯 torch 元数据运算（stride/shape/storage_offset/
set_/view，零字节拷贝），CPU 张量可原样跑。我们构造两个「被 runner 超额对齐分配」的
KV 张量，直接调 implementation 里逐字保真的 SimpleCPUOffloadNPUWorker._build_block_views，
并排打印 storage.nbytes() 会给出的错位尺寸 vs shape/stride 精确框出的 num_blocks×page 数据区。

用法：python3 run_build_block_views.py  → 原始输出重定向到 build_block_views.json
"""
import json
import sys
from pathlib import Path

import torch

IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))

from worker import SimpleCPUOffloadNPUWorker  # noqa: E402  逐字保真的极简路径 worker

build = SimpleCPUOffloadNPUWorker._build_block_views
DT = torch.bfloat16          # element_size = 2 bytes
EL = torch.tensor([], dtype=DT).element_size()

out = {"element_size": EL, "cases": {}}

# ────────────────────────────────────────────────────────────────────────
# Case A —— 单段 blocks-outermost，runner 超额对齐：前导偏移 + 尾部 padding
#   逻辑 KV：num_blocks=4，每 block page=4 个 bf16 元素 = 8 字节 → 数据区 32 字节
#   底层 storage 被撑到 48 字节：前面留 6 字节对齐偏移、后面留 10 字节 padding
# ────────────────────────────────────────────────────────────────────────
NUM_BLOCKS_A = 4
PAGE_ELEMS_A = 4                     # stride(0) in elements
LEADING_BYTES = 6                   # 对齐前导偏移
TRAILING_BYTES = 10                # 尾部 padding
storage_elems = (LEADING_BYTES + NUM_BLOCKS_A * PAGE_ELEMS_A * EL + TRAILING_BYTES) // EL
base = torch.arange(storage_elems, dtype=DT)              # 48 字节底层 storage
tensor_a = torch.empty(0, dtype=DT).set_(
    base.untyped_storage(),
    LEADING_BYTES // EL,           # storage_offset = 3 个元素 = 6 字节
    (NUM_BLOCKS_A, PAGE_ELEMS_A),  # shape=(4,4)，shape[0]=4 >= num_blocks=4 → 单段分支
    (PAGE_ELEMS_A, 1),
)
nbytes_a = tensor_a.untyped_storage().nbytes()
page_bytes_a = tensor_a.stride(0) * EL
data_bytes_a = NUM_BLOCKS_A * page_bytes_a
naive_blocks_a = nbytes_a // page_bytes_a               # 用 nbytes 会数出的错位块数
views_a = build("layerA", tensor_a, NUM_BLOCKS_A)
va = views_a["layerA"]
out["cases"]["A_single_segment"] = {
    "num_blocks": NUM_BLOCKS_A,
    "shape": list(tensor_a.shape),
    "stride0_elems": tensor_a.stride(0),
    "storage_offset_elems": tensor_a.storage_offset(),
    "storage_offset_bytes": tensor_a.storage_offset() * EL,
    "leading_offset_bytes": LEADING_BYTES,
    "trailing_padding_bytes": TRAILING_BYTES,
    "storage_nbytes": nbytes_a,
    "page_size_bytes": page_bytes_a,
    "data_bytes_shape_stride": data_bytes_a,
    "naive_blocks_from_nbytes": naive_blocks_a,
    "phantom_blocks": naive_blocks_a - NUM_BLOCKS_A,
    "view_keys": list(views_a.keys()),
    "view_shape": list(va.shape),
    "view_dtype": str(va.dtype),
    "bytes_skipped_total": nbytes_a - data_bytes_a,
}

# ────────────────────────────────────────────────────────────────────────
# Case B —— 多段 (N, num_blocks_physical, …)，K|V 堆在一个分配里（N=2）
#   物理 shape=(2, 6, 4)：每段物理 6 块，但 connector 只认 num_blocks=4
#   → 每段视图只取前 4 块，段间用 stride(0) 跨过 2 个未用物理块
# ────────────────────────────────────────────────────────────────────────
NUM_BLOCKS_B = 4
PHYS_BLOCKS_B = 6
INNER_B = 4
tensor_b = torch.zeros(2, PHYS_BLOCKS_B, INNER_B, dtype=DT)  # 连续，stride=(24,4,1)
nbytes_b = tensor_b.untyped_storage().nbytes()
seg_page_bytes = tensor_b.stride(1) * EL
seg_data_bytes = NUM_BLOCKS_B * seg_page_bytes
seg_stride_bytes = tensor_b.stride(0) * EL
n_segments = tensor_b.shape[0]
total_bytes_b = (n_segments - 1) * seg_stride_bytes + seg_data_bytes
views_b = build("layerB", tensor_b, NUM_BLOCKS_B)
out["cases"]["B_multi_segment"] = {
    "num_blocks": NUM_BLOCKS_B,
    "phys_blocks_per_seg": PHYS_BLOCKS_B,
    "shape": list(tensor_b.shape),
    "stride0_elems": tensor_b.stride(0),
    "stride1_elems": tensor_b.stride(1),
    "n_segments": n_segments,
    "seg_page_size_bytes": seg_page_bytes,
    "seg_data_bytes": seg_data_bytes,
    "seg_stride_bytes": seg_stride_bytes,
    "storage_nbytes": nbytes_b,
    "total_bytes_shape_stride": total_bytes_b,
    "bytes_skipped_total": nbytes_b - (n_segments * seg_data_bytes),
    "view_keys": list(views_b.keys()),
    "seg0_view_shape": list(views_b["layerB.0"].shape),
    "seg1_view_shape": list(views_b["layerB.1"].shape),
    "seg1_start_byte": (n_segments - 1) * seg_stride_bytes,
}

json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
sys.stdout.write("\n")
