# m12 chunk 粒度与 OffloadKey：block_size/blocks_per_chunk 互斥、键打包、fs 三级分桶命名。
import os
import tempfile

from _driver_common import TESTS, dump, key

import sys

sys.path.insert(0, str(TESTS))
from _kv_harness import make_kv_config, make_vllm_config  # noqa: E402

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (  # noqa: E402
    build_offloading_config,
)
from vllm.v1.kv_offload.base import (  # noqa: E402
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.file_mapper import FileMapper  # noqa: E402

# ── ① OffloadKey 打包：block_hash + group_idx(4B) ──
h = b"\x01" * 32
k = make_offload_key(h, 7)
key_packing = {
    "block_hash_len": len(h),
    "group_idx": 7,
    "key_total_len": len(k),
    "group_idx_tail_len": 4,
    "roundtrip_hash_equal": get_offload_block_hash(k) == h,
    "roundtrip_group_equal": get_offload_group_idx(k) == 7,
    "why_bytes": "免去 tuple 装箱 GC——池内唯一键直接当 dict key / 文件名 / 网络载荷",
    "anchor": "vllm/v1/kv_offload/base.py:L29-L41",
}

# ── ② chunk 粒度：block_size(=tokens_per_chunk) 与 blocks_per_chunk 二选一 ──
cfg = make_kv_config()  # GPU block_size=4 → 每 token 块 4
built = build_offloading_config(
    make_vllm_config(extra_config={"cpu_bytes_to_use": 1, "block_size": 8}), cfg
)
chunk_cfg = {
    "gpu_block_tokens": 4,
    "offload_block_size_tokens": 8,
    "blocks_per_chunk": built.cache.blocks_per_chunk,
    "tokens_per_hash": built.cache.tokens_per_hash,
    "hashes_per_chunk": (4 * built.cache.blocks_per_chunk) // built.cache.tokens_per_hash,
    "worker_kv_bytes_per_block": built.worker_kv_bytes_per_block,
    "cpu_page_multiplier": "CPU 页 = GPU 页 × blocks_per_chunk（簿记条数 vs 查找粒度的权衡）",
}
try:
    build_offloading_config(
        make_vllm_config(
            extra_config={"cpu_bytes_to_use": 1, "block_size": 8, "blocks_per_chunk": 2}
        ),
        make_kv_config(),
    )
    chunk_cfg["mutual_exclusion"] = "resolved"
except ValueError as e:
    chunk_cfg["mutual_exclusion"] = f"ValueError: {e}"

# ── ③ fs 层三级分桶命名：<root>/<model>_<digest>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin ──
tmp = tempfile.mkdtemp(prefix="ch38_m12_", dir=str(TESTS.parent))
fm = FileMapper(
    root_dir=tmp,
    model_name="org/name",
    tokens_per_hash=16,
    blocks_per_file=1,
    tp_size=1,
    pp_size=1,
    pcp_size=1,
    dcp_size=1,
    rank=3,
    dtype="float32",
)
name5 = fm.get_file_name(key(5)).replace("\\", "/")
h5 = (5).to_bytes(32, "big").hex()


def rel_parts(seed):
    return fm.get_file_name(key(seed)).replace("\\", "/").split("/")[-4:]


# 同 <hhh> 前缀的键共享第一级桶（限目录扇出）
parts0, parts1 = rel_parts(0), rel_parts(1)
hfar = ((1 << 252)).to_bytes(32, "big").hex()
parts_far = rel_parts(1 << 252)
bucket_case = {
    "key_seed": 5,
    "relpath_parts_count": len(rel_parts(5)),
    "model_dir_suffix": "_r3",
    "model_dir_flattened": "org_name（HF 路径斜杠折平）",
    "bucket1": h5[:3],
    "bucket2": f"{h5[3:5]}_g0",
    "file_name": f"{h5}.bin",
    "seeds_0_1_share_bucket1": parts0[1] == parts1[1] == "000",
    "far_seed_bucket1": parts_far[1],
    "far_seed_differs": parts_far[1] != "000",
    "anchor": "vllm/v1/kv_offload/file_mapper.py:L107-L115 get_file_name",
    "note": "每 rank 写自己的 _r<rank> 兄弟目录——多 rank 安全共享同一 root_dir；digest 含 block_size/并行/dtype",
}

import shutil  # noqa: E402

shutil.rmtree(tmp, ignore_errors=True)

dump(
    "m12.json",
    {
        "offload_key_packing": key_packing,
        "chunk_granularity": chunk_cfg,
        "fs_three_level_buckets": bucket_case,
    },
)
