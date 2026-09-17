# m2 CPU 池开张：定价公式（PAGESIZE 对齐/跨 worker 分摊）+ canonical 规范化 + /dev/shm 共享区几何。
import mmap

from _driver_common import TESTS, dump

import sys

sys.path.insert(0, str(TESTS))
from _kv_harness import (  # noqa: E402
    LAYER_NAMES,
    assemble_engine,
    make_kv_caches,
    make_kv_config,
    make_vllm_config,
)

from vllm.utils.math_utils import round_up  # noqa: E402
from vllm.v1.kv_offload.config import (  # noqa: E402
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec  # noqa: E402
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion  # noqa: E402


def cpu_cfg(cpu_bytes, world_size, blocks_per_chunk, worker_bytes):
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(tokens_per_block=16, layer_names=tuple(LAYER_NAMES)),),
        model=OffloadingModelConfig(name="dummy-model", dtype="float32"),
        worker_kv_bytes_per_block=worker_bytes,
        enable_kv_cache_events=False,
        extra_config={"cpu_bytes_to_use": cpu_bytes, "blocks_per_chunk": blocks_per_chunk},
        engine_id="e0",
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=blocks_per_chunk),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=world_size,
            tp_size=world_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=True,
        ),
    )


PAGESIZE = mmap.PAGESIZE  # = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

def sizing(cpu_bytes, world_size, blocks_per_chunk, worker_bytes):
    spec = CPUOffloadingSpec(cpu_cfg(cpu_bytes, world_size, blocks_per_chunk, worker_bytes))
    raw_chunk = worker_bytes * world_size * blocks_per_chunk
    aligned = round_up(raw_chunk, PAGESIZE)
    return {
        "cpu_bytes_to_use": cpu_bytes,
        "cpu_bytes_to_use_mib": cpu_bytes // (1 << 20),
        "world_size": world_size,
        "blocks_per_chunk": blocks_per_chunk,
        "worker_kv_bytes_per_block": worker_bytes,
        "kv_bytes_per_block_all_workers": worker_bytes * world_size,
        "kv_bytes_per_chunk_raw": raw_chunk,
        "kv_bytes_per_chunk_aligned": spec.kv_bytes_per_chunk,
        "alignment": PAGESIZE,
        "padding_bytes_per_chunk": aligned - raw_chunk,
        "num_blocks": spec.num_blocks,
        "cpu_page_size_per_worker": spec.cpu_page_size_per_worker,
        "check_num_blocks": cpu_bytes // aligned,
        "check_page_per_worker": raw_chunk // world_size,
        "total_pool_bytes": spec.num_blocks * spec.kv_bytes_per_chunk,
    }


sizing_cases = {
    # 主例：2048B chunk 对齐到 4096 → 每 chunk 垫 2048B padding（非平凡分支）
    "pad_case_100MiB": sizing(100 << 20, 2, 2, 512),
    # 对照：4096B 恰好页对齐 → padding 0
    "no_pad_case_100MiB": sizing(100 << 20, 2, 2, 1024),
    # 小池：8192B 预算只装 2 块
    "tiny_8KiB": sizing(8192, 2, 2, 512),
    # 单 worker
    "world1": sizing(1 << 20, 1, 1, 512),
}

missing_gate = {}
cfg = cpu_cfg(1 << 20, 2, 2, 512)
object.__setattr__(cfg, "extra_config", {"blocks_per_chunk": 2})
try:
    CPUOffloadingSpec(cfg)
    missing_gate["result"] = "constructed"
except Exception as e:
    missing_gate["result"] = f"{type(e).__name__}: {e}"

# ── /dev/shm 共享区几何（源码逐字保留；host 无 /dev/shm，走 no-mmap 回退）──
shm = {
    "mmap_path_pattern": "/dev/shm/vllm_offload_{engine_id}.mmap",
    "block_size_alignment": SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT,
    "row_layout": "|--- W0-B0---|---- W1-B0---| ... | maybe-pad |（spec.py 注释原文）",
    "worker_slot_offset_formula": "rank * cpu_page_size_per_worker（shared_offload_region.py:L70-L72）",
    "scheduler_side_rank": "rank=None（tiering/spec.py:L170-L215——调度器进程也 mmap 同一物理池）",
    "host_note": "Windows host 无 /dev/shm：is_cuda_alike()=False → spec.create_worker 走 no-mmap 张量回退分支（源码自带）；真部署全体 TP worker + 调度器进程 mmap 同一文件",
    "pin": "cudaHostRegister 钉页（gpu_worker.py:L123-L150；host 跳过并打 info 日志）",
}

# ── canonical 规范化：任意布局 → (num_blocks, page) 张量视图 ──
kv_config = make_kv_config()  # 8 块 × 512B 页，2 层共享同一物理张量
eng = assemble_engine(
    make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}), kv_config, make_kv_caches(kv_config)
)
hw = eng.host_worker
canonical = {
    "num_canonical_tensors": len(hw.gpu_tensors),
    "gpu_tensor_shape": list(hw.gpu_tensors[0].shape),
    "gpu_tensor_dtype": str(hw.gpu_tensors[0].dtype),
    "cpu_tensor_shape": list(hw.cpu_tensors[0].shape),
    "num_groups": len(hw.group_refs),
    "refs_per_group": len(hw.group_refs[0]),
    "ref_tensor_indices": [r.tensor_idx for r in hw.group_refs[0]],
    "ref_page_size_bytes": [r.page_size_bytes for r in hw.group_refs[0]],
    "layers_share_one_physical_tensor": True,
    "physical_tensor_count_in_kv_caches": len({t.data_ptr() for t in eng.kv_caches.values()}),
    "blocks_per_chunk_of_engine": hw.blocks_per_chunk,
    "num_cpu_blocks_of_engine": hw.cpu_tensors[0].shape[0],
    "anchor": "offloading/worker.py:L69-L243（register_kv_caches 逐层→CanonicalKVCaches）",
}

dump(
    "m02.json",
    {
        "sizing_cases": sizing_cases,
        "missing_cpu_bytes_gate": missing_gate,
        "shm_region": shm,
        "canonical_normalization": canonical,
    },
)
