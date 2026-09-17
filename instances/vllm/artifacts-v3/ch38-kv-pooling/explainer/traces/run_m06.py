# m6 DMA 引擎：compute_sub_block_ptrs 指针数学（1:1/展开/半块跳越）+ 描述符三缓冲 +
# 真实 store job 的描述符装配（组×层展开）+ 三戒时序（host 不可观察，锚源码注释）。
from _driver_common import TESTS, dump

import sys

sys.path.insert(0, str(TESTS))
import ctypes  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from _kv_harness import (  # noqa: E402
    assemble_engine,
    fill_block_hashes,
    make_kv_caches,
    make_kv_config,
    make_request,
    make_scheduler_output,
    make_vllm_config,
)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.v1.kv_offload.cpu.gpu_worker import (  # noqa: E402
    _new_descriptor_buffers,
    compute_sub_block_ptrs,
)

# ── ① compute_sub_block_ptrs 1:1（blocks_per_chunk=1 快路）──
t = torch.arange(200, dtype=torch.int8)
out = np.empty(2, dtype=np.uint64)
compute_sub_block_ptrs(np.array([0, 1]), 1, out, t.view(2, 100))
one_to_one = {
    "tensor_rows": 2,
    "row_stride": 100,
    "block_ids": [0, 1],
    "ptr_offsets_from_base": [int(p) - t.data_ptr() for p in out],
}

# ── ② 展开 + skip（blocks_per_chunk=2：CPU 块 → 2 个 GPU 子块指针）──
t2 = torch.arange(600, dtype=torch.int8)
view = t2.view(3, 200)  # 3 个 CPU 块、行距 200、每块 2 子块 × 100B
out2 = np.empty(3, dtype=np.uint64)
compute_sub_block_ptrs(np.array([1, 2]), 2, out2, view, skip_count=1)
out3 = np.empty(2, dtype=np.uint64)
compute_sub_block_ptrs(np.array([1, 2]), 2, out3, view, skip_count=0)
expansion = {
    "cpu_blocks": 3,
    "row_stride": 200,
    "sub_block_bytes": 100,
    "blocks_per_chunk": 2,
    "block_ids": [1, 2],
    "skip_1_offsets": [int(p) - t2.data_ptr() for p in out2],
    "skip_0_first_two_offsets": [int(p) - t2.data_ptr() for p in out3],
    "meaning": "skip=1 = 从 CPU 块 1 的第 2 个子块起读（GPU 首块不对齐 offload 块边界的半块跳越）",
}

# ── ③ 描述符三缓冲与 swap_blocks_batch（host seam：memmove 逐描述符搬运）──
s, d, z = _new_descriptor_buffers(4)
src = torch.arange(64, dtype=torch.int8)
dst = torch.zeros(64, dtype=torch.int8)
ops.swap_blocks_batch(
    torch.tensor([src.data_ptr(), src.data_ptr() + 32], dtype=torch.int64),
    torch.tensor([dst.data_ptr() + 32, dst.data_ptr()], dtype=torch.int64),
    torch.tensor([32, 32], dtype=torch.int64),
    is_src_access_order_any=True,
)
descriptor_buffers = {
    "buffer_dtype": str(s.dtype),
    "buffer_shape": list(s.shape),
    "num_descriptors": 2,
    "sizes": [32, 32],
    "dst_half0_equals_src_last32": bytes(dst[32:64].numpy()) == bytes(src[:32].numpy()),
    "dst_half1_equals_src_first32": bytes(dst[0:32].numpy()) == bytes(src[32:].numpy()),
    "src_sum": int(src.sum()),
    "dst_sum": int(dst.sum()),
    "seam": "真源 C++ cache_kernels.cu swap_blocks_batch 批量拷贝 kernel；host = ctypes.memmove 逐描述符等价搬运",
}

# ── ④ 真实 store job 的描述符装配：组×层展开（transfer_async 同式数学）──
kv_config = make_kv_config(num_blocks=16)  # 块号 0-9 全部落在张量内
eng = assemble_engine(
    make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20, "blocks_per_chunk": 3, "offload_prompt_only": False}),
    kv_config,
    make_kv_caches(kv_config),
)
req = make_request("r1", list(range(36)))
eng.scheduler.on_new_request(req)
fill_block_hashes(req, 12)
block_ids = [1, 5, 6, 7, 2, 4, 9, 3, 8]
meta = eng.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": (block_ids,)}],
        num_scheduled_tokens={"r1": 36},
    )
)
job = next(iter(meta.store_jobs.values()))
gpu_spec = job.src_spec
hw = eng.host_worker
# 与 transfer_async 同式：num_copy_ops = Σ group_size × len(group_data_refs)
num_copy_ops = 0
for group_size, refs in zip(gpu_spec.group_sizes, hw.group_refs):
    num_copy_ops += group_size * len(refs)
page_size = hw.group_refs[0][0].page_size_bytes
# 直接执行描述符搬运并核字节（HostOffloadingWorker 与 handler 同一寻址语义）
layer0 = eng.kv_caches[list(eng.kv_caches)[0]]
for i, bid in enumerate(block_ids):
    layer0[bid] = torch.full_like(layer0[bid], 60 + i)
before = [int(layer0[b].sum()) for b in block_ids]
ok = hw.submit_store(0, gpu_spec, job.dst_spec)
after = [int(hw.cpu_tensors[0][slot].sum()) for slot in [int(b) for b in job.dst_spec.block_ids]]
# blocks_per_chunk=3：每个 CPU 槽 = 3 个 GPU 块拼接 → 期望槽校验和 = 每 3 块之和
expected_slot_sums = [sum(before[i : i + 3]) for i in range(0, len(before), 3)]
real_job = {
    "group_sizes": [int(g) for g in gpu_spec.group_sizes],
    "refs_per_group": len(hw.group_refs[0]),
    "num_canonical_tensors": len(hw.gpu_tensors),
    "num_copy_ops": num_copy_ops,
    "descriptor_size_bytes_each": page_size,
    "total_transfer_bytes": num_copy_ops * page_size,
    "src_gpu_blocks": [int(b) for b in gpu_spec.block_ids],
    "dst_cpu_slots": [int(b) for b in job.dst_spec.block_ids],
    "submit_returned": ok,
    "gpu_block_sums_before": before,
    "cpu_slot_sums_after": after,
    "expected_slot_sums_blocks_concatenated": expected_slot_sums,
    "byte_moved_verdict": expected_slot_sums == after,
}

# ── ⑤ DMA 三戒（时序 host 不可观察——锚源码注释原文）──
three_rules = {
    "rule_1_store_waits_compute": "gpu_worker.py:L372-L373：GPU→CPU 先 stream.wait_stream(计算流)——等模型写完再搬",
    "rule_2_same_direction_serial": "gpu_worker.py:L374-L378：新 transfer 的 stream wait_event(前一 transfer 的 end_event)——同向保序",
    "rule_3_load_src_access_any": "gpu_worker.py:L379-L385：CPU→GPU 才开 CU_MEMCPY_SRC_ACCESS_ORDER_ANY（pinned host 无并发 GPU 写；GPU→CPU 读活 KV 必须保 STREAM 序）",
    "completion": "gpu_worker.py:L419-L441：get_finished 按事件 query() 出队（轮询非回调）；wait()=事件 synchronize",
    "kernel_choice": "gpu_worker.py:L40-L42：GPU→CPU 恒用专用拷贝引擎 swap_blocks_batch（带宽受限，专用引擎胜过 Triton）",
    "pooling": "stream/事件/描述符缓冲全池化复用（_stream_pool/_event_pool/_buffer_pool，L431-L436）",
    "host_limitation": "HostStream 替身 wait_stream/wait_event no-op——三戒时序在 host 不可观察；控制流与字节流不变（impl-notes §3/§5）",
}

dump(
    "m06.json",
    {
        "ptr_math_one_to_one": one_to_one,
        "ptr_math_expansion_skip": expansion,
        "descriptor_buffers_and_batch": descriptor_buffers,
        "real_store_job_descriptors": real_job,
        "three_ordering_rules_source_anchored": three_rules,
    },
)
