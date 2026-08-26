# ch12 m11 驱动：下一拍 GPU 回填 _prepare_input_ids 三岔口——
# 正常拍整段直拷 / async 拍批次未变单 slice 直拷 / 批次变过按 index scatter；
# _compute_prev_positions 槽位映射（-1=新请求）。
# 真源锚点：gpu_model_runner.py:L1769-L1782（prev_positions）、L1784-L1877（三岔口）、
# L1878-L1891（scatter）。
# HOST SEAM：CPU 张量代 GPU 张量——索引算术与控制流与 pin 逐字一致（非 pinned/DMA）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402

cfg = VllmConfig(scheduler_config=SchedulerConfig(async_scheduling=True), max_model_len=64)
runner = GPUModelRunner(cfg, max_num_reqs=8, max_model_len=64, vocab_size=16)


class SO:
    scheduled_spec_decode_tokens = {}


def fill(reqs):
    for rid, prompt in reqs.items():
        runner.input_batch.add_request(rid, prompt)
        runner.requests[rid] = type(
            "ReqState", (), {"num_tokens": len(prompt), "all_token_ids": list(prompt)}
        )()


def prime_prev(tokens):
    runner.input_batch.prev_sampled_token_ids = torch.tensor(
        [[t] for t in tokens], dtype=torch.int32)
    runner.input_batch.prev_req_id_to_index = {
        rid: i for i, rid in enumerate(runner.input_batch.req_ids)}


def reset_input_ids():
    runner.input_ids.cpu[:] = 0
    runner.input_ids.gpu[:] = 0


trace = {
    "mechanism": "m11 下一拍 GPU 回填（_prepare_input_ids 三岔口）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/worker/gpu_model_runner.py:L1769-L1782, L1784-L1877, L1878-L1891",
    "host_seam_note": "HOST SEAM：CPU 张量对（.cpu/.gpu 同机）代 CpuGpuBuffer 的 pinned+DMA 面；scatter/直拷的索引算术与分支判定逐字同 pin",
    "cases": [],
}

# -------- 路径一：正常拍（prev 为 None）——整段 copy_to_gpu
reset_input_ids()
runner.input_batch.prev_sampled_token_ids = None
fill({"req-0": [1, 2]})
runner.input_ids.cpu[:2] = [1, 2]
runner._prepare_input_ids(SO(), num_reqs=1, total_num_scheduled_tokens=2,
                          cu_num_tokens=np.array([2], dtype=np.int64))
trace["cases"].append({
    "case": "正常拍（prev_sampled_token_ids=None）",
    "input_ids_cpu_side": [1, 2],
    "input_ids_gpu_result": [int(v) for v in runner.input_ids.gpu[:2]],
    "path": "copy_to_gpu(total)——GPU 上没有上一拍可回填，token 全来自 CPU 侧账本",
    "anchor": "gpu_model_runner.py:L1801-L1807",
})

# -------- 路径二：async 拍、批次未变未重排——common-case 单 slice 直拷
# 证明『token 来自 prev 而非 CPU』：CPU 侧故意放 0
reset_input_ids()
runner2_reqs = {"req-0": [1, 2], "req-1": [3, 4]}
for rid in list(runner.input_batch.req_ids):
    runner.input_batch.remove_request(rid)
fill(runner2_reqs)
prime_prev([7, 8])  # 上拍采出 7/8
runner._compute_prev_positions(2)
runner.input_ids.cpu[:2] = [0, 0]  # CPU 侧没有可用信息
runner._prepare_input_ids(SO(), num_reqs=2, total_num_scheduled_tokens=2,
                          cu_num_tokens=np.array([1, 2], dtype=np.int64))
trace["cases"].append({
    "case": "async 拍·批次未变未重排（common-case）",
    "prev_positions": [int(v) for v in runner.prev_positions.np[:2]],
    "cu_num_tokens": [1, 2],
    "input_ids_cpu_side_deliberately_zeroed": [0, 0],
    "input_ids_gpu_result": [int(v) for v in runner.input_ids.gpu[:2]],
    "path": "单 slice 直拷 input_ids.gpu[:N] ← prev_sampled_token_ids[:N,0]（L1868-L1877）",
    "proof": "CPU 侧全 0 而结果 [7,8]——token 只能来自 prev 的 GPU 张量（真 token 未过 CPU）",
})

# -------- 路径三：批次变过/重排——按 index scatter
reset_input_ids()
for rid in list(runner.input_batch.req_ids):
    runner.input_batch.remove_request(rid)
# 重排落位：req-2 提到最前 + 新请求 req-x 垫中间 + req-0 在后（req-1 已离开批次）
for rid, prompt in (("req-2", [5, 6]), ("req-x", [11, 12]), ("req-0", [1, 2])):
    runner.input_batch.add_request(rid, prompt)
    runner.requests[rid] = type(
        "ReqState", (), {"num_tokens": len(prompt), "all_token_ids": list(prompt)})()
runner.input_batch.prev_req_id_to_index = {"req-0": 0, "req-1": 1, "req-2": 2}
runner.input_batch.prev_sampled_token_ids = torch.tensor(
    [[7], [8], [9]], dtype=torch.int32)  # 上拍三个请求各采出 7/8/9
runner._compute_prev_positions(3)
prev_positions = [int(v) for v in runner.prev_positions.np[:3]]
runner.input_ids.cpu[:3] = [100, 101, 102]  # CPU 侧可见部分（新请求 req-x 的 prompt 尾）
runner.input_ids.gpu[:3] = torch.tensor([100, 101, 102], dtype=torch.int32)
runner._prepare_input_ids(SO(), num_reqs=3, total_num_scheduled_tokens=3,
                          cu_num_tokens=np.array([1, 2, 3], dtype=np.int64))
trace["cases"].append({
    "case": "async 拍·批次重排（req-2 提前 + 新请求 req-x 插中间 + req-1 离场）",
    "prev_positions": prev_positions,
    "prev_positions_note": "req-2: 0→上拍槽2、req-x: -1（新请求无上拍）、req-0: 上拍槽0；req-1 已不在批",
    "prev_sampled": [7, 8, 9],
    "input_ids_cpu_side": [100, 101, 102],
    "input_ids_gpu_result": [int(v) for v in runner.input_ids.gpu[:3]],
    "path": "scatter 兜底：index=[flattened 采样位] ← src=prev[prev_indices,0]（L1878-L1891）",
    "proof": "位置0=9（从 prev 行 2 回填）、位置1=101（新请求保留 CPU 值）、位置2=7（从 prev 行 0 回填）",
})

trace["prev_positions_mapping_note"] = (
    "_compute_prev_positions（L1769-L1782）：当前批每个槽位查 prev_req_id_to_index——"
    "上拍在批的请求得到它的旧槽号、新请求得到 -1（scatter 循环里 -1 直接 continue）"
)

out = os.path.join(os.path.dirname(__file__), "m11_gpu_backfill.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
