"""Driver for m7 (block_id 跨进程契约：新请求全量块表 / 在跑请求增量
new_block_ids / worker 镜像) — host run against the ch13 companion.
同时采 m8（清零账）、m14（读写两条腿的读腿出口）、m15（commit 先行）的证据。

HOST SEAM：GPUModelRunner 在 CPU 设备跑（kv_caches/块表/清零全走 CPU 镜像）；
CpuGpuBuffer 的 .gpu 与 .cpu 同为 CPU 张量、commit 是真拷贝——双镜像契约逐字成立。

Part A（单组，页表语义精确）：三拍过线
  拍1 新请求 r1 33 token：全量块表 ([1,2,3],) + worker 建档 add_row
  拍1.5 在跑请求无新块 → get_block_ids(allow_none=True) = None（不占带宽）
  拍2 r1 长到 49：增量 [4] → block_ids.extend + append_row 差量追加
  拍3 r2 16 token 入场：全量 ([5],)
  附：commit 只拷活跃行；_get_block_table 的 pad 行填 NULL_BLOCK_ID=0
Part B（混合精度两组，needs_kv_cache_zeroing=True）：清零账活体
  陈旧字节 7 铺满 → 拍1 清 [1..6]（两层）→ 块 0（null）不动 → 排干语义
Part C（单组）：抢占-恢复的整表替换（resumed 语义）
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from implementation.output import (  # noqa: E402
    CachedRequestData,
    SchedulerOutput,
)
from implementation.request import Request, RequestStatus  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402

BLOCK_SIZE = 16
LAYER = "model.layers.0.self_attn.attn"
LAYER_B = "model.layers.1.self_attn.attn"


def make_spec(dtype=torch.float16, **kw):
    return FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128, dtype=dtype
    )


def make_config(num_blocks: int, num_groups: int = 1) -> KVCacheConfig:
    spec = make_spec()
    if num_groups == 2:
        spec_b = make_spec(dtype=torch.float32)
        groups = [
            KVCacheGroupSpec(layer_names=[LAYER], kv_cache_spec=spec),
            KVCacheGroupSpec(layer_names=[LAYER_B], kv_cache_spec=spec_b),
        ]
        tensors = [
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[LAYER]),
            KVCacheTensor(size=num_blocks * spec_b.page_size_bytes, shared_by=[LAYER_B]),
        ]
    else:
        groups = [KVCacheGroupSpec(layer_names=[LAYER], kv_cache_spec=spec)]
        tensors = [
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[LAYER])
        ]
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups
    )


def make_scheduler(num_blocks: int, num_groups: int = 1) -> Scheduler:
    return Scheduler(
        kv_cache_config=make_config(num_blocks, num_groups),
        max_model_len=256,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )


def make_runner(num_blocks: int, num_groups: int = 1) -> GPUModelRunner:
    config = make_config(num_blocks, num_groups)
    return GPUModelRunner(
        kv_cache_config=config,
        block_size=BLOCK_SIZE,
        max_num_reqs=4,
        max_blocks_per_req=8,
        max_num_batched_tokens=128,
        device=torch.device("cpu"),  # HOST SEAM
    )


def make_request(req_id: str, n: int) -> Request:
    req = Request(request_id=req_id, prompt_token_ids=list(range(n)))
    req.status = RequestStatus.WAITING
    return req


def cached_data(req_ids, new_block_ids, computed, resumed=(), outputs=None):
    return CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(resumed),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=list(new_block_ids),
        num_computed_tokens=list(computed),
        num_output_tokens=outputs if outputs is not None else [0] * len(list(req_ids)),
    )


def sched_output(new_reqs, cached, nst, zero):
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=nst,
        total_num_scheduled_tokens=sum(nst.values()),
        finished_req_ids=set(),
        new_block_ids_to_zero=zero,
    )


def main():
    out = {
        "driver": "run_m7_wire_contract.py",
        "mechanism": "m7 block_id 跨进程契约（scheduler.py:L1144-L1149 / L1451-L1453 / gpu_model_runner.py:L1442-L1474 / block_table.py:L138-L154）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版（HOST SEAM：runner 在 CPU 设备，双镜像/清零走 CPU 镜像——契约逐字成立）",
        "environment_note": "host CPU 取证：CpuGpuBuffer 的 .gpu 与 .cpu 同为 CPU 张量、commit 是真拷贝；KVBlockZeroer 走 ctypes.memset 的 CPU 分支（同一张绝对地址表）；CUDA 分支容器内真跑，数值无差",
        "config": {"block_size": BLOCK_SIZE, "null_block_id": 0, "pad_slot_id": -1},
        "partA_single_group": [],
        "partB_zeroing": {},
        "partC_preempt_resume": {},
    }

    # ---------------- Part A：单组三拍 ----------------
    NUM = 10
    sched = make_scheduler(NUM)
    runner = make_runner(NUM)
    partA = out["partA_single_group"]

    r1 = make_request("r1", 33)
    b1 = sched.allocate_slots_for_waiting(r1, 33, 0, None)
    full1 = b1.get_block_ids()
    zero1 = sched._get_new_block_ids_to_zero()  # uniform 单组 → None
    new_data1 = sched.make_new_reqs_data([r1], {r1.request_id: b1})
    # 过线快照：真实两进程经 IPC 序列化（天然拷贝）；单进程 driver 手工快照，
    # 防 worker 侧 extend 就地改动共享 list 污染记录
    wire1_snapshot = [list(x) for x in new_data1[0].block_ids]
    partA.append({
        "beat": "1 新请求 r1（33 token）",
        "wire_new_reqs_block_ids": wire1_snapshot,
        "wire_payload": "全量块表 ([1,2,3],) + prompt_token_ids 33 个 + num_computed_tokens——首帧整箱装备",
        "worker_mirror_after": None,
        "page_table_row": None,
        "new_block_ids_to_zero": zero1,
        "zero_note": "uniform 单组 needs_kv_cache_zeroing=False → None（清零通道关——Part B 开它）",
    })
    runner._update_states(sched_output(new_data1, cached_data([], [], []), {"r1": 33}, zero1))
    row1 = runner.input_batch.req_id_to_index["r1"]
    partA[-1]["worker_mirror_after"] = [list(x) for x in runner.requests["r1"].block_ids]
    partA[-1]["page_table_row"] = [int(x) for x in runner.input_batch.block_table.block_table.np[row1][:3]]

    # 拍1.5：在跑请求无新块 → allow_none=True → None（不占带宽）
    r1.status = RequestStatus.RUNNING
    r1.num_computed_tokens = 33
    empty = sched.kv_cache_manager.empty_kv_cache_blocks
    cd = sched._make_cached_request_data([r1], [], {r1.request_id: 33}, {r1.request_id: empty})
    partA.append({
        "beat": "1.5 在跑请求本步无新块",
        "wire_cached_new_block_ids": str(cd.new_block_ids),
        "allow_none_semantics": "get_block_ids(allow_none=True)：全组空 → None（kv_cache_manager.py:L89-L91）——空增量不占带宽",
    })

    # 拍2：r1 长到 49 → 增量 [4]
    b2 = sched.allocate_slots_for_running(r1, 16)
    r1.num_computed_tokens = 49
    cd2 = sched._make_cached_request_data([r1], [], {r1.request_id: 16}, {r1.request_id: b2})
    partA.append({
        "beat": "2 r1 长到 49 token（+16）",
        "wire_cached_new_block_ids": [str(x) for x in cd2.new_block_ids],
        "wire_payload": "增量 ([4],)——只发本步新块",
    })
    runner._update_states(sched_output([], cd2, {"r1": 16}, None))
    partA[-1]["worker_mirror_after"] = [list(x) for x in runner.requests["r1"].block_ids]
    partA[-1]["page_table_row"] = [int(x) for x in
        runner.input_batch.block_table.block_table.np[row1][:4]]
    partA[-1]["num_blocks_per_row"] = int(runner.input_batch.block_table.num_blocks_per_row[row1])
    partA[-1]["mirror_note"] = "差量：block_ids[0].extend([4]) + block_table.append_row([4], row0)——行内偏移由 num_blocks_per_row 记账"

    # 拍3：r2 16 token 入场
    r2 = make_request("r2", 16)
    b3 = sched.allocate_slots_for_waiting(r2, 16, 0, None)
    cd3_running = sched._make_cached_request_data(
        [r1], [], {r1.request_id: 1}, {r1.request_id: empty}
    )
    new_data2 = sched.make_new_reqs_data([r2], {r2.request_id: b3})
    wire2_snapshot = [list(x) for x in new_data2[0].block_ids]
    partA.append({
        "beat": "3 新请求 r2（16 token）",
        "wire_new_reqs_block_ids": wire2_snapshot,
        "same_frame_cached_new_block_ids": [str(x) for x in cd3_running.new_block_ids],
        "note": "同一帧里 r2 全量 ([5],)、r1 增量 None——两种包裹同帧过江",
    })
    runner._update_states(sched_output(new_data2, cd3_running, {"r1": 1, "r2": 16}, None))
    row2 = runner.input_batch.req_id_to_index["r2"]
    partA[-1]["page_table_row_r2"] = [int(x) for x in runner.input_batch.block_table.block_table.np[row2][:1]]
    partA[-1]["page_table_row_r1"] = [int(x) for x in runner.input_batch.block_table.block_table.np[row1][:4]]

    # m15 证据：commit 只拷活跃行
    bt = runner.input_batch.block_table
    bt.block_table.cpu[3, 0] = 9  # 行 3 本拍不活跃（CPU 侧写脏）
    bt.commit_block_table(2)
    m15 = {
        "scene": "2 个活跃请求（行 0/1），行 3 被 CPU 侧写脏但不活跃",
        "commit_active_rows": 2,
        "gpu_row0_after": [int(x) for x in bt.block_table.gpu[0][:4]],
        "gpu_row3_after": int(bt.block_table.gpu[3][0].item()),
        "cpu_row3": int(bt.block_table.cpu[3][0].item()),
        "first_line_anchor": "_prepare_inputs 第一句就是 commit_block_table（gpu_model_runner.py:L1977-L1979，注释原话 OPTIMIZATION: Start copying the block table first … overlap the copy with the following CPU operations）",
        "note": "GPU 镜像行 3 仍是 0（没被拷）；活跃行 0 已同步 [1,2,3,4]——每拍只拷活跃行（block_table.py:L213-L214）",
    }
    out["m15_commit_evidence"] = m15

    # m14 证据：读腿出口 _get_block_table 的 pad 行
    padded = runner._get_block_table(num_reqs=2, num_reqs_padded=4)
    out["m14_read_leg_evidence"] = {
        "write_leg": "写 KV 走 slot_mapping（每 token 一个物理槽位，直寻址——数字见 traces/m9_slot_identity.json）",
        "read_leg": "读历史 KV 走块表张量：get_device_tensor(num_reqs_padded) 交 attention metadata builder（gpu_model_runner.py:L2325-L2341）",
        "padded_tensor_shape": list(padded.shape),
        "pad_rows_filled_with": 0,
        "pad_rows_detail": f"行 [2:4] 全为 {int(padded[2][0].item())} = NULL_BLOCK_ID——块 0 是 null 块，读它永远安全（CUDA graph padding 的语义）",
        "f7": "写=直寻址、读=穿块表间接寻址——间接寻址的代价与 kernel 内景 → ch22（F7 伏笔埋点）",
    }

    # ---------------- Part B：混合精度两组 → 清零通道活体 ----------------
    sched2 = make_scheduler(NUM, num_groups=2)
    runner2 = make_runner(NUM, num_groups=2)
    rb = make_request("rb", 33)
    bb = sched2.allocate_slots_for_waiting(rb, 33, 0, None)
    ids2 = bb.get_block_ids()
    zb = sched2._get_new_block_ids_to_zero()
    # 铺陈旧字节（上一任主人留下的）
    for name, kv in runner2.kv_caches.items():
        kv.view(torch.int32).fill_(7)
    new_datab = sched2.make_new_reqs_data([rb], {rb.request_id: bb})
    wireb_snapshot = [list(x) for x in new_datab[0].block_ids]
    runner2._update_states(sched_output(new_datab, cached_data([], [], []), {"rb": 33}, zb))
    spec0 = runner2.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    spec1 = runner2.kv_cache_config.kv_cache_groups[1].kv_cache_spec
    el0 = spec0.page_size_bytes // 4
    el1 = spec1.page_size_bytes // 4

    def block_nonzero(runner, name, block_id, el):
        flat = runner.kv_caches[name].view(torch.int32)
        return int(flat[block_id * el:(block_id + 1) * el].abs().sum())

    partB = {
        "config": "两组混合精度（fp16 + fp32）→ needs_kv_cache_zeroing=True（kv_cache_interface.py:L1013-L1022）",
        "wire_new_reqs_block_ids": wireb_snapshot,
        "new_block_ids_to_zero_beat1": zb,
        "stale_bytes_before": 7,
        "stale_note": "块从自由队列回收，上一任主人留下的字节还躺在显存里（注释原话 to prevent stale NaN/data from corrupting attention or SSM computation，gpu_model_runner.py:L1219-L1222）",
        "after_zero": {
            "layer0_block1_nonzero": block_nonzero(runner2, LAYER, 1, el0),
            "layer0_block6_nonzero": block_nonzero(runner2, LAYER, 6, el0),
            "layer1_block1_nonzero": block_nonzero(runner2, LAYER_B, 1, el1),
            "layer1_block6_nonzero": block_nonzero(runner2, LAYER_B, 6, el1),
            "layer0_block0_nonzero_stale_kept": block_nonzero(runner2, LAYER, 0, el0),
            "layer0_page_el": el0,
            "layer1_page_el": el1,
        },
        "drain_semantics": {
            "second_call": sched2._get_new_block_ids_to_zero(),
            "note": "take_new_block_ids 每步排干（注释 does not grow unbounded，scheduler.py:L1261-L1265）；排干后为空 → None（L1272 `or None`）",
        },
    }
    out["partB_zeroing"] = partB

    # ---------------- Part C：抢占-恢复整表替换 ----------------
    NUM3 = 10
    sched3 = make_scheduler(NUM3)
    runner3 = make_runner(NUM3)
    rc = make_request("rc", 16)
    bc = sched3.allocate_slots_for_waiting(rc, 16, 0, None)
    rc.status = RequestStatus.RUNNING
    sched3.running.append(rc)
    rc.num_computed_tokens = 16
    rc.append_output_token_ids(7)  # 生成 1 个输出 token：终长 17
    nd = sched3.make_new_reqs_data([rc], {rc.request_id: bc})
    runner3._update_states(sched_output(nd, cached_data([], [], []), {"rc": 16}, None))
    row_c = runner3.input_batch.req_id_to_index["rc"]
    before_preempt = {
        "rc_block_ids": [list(x) for x in runner3.requests["rc"].block_ids],
        "page_table_row": [int(x) for x in runner3.input_batch.block_table.block_table.np[row_c][:1]],
    }
    # 抢占（ch11 外部行为；块侧两件事：free 全部块 + computed 归零）
    sched3._preempt_request(rc)
    # worker 侧摘批（真实由 preempted_req_ids 通知——本章 facet 手工模拟；
    # CachedRequestState 留在 requests 里，恢复时复用——真实语义如此）
    runner3.input_batch.remove_request("rc")
    # 恢复：重算整段（17 token → 2 块）
    rc.status = RequestStatus.PREEMPTED
    br = sched3.allocate_slots_for_waiting(rc, 17, 0, None)
    resumed_ids = br.get_block_ids()
    cdr = sched3._make_cached_request_data(
        [], [rc], {rc.request_id: 17}, {rc.request_id: br}
    )
    runner3._update_states(sched_output([], cdr, {"rc": 17}, None))
    partC = {
        "before_preempt": before_preempt,
        "preempt_freed": [1],
        "preempt_note": "rc 原持 [1]（1 块）；逆序归还挂队尾，恢复重算 17 token 拿到新鲜块 [2,3]",
        "resumed_wire": {
            "resumed_req_ids": list(cdr.resumed_req_ids),
            "new_block_ids": [str(x) for x in cdr.new_block_ids],
            "note": "resumed 请求的 new_block_ids 是整体替换而非追加（output.py:L117-L121 注释语义）",
        },
        "worker_after": {
            "rc_block_ids": [list(x) for x in runner3.requests["rc"].block_ids],
            "req_index_was_none": True,
            "note": "assert req_index is None + block_ids = new_block_ids 整表替换（gpu_model_runner.py:L1447-L1452）——恢复者的块全换新",
        },
    }
    out["partC_preempt_resume"] = partC

    # 校验
    assert full1 == ([1, 2, 3],)
    assert cd.new_block_ids == [None]
    assert cd2.new_block_ids == [([4],)]
    assert new_data2[0].block_ids == ([5],)
    assert m15["gpu_row3_after"] == 0 and m15["gpu_row0_after"] == [1, 2, 3, 4]
    assert partA[2]["worker_mirror_after"] == [[1, 2, 3, 4]]
    assert partA[3]["page_table_row_r1"] == [1, 2, 3, 4]
    assert padded[2][0].item() == 0 and padded[3][0].item() == 0
    assert ids2 == ([1, 2, 3], [4, 5, 6]) and zb == [1, 2, 3, 4, 5, 6]
    assert partB["after_zero"]["layer0_block1_nonzero"] == 0
    assert partB["after_zero"]["layer1_block6_nonzero"] == 0
    assert partB["after_zero"]["layer0_block0_nonzero_stale_kept"] > 0
    assert partB["drain_semantics"]["second_call"] is None
    assert before_preempt["rc_block_ids"] == [[1]]
    assert resumed_ids == ([2, 3],)
    assert partC["worker_after"]["rc_block_ids"] == [[2, 3]]

    dst = Path(__file__).resolve().parent / "m7_wire_contract.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(out["partA_single_group"], ensure_ascii=False, indent=1))
    print(json.dumps(out["m15_commit_evidence"], ensure_ascii=False))
    print(json.dumps(out["m14_read_leg_evidence"], ensure_ascii=False))
    print(json.dumps(out["partB_zeroing"], ensure_ascii=False, indent=1))
    print(json.dumps(out["partC_preempt_resume"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
