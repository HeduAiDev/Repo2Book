"""Driver for m12 (终局逆序 free："tail blocks are freed first") — host run
against the ch13 companion.

上半：裸池上分配 [1..5] → 逆序 free → 队尾驱逐序 [5,4,3,2,1] → 再分配
5+2 块观察「先耗新鲜块、再按尾块优先复用」。
下半：Scheduler._free_blocks 端到端（48-token 请求 3 块 → 完成 → 全回池、
销账、驱逐序挂队尾）。
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from implementation.block_pool import BlockPool  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from implementation.request import Request, RequestStatus  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402

BLOCK_SIZE = 16
LAYER = "model.layers.0.self_attn.attn"


def make_scheduler(num_blocks: int) -> Scheduler:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[LAYER])
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[LAYER], kv_cache_spec=spec)],
    )
    return Scheduler(
        kv_cache_config=config,
        max_model_len=256,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )


def make_request(req_id: str, n: int) -> Request:
    req = Request(request_id=req_id, prompt_token_ids=list(range(n)))
    req.status = RequestStatus.WAITING
    return req


def main():
    steps = []

    # 上半：裸池
    pool = BlockPool(num_gpu_blocks=11, enable_caching=False, hash_block_size=BLOCK_SIZE)
    allocated = pool.get_new_blocks(5)
    alloc_ids = [b.block_id for b in allocated]
    steps.append({
        "step": "1 分配 5 块",
        "queue": [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        "request_holds": alloc_ids,
        "num_free": pool.get_num_free_blocks(),
        "note": "ids 1..5（0 被 null_block 占）；新鲜块队头还剩 6..10",
    })

    pool.free_blocks(reversed(allocated))  # 终局逆序 free（single_type free 的做法）
    steps.append({
        "step": "2 终局 free（reversed 传入）",
        "queue": [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        "num_free": pool.get_num_free_blocks(),
        "note": "归还序 [5,4,3,2,1] append_n 挂队尾——尾块 5 最先处于被驱逐位（'tail blocks are freed first'，single_type:L519-L527 docstring 原话）",
    })

    first = pool.get_new_blocks(5)
    steps.append({
        "step": "3 再分配 5 块",
        "queue": [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        "got": [b.block_id for b in first],
        "num_free": pool.get_num_free_blocks(),
        "note": "拿到 [6,7,8,9,10]——先耗尽从未用过的新鲜块",
    })

    second = pool.get_new_blocks(2)
    steps.append({
        "step": "4 再分配 2 块",
        "queue": [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        "got": [b.block_id for b in second],
        "num_free": pool.get_num_free_blocks(),
        "note": "拿到 [5,4]——新鲜块耗尽后按归还序（尾块优先）复用——LRU 尾优先驱逐序的可观测语义",
    })

    # 下半：Scheduler 端到端
    sched = make_scheduler(num_blocks=5)
    single = sched.kv_cache_manager.coordinator.single_type_managers[0]
    req = make_request("req-0", 48)
    sched.requests["req-0"] = req
    blocks = sched.allocate_slots_for_waiting(req, 48, 0, None)
    e2e_alloc = blocks.get_block_ids()[0]
    e2e = [{
        "step": "5 入场（48-token prompt）",
        "request_holds": e2e_alloc,
        "num_free": sched.kv_cache_manager.block_pool.get_num_free_blocks(),
        "in_req_to_blocks": "req-0" in single.req_to_blocks,
        "note": "3 块 [1,2,3]；可用 4 块剩 1",
    }]

    req.status = RequestStatus.FINISHED_STOPPED
    assert req.is_finished()
    sched._free_blocks(req)
    e2e.append({
        "step": "6 _free_blocks（请求完成）",
        "queue_tail": [b.block_id for b in sched.kv_cache_manager.block_pool.free_block_queue.get_all_free_blocks()][-3:],
        "num_free": sched.kv_cache_manager.block_pool.get_num_free_blocks(),
        "in_req_to_blocks": "req-0" in single.req_to_blocks,
        "in_requests": "req-0" in sched.requests,
        "note": "manager.free → pop_blocks_for_free 摘账 → reversed 交给 block_pool.free_blocks（ref_cnt−1 归零回池）——块回池，驱逐序 [3,2,1]",
    })

    assert alloc_ids == [1, 2, 3, 4, 5]
    assert [b.block_id for b in second] == [5, 4]
    assert e2e_alloc == [1, 2, 3]
    assert e2e[1]["num_free"] == 4 and not e2e[1]["in_req_to_blocks"]

    out = {
        "driver": "run_m12_reverse_free.py",
        "mechanism": "m12 终局逆序 free（scheduler.py:L2329-L2354 / kv_cache_manager.py:L567-L578 / single_type:L519-L527 / block_pool.py:L719-L742）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版（caching 关：free_blocks 全走 append_n 保传入序；'tail first' 的驱逐优先级由 reversed 入队序体现）",
        "config": {"pool_a_blocks": 11, "pool_b_blocks": 5, "block_size": BLOCK_SIZE},
        "bare_pool_steps": steps,
        "scheduler_e2e": e2e,
        "docstring_quote": "Free blocks in reverse order so that the tail blocks are freed first（single_type_kv_cache_manager.py:L522-L525）",
        "why": "逆序是 ch15 LRU 不变量的半边：若按分配序还块，队头会变成前缀的头几块，驱逐从最长可复用前缀的腰部斩断（deepread why_chains[3]）；哈希保留 → ch15",
    }

    dst = Path(__file__).resolve().parent / "m12_reverse_free.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(steps, ensure_ascii=False, indent=1))
    print(json.dumps(e2e, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
