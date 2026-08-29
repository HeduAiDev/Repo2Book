# ch16 m11 所有权转移·终局接管：request_finished→True = connector 接管块的
# 异步释放——块不释放、请求留在 self.requests（has_finished_requests 知道
# 还有账），直到 get_finished 报 finished_sending 才 _free_blocks
# （scheduler.py:L2300-L2327 + L2577-L2612 + L2714-L2741）。
# 场景：64-token 请求（异步 ext 32 → 2 块 ext + 后续 2 块）以 producer 收尾。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (  # noqa: E402
    dump,
    free_count,
    full_free,
    make_request,
    script_scheduler,
    worker_output,
)
from implementation.request import RequestStatus  # noqa: E402

steps = []
s = script_scheduler([(32, True)])
s.connector.finish_answer = True                 # request_finished 返回 True=接管
req = make_request("r1", range(64))
s.add_request(req)
out = s.schedule()
ids = s.kv_cache_manager.get_block_ids("r1")[0]
steps.append({
    "step": 1,
    "event": "异步准入（ext 32）",
    "blocks_on_table": len(ids),                 # 2
    "free_blocks": free_count(s),                # 63−2=61
})

req.status = RequestStatus.FINISHED_STOPPED      # 请求完成（生产者送完 64 token）
used_before = full_free(s) - free_count(s)
s._free_request(req)
free_after_free_req = free_count(s)
finished_ev = [
    e for e in s.connector.events if e[0] == "request_finished"
][-1]
steps.append({
    "step": 2,
    "event": "请求完成 → _free_request 问 connector",
    "request_finished_answer": True,             # 接管
    "handoff_block_table_len": len(finished_ev[2]),   # 整块表交接
    "blocks_still_held": full_free(s) - free_after_free_req == used_before,  # 4 块没放
    "free_blocks_unchanged": free_after_free_req, # 61：接管=不释放
    "r1_still_in_requests": "r1" in s.requests,  # 留账
    "has_finished_requests": s.has_finished_requests(),   # 引擎知道还有账
})

# worker 报 finished_sending → _update_from_kv_xfer_finished → _free_blocks
s.update_from_output(out, worker_output(finished_sending={"r1"}))
steps.append({
    "step": 3,
    "event": "worker get_finished 报 finished_sending",
    "r1_removed_from_requests": "r1" not in s.requests,
    "free_blocks": free_count(s),                # 63：块全部归还
    "pool_free_baseline": full_free(s),
})

# 对照：SupportsHMA 逐组交接（混合模型的必经门）
s2 = script_scheduler([(32, True)], hma=True)
s2.connector.finish_answer = True
req2 = make_request("r1", range(64))
s2.add_request(req2)
s2.schedule()
req2.status = RequestStatus.FINISHED_STOPPED
s2._free_request(req2)
hma_ev = [
    e for e in s2.connector.events
    if e[0] == "request_finished_all_groups"
][-1]

trace = {
    "mechanism": "m11 终局接管（request_finished→True）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {
        "block_size": 16,
        "prompt_tokens": 64,
        "kv_role": "kv_consumer（finish_answer 编程为 True 模拟 producer 接管）",
        "pool_blocks": 64,
        "pool_free_baseline": full_free(s),
    },
    "steps": steps,
    "hma_variant": {
        "called": "request_finished_all_groups",
        "per_group_tables": [list(g) for g in hma_ev[2]],  # 逐组块表
        "num_groups": len(hma_ev[2]),
    },
    "语义": "『已交接未送达』挂起态：账本说请求完了、物理块却归 connector 管——"
            "get_finished 的 finished_sending 是唯一的放行票据。",
}
print(dump("m11", trace))
