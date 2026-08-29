# ch16 m4 异步加载路径：allocate_slots(ext=32, delay_cache_blocks=True)——
# 『已分配未缓存』窗口：块已挂表但 num_cached_block=0、block_hash=None；
# 请求置 WAITING_FOR_REMOTE_KVS、num_computed_tokens 先行设置但零前向
# （scheduler.py:L1023-L1053 / kv_cache_manager.py:L529-L565 的 delay 早退）。
# 场景：64-token prompt、外部可加载 32（异步）。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import dump, make_request, script_scheduler  # noqa: E402

s = script_scheduler([(32, True)])
req = make_request("r1", range(64))
s.add_request(req)
out = s.schedule()

mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
ids = s.kv_cache_manager.get_block_ids("r1")[0]

trace = {
    "mechanism": "m4 异步加载『已分配未缓存』窗口",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "params": {
        "block_size": 16,
        "prompt_tokens": 64,
        "external_tokens": 32,
        "load_kv_async": True,
        "场景": "connector 答 (32, True)：外部 32 token 异步加载",
    },
    "layout_five_segments": {
        "comp": 0,
        "new_comp": 0,
        "ext_comp": 32,     # not cached by vLLM, but cached by connector
        "new": 0,           # async 期不算新 token（scheduler.py:L866-L869）
        "lookahead": 0,
    },
    "observations": {
        "r1_status": req.status.name,                  # WAITING_FOR_REMOTE_KVS
        "r1_in_skipped_waiting": req in list(s.skipped_waiting),
        "num_computed_tokens_set_ahead": req.num_computed_tokens,   # 32 先行
        "scheduled_tokens_this_step": 0,               # 零前向
        "blocks_on_table": len(ids),                   # 2 块 ext
        "block_ids": ids,
        "num_cached_block": mgr.num_cached_block.get("r1", 0),      # 0 —— 账上没缓存
        "first_block_hash_is_none": mgr.req_to_blocks["r1"][0].block_hash is None,
        "free_blocks": s.kv_cache_manager.block_pool.get_num_free_blocks(),
        "pool_free_baseline": s.kv_cache_manager.block_pool.num_gpu_blocks - 1,
    },
    "窗口含义": "块表挂 2 块（账实分离：块被本请求占用），哈希表/缓存账 0——"
               "传输完成前这些块对外不可命中；失败时按第一个坏块截断重算。",
}
print(dump("m4", trace))
