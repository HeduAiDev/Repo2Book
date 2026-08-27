"""Driver for m5 (需块预测 get_num_blocks_to_allocate：cdiv 主算术 + running
fast-path + 可驱逐命中块计数) — host run against the ch13 companion.

六问六答（预测器与分配器同构——每个预测值都跟一次真实分配对账）：
  a) 新请求 100 token -> 7        （cdiv 主算术）
  b) 新请求 33 token  -> 3        （非整除：33/16 向上取整）
  c) running 已持 7、目标 113 -> 1（fast-path 差值）
  d) running 已持 7、目标 112 -> 0（恰好对齐——块界上的免账）
  e) spec 拒绝：目标 64 < 已持 7 块 -> 0（max(需-有, 0) 钳零）
  f) 32 token 带 1 块可驱逐命中 -> 2（1 新块 + 1 可驱逐命中块也占容量）
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.request import Request, RequestStatus  # noqa: E402

BLOCK_SIZE = 16
LAYER = "model.layers.0.self_attn.attn"


def make_manager(num_blocks: int = 20) -> KVCacheManager:
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
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=512,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )


def make_request(req_id: str, n: int) -> Request:
    req = Request(request_id=req_id, prompt_token_ids=list(range(n)))
    req.status = RequestStatus.WAITING
    return req


def main():
    mgr = make_manager()
    single = mgr.coordinator.single_type_managers[0]
    cases = []

    def ask(case, req_id, num_tokens, held, path_note):
        n = single.get_num_blocks_to_allocate(
            request_id=req_id,
            num_tokens=num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=num_tokens,
            num_local_computed_tokens=num_tokens,
            num_tokens_main_model=num_tokens,
        )
        cases.append({
            "case": case,
            "num_tokens": num_tokens,
            "held_blocks": held,
            "predicted": n,
            "cdiv": -(-num_tokens // BLOCK_SIZE),
            "path": path_note,
        })
        return n

    # a/b：fresh 请求走 cdiv 主算术（L178）
    a = ask("a 新请求 100 token", "fresh-a", 100, 0, "cdiv 主算术（L178）：100/16 向上取整")
    b = ask("b 新请求 33 token（非整除）", "fresh-b", 33, 0, "cdiv 主算术：33/16 = 2 余 1 -> 3")

    # 对账 a：真实分配恰好 7 块（预测器 ≡ 分配器）
    ra = make_request("fresh-a", 100)
    got = mgr.allocate_slots(ra, 100).get_block_ids()[0]
    alloc_check_a = {"predicted": a, "actually_allocated": len(got),
                     "block_ids": got, "match": len(got) == a}

    # c/d/e：running fast-path（L194-L200）——num_cached_block 账位由 driver 手工
    # 登记以走 fast-path 支（真实部署 enable_prefix_caching 默认 True 时由
    # cache_blocks 自然登记；本章精简版 False 支恒走慢路径、数学同构）
    rc = make_request("run-c", 100)
    mgr.allocate_slots(rc, 100)
    rc.status = RequestStatus.RUNNING
    single.num_cached_block["run-c"] = 7

    def ask_fast(case, num_tokens):
        n = single.get_num_blocks_to_allocate(
            request_id="run-c",
            num_tokens=num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=num_tokens,
            num_local_computed_tokens=num_tokens,
            num_tokens_main_model=num_tokens,
        )
        cases.append({
            "case": case,
            "num_tokens": num_tokens,
            "held_blocks": 7,
            "predicted": n,
            "cdiv": -(-num_tokens // BLOCK_SIZE),
            "path": "running fast-path（L194-L200）：max(cdiv − 已持 7, 0)",
        })
        return n

    c = ask_fast("c running 已持 7 块、长到 113 token", 113)
    d = ask_fast("d running 已持 7 块、长到 112 token（恰对齐）", 112)
    e = ask_fast("e spec 拒绝草稿：目标回缩到 64 token", 64)

    # f：可驱逐命中块计数（L220-L225）——32 token、1 块 ref_cnt==0 的命中块
    (free_block,) = mgr.block_pool.get_new_blocks(1)  # 块 1
    mgr.block_pool.free_blocks([free_block])  # 回自由队列 -> 驱逐候选
    f = single.get_num_blocks_to_allocate(
        request_id="fresh-f",
        num_tokens=32,
        new_computed_blocks=[free_block],
        total_computed_tokens=16,
        num_local_computed_tokens=16,
        num_tokens_main_model=32,
    )
    cases.append({
        "case": "f 32 token 带 1 块可驱逐命中",
        "num_tokens": 32,
        "held_blocks": 0,
        "hit_blocks": 1,
        "hit_ref_cnt": free_block.ref_cnt,
        "predicted": f,
        "cdiv": 2,
        "path": "慢路径：num_new = max(cdiv 2 − max(0, 命中 1), 0) = 1；可驱逐命中块 +1 -> 2（touch 时要占容量，L220-L225）",
    })

    # 对账 c：真实差值分配恰好 1 块
    rc.num_computed_tokens = 100
    got_c = mgr.allocate_slots(rc, 13).get_block_ids()[0]
    alloc_check_c = {
        "note": "长到 113 token（100 已算 + 13 新）",
        "predicted": c,
        "actually_allocated": len(got_c),
        "block_ids": got_c,
        "match": len(got_c) == c,
    }

    assert (a, b, c, d, e, f) == (7, 3, 1, 0, 0, 2)
    assert alloc_check_a["match"] and alloc_check_c["match"]

    out = {
        "driver": "run_m5_need_prediction.py",
        "mechanism": "m5 需块预测 get_num_blocks_to_allocate（single_type_kv_cache_manager.py:L144-L230）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版",
        "config": {"block_size": BLOCK_SIZE, "pool_blocks": 20},
        "cases": cases,
        "predictor_matches_allocator": [alloc_check_a, alloc_check_c],
        "usage": "预测值 > 空闲 → allocate_slots return None（kv_cache_manager.py:L510-L527）——ch11 抢占唯一触发信号的内因",
        "fastpath_setup_note": "c/d/e 的 num_cached_block 账位由 driver 手工登记（真实开缓存部署由 cache_blocks 自然推进；精简版 False 支走慢路径，两者数学同构：max(cdiv − max(0, 已持), 0)）",
    }

    dst = Path(__file__).resolve().parent / "m5_need_prediction.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(cases, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
