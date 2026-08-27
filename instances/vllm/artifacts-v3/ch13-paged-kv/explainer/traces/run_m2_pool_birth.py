"""Driver for m2 (KVCacheBlock 七字段元数据与池的构造) — host run against the
ch13 companion. Figure evidence for the "块的身份证 + 池的出生" 图：
blocks 数组一次预构、null_block 从队头 popleft 占 block_id=0（is_null=True、
ref_cnt 不维护）、自由队列整串互串、get_usage 分母减 null、块 id 从 1 起分配。
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.block_pool import BlockPool  # noqa: E402

BLOCK_SIZE = 16
N = 6


def main():
    pool = BlockPool(
        num_gpu_blocks=N, enable_caching=False, hash_block_size=BLOCK_SIZE
    )
    queue = [
        b.block_id for b in pool.free_block_queue.get_all_free_blocks()
    ]

    # 块 1 的七字段快照（身份证正面照）
    b1 = pool.blocks[1]
    field_snapshot = {
        "block_id": b1.block_id,
        "ref_cnt": b1.ref_cnt,
        "_block_hash": b1.block_hash,
        "_block_hash_num_tokens": b1.block_hash_num_tokens,
        "prev_free_block_id": (
            b1.prev_free_block.block_id if b1.prev_free_block else None
        ),
        "next_free_block_id": (
            b1.next_free_block.block_id if b1.next_free_block else None
        ),
        "is_null": b1.is_null,
        "note": "prev 指向队头哨兵（block_id=-1）——块 1 是队列第一个真实块；两个哈希字段是 ch15 的缓存账位（本章恒空）",
    }

    out = {
        "driver": "run_m2_pool_birth.py",
        "mechanism": "m2 KVCacheBlock 七字段与池的构造（kv_cache_utils.py:L118-L138 / block_pool.py:L162-L191）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版",
        "config": {"num_gpu_blocks": N, "block_size": BLOCK_SIZE},
        "pool_birth": {
            "blocks_array_len": len(pool.blocks),
            "blocks_ids": [b.block_id for b in pool.blocks],
            "preallocated_once": "KVCacheBlock(idx) for idx in range(num_gpu_blocks)（L175-L177 对象数组一次预构）",
            "null_block": {
                "is_blocks_0": pool.null_block is pool.blocks[0],
                "block_id": pool.null_block.block_id,
                "is_null": pool.null_block.is_null,
                "ref_cnt": pool.null_block.ref_cnt,
                "source_comment": "The ref_cnt of null_block is not maintained, needs special care to avoid freeing it（block_pool.py:L187-L189 注释原话）",
                "born_by": "free_block_queue.popleft()（L190——从队头摘走 0 号）",
            },
            "free_queue_initial_order": queue,
            "num_free_after_birth": pool.get_num_free_blocks(),
            "first_allocatable_block_id": 1,
            "sentinels": {
                "fake_head_block_id": pool.free_block_queue.fake_free_list_head.block_id,
                "fake_tail_block_id": pool.free_block_queue.fake_free_list_tail.block_id,
                "fake_head_next": pool.free_block_queue.fake_free_list_head.next_free_block.block_id,
                "block1_prev": (
                    pool.blocks[1].prev_free_block.block_id
                ),
            },
        },
        "block1_seven_fields": field_snapshot,
        "usage_ledger": [],
    }

    # get_usage：分母 = num_gpu_blocks - 1（减 null 块）——vLLM 运行日志
    # "GPU KV cache usage" 的出处
    out["usage_ledger"].append(
        {"after": "birth", "held": 0, "free": pool.get_num_free_blocks(),
         "usage": pool.get_usage()}
    )
    got = pool.get_new_blocks(2)
    out["usage_ledger"].append(
        {"after": "alloc 2", "held_block_ids": [b.block_id for b in got],
         "ref_cnt_all": [b.ref_cnt for b in got], "free": pool.get_num_free_blocks(),
         "usage": round(pool.get_usage(), 2),
         "arithmetic": "1 - 3/5 = 0.4（分母 5 = 6 - 1，减 null 块）"}
    )

    assert pool.get_num_free_blocks() == 3
    assert round(pool.get_usage(), 2) == 0.4

    dst = Path(__file__).resolve().parent / "m2_pool_birth.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(out["pool_birth"], ensure_ascii=False))
    print(json.dumps(out["usage_ledger"], ensure_ascii=False))


if __name__ == "__main__":
    main()
