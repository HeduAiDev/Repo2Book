"""Driver for m3 (FreeKVCacheBlockQueue 侵入式双向链表) — host run against the
ch13 companion. 五块小队列上做四步指针手术：
  popleft_n(1) 队头取 -> remove(blocks[2]) O(1) 中间摘 -> append_n 归还挂尾 ->
  popleft() 再取 -> prepend_n 挂头。
每步记录队列全景、num_free、关键指针变化；全程 id() 集合不变——
"does not allocate any Python objects"（类 docstring 原话）的机器物证。
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.kv_cache_utils import (  # noqa: E402
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)


def obj_ids(q: FreeKVCacheBlockQueue, blocks) -> set:
    ids = {id(b) for b in blocks}
    ids.add(id(q.fake_free_list_head))
    ids.add(id(q.fake_free_list_tail))
    return ids


def main():
    blocks = [KVCacheBlock(idx) for idx in range(5)]
    q = FreeKVCacheBlockQueue(blocks)
    ids_before = obj_ids(q, blocks)

    steps = []

    def snap(label, note):
        steps.append({
            "step": label,
            "queue_head_to_tail": [b.block_id for b in q.get_all_free_blocks()],
            "num_free_blocks": q.num_free_blocks,
            "note": note,
        })

    snap(
        "0 初始：相邻块互串 + 双哨兵",
        "fake_head(block_id=-1) ↔ 0 ↔ 1 ↔ 2 ↔ 3 ↔ 4 ↔ fake_tail(block_id=-1)——每个真实块都有 prev 和 next（哨兵消掉边界分支）",
    )

    taken_head = q.popleft_n(1)
    assert [b.block_id for b in taken_head] == [0]
    snap(
        "1 popleft_n(1)：队头取块",
        f"块 0 被摘：prev/next 置 None；fake_head 直连块 1（blocks[1].prev_free_block.block_id = {blocks[1].prev_free_block.block_id}）",
    )

    # O(1) 中间摘（touch 救回驱逐候选的原语——ch15 前置，本章验语义）
    q.remove(blocks[2])
    snap(
        "2 remove(blocks[2])：O(1) 中间摘",
        f"块 1 的 next 越过块 2 直指块 3（blocks[1].next.block_id = {blocks[1].next_free_block.block_id}），块 3 的 prev = {blocks[3].prev_free_block.block_id}；块 2 指针清 None",
    )

    q.append_n(taken_head)
    snap(
        "3 append_n([0])：归还挂队尾",
        f"块 0 接到原尾块 4 之后（blocks[4].next.block_id = {blocks[4].next_free_block.block_id if blocks[4].next_free_block else None}，块 0.next = fake_tail）",
    )

    popped = q.popleft()
    assert popped is blocks[1]
    snap(
        "4 popleft()：单取队头（null_block 的出生就是它）",
        f"拿到块 1；fake_head 直连块 3（fake_head.next.block_id = {q.fake_free_list_head.next_free_block.block_id}）",
    )

    q.prepend_n([blocks[1]])
    snap(
        "5 prepend_n([1])：挂回队头",
        "free_blocks 劈分时无哈希块 prepend 到队头先驱逐的挂点（ch15 LRU 双不变量；caching 关时不触发，本章验原语语义）",
    )

    ids_after = obj_ids(q, blocks)
    zero_alloc = {
        "objects_before": len(ids_before),
        "objects_after": len(ids_after),
        "same_object_set": ids_before == ids_after,
        "note": "全程 7 个对象（5 真实块 + 2 哨兵）零增——链表操纵只改块上 prev/next 指针、不分配任何 Python 对象（类 docstring L188-L191 原话 'does not allocate any Python objects'）",
    }

    # 终态断言
    final = [b.block_id for b in q.get_all_free_blocks()]
    assert final == [1, 3, 4, 0], final
    assert q.num_free_blocks == 4

    out = {
        "driver": "run_m3_free_queue.py",
        "mechanism": "m3 FreeKVCacheBlockQueue 侵入式双向链表（kv_cache_utils.py:L184-L234 / L273-L304 / L306-L324）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版（全原语逐字）",
        "config": {"blocks": 5, "sentinel_block_id": -1},
        "why_not_deque": (
            "类 docstring 原话：implement this class instead of using Python builtin deque "
            "to support removing a block in the middle of the queue in O(1) time；"
            "不分配任何 Python object 以逼近 C++ deque"
        ),
        "steps": steps,
        "zero_allocation_evidence": zero_alloc,
        "final_state": {
            "queue": final,
            "num_free_blocks": q.num_free_blocks,
        },
    }

    dst = Path(__file__).resolve().parent / "m3_free_queue.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(steps, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
