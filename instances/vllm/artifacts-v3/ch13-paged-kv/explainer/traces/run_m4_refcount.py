"""Driver for m4 (引用计数生命周期：get_new_blocks +1 / free −1 归零入队 /
touch +1 出队) — host run against the ch13 companion.

单块的一生（pool 6 块、可用 5）：r1 首配 +1 → r2 命中同前缀 touch 再 +1
（ref_cnt=2，共享）→ r1 结束 −1（=1，不回池——r2 还在用）→ r2 结束 −1
（=0，归零回池挂队尾）→ touch 救回（0→1 且出队）→ 再 free（归零回池）→
null_block 特判（is_null 永不回队）。
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
    pool = BlockPool(num_gpu_blocks=N, enable_caching=False, hash_block_size=BLOCK_SIZE)
    log = []

    def rec(action, block_id, before, after, free, extra=""):
        log.append({
            "action": action,
            "block": block_id,
            "ref_cnt": f"{before} -> {after}",
            "ref_before": before,
            "ref_after": after,
            "free_blocks": free,
            "returned_to_pool": extra or "",
        })

    (b,) = pool.get_new_blocks(1)
    rec("r1 首次分配 get_new_blocks(1)", b.block_id, 0, 1, pool.get_num_free_blocks(),
        "否（新主人登记）")

    pool.touch([b])  # r2 命中同一前缀（ch15 场景预演：ref_cnt≠0 不出队）
    rec("r2 命中同前缀 touch([b])", b.block_id, 1, 2, pool.get_num_free_blocks(),
        "否（ref_cnt≠0 无需出队——块本就不在自由队列）")

    pool.free_blocks([b])  # r1 结束
    rec("r1 结束 free_blocks([b])", b.block_id, 2, 1, pool.get_num_free_blocks(),
        "否（ref_cnt=1 未归零——r2 还在用，这正是共享的物理意义）")

    pool.free_blocks([b])  # r2 结束
    queue = [x.block_id for x in pool.free_block_queue.get_all_free_blocks()]
    rec("r2 结束 free_blocks([b])", b.block_id, 1, 0, pool.get_num_free_blocks(),
        f"是（归零且非 null → append_n 挂队尾；队尾段 {queue[-1:]}）")

    pool.touch([b])  # 驱逐候选被救回
    rec("touch([b]) 救回驱逐候选", b.block_id, 0, 1, pool.get_num_free_blocks(),
        "出队（ref_cnt==0 且非 null → free_block_queue.remove，空闲 5→4）")

    pool.free_blocks([b])
    rec("再 free_blocks([b])", b.block_id, 1, 0, pool.get_num_free_blocks(),
        "是（归零回池）")

    free_before_null = pool.get_num_free_blocks()
    pool.free_blocks([pool.null_block])
    rec("free_blocks([null_block]) 占位块特判", 0, "不维护", "不维护",
        pool.get_num_free_blocks(),
        f"否（is_null=True 分支挡住——空闲 {free_before_null} 不变；ref_cnt 本就不维护）")

    # 校验
    assert [r["ref_before"] for r in log[:5]] == [0, 1, 2, 1, 0]
    assert [r["ref_after"] for r in log[:5]] == [1, 2, 1, 0, 1]
    assert log[0]["free_blocks"] == 4
    assert log[1]["free_blocks"] == 4  # touch 不减空闲（ref≠0）
    assert log[2]["free_blocks"] == 4
    assert log[3]["free_blocks"] == 5
    assert log[4]["free_blocks"] == 4
    assert log[5]["free_blocks"] == 5
    assert log[6]["free_blocks"] == 5

    out = {
        "driver": "run_m4_refcount.py",
        "mechanism": "m4 引用计数生命周期（block_pool.py:L647-L677 / L702-L742）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版",
        "config": {"num_gpu_blocks": N, "usable": N - 1, "block_size": BLOCK_SIZE},
        "life_of_one_block": log,
        "summary": {
            "max_ref_cnt_seen": 2,
            "owners_at_ref2": ["r1（首配）", "r2（touch 命中）"],
            "free_rule": "ref_cnt−1 后 == 0 且非 null 才回自由队列（L730-L742）",
            "touch_rule": "+1；ref_cnt==0 且非 null 先 remove 出队（L710-L715）——ch15 前缀命中救回驱逐候选",
            "null_rule": "is_null 特判永不回队；ref_cnt 不维护（L187-L189 注释）",
        },
    }

    dst = Path(__file__).resolve().parent / "m4_refcount.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
