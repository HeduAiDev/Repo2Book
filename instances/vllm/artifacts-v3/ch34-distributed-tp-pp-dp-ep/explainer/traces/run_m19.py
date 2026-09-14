# m19 取证：wave 共识。
#   - sync_truth：2 元素 SUM 双共识真值表（含『部分 rank 在等共识→仍算有活』
#     的 docstring 边界）+ MAX≡OR。
#   - wave_lockstep：DPEngineCoreProc 忙循环真跑 40 拍——dummy batch 维持锁步、
#     第 32 拍 2 元素 SUM AR、dp_rank0 上报 wave_complete(-1 哨兵)、current_wave+1。
from __future__ import annotations

from _driver_common import clean_tmp, dump


def main() -> None:
    from _trace_jobs import _free_port
    from _trace_pool import TracePool

    doc = {
        "mechanisms": ["m19"],
        "params": "2 引擎 stateless gloo dp_group；忙循环预算 40 拍",
        "combos": [
            [[True, False], [False, False]],
            [[False, False], [False, False]],
            [[False, False], [True, True]],
            [[False, True], [False, False]],
            [[False, False], [True, False]],
            [[True, True], [False, True]],
        ],
    }
    pool = TracePool(2)
    try:
        doc["sync_truth"] = pool.map(
            "sync_truth", {"port": _free_port(), "combos": doc["combos"]}
        )
        doc["wave_lockstep"] = pool.map(
            "wave_lockstep", {"port": _free_port(), "max_iters": 40}
        )
    finally:
        pool.close()
    dump("m19", doc)
    clean_tmp()


if __name__ == "__main__":
    main()
