# m01/m05/m08 取证：GroupCoordinator 双群组解剖 + all_reduce 派发双路径 + PP 张量字典。
#   - tp_anatomy（world=2，真建组）：partial 1+2 → 3；直接路径与强开 custom-op 路径
#     两条都走通；barrier/对象广播走 cpu_group；回退链前六级在 host 的构造面状态。
#   - tp1_shortcircuit（world=1）：world_size==1 短路返回同一张量对象。
#   - pp_dict（world=2）：metadata 走 cpu_group 对象通道、2 个张量各自 isend/irecv。
from __future__ import annotations

from _driver_common import CHAPTER, TRACES, clean_tmp, dump


def main() -> None:
    from _trace_jobs import _fresh_store
    from _trace_pool import TracePool

    doc = {"mechanisms": ["m1", "m5", "m8"], "params": "TP=2（2 进程 gloo）与 TP=1（1 进程）"}

    pool2 = TracePool(2)
    try:
        doc["tp_anatomy"] = pool2.map("tp_anatomy", {"store": _fresh_store()})
        doc["pp_dict"] = pool2.map("pp_dict", {"store": _fresh_store()})
    finally:
        pool2.close()

    pool1 = TracePool(1)
    try:
        doc["tp1_shortcircuit"] = pool1.map("tp1_shortcircuit", {"store": _fresh_store()})
    finally:
        pool1.close()

    dump("m01", doc)
    clean_tmp()


if __name__ == "__main__":
    main()
