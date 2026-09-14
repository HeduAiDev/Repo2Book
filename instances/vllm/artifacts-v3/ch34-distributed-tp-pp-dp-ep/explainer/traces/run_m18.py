# m18/m20 取证：DPCoordinator 三 socket 控制面（真 ZMQ XPUB/XSUB/PULL）。
# 事件链：READY×2 → stats 上报 [2,1,0.6] → 前端 100ms 级发布（含实测时延）→
# FIRST_REQ → START_DP_WAVE(0, exclude=1) → wave_complete → (None, 1, False) →
# stale wave 上报 → 补广播 (3, exclude=0)。
from __future__ import annotations

from _driver_common import clean_tmp, dump


def main() -> None:
    from _trace_pool import TracePool

    doc = {"mechanisms": ["m18", "m20"], "params": "1 进程内真跑 coordinator + 双引擎 XSUB + 前端 XSUB"}
    pool = TracePool(1)
    try:
        doc["coord_timed"] = pool.map("coord_timed", {})[0]
    finally:
        pool.close()
    dump("m18", doc)
    clean_tmp()


if __name__ == "__main__":
    main()
