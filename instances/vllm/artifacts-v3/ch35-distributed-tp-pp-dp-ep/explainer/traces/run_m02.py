# m02/m03 取证：8 GPU（TP=2/PP=2/DP=2/PCP=1）5 维 rank 张量四刀。
# MoE 与 dense 各跑一遍——EP 组只在 MoE 建；TP 组独享 mq_broadcaster；
# EP 组独享 use_all2all；每维组数 × 双群组 = ProcessGroup 总数（44）。
from __future__ import annotations

from _driver_common import clean_tmp, dump


def main() -> None:
    from _trace_jobs import _fresh_store
    from _trace_pool import TracePool

    doc = {
        "mechanisms": ["m2", "m3"],
        "params": "8 rank，(ExternalDP, DP, PP, PCP, TP) = (1, 2, 2, 1, 2)",
    }
    pool = TracePool(8)
    try:
        doc["moe"] = pool.map("cuts_full", {"is_moe": True, "store": _fresh_store()})
        doc["dense"] = pool.map("cuts_full", {"is_moe": False, "store": _fresh_store()})
    finally:
        pool.close()
    dump("m02", doc)
    clean_tmp()


if __name__ == "__main__":
    main()
