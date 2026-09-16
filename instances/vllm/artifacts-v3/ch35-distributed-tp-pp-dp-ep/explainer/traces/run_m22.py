# m22 取证：EP dispatch/combine（默认 AgRs=all_gatherv/reduce_scatterv）。
# 2 rank 各 3 token、topk 一奇一偶跨 rank 命中（专家 e 归 rank e%2）：
# dispatch 后每 rank 全网 6 行；本地只算命中行（rank0 6 行 / rank1 4 行）；
# combine 归位 owner 视角 3 行 = 原值（0.5+0.5 权重）。MoE prepare/finalize
# 消费现场同参数再跑一遍。AgRs vs 真 A2A 的投递量对账由两侧实测推得。
from __future__ import annotations

from _driver_common import clean_tmp, dump


def main() -> None:
    from _trace_jobs import _fresh_store
    from _trace_pool import TracePool

    doc = {
        "mechanisms": ["m22"],
        "params": "2 rank × 3 token × hidden 4；topk=2（一奇一偶）；w=0.5×2",
    }
    pool = TracePool(2)
    try:
        doc["ep_full"] = pool.map("ep_full", {"store": _fresh_store()})
    finally:
        pool.close()
    r0, r1 = doc["ep_full"]
    world = 2
    gathered = r0["gathered_rows"]
    doc["wire_analysis"] = {
        "agrs_deliveries": gathered * world,
        "agrs_note": "all_gatherv 把每行投递给全部 rank（含不需要它的 rank）",
        "true_a2a_deliveries": r0["n_hit_rows"] + r1["n_hit_rows"],
        "true_a2a_note": "真 A2A 只投递各 rank 命中行（rank0 需 6 行、rank1 需 4 行）",
        "overhead_rows": gathered * world - (r0["n_hit_rows"] + r1["n_hit_rows"]),
        "overhead_pct_of_agrs": round(
            100
            * (gathered * world - (r0["n_hit_rows"] + r1["n_hit_rows"]))
            / (gathered * world)
        ),
        "agrs_premium_vs_a2a_pct": round(
            100
            * (
                gathered * world / (r0["n_hit_rows"] + r1["n_hit_rows"]) - 1
            )
        ),
    }
    dump("m22", doc)
    clean_tmp()


if __name__ == "__main__":
    main()
