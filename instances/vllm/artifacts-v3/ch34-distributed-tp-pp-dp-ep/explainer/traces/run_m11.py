# m11 取证：PP×TP 切片优化。
#   - 谓词真值表（单进程，真方法）：numel 整除 TP 才默认开；None 组不开；
#     all_gather_tensors 逐张量覆写开/关。
#   - 4 进程（TP=2×PP=2）真 isend/irecv：hidden [4,4]=16 元素只发 [2,4]=8 元素
#     切片；接收端 all_gather 重建 [4,4] 与原张量逐元素相等；线上元素 2×8=16
#     vs 不优化 2×16=32（省一半）。
from __future__ import annotations

from _driver_common import clean_tmp, dump


def _predicate_cases() -> list[dict]:
    from vllm.distributed.parallel_state import GroupCoordinator

    class _TP:
        world_size = 2

    gc = object.__new__(GroupCoordinator)
    cases = [
        ("hidden", 8, _TP, None, True),   # 8%2==0 → 默认开
        ("hidden", 7, _TP, None, False),  # 7%2==1 → 不开
        ("hidden", 8, None, None, False),  # 无 all_gather 组 → 不开
        ("hidden", 7, _TP, {"hidden": True}, True),    # 逐张量强开
        ("hidden", 8, _TP, {"hidden": False}, False),  # 逐张量强关
    ]
    return [
        {
            "key": k, "numel": n,
            "has_group": g is not None,
            "override": o,
            "use_all_gather": gc._should_use_all_gather(k, n, g, o),
        }
        for k, n, g, o, _ in cases
    ]


def main() -> None:
    from _trace_jobs import _fresh_store
    from _trace_pool import TracePool

    doc = {
        "mechanisms": ["m11"],
        "params": "TP=2×PP=2（4 进程）；hidden [4,4] float32",
        "predicate_cases": _predicate_cases(),
    }
    pool = TracePool(4)
    try:
        doc["slice_capture"] = pool.map("slice_capture", {"store": _fresh_store()})
    finally:
        pool.close()
    # 线上元素对账（由两侧实测推得：发送侧 sent_numel、接收侧重建形状）
    senders = [r for r in doc["slice_capture"] if r["role"] == "pp0"]
    doc["wire_volume"] = {
        "full_numel_per_sender": senders[0]["full_numel"],
        "sent_numel_per_sender": senders[0]["sent_numel"],
        "wire_with_opt": sum(r["sent_numel"] for r in senders),
        "wire_without_opt": sum(r["full_numel"] for r in senders),
        "sent_slice_formula": "hidden.reshape(2, -1)[tp_rank]（源码 L1061-L1063 同式）",
        "note": "PP 段间张量在 TP 组内本就 replicated——p 份重复发送是浪费",
    }
    dump("m11", doc)
    clean_tmp()


if __name__ == "__main__":
    main()
