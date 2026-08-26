"""Driver for m20 (remove_all 单元素快路径 + 'OPTIMIZATION: Avoid list(set)'
——Python 热路径连函数调用都要省) — host run, pin vLLM v0.27.1
(vllm/v1/core/sched/utils.py:L62-L91 + scheduler.py:L2117-L2118).

Structural facts (machine-independent):
  - empty removal set: returns the SAME list object (no-op).
  - single-item set: fast path — in-place list.remove, returns the ORIGINAL
    object (identity preserved), no new list allocated.
  - multi-item set: falls back to a list comprehension — returns a NEW list,
    the original is untouched.
Call sites in this chapter's hot path: update_from_output removes stopped
RUNNING requests (usually exactly ONE per beat — the fast path is the common
case by construction), finish_requests removes aborted requests.
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.utils import remove_all  # noqa: E402


def main():
    out = {
        "driver": "run_m20_remove_all.py",
        "mechanism": "m20 remove_all 单元素快路径（utils.py:L62-L91 + scheduler.py:L2117-L2118 'OPTIMIZATION: Avoid list(set)'）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/（utils.py 130 行无删除）",
        "cases": [],
    }

    # empty set
    lst = ["r1", "r2", "r3"]
    ret = remove_all(lst, set())
    out["cases"].append({
        "case": "空集合 no-op", "removed_count": 0,
        "same_object": ret is lst, "length": len(lst), "content": list(lst),
    })

    # single item — fast path
    lst = ["r1", "r2", "r3"]
    ret = remove_all(lst, {"r2"})
    out["cases"].append({
        "case": "单元素快路径", "removed_count": 1,
        "same_object": ret is lst, "length": len(lst), "content": list(lst),
        "note": "in-place list.remove + 原对象返回：零新列表分配——update_from_output 每拍停 1 个请求是常态，快路径即主路径",
    })

    # multi item — comprehension
    lst = ["r1", "r2", "r3", "r4"]
    ret = remove_all(lst, {"r1", "r3"})
    out["cases"].append({
        "case": "多元素重建", "removed_count": 2,
        "same_object": ret is not lst,
        "original_length": len(lst), "original_content": list(lst),
        "new_length": len(ret), "new_content": list(ret),
        "note": "list comprehension 重建新列表，原列表不动；调用方必须用返回值",
    })

    c0, c1, c2 = out["cases"]
    assert c0["same_object"] is True and c0["length"] == 3
    assert c1["same_object"] is True and c1["content"] == ["r1", "r3"]
    assert c2["same_object"] is True and c2["original_content"] == ["r1", "r2", "r3", "r4"]
    assert c2["new_content"] == ["r2", "r4"]
    out["call_sites"] = [
        "update_from_output：remove_all(self.running, stopped_running_reqs)（scheduler.py:L1946-L1948）——每拍至多停 1 个请求，单元素快路径是构造上的常态",
        "finish_requests：remove_all(self.running, running_requests_to_remove)（scheduler.py:L2278-L2281）",
        "同类尾部优化：scheduler.py:L2117-L2118 注释 'OPTIMIZATION: Avoid list(set)'——热循环里连 set(list) 都要省",
    ]

    dest = Path(__file__).with_name("m20_remove_all.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for c in out["cases"]:
        print(c["case"], {k: v for k, v in c.items() if k not in ("case", "note")})


if __name__ == "__main__":
    main()
