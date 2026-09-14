# ch34 explainer 驱动公共件：路径铺设 + JSON 落盘（保 LF）。
from __future__ import annotations

import json
import sys
from pathlib import Path

TRACES = Path(__file__).resolve().parent
CHAPTER = TRACES.parents[1]
IMPL = CHAPTER / "implementation"
TESTS = CHAPTER / "tests"
for _p in (str(TRACES), str(TESTS), str(IMPL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def dump(name: str, doc: dict) -> None:
    out = TRACES / f"{name}.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def clean_tmp() -> None:
    import shutil

    tmp = TRACES / ".tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tests_tmp = TESTS / ".tmp"
    if tests_tmp.exists():
        shutil.rmtree(tests_tmp)
