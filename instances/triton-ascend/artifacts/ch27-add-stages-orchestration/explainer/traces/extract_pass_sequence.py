#!/usr/bin/env python3
"""Deterministic extractor for ch27 (add_stages orchestration).

本章无精简版：真实的 triton-ascend 下降流水线要跑起来需要 Ascend 工具链
(bishengir-compile + CANN + NPU 硬件)，host 上无法执行。但本章的教学对象是
**编排**本身——哪些 pass、按什么顺序、带什么开关——这一信息完全静态可从
compiler.py 源码抽取，不需要运行编译器。

本脚本用纯文本解析把 make_ttir(第一段 ttir)与 ttir_to_linalg(第二段 ttadapter)
里挂到 pass_manager 上的 pass 调用**按源码出现顺序**抽出来，连同行号一起打印，
作为 explainer.json 里 worked_example 逐轮表的确定性证据(source-of-truth)。

用法:
    python3 extract_pass_sequence.py > pass_sequence.txt
"""
import json
import re
from pathlib import Path

# host 上的规范源码路径(仓内相对)——脚本按仓库根定位。
REPO = Path(__file__).resolve().parents[6]
SRC = REPO / "instances/triton-ascend/source/third_party/ascend/backend/compiler.py"

lines = SRC.read_text(encoding="utf-8").splitlines()


def span(start_marker, end_marker):
    """返回 [start, end) 行号区间(1-based)。"""
    s = e = None
    for i, ln in enumerate(lines, 1):
        if s is None and start_marker in ln:
            s = i
        elif s is not None and end_marker in ln and i > s:
            e = i
            break
    return s, e


# add_<name>( 调用，含 passes.common./passes.ttir./ascend.passes.ttir. 三种命名空间
CALL = re.compile(r'(passes\.common|passes\.ttir|ascend\.passes\.ttir)\.(add_\w+)\s*\(')


def collect(s, e):
    out = []
    for i in range(s, e):
        ln = lines[i - 1]
        m = CALL.search(ln)
        if m:
            ns, name = m.group(1), m.group(2)
            out.append({"line": i, "ns": ns, "pass": name})
    return out


ttir_s, ttir_e = span("def make_ttir(", "return mod")
lin_s, lin_e = span("def ttir_to_linalg(", "def __get_metadata_attr_by_callback")

make_ttir_passes = collect(ttir_s, ttir_e)
ttir_to_linalg_all = collect(lin_s, lin_e)

# 自动调度块是可选的(if metadata["add_auto_scheduling"]: 默认 False)。
sched_s, sched_e = None, None
for i in range(lin_s, lin_e):
    if 'metadata["add_auto_scheduling"]' in lines[i - 1]:
        sched_s = i + 1
    if sched_s and lines[i - 1].strip() == "" and i > sched_s and sched_e is None:
        sched_e = i
        break

sched_passes = [p for p in ttir_to_linalg_all if sched_s and sched_s <= p["line"] < sched_e]
mainline_passes = [p for p in ttir_to_linalg_all if not (sched_s and sched_s <= p["line"] < sched_e)]

report = {
    "source_file": "third_party/ascend/backend/compiler.py",
    "make_ttir": {"span": [ttir_s, ttir_e], "count": len(make_ttir_passes), "passes": make_ttir_passes},
    "ttir_to_linalg_mainline": {"count": len(mainline_passes), "passes": mainline_passes},
    "ttir_to_linalg_auto_scheduling_block": {
        "gated_by": 'metadata["add_auto_scheduling"] (default False, compiler.py:L770)',
        "span": [sched_s, sched_e], "count": len(sched_passes), "passes": sched_passes,
    },
}

print(json.dumps(report, indent=2, ensure_ascii=False))
print("\n=== make_ttir 顺序(第一段 ttir，与基座共享) ===")
for k, p in enumerate(make_ttir_passes, 1):
    print(f"  {k:2d}. L{p['line']}  {p['ns']}.{p['pass']}")
print(f"\n=== ttir_to_linalg 主线顺序(第二段 ttadapter，默认路径) ===")
for k, p in enumerate(mainline_passes, 1):
    print(f"  {k:2d}. L{p['line']}  {p['ns']}.{p['pass']}")
print(f"\n=== 可选自动调度块(add_auto_scheduling=True 才进链) ===")
for k, p in enumerate(sched_passes, 1):
    print(f"  {k:2d}. L{p['line']}  {p['ns']}.{p['pass']}")
