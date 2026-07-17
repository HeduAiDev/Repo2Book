#!/usr/bin/env python3
"""数值一致性 linter — 正文数值推演表 vs explainer 素材,数字不许漂移。

约定(writer 契约同款):数值推演表的 markdown 表格前放一行 `<!-- trace: <mechanism_id> -->`
(HTML 注释,读者不可见;标记与表格间允许空行)。writer 可自由改排版/措辞,数字不可改。

阻断项:标记的机制在 explainer 里不存在;标记后没有紧跟表格;表内出现素材之外的数字(漂移);
        dossier 里 needs_worked_example 的机制在正文无任何 trace 标记(覆盖缺口)。
警告项:无 explainer.json(v2 旧章,跳过检查)。
用法:python3 lint_trace_consistency.py <chapter_dir>   阻断项存在则 exit 1。
"""
import json
import re
import sys
from pathlib import Path

NUM = re.compile(r'-?\d+(?:\.\d+)?')
MARK = re.compile(r'<!--\s*trace:\s*([\w-]+)\s*-->')
# 引用类记号不是推演数据:章节引用(§7 / 第 7 章)、源码行号引用(L308 / L308-310)。
# 对称地从两侧剔除,避免『§1 表』的 1、『L308』的 308 被当成漂移数字。
REF = re.compile(r'§\s*\d+|第\s*\d+\s*[章节]|\bL\d+(?:[-–]\d+)?')
# 显式标注为『虚构/假设/hypothetical』的整行 = 教学用反例,非 run 实测值;仅在正文侧剔除
# (explainer 素材是运行验证过的,不含虚构行),故只减少误报、绝不掩盖真实 drift。
HYPO = re.compile(r'虚构|假设|hypothetical', re.I)


def _nums(text: str) -> set:
    return {float(t) for t in NUM.findall(REF.sub(' ', text))}


def _nums_narrative(table: str) -> set:
    kept = "\n".join(ln for ln in table.splitlines() if not HYPO.search(ln))
    return _nums(kept)


def _tables_after_marks(md: str):
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        m = MARK.search(ln)
        if not m:
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        rows = []
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            rows.append(lines[j])
            j += 1
        yield m.group(1), "\n".join(rows)


def lint_trace_consistency(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    res = {"invalid": [], "drift": [], "coverage": [], "warn": []}
    ef = d / "explainer" / "explainer.json"
    if not ef.exists():
        res["warn"].append("  无 explainer.json(v2 旧章?)——跳过一致性检查")
        return res
    nar = d / "narrative" / "chapter.md"
    if not nar.exists():
        res["invalid"].append("  narrative/chapter.md 缺失")
        return res
    try:
        doc = json.loads(ef.read_text(encoding="utf-8"))
    except ValueError as e:
        res["invalid"].append(f"  explainer.json 不合法: {e}")
        return res
    allowed = {}
    for m in doc.get("mechanisms", []):
        table = ((m.get("worked_example") or {}).get("table")) or {}
        # ensure_ascii=False: 否则 CJK 被转成 \uXXXX,转义序列里的十六进制数字会被
        # NUM 正则吃进 allowed 集合,污染放行名单 → 全中文表格的漂移检测形同虚设。
        # (exp-0717-5:ch25 pessimistic-seed 2^62 vs explainer 2^30 的真实 drift 被此 bug 掩盖)
        allowed[m.get("mechanism_id")] = _nums(json.dumps(table, ensure_ascii=False))
    marked = set()
    for mid, table in _tables_after_marks(nar.read_text(encoding="utf-8")):
        marked.add(mid)
        if mid not in allowed:
            res["invalid"].append(f"  标记 trace:{mid} 在 explainer 里不存在")
            continue
        if not table:
            res["invalid"].append(f"  标记 trace:{mid} 后没有紧跟 markdown 表格")
            continue
        extra = _nums_narrative(table) - allowed[mid]
        if extra:
            res["drift"].append(
                f"  trace:{mid} 表内数字 {sorted(extra)} 不在 explainer 素材里(数字不可改,排版随意)")
    df = d / "dossier" / "dossier.json"
    if df.exists():
        try:
            need = {m["id"] for m in json.loads(df.read_text(encoding="utf-8")).get("mechanisms", [])
                    if m.get("needs_worked_example")}
        except ValueError:
            need = set()
        for mid in sorted(need - marked):
            res["coverage"].append(f"  机制 {mid} 的数值推演表未进正文(缺 <!-- trace: {mid} --> 标记)")
    return res


def print_report(res: dict, cd: str) -> int:
    print(f"Trace-Consistency Lint: {cd}\n{'=' * 60}")
    blocking = sum(len(v) for k, v in res.items() if k != "warn")
    for k, issues in res.items():
        for i in issues:
            print(("⚠️ " if k == "warn" else "❌ ") + f"{k}: {i}")
    if blocking == 0:
        print("✓ 正文数值推演表与 explainer 素材一致")
        return 0
    print(f"\n{'=' * 60}\n🔴 {blocking} BLOCKING")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_trace_consistency.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_trace_consistency(sys.argv[1]), sys.argv[1]))
