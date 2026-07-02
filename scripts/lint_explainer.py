#!/usr/bin/env python3
"""Explainer 素材 linter — 校验 explainer/explainer.json(illustrator/writer 的素材真相源)。

对照 dossier.mechanisms 账本逐机制对账:
阻断项:explainer.json 缺失/不合法;needs_worked_example 机制无条目;needs_figure 机制无条目;
        intuition/quantified/invariant 空;逐轮表 <2 行;trace_source 非法;run 无 trace_ref 或
        表格数字在 trace 原始输出里找不到(数字不许编);manual 无 manual_reason;needs_figure
        机制无 figure_spec;figure_spec 缺 claim/template/caption_draft、numbers 缺
        value/provenance。**figure-only 机制**(dossier needs_worked_example=false 且条目自身
        无 worked_example)不强求直觉/数值推演/不变量——只校验其 figure_specs。
警告项:manual trace 未经运行验证。
用法:python3 lint_explainer.py <chapter_dir>   阻断项存在则 exit 1。
"""
import json
import re
import sys
from pathlib import Path

NUM = re.compile(r'-?\d+(?:\.\d+)?')
TEMPLATES = {"state-table", "swimlane", "layout", "tensor-flow",
             "before-after", "state-machine", "flow", "tiling"}


def _nums(text: str) -> set:
    return {float(t) for t in NUM.findall(text)}


def lint_explainer(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    res = {"invalid": [], "mechanism": [], "trace": [], "figure": [], "warn": []}
    ef = d / "explainer" / "explainer.json"
    if not ef.exists():
        res["invalid"].append("  explainer/explainer.json 缺失")
        return res
    try:
        doc = json.loads(ef.read_text(encoding="utf-8"))
    except ValueError as e:
        res["invalid"].append(f"  JSON 不合法: {e}")
        return res
    dossier = {}
    df = d / "dossier" / "dossier.json"
    if df.exists():
        try:
            dossier = json.loads(df.read_text(encoding="utf-8"))
        except ValueError:
            pass
    want_we = {m["id"] for m in dossier.get("mechanisms", []) if m.get("needs_worked_example")}
    want_fig = {m["id"] for m in dossier.get("mechanisms", []) if m.get("needs_figure")}
    got = {m.get("mechanism_id"): m for m in doc.get("mechanisms", [])}
    for mid in sorted(want_we - set(got)):
        res["mechanism"].append(f"  {mid}: dossier 要求 worked example,explainer 无条目")
    for mid in sorted(want_fig - set(got)):
        res["figure"].append(f"  {mid}: dossier 要求配图,explainer 无条目")
    for mid, m in got.items():
        we = m.get("worked_example") or {}
        # figure-only 机制(dossier needs_worked_example=false 且条目本身无 worked_example)
        # 不强求直觉/数值推演——只在 dossier 要求 worked example 或条目自带时才校验。
        if mid in want_we or m.get("worked_example"):
            for k in ("intuition", "quantified"):
                if not m.get(k):
                    res["mechanism"].append(f"  {mid}: {k} 为空")
            inv = m.get("invariant") or {}
            if not (inv.get("claim") and inv.get("argument")):
                res["mechanism"].append(f"  {mid}: invariant.claim/argument 为空(要单调量或基例+归纳步,不是断言)")
            rows = ((we.get("table") or {}).get("rows")) or []
            if len(rows) < 2:
                res["mechanism"].append(f"  {mid}: 逐轮表不足 2 轮(rows={len(rows)})")
            ts = we.get("trace_source")
            if ts == "run":
                tr = d / "explainer" / (we.get("trace_ref") or "")
                if not we.get("trace_ref") or not tr.exists():
                    res["trace"].append(f"  {mid}: trace_source=run 但 trace_ref 缺失/文件不存在")
                else:
                    have = _nums(tr.read_text(encoding="utf-8", errors="replace"))
                    for row in rows:
                        for cell in row:
                            for v in _nums(str(cell)):
                                if v not in have:
                                    res["trace"].append(
                                        f"  {mid}: 表格数字 {v:g} 在 {we['trace_ref']} 里找不到(数字不许编)")
            elif ts == "manual":
                if not we.get("manual_reason"):
                    res["trace"].append(f"  {mid}: trace_source=manual 必须写 manual_reason(降级原因)")
                else:
                    res["warn"].append(f"  {mid}: manual trace 未经运行验证")
            else:
                res["trace"].append(f"  {mid}: trace_source={ts!r} 非法(run|manual)")
        specs = m.get("figure_specs") or []
        if mid in want_fig and not specs:
            res["figure"].append(f"  {mid}: dossier 要求配图,figure_specs 为空")
        for s in specs:
            fid = s.get("figure_id") or "?"
            for k in ("claim", "caption_draft"):
                if not s.get(k):
                    res["figure"].append(f"  {fid}: {k} 为空")
            if s.get("template") not in TEMPLATES:
                res["figure"].append(f"  {fid}: template={s.get('template')!r} 非法({sorted(TEMPLATES)})")
            for n in s.get("numbers") or []:
                if not (n.get("value") and n.get("provenance")):
                    res["figure"].append(f"  {fid}: numbers 项缺 value/provenance(图中数字皆有出处)")
    return res


def print_report(res: dict, cd: str) -> int:
    print(f"Explainer Lint: {cd}\n{'=' * 60}")
    blocking = sum(len(v) for k, v in res.items() if k != "warn")
    for k, issues in res.items():
        for i in issues:
            print(("⚠️ " if k == "warn" else "❌ ") + f"{k}: {i}")
    if blocking == 0:
        print("✓ explainer 素材检查通过(数字可溯源/逐轮表/不变量/figure-spec 齐)")
        return 0
    print(f"\n{'=' * 60}\n🔴 {blocking} BLOCKING")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_explainer.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_explainer(sys.argv[1]), sys.argv[1]))
