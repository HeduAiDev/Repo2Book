#!/usr/bin/env python3
"""Dossier 机制清单 linter — 校验 dossier.json 的 mechanisms[](v3 素材先行流水线的账本)。

mechanisms 是"一图讲一机制、一例讲一算法"的覆盖度账本:explainer 按它产素材、
illustrator 按它配图、reviewer 按它对账。

阻断项:JSON 不合法/缺 mechanisms;机制缺必填字段、枚举非法、id 重复;
        kind=algorithm 但 needs_worked_example!=true;source_anchors 格式非法/文件不存在/行号越界;
        paper_origin 格式非法(arXiv id/URL、sections 非空)。
警告项:实例 source/ 不在(跳过锚点行号核验)。警告:algorithm 无 paper_origin;prereq 章目录缺失。
用法:python3 lint_dossier.py <chapter_dir>   阻断项存在则 exit 1。
"""
import json
import re
import sys
from pathlib import Path

KINDS = {"algorithm", "dataflow", "layout", "protocol", "config"}
DIFF = {"core", "supporting"}
ANCHOR = re.compile(r'^([\w./-]+\.\w+):L(\d+)(?:-L?(\d+))?$')
PAPER_ID = re.compile(r'^(arXiv:\d{4}\.\d{4,5}(v\d+)?|https?://\S+)$')


def _source_root(chapter_dir: Path):
    for p in chapter_dir.resolve().parents:
        if p.name == "artifacts":
            return p.parent / "source"
    return None


def lint_dossier(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    res = {"invalid": [], "mechanism": [], "anchor": [], "warn": []}
    f = d / "dossier" / "dossier.json"
    if not f.exists():
        res["invalid"].append("  dossier/dossier.json 缺失")
        return res
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
    except ValueError as e:
        res["invalid"].append(f"  JSON 不合法: {e}")
        return res
    mechs = doc.get("mechanisms")
    if not isinstance(mechs, list) or not mechs:
        res["invalid"].append("  缺 mechanisms[](v3 机制清单——覆盖度/配图/深度的账本)")
        return res
    src = _source_root(d)
    if src is None or not src.exists():
        res["warn"].append("  找不到实例 source/,跳过锚点行号核验")
        src = None
    seen = set()
    for i, m in enumerate(mechs):
        mid = m.get("id") or f"#{i}"
        if m.get("id") in seen:
            res["mechanism"].append(f"  {mid}: id 重复")
        seen.add(m.get("id"))
        for k in ("id", "name", "kind", "source_anchors", "difficulty"):
            if not m.get(k):
                res["mechanism"].append(f"  {mid}: 缺 {k}")
        if m.get("kind") not in KINDS:
            res["mechanism"].append(f"  {mid}: kind={m.get('kind')!r} 非法(应为 {sorted(KINDS)})")
        if m.get("difficulty") not in DIFF:
            res["mechanism"].append(f"  {mid}: difficulty={m.get('difficulty')!r} 非法(core|supporting)")
        if m.get("kind") == "algorithm" and m.get("needs_worked_example") is not True:
            res["mechanism"].append(f"  {mid}: kind=algorithm 必须 needs_worked_example=true")
        po = m.get("paper_origin")
        if po is not None:
            if not isinstance(po, dict):
                res["mechanism"].append(f"  {mid}: paper_origin 须为对象 {{paper, sections}}")
            else:
                if not PAPER_ID.match(str(po.get("paper", ""))):
                    res["mechanism"].append(f"  {mid}: paper_origin.paper 格式非法(应为 arXiv:NNNN.NNNNN 或 URL)")
                if not isinstance(po.get("sections"), list) or not po.get("sections"):
                    res["mechanism"].append(f"  {mid}: paper_origin.sections 须为非空列表(§/Eq 锚)")
        elif doc.get("kind") == "primer":
            res["mechanism"].append(f"  {mid}: primer 章每个机制必填 paper_origin")
        elif m.get("kind") == "algorithm":
            res["warn"].append(f"  {mid}: kind=algorithm 且无 paper_origin——确认该算法确无论文出处")
        pr = m.get("prereq")
        if pr:
            arts = d.resolve()
            arts = next((p for p in arts.parents if p.name == "artifacts"), None)
            if arts is None or not list(arts.glob(pr + "-*")):
                res["warn"].append(f"  {mid}: prereq={pr} 对应章目录尚不存在(原理章未建则属正常)")
        for a in m.get("source_anchors") or []:
            am = ANCHOR.match(a)
            if not am:
                res["anchor"].append(f"  {mid}: 锚点格式非法 {a!r}(应为 path:Lnnn[-Lnnn])")
                continue
            if src is None:
                continue
            fp = src / am.group(1)
            if not fp.exists():
                res["anchor"].append(f"  {mid}: 文件不存在 {am.group(1)}")
                continue
            n = sum(1 for _ in fp.open(encoding="utf-8", errors="replace"))
            end = int(am.group(3) or am.group(2))
            if end > n:
                res["anchor"].append(f"  {mid}: 行号越界 {a}(文件共 {n} 行)")
    return res


def print_report(res: dict, cd: str) -> int:
    print(f"Dossier Lint: {cd}\n{'=' * 60}")
    blocking = sum(len(v) for k, v in res.items() if k != "warn")
    for k, issues in res.items():
        for i in issues:
            print(("⚠️ " if k == "warn" else "❌ ") + f"{k}: {i}")
    if blocking == 0:
        print("✓ dossier 机制清单检查通过")
        return 0
    print(f"\n{'=' * 60}\n🔴 {blocking} BLOCKING")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_dossier.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_dossier(sys.argv[1]), sys.argv[1]))
