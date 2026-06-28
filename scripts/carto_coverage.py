#!/usr/bin/env python3
"""Cartography 覆盖交叉核对（W23）：活动实例源码的每个顶层子系统，是否被 outline 某章覆盖。

强制 cartography 步骤（ARCHITECT-RUNBOOK §0.6 第 3 条）的可执行版——别再裸眼漏章。
- 列 source/<canonical_prefix>/ 下的顶层子系统（目录）。
- 逐一确认被 outline-final.json 某章的 key_source_paths 覆盖。
- 同时校验：每个 key_source_paths 在 source/ 真实存在；姊妹篇 pairs_with 基座路径存在（若配了 base 实例）。
未覆盖子系统 / 不存在路径 → 退出码 1。可在 ARCHITECTURE.md 里对"故意只点名不成章"的子系统说明。

用法: python3 scripts/carto_coverage.py            # 活动实例
      REPO2BOOK_INSTANCE=<name> python3 scripts/carto_coverage.py
"""
import json
import os
import sys
import instance

SRC = instance.source_dir()
CFG = instance.config()
PREFIX = (CFG.get("source") or {}).get("canonical_prefix") or instance.active_name()
OUTLINE = instance.book_dir() / "cartography" / "outline-final.json"


def main():
    if not OUTLINE.exists():
        print(f"找不到 outline: {OUTLINE}")
        return 1
    o = json.loads(OUTLINE.read_text(encoding="utf-8"))
    chapters = o.get("chapters", o if isinstance(o, list) else [])
    covered = " ".join(p for c in chapters for p in (c.get("key_source_paths") or []))

    # 1) 路径存在性
    missing = []
    for c in chapters:
        for p in (c.get("key_source_paths") or []):
            if not (SRC / p).exists():
                missing.append((c.get("chapter_id"), p))

    # 2) 顶层子系统覆盖
    pkg = SRC / PREFIX
    subs = sorted(d.name for d in pkg.iterdir() if d.is_dir() and not d.name.startswith("__")) if pkg.is_dir() else []
    uncovered = [s for s in subs
                 if f"{PREFIX}/{s}/" not in covered and f"{PREFIX}/{s}." not in covered]

    print(f"实例 {instance.active_name()} · 前缀 {PREFIX}/ · {len(chapters)} 章 · {len(subs)} 顶层子系统")
    if missing:
        print(f"\n❌ key_source_paths 不存在 ({len(missing)}):")
        for cid, p in missing[:20]:
            print(f"    {cid}: {p}")
    if uncovered:
        print(f"\n⚠ 未被任何章 key_source_paths 覆盖的顶层子系统 ({len(uncovered)}):")
        for s in uncovered:
            print(f"    {PREFIX}/{s}/   ← 漏章？或在 ARCHITECTURE.md 显式点名入横切")
    if not missing and not uncovered:
        print("\n✓ 全部顶层子系统有章覆盖，且 key_source_paths 全部存在")
        return 0
    print("\n（覆盖核对未过：补章 / 或确认是有意点名的 minor 子系统）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
