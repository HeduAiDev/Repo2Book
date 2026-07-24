#!/usr/bin/env python3
"""figures.json 登记完整性门禁：每章 diagrams/figure-manifest.json 里的每张图，
都必须在 Book Bible 的 figures.json 里登记（archivist 归档职责）。

背景：工作流/手动归档反复漏登本章图——尤其 chapter-map（ch23/25/26/27/33 五次）。
门禁只看渲染产物与 manifest 的 blind_review，看不出「图在盘、manifest PASS，但
bible/figures.json 没这一条」。这个 linter 把它补成确定性检查，替掉 Lead 每章手工 grep。

命名约定：manifest 的 `chapter-map` 在 bible 里登记为 `fig-<chNN>-chapter-map`
（防跨章撞 id）；其余 figure_id 两边一致。

用法：python3 scripts/lint_figures_registered.py            # 扫活动实例全部章
      python3 scripts/lint_figures_registered.py <chapter_dir> [更多...]
退出码 1 = 有未登记图。
"""
import json
import sys
import glob
import re
import pathlib
import instance

CHNN = re.compile(r'(ch\d{2})')


def bible_figures_path():
    return instance.book_dir() / "bible" / "figures.json"


def load_bible_index(bp):
    """→ {chapter_id: set(figure_id)}。"""
    idx = {}
    if not bp.is_file():
        return idx
    try:
        m = json.loads(bp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return idx
    figs = m if isinstance(m, list) else m.get("figures", [])
    for f in figs:
        if isinstance(f, dict) and f.get("chapter_id") and f.get("figure_id"):
            idx.setdefault(f["chapter_id"], set()).add(f["figure_id"])
    return idx


def manifest_figure_ids(chapter_dir):
    mp = pathlib.Path(chapter_dir) / "diagrams" / "figure-manifest.json"
    if not mp.is_file():
        return []
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    figs = m if isinstance(m, list) else m.get("figures", [])
    return [f.get("figure_id") for f in figs if isinstance(f, dict) and f.get("figure_id")]


_FIG = re.compile(r'^fig-')
_CHNN = re.compile(r'^ch\d{2}-')


def _core(fid):
    """归一化 figure_id：去掉可选的 `fig-` 前缀、再去掉可选的 `chNN-` 前缀，
    得到「核心名」。各章登记体例不一(有的 fig-chNN-m1-x、有的 fig-m1-x、
    有的 chNN-m1-x、chapter-map↔fig-chNN-chapter-map),核心名统一后可比。"""
    s = _FIG.sub('', fid, count=1)
    s = _CHNN.sub('', s, count=1)
    return s


def check(chapter_dir, bible_idx):
    """→ [未登记的 figure_id...]（按归一化核心名比对，容忍 fig-/chNN- 前缀差异）。"""
    p = pathlib.Path(chapter_dir)
    mnum = CHNN.search(p.name)
    if not mnum:
        return []
    cid = mnum.group(1)
    registered_cores = {_core(f) for f in bible_idx.get(cid, set())}
    missing = []
    for fid in manifest_figure_ids(chapter_dir):
        if _core(fid) not in registered_cores:
            missing.append(fid)
    return missing


def main():
    args = sys.argv[1:]
    bp = bible_figures_path()
    bible_idx = load_bible_index(bp)
    if args:
        dirs = args
    else:
        # chapters_glob() 给的是 narrative/chapter.md 路径；取其上两级=章目录
        dirs = sorted({str(pathlib.Path(g).resolve().parents[1])
                       for g in glob.glob(instance.chapters_glob())})
    total = 0
    for d in dirs:
        missing = check(d, bible_idx)
        if missing:
            total += len(missing)
            name = pathlib.Path(d).name
            print(f"❌ {name}: {len(missing)} 张图未登记进 bible/figures.json")
            for fid in missing:
                print(f"    {fid}")
    if total == 0:
        print("✓ 每章 manifest 图均已登记进 bible/figures.json")
        return 0
    print(f"\n共 {total} 张图未登记（archivist 归档漏登，须补进 figures.json）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
