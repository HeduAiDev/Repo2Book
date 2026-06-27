#!/usr/bin/env python3
"""确定性行号重映射：把书中精确的 `vllm/<path>.py:Lx-Ly` 引用从旧 rev 平移到新 rev。

用 difflib.SequenceMatcher 对两版文件做行级对齐（不解析 diff 文本，稳）。
分类每处引用：
  identity      — 文件零变化，行号不动；
  shift         — 引用区间落在未改区，安全平移到新行号；
  content-change— 引用区间触到了被改/删的行，需重抽片段（不可机械改号）；
  unmappable    — 文件在新 rev 不存在 / 解析不出。

用法:
  python3 scripts/remap_lines_v021.py report            # 全书统计（默认）
  python3 scripts/remap_lines_v021.py apply             # 应用 shift 类重映射（就地改文件），打印 content-change 清单
  python3 scripts/remap_lines_v021.py dump <file.md|.py> # 单文件逐条
旧/新 rev 与源码根在常量里。
"""
import re
import sys
import glob
import os
import subprocess
import difflib

SRC = "instances/vllm/source"
OLD = "f3fef1235"
NEW = "v0.21.0"

# vllm/a/b.py:381-383  |  :L381  |  :L381-L412  |  :381
CITE = re.compile(r"(vllm/[\w./]+\.py):L?(\d+)(?:\s*-\s*L?(\d+))?")

_blob = {}


def show(rev, path):
    key = (rev, path)
    if key not in _blob:
        r = subprocess.run(["git", "-C", SRC, "show", f"{rev}:{path}"],
                           capture_output=True, text=True)
        _blob[key] = r.stdout.split("\n") if r.returncode == 0 else None
    return _blob[key]


_maps = {}
_tree = None


def _newtree():
    global _tree
    if _tree is None:
        r = subprocess.run(["git", "-C", SRC, "ls-tree", "-r", "--name-only", NEW],
                           capture_output=True, text=True)
        _tree = [p for p in r.stdout.split("\n") if p.startswith("vllm/") and p.endswith(".py")]
    return _tree


def resolve_path(path):
    """Abbreviated `vllm/.../suffix.py` -> unique real full path in the tree (else path)."""
    if "/.../" not in path:
        return path
    suffix = path.split("/.../", 1)[1]
    cands = [p for p in _newtree() if p.endswith("/" + suffix)]
    if len(cands) == 1:
        return cands[0]
    if cands:  # disambiguate by shortest path (closest to root under vllm/)
        return min(cands, key=len)
    return path


def build_map(path):
    """returns (old2new dict 1-based, changed_old set 1-based, exists_new bool, identical bool)."""
    if path in _maps:
        return _maps[path]
    real = resolve_path(path)
    a = show(OLD, real)
    b = show(NEW, real)
    if a is None:
        res = (None, None, None, None)            # didn't exist in old either
    elif b is None:
        res = (None, None, False, False)          # removed in new
    elif a == b:
        res = ({i + 1: i + 1 for i in range(len(a))}, set(), True, True)
    else:
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        o2n = {}
        chg = set()
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    o2n[i1 + k + 1] = j1 + k + 1
            else:  # replace / delete / insert
                for k in range(i1, i2):
                    chg.add(k + 1)
                    o2n[k + 1] = None
        res = (o2n, chg, True, False)
    _maps[path] = res
    return res


def nearest_new(o2n, old):
    """map an old line to a new line, falling back to nearest mapped neighbour."""
    if old in o2n and o2n[old] is not None:
        return o2n[old]
    for d in range(1, 200):
        for cand in (old - d, old + d):
            if o2n.get(cand):
                return o2n[cand]
    return None


def classify(path, ls, le):
    o2n, chg, exists, identical = build_map(path)
    if exists is None:
        return "unmappable", None
    if exists is False:
        return "unmappable", None
    if identical:
        return "identity", (ls, le)
    touched = any((L in chg) for L in range(ls, le + 1))
    if touched:
        ns, ne = nearest_new(o2n, ls), nearest_new(o2n, le)
        return "content-change", (ns, ne)
    return "shift", (nearest_new(o2n, ls), nearest_new(o2n, le))


def iter_citations(text):
    for m in CITE.finditer(text):
        path = m.group(1)
        ls = int(m.group(2))
        le = int(m.group(3)) if m.group(3) else ls
        yield m, path, ls, le


def files():
    return (sorted(glob.glob("instances/vllm/artifacts/ch*/narrative/chapter.md"))
            + sorted(glob.glob("instances/vllm/artifacts/ch*/implementation/*.py")))


def report():
    cats = {"identity": 0, "shift": 0, "content-change": 0, "unmappable": 0}
    per_ch = {}
    for f in files():
        ch = f.split("/")[3]
        text = open(f, encoding="utf-8").read()
        for m, path, ls, le in iter_citations(text):
            cat, _ = classify(path, ls, le)
            cats[cat] += 1
            d = per_ch.setdefault(ch, {"shift": 0, "content-change": 0, "identity": 0, "unmappable": 0})
            d[cat] += 1
    print(f"{'chapter':40} {'ident':>5} {'shift':>5} {'CHG':>5} {'unmap':>5}")
    for ch in sorted(per_ch):
        d = per_ch[ch]
        mark = "!!" if d["content-change"] or d["unmappable"] else "  "
        print(f"{mark}{ch:38} {d['identity']:5} {d['shift']:5} {d['content-change']:5} {d['unmappable']:5}")
    print(f"\nTOTAL  identity={cats['identity']}  shift={cats['shift']}  "
          f"content-change={cats['content-change']}  unmappable={cats['unmappable']}")
    print(f"precise citations total = {sum(cats.values())}")


def apply():
    """rewrite shift-class citations in place; collect content-change/unmappable for re-sync."""
    flagged = []
    changed_files = 0
    for f in files():
        text = open(f, encoding="utf-8").read()
        out = []
        last = 0
        dirty = False
        for m, path, ls, le in iter_citations(text):
            cat, newrange = classify(path, ls, le)
            if cat == "shift" and newrange and newrange[0] and newrange[1]:
                ns, ne = newrange
                # preserve the original textual style (L-prefix? range?)
                orig = m.group(0)
                hasL = ":L" in orig
                if m.group(3):
                    rep = f"{path}:{'L' if hasL else ''}{ns}-{'L' if hasL else ''}{ne}"
                else:
                    rep = f"{path}:{'L' if hasL else ''}{ns}"
                out.append(text[last:m.start()])
                out.append(rep)
                last = m.end()
                dirty = True
            elif cat in ("content-change", "unmappable"):
                flagged.append((f.split("/")[3], f.split("/")[-1], path, ls, le, cat, newrange))
        out.append(text[last:])
        if dirty:
            open(f, "w", encoding="utf-8").write("".join(out))
            changed_files += 1
    print(f"applied shift-remaps in {changed_files} files")
    print(f"flagged for re-sync (content-change/unmappable): {len(flagged)}")
    import json
    json.dump([{"chapter": c, "file": fn, "path": p, "old": [ls, le], "cat": cat,
                "new_approx": nr} for (c, fn, p, ls, le, cat, nr) in flagged],
              open("instances/vllm/book/_v021-update/_resync-flags.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("flags written to instances/vllm/book/_v021-update/_resync-flags.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        report()
    elif cmd == "apply":
        apply()
    elif cmd == "dump" and len(sys.argv) > 2:
        text = open(sys.argv[2], encoding="utf-8").read()
        for m, path, ls, le in iter_citations(text):
            cat, nr = classify(path, ls, le)
            print(f"{cat:14} {path}:{ls}-{le} -> {nr}")
