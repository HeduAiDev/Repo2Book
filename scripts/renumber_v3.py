#!/usr/bin/env python3
"""v3 重编号（2026-09-16）：在 ch28 前插入新章「DeepSeek-V4 原理」——
原 ch28 起全部 +1（全书 41 章）。仿 renumber_chapters.py（v2 版）的迁移纪律，
但按 artifacts-v3 / l2-specs / bible-v3 / pedagogy-plan 的实际形态实现。

规则（顺序敏感）：
  1. 目录 git mv（降序，避免撞名）：ch28→29 ch29→30 ch30→31 ch31→32 ch34→35
     ch36→37 ch37→38 ch38→39
  2. l2-specs 文件改名（同映射，仅存在的）
  3. 文本替换 pass（覆盖 artifacts-v3/**、book/cartography/**、book/bible/*.json、
     trace/state.json）：
     a) 目录名整串  ch28-deepseek-v4-assembly → ch29-deepseek-v4-assembly ...
     b) 图前缀     L2-ch28 → L2-ch29 ；ch28-fig- → ch29-fig-
     c) 「第 N 章」  N≥28 → N+1（.md 文件）
     d) 裸 chNN     N≥28 → N+1（仅 .json 文件；.md 里裸 chNN 罕见，避免误伤不动）
     e) l2-spec 数字字段 "chapter": 28 → 29
  4. 校验：无旧目录名残留；所有 ../diagrams/L2-chNN.png 链接有实文件；
     所有跨章链接 ../../chNN-slug/ 指向存在的目录。

用法：python scripts/renumber_v3.py [--dry-run]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INST = ROOT / "instances" / "vllm"
SHIFT = 28          # 从第 28 章起 +1
DRY = "--dry-run" in sys.argv

MOVES = {  # old dir -> old num mapping 由 old 目录名推出；new = num+1
    "ch28-deepseek-v4-assembly": 28,
    "ch29-sampler-pipeline": 29,
    "ch30-grammar-compilation": 30,
    "ch31-bitmask-enforcement": 31,
    "ch34-distributed-tp-pp-dp-ep": 34,
    "ch36-pd-disaggregation": 36,
    "ch37-kv-pooling": 37,
    "ch38-openai-serving-multiturn": 38,
}


def new_dir(old: str) -> str:
    n = int(old[2:4])
    return f"ch{n + 1:02d}" + old[4:]


def bump_num(m):
    """字符串里出现的章号（28..40）→ +1；<28 或 >40 不动。"""
    n = int(m.group(1))
    return f"ch{n + 1:02d}" if SHIFT <= n <= 40 else m.group(0)


def bump_zh(m):
    n = int(m.group(1))
    return f"第 {n + 1} 章" if SHIFT <= n <= 40 else m.group(0)


def rewrite_text(text: str, is_md: bool) -> str:
    # b) 所有 chNN token 统一单次 bump（词边界防 batch28 误伤；覆盖裸 chNN、
    #    chNN-fig-、L2-chNN、以及 chNN-slug 目录名——**必须先于目录名整串替换**，
    #    否则目录名会被咬两次（规则 a 换成新号、规则 b 再 +1）。2026-09-16 翻车记。
    text = re.sub(r"(?<![0-9A-Za-z])ch(\d{2})(?![0-9])", bump_num, text)
    # a) 目录名整串兜底（统一 bump 之后旧目录名理论上已消失，此步为安全网/幂等）
    for old, n in MOVES.items():
        text = text.replace(old, new_dir(old))
    # c) 「第 N 章」
    text = re.sub(r"第\s*(\d+)\s*章", bump_zh, text)
    # e) l2-spec 数字章号字段
    if not is_md:
        text = re.sub(r'("chapter"\s*:\s*(\d+))',
                      lambda m: '"chapter": ' + (str(int(m.group(2)) + 1) if SHIFT <= int(m.group(2)) <= 40 else m.group(2)),
                      text)
    return text


# v2 封版档案：章号是 v2 自己的编号体系，绝不可随 v3 位移（2026-07-06 renumber 的教训）
V2_ARCHIVE = {
    "map.json", "outline-final.json", "arch-model.json", "chapter-queue.json",
    "draft-outline.json", "outline-v3-draft.md",
    "concepts.json", "interfaces.json", "arc-map.json", "glossary.json",
}


def targets():
    out = []
    for p in ["artifacts-v3/*/narrative/*.md", "artifacts-v3/*/dossier/*.json",
              "artifacts-v3/*/explainer/*.json", "artifacts-v3/*/research/*.json",
              "artifacts-v3/*/reviews/*.json", "artifacts-v3/*/diagrams/*.json",
              "artifacts-v3/*/diagrams/*.py", "artifacts-v3/*/tests/*.json",
              "artifacts-v3/*/implementation/**/*.md",   # impl-notes 可能引未来章号
              "book/cartography/pedagogy-plan.json", "book/cartography/papers-map.json",
              "book/cartography/ARCHITECTURE.md", "book/cartography/FIGURE-SYSTEM.md",
              "book/cartography/WRITING-CONTRACT-v3.md",
              "book/cartography/l2-specs/*.json",
              "book/bible/glossary-v3.json", "book/bible/concepts-v3.json",
              "book/bible/interfaces-v3.json", "book/bible/figures.json",
              "book/bible/foreshadow-v3.json"]:
        out += sorted(INST.glob(p))
    return [f for f in out
            if not f.name.startswith("renumber-") and f.name not in V2_ARCHIVE]


def main():
    # 阶段一：目录迁移（降序）
    moves_sorted = sorted(MOVES.items(), key=lambda kv: -kv[1])
    for old, _ in moves_sorted:
        src, dst = INST / "artifacts-v3" / old, INST / "artifacts-v3" / new_dir(old)
        if not src.exists():
            print(f"skip (不存在): {old}")
            continue
        if dst.exists():
            print(f"✗ 目标已存在，中止: {dst}")
            sys.exit(1)
        print(f"mv {old} -> {dst.name}")
        if not DRY:
            try:
                subprocess.run(["git", "mv", str(src), str(dst)], check=True, capture_output=True, cwd=str(ROOT))
            except subprocess.CalledProcessError:
                src.rename(dst)

    # 阶段二：l2-specs 改名（降序）
    for old, n in moves_sorted:
        sp_old = INST / "book/cartography/l2-specs" / f"ch{n}.json"
        sp_new = INST / "book/cartography/l2-specs" / f"ch{n + 1}.json"
        if sp_old.exists():
            print(f"mv l2-specs/{sp_old.name} -> {sp_new.name}")
            if not DRY:
                try:
                    subprocess.run(["git", "mv", str(sp_old), str(sp_new)], check=True, capture_output=True, cwd=str(ROOT))
                except subprocess.CalledProcessError:
                    sp_old.rename(sp_new)   # 未跟踪文件 git mv 会拒

    # 阶段三：文本替换
    changed = 0
    for f in targets():
        try:
            t = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        if f.name == "figures.json":
            # 混合账本：仅重写 book=="v3" 的条目，v2 条目（无 book 字段）原样保留
            try:
                arr = json.loads(t)
            except json.JSONDecodeError:
                print(f"✗ figures.json 解析失败，跳过"); continue
            for e in arr:
                if isinstance(e, dict) and e.get("book") == "v3":
                    ne = {k: (rewrite_text(v, False) if isinstance(v, str) else v) for k, v in e.items()}
                    e.clear(); e.update(ne)
            nt = json.dumps(arr, ensure_ascii=False, indent=1) + "\n"
            if nt != t:
                changed += 1
                print(f"edit {f.relative_to(INST)} (v3 条目)")
                if not DRY:
                    f.write_text(nt, encoding="utf-8", newline="\n")
            continue
        nt = rewrite_text(t, is_md=f.suffix == ".md")
        if nt != t:
            changed += 1
            print(f"edit {f.relative_to(INST)}")
            if not DRY:
                f.write_text(nt, encoding="utf-8", newline="\n")

    # state.json 特例：v2/v3 混装，只动 v3 子对象与 status 段（chapters 字典是 v2 编号，禁动）
    sp = INST / "trace" / "state.json"
    if sp.exists():
        raw = sp.read_text(encoding="utf-8")
        d = json.loads(raw)
        n_d = json.loads(json.dumps(d, ensure_ascii=False))
        if isinstance(n_d.get("status"), str):
            n_d["status"] = rewrite_text(n_d["status"], False)
        if isinstance(n_d.get("v3"), dict):
            n_d["v3"] = json.loads(rewrite_text(json.dumps(n_d["v3"], ensure_ascii=False), False))
        ns = json.dumps(n_d, ensure_ascii=False, indent=2) + "\n"
        if ns != json.dumps(d, ensure_ascii=False, indent=2) + "\n":
            changed += 1
            print("edit trace/state.json (仅 v3 子对象+status)")
            if not DRY:
                sp.write_text(ns, encoding="utf-8", newline="\n")

    print(f"\n{'DRY-RUN ' if DRY else ''}files changed = {changed}")

    if DRY:
        return
    # 阶段四：校验
    problems = []
    bad = []
    pat_bare = re.compile(r"\bch(2[89]|3[0-8])-[\w\-]+")
    for f in targets() + sorted(INST.glob("artifacts-v3/*/narrative/*.md")):
        t = f.read_text(encoding="utf-8", errors="replace")
        for m in pat_bare.finditer(t):
            # 命中旧号 chNN- 前缀即残留（新号目录 ch29..ch39 不在此正则类里？在——需按映射排除）
            seg = m.group(0)
            if seg in MOVES:  # 恰是旧目录名 → 残留
                bad.append(f"{f.relative_to(INST)}: 残留旧目录名 {seg}")
    link = re.compile(r"\]\((?:\.\./)+(ch\d{2}-[\w\-]+|diagrams/L2-ch\d{2}\.(?:png|svg))")
    for md in sorted(INST.glob("artifacts-v3/*/narrative/*.md")):
        for mo in link.finditer(md.read_text(encoding="utf-8", errors="replace")):
            tgt = mo.group(1)
            # 文件在 <chap>/narrative/x.md；../../ 从 narrative/ 起算 = artifacts-v3/
            p = (md.parent / ".." / ".." / tgt).resolve()
            if not p.exists():
                problems.append(f"{md.relative_to(INST)}: 悬空 → {tgt}")
    print("残留旧名:", len(bad))
    for b in bad[:20]:
        print("  ", b)
    print("悬空链接:", len(problems))
    for p in problems[:20]:
        print("  ", p)
    sys.exit(1 if (bad or problems) else 0)


if __name__ == "__main__":
    main()
