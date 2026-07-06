#!/usr/bin/env python3
"""章节重编号引擎——补章/交错的通用迁移工具(spec 2026-07-05-chapter-interleave)。

plan.json: {"moves": [{"old_dir": "ch34-primer-eplb", "new_id": "ch09"}, ...]}(显式全量)
两阶段幂等:①目录经临时名迁移(git mv,非 git 环境回退 rename);②全量引用重写——
  链接路径规范化(../chNN → ../../chNN,修全书历史笔误)→ 目录名整串替换 →
  裸 chNN(不跟 -)占位符同步替换 → 「第 N 章」单趟同步替换。
每处替换写迁移日志;--dry-run 只报不改;validate() 扫悬空引用。

用法:
  python3 scripts/renumber_chapters.py --instance vllm-ascend --plan <plan.json> [--dry-run]
  python3 scripts/renumber_chapters.py --instance vllm-ascend --insert <slug>@before:<目标dir>
  python3 scripts/renumber_chapters.py --instance vllm-ascend --validate
"""
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRPAT = re.compile(r'^ch(\d{2})-(.+)$')


@dataclass
class Move:
    old_dir: str
    new_id: str

    @property
    def old_id(self):
        return self.old_dir[:4]

    @property
    def new_dir(self):
        return self.new_id + self.old_dir[4:]


@dataclass
class Report:
    planned_moves: int = 0
    done_moves: int = 0
    skipped_moves: int = 0
    files_changed: int = 0
    log: list = field(default_factory=list)


def parse_moves(raw):
    return [Move(m["old_dir"], m["new_id"]) for m in raw]


def load_plan(path):
    return parse_moves(json.loads(Path(path).read_text(encoding="utf-8"))["moves"])


def _mv(src: Path, dst: Path):
    try:
        subprocess.run(["git", "mv", str(src), str(dst)], check=True, capture_output=True,
                       cwd=str(ROOT))
    except (subprocess.CalledProcessError, FileNotFoundError):
        src.rename(dst)


def _rewrite_targets(inst: Path):
    pats = ["artifacts/*/narrative/*.md", "artifacts/*/dossier/*.json", "artifacts/*/explainer/*.json",
            "artifacts/*/reviews/*.json", "artifacts/*/retrofit/*.json", "artifacts/*/diagrams/*.json",
            "book/cartography/*.json", "book/bible/*.json", "book/assets/roadmap/roadmap.py",
            "trace/state.json", "INSTANCE.md"]
    out = []
    for p in pats:
        out += sorted(inst.glob(p))
    # 豁免重编号 plan 存档自身(否则引擎会改写自己的映射记录,毁掉考古依据)
    return [f for f in out if not f.name.startswith('renumber-')]


def _rewrite_text(text: str, moves, report, fname):
    orig = text
    # 0) 链接路径规范化:](../chNN- → ](../../chNN-(历史笔误,narrative/ 出发需两层)
    text = re.sub(r'\]\(\.\./(ch\d{2}-)', r'](../../\1', text)
    # 1) 目录名整串替换(slug 唯一,顺序安全)
    for m in moves:
        text = text.replace(m.old_dir, m.new_dir)
    # 2) 裸 chNN(后不跟 -):占位符两趟同步替换
    idmap = {m.old_id: m.new_id for m in moves}
    for old, _ in idmap.items():
        text = re.sub(r'\b' + old + r'\b(?!-)', '\x00' + old + '\x00', text)
    for old, new in idmap.items():
        text = text.replace('\x00' + old + '\x00', new)
    # 3) 「第 N 章」单趟同步替换(半角数字,N∈映射集)
    nummap = {str(int(m.old_id[2:])): str(int(m.new_id[2:])) for m in moves}

    def _num(mo):
        n = mo.group(1)
        return '第 ' + nummap.get(n, n) + ' 章' if n in nummap else mo.group(0)

    text = re.sub(r'第\s*(\d{1,3})\s*章', _num, text)
    if text != orig:
        report.files_changed += 1
        report.log.append(f"rewrote {fname}")
    return text


def apply(inst: Path, moves, dry_run: bool) -> Report:
    rep = Report()
    todo = [m for m in moves if (inst / "artifacts" / m.old_dir).exists()]
    rep.planned_moves = len(todo)
    rep.skipped_moves = len(moves) - len(todo)
    if dry_run:
        for f in _rewrite_targets(inst):
            t = f.read_text(encoding="utf-8", errors="replace")
            r = Report()
            _rewrite_text(t, todo, r, f.name)
            rep.files_changed += r.files_changed
            rep.log += r.log
        rep.log.insert(0, f"[dry-run] moves={rep.planned_moves} skipped={rep.skipped_moves}")
        return rep
    # 阶段一:目录经临时名迁移(artifacts + book/papers)
    for base in (inst / "artifacts", inst / "book" / "papers"):
        if not base.exists():
            continue
        for m in todo:
            src = base / m.old_dir
            if src.exists():
                _mv(src, base / ("__tmp__" + m.new_dir))
        for m in todo:
            tmp = base / ("__tmp__" + m.new_dir)
            if tmp.exists():
                _mv(tmp, base / m.new_dir)
                rep.done_moves += 1
    # 阶段二:引用重写
    for f in _rewrite_targets(inst):
        t = f.read_text(encoding="utf-8", errors="replace")
        nt = _rewrite_text(t, todo, rep, str(f.relative_to(inst)))
        if nt != t:
            f.write_text(nt, encoding="utf-8")
    return rep


def validate(inst: Path):
    probs = []
    dirs = {d.name for d in (inst / "artifacts").iterdir() if d.is_dir()}
    link = re.compile(r'\]\((?:\.\./)+(ch\d{2}-[\w\-]+)/')
    for f in sorted(inst.glob("artifacts/*/narrative/*.md")):
        for mo in link.finditer(f.read_text(encoding="utf-8", errors="replace")):
            if mo.group(1) not in dirs:
                probs.append(f"{f.relative_to(inst)}: 悬空跨章链接 → {mo.group(1)}")
    return probs


def build_insert_plan(inst: Path, new_slug: str, before_dir: str):
    dirs = sorted(d.name for d in (inst / "artifacts").iterdir()
                  if d.is_dir() and DIRPAT.match(d.name))
    if before_dir not in dirs:
        raise SystemExit(f"目标章不存在: {before_dir}")
    pos = int(before_dir[2:4])
    moves = [{"old_dir": d, "new_id": f"ch{int(d[2:4]) + 1:02d}"}
             for d in dirs if int(d[2:4]) >= pos]
    return {"new_chapter_dir": f"ch{pos:02d}-{new_slug}", "moves": moves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--plan")
    ap.add_argument("--insert")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    inst = ROOT / "instances" / a.instance
    if a.validate:
        probs = validate(inst)
        print("\n".join(probs) if probs else "✓ 无悬空跨章链接")
        sys.exit(1 if probs else 0)
    if a.insert:
        slug, _, before = a.insert.partition("@before:")
        print(json.dumps(build_insert_plan(inst, slug, before), ensure_ascii=False, indent=2))
        return
    rep = apply(inst, load_plan(a.plan), a.dry_run)
    print("\n".join(rep.log))
    print(f"moves done={rep.done_moves} skipped={rep.skipped_moves} files_changed={rep.files_changed}")
    probs = [] if a.dry_run else validate(inst)
    if probs:
        print("\n".join(probs))
        sys.exit(1)


if __name__ == "__main__":
    main()
