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
    unresolved_sections: list = field(default_factory=list)


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
            "artifacts/*/diagrams/*.py",
            "book/cartography/*.json", "book/bible/*.json", "book/assets/roadmap/roadmap.py",
            "trace/state.json", "INSTANCE.md"]
    out = []
    for p in pats:
        out += sorted(inst.glob(p))
    # 豁免重编号 plan 存档自身(否则引擎会改写自己的映射记录,毁掉考古依据)
    return [f for f in out if not f.name.startswith('renumber-')]


def _own_diagram_move(fname: str, moves):
    """若 fname 是某条 move 自身章节下的 artifacts/<chapter_dir>/diagrams/*.py 脚本,返回对应 Move,否则 None。

    §N.M 徽标只应指"本章自身",不该被论文引用(如 "arXiv:2210.17323 §3.1")或正文自引用锚点
    (如 "[§2.9](#anchor)")误伤——因此严格限定:仅 diagrams/*.py 文件、且章节目录段命中某条
    move 的 old_dir(dry-run,目录尚未迁移)或 new_dir(apply,阶段一已把目录迁到新名)。
    """
    parts = fname.replace('\\', '/').split('/')
    if not parts or not parts[-1].endswith('.py') or 'diagrams' not in parts:
        return None
    idx = parts.index('diagrams')
    if idx == 0:
        return None
    chapter_dir = parts[idx - 1]
    for m in moves:
        if chapter_dir in (m.old_dir, m.new_dir):
            return m
    return None


def _chapter_dir_of(fname: str):
    """从实例内相对路径取章节目录段(artifacts/<chapter_dir>/...),取不到返回 None。"""
    parts = fname.replace('\\', '/').split('/')
    if len(parts) >= 2 and parts[0] == 'artifacts':
        return parts[1]
    return None


def _refers_to(window: str, tgt: "Move", own_new):
    """window 内是否有确凿线索指向 tgt 章(目录名/「第 N 章」/上下一章相对词)。"""
    # 只认"新号"线索:规则 5 跑在规则 1(目录名)与规则 3(「第 N 章」)之后,窗口里的目录名与章号
    # 此刻必已是新值。若也认旧号,就会在第二趟把已经改好的 §N.M 再推一格——真实翻车样本见
    # test_no_double_shift_when_chapter_word_equals_another_moves_old_id。
    if tgt.new_dir in window:
        return True
    tgt_new = int(tgt.new_id[2:])
    if re.search(r'第\s*' + str(tgt_new) + r'\s*章', window):
        return True
    if own_new is not None:
        if '下一章' in window and tgt_new == own_new + 1:
            return True
        if ('上一章' in window or '前一章' in window) and tgt_new == own_new - 1:
            return True
    return False


def _window_points_at(window: str, n: int, cur_num=None):
    """窗口里是否已有指向"第 n 章"的线索(目录名 chNN- 或「第 n 章」)。

    用于识别"这处 §n.M 本来就对"——例如「第 36 章 §36.8」:36 虽恰是某条 move 的旧号,
    但窗口明明白白指着现在的第 36 章,它已经自洽,既不该改也不该报。
    """
    if re.search(r'\bch0*' + str(n) + r'-', window):
        return True
    if re.search(r'第\s*' + str(n) + r'\s*章', window):
        return True
    if cur_num is not None:   # 「(上|下)一章 §N.M」——相对词也是确凿线索
        if '下一章' in window and n == cur_num + 1:
            return True
        if ('上一章' in window or '前一章' in window) and n == cur_num - 1:
            return True
    return False


def _rewrite_sections(text: str, moves, chapter_dir: str):
    """正文小节号重编号:`## N.M` 标题 + `§N.M` 徽标/引用 + 标题派生的章内锚点。
    返回 (新文本, unresolved 列表)。

    分三档(见 scripts/tests/test_renumber_sections.py 的来源说明):
      A 小节标题(^##..######):本章旧号即改——标题绝不会是论文引用。
      B 本章自引 §N.M(N == 本章旧号):即改。
      C 跨章引 §N.M:仅当同行近处有指向该章的确凿线索时才改;否则原样保留并登记 unresolved。
    标题一改,GitHub 由标题派生的锚点(`## 36.2 顶层编排` → `#362-顶层编排`)随之变号,
    故按本趟真实改过的标题收集 前缀映射,同步改写 `](#362-…)` → `](#382-…)`,否则章内链接全断。
    条件都绑定到"某个具体的旧号→新号"而非号段,故天然幂等、且不会在同趟内级联位移。
    """
    unresolved = []
    by_old = {int(m.old_id[2:]): m for m in moves}
    own = next((m for m in moves if chapter_dir in (m.old_dir, m.new_dir)), None)
    own_old = int(own.old_id[2:]) if own else None
    own_new = int(own.new_id[2:]) if own else None
    cur_mo = re.match(r'ch(\d+)', chapter_dir or '')
    cur_num = int(cur_mo.group(1)) if cur_mo else None
    anchor_map = {}

    lines = text.split('\n')
    for li, line in enumerate(lines):
        if own is not None:
            mo = re.match(r'(#{2,6}\s*)(\d{1,3})((?:\.\d{1,3})+)', line)
            if mo and int(mo.group(2)) == own_old:
                old_num, tail = mo.group(2) + mo.group(3), mo.group(3)
                new_num = str(own_new) + tail
                anchor_map[old_num.replace('.', '')] = new_num.replace('.', '')
                line = mo.group(1) + new_num + line[mo.end():]

        def _sec(mo):
            n = int(mo.group(1))
            if own is not None and n == own_old:
                return '§' + str(own_new) + '.' + mo.group(2)
            tgt = by_old.get(n)
            if tgt is not None and n == cur_num:
                # §N 恰是本章"现在"的号 → 自引,与某条 move 的旧号撞号纯属巧合。两种撞法都会发生:
                #   ① 新章补进被腾空的号位(ch31-structured-output 的 §31.x 是它自己的小节);
                #   ② 移动章改完号后,新号又正好是另一条 move 的旧号(ch35 的 §35.x vs 旧 ch35→ch37)。
                # 不改也不报——上面的分支 B 已先处理过"本章旧号",走到这里的只可能是自引。
                return mo.group(0)
            if tgt is not None:
                window = _sec.line[max(0, mo.start() - 160):mo.start() + 160]
                if _refers_to(window, tgt, own_new):
                    return '§' + str(int(tgt.new_id[2:])) + '.' + mo.group(2)
                if _window_points_at(window, n, cur_num):
                    return mo.group(0)   # 已自洽(见 _window_points_at),静默放过
                unresolved.append(f'{chapter_dir}:L{li + 1} §{mo.group(1)}.{mo.group(2)}'
                                  f' → 近处无指向 {tgt.new_dir} 的线索,未改,请人核')
            return mo.group(0)

        _sec.line = line
        lines[li] = re.sub(r'§(\d{1,3})\.(\d{1,3})', _sec, line)
    text = '\n'.join(lines)
    for old_a, new_a in anchor_map.items():
        text = text.replace('](#' + old_a + '-', '](#' + new_a + '-')
    return text, unresolved


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
    # 4) 「§N.M」章内小节锚点/徽标——仅限"本章自身"的 diagrams/*.py 脚本(见 _own_diagram_move),
    #    且仅当捕获的 N(int() 归一,兼容零填充如 "§09.2")恰等于该脚本所属章节的旧号时才重写,
    #    M(节内序号)不变。论文 §引用(如 "arXiv:2210.17323 §3.1")与正文自引用锚点
    #    (如 "[§2.9](#anchor)")均不落在 diagrams/*.py 内,天然豁免。
    own_move = _own_diagram_move(fname, moves)
    if own_move is not None:
        local_nummap = {str(int(own_move.old_id[2:])): str(int(own_move.new_id[2:]))}

        def _sec(mo):
            n = str(int(mo.group(1)))
            return '§' + local_nummap[n] + '.' + mo.group(2) if n in local_nummap else mo.group(0)

        text = re.sub(r'§(\d{1,3})\.(\d{1,3})', _sec, text)
    # 5) 正文 `## N.M` 小节标题与 §N.M 引用(exp-2026-07-20-02)——原引擎只改 diagrams/*.py,
    #    导致移动章正文滞留旧号、与已改的图徽标打架、lint_chapter_map --require 全红。
    if fname.replace('\\', '/').endswith('.md') and '/narrative/' in fname.replace('\\', '/'):
        chdir = _chapter_dir_of(fname)
        if chdir:
            text, un = _rewrite_sections(text, moves, chdir)
            report.unresolved_sections += un
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
            _rewrite_text(t, todo, r, str(f.relative_to(inst)))
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


def apply_sections_only(inst: Path, moves, dry_run: bool) -> Report:
    """补丁模式:目录已迁完、只有正文小节号滞留旧号时,单跑规则 5(exp-2026-07-20-02 修复通道)。

    幂等——_rewrite_sections 的条件都绑定具体旧号,跑第二遍是 no-op。
    """
    rep = Report()
    for f in sorted(inst.glob("artifacts/*/narrative/*.md")):
        chdir = f.parent.parent.name
        t = f.read_text(encoding="utf-8")
        nt, un = _rewrite_sections(t, moves, chdir)
        rep.unresolved_sections += un
        if nt != t:
            rep.files_changed += 1
            rep.log.append(f"sections rewritten {f.relative_to(inst)}")
            if not dry_run:
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
    ap.add_argument("--sections-only", action="store_true",
                    help="目录已迁完,只补正文 §N.M / `## N.M` 小节号(见 apply_sections_only)")
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
    moves = load_plan(a.plan)
    rep = (apply_sections_only(inst, moves, a.dry_run) if a.sections_only
           else apply(inst, moves, a.dry_run))
    print("\n".join(rep.log))
    if rep.unresolved_sections:
        print("⚠️ 以下 §N.M 无法确证所指章节,未改,请人核:")
        print("\n".join("   " + u for u in rep.unresolved_sections))
    print(f"moves done={rep.done_moves} skipped={rep.skipped_moves} files_changed={rep.files_changed}")
    probs = [] if a.dry_run else validate(inst)
    if probs:
        print("\n".join(probs))
        sys.exit(1)


if __name__ == "__main__":
    main()
