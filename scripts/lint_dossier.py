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
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台打印符号免疫(exp-2026-08-17)
from pathlib import Path

KINDS = {"algorithm", "dataflow", "layout", "protocol", "config"}
DIFF = {"core", "supporting"}
ANCHOR = re.compile(r'^([\w./-]+\.\w+):L(\d+)(?:-L?(\d+))?$')
PAPER_ID = re.compile(r'^(arXiv:\d{4}\.\d{4,5}(v\d+)?|https?://\S+)$')
# embed_verbatim(exp-2026-07-18-02):embed_excerpts 与 pin blob 逐字比对用
EMBED_PATH = re.compile(r'^[\w./-]+\.\w+$')
EMBED_LINES = re.compile(r'^L(\d+)(?:-L?(\d+))?$')
# 注记行豁免:`# … 省略 …`/`// … 省略`/`# SOURCE:`/`# PAPER:`/`# SUBTRACTED:` 打头
NOTE_LINE = re.compile(r'^\s*(?:#|//)\s*(?:…|\.{3})?\s*(?:省略|SOURCE:|PAPER:|SUBTRACTED:)')


def _source_root(chapter_dir: Path):
    for p in chapter_dir.resolve().parents:
        if p.name.startswith("artifacts"):
            return p.parent / "source"
    return None


def _norm_line(line: str) -> str:
    """空白归一:expandtabs → rstrip → 行内连续空白折叠为单空格 → \\uXXXX 转写还原。
    缩进(行首空白)不折叠——Python/MLIR 缩进承载语义,缩进错=默写错,必须抓。
    \\uXXXX/\\UXXXXXXXX ASCII 转写还原为字符:analyst 把 pin 里的 ⊕/𝔽 等抄成
    `\\u2295` 转写属编码转录差,非默写(oracle 对表 ch23 实证),归一后等价。"""
    line = line.expandtabs().rstrip()
    m = re.match(r'^(\s*)(.*)$', line)
    line = m.group(1) + re.sub(r'\s+', ' ', m.group(2))
    def _u(mm):
        try:
            return chr(int(mm.group(1), 16))
        except ValueError:
            return mm.group(0)
    return re.sub(r'\\[uU]([0-9a-fA-F]{4,8})', _u, line)


def _dedent_norm(lines):
    """去公共缩进(textwrap.dedent 同义,但对已归一行):整体平移容差、相对缩进保留。"""
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    if not indents:
        return lines
    k = min(indents)
    return [l[k:] if l.strip() else l for l in lines]


def _ellipsis_split(norm: str):
    """省略感知:dossier 行内含 …/... 即视为 analyst 截断标注。
    返回 (首个省略号前的前缀, 是否含省略)。纯省略行前缀为空。
    oracle 对表实证三形态:整行 `...`(裸/`# ...`/`// ...`)、行尾截断
    `…a scalar is loaded. ...`、行中压缩 `raise NameError(...)`。"""
    mm = re.search(r'(?:\.{3}|…)', norm)
    if not mm:
        return norm, False
    prefix = norm[:mm.start()].rstrip()
    # 纯注释省略行(# ... / // ... / 裸 ...)前缀只剩注释符或空 → 整行跳过
    if re.fullmatch(r'\s*(?:#|//|/\*)?\s*', prefix):
        return "", True
    return prefix, True


def _pin_lines(src: Path, rel: str, cache: dict):
    """取 pin 内容:优先 git show HEAD:<path>(免疫脏工作区),fallback 工作区文件。
    返回 list[str] 或 None(两者都取不到)。"""
    if rel in cache:
        return cache[rel]
    text = None
    try:
        import subprocess
        r = subprocess.run(["git", "-C", str(src), "show", f"HEAD:{rel}"],
                           capture_output=True, timeout=30)
        if r.returncode == 0:
            text = r.stdout.decode("utf-8", errors="replace")
    except OSError:
        pass
    if text is None:
        fp = src / rel
        if fp.exists():
            text = fp.read_text(encoding="utf-8", errors="replace")
    cache[rel] = text.splitlines() if text is not None else None
    return cache[rel]


def _check_embed_verbatim(doc: dict, src, res: dict):
    """embed_verbatim(SDD 2026-07-18):dossier embed_excerpts 的 code 与 pin blob
    空白归一后逐字比对。全量模式(行数=区间)逐行严格;子集模式(analyst 区间内抽行,
    全书实测 34%)按序子序列匹配。三层闭环补『pin↔dossier』一环(dossier↔正文=lint_fidelity)。"""
    excerpts = doc.get("embed_excerpts")
    if not isinstance(excerpts, list):
        return
    comparable = []
    for i, e in enumerate(excerpts):
        if not isinstance(e, dict):
            continue
        rel, lines = str(e.get("path", "")), str(e.get("lines", ""))
        # primer 论文条目(§/Eq 锚、无文件 path)/n-a 行号/external-source 快照:合法形态,静默跳过
        if not EMBED_PATH.match(rel) or "external-source" in rel:
            continue
        lm = EMBED_LINES.match(lines)
        if not lm or not isinstance(e.get("code"), str):
            continue
        comparable.append((i, e, rel, int(lm.group(1)), int(lm.group(2) or lm.group(1))))
    if not comparable:
        return
    if src is None:
        res["warn"].append("  embed_verbatim: 找不到实例 source/,跳过 embed_excerpts 逐字核验")
        return
    cache = {}
    for i, e, rel, start, end in comparable:
        tag = f"embed_excerpts[{i}] {rel}:L{start}" + (f"-L{end}" if end != start else "")
        pin = _pin_lines(src, rel, cache)
        if pin is None:
            # 不在 pin:前瞻 primer 读上游码/跨仓引用(vllm-ascend 引 vllm/*)/论文包路径
            # ——多真相源形态合法存在,降 warn 人核(oracle 对表:31 例全为此类,仅 1 例疑真错)
            res["warn"].append(f"  embed: {tag} 文件不在 pin(前瞻/跨仓/论文包引用或路径错,人核)")
            continue
        if end > len(pin):
            res["embed"].append(f"  {tag}: 行号越界(文件共 {len(pin)} 行)")
            continue
        region = _dedent_norm([_norm_line(x) for x in pin[start - 1:end]])
        # 每行 → (归一文本, 省略前缀, 是否截断);注记行与纯省略行(`...`/`# ...`)剔除
        kept = []
        for raw in e["code"].splitlines():
            cl = _norm_line(raw)
            if NOTE_LINE.match(cl):
                continue
            prefix, trunc = _ellipsis_split(cl)
            if trunc and not prefix:
                continue  # 纯省略标注行
            kept.append((cl, prefix, trunc))
        # 统一缩进容差:analyst 常把嵌套代码整体 dedent 后内嵌(oracle 对表 ch27 .td 8 例)
        # ——相对缩进仍严格(两侧各自去公共缩进),整体平移不算默写。
        deds = _dedent_norm([c for c, _, _ in kept])
        code = []
        for (cl, prefix, trunc), dcl in zip(kept, deds):
            shift = len(cl) - len(dcl)
            code.append((dcl, prefix[shift:] if trunc else "", trunc))

        def _match(dl, pl):
            """截断行按前缀匹配,整行按全等。"""
            cl, prefix, trunc = dl
            return pl.startswith(prefix) if trunc else pl == cl

        if len(code) == end - start + 1:
            # 全量模式:逐行严格(截断行容前缀)
            for j, (dl, pl) in enumerate(zip(code, region)):
                if not _match(dl, pl):
                    res["embed"].append(
                        f"  {tag} 第 {start + j} 行与 pin 不符:dossier={dl[0].strip()!r}  pin={pl.strip()!r}")
                    break
        else:
            # 子集模式:非空行按序子序列匹配(空行不作锚,避免到处配上假阴)
            pi = 0
            for dl in code:
                if not dl[0].strip():
                    continue
                while pi < len(region) and not _match(dl, region[pi]):
                    pi += 1
                if pi >= len(region):
                    import difflib
                    near = difflib.get_close_matches(dl[0], region, n=1)
                    hint = f"  pin 区间内最相近:{near[0].strip()!r}" if near else ""
                    # 子集模式暂 warn(SDD §4 分阶段):存量 117 例混杂改行宽/重复行贪心
                    # 错配/列表重排等转录形态,假阳未清零前不升 blocking;全量/越界类已 blocking。
                    res["warn"].append(
                        f"  embed: {tag} dossier 行在 pin 区间内按序匹配不到(默写/乱序/行号错,人核):"
                        f"{dl[0].strip()!r}{hint}")
                    break
                pi += 1


def lint_dossier(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    res = {"invalid": [], "mechanism": [], "anchor": [], "embed": [], "warn": []}
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
            note = m.get("paper_origin_note")
            if isinstance(note, str) and note.strip():
                res["warn"].append(f"  {mid}: 无 paper_origin(注记豁免: {note.strip()[:40]})")
            else:
                res["mechanism"].append(f"  {mid}: primer 章每个机制必填 paper_origin")
        elif m.get("kind") == "algorithm":
            res["warn"].append(f"  {mid}: kind=algorithm 且无 paper_origin——确认该算法确无论文出处")
        pr = m.get("prereq")
        if pr:
            arts = d.resolve()
            arts = next((p for p in arts.parents if p.name.startswith("artifacts")), None)
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
    _check_embed_verbatim(doc, src, res)
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
