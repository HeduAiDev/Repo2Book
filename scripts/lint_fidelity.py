#!/usr/bin/env python3
"""Fidelity linter — enforces the subtract-only companion contract.

Checks (blocking unless noted):
  1. Every top-level def/class in implementation/*.py has a `# SOURCE: vllm/...` ref in its span.
  2. No invention markers (# ADDED / # TOY / # FAKE / # INVENTED).
  3. Narrative grounds in real vLLM: count of `vllm/...py` refs >= refs to `implementation/`,
     and >= MIN_VLLM_REFS.
  4. (warning) at least one `# SUBTRACTED:` marker present.

Usage: python3 lint_fidelity.py <chapter_dir>
Exit 1 if any blocking issue.
"""
import ast
import json
import re
import sys
from pathlib import Path

MIN_VLLM_REFS = 5
INVENTION_MARKERS = ("# ADDED", "# TOY", "# FAKE", "# INVENTED")

# 真实源码前缀：活动实例的规范前缀(如 vllm_ascend) + 对照基座前缀(如 vllm)。
# 与 lint_source_grounding.py 保持一致——off-spine 实例章节的「真实源码」是本仓规范前缀，
# 不只是基座 vllm/。
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import instance as _instance
    _SRC_PREFIXES = list(_instance.canonical_prefixes())
    try:
        _root = json.load(open(Path(__file__).resolve().parent.parent / "repo2book.json"))
        _dep = (_root.get("instances", {}).get(_instance.active_name(), {}) or {}).get("depends_on")
        if _dep:
            _bsrc = json.load(open(
                Path(__file__).resolve().parent.parent / "instances" / _dep / "repo2book.json"
            )).get("source", {})
            for _bp in (_bsrc.get("canonical_prefixes") or [_bsrc.get("canonical_prefix") or _dep]):
                if _bp and _bp not in _SRC_PREFIXES:
                    _SRC_PREFIXES.append(_bp)
    except Exception:
        pass
except Exception:
    _SRC_PREFIXES = ["vllm"]
# 长前缀优先，避免 vllm 抢先匹配 vllm_ascend（其实前缀后强制 '/'，二者互斥，但仍按长度排序更稳）
_SRC_REF_RE = re.compile(
    r"(?:" + "|".join(re.escape(p) for p in sorted(_SRC_PREFIXES, key=len, reverse=True)) + r")/[\w/]+\.py"
)


# ── 省略标记闸门（lint-exp-002）──────────────────────────────────────────
# dossier.embed_excerpts 是「要内嵌的真实源码片段」的唯一真相源：每条登记 path/lines/
# code/elide。当一条摘录本身就含内部跳跃（elide 非空、code 里已有 …/... 省略号、或
# lines 声明了 2 段以上非连续区间），意味着 writer 把这条摘录嵌进正文时也必须让读者
# 看得见"这里跳过了一段"——否则读者对照真实源码会发现正文悄悄抽掉了内容却毫无提示。
# 这里只做「dossier 已登记为有省略 → 正文对应代码段是否可见省略标记」的核对，不猜测
# dossier 之外的裁剪（那类判断噪音太大，留给人工评审）。
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")
_LNUM_RE = re.compile(r"L(\d+)")
_FENCE_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.S)


def _dossier_embed_excerpts(chapter_dir: Path):
    """解析 dossier.embed_excerpts，为每条摘录标出「path/起始行号/最大行号/是否已登记省略」。"""
    p = chapter_dir / "dossier" / "dossier.json"
    if not p.exists():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return []
    out = []
    for e in doc.get("embed_excerpts") or []:
        if not isinstance(e, dict):
            continue
        path = e.get("path")
        lines_field = e.get("lines") or ""
        nums = [int(n) for n in _LNUM_RE.findall(lines_field)]
        if not path or not nums:
            continue  # lines 字段不含可解析行号（如论文 §/Eq 引用、纯文字摘要）——跳过，避免误判
        code = e.get("code") or ""
        # 注意：`elide` 字段在真实语料里常用来说明"这段代码展示了但正文不展开讨论的旁支"，
        # 并不总是意味着"这里有一段代码被裁掉了"（analyst 实践里两种用法混用）。只有
        # code 字段本身已含省略号、或 lines 声明了 2 段以上不连续区间，才是"确有裁剪"
        # 的可靠信号——否则对 elide 字段过度敏感会在既有章节上大量误报（已实测 ch01/
        # ch20 等章节的 elide 均属"旁支说明"而非"内容裁剪"）。
        has_documented_gap = bool(_ELLIPSIS_RE.search(code)) or len(nums) > 2
        out.append({
            "path": path,
            "first_la": nums[0],
            "max_l": max(nums),
            "has_documented_gap": has_documented_gap,
        })
    return out


def _check_elision(chapter_dir: Path, narrative_text: str):
    """两条检查：
    1) elision_gap —— dossier 登记该摘录含省略，但正文对应代码段看不到省略标记。
    2) non_adjacent_splice —— 同一代码块内相邻两条摘录的登记区间并不相邻，
       但块内看不到省略标记（读者会误以为两段代码在源码里紧挨着）。
    仅在能按 (path, 起始行号) 精确匹配到 dossier 条目时才判定，匹配不上一律跳过——
    宁可漏判，不可对无法验证的写法（如整块只有一条不带行号的说明性注释）误报。
    """
    gap_issues, splice_issues = [], []
    entries = _dossier_embed_excerpts(chapter_dir)
    if not entries:
        return gap_issues, splice_issues
    by_key = {}
    for e in entries:
        by_key.setdefault((e["path"], e["first_la"]), e)

    alt = "|".join(re.escape(p) for p in sorted(_SRC_PREFIXES, key=len, reverse=True))
    marker_re = re.compile(
        r"^#\s*(?P<path>(?:" + alt + r")/[\w./-]+\.py)(?::L(?P<la>\d+))?.*$", re.M
    )

    for block_m in _FENCE_RE.finditer(narrative_text):
        block = block_m.group(1)
        markers = list(marker_re.finditer(block))
        if not markers:
            continue
        bounds = []
        for i, mm in enumerate(markers):
            nl = block.find("\n", mm.start())
            seg_start = nl + 1 if nl != -1 else len(block)
            seg_end = markers[i + 1].start() if i + 1 < len(markers) else len(block)
            bounds.append((mm, seg_start, seg_end))

        for i, (mm, seg_start, seg_end) in enumerate(bounds):
            path = mm.group("path")
            la = int(mm.group("la")) if mm.group("la") else None
            segment_text = block[seg_start:seg_end]
            has_marker = bool(_ELLIPSIS_RE.search(segment_text))
            entry = by_key.get((path, la)) if la is not None else None

            if entry and entry["has_documented_gap"] and not has_marker:
                gap_issues.append(
                    f"  {path}:L{la} —— dossier 登记该摘录含省略(elide/多段区间)，"
                    f"正文对应代码段内未见省略标记(…/...)"
                )

            if i + 1 < len(bounds):
                nmm = bounds[i + 1][0]
                npath = nmm.group("path")
                nla = int(nmm.group("la")) if nmm.group("la") else None
                nentry = by_key.get((npath, nla)) if nla is not None else None
                if entry and nentry:
                    adjacent = path == npath and nentry["first_la"] <= entry["max_l"] + 1
                    if not adjacent and not has_marker:
                        splice_issues.append(
                            f"  {path}:L{la} 与 {npath}:L{nla} 在同一代码块内首尾拼接，"
                            f"两段登记区间不相邻(止于 L{entry['max_l']} / 起于 L{nentry['first_la']})，"
                            f"块内未见省略标记(…/...)"
                        )
    return gap_issues, splice_issues


def _spans_missing_source(pyfile: Path):
    src = pyfile.read_text(encoding="utf-8")
    lines = src.splitlines()
    out = []
    try:
        tree = ast.parse(src, filename=str(pyfile))
    except SyntaxError as e:
        return [f"  {pyfile.name}: syntax error {e}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            ctx = "\n".join(lines[max(0, start - 1):end])
            if "# SOURCE:" not in ctx:
                out.append(f"  {pyfile.name}:{node.lineno} `{node.name}` 无 # SOURCE: 引用")
    return out


def lint_fidelity(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    impl = d / "implementation"
    narrative = d / "narrative" / "chapter.md"
    res = {"missing_source": [], "invention": [], "narrative_grounding": [],
           "over_subtraction": [], "no_subtraction": [],
           "elision_gap": [], "non_adjacent_splice": []}
    # rglob：递归扫子目录——与真实源码同构的 backend/ 等子目录布局应被支持（顶层 glob 会漏判）。
    pyfiles = [p for p in impl.rglob("*.py") if p.name != "__init__.py"] if impl.exists() else []
    subtraction_seen = False
    for p in pyfiles:
        text = p.read_text(encoding="utf-8")
        res["missing_source"] += _spans_missing_source(p)
        for m in INVENTION_MARKERS:
            if m in text:
                res["invention"].append(f"  {p.name}: 禁止标记 {m}")
        if "# SUBTRACTED:" in text:
            subtraction_seen = True
    if pyfiles and not subtraction_seen:
        res["no_subtraction"].append("  无任何 # SUBTRACTED: 标记（只做减法应有删除注释）")
    if narrative.exists():
        nt = narrative.read_text(encoding="utf-8")
        vllm_refs = len(_SRC_REF_RE.findall(nt))
        comp_refs = len(re.findall(r"implementation/[\w/]+\.py", nt))
        if vllm_refs < MIN_VLLM_REFS:
            res["narrative_grounding"].append(f"  真实源码引用仅 {vllm_refs} 处（需 >= {MIN_VLLM_REFS}）")
        if comp_refs > vllm_refs:
            res["narrative_grounding"].append(
                f"  叙事引用精简版({comp_refs}) 多于真实源码({vllm_refs}) — 喧宾夺主")
        gap_issues, splice_issues = _check_elision(d, nt)
        res["elision_gap"] += gap_issues
        res["non_adjacent_splice"] += splice_issues

    # 过度删减/误删：dossier 声明的 must_keep 符号必须出现在精简版
    dossier = d / "dossier" / "dossier.json"
    if dossier.exists() and pyfiles:
        impl_text = "\n".join(p.read_text(encoding="utf-8") for p in pyfiles)
        try:
            doss = json.loads(dossier.read_text(encoding="utf-8"))
            must_keep = (doss.get("subtraction_plan") or {}).get("must_keep") or []
        except (ValueError, AttributeError):
            must_keep = []
        for entry in must_keep:
            sym = entry.get("symbol") if isinstance(entry, dict) else entry
            if not sym:
                continue
            leaf = str(sym).split(".")[-1]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", leaf) and leaf not in impl_text:
                res["over_subtraction"].append(
                    f"  must_keep 符号 `{sym}` 未出现在精简版（疑似过度删减/误删）")
    return res


def print_report(res: dict, chapter_dir: str) -> int:
    total = sum(len(v) for v in res.values())
    print(f"Fidelity Lint: {chapter_dir}\n{'=' * 60}")
    if total == 0:
        print("✓ 保真度检查全部通过！")
        return 0
    for k, issues in res.items():
        if issues:
            print(f"\n❌ {k} ({len(issues)}):")
            for i in issues:
                print(i)
    # elision_gap / non_adjacent_splice：lint-exp-002 落地时对全书语料(vllm + vllm-ascend
    # 全部章节)做过实测——按简报原意判 blocking 会在 ~15 个既有非 primer 章节上新增
    # BLOCKING（根因：dossier `elide` 字段在真实写作里常用来说明"代码已展示但正文不
    # 展开讨论的旁支"，并非总是"内容被裁掉"，启发式无法完全区分两种用法）。按
    # HARD RULE 防回归要求降级为非阻断提示，先跑几轮观察真实误报率，稳定后再考虑收紧。
    blocking = (len(res["missing_source"]) + len(res["invention"])
                + len(res["narrative_grounding"]) + len(res["over_subtraction"]))
    print(f"\n{'=' * 60}")
    print(f"🔴 {blocking} BLOCKING" if blocking else "🟢 仅警告（无 BLOCKING）")
    return 1 if blocking else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_fidelity.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_fidelity(sys.argv[1]), sys.argv[1]))
