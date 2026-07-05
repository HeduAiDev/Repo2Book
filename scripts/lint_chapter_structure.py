#!/usr/bin/env python3
"""Chapter-structure linter — Roadmap present + self-contained embedded source + no scaffold leakage.

Usage: python3 lint_chapter_structure.py <chapter.md>
Exit 1 if any issue (all blocking).
"""
import re
import sys
from pathlib import Path

MIN_SOURCE_BLOCKS = 2

# 内嵌「真源码」路径前缀：活动实例的规范前缀(如 vllm_ascend) + 对照基座(如 vllm) + C/C++ 源目录(csrc)。
# 与 lint_fidelity.py / lint_source_grounding.py 一致——off-spine 实例章节的真源码不只是基座 vllm/。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json as _json
_PREFIXES = ["vllm", "csrc"]
try:
    import instance as _instance
    _p = _instance.canonical_prefix()
    if _p and _p not in _PREFIXES:
        _PREFIXES.append(_p)
    try:
        _root = _json.load(open(Path(__file__).resolve().parent.parent / "repo2book.json"))
        _dep = (_root.get("instances", {}).get(_instance.active_name(), {}) or {}).get("depends_on")
        if _dep:
            _bp = _json.load(open(Path(__file__).resolve().parent.parent / "instances" / _dep / "repo2book.json")).get("source", {}).get("canonical_prefix") or _dep
            if _bp and _bp not in _PREFIXES:
                _PREFIXES.append(_bp)
    except Exception:
        pass
except Exception:
    pass
# 块内含真源码路径标注（规范前缀 + 代码后缀）即算「内嵌真源码」
_SRC_PATH_RE = re.compile(
    r"(?:" + "|".join(re.escape(p) for p in sorted(set(_PREFIXES), key=len, reverse=True)) +
    r")/[\w./-]+\.(?:py|cpp|cc|cxx|h|hpp|cu)")
_CODE_FENCE_RE = re.compile(r"```(?:python|py|cpp|c\+\+|cc|cxx|c|cuda)\b.*?```", re.S | re.I)

# ── core 机制源码层机检（lint-exp-008）─────────────────────────────────────
# 全章级"内嵌源码块数量 >= 2"通不出「逐个 difficulty=core 机制是否各自有内嵌源码」这
# 一粒度的缺口（ch33 的 expected-accepted-length/walltime-speedup 两条 core 机制全章
# 只靠一句 prose 带过参数名、从未被嵌入代码块，直到人工评审才发现）。这里逐个核对
# dossier.mechanisms 里 difficulty=core 的 source_anchors 是否与正文某代码块标注的
# 行号区间相交。
_MECH_ANCHOR_RE = re.compile(r"([\w./-]+\.(?:py|cpp|cc|cxx|h|hpp|cu)):L(\d+)-L(\d+)")
_alt_prefixes = "|".join(re.escape(p) for p in sorted(set(_PREFIXES), key=len, reverse=True))
_BLOCK_MARKER_RE = re.compile(
    r"^#\s*(?P<path>(?:" + _alt_prefixes + r")/[\w./-]+\.(?:py|cpp|cc|cxx|h|hpp|cu))"
    r"(?::L(?P<la>\d+)(?:-L?(?P<lb>\d+))?)?.*$",
    re.M,
)


def _embedded_block_ranges(text: str):
    """扫正文全部内嵌源码块，逐条 marker 解析出 (path, La, Ld) —— 块头未给出结束行号时，
    用该 marker 到下一 marker(或块尾)之间的非空行数近似 Ld，供与 dossier source_anchors
    做区间相交判定。"""
    ranges = []
    for block_m in _CODE_FENCE_RE.finditer(text):
        block = block_m.group(0)
        markers = list(_BLOCK_MARKER_RE.finditer(block))
        for i, mm in enumerate(markers):
            path = mm.group("path")
            la = mm.group("la")
            if la is None:
                continue
            la = int(la)
            lb = mm.group("lb")
            if lb is not None:
                ranges.append((path, la, int(lb)))
                continue
            nl = block.find("\n", mm.start())
            seg_start = nl + 1 if nl != -1 else len(block)
            seg_end = markers[i + 1].start() if i + 1 < len(markers) else len(block)
            segment = block[seg_start:seg_end]
            nonblank = sum(1 for ln in segment.splitlines() if ln.strip())
            ld = la + max(nonblank - 1, 0)
            ranges.append((path, la, ld))
    return ranges


def _core_mechanism_source_check(chapter_dir: Path, text: str):
    """返回 (missing, ) —— 每个 difficulty=core 机制若其全部 source_anchors 都与正文
    任何内嵌源码块区间不相交，报告该 mechanism_id。非 core 机制、无 dossier、
    无 mechanisms 字段时一律跳过（不误伤）。"""
    issues = []
    dossier_path = chapter_dir / "dossier" / "dossier.json"
    if not dossier_path.exists():
        return issues
    try:
        doc = _json.loads(dossier_path.read_text(encoding="utf-8"))
    except ValueError:
        return issues
    core_mechs = [
        m for m in (doc.get("mechanisms") or [])
        if isinstance(m, dict) and m.get("difficulty") == "core"
    ]
    if not core_mechs:
        return issues

    block_ranges = _embedded_block_ranges(text)
    for mech in core_mechs:
        mech_id = mech.get("id", "?")
        anchors = mech.get("source_anchors") or []
        parsed_anchors = []
        for a in anchors:
            am = _MECH_ANCHOR_RE.match(a.strip()) if isinstance(a, str) else None
            if am:
                parsed_anchors.append((am.group(1), int(am.group(2)), int(am.group(3))))
        if not parsed_anchors:
            continue  # source_anchors 格式不可解析（如非 path:La-Lb），跳过避免误报
        hit = any(
            apath == bpath and max(ala, bla) <= min(alb, bld)
            for apath, ala, alb in parsed_anchors
            for bpath, bla, bld in block_ranges
        )
        if not hit:
            anchors_str = ", ".join(anchors)
            issues.append(
                f"  difficulty=core 机制 `{mech_id}` 的 source_anchors({anchors_str}) "
                f"未与正文任何内嵌源码块的标注区间相交——三层（直觉/机制/源码）里缺源码层"
            )
    return issues


def lint_structure(md_path: str) -> dict:
    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")
    res = {"no_roadmap": [], "no_embedded_source": [], "scaffold_leak": [], "halfwidth_punct": [],
           "core_mechanism_missing_source": []}

    head = "\n".join(text.splitlines()[:60])
    if not re.search(r"(roadmap|路线图|你在这里)", head, re.I):
        res["no_roadmap"].append("  开头 60 行内无 Roadmap/路线图/你在这里 段")

    blocks = _CODE_FENCE_RE.findall(text)
    embedded = [b for b in blocks if _SRC_PATH_RE.search(b)]
    if len(embedded) < MIN_SOURCE_BLOCKS:
        res["no_embedded_source"].append(
            f"  内嵌真源码块仅 {len(embedded)}（需 >= {MIN_SOURCE_BLOCKS}，块内含规范源码路径标注，如 "
            f"{'/'.join(sorted(set(_PREFIXES)))}/…）")

    # 零脚手架泄漏（读者视角）：正文不得含本仓库脚手架痕迹
    scaffold = [
        (r"instances/[\w.-]+/source", "出现脚手架路径 instances/<instance>/source（应用规范源码路径，如 vllm/…）"),
        (r"\bCell\s*\d+\b", "出现 'Cell N' 脚手架标题（应用自然标题）"),
        (r"impl-notes\.md|dossier", "引用内部脚手架文件（impl-notes.md/dossier）"),
        (r"must_keep|subtraction_plan|embed_excerpt", "引用内部 dossier 机制术语（must_keep/subtraction_plan/embed_excerpt——读者视角不该出现）"),
        (r"详[见细]文档|完整文档见|这里只?截取", "提到出版物中不存在的外部文档/截取说明"),
    ]
    for pat, msg in scaffold:
        if re.search(pat, text):
            res["scaffold_leak"].append(f"  {msg}")

    # 中文之间误用半角逗号（应全角 '，'）；排除代码块
    no_code = re.sub(r'```.*?```', '', text, flags=re.S)
    for mm in re.finditer(r'[一-鿿],', no_code):
        ctx = no_code[max(0, mm.start() - 6):mm.start() + 2].replace('\n', ' ')
        res["halfwidth_punct"].append(f"  中文后误用半角逗号（应 '，'）：…{ctx}…")

    # core 机制源码层机检（lint-exp-008）：向后兼容旧调用点——入参仍是单个
    # chapter.md 文件路径（如 `{chapter}/narrative/chapter.md`），chapter_dir 从其
    # 父目录的父目录推导，无 dossier/ 时静默跳过（不影响旧调用行为）。
    chapter_dir = md_path.resolve().parent.parent
    res["core_mechanism_missing_source"] += _core_mechanism_source_check(chapter_dir, text)
    return res


# 全部既有检查历来即"有则阻断"。新增的 core_mechanism_missing_source（lint-exp-008）
# 判据虽是确定性区间相交运算，但落地实测发现既有章节命中：ch31/ch32 的 core 机制用
# `# path:L721,L729,L731` 逗号列表式 marker 或纯 prose 引用锚点区间——本检查的 marker
# 区间解析对这类写法不完全（会漏认），按"既有章节不得新增 BLOCKING"的防回归硬规则
# 降级为 warn（⚠️，不计入返回码）；待 marker 解析覆盖逗号列表等写法、并跑几轮观察
# 误报率后再考虑升为 blocking。
_BLOCKING_KEYS = (
    "no_roadmap", "no_embedded_source", "scaffold_leak", "halfwidth_punct",
)


def print_report(res: dict, path: str) -> int:
    total = sum(len(v) for v in res.values())
    print(f"Chapter-Structure Lint: {path}\n{'=' * 60}")
    if total == 0:
        print("✓ 结构检查通过（Roadmap + 自包含源码 + 零脚手架泄漏）")
        return 0
    for k, issues in res.items():
        mark = "❌" if k in _BLOCKING_KEYS else "⚠️"
        for i in issues:
            print(f"{mark} {k}: {i}")
    blocking = sum(len(res.get(k, [])) for k in _BLOCKING_KEYS)
    return 1 if blocking else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_chapter_structure.py <chapter.md>")
        sys.exit(1)
    sys.exit(print_report(lint_structure(sys.argv[1]), sys.argv[1]))
