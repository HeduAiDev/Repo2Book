#!/usr/bin/env python3
"""论文根基 linter — primer(原理章)的替代门禁,与 lint_fidelity 成对:
primer 章豁免 subtract-only,但参考实现与推导必须锚定论文。

启用条件:dossier/dossier.json 顶层 "kind":"primer"。非 primer 章一切为空、exit 0。

阻断项:implementation/*.py 有 def/class 缺 `# PAPER:` 锚(定义行上下 3 行内);
        narrative 无任何 arXiv id(推导无出处)。
警告项:某 `$$` 公式块 ±10 行内无引用锚(§/Eq/式/arXiv);
        dossier paper_origin.sections 的小节号在论文包 paper.md 里 grep 不到。
用法:python3 lint_paper_grounding.py <chapter_dir>   阻断项存在则 exit 1。
"""
import json
import re
import sys
from pathlib import Path

ARXIV = re.compile(r'arXiv[:\s/]*(\d{4}\.\d{4,5})', re.I)
ANCHOR = re.compile(r'§|Eq\.?|arXiv|式\s*\(|PAPER', re.I)
DEF = re.compile(r'^\s*(?:def|class)\s+(\w+)')

# ── symbol_context: primer 公式符号裸奔检查 ──
GREEK_UNICODE = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "π", "rho": "ρ", "varrho": "ρ", "sigma": "σ",
    "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
SYMBOL_WHITELIST = {"max", "min", "exp", "log", "softmax", "argmax", "mathrm"}
RE_MATHRM_WORD = re.compile(r'\\mathrm\{(\w+)\}')
RE_CMD_WORD = re.compile(r'\\([a-zA-Z]+)')
RE_ISOLATED_LETTER = re.compile(r'(?<![A-Za-z])([A-Za-z])(?:_\{?([A-Za-z0-9]+)\}?)?(?![A-Za-z])')
RE_MULTI_EQ_SEP = re.compile(r'\\qquad|\\quad')


def _extract_symbol_candidates(zone: str):
    """从一段(等号左侧或整行)LaTeX 文本里提取符号候选:(kind, display, checkword)。"""
    cands = []

    def _sub_mathrm(m):
        word = m.group(1)
        if word.lower() not in SYMBOL_WHITELIST and not word.isdigit():
            cands.append(("mathrm", f"\\mathrm{{{word}}}", word))
        return " " * len(m.group(0))

    zone = RE_MATHRM_WORD.sub(_sub_mathrm, zone)

    def _sub_cmd(m):
        word = m.group(1)
        if word in GREEK_UNICODE:
            cands.append(("greek", "\\" + word, word))
        return " " * len(m.group(0))

    zone = RE_CMD_WORD.sub(_sub_cmd, zone)

    for m in RE_ISOLATED_LETTER.finditer(zone):
        base = m.group(1)
        if base.lower() in SYMBOL_WHITELIST:
            continue
        cands.append(("letter", base, base))
    return cands


def _bare_word_present(word: str, text: str) -> bool:
    return re.search(r'(?<![A-Za-z0-9_])' + re.escape(word) + r'(?![A-Za-z0-9_])', text) is not None


def _symbol_mentioned(kind: str, checkword: str, text: str) -> bool:
    if kind == "greek":
        if ("\\" + checkword) in text:
            return True
        uni = GREEK_UNICODE.get(checkword)
        if uni and uni in text:
            return True
        return _bare_word_present(checkword, text)
    if kind == "mathrm":
        if ("\\mathrm{" + checkword + "}") in text:
            return True
        return _bare_word_present(checkword, text)
    return _bare_word_present(checkword, text)


def _nearby_prose(lines, idx: int, before: bool, limit: int = 3) -> str:
    """从 idx(某 $$ 分隔行)向外收集最多 limit 个非空行(prose 或表格皆可)。"""
    collected = []
    i, step = (idx - 1, -1) if before else (idx + 1, 1)
    while 0 <= i < len(lines) and len(collected) < limit:
        if lines[i].strip():
            collected.append(lines[i])
        i += step
    return "\n".join(collected)


# ── key_figure_missing: key_figures ↔ 重绘图注对应 ──
IMG_CAPTION = re.compile(r'!\[([^\]]*)\]\(')
FIG_NUM_IN_TEXT = re.compile(r'[Ff]ig\.?\s*(\d+)')


def _is_redraw_caption(caption: str) -> bool:
    return "重绘自" in caption or ("按" in caption and "描述重绘" in caption)


def _redrawn_fig_numbers(text: str) -> set:
    nums = set()
    for m in IMG_CAPTION.finditer(text):
        caption = m.group(1)
        if _is_redraw_caption(caption):
            # findall(非 search):图注可能「重绘自…Fig.2 与 Fig.3 合并」同时提及多个 Fig 号,
            # 全收——只取首个会把后面被合并的原图漏判成"未重绘"孤儿。
            for num in FIG_NUM_IN_TEXT.findall(caption):
                nums.add(num)
    return nums


def _fig_number(fig_field) -> str:
    if not isinstance(fig_field, str):
        return None
    m = re.search(r'(\d+)', fig_field)
    return m.group(1) if m else None


def lint_paper_grounding(chapter_dir: str, expect_primer: bool = False) -> dict:
    d = Path(chapter_dir)
    res = {
        "impl": [], "citation": [], "formula": [], "paper_ref": [], "warn": [], "expect": [],
        "symbol_context": [], "key_figure_missing": [],
    }
    df = d / "dossier" / "dossier.json"
    try:
        doc = json.loads(df.read_text(encoding="utf-8")) if df.exists() else {}
    except ValueError:
        doc = {}
    if doc.get("kind") != "primer":
        if expect_primer:
            res["expect"].append(
                '  期望 primer 章但 dossier 顶层缺 "kind":"primer"(lint 分流开关)——analyst 须补写'
            )
        res["warn"].append("  非 primer 章(dossier 顶层无 kind:primer)——本检查跳过")
        return res

    # 1) 参考实现每个 def/class 有 # PAPER: 锚(定义行上 3 行或下 3 行内)
    for py in sorted((d / "implementation").glob("*.py")) if (d / "implementation").exists() else []:
        lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, ln in enumerate(lines):
            m = DEF.match(ln)
            if not m:
                continue
            window = lines[max(0, i - 3): i + 4]
            if not any("# PAPER:" in w for w in window):
                res["impl"].append(f"  {py.name}:{i+1} {m.group(1)} 缺 `# PAPER: §x Eq.y` 锚")

    # 2) 正文:必须有 arXiv id;每个 $$ 块 ±10 行内应有引用锚
    nar = d / "narrative" / "chapter.md"
    text = None
    if nar.exists():
        text = nar.read_text(encoding="utf-8")
        if not ARXIV.search(text):
            res["citation"].append("  正文无任何 arXiv id——推导必须给论文出处")
        lines = text.splitlines()
        starts = [i for i, ln in enumerate(lines) if ln.strip() == "$$"]
        for a, b in zip(starts[0::2], starts[1::2]):
            lo, hi = max(0, a - 10), min(len(lines), b + 11)
            ctx = "\n".join(lines[lo:a] + lines[b + 1:hi])
            if not ANCHOR.search(ctx):
                res["formula"].append(f"  L{a+1} 公式块 ±10 行内无引用锚(§/Eq/arXiv)")

        # 2b) symbol_context(WARN):每个 $$ 块「引入」的符号——全式扫描(等号两侧都提取候选,
        #     不再只取 LHS),真实论文公式的 RHS 常定义新符号(如 softmax 分母里的算子基名、
        #     复合表达式的分母变量),只看 LHS 会系统性漏检这些符号的裸奔。
        #     须在首现公式块 ±3 个非空行(prose 或表格)内被提及,否则视为裸奔
        table_blob = "\n".join(ln for ln in lines if ln.strip().startswith("|"))
        seen_symbols = set()
        for a, b in zip(starts[0::2], starts[1::2]):
            for line in lines[a + 1:b]:
                for seg in RE_MULTI_EQ_SEP.split(line):
                    for kind, display, checkword in _extract_symbol_candidates(seg):
                        key = (kind, checkword)
                        if key in seen_symbols:
                            continue
                        seen_symbols.add(key)
                        evidence = "\n".join([
                            _nearby_prose(lines, a, before=True),
                            _nearby_prose(lines, b, before=False),
                            table_blob,
                        ])
                        if not _symbol_mentioned(kind, checkword, evidence):
                            res["symbol_context"].append(
                                f"  L{a+1} 符号 {display} 首现公式块 ±3 行内无提及(prose/表格均无)"
                            )
    else:
        res["warn"].append("  narrative/chapter.md 尚不存在(写作前跑属正常)")

    # 3) dossier paper_origin.sections 可在论文包里找到(WARNING)
    #    双论文 primer 章可能有 paper.md + paper-dsa.md 等多份,拼接全部 *.md 再 grep
    inst_book = d.resolve().parent.parent / "book"
    pack_dir = inst_book / "papers" / d.resolve().name
    pack_dir_missing = not pack_dir.exists()
    pack_files = sorted(pack_dir.glob("*.md")) if pack_dir.exists() else []
    ptext = (
        "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in pack_files)
        if pack_files
        else None
    )
    if ptext is None:
        res["warn"].append(f"  论文包缺失:{pack_dir}(发车前应先落盘)")
    else:
        for mech in doc.get("mechanisms", []):
            po = mech.get("paper_origin")
            if not isinstance(po, dict):
                continue
            for s in po.get("sections") or []:
                key = s.replace("§", "").replace("Eq.", "").strip()
                if key and key not in ptext:
                    res["paper_ref"].append(f"  {mech.get('id')}: 小节 {s} 在论文包里找不到")

    # 4) key_figures ↔ 重绘图注对应(BLOCKING):论文包 meta.json 策展的每张关键图
    #    须在正文找到「重绘自…Fig.N」/「按…Fig.N 描述重绘」的图注;反向孤儿图注同样报错。
    #    meta.json 无 key_figures 字段 → WARN(策展缺口),不阻断。
    #    整个论文包目录都不存在时,上面已记过一条"论文包缺失"warn——
    #    此时 meta.json 必然也不存在,不再叠加同根因的第二条"缺 key_figures"warn。
    if text is not None:
        meta_path = pack_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                meta = {}
        raw_kfs = meta.get("key_figures")
        if raw_kfs is None and not pack_dir_missing:
            res["warn"].append(f"  论文包 meta.json 缺 key_figures 字段(策展缺口):{meta_path}")
        kfs = raw_kfs or []
        redrawn = _redrawn_fig_numbers(text)
        registered = set()
        for kf in kfs:
            if not isinstance(kf, dict):
                continue
            num = _fig_number(kf.get("fig"))
            if not num:
                continue
            registered.add(num)
            if num not in redrawn:
                res["key_figure_missing"].append(
                    f"  key_figures {kf.get('fig')} 缺章内对应图注"
                    f"(须含「重绘自…Fig.{num}」或「按…Fig.{num} 描述重绘」)"
                )
        for num in sorted(redrawn - registered):
            res["key_figure_missing"].append(
                f"  图注「重绘自…Fig.{num}」未在论文包 key_figures 登记(孤儿重绘)"
            )
    return res


def print_report(res: dict, cd: str) -> int:
    print(f"Paper-Grounding Lint: {cd}\n{'=' * 60}")
    blocking = (
        len(res["impl"]) + len(res["citation"]) + len(res.get("expect", []))
        + len(res.get("key_figure_missing", []))
    )
    for k, issues in res.items():
        mark = "❌ " if k in ("impl", "citation", "expect", "key_figure_missing") else "⚠️ "
        for i in issues:
            print(mark + f"{k}: {i}")
    if blocking == 0:
        print("✓ 无 BLOCKING")
        return 0
    print(f"\n{'=' * 60}\n🔴 {blocking} BLOCKING")
    return 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    expect_primer = "--expect-primer" in argv
    argv = [a for a in argv if a != "--expect-primer"]
    if len(argv) < 1:
        print("Usage: python3 lint_paper_grounding.py <chapter_dir> [--expect-primer]")
        sys.exit(1)
    sys.exit(print_report(lint_paper_grounding(argv[0], expect_primer=expect_primer), argv[0]))
