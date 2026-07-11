#!/usr/bin/env python3
"""
Formula Linter — auto-detect LaTeX rendering issues in chapter narratives.

Usage:
    python scripts/lint_formulas.py artifacts/01-self-attention-fundamentals/narrative/chapter.md

Checks:
    1. \text{} in formulas → use \mathrm{} instead
    2. \boxed{} → not supported in basic renderers
    3. \tag*{} → not supported in basic renderers
    4. \frac inside inline $...$ (should be block $$...$$)
    5. $$ on same line as content (needs separate lines)
    6. Too many inline formulas per paragraph
    7. Underscores in inline math that could collide with markdown
"""

import re
import sys
from pathlib import Path

# ── 单符号/简单变量 inline 公式判定（lint-exp-011）──
# 判定为"简单"（不计入 too_many_inline_formulas 的密度分母）：单个希腊/拉丁字母，
# 可带上下标（如 \delta、W^{UK}、L_{kv}、\alpha_i），或单个数字。
# 含运算符（=、\frac、\sum、\prod、\int 等）或组合表达式一律视为"复杂"。
_SIMPLE_INLINE_RE = re.compile(
    r'^\\?[A-Za-z]+(?:_\{[^{}]*\}|_[A-Za-z0-9]+|\^\{[^{}]*\}|\^[A-Za-z0-9]+)*$'
)
_SIMPLE_NUMBER_RE = re.compile(r'^-?\d+(?:\.\d+)?$')


def _is_simple_inline_formula(content: str) -> bool:
    """单符号/简单变量 → True（不计入密度）；复杂表达式 → False（计入密度）。"""
    c = content.strip()
    if not c:
        return True
    if len(c) > 30:
        return False
    if _SIMPLE_NUMBER_RE.match(c):
        return True
    return bool(_SIMPLE_INLINE_RE.match(c))


def lint_formulas(filepath: str) -> dict:
    """Run all formula checks. Returns {check_name: [issues]}."""
    text = Path(filepath).read_text(encoding="utf-8")
    lines = text.split("\n")

    results = {}

    # ── Check 1: \text{} usage ──
    issues = []
    for i, line in enumerate(lines, 1):
        if "\\text{" in line:
            issues.append(f"  Line {i}: \\text{{...}} found — use \\mathrm{{...}} instead")
    results["text_instead_of_mathrm"] = issues

    # ── Check 2: \boxed{} ──
    issues = []
    for i, line in enumerate(lines, 1):
        if "\\boxed{" in line:
            issues.append(
                f"  Line {i}: \\boxed{{...}} found — requires amsmath, "
                f"use bold markdown header above formula instead"
            )
    results["boxed_requires_amsmath"] = issues

    # ── Check 3: \tag*{} ──
    issues = []
    for i, line in enumerate(lines, 1):
        if "\\tag{" in line or "\\tag*{" in line:
            issues.append(
                f"  Line {i}: \\tag{{}} found — requires amsmath, "
                f"put annotation outside $$ block"
            )
    results["tag_requires_amsmath"] = issues

    # ── Check 4: \frac inside inline $ ──
    issues = []
    # Find inline math spans: $...$ where ... contains \frac
    for i, line in enumerate(lines, 1):
        # Remove $$ blocks first (they're fine)
        cleaned = re.sub(r'\$\$[^$]*\$\$', '', line)
        # Find $...$ with \frac inside
        for m in re.finditer(r'\$([^$]+)\$', cleaned):
            if '\\frac' in m.group(1):
                issues.append(
                    f"  Line {i}: \\frac inside inline $...$ → "
                    f'"{m.group()[:50]}..." — promote to $$...$$ block'
                )
    results["frac_in_inline_math"] = issues

    # ── Check 5: $$ on same line as formula content ──
    issues = []
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("$$") and len(line.strip()) > 2:
            # $$ followed by content on same line
            content = line.strip()[2:].strip()
            if content and not content.startswith("$$"):
                issues.append(
                    f"  Line {i}: $$ on same line as formula content — "
                    f"move content to next line"
                )
    results["block_math_on_separate_lines"] = issues

    # ── Check 6: Inline formulas with ≥3 $...$ per paragraph ──
    issues = []
    paragraph_lines = []
    current_start = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#") or stripped.startswith("```"):
            if paragraph_lines:
                para = " ".join(paragraph_lines)
                inline_contents = re.findall(r'(?<!\$)\$(?!\$)([^$]+)\$(?!\$)', para)
                complex_contents = [c for c in inline_contents if not _is_simple_inline_formula(c)]
                inline_count = len(complex_contents)
                if inline_count >= 3:
                    issues.append(
                        f"  Lines {current_start}-{i-1}: {inline_count} inline formulas "
                        f"in one paragraph — consider promoting some to block formulas"
                    )
                paragraph_lines = []
                current_start = i + 1
        else:
            if not paragraph_lines:
                current_start = i
            # Remove $$ blocks from count
            cleaned = re.sub(r'\$\$[^$]*\$\$', '', stripped)
            paragraph_lines.append(cleaned)
    results["too_many_inline_formulas"] = issues

    # ── Check 7: Complex inline formulas (>30 chars inside $...$) ──
    issues = []
    for i, line in enumerate(lines, 1):
        cleaned = re.sub(r'\$\$[^$]*\$\$', '', line)
        for m in re.finditer(r'\$(?!\$)([^$]+)\$(?!\$)', cleaned):
            content = m.group(1)
            if len(content) > 30:
                issues.append(
                    f"  Line {i}: Complex inline formula ({len(content)} chars) → "
                    f'"{content[:60]}..." — should be block formula'
                )
    results["complex_inline_formulas"] = issues

    # ── Check 8: CJK inside math (strict KaTeX: unicodeTextInMathMode) ──
    issues = []
    cjk = re.compile(r'[一-鿿　-〿（），：；]')
    for m in re.finditer(r'^\$\$\n(.*?)\n\$\$', text, re.S | re.M):
        body = m.group(1)
        if cjk.search(body):
            ln = text[:m.start()].count("\n") + 2
            frag = next(s for s in body.split("\n") if cjk.search(s))
            issues.append(
                f"  Line {ln}: CJK inside $$ block → \"{frag.strip()[:50]}\" — "
                f"strict KaTeX rejects; move CJK annotation out of math or use symbols"
            )
    in_fence = False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        cleaned = re.sub(r'\$\$[^$]*\$\$', '', line)
        for m in re.finditer(r'(?<!\$)\$(?!\$)([^$\n]+)\$(?!\$)', cleaned):
            if cjk.search(m.group(1)):
                issues.append(
                    f"  Line {i}: CJK inside inline $...$ → \"{m.group(1)[:50]}\" — "
                    f"strict KaTeX rejects; move CJK out of math"
                )
    results["cjk_in_math"] = issues

    # ── Check 9: inline math adjacent to CJK/fullwidth/alnum (GitHub won't render) ──
    # cmark-gfm 数学扩展要求 $...$ 两侧紧邻空白/ASCII 标点；紧贴 CJK 字符、
    # 全角标点或字母数字时整段不渲染。修法：$ 定界符与相邻字符间补 ASCII 空格。
    issues = []
    inline_re = re.compile(r'(?<![\$`])\$(?!\$)([^$\n]+?)\$(?!\$)')
    in_fence = False
    in_disp = False
    for i, line in enumerate(lines, 1):
        st = line.strip()
        if st.startswith("```"):
            in_fence = not in_fence
            continue
        if st == "$$":
            in_disp = not in_disp
            continue
        if in_fence or in_disp:
            continue
        parts = line.split("`")
        for k in range(0, len(parts), 2):  # 偶数段 = 非代码 span
            seg = parts[k]
            for m in inline_re.finditer(seg):
                prev = seg[m.start() - 1] if m.start() > 0 else " "
                nxt = seg[m.end()] if m.end() < len(seg) else " "
                if (prev != " " and (ord(prev) > 127 or prev.isalnum())) or \
                   (nxt != " " and (ord(nxt) > 127 or nxt.isalnum())):
                    issues.append(
                        f"  Line {i}: inline math \"{m.group(0)[:40]}\" 紧邻 CJK/全角/字母数字"
                        f"——GitHub 不渲染；在 $ 与相邻字符间补空格"
                    )
    results["inline_math_adjacency_github"] = issues

    # ── Check 10: LaTeX 数学命令混在行内 code span(反引号)里 —— 不渲染,显示裸源码 ──
    # 数学记号应用 $...$;写成 `\mathbb{R}`/`\mid`/`\le` 这类反引号代码,读者看到的是
    # 字面反斜杠命令(2026-07-12 用户发现 ch37-dspark 35 处)。正则只匹配数学命令(不含
    # 正则转义 \d\n\t / Windows 路径 \ 等),避免误伤正常代码。
    _latex_cmd = re.compile(
        r'\\(?:in|le|ge|mid|top|times|cdot|frac|sum|prod|sqrt|mathbb|mathrm|mathcal|'
        r'alpha|beta|gamma|delta|theta|sigma|lambda|mu|nu|approx|otimes|leq|geq|neq|'
        r'forall|exists|langle|rangle|partial|nabla|infty|leftarrow|rightarrow|longrightarrow|'
        r'min|max|log|exp|lceil|rceil|lfloor|rfloor|le|ge)\b')
    _code_span = re.compile(r'`([^`]+)`')
    issues = []
    in_fence = False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for sp in _code_span.finditer(line):
            if _latex_cmd.search(sp.group(1)):
                issues.append(
                    f"  Line {i}: LaTeX 命令写在反引号 code span 里 → `{sp.group(1)[:45]}` "
                    f"—— 不渲染,显示裸源码;数学记号请用 $...$")
    results["latex_in_code_span"] = issues

    return results


def print_report(results: dict, filepath: str):
    """Pretty-print the lint results."""
    total_issues = sum(len(v) for v in results.values())
    print(f"Formula Lint: {filepath}")
    print(f"{'=' * 60}")

    if total_issues == 0:
        print("✓ All formula checks passed!")
        return 0

    for check_name, issues in results.items():
        if issues:
            label = check_name.replace("_", " ").title()
            print(f"\n❌ {label} ({len(issues)} issue(s)):")
            for issue in issues:
                print(issue)

    print(f"\n{'=' * 60}")
    print(f"Total: {total_issues} issue(s) found")
    print()

    # Severity
    blocking = (
        len(results.get("text_instead_of_mathrm", []))
        + len(results.get("boxed_requires_amsmath", []))
        + len(results.get("tag_requires_amsmath", []))
        + len(results.get("frac_in_inline_math", []))
        + len(results.get("block_math_on_separate_lines", []))
        # 真·渲染失败类(GitHub 静默不渲染整段/strict KaTeX 报错)——提升为阻断,
        # 否则章节带着不渲染的公式发布(2026-07-12 用户发现,20 处 adjacency 曾漏发)。
        + len(results.get("inline_math_adjacency_github", []))
        + len(results.get("cjk_in_math", []))
        + len(results.get("latex_in_code_span", []))
    )
    if blocking > 0:
        print(f"🔴 {blocking} BLOCKING issue(s) — auto-REJECT")
    else:
        print("🟢 No blocking issues")
    return blocking


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lint_formulas.py <chapter.md>")
        sys.exit(1)

    filepath = sys.argv[1]
    results = lint_formulas(filepath)
    blocking = print_report(results, filepath)
    sys.exit(1 if blocking else 0)
