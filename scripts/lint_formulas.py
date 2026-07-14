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
import unicodedata
from pathlib import Path

# ── CommonMark「flanking」判定(Check 12/13 共用)───────────────────────────────
# 关键事实:CJK 汉字既不是 Unicode 空白、也不是 Unicode 标点,而全角标点(，。：「」——)
# 是标点。定界符两侧一个是汉字、一个是全角标点时,flanking 条件不成立 → GitHub 原样吐出
# 字面 ** / $,连带其中的数学一起不渲染。(2026-07-13 经 GitHub markdown API 实测标定)
_ASCII_PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def _is_punct(ch: str) -> bool:
    if ch is None:
        return False
    return ch in _ASCII_PUNCT or unicodedata.category(ch).startswith('P')


def _is_ws(ch: str) -> bool:
    return ch is None or ch.isspace()


def _prev_ch(s: str, i: int):
    return s[i - 1] if i > 0 else None      # 行首视作空白


def _next_ch(s: str, i: int, n: int):
    return s[i + n] if i + n < len(s) else None   # 行尾视作空白


def _left_flanking(s: str, i: int, n: int) -> bool:
    p, q = _prev_ch(s, i), _next_ch(s, i, n)
    if _is_ws(q):
        return False
    return (not _is_punct(q)) or _is_ws(p) or _is_punct(p)


def _right_flanking(s: str, i: int, n: int) -> bool:
    p, q = _prev_ch(s, i), _next_ch(s, i, n)
    if _is_ws(p):
        return False
    return (not _is_punct(p)) or _is_ws(q) or _is_punct(q)


def _can_open(s: str, i: int, n: int) -> bool:
    """`*` 系定界符能否作为开定界符(强调/加粗)。"""
    return _left_flanking(s, i, n)


def _can_close(s: str, i: int, n: int) -> bool:
    """`*` 系定界符能否作为闭定界符。"""
    return _right_flanking(s, i, n)


def _underscore_can_open(s: str, i: int) -> bool:
    """`_` 的开定界符规则比 `*` 严:还须「非 right-flanking 或前接标点」。"""
    if not _left_flanking(s, i, 1):
        return False
    return (not _right_flanking(s, i, 1)) or _is_punct(_prev_ch(s, i))


def _has_emphasis_openable_underscore(content: str) -> bool:
    """数学内容里是否存在能开 <em> 的 `_`(典型:`}_{` / `]_{` —— 前后皆标点)。"""
    for m in re.finditer(r'_', content):
        if _underscore_can_open(content, m.start()):
            return True
    return False


_GH_MATH_ESCAPE = re.compile(r'\$`[^`]*`\$')     # GitHub 官方行内数学转义,整体屏蔽


def _mask_line(line: str) -> str:
    """把 $`…`$ 转义数学与普通 code span 抹成等长空白,再做 $ 配对/flanking 判定。"""
    masked = _GH_MATH_ESCAPE.sub(lambda m: " " * len(m.group(0)), line)
    return re.sub(r'`[^`]*`', lambda m: " " * len(m.group(0)), masked)


def _inline_math_spans(line: str):
    """按顺序配对 $,跳过 code span 与 $`…`$。产出 (open_idx, close_idx, content)。"""
    masked = _mask_line(line)
    pos = [m.start() for m in re.finditer(r'(?<!\\)(?<!\$)\$(?!\$)', masked)]
    for a, b in zip(pos[0::2], pos[1::2]):
        yield a, b, line[a + 1:b]

# ── 单符号/简单变量 inline 公式判定（lint-exp-011）──
# 判定为"简单"（不计入 too_many_inline_formulas 的密度分母）：单个希腊/拉丁字母，
# 可带上下标（如 \delta、W^{UK}、L_{kv}、\alpha_i），或单个数字。
# 含运算符（=、\frac、\sum、\prod、\int 等）或组合表达式一律视为"复杂"。
_SIMPLE_INLINE_RE = re.compile(
    r'^\\?[A-Za-z]+(?:_\{[^{}]*\}|_[A-Za-z0-9]+|\^\{[^{}]*\}|\^[A-Za-z0-9]+)*$'
)
_SIMPLE_NUMBER_RE = re.compile(r'^-?\d+(?:\.\d+)?$')


def _unwrap(content: str) -> str:
    """行内数学一律写 GitHub 转义 $`…`$;各项检查看到的 content 可能带一层反引号,剥掉。"""
    c = content.strip()
    if len(c) >= 2 and c[0] == "`" and c[-1] == "`":
        return c[1:-1].strip()
    return c


def _is_simple_inline_formula(content: str) -> bool:
    """单符号/简单变量 → True（不计入密度）；复杂表达式 → False（计入密度）。"""
    c = _unwrap(content)
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
            content = _unwrap(m.group(1))
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
        masked = _mask_line(line)
        for m in inline_re.finditer(masked):
            prev = line[m.start() - 1] if m.start() > 0 else " "
            nxt = line[m.end()] if m.end() < len(line) else " "
            if (prev != " " and (ord(prev) > 127 or prev.isalnum())) or \
               (nxt != " " and (ord(nxt) > 127 or nxt.isalnum())):
                issues.append(
                    f"  Line {i}: inline math \"{line[m.start():m.end()][:40]}\" 紧邻 CJK/全角/字母数字"
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
            # GitHub 官方行内数学转义 $`…`$ —— 是数学,不是 code span,放行(见 Check 13)
            before = line[sp.start() - 1] if sp.start() > 0 else ""
            after = line[sp.end()] if sp.end() < len(line) else ""
            if before == "$" and after == "$":
                continue
            if _latex_cmd.search(sp.group(1)):
                issues.append(
                    f"  Line {i}: LaTeX 命令写在反引号 code span 里 → `{sp.group(1)[:45]}` "
                    f"—— 不渲染,显示裸源码;数学记号请用 $...$")
    results["latex_in_code_span"] = issues

    # ── Check 11: 行内 $…$ 内侧带空格($ x $) —— GitHub 不渲染 ──
    # cmark-gfm 要求开定界符后不紧跟空白、闭定界符前不带空白;`$ h $` 两头都犯 → 整段
    # 显示原始 `$ h $`。注意:空格该留在 $ 的**外侧**(与相邻 CJK 之间),不是内侧。
    # (2026-07-13 用户发现:某章 183 处 `$ x $`;Check 9 只查外侧,漏了内侧。)
    issues = []
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
        for a, b, content in _inline_math_spans(line):
            if content and (content[0].isspace() or content[-1].isspace()):
                issues.append(
                    f"  Line {i}: 行内数学内侧带空格 → \"{line[a:b + 1][:40]}\" —— "
                    f"GitHub 不渲染;空格应在 $ 的外侧(与相邻 CJK 之间),不是内侧")
    results["inner_space_in_inline_math"] = issues

    # ── Check 12: **粗体** 定界符被 CJK/全角标点卡住 flanking —— 整段吐字面 ** ──
    # CommonMark:开定界符须 left-flanking、闭定界符须 right-flanking。CJK 汉字既不是
    # 空白也不是标点,于是「是**「编译」…**」的开定界符(前接汉字、后接全角括号)不成立,
    # 「…读:**第」的闭定界符(前接全角冒号、后接汉字)也不成立 → ** 原样显示,**且其中
    # 的 $…$ 数学一并陪葬**。修法:在 ** 外侧补半角空格(与 $ 同一条「空格在外侧」规则)。
    # (2026-07-13 用 GitHub markdown API 实测确认;详见 experience-ledger exp-0713-1)
    issues = []
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
        # masked 只用来「定位」真正的 ** 运行(排除 code span 内的);flanking 判定必须回到
        # 原始 line —— 用空白掩码算 flanking 会把 `code`**x** 里紧邻反引号的 ** 误判成
        # 「后接空白」(2026-07-13 自查抓出 406 处假阳)。掩码等长,下标可直接复用。
        masked = _mask_line(line)
        runs = [m for m in re.finditer(r'(?<!\*)\*\*(?!\*)', masked)]
        for opener, closer in zip(runs[0::2], runs[1::2]):
            if not _can_open(line, opener.start(), 2):
                issues.append(
                    f"  Line {i}: 粗体开定界符 flanking 不成立 → "
                    f"\"{line[max(0, opener.start() - 4):opener.end() + 12]}\" —— GitHub 吐字面 **"
                    f"(其中的 $…$ 也不渲染);在 ** 外侧补半角空格")
            elif not _can_close(line, closer.start(), 2):
                issues.append(
                    f"  Line {i}: 粗体闭定界符 flanking 不成立 → "
                    f"\"{line[max(0, closer.start() - 12):closer.end() + 4]}\" —— GitHub 吐字面 **"
                    f"(其中的 $…$ 也不渲染);在 ** 外侧补半角空格")
    results["emphasis_flanking_cjk"] = issues

    # ── Check 13: 行内数学含「可开强调」的下划线 → markdown 先把 _ 吃成 <em> ──
    # `$\mathbf{q}_{t,j}$` 里的 _ 前接 `}`(标点)、后接 `{`(标点) → left-flanking 成立 →
    # 能开 <em>,与本段另一个 _ 配对后跨过 $ 定界符,数学整段不渲染。`$a_i$` 免疫(_ 前接
    # 字母,开不了)。方括号/圆括号下标(如 [\hat H^{-1}]_{jj})同理且无法靠改记号回避,
    # 故统一用 GitHub 官方转义 $`…`$ 包住(实测在正文/表格/粗体/列表中均渲染)。
    issues = []
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
        for a, b, content in _inline_math_spans(line):
            if _has_emphasis_openable_underscore(content):
                issues.append(
                    f"  Line {i}: 行内数学含可开强调的下划线 → \"{line[a:b + 1][:45]}\" —— "
                    f"markdown 会把 _ 吃成 <em>,整段数学不渲染;改用 GitHub 转义 $`…`$")
    results["markdown_hostile_inline_math"] = issues

    # ── Check 14: 块级数学禁用 $$…$$,一律 ```math 围栏 ──
    # GitHub 在数学扩展拿到内容前先做 CommonMark 反斜杠转义:$$ 块里 \, \; \! \{ \_ \\ 等
    # 「反斜杠+标点」命令全被吃掉反斜杠 → 细间距变字面逗号、\left\{ 变非法 \left{ 报
    # Missing delimiter、aligned 换行被砍(2026-07-14 用户报 \left 报错后 API 实测,
    # 全书 242 块中 139 块受影响)。```math 围栏是代码围栏语义,逐字节免疫(引用块内同样成立)。
    # 存量转换:python3 scripts/fix_display_math_fence.py <file.md>
    issues = []
    in_code = False
    for i, line in enumerate(lines, 1):
        rest = re.sub(r'^(\s*(?:>\s?)*)', '', line)
        if rest.startswith("```"):
            in_code = not in_code
            continue
        if not in_code and rest.strip() == "$$":
            issues.append(
                f"  Line {i}: 块级数学用了 $$ —— GitHub 会先吃掉块内 \\, \\; \\! \\{{ 等的反斜杠"
                f"(公式错渲/报错);改用 ```math 围栏(fix_display_math_fence.py 可批转)")
    results["display_math_dollar_block"] = issues

    # ── Check 8b: ```math 围栏体内禁 CJK(围栏被 in_fence 跳过,这里单独扫) ──
    cjk_re = re.compile(r'[一-鿿　-〿＀-￯]')
    issues = []
    in_math_fence = False
    in_other_fence = False
    for i, line in enumerate(lines, 1):
        rest = re.sub(r'^(\s*(?:>\s?)*)', '', line)
        if rest.startswith("```"):
            if in_math_fence:
                in_math_fence = False
            elif in_other_fence:
                in_other_fence = False
            elif rest.strip() == "```math":
                in_math_fence = True
            else:
                in_other_fence = True
            continue
        if in_math_fence and cjk_re.search(rest):
            issues.append(
                f"  Line {i}: ```math 围栏内含 CJK → \"{rest.strip()[:50]}\" — "
                f"strict KaTeX 拒绝;中文移出公式")
    results["cjk_in_math_fence"] = issues

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
        + len(results.get("inner_space_in_inline_math", []))
        + len(results.get("emphasis_flanking_cjk", []))
        + len(results.get("markdown_hostile_inline_math", []))
        + len(results.get("display_math_dollar_block", []))
        + len(results.get("cjk_in_math_fence", []))
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
