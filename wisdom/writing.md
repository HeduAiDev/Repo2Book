# Wisdom: Writing Patterns

Universal writing patterns for technical book chapters. These produce chapters that
are both deeply educational AND source-grounded — without either extreme.

---

## W06: Formula lint: \text → \mathrm (BLOCKING)

**Discovered by**: writer, reviewer
**Confirmed in**: vllm (all 13 chapters)
**Severity**: BLOCKING — auto-REJECT in review

### Pattern
LaTeX `\text{}` renders inconsistently across markdown renderers. Use `\mathrm{}` for
Roman text in math mode. Chinese characters in formulas should be moved OUTSIDE `$$`.

```latex
% WRONG
$$ T_{\text{static}} = 800 + 100 = 900 \text{ 步} $$

% RIGHT
$$ T_{\mathrm{static}} = 800 + 100 = 900 $$
也就是 900 步。
```

### All Blocking Formula Rules
| Issue | Fix |
|-------|-----|
| `\text{}` | `\mathrm{}` |
| `\boxed{}` | Markdown bold header above |
| `\tag*{}` | Annotation outside `$$` |
| `\frac` in inline `$` | Promote to block `$$` |
| `$$` same line | Separate lines |

### Affected Roles
- **writer**: Run `lint_formulas.py` before claiming complete
- **reviewer**: AUTO-REJECT on any blocking formula issue
- **tester**: No direct impact (but tests should verify linter runs)

---

## W07: Code walkthrough — read implementation FIRST

**Discovered by**: writer
**Confirmed in**: vllm (04-continuous-batching)
**Severity**: BLOCKING — wrong line numbers destroy reader trust

### Pattern
The Writer has been writing code walkthroughs with incorrect line numbers. The
implementation file may change after the Writer last read it (due to Implementer fixes).

### Solution
1. Read the implementation file RIGHT BEFORE writing the walkthrough
2. Reference exact line numbers from the file you just read
3. Verify: grep for the function signature to confirm line number
4. If the implementation changes, re-read and update line numbers

### Affected Roles
- **writer**: MANDATORY — read implementation before every walkthrough section
- **reviewer**: Spot-check 3 random line number references against actual file
- **implementer**: Notify writer when making changes that affect line numbers

---

## W09: 大白话 style — formality spectrum per cell

**Discovered by**: writer, reviewer
**Confirmed in**: vllm (all chapters)
**Severity**: MEDIUM

### Pattern
Each cell has a different target formality. Cell 2 (hook) should be casual/conversational
(like a friend explaining). Cell 4 (theory) should be rigorous. Cell 11 (summary) should
return to casual.

### Formality Spectrum
```
Cell 2 (Hook):            ██░░░░░  casual, 口语
Cell 3 (Problem):         ███░░░░  concrete, 例子驱动
Cell 4 (Theory):          ██████░  rigorous, 数学语言
Cell 5 (Walkthrough):     ████░░░  detailed, 逐行解释
Cell 6 (Implementation):  ████░░░  technical, 别说废话
Cell 7 (Numerical):       ███░░░░  concrete, 像在演示
Cell 9 (Source Mapping):  █████░░  precise, 对照表
Cell 10 (Verification):   ███░░░░  factual, 像报告
Cell 11 (Summary):        ██░░░░░  casual, 大白话
```

### Affected Roles
- **writer**: Match formality to cell number
- **reviewer**: Flag mismatched formality (Cell 4 in 口语, Cell 2 in 书面语)
- **book-editor**: This is in the style guide, not just the review checklist

---

## Other Writing Patterns

### Every section: Source Trail first, then Theory
Readers need to know WHERE in the code before understanding WHY. Source → Theory order
creates a physical anchor that makes abstract concepts feel concrete.

### Diagram validation before PNG
Always: `xmllint --noout → validate_svg.py → convert -density 150`. Three steps,
must all pass. A broken diagram is worse than no diagram.

### Running output must match actual code
The numerical trace MUST be actual output from running the companion `implementation/`.
Never fabricate output — readers will try to reproduce it and lose trust.

---

## v2 追加（源码解读型；取代上面旧 "Cell N / convert" 措辞）

旧条目里的 "Cell N" 章节结构与 `convert -density` 渲染均已**废弃**——v2 是**源码解读型**：正文以真实源码为主线、自包含内嵌；图用 `rsvg-convert -z 2`（**勿用 ImageMagick convert**，丢中文/错位）。以下为 v2 + v0.21.0 重基 + vllm-ascend 试点打磨出的新写作经验（明细见 INDEX W13–W21）。

### W13 内嵌真源码逐字保留（reviewer 必查）
内嵌片段是"真实源码"的承诺——**逐字保留**含完整签名与类型标注（别把 `num_heads: int | None = None` 改成 `num_heads=None`）。删无关分支用 `# … 省略：… `。把**非相邻**的方法拼进一个代码块时，中间也要加省略标注，否则读者以为它们在源码里紧挨着。

### W14 主线术语首现处一次性中文注解
全章反复用的主线术语，首现处括注一次中文译名（与 glossary 对齐），如「qualname（全限定类名）」「OOT（out-of-tree）」，后文沿用英文。

### W15 图：一图一核心对比，几何要干净
`lint_diagram_geometry` 校验文字不越界/不相撞/不压框/不裁切、箭头接框边。信息密度别过高：核心对比留图里，细节下沉正文。

### W16 锚点与标点（确定性 lint）
章内回指 `#` 锚点须解析到本章标题的 GitHub-slug（`lint_anchors`）；中文句子标点全角（`lint_punct`）。

### W21 姊妹篇/衍生仓的写法
解读"插件/衍生仓"（如 vllm-ascend 之于 vLLM）：主线讲衍生仓源码（`vllm_ascend/…`），**每章钉一个对位基座章**，对照基座源码（已在 `instances/<base>/source`，直引规范 `vllm/…`）说清"它顶替/扩展了哪一站"。两边都用规范路径。
