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
The numerical trace in Cell 7 MUST be actual output from running `python3 implementation/{module}.py`.
Never fabricate output — readers will try to reproduce it and lose trust.
