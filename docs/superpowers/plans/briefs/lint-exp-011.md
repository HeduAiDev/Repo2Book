# SDD 任务简报 lint-exp-011 — lint_formulas:单符号/简单变量 inline 不计入密度告警

> 来源：`instances/vllm-ascend/book/retro/retro-2026-07-05.json` candidate `exp-undated-011`（Lead 已批准，本简报为落地前的 SDD 任务定义，不改代码）。

## 背景 / 现象

`scripts/lint_formulas.py` 的 **Check 6（`too_many_inline_formulas`）** 按"段落内
`$...$` 计数 ≥3 即报"的启发式，对数学密集章节（primer 原理章尤甚）产生大量噪音：

- ch26：报 11 处"单段内联公式过多"，人工核查全部是单符号/简单变量、合规；
- ch31：报 29 处密度警告，逐个核查均为单符号（如 `$W^{UK}$`、`$\delta$`）合规。

**代码现状核实**：该检查（`results["too_many_inline_formulas"]`）**已经是非阻断**——
`print_report` 的 `blocking` 汇总（第 148-154 行）本就不包含 `too_many_inline_formulas`，
只统计 `text_instead_of_mathrm`/`boxed_requires_amsmath`/`tag_requires_amsmath`/
`frac_in_inline_math`/`block_math_on_separate_lines` 五项。**问题不在"是否阻断"，而在**
"reviewer 仍要逐条人工排噪"——非阻断告警一样占用评审注意力（`formula-structure` 维度需要
逐条过一遍才能判定"无 blocking"，噪音多则拖慢评审）。

## 规则描述（拟）

改进 Check 6 的密度启发式：段落内 inline 公式计数**只统计"非单符号、非简单变量"的 inline
公式**，即：

- 判定为"单符号/简单变量"（不计入密度分母）的模式包括：单个希腊字母/拉丁字母（可带上下标，如
  `$\delta$`、`$W^{UK}$`、`$L_{kv}$`、`$\alpha_i$`）、单个数字、单个已知记号变量引用（无运算符、
  无多个变量组合）；
- 判定为"复杂 inline 公式"（计入密度分母）的模式：含 `\frac`、含 `=`/`\sum`/`\prod`/`\int` 等
  运算符、或字符长度超过某阈值（可复用现有 Check 7 的 30 字符阈值作为参考基线）、或含 ≥2 个
  独立变量的组合表达式；
- 段落内"复杂 inline 公式"计数 ≥3 才触发 `too_many_inline_formulas`。

## blocking / warn 定级建议

- 沿用现状：`too_many_inline_formulas` 继续为 **非阻断（warn）**——本次只收紧触发条件、降低
  噪音，不提升定级。真正应 blocking 的是既有的 5 项（`\text{}`/`\boxed{}`/`\tag{}`/inline
  `\frac`/`$$` 同行内容），这些判据本身是确定性的，无需改动。

## 测试用例草案（参照 `scripts/tests/test_lint_formulas.py` 风格；当前无此测试文件，需新建）

```python
# 新建 scripts/tests/test_lint_formulas.py
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_formulas import lint_formulas

def test_single_symbol_inline_not_counted(tmp_path):
    """一段话里 5 个 $\\delta$/$W^{UK}$/$L_{kv}$ 等单符号 inline → 不触发 too_many_inline_formulas。"""
    p = tmp_path / "chapter.md"
    p.write_text(
        "本节讨论 $\\delta$、$W^{UK}$、$L_{kv}$、$\\alpha_i$、$\\beta$ 五个记号的关系。\n",
        encoding="utf-8")
    res = lint_formulas(str(p))
    assert not res["too_many_inline_formulas"]

def test_complex_inline_formulas_still_flagged(tmp_path):
    """一段话里 3+ 个含运算符/组合表达式的复杂 inline 公式 → 仍应触发。"""
    p = tmp_path / "chapter.md"
    p.write_text(
        "有 $\\frac{a}{b}=c$，也有 $x+y=z$，还有 $\\sum_i p_i q_i$ 这几种表达式混在一段话里。\n",
        encoding="utf-8")
    res = lint_formulas(str(p))
    assert res["too_many_inline_formulas"]

def test_mixed_paragraph_only_complex_ones_counted(tmp_path):
    """混合段落：2 个单符号 + 3 个复杂公式 → 按复杂公式数 3 触发（不是总数 5）。"""
    ...

def test_too_many_inline_formulas_remains_non_blocking(tmp_path):
    """即便触发 too_many_inline_formulas，blocking 汇总仍不应包含它（回归防升级为阻断）。"""
    ...
```

## 提醒

请 Lead 走 TDD 小任务落地。这是本轮 6 条 linter 简报里改动范围最小的一条（只动 Check 6 的一个
正则/分类函数），建议优先安排，可用 ch26/ch31 的正文原文直接做真实回归用例。
