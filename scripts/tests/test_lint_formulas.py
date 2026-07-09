import pathlib
import sys

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
    p = tmp_path / "chapter.md"
    p.write_text(
        "这段有 $\\delta$ 和 $\\beta$ 两个单符号，还有 $\\frac{a}{b}=c$、$x+y=z$、"
        "$\\sum_i p_i q_i$ 三个复杂表达式。\n",
        encoding="utf-8")
    res = lint_formulas(str(p))
    assert res["too_many_inline_formulas"]
    assert "3 inline formulas" in res["too_many_inline_formulas"][0]


def test_two_complex_formulas_not_flagged(tmp_path):
    """只有 2 个复杂公式（低于阈值 3）→ 不触发，即便掺杂了若干单符号。"""
    p = tmp_path / "chapter.md"
    p.write_text(
        "这段有 $\\delta$、$\\beta$、$W^{UK}$ 三个单符号，还有 $\\frac{a}{b}=c$、$x+y=z$ "
        "两个复杂表达式。\n",
        encoding="utf-8")
    res = lint_formulas(str(p))
    assert not res["too_many_inline_formulas"]


def test_too_many_inline_formulas_remains_non_blocking(tmp_path):
    """即便触发 too_many_inline_formulas，blocking 汇总仍不应包含它（回归防升级为阻断）。"""
    import io
    import contextlib
    from lint_formulas import print_report

    p = tmp_path / "chapter.md"
    p.write_text(
        "有 $x+y=z$，也有 $\\sum_i p_i q_i$，还有 $\\alpha+\\beta=\\gamma$ 这几种表达式混在一段话里。\n",
        encoding="utf-8")
    res = lint_formulas(str(p))
    assert res["too_many_inline_formulas"]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(res, str(p))
    out = buf.getvalue()
    assert "No blocking issues" in out or "🟢" in out


def test_cjk_in_display_math_flagged(tmp_path):
    ch = tmp_path / "chapter.md"
    ch.write_text("x\n\n$$\n\\mathrm{利用率} = 1\n$$\n", encoding="utf-8")
    from lint_formulas import lint_formulas
    res = lint_formulas(str(ch))
    assert len(res["cjk_in_math"]) == 1


def test_cjk_in_inline_math_flagged(tmp_path):
    ch = tmp_path / "chapter.md"
    ch.write_text("总量 $N_{总} = 3$ 个。\n", encoding="utf-8")
    from lint_formulas import lint_formulas
    res = lint_formulas(str(ch))
    assert len(res["cjk_in_math"]) == 1


def test_cjk_outside_math_ok(tmp_path):
    ch = tmp_path / "chapter.md"
    ch.write_text("利用率是 $u = 1$。\n\n$$\nu = \\frac{a}{b}\n$$\n", encoding="utf-8")
    from lint_formulas import lint_formulas
    res = lint_formulas(str(ch))
    assert res["cjk_in_math"] == []


def test_inline_math_cjk_adjacency_flagged(tmp_path):
    ch = tmp_path / "chapter.md"
    ch.write_text("约定：$t$ 是索引（$i=1,\\ldots,n$）。\n", encoding="utf-8")
    from lint_formulas import lint_formulas
    res = lint_formulas(str(ch))
    assert len(res["inline_math_adjacency_github"]) >= 2  # ：$t$ 与（$i$）


def test_inline_math_padded_ok(tmp_path):
    ch = tmp_path / "chapter.md"
    ch.write_text("约定： $t$ 是索引（ $i=1,\\ldots,n$ ）。\n", encoding="utf-8")
    from lint_formulas import lint_formulas
    res = lint_formulas(str(ch))
    assert res["inline_math_adjacency_github"] == []


def test_inline_adjacency_skips_code_and_display(tmp_path):
    ch = tmp_path / "chapter.md"
    ch.write_text("代码 `x=$a$，b` 不查。\n\n$$\na_{t}=1\n$$\n", encoding="utf-8")
    from lint_formulas import lint_formulas
    res = lint_formulas(str(ch))
    assert res["inline_math_adjacency_github"] == []
