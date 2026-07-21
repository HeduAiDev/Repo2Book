"""引文里的**裸 `…` 行**是省略标记,不该被 citation_range 当成"源码里找不到的行"(exp-2026-07-20-07)。

来源:ch06 引 vec_ops.py 的两段函数时,把 docstring 中段省略成单独一行 `    …`。
`_NOTE_RE` 只认 `#`/`//` 注释里的省略号,裸 `…` 行漏网 → citation_range 报
"第 N 行 '…' 在该区间内按序找不到"。这类噪音正是我把 citation_range 降为 warn 的原因,
而 warn 又让一处**真的**篡改(ch06 把 gather.py 的 `* K + k_offs` 改成 `* N + n_offs`)从眼皮下溜过。
降噪 = 为将来把该检查升级成 blocking 铺路。

只豁免 `…`(U+2026),**不**豁免裸 `...`——后者是合法 Python(Ellipsis,`def f(): ...`),
豁免它可能掩盖真实不符。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_fidelity import _check_citation_ranges

SRC = ("def alpha(x):\n"        # L1
       '    """doc line A\n'    # L2
       "    doc line B\n"       # L3
       "    doc line C\n"       # L4
       '    """\n'              # L5
       "    return x + 1\n")    # L6


def _mk(tmp, narrative, src=SRC):
    inst = tmp / "instances" / "demo"
    ch = inst / "artifacts" / "ch01"
    (ch / "narrative").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    (inst / "source" / "pkg").mkdir(parents=True)
    (inst / "source" / "pkg" / "m.py").write_text(src, encoding="utf-8")
    return str(ch), narrative


def test_bare_ellipsis_line_is_treated_as_elision(tmp_path):
    """docstring 中段用裸 `…` 省略 → 不该报。"""
    nar = ('```python\n# pkg/m.py:L1-L6\n'
           'def alpha(x):\n    """doc line A\n    …\n    """\n    return x + 1\n```\n')
    ch, nar_text = _mk(tmp_path, nar)
    issues = _check_citation_ranges(ch, nar_text)
    assert issues == [], issues


def test_real_tampering_still_caught(tmp_path):
    """把源码改掉一行(x+1 → x+2)仍必须报——降噪不得放宽真篡改。"""
    nar = ('```python\n# pkg/m.py:L1-L6\n'
           'def alpha(x):\n    """doc line A\n    …\n    """\n    return x + 2\n```\n')
    ch, nar_text = _mk(tmp_path, nar)
    issues = _check_citation_ranges(ch, nar_text)
    assert issues, "篡改行未被抓出"
    assert "x + 2" in str(issues)


def test_bare_python_ellipsis_not_exempted(tmp_path):
    """裸 `...` 是合法 Python(Ellipsis),不豁免:源码没有该行时仍要报。"""
    nar = ('```python\n# pkg/m.py:L1-L6\n'
           'def alpha(x):\n    ...\n    return x + 1\n```\n')
    ch, nar_text = _mk(tmp_path, nar)
    issues = _check_citation_ranges(ch, nar_text)
    assert issues, "裸 ... 被当成省略标记豁免了,可能掩盖真实不符"
