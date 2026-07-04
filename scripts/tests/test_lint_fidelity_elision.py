import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_fidelity import lint_fidelity


def _mk(tmp, narrative, embed_excerpts, impl_files=None):
    d = tmp / "ch"
    (d / "narrative").mkdir(parents=True)
    (d / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    (d / "implementation").mkdir(parents=True)
    for name, body in (impl_files or {}).items():
        (d / "implementation" / name).write_text(body, encoding="utf-8")
    (d / "dossier").mkdir(parents=True)
    (d / "dossier" / "dossier.json").write_text(
        json.dumps({"embed_excerpts": embed_excerpts}), encoding="utf-8"
    )
    return str(d)


def test_gap_without_elision_marker_flagged(tmp_path):
    """dossier 摘录的 code 字段本身已含省略号（真正的内容跳跃，而非仅 elide 旁支说明），
    但正文对应代码段无 …/... 标记 → FAIL。"""
    excerpts = [{
        "path": "vllm/a.py", "lines": "L10-L30",
        "code": "def f():\n    pass\n...\ndef g():\n    pass\n",
        "elide": ["中间一大段与本节无关的分支被省略"],
    }]
    narrative = (
        "## 一节\n\n"
        "```python\n"
        "# vllm/a.py:L10\n"
        "def f():\n"
        "    pass\n"
        "```\n"
    )
    res = lint_fidelity(_mk(tmp_path, narrative, excerpts))
    assert res["elision_gap"], "dossier code 字段已标省略但正文无标记应报告"


def test_gap_with_elision_marker_passes(tmp_path):
    """同上跨度缺口，但代码块含 `# … 省略 …` → 不报。"""
    excerpts = [{
        "path": "vllm/a.py", "lines": "L10-L30",
        "code": "def f():\n    pass\n...\ndef g():\n    pass\n",
        "elide": ["中间一大段与本节无关的分支被省略"],
    }]
    narrative = (
        "## 一节\n\n"
        "```python\n"
        "# vllm/a.py:L10\n"
        "def f():\n"
        "    pass\n"
        "    # … 省略：与本节无关的分支 …\n"
        "```\n"
    )
    res = lint_fidelity(_mk(tmp_path, narrative, excerpts))
    assert not res["elision_gap"]


def test_elide_note_alone_without_ellipsis_not_flagged(tmp_path):
    """真实语料回归：`elide` 字段常用来说明"代码里已经显示、只是正文不展开讨论的旁支"，
    并非"内容被裁掉"——code 字段本身没有省略号、lines 也只是单段连续区间时，
    即便 elide 非空也不应误报（ch01/ch20 等既有章节的真实写法）。"""
    excerpts = [{
        "path": "vllm/a.py", "lines": "L10-L20",
        "code": "def f():\n    if branch():\n        return other()\n    return main()\n",
        "elide": ["if 分支是旁支，正文只讲 main() 主线"],
    }]
    narrative = (
        "## 一节\n\n"
        "```python\n"
        "# vllm/a.py:L10\n"
        "def f():\n"
        "    if branch():\n"
        "        return other()\n"
        "    return main()\n"
        "```\n"
    )
    res = lint_fidelity(_mk(tmp_path, narrative, excerpts))
    assert not res["elision_gap"]


def test_non_adjacent_blocks_concatenated_without_marker_flagged(tmp_path):
    """dossier 声明两个不相邻 lines 区间(不同摘录条目)，chapter.md 代码块无缝拼接展示 → FAIL。"""
    excerpts = [
        {"path": "vllm/a.py", "lines": "L10-L20", "code": "def f():\n    pass\n", "elide": []},
        {"path": "vllm/a.py", "lines": "L80-L90", "code": "def g():\n    pass\n", "elide": []},
    ]
    narrative = (
        "## 一节\n\n"
        "```python\n"
        "# vllm/a.py:L10\n"
        "def f():\n"
        "    pass\n"
        "# vllm/a.py:L80\n"
        "def g():\n"
        "    pass\n"
        "```\n"
    )
    res = lint_fidelity(_mk(tmp_path, narrative, excerpts))
    assert res["non_adjacent_splice"], "非相邻区间无缝拼接且无省略标记应报告"


def test_non_adjacent_blocks_with_marker_passes(tmp_path):
    """同上非相邻拼接，但块内有省略标记分隔 → 不报。"""
    excerpts = [
        {"path": "vllm/a.py", "lines": "L10-L20", "code": "def f():\n    pass\n", "elide": []},
        {"path": "vllm/a.py", "lines": "L80-L90", "code": "def g():\n    pass\n", "elide": []},
    ]
    narrative = (
        "## 一节\n\n"
        "```python\n"
        "# vllm/a.py:L10\n"
        "def f():\n"
        "    pass\n"
        "# … 省略：中间不相关的几个方法 …\n"
        "# vllm/a.py:L80\n"
        "def g():\n"
        "    pass\n"
        "```\n"
    )
    res = lint_fidelity(_mk(tmp_path, narrative, excerpts))
    assert not res["non_adjacent_splice"]


def test_adjacent_entries_no_marker_needed(tmp_path):
    """两条摘录区间恰好相邻(L10-L20 接 L21-L30)→ 不需要省略标记，也不报非相邻拼接。"""
    excerpts = [
        {"path": "vllm/a.py", "lines": "L10-L20", "code": "def f():\n    pass\n", "elide": []},
        {"path": "vllm/a.py", "lines": "L21-L30", "code": "def g():\n    pass\n", "elide": []},
    ]
    narrative = (
        "## 一节\n\n"
        "```python\n"
        "# vllm/a.py:L10\n"
        "def f():\n"
        "    pass\n"
        "# vllm/a.py:L21\n"
        "def g():\n"
        "    pass\n"
        "```\n"
    )
    res = lint_fidelity(_mk(tmp_path, narrative, excerpts))
    assert not res["non_adjacent_splice"]
    assert not res["elision_gap"]


def test_contiguous_full_block_no_false_positive(tmp_path):
    """行号区间与展示行数完全吻合、无拼接、dossier 未登记省略 → 不报（回归防误报）。"""
    excerpts = [{
        "path": "vllm/a.py", "lines": "L10-L12",
        "code": "def f():\n    return 1\n", "elide": [],
    }]
    narrative = (
        "## 一节\n\n"
        "```python\n"
        "# vllm/a.py:L10\n"
        "def f():\n"
        "    return 1\n"
        "```\n"
    )
    res = lint_fidelity(_mk(tmp_path, narrative, excerpts))
    assert not res["elision_gap"]
    assert not res["non_adjacent_splice"]


def test_marker_without_line_number_skipped_not_false_positive(tmp_path):
    """块内 marker 不带行号(如「三个非相邻方法拼在一起看」纯说明性注释)——
    无法按 (path, la) 匹配到 dossier 条目时应跳过，不误报(真实语料里 ch01 就是这种写法)。"""
    excerpts = [
        {"path": "vllm/a.py", "lines": "L10-L20", "code": "def f():\n    pass\n", "elide": []},
    ]
    narrative = (
        "## 一节\n\n"
        "```python\n"
        "# vllm/a.py（三个非相邻方法拼在一起看，各自省略无关分支）\n"
        "def f():\n"
        "    pass\n"
        "def g():\n"
        "    pass\n"
        "```\n"
    )
    res = lint_fidelity(_mk(tmp_path, narrative, excerpts))
    assert not res["elision_gap"]
    assert not res["non_adjacent_splice"]


def test_elision_checks_are_non_blocking(tmp_path):
    """elision_gap / non_adjacent_splice 报告问题，但降级为非阻断提示——
    对全书语料实测(见 lint_fidelity.py print_report 注释)发现 dossier `elide` 字段的
    真实写法会让"确定性"判据在 ~15 个既有非 primer 章节上误报，故不计入 blocking 汇总。"""
    import io
    import contextlib
    from lint_fidelity import print_report

    excerpts = [{
        "path": "vllm/a.py", "lines": "L10-L30",
        "code": "def f():\n    pass\n...\ndef g():\n    pass\n", "elide": ["省略了中间分支"],
    }]
    narrative = (
        "## 一节\n\n"
        "```python\n"
        "# vllm/a.py:L10\n"
        "def f():\n"
        "    pass\n"
        "```\n\n"
        "vllm/a.py:L10 vllm/a.py:L11 vllm/a.py:L12 vllm/a.py:L13 vllm/a.py:L14\n"
    )
    res = lint_fidelity(_mk(tmp_path, narrative, excerpts))
    assert res["elision_gap"], "问题本身仍应被报告出来（供 writer 自查），只是不计入 blocking"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = print_report(res, str(tmp_path))
    assert code == 0
    assert "🔴" not in buf.getvalue(), "不应出现红色 BLOCKING 汇总标记"
