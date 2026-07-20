"""citation_range 检查(exp-2026-07-20-01)——正文 ```python 块的 `# path:La-Lb` 行号区间须真对应。

来源:vLLM ch31 评审——lint_fidelity 只验『引文出现在所指文件中』,不验 [a,b] 精确性,
放过 3 处区间错(L94-98 应 L72-74 / L286-295 应 L286-296 / L1358-1369 应 L1359-1372)。
与 lint_dossier 的 embed_verbatim 同类,只是作用在正文引文上。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_fidelity import _check_citation_ranges

SRC = ("import os\n"          # L1
       "\n"                    # L2
       "def alpha(x):\n"       # L3
       "    return x + 1\n"    # L4
       "\n"                    # L5
       "def beta(y):\n"        # L6
       "    return y * 2\n")   # L7


def _mk(tmp, narrative: str, src_text=SRC):
    inst = tmp / "inst"
    ch = inst / "artifacts" / "ch01"
    (ch / "narrative").mkdir(parents=True)
    (inst / "source" / "vllm").mkdir(parents=True)
    (inst / "source" / "vllm" / "mod.py").write_text(src_text, encoding="utf-8")
    (ch / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    return ch


def _blk(cite, body):
    return f"文字\n\n```python\n# {cite}\n{body}```\n\n更多文字\n"


def test_correct_range_passes(tmp_path):
    ch = _mk(tmp_path, _blk("vllm/mod.py:L3-L4", "def alpha(x):\n    return x + 1\n"))
    assert _check_citation_ranges(ch, (ch / "narrative" / "chapter.md").read_text()) == []


def test_wrong_range_blocks(tmp_path):
    # 引的是 L3-L4 的内容，却标成 L6-L7（ch31 issue-1 同型）
    ch = _mk(tmp_path, _blk("vllm/mod.py:L6-L7", "def alpha(x):\n    return x + 1\n"))
    issues = _check_citation_ranges(ch, (ch / "narrative" / "chapter.md").read_text())
    assert issues, "标错区间必须报"
    assert "mod.py" in issues[0] and "L6" in issues[0]


def test_off_by_one_end_blocks(tmp_path):
    # 引到 L6 的内容却只标到 L4（ch31 issue-2 同型：末行落在区间外）
    ch = _mk(tmp_path, _blk("vllm/mod.py:L3-L4",
                            "def alpha(x):\n    return x + 1\n\ndef beta(y):\n"))
    issues = _check_citation_ranges(ch, (ch / "narrative" / "chapter.md").read_text())
    assert issues


def test_leading_blank_line_offset_blocks(tmp_path):
    # 区间起点落在空行上（ch31 issue-3 同型：L1358 是空行）
    ch = _mk(tmp_path, _blk("vllm/mod.py:L2-L4", "def alpha(x):\n    return x + 1\n"))
    issues = _check_citation_ranges(ch, (ch / "narrative" / "chapter.md").read_text())
    assert issues


def test_elision_tolerated(tmp_path):
    ch = _mk(tmp_path, _blk("vllm/mod.py:L1-L7",
                            "import os\n# … 省略 …\ndef beta(y):\n    return y * 2\n"))
    assert _check_citation_ranges(ch, (ch / "narrative" / "chapter.md").read_text()) == []


def test_missing_file_is_skipped_not_blocking(tmp_path):
    ch = _mk(tmp_path, _blk("vllm/nope.py:L1-L2", "whatever\n"))
    assert _check_citation_ranges(ch, (ch / "narrative" / "chapter.md").read_text()) == []
