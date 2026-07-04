import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_source_grounding import lint_source_grounding

NARRATIVE = """# 第 33 章 测试章

## 背景

这里没有任何源码引用，用于触发"无来源"检查。

## 实现细节

vllm/attention/backend.py:120 这里有真实引用。
"""


def _mk(tmp, kind=None, impl_comment="# hello world\n", dossier=True):
    ch = tmp / "inst" / "artifacts" / "ch33-test"
    (ch / "narrative").mkdir(parents=True)
    (ch / "implementation").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(NARRATIVE, encoding="utf-8")
    (ch / "implementation" / "impl.py").write_text(impl_comment, encoding="utf-8")
    if dossier:
        (ch / "dossier").mkdir(parents=True)
        doc = {}
        if kind:
            doc["kind"] = kind
        (ch / "dossier" / "dossier.json").write_text(json.dumps(doc), encoding="utf-8")
    return str(ch)


def test_non_primer_minimal_fixture_still_blocks(tmp_path):
    """回归：非 primer 章，行为与修复前字节级一致——缺来源 section 与 <3 锚点均应 BLOCK。"""
    ch = _mk(tmp_path, kind=None, impl_comment="# just a comment\n")
    r = lint_source_grounding(ch)
    assert r["narrative_vllm_refs"], "非 primer 缺来源的 section 应报告"
    assert r["implementation_references"], "非 primer <3 SOURCE/REFERENCE 锚点应报告"


def test_non_primer_no_dossier_still_blocks(tmp_path):
    """没有 dossier.json 时（旧章节/边界情况），非 primer 行为不变。"""
    ch = _mk(tmp_path, kind=None, impl_comment="# just a comment\n", dossier=False)
    r = lint_source_grounding(ch)
    assert r["narrative_vllm_refs"]
    assert r["implementation_references"]


def test_primer_with_paper_anchors_passes(tmp_path):
    """primer 章：# PAPER: 锚点计入实现锚点计数，且跳过逐 section 来源检查。"""
    impl = (
        "# PAPER: §3.1 Eq.1\n"
        "def a(): pass\n\n"
        "# PAPER: §3.2 Eq.2\n"
        "def b(): pass\n\n"
        "# PAPER: §3.3 Eq.3\n"
        "def c(): pass\n"
    )
    ch = _mk(tmp_path, kind="primer", impl_comment=impl)
    r = lint_source_grounding(ch)
    assert not r["narrative_vllm_refs"], "primer 章应跳过缺来源 section 检查"
    assert not r["implementation_references"], "primer 章 >=3 个 # PAPER 锚点应视为通过"


def test_primer_with_too_few_paper_anchors_still_blocks(tmp_path):
    """primer 章即便跳过 section 检查，实现锚点数仍需 >= 3。"""
    impl = "# PAPER: §3.1 Eq.1\ndef a(): pass\n"
    ch = _mk(tmp_path, kind="primer", impl_comment=impl)
    r = lint_source_grounding(ch)
    assert r["implementation_references"], "primer 章 <3 个锚点仍应 BLOCK"
